

from typing import Dict, Tuple, Optional

import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import freerec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3, help="the number of layers for FSC/BSC")

cfg.add_argument("--mfiles", type=str, default="textual_modality.pkl,visual_modality.pkl", help="the files saving modality")
cfg.add_argument("--num-neighbors", type=str, default='5-1', help="for kNN graph")
cfg.add_argument("--gamma", type=float, default=0.2)
cfg.add_argument("--patience", type=int, default=30, help="early stopping patience (default: 30)")
cfg.add_argument("--early-stop-warmup", type=int, default=100, help="early stopping warmup epochs (default: 100)")

cfg.set_defaults(
    description="STAIR",
    root="../../data",
    dataset='Amazon2014Baby_550_MMRec',
    epochs=500,
    batch_size=1024,
    optimizer='adamwsevo',
    lr=1e-3,
    weight_decay=0.1,
    seed=1,
    monitors=["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
    which4best="NDCG@20",
)
cfg.compile()


if isinstance(cfg.mfiles, str):
    cfg.mfiles = cfg.mfiles.split(',')
if isinstance(cfg.num_neighbors, str):
    cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-'))) 

# beta3 here is the 1 - beta_j for BSC
cfg.beta3 = (0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)).to(cfg.device)

class STAIR(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet
    ) -> None:
        super().__init__(dataset)

        self.num_layers = cfg.num_layers

        self.User.add_module(
            "embeddings", nn.Embedding(
                self.User.count, cfg.embedding_dim
            )
        )

        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count, cfg.embedding_dim
            )
        )

        self.register_buffer(
            "Adj",
            self.dataset.train().to_normalized_adj(
                normalization='sym'
            )
        )

        self.reset_parameters()

        self.prepare(dataset.path)

        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def marked_params(self):
        params = [
            {
                'params': self.User.parameters(),
                'smoother': None
            },
            {
                'params': self.Item.parameters(), 
                'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann')
            },
        ]
        return params

    def whitening(self, feats: torch.Tensor):
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats, dtype=torch.float32)
        else:
            feats = feats.float()
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        if feats.size(1) < cfg.embedding_dim:
            reps = math.ceil(cfg.embedding_dim / feats.size(1))
            scale = math.sqrt(feats.size(1) / cfg.embedding_dim)
            feats = (feats.repeat(1, reps)[:, :cfg.embedding_dim]) * scale
        else:
            feats = feats[:, :cfg.embedding_dim]
        return feats * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        r"""
        Compute the kNN graph.
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = features.float()
        features = F.normalize(features, dim=-1) # (N, D)
        sim = features @ features.t() # (N, N)
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(
            sim, k, symmetric=False
        )
        return edge_index

    def prepare(self, path: str):
        from freerec.utils import import_pickle

        mfeats = []
        for mfile in cfg.mfiles:
            mpath = os.path.join(path, mfile)
            if not os.path.exists(mpath):
                # Fallback search across common data locations
                for cand in [
                    os.path.join(cfg.root, cfg.dataset, mfile),
                    os.path.join("/kaggle/data", cfg.dataset, mfile),
                    os.path.join("/kaggle/working/STAIR/data", cfg.dataset, mfile),
                    os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset, mfile),
                    os.path.join("data", cfg.dataset, mfile),
                ]:
                    if os.path.exists(cand):
                        mpath = cand
                        break
            mfeats.append(import_pickle(mpath))

        edge_index = torch.cat(
            [self.get_knn_graph(feats, k) for feats, k in zip(mfeats, cfg.num_neighbors)],
            dim=1
        )
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(
            edge_index, edge_weight, reduce='sum'
        )
        edge_index, edge_weight = freerec.graph.to_undirected(
            edge_index, edge_weight, reduce='max'
        )
        edge_index, edge_weight = freerec.graph.to_normalized(
            edge_index, edge_weight,
            normalization='sym'
        )
        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight,
            size=(self.Item.count, self.Item.count)
        )
        self.register_buffer(
            'mAdj',
            mAdj.to_sparse_csr()
        )

        # MI
        mfeats = [self.whitening(mfeat) * k for mfeat, k in zip(mfeats, cfg.num_neighbors)]
        mfeats = sum(mfeats).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats)

        edge_index = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, normalization='left')
        R = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.User.count, self.Item.count)
        ).to_sparse_csr()

        self.User.embeddings.weight.data.copy_(R @ mfeats)

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_pairs_source(
        ).gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        ) # (N, D)

        features = allEmbds
        smoothed = allEmbds
        
        # FSC
        beta = 1 - cfg.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        userEmbds = userEmbds[users] # (B, 1, D)
        iposEmbds = itemEmbds[positives] # (B, 1, D)
        inegEmbds = itemEmbds[negatives] # (B, K, D)

        rec_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userEmbds, iposEmbds),
            torch.einsum("BKD,BKD->BK", userEmbds, inegEmbds)
        )
        return rec_loss

    def reset_ranking_buffers(self):
        """This method will be executed before evaluation."""
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum("BKD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]] # (B, 101, D)
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)


class CoachForSTAIR(freerec.launcher.Coach):

    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.marked_params(), lr=self.cfg.lr, 
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamsevo':
            self.optimizer = AdamSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamwsevo':
            self.optimizer = AdamWSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unexpected optimizer {self.cfg.optimizer} ..."
            )

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(
                loss.item(), 
                n=len(data[self.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_ndcg20 = -1.0
        self.best_epoch = 0
        self.patience_counter = 0
        ds_lower = getattr(self.cfg, 'dataset', '').lower()
        default_warmup = 100 if any(k in ds_lower for k in ['electronic', 'clothing']) else 200
        default_patience = 20 if any(k in ds_lower for k in ['electronic', 'clothing']) else 30
        self.early_stop_warmup = getattr(self.cfg, 'early_stop_warmup', default_warmup)
        self.patience = getattr(self.cfg, 'patience', default_patience)

    def evaluate(self, epoch: int = 0, mode: str = 'valid'):
        super().evaluate(epoch, mode=mode)
        if mode == 'valid':
            try:
                ds_lower = getattr(self.cfg, 'dataset', '').lower()
                enable_es = any(k in ds_lower for k in ['clothing', 'electronic'])

                meters = getattr(self, 'meters', None)
                if meters is None and hasattr(self, 'monitor') and hasattr(self.monitor, 'meters'):
                    meters = self.monitor.meters

                def get_val(name):
                    if meters is not None:
                        for k, v in meters.items():
                            if name.lower() in str(k).lower():
                                return getattr(v, 'avg', getattr(v, 'val', None))
                    return None

                n20 = get_val('NDCG@20')
                if n20 is not None:
                    if n20 > self.best_ndcg20:
                        self.best_ndcg20 = n20
                        self.best_epoch = epoch
                        self.patience_counter = 0
                    else:
                        if epoch + 1 >= self.early_stop_warmup:
                            self.patience_counter += 1
                        else:
                            self.patience_counter = 0

                    if enable_es:
                        if epoch + 1 >= self.early_stop_warmup:
                            if self.patience_counter > 0:
                                print(
                                    f"  ⏳ [Early Stopping] Patience: {self.patience_counter}/{self.patience} "
                                    f"(Best NDCG@20: {self.best_ndcg20:.4f} @Epoch {self.best_epoch})"
                                )
                            if self.patience_counter >= self.patience:
                                print(
                                    f"\n🛑 [EARLY STOPPING TRIGGERED @Epoch {epoch + 1}] "
                                    f"Kích hoạt dừng sớm sau {self.patience} epochs không cải thiện NDCG@20 "
                                    f"(tính từ sau Epoch {self.early_stop_warmup}, Best Epoch: {self.best_epoch}, Best NDCG@20: {self.best_ndcg20:.4f}).\n"
                                )
                                self.cfg.epochs = epoch + 1
            except Exception:
                pass


def main():

    # Robust auto-bridge for FreeRec:
    processed_dir = os.path.join(cfg.root, "Processed", cfg.dataset)
    if os.path.islink(processed_dir) and not os.path.exists(processed_dir):
        try:
            os.unlink(processed_dir)
        except Exception:
            pass

    if not os.path.exists(processed_dir) or (os.path.isdir(processed_dir) and not os.listdir(processed_dir)):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
        candidates = [
            os.path.join(cfg.root, cfg.dataset),
            os.path.join("/kaggle/data", cfg.dataset),
            os.path.join("/kaggle/data/Processed", cfg.dataset),
            os.path.join("/kaggle/working/STAIR/data", cfg.dataset),
            os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset),
            os.path.join(script_dir, "data", cfg.dataset),
            os.path.join("data", cfg.dataset),
        ]
        for cand in candidates:
            if os.path.exists(cand) and os.path.isdir(cand) and os.path.abspath(cand) != os.path.abspath(processed_dir) and len(os.listdir(cand)) > 0:
                os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
                try:
                    os.symlink(cand, processed_dir)
                    print(f"[DataSet] >>> Auto-bridged symlink: {cand} -> {processed_dir}")
                except Exception:
                    import shutil
                    shutil.copytree(cand, processed_dir, dirs_exist_ok=True)
                    print(f"[DataSet] >>> Auto-bridged copied: {cand} -> {processed_dir}")
                break

    # Robust dataset loading:
    # 1. freerec.data.datasets contains a submodule named 'tiktok', so getattr(...) returns a module
    #    rather than a class when cfg.dataset == 'tiktok'.
    # 2. FreeRec's ValidSampler/TestSampler requires dataset.TASK is MATCHING. We define it on
    #    RecDataSet class, BaseSet class, and pass tasktag to ensure it is always present.
    tasktag = getattr(cfg, 'tasktag', None) or getattr(freerec.data.tags, 'MATCHING', None)
    if hasattr(freerec.data.datasets, 'RecDataSet'):
        freerec.data.datasets.RecDataSet.TASK = tasktag
    if hasattr(freerec.data.datasets, 'base') and hasattr(freerec.data.datasets.base, 'BaseSet'):
        freerec.data.datasets.base.BaseSet.TASK = tasktag

    ds_cls = getattr(freerec.data.datasets, cfg.dataset, None)
    if isinstance(ds_cls, type):
        try:
            dataset = ds_cls(root=cfg.root)
        except Exception:
            try:
                from freerec.data.datasets.base import MatchingRecDataSet
                dataset = MatchingRecDataSet(cfg.root, cfg.dataset, tasktag=tasktag)
            except Exception:
                dataset = freerec.data.datasets.RecDataSet(
                    cfg.root, cfg.dataset, tasktag=tasktag
                )
    else:
        try:
            from freerec.data.datasets.base import MatchingRecDataSet
            dataset = MatchingRecDataSet(cfg.root, cfg.dataset, tasktag=tasktag)
        except Exception:
            dataset = freerec.data.datasets.RecDataSet(
                cfg.root, cfg.dataset, tasktag=tasktag
            )

    # Ensure TASK attribute is always attached to dataset instance
    if not hasattr(dataset, 'TASK') or dataset.TASK is None:
        dataset.TASK = tasktag

    model = STAIR(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()

    save_dir = getattr(cfg, 'CHECKPOINT_PATH', getattr(cfg, 'root_dir', '.'))
    best_pt_path = os.path.join(save_dir, getattr(cfg, 'BEST_FILENAME', 'best.pt'))
    best_pth_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_pt_path) and not os.path.exists(best_pth_path):
        try:
            import shutil
            shutil.copy2(best_pt_path, best_pth_path)
            print(f"[Coach] >>> Đã lưu bản sao mô hình tối ưu: {best_pth_path}")
        except Exception:
            pass

    print("\n[Coach] >>> HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ TỐI ƯU THÀNH CÔNG (Exit Code: 0)!")


if __name__ == "__main__":
    main()