"""
main_stair_lia_v3.py -- STAIR-LIA v3 (Local Interest Aligned)
=============================================================
Kiến trúc: ZCA Whitening + Offline Bidirectional ROI Attention (CLID)
           + Lightweight Fusion (alpha-gating) + Denoised kNN Graph trên E_fused
           + BPR Loss (+ optional ROI InfoNCE contrastive loss)
"""
from typing import Dict, Tuple
import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
# Compatibility fallback for torchdata.datapipes in PyTorch 2.x / Kaggle environments
try:
    import torchdata.datapipes as dp
except (ImportError, ModuleNotFoundError):
    import sys, types
    import torch.utils.data
    
    if 'torchdata' not in sys.modules:
        td = types.ModuleType('torchdata')
        sys.modules['torchdata'] = td
    else:
        td = sys.modules['torchdata']
        
    if 'torchdata.datapipes' not in sys.modules:
        dp_mod = types.ModuleType('torchdata.datapipes')
        iter_mod = types.ModuleType('torchdata.datapipes.iter')
        map_mod = types.ModuleType('torchdata.datapipes.map')
        
        class IterDataPipe(torch.utils.data.IterableDataset):
            pass
            
        class MapDataPipe(torch.utils.data.Dataset):
            pass
            
        iter_mod.IterDataPipe = IterDataPipe
        map_mod.MapDataPipe = MapDataPipe
        
        dp_mod.iter = iter_mod
        dp_mod.map = map_mod
        td.datapipes = dp_mod
        
        sys.modules['torchdata.datapipes'] = dp_mod
        sys.modules['torchdata.datapipes.iter'] = iter_mod
        sys.modules['torchdata.datapipes.map'] = map_mod

import freerec


from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

# Khớp phiên bản freerec==0.8.5 trong môi trường Kaggle
freerec.declare(version='0.8.5')

# ============================================================================
# Config
# ============================================================================
cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim",  type=int,   default=64)
cfg.add_argument("--num-layers",     type=int,   default=3)
cfg.add_argument("--mfiles",         type=str,   default="textual_modality.pkl,visual_modality.pkl")
cfg.add_argument("--num-neighbors",  type=str,   default='5-1')
cfg.add_argument("--gamma",          type=float, default=0.2)

# LIA-specific
cfg.add_argument("--lia-precomputed-dir", type=str, default="preprocessed_lia",
                 help="Thư mục chứa E_fused_t.pt / E_fused_v.pt")
cfg.add_argument("--lia-alpha",      type=float, default=0.5,
                 help="Weight khởi tạo cho text vs visual trong modal fusion")
cfg.add_argument("--lia-roi-cl",     type=float, default=0.0,
                 help="Weight cho ROI Contrastive loss (0.0 = disable, 0.01 = enable)")
cfg.add_argument("--lia-temperature",type=float, default=0.07,
                 help="Temperature cho InfoNCE ROI contrastive loss")

cfg.set_defaults(
    description="STAIR-LIA-v3",
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

# Fallback to CPU if CUDA fails
if torch.cuda.is_available():
    cfg.device = 'cuda'
else:
    cfg.device = 'cpu'

try:
    _ = torch.tensor([1.0], device=cfg.device)
except Exception:
    cfg.device = 'cpu'
    print("Switched to CPU due to CUDA error")

cfg.mfiles        = cfg.mfiles.split(',')
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-')))

# BSC Smoother spectral decay
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


def roi_contrastive_loss(
    roi_t: torch.Tensor,
    roi_v: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    roi_t = F.normalize(roi_t, p=2, dim=-1)
    roi_v = F.normalize(roi_v, p=2, dim=-1)
    logits = roi_v @ roi_t.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_v2t = F.cross_entropy(logits,   labels)
    loss_t2v = F.cross_entropy(logits.t(), labels)
    return (loss_v2t + loss_t2v) / 2.0


class STAIR_LIA_v3(freerec.models.GenRecArch):
    """
    STAIR-LIA v3 (Local Interest Aligned).
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)
        self.num_layers = cfg.num_layers

        self.User.add_module('embeddings', nn.Embedding(self.User.count, cfg.embedding_dim))
        self.Item.add_module('embeddings', nn.Embedding(self.Item.count, cfg.embedding_dim))

        self.register_buffer(
            'Adj',
            self.dataset.train().to_normalized_adj(normalization='sym').to(cfg.device)
        )

        self.alpha_raw = nn.Parameter(torch.zeros(1))

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)

    def marked_params(self):
        return [
            {'params': self.User.parameters(), 'smoother': None},
            {
                'params': self.Item.parameters(),
                'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann'),
            },
            {'params': [self.alpha_raw], 'smoother': None, 'lr': cfg.lr * 0.1, 'weight_decay': 0.0},
        ]

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        return edge_index

    def build_denoised_knn_graph(self, E_fused: torch.Tensor, num_neighbors_list) -> None:
        print("[LIA] Building Denoised kNN Graph on clean E_fused...")
        k_total = sum(num_neighbors_list)
        edge_index = self.get_knn_graph(E_fused.cpu(), k=k_total)
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce='sum')
        edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce='max')
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, edge_weight, normalization='sym')
        
        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.Item.count, self.Item.count)
        )
        self.register_buffer('mAdj', mAdj.to_sparse_csr().to(cfg.device))
        print(f"[LIA] Denoised kNN Graph ready: {edge_index.shape[1]} edges (k={k_total})")

    def whitening(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.cpu().float()
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def prepare(self, path: str):
        from freerec.utils import import_pickle

        candidate_dirs = [
            os.path.join(path, cfg.lia_precomputed_dir, cfg.dataset),
            os.path.join(path, "preprocessed_lia", cfg.dataset),
            os.path.join(path, cfg.dataset),
            os.path.join(path),
            os.path.join("/kaggle/working/preprocessed_lia", cfg.dataset),
            os.path.join("/kaggle/data/preprocessed_lia", cfg.dataset),
            os.path.join("/kaggle/data", cfg.dataset, "preprocessed_lia", cfg.dataset),
            os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset, "preprocessed_lia", cfg.dataset),
        ]

        found_dir = None
        for d in candidate_dirs:
            if os.path.exists(os.path.join(d, "E_fused_t.pt")) and os.path.exists(os.path.join(d, "E_fused_v.pt")):
                found_dir = d
                break

        if found_dir:
            print(f"[LIA] Loading precomputed E_fused from: {found_dir}")
            E_fused_t = torch.load(os.path.join(found_dir, "E_fused_t.pt"), map_location='cpu').float()
            E_fused_v = torch.load(os.path.join(found_dir, "E_fused_v.pt"), map_location='cpu').float()
            
            # === [SCALE FIX]: Chuẩn hóa L2-norm = 1.0 đồng bộ với hệ quy chiếu STAIR baseline ===
            E_fused_t = F.normalize(E_fused_t, p=2, dim=-1)
            E_fused_v = F.normalize(E_fused_v, p=2, dim=-1)

            roi_t_p = os.path.join(found_dir, "E_roi_t.pt")
            roi_v_p = os.path.join(found_dir, "E_roi_v.pt")
            if os.path.exists(roi_t_p) and os.path.exists(roi_v_p):
                E_roi_t = F.normalize(torch.load(roi_t_p, map_location='cpu').float(), p=2, dim=-1)
                E_roi_v = F.normalize(torch.load(roi_v_p, map_location='cpu').float(), p=2, dim=-1)
                self.register_buffer('E_roi_t', E_roi_t.to(cfg.device))
                self.register_buffer('E_roi_v', E_roi_v.to(cfg.device))
            else:
                self.E_roi_t = None
                self.E_roi_v = None
        else:
            print(f"[LIA] Precomputed files not found in candidate paths. Fallback to SVD Whitening...")
            mfeats_raw = [
                import_pickle(os.path.join(path, mfile)) for mfile in cfg.mfiles
            ]
            E_fused_t = F.normalize(self.whitening(mfeats_raw[0].float()), p=2, dim=-1)
            E_fused_v = F.normalize(self.whitening(mfeats_raw[1].float()), p=2, dim=-1)
            self.E_roi_t = None
            self.E_roi_v = None

        print(f"[LIA] E_fused_t: {tuple(E_fused_t.shape)}, mean_norm={E_fused_t.norm(dim=-1).mean():.3f}")
        print(f"[LIA] E_fused_v: {tuple(E_fused_v.shape)}, mean_norm={E_fused_v.norm(dim=-1).mean():.3f}")

        # 1. Xây Denoised kNN Graph trên CPU với vector đã chuẩn hóa L2
        E_for_graph = F.normalize(0.5 * E_fused_t + 0.5 * E_fused_v, p=2, dim=-1)
        self.build_denoised_knn_graph(E_for_graph, cfg.num_neighbors)

        self.register_buffer('E_fused_t', E_fused_t.to(cfg.device))
        self.register_buffer('E_fused_v', E_fused_v.to(cfg.device))

        # 2. Khởi tạo User embeddings qua R @ E_modal_init trên CPU
        E_modal_init = F.normalize(0.5 * E_fused_t + 0.5 * E_fused_v, p=2, dim=-1).cpu()
        edge_index_ui = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index_ui, edge_weight_ui = freerec.graph.to_normalized(
            edge_index_ui, normalization='left'
        )
        R = torch.sparse_coo_tensor(
            edge_index_ui, edge_weight_ui,
            size=(self.User.count, self.Item.count)
        ).to_sparse_csr()
        
        user_init = (R @ E_modal_init).float()
        self.User.embeddings.weight.data.copy_(user_init)

        self.Item.embeddings.weight.data.zero_()
        print("[LIA] Setup completed successfully!")

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_pairs_source(
        ).gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def get_modal_item_embeddings(self) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha_raw).to(self.E_fused_t.device)
        e_modal = alpha * self.E_fused_t + (1.0 - alpha) * self.E_fused_v
        return F.normalize(e_modal, p=2, dim=-1)

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        modal_items = self.get_modal_item_embeddings()
        total_items = self.Item.embeddings.weight + modal_items

        allEmbds = torch.cat((self.User.embeddings.weight, total_items), dim=0)
        features = allEmbds
        smoothed = allEmbds
        beta     = 1 - cfg.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(avgEmbds, (self.User.count, self.Item.count))
        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode()
        users     = data[self.User]
        positives = data[self.Item]
        negatives = data[self.INeg]

        bpr_loss = self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[positives]),
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[negatives]),
        )

        if cfg.lia_roi_cl > 0.0 and self.E_roi_t is not None:
            pos_items_flat = positives.view(-1)
            roi_t_batch = self.E_roi_t[pos_items_flat]
            roi_v_batch = self.E_roi_v[pos_items_flat]
            cl_loss = roi_contrastive_loss(roi_t_batch, roi_v_batch, cfg.lia_temperature)
            return bpr_loss + cfg.lia_roi_cl * cl_loss
        return bpr_loss

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum('BKD,ND->BN', userEmbds, itemEmbds)

    def recommend_from_pool(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum('BKD,BKD->BK', userEmbds, itemEmbds)


class CoachForSTAIR_LIA_v3(freerec.launcher.Coach):
    def set_optimizer(self):
        self.optimizer = AdamWSEvo(
            self.model.marked_params(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )
        print(f"[Optimizer] AdamWSEvo ready (base lr={self.cfg.lr})")

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.monitor(
                loss.item(), n=len(data[self.User]),
                reduction='mean', mode='train', pool=['LOSS'],
            )
        alpha_val = torch.sigmoid(self.model.alpha_raw).item()
        print(f"  [alpha @epoch {epoch:3d}]: {alpha_val:.4f} (text weight)", flush=True)


def main():
    # Robust dataset loading:
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

    if not hasattr(dataset, 'TASK') or dataset.TASK is None:
        dataset.TASK = tasktag

    model = STAIR_LIA_v3(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_LIA_v3(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == '__main__':
    main()
