# -*- coding: utf-8 -*-
"""
main_stair_ne_nlgcl_v5_plus.py — STAIR-NE-NLGCL v5+ (v3-Refined) Training Script
================================================================================
Kế thừa trọn vẹn sự tinh gọn tối ưu của v5 (100% Direct Gradient Flow):
1. Bỏ hoàn toàn Projection Head -> InfoNCE tác động trực tiếp vào H^(0) và H^(1).
2. Bỏ hoàn toàn Regularized Diagonal Spectral Projector -> Loại bỏ ma sát tối ưu.
3. Giữ nguyên Sign-Preserving Spectral Perturbation (|noise| >= 0) -> Bảo toàn góc phần tư 64D.
4. Linear HANS (Hardness-Aware Negative Scheduling):
   psi = 1.0 + gamma_h * clamp(cos_sim, min=0.0) với gamma_h = 0.15 (không làm méo tau_eff).
5. Hard-Threshold MFNA (Modality False Negative Attenuation):
   Nếu sim_modal > 0.85 -> mask = 0.0 (loại bỏ hoàn toàn near-duplicates), ngược lại 1.0.
6. Constant Contrastive Pressure:
   lambda_cl = 0.010 (warmup 0 -> 0.010 trong 50 epoch đầu, sau đó cố định 100%).

Usage:
    python main_stair_ne_nlgcl_v5_plus.py --config configs/Amazon2014Baby_550_MMRec.yaml
    python main_stair_ne_nlgcl_v5_plus.py --config configs/Amazon2014Sports_550_MMRec.yaml
"""

import math
import os
import sys
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# ── Compatibility Patch for torchdata in PyTorch 2.x / Python 3.12 / Kaggle ──
try:
    import torchdata
    import torchdata.datapipes as dp
except Exception:
    dp = None

if dp is None or 'torchdata.datapipes' not in sys.modules:
    if 'torchdata' not in sys.modules:
        td = types.ModuleType('torchdata')
        sys.modules['torchdata'] = td
    else:
        td = sys.modules['torchdata']

    dp = types.ModuleType('torchdata.datapipes')
    td.datapipes = dp
    sys.modules['torchdata.datapipes'] = dp

# Ensure dp.iter and IterDataPipe exist
if not hasattr(dp, 'iter'):
    iter_mod = types.ModuleType('torchdata.datapipes.iter')
    dp.iter = iter_mod
    sys.modules['torchdata.datapipes.iter'] = iter_mod
if not hasattr(dp.iter, 'IterDataPipe'):
    class IterDataPipe(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter([])
    dp.iter.IterDataPipe = IterDataPipe

# Ensure dp.map and MapDataPipe exist
if not hasattr(dp, 'map'):
    map_mod = types.ModuleType('torchdata.datapipes.map')
    dp.map = map_mod
    sys.modules['torchdata.datapipes.map'] = map_mod
if not hasattr(dp.map, 'MapDataPipe'):
    class MapDataPipe(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            raise NotImplementedError
        def __len__(self):
            return 0
    dp.map.MapDataPipe = MapDataPipe

# Ensure functional_datapipe decorator exists on dp
if not hasattr(dp, 'functional_datapipe'):
    def functional_datapipe(name, enable_df_datapipes_support=False):
        def decorator(cls):
            def method(self, *args, **kwargs):
                return cls(self, *args, **kwargs)
            if hasattr(dp, 'iter') and hasattr(dp.iter, 'IterDataPipe'):
                setattr(dp.iter.IterDataPipe, name, method)
            if hasattr(dp, 'map') and hasattr(dp.map, 'MapDataPipe'):
                setattr(dp.map.MapDataPipe, name, method)
            try:
                if hasattr(torch.utils.data, 'IterDataPipe'):
                    setattr(torch.utils.data.IterDataPipe, name, method)
                if hasattr(torch.utils.data, 'MapDataPipe'):
                    setattr(torch.utils.data.MapDataPipe, name, method)
            except Exception:
                pass
            return cls
        return decorator
    dp.functional_datapipe = functional_datapipe

import freerec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

from models.stair_ne_nlgcl_v5_plus import STAIR_NE_NLGCL_v5_Plus

freerec.declare(version='0.8.5')

# ═════════════════════════════════════════════════════════════════════════════
# Configuration Setup: STAIR Baseline + STAIR-NE-NLGCL v5+ (v3-Refined)
# ═════════════════════════════════════════════════════════════════════════════
cfg = freerec.parser.Parser()

# ── STAIR Baseline Parameters ──
cfg.add_argument("--embedding-dim", type=int, default=64,
                 help="Latent vector embedding dimension D (default: 64)")
cfg.add_argument("--num-layers", type=int, default=3,
                 help="Number of layers for FSC/BSC (default: 3)")
cfg.add_argument("--mfiles", type=str,
                 default="textual_modality.pkl,visual_modality.pkl",
                 help="Comma-separated modality feature files")
cfg.add_argument("--num-neighbors", type=str, default='5-1',
                 help="kNN counts per modality, e.g. '5-1'")
cfg.add_argument("--gamma", type=float, default=0.2,
                 help="Spectral decay exponent for beta3 (default: 0.2)")

# ── STAIR-NE-NLGCL v5+ Parameters ──
cfg.add_argument("--tau", type=float, default=0.20,
                 help="Temperature tau for InfoNCE softmax (default: 0.20)")
cfg.add_argument("--alpha-dir", type=float, default=0.50,
                 help="Direction balance: alpha*L_{u->i} + (1-alpha)*L_{i->u} (default: 0.50)")
cfg.add_argument("--eps", type=float, default=0.08,
                 help="Sign-preserving noise amplitude epsilon (default: 0.08)")
cfg.add_argument("--tau-thresh", type=float, default=0.85,
                 help="Semantic similarity threshold for false negative masking (default: 0.85)")
cfg.add_argument("--lambda-cl", type=float, default=0.010,
                 help="Constant contrastive loss weight (default: 0.010)")
cfg.add_argument("--gamma-h", type=float, default=0.15,
                 help="Linear HANS hardness penalty coefficient (default: 0.15)")
cfg.add_argument("--warmup-epochs", type=int, default=50,
                 help="Warmup epochs for lambda (default: 50)")

cfg.set_defaults(
    description="STAIR-NE-NLGCL-v5-Plus",
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

cfg.mfiles        = cfg.mfiles.split(',')
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-')))

# BSC Smoother spectral decay beta3
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ═════════════════════════════════════════════════════════════════════════════
# STAIR-NE-NLGCL v5+ (v3-Refined) Model Architecture
# ═════════════════════════════════════════════════════════════════════════════
class STAIR_NE_NLGCL_v5_Plus_Model(freerec.models.GenRecArch):
    """
    STAIR-NE-NLGCL v5+ (v3-Refined) Model Architecture:
    Combines STAIR Forward Stepwise Convolution with Clean Direct Contrastive Learning:
    1. No Projection Head (100% Direct InfoNCE Gradient Flow to H^(0) and H^(1))
    2. No Regularized Diagonal Projector (Zero optimization friction)
    3. True Sign-Preserving Spectral Perturbation (|noise| >= 0)
    4. Clean Linear HANS (psi = 1 + gamma_h * max(0, cos))
    5. Hard-Threshold MFNA (mask = I(sim_modal <= 0.85))
    6. Constant Lambda CL (0.010 with 50-epoch linear warmup)
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)
        self.num_layers = cfg.num_layers

        self.User.add_module(
            'embeddings', nn.Embedding(self.User.count, cfg.embedding_dim)
        )
        self.Item.add_module(
            'embeddings', nn.Embedding(self.Item.count, cfg.embedding_dim)
        )

        self.register_buffer(
            'Adj',
            self.dataset.train().to_normalized_adj(normalization='sym')
        )
        self.register_buffer('beta3', cfg.beta3)

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

        # Clean STAIR-NE-NLGCL v5+ Contrastive Module
        self.ne_nlgcl_v5_plus = STAIR_NE_NLGCL_v5_Plus(
            n_users       = self.User.count,
            n_items       = self.Item.count,
            tau           = cfg.tau,
            alpha_dir     = cfg.alpha_dir,
            eps           = cfg.eps,
            tau_thresh    = cfg.tau_thresh,
            lambda_cl     = cfg.lambda_cl,
            gamma_h       = cfg.gamma_h,
            warmup_epochs = cfg.warmup_epochs,
        )

        self.last_cl_loss: Optional[float] = None

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
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = features.float()
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
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
                for cand in [
                    os.path.join(cfg.root, cfg.dataset, mfile),
                    os.path.join("/kaggle/data", cfg.dataset, mfile),
                    os.path.join("/kaggle/data/Processed", cfg.dataset, mfile),
                    os.path.join("/kaggle/working/STAIR/data", cfg.dataset, mfile),
                    os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset, mfile),
                    os.path.join("data", cfg.dataset, mfile),
                    os.path.join("data/Processed", cfg.dataset, mfile),
                ]:
                    if os.path.exists(cand):
                        mpath = cand
                        break
            mfeats.append(import_pickle(mpath))

        edge_index = torch.cat(
            [self.get_knn_graph(feats, k)
             for feats, k in zip(mfeats, cfg.num_neighbors)],
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
            edge_index, edge_weight, normalization='sym'
        )
        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight,
            size=(self.Item.count, self.Item.count)
        )
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        # Whitened modal feature initialization
        mfeats_w = [
            self.whitening(mfeat) * k
            for mfeat, k in zip(mfeats, cfg.num_neighbors)
        ]
        mfeats_init = sum(mfeats_w).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats_init)

        edge_index_ui = self.dataset.train().to_bigraph(
            edge_type='u2i'
        )['u2i'].edge_index
        edge_index_ui, edge_weight_ui = freerec.graph.to_normalized(
            edge_index_ui, normalization='left'
        )
        R = torch.sparse_coo_tensor(
            edge_index_ui, edge_weight_ui,
            size=(self.User.count, self.Item.count)
        ).to_sparse_csr()
        user_profiles_init = R @ mfeats_init
        self.User.embeddings.weight.data.copy_(user_profiles_init)

        # Register raw modal features for In-batch Dynamic False Negative Attenuation
        self.register_buffer('item_modals_raw', mfeats_init.detach().clone())

    def sure_trainpipe(self, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_pairs_source()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward Stepwise Convolution with Layer Intermediates capture:
        Returns:
            userEmbds:    (N_u, D) final aggregated user representations
            itemEmbds:    (N_i, D) final aggregated item representations
            layer_embeds: [H^0, H^1, ..., H^L] per-layer representations
        """
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight),
            dim=0,
        )

        layer_embeds = [allEmbds]
        features = allEmbds
        smoothed = allEmbds

        beta = (1.0 - self.beta3).to(allEmbds.device)
        norm_correction = 1.0 - beta ** (self.num_layers + 1)

        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
            layer_embeds.append(features)

        avgEmbds = smoothed.mul(1.0 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        return userEmbds, itemEmbds, layer_embeds

    def encode_for_eval(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluation encode function (zero overhead)."""
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight),
            dim=0,
        )
        features = allEmbds
        smoothed = allEmbds
        beta = (1.0 - self.beta3).to(allEmbds.device)
        norm_correction = 1.0 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1.0 - beta).div(norm_correction)
        return torch.split(avgEmbds, (self.User.count, self.Item.count))

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        """
        Training step:
        L_total = L_BPR + lambda_cl(t) * L_NE-NLGCL_v5+
        """
        userEmbds, itemEmbds, layer_embeds = self.encode()

        users     = data[self.User]
        positives = data[self.Item]
        negatives = data[self.INeg]

        # 1. Pairwise BPR Ranking Loss
        rec_loss = self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[positives]),
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[negatives]),
        )

        # 2. STAIR-NE-NLGCL v5+ Contrastive Loss
        if self.training:
            beta = (1.0 - self.beta3).to(userEmbds.device)
            i_mod = self.item_modals_raw if hasattr(self, 'item_modals_raw') else None

            weighted_cl_loss, raw_cl_loss = self.ne_nlgcl_v5_plus(
                layer_embeds = layer_embeds,
                users        = users,
                positives    = positives,
                beta         = beta,
                item_modals  = i_mod,
            )
            self.last_cl_loss = raw_cl_loss
            return rec_loss + weighted_cl_loss

        return rec_loss

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode_for_eval()
        self.ranking_buffer = {
            self.User: userEmbds.detach().clone(),
            self.Item: itemEmbds.detach().clone(),
        }

    def recommend_from_full(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum('BKD,ND->BN', userEmbds, itemEmbds)

    def recommend_from_pool(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum('BKD,BKD->BK', userEmbds, itemEmbds)


# ═════════════════════════════════════════════════════════════════════════════
# Coach Class for STAIR-NE-NLGCL v5+
# ═════════════════════════════════════════════════════════════════════════════
class CoachForSTAIR_NE_NLGCL_v5_Plus(freerec.launcher.Coach):

    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'adamwsevo':
            self.optimizer = AdamWSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adamsevo':
            self.optimizer = AdamSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"CoachForSTAIR_NE_NLGCL_v5_Plus does not support {self.cfg.optimizer} optimizer"
            )

    def train_per_epoch(self, epoch: int):
        self.model.train()
        self.model.ne_nlgcl_v5_plus.update_epoch(epoch + 1)
        total_cl_loss = 0.0
        cl_batches = 0

        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                loss.item(), n=len(data[self.User]),
                reduction="mean", mode='train', pool=['LOSS'],
            )

            if hasattr(self.model, 'last_cl_loss') and self.model.last_cl_loss is not None:
                total_cl_loss += self.model.last_cl_loss
                cl_batches += 1

        if cl_batches > 0:
            avg_cl_loss = total_cl_loss / float(cl_batches)
            gamma_h, curr_lambda = self.model.ne_nlgcl_v5_plus.get_current_params()
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == self.cfg.epochs:
                print(
                    f"  [v5+ Epoch {epoch + 1:03d}] gamma_h: {gamma_h:.4f} | "
                    f"lambda: {curr_lambda:.5f} | avg_cl_loss: {avg_cl_loss:.6f}"
                )


# ═════════════════════════════════════════════════════════════════════════════
# Main Execution Entry Point
# ═════════════════════════════════════════════════════════════════════════════
def main():
    # Auto-bridge dataset for FreeRec:
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

    model = STAIR_NE_NLGCL_v5_Plus_Model(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_NE_NLGCL_v5_Plus(
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
