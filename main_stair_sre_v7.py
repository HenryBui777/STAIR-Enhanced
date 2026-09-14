"""
main_stair_sre_v6.py — STAIR-SRE v6 Training Script
======================================================
Stepwise Spectral-Refined Contrastive Learning

Phase 3 — Dot 1: Core Architecture with 4 Pillars
  Pillar 1: Diagonal Spectral-scaling Projector (0-rotation)
  Pillar 2: Soft Spectral Swapping (Bernoulli hard negative generation)
  Pillar 3: Adaptive False Negative Attenuation (smooth, outside exp)
  Pillar 4: Hierarchical Layer-wise Contrastive Alignment (NLGCL heritage)

Target: Breakout ≥ +5.0% synchronously across all 3 datasets:
  Amazon Baby:        Recall@20 ≥ 0.1095  (+5.09%) | NDCG@20 ≥ 0.0480  (+5.73%)
  Amazon Sports:      Recall@20 ≥ 0.1168  (+5.13%) | NDCG@20 ≥ 0.0530  (+6.00%)
  Amazon Electronics: Recall@20 ≥ 0.0705  (+6.02%) | NDCG@20 ≥ 0.0325  (+7.26%)
"""

from typing import Dict, Tuple, List
import os
import sys
import types
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

if not hasattr(dp, 'iter'):
    iter_mod = types.ModuleType('torchdata.datapipes.iter')
    dp.iter = iter_mod
    sys.modules['torchdata.datapipes.iter'] = iter_mod
if not hasattr(dp.iter, 'IterDataPipe'):
    class IterDataPipe(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter([])
    dp.iter.IterDataPipe = IterDataPipe

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
from freerec.data.tags import USER, ITEM, ID

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

from models.stair_sre_v6 import (
    DiagonalSpectralProjector,
    RegularizedDiagonalSpectralProjector,
    StepwiseSRELoss,
    StepwiseSREv2Loss,
    StepwiseSREv1_1Loss,
)

freerec.declare(version='0.8.5')

# =========================================================================
# Config: STAIR baseline args + SRE v6 args
# =========================================================================
cfg = freerec.parser.Parser()

# -- STAIR baseline args --
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3,
                 help="Number of layers for FSC/BSC")
cfg.add_argument("--mfiles", type=str,
                 default="textual_modality.pkl,visual_modality.pkl",
                 help="Comma-separated modality feature files")
cfg.add_argument("--num-neighbors", type=str, default='5-1',
                 help="kNN counts per modality, e.g. '5-1'")
cfg.add_argument("--gamma", type=float, default=0.2,
                 help="Spectral decay exponent for beta3")

# -- STAIR-SRE v6 specific args --
cfg.add_argument("--lambda-sre", type=float, default=1e-4,
                 help="Weight for SRE contrastive loss (0.0 = disabled)")
cfg.add_argument("--sre-tau", type=float, default=0.2,
                 help="Temperature for SRE InfoNCE loss")
cfg.add_argument("--sre-G", type=int, default=1,
                 help="Number of FSC layer gaps to contrast")
cfg.add_argument("--sre-alpha", type=float, default=0.5,
                 help="Balance between user-side and item-side CL")
cfg.add_argument("--sre-eps", type=float, default=0.1,
                 help="Spectral noise amplitude epsilon")
cfg.add_argument("--sre-tau-atten", type=float, default=0.35,
                 help="Threshold above which false negative attenuation is activated (default: 0.35 in v1.1)")
cfg.add_argument("--sre-swap-mode", type=str, default="cross_neg", choices=["cross_neg", "pos_neg"],
                 help="Spectral hard negative swapping mode: 'cross_neg' (v1.1 CNSS) or 'pos_neg' (v1 legacy)")
cfg.add_argument("--reg-w", type=float, default=1e-4,
                 help="Weight for Diagonal Spectral Projector anchoring loss lambda_w * ||w - 1||^2")
cfg.add_argument("--lr-proj", type=float, default=None,
                 help="Separate learning rate for spectral projector w (default: None -> cfg.lr * 0.1)")
cfg.add_argument("--sre-raw-hop", action="store_true", default=False,
                 help="Use raw A^g instead of smoothed for CL")
cfg.add_argument("--sre-debug", action="store_true", default=False,
                 help="Enable diagnostic logging for first few batches")

# Parse configurations
cfg.compile()

# Parse list fields
cfg.mfiles = cfg.mfiles.split(',')
cfg.num_neighbors = [int(k) for k in cfg.num_neighbors.split('-')]

if len(cfg.mfiles) != len(cfg.num_neighbors):
    raise ValueError(
        f"Length of mfiles ({len(cfg.mfiles)}) must match "
        f"num_neighbors ({len(cfg.num_neighbors)})"
    )

# BSC Smoother spectral decay beta3
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# =========================================================================
# STAIR-SRE v6 Model Class
# =========================================================================
class STAIR_SRE_Model(freerec.models.GenRecArch):
    """
    STAIR-SRE v6 Model:
    Combines STAIR Forward Stepwise Convolution with:
      1. Diagonal Spectral-scaling Projector (0-rotation Hadamard scaling)
      2. Soft Spectral Swapping (Bernoulli-based hard negative generation)
      3. Adaptive False Negative Attenuation (smooth (1-W) outside exp)
      4. Hierarchical Layer-wise Natural Contrastive Alignment (NLGCL heritage)
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

        # -- Pillar 1: Regularized Diagonal Spectral-scaling Projector --
        self.spectral_projector = DiagonalSpectralProjector(
            dim=cfg.embedding_dim,
            reg_weight=cfg.reg_w,
        )

        # -- Pillars 2-4: StepwiseSRELoss (v1.1) --
        beta_for_sre = (1.0 - cfg.beta3).to(cfg.device)
        self.sre_loss = StepwiseSRELoss(
            n_users   = self.User.count,
            n_items   = self.Item.count,
            beta      = beta_for_sre,
            G         = cfg.sre_G,
            tau       = cfg.sre_tau,
            alpha     = cfg.sre_alpha,
            eps       = cfg.sre_eps,
            tau_atten = cfg.sre_tau_atten,
            swap_mode = cfg.sre_swap_mode,
            debug     = cfg.sre_debug,
        )

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def marked_params(self):
        proj_lr = cfg.lr_proj if getattr(cfg, 'lr_proj', None) is not None else (cfg.lr * 0.1)
        return [
            {'params': self.User.parameters(), 'smoother': None},
            {
                'params': self.Item.parameters(),
                'smoother': Smoother(
                    self.mAdj, beta=cfg.beta3,
                    L=cfg.num_layers, aggr='neumann'
                ),
            },
            # Diagonal Spectral Projector params: anchored LR (0.1x) to prevent spectral drift
            {
                'params': self.spectral_projector.parameters(),
                'lr': proj_lr,
                'smoother': None,
            },
        ]

    def whitening(self, feats: torch.Tensor):
        """SVD Whitening — identical to STAIR baseline."""
        feats = feats - feats.mean(0, keepdim=True)
        _, S, V = torch.pca_lowrank(feats, q=cfg.embedding_dim, center=False)
        return feats @ V @ torch.diag(S.pow(-1))

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        features = F.normalize(features, p=2, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        return edge_index

    def prepare(self, path: str):
        """Modality Initialization + Profile Buffers for FN Attenuation."""
        try:
            from freerec.utils import import_pickle
        except (ImportError, AttributeError):
            import pickle
            def import_pickle(fpath):
                with open(fpath, 'rb') as f:
                    return pickle.load(f)

        mfeats = [
            import_pickle(os.path.join(path, mfile))
            for mfile in cfg.mfiles
        ]

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

        # MI: whitened modal feature initialization
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

        # Register raw modal features & user profiles for Adaptive FN Attenuation
        self.register_buffer('item_modals_raw', mfeats_init.detach().clone())
        self.register_buffer('user_profiles_raw', user_profiles_init.detach().clone())

    def sure_trainpipe(self, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_pairs_source()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

    # =========================================================================
    # encode(): FSC with Diagonal Spectral Projector + Layer Embeds Capture
    # =========================================================================
    def encode(self) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward Stepwise Convolution with:
        - Pillar 1: Diagonal Spectral Projector applied to Item embeddings
        - Layer embed capture for contrastive learning

        Returns:
            userEmbds:    (N_u, D) final aggregated user representations
            itemEmbds:    (N_i, D) final aggregated item representations
            layer_embeds: [H^0, H^1, ..., H^L] per-layer intermediates
        """
        # Apply Diagonal Spectral Projector to Item embeddings (0-rotation)
        item_embeds_proj = self.spectral_projector(
            self.Item.embeddings.weight
        )

        allEmbds = torch.cat(
            (self.User.embeddings.weight, item_embeds_proj),
            dim=0,
        )

        layer_embeds = [allEmbds]
        features = allEmbds
        smoothed = allEmbds

        # beta = 1 - beta3 for FSC propagation
        beta = (1.0 - self.beta3).to(allEmbds.device)
        norm_correction = 1.0 - beta ** (self.num_layers + 1)

        raw_hop = getattr(cfg, 'sre_raw_hop', False)
        if raw_hop:
            raw_features = allEmbds

        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features

            if raw_hop:
                raw_features = self.Adj @ raw_features
                layer_embeds.append(raw_features)
            else:
                layer_embeds.append(features)

        avgEmbds = smoothed.mul(1.0 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        return userEmbds, itemEmbds, layer_embeds

    def encode_for_eval(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Baseline-compatible encode for evaluation."""
        item_embeds_proj = self.spectral_projector(
            self.Item.embeddings.weight
        )
        allEmbds = torch.cat(
            (self.User.embeddings.weight, item_embeds_proj),
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

    # =========================================================================
    # fit(): Combines BPR + STAIR-SRE Losses
    # =========================================================================
    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        """
        Training step: L = L_BPR + lambda_sre * L_SRE
        """
        userEmbds, itemEmbds, layer_embeds = self.encode()

        users     = data[self.User]
        positives = data[self.Item]
        negatives = data[self.INeg]

        # -- Pairwise BPR Ranking Loss --
        rec_loss = self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[positives]),
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[negatives]),
        )

        # -- Pillar 1: Spectral Projector L2 Anchoring Loss --
        reg_w_loss = self.spectral_projector.get_anchoring_loss()

        # -- STAIR-SRE v1.1 Contrastive Loss --
        if self.training and cfg.lambda_sre > 0.0:
            beta = (1.0 - self.beta3).to(userEmbds.device)

            # Always provide user profiles and item modals for Adaptive FN Attenuation
            u_prof = self.user_profiles_raw[users.view(-1)]
            i_mod = self.item_modals_raw[positives.view(-1)]

            cl_loss = self.sre_loss(
                layer_embeds  = layer_embeds,
                users         = users,
                positives     = positives,
                beta          = beta,
                user_profiles = u_prof,
                item_modals   = i_mod,
            )
            return rec_loss + reg_w_loss + cfg.lambda_sre * cl_loss

        return rec_loss + reg_w_loss

    # =========================================================================
    # Evaluation Methods (IDENTICAL to STAIR baseline)
    # =========================================================================
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


# =========================================================================
# Coach (Training Loop)
# =========================================================================
class CoachForSTAIR_SRE(freerec.launcher.Coach):

    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.marked_params(), lr=self.cfg.lr,
                momentum=self.cfg.momentum, nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
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
        elif self.cfg.optimizer.lower() == 'adamsevo':
            self.optimizer = AdamSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adamwsevo':
            self.optimizer = AdamWSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"CoachForSTAIR_SRE does not support {self.cfg.optimizer} optimizer"
            )

    def train_per_epoch(self, epoch: int):
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

    # Robust dataset loading: freerec.data.datasets contains a submodule named 'tiktok',
    # so getattr(...) returns a module rather than a class when cfg.dataset == 'tiktok'.
    ds_cls = getattr(freerec.data.datasets, cfg.dataset, None)
    if isinstance(ds_cls, type):
        try:
            dataset = ds_cls(root=cfg.root)
        except Exception:
            dataset = freerec.data.datasets.RecDataSet(
                cfg.root, cfg.dataset, tasktag=getattr(cfg, "tasktag", None)
            )
    else:
        dataset = freerec.data.datasets.RecDataSet(
            cfg.root, cfg.dataset, tasktag=getattr(cfg, "tasktag", None)
        )

    model = STAIR_SRE_Model(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_SRE(
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
