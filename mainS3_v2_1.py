# -*- coding: utf-8 -*-
"""
mainS3_v2_1.py — STAIR-SRE-ANS v2.1 Official Training Pipeline
===========================================================
Stepwise Spectral-Refined Contrastive Learning with Adaptive Negative Scheduling
and Continuous Spectral Difficulty Decoupling (Phase 3 -- Batch 2.1 / v2.1 (Decoupled Layer-0 CL & Thresholded Gating))

5 Technology Pillars Integrated:
  Pillar 1: Regularized Diagonal Spectral Projector (0-rotation)
  Pillar 2: Continuous Spectral Difficulty Decoupling (No hard dim-32 boundary)
  Pillar 3: Gated Top-K Selection (Selection_Score = Difficulty * Attenuation)
  Pillar 4: Thresholded Cosine-Gated MFNA (M_gated = Meta * max(0, cos))
  Pillar 5: Decoupled HANS Scheduler & Training-Guarded FIFO Memory Bank Q

Target Performance (Breakthrough >= +5.0% synchronously across all 3 datasets):
  Amazon Baby:        Recall@20 >= 0.1095 (+5.09%) | NDCG@20 >= 0.0480 (+5.73%)
  Amazon Sports:      Recall@20 >= 0.1168 (+5.13%) | NDCG@20 >= 0.0530 (+6.00%)
  Amazon Electronics: Recall@20 >= 0.0705 (+6.02%) | NDCG@20 >= 0.0325 (+7.26%)
"""

from typing import Dict, Tuple, List, Optional
import os
import sys
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# =========================================================================
# Compatibility Patch for torchdata in PyTorch 2.x / Python 3.12 / Kaggle
# =========================================================================
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

from models.stair_sre_ans_v2_1 import (
    RegularizedDiagonalSpectralProjector,
    StepwiseSREANSLoss_v21,
)

freerec.declare(version='0.8.5')

# =========================================================================
# Config Parser: STAIR Baseline Args + STAIR-SRE-ANS v2 Args
# =========================================================================
cfg = freerec.parser.Parser()

# -- STAIR Baseline Parameters --
cfg.add_argument("--embedding-dim", type=int, default=64,
                 help="Latent vector embedding dimension D")
cfg.add_argument("--num-layers", type=int, default=3,
                 help="Number of layers L for Forward/Backward Stepwise Convolution")
cfg.add_argument("--mfiles", type=str,
                 default="textual_modality.pkl,visual_modality.pkl",
                 help="Comma-separated multimodal feature file paths")
cfg.add_argument("--num-neighbors", type=str, default='5-1',
                 help="kNN neighbor counts per modality, e.g. '5-1'")
cfg.add_argument("--gamma", type=float, default=0.2,
                 help="Spectral decay power gamma for beta3 (default: 0.2, overridden by dataset YAML)")

# -- STAIR-SRE-ANS v2 Specific Parameters --
cfg.add_argument("--lambda-ans", type=float, default=5e-5,
                 help="Weight lambda_ans for SRE-ANS contrastive loss (default: 5e-5)")
cfg.add_argument("--ans-tau", type=float, default=0.20,
                 help="InfoNCE temperature tau for SRE-ANS loss (default: 0.20)")
cfg.add_argument("--queue-size", type=int, default=4096,
                 help="Capacity of cross-batch FIFO Memory Bank Q (default: 4096, 8192 for Electronics)")
cfg.add_argument("--warmup-epochs", type=int, default=50,
                 help="Number of bootstrap warmup epochs before activating HANS (default: 50)")
cfg.add_argument("--gamma-max", type=float, default=0.35,
                 help="Ceiling cap for hard negative penalty coefficient gamma_h (default: 0.35)")
cfg.add_argument("--hn-ratio-max", type=float, default=0.40,
                 help="Ceiling cap for hard negative selection ratio hn_ratio (default: 0.40)")
cfg.add_argument("--subspace-alpha", type=float, default=0.50,
                 help="Balance between low-frequency and high-frequency spectral bands (default: 0.50)")
cfg.add_argument("--reg-w", type=float, default=1e-4,
                 help="L2 anchoring regularization weight lambda_w * ||w - 1||^2 (default: 1e-4)")
cfg.add_argument("--lr-proj", type=float, default=None,
                 help="Dedicated learning rate for spectral projector w (default: None -> cfg.lr * 0.1)")
cfg.add_argument("--hans-window", type=int, default=10,
                 help="Sliding window size for HANS loss-gated trigger (default: 10, 5 for Electronics)")
cfg.add_argument("--ans-debug", action="store_true", default=False,
                 help="Enable diagnostic logging for first few batches")

# Parse CLI and YAML configs
cfg.compile()

# Parse list fields
cfg.mfiles = cfg.mfiles.split(',')
cfg.num_neighbors = [int(k) for k in cfg.num_neighbors.split('-')]

if len(cfg.mfiles) != len(cfg.num_neighbors):
    raise ValueError(
        f"Length of mfiles ({len(cfg.mfiles)}) must match "
        f"num_neighbors ({len(cfg.num_neighbors)})"
    )

# Backward Stepwise Convolution Smoother spectral decay beta3
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / float(cfg.embedding_dim)).pow(cfg.gamma)
).to(cfg.device)


# =========================================================================
# STAIR_SRE_ANS_v2_1 Model Class
# =========================================================================
class STAIR_SRE_ANS_v2_1(freerec.models.GenRecArch):
    """
    STAIR-SRE-ANS v2 Model:
    Integrates Forward Stepwise Convolution with 5 Pillars:
      1. Regularized Diagonal Spectral Projector (0-rotation, L2 anchoring)
      2. Continuous Spectral Difficulty Decoupling (No hard dim-32 boundary)
      3. Gated Top-K Selection (Selection_Score = Difficulty * Attenuation)
      4. Thresholded Cosine-Gated MFNA (M_gated = Meta * max(0, cos))
      5. Decoupled HANS Scheduler & Training-Guarded FIFO Memory Bank Q
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super(STAIR_SRE_ANS_v2_1, self).__init__(dataset)
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

        # -- Pillar 1: Regularized Diagonal Spectral Projector --
        self.spectral_projector = RegularizedDiagonalSpectralProjector(
            dim=cfg.embedding_dim,
            reg_weight=cfg.reg_w,
        )

        # -- Pillars 2-5: Stepwise SRE-ANS v2 Loss --
        # Dynamically inject the exact FSC spectral curve (1.0 - beta3) to guarantee 100% mathematical consistency
        self.ans_loss = StepwiseSREANSLoss_v21(
            dim=cfg.embedding_dim,
            tau=cfg.ans_tau,
            queue_size=cfg.queue_size,
            warmup_epochs=cfg.warmup_epochs,
            total_epochs=cfg.epochs,
            gamma_max=cfg.gamma_max,
            hn_ratio_max=cfg.hn_ratio_max,
            subspace_alpha=cfg.subspace_alpha,
            beta=(1.0 - self.beta3),
            gamma=cfg.gamma,
        )

        # State tracking for decoupled HANS monitoring
        self.last_cl_loss: Optional[float] = None

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def marked_params(self):
        """
        Assigns customized optimization parameters:
        - Projector w gets lower learning rate (0.1x) to prevent coordinate drift.
        - Item embeddings use Neumann Smoother with beta3 according to STAIR BSC.
        """
        proj_lr = cfg.lr_proj if getattr(cfg, 'lr_proj', None) is not None else (cfg.lr * 0.1)
        params_groups = [
            {'params': self.User.parameters(), 'smoother': None},
            {
                'params': self.Item.parameters(),
                'smoother': Smoother(
                    self.mAdj, beta=cfg.beta3,
                    L=cfg.num_layers, aggr='neumann'
                ),
            },
            {
                'params': self.spectral_projector.parameters(),
                'lr': proj_lr,
                'smoother': None,
            },
        ]
        if hasattr(self.ans_loss, 'proj_head'):
            params_groups.append({
                'params': self.ans_loss.proj_head.parameters(),
                'lr': cfg.lr,
                'smoother': None,
            })
        return params_groups

    def whitening(self, feats: torch.Tensor):
        """SVD Whitening -- strictly identical to STAIR baseline (torch.linalg.svd)."""
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        features = F.normalize(features, p=2, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.0)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        return edge_index

    def prepare(self, path: str):
        """Modality Initialization + Profile Buffers."""
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

        # SVD Whitened modal feature initialization
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

    def sure_trainpipe(self, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_pairs_source()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Stepwise Convolution with Diagonal Spectral Projector:
        Returns:
            userEmbds: (N_u, D) aggregated user embeddings
            itemEmbds: (N_i, D) aggregated item embeddings
        """
        # Element-wise scaling on Item embeddings (0-rotation)
        item_embeds_proj = self.spectral_projector(
            self.Item.embeddings.weight
        )

        allEmbds = torch.cat(
            (self.User.embeddings.weight, item_embeds_proj),
            dim=0,
        )

        features = allEmbds
        smoothed = allEmbds

        # beta = 1 - beta3 for FSC forward propagation
        beta = (1.0 - self.beta3).to(allEmbds.device)
        norm_correction = 1.0 - beta ** (self.num_layers + 1)

        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features

        avgEmbds = smoothed.mul(1.0 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        return userEmbds, itemEmbds

    def encode_for_eval(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluation encode matching STAIR baseline."""
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

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        """
        Computes training loss: L_total = L_BPR + L_reg_w + lambda_ans * L_ANS
        """
        userEmbds, itemEmbds = self.encode()

        users     = data[self.User]
        positives = data[self.Item]
        negatives = data[self.INeg]

        # 1. Pairwise BPR Ranking Loss
        rec_loss = self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[positives]),
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[negatives]),
        )

        # 2. Pillar 1: Spectral Projector L2 Anchoring Loss
        reg_w_loss = self.spectral_projector.get_anchoring_loss()

        # 3. Decoupled Contrastive Loss on Layer-0 Latent Space (STAIR-SRE-ANS v2.1)
        # Keeps H^(L) 100% for BPR ranking; optimizes representation alignment on raw features
        if self.training and cfg.lambda_ans > 0.0:
            item_raw_proj = self.spectral_projector(self.Item.embeddings.weight)
            user_raw = self.User.embeddings.weight

            cl_loss = self.ans_loss(
                u_raw       = user_raw,
                i_raw       = item_raw_proj,
                batch_users = users,
                batch_items = positives,
            )
            self.last_cl_loss = float(cl_loss.detach().item())
            return rec_loss + reg_w_loss + cfg.lambda_ans * cl_loss

        self.last_cl_loss = None
        return rec_loss + reg_w_loss

    # =========================================================================
    # Evaluation Buffers (IDENTICAL to STAIR Baseline)
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
# CoachForSTAIR_SRE_ANS_v2_1 (Training Loop with Decoupled HANS Scheduling)
# =========================================================================
class CoachForSTAIR_SRE_ANS_v2_1(freerec.launcher.Coach):

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
                f"CoachForSTAIR_SRE_ANS_v2_1 does not support {self.cfg.optimizer} optimizer"
            )

    def train_per_epoch(self, epoch: int):
        self.model.train()
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

            # Accumulate contrastive loss for decoupled HANS scheduling
            if hasattr(self.model, 'last_cl_loss') and self.model.last_cl_loss is not None:
                total_cl_loss += self.model.last_cl_loss
                cl_batches += 1

        # ------------------------------------------------------------------
        # Update HANS Scheduler at Epoch Boundary
        # ------------------------------------------------------------------
        if cl_batches > 0:
            avg_cl_loss = total_cl_loss / float(cl_batches)
            window_size = getattr(self.cfg, 'hans_window', 10)
            self.model.ans_loss.update_scheduler(
                current_cl_loss=avg_cl_loss,
                window=window_size,
                threshold=0.99,
            )

            # Diagnostic logging every 10 epochs or at epoch 0
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  [HANS Epoch {epoch + 1:03d}] gamma_h: {self.model.ans_loss.gamma_h:.4f} | "
                    f"hn_ratio: {self.model.ans_loss.hn_ratio:.4f} | "
                    f"avg_cl_loss: {avg_cl_loss:.6f}"
                )


def main():
    # Robust auto-bridge for FreeRec:
    # FreeRec expects data in os.path.join(cfg.root, "Processed", cfg.dataset).
    # If it is located in cfg.root/{cfg.dataset} or any other standard location, symlink or copy it
    # so FreeRec never triggers fragile Zenodo downloads that return 403 Forbidden.
    processed_dir = os.path.join(cfg.root, "Processed", cfg.dataset)
    if not os.path.exists(processed_dir) or not os.listdir(processed_dir):
        candidates = [
            os.path.join(cfg.root, cfg.dataset),
            os.path.join("/kaggle/data", cfg.dataset),
            os.path.join("/kaggle/data/Processed", cfg.dataset),
            os.path.join("data", cfg.dataset),
            os.path.join("data/Processed", cfg.dataset),
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

    model = STAIR_SRE_ANS_v2_1(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_SRE_ANS_v2_1(
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
