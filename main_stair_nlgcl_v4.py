"""
main_stair_nlgcl_v4.py — STAIR-NLGCL v4 Training Script
=========================================================
STAIR Baseline + Zero-cost Neighborhood-enriched Graph Contrastive Learning.

This file is a self-contained training script following the exact same pattern
as STAIR's original main.py. It can be invoked identically:

    python main_stair_nlgcl_v4.py --config configs/Amazon2014Baby_550_MMRec.yaml

New hyperparameters (all with safe defaults):
    --lambda-nlgcl  : Weight for NLGCL loss term (default: 0.01, based on NLGCL+ multimodal SOTA)
    --nlgcl-tau     : Temperature for InfoNCE (default: 0.2)
    --nlgcl-G       : Number of contrastive gaps (default: 1)
    --nlgcl-alpha   : Balance between user-CL and item-CL (default: 0.5)
"""

from typing import Dict, List, Tuple

import torch, os, math
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

freerec.declare(version='0.8.5')

# ═══════════════════════════════════════════════════════════════════════════
# Config: STAIR baseline args + NLGCL-specific args
# ═══════════════════════════════════════════════════════════════════════════
cfg = freerec.parser.Parser()

# ── STAIR baseline args (identical to main.py) ──
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3,
                 help="Number of layers for FSC/BSC")
cfg.add_argument("--mfiles", type=str,
                 default="textual_modality.pkl,visual_modality.pkl",
                 help="Comma-separated modality feature files")
cfg.add_argument("--num-neighbors", type=str, default='5-1',
                 help="kNN counts per modality, e.g. '5-1'")
cfg.add_argument("--gamma", type=float, default=0.2,
                 help="Spectral decay exponent for β₃")

# ── NLGCL-specific args ──
cfg.add_argument("--lambda-nlgcl", type=float, default=0.1,
                 help="Weight for NLGCL contrastive loss (0.0 = disabled)")
cfg.add_argument("--nlgcl-tau", type=float, default=0.2,
                 help="Temperature τ for InfoNCE softmax")
cfg.add_argument("--nlgcl-G", type=int, default=1,
                 help="Number of contrastive gaps G (contrast layer g vs g+1)")
cfg.add_argument("--nlgcl-alpha", type=float, default=0.5,
                 help="Balance: α·L_user + (1-α)·L_item")
cfg.add_argument("--nlgcl-raw-hop", action="store_true", default=False,
                 help="Use raw graph hops (A^l E) for NLGCL instead of beta-damped FSC intermediates")

cfg.set_defaults(
    description="STAIR-NLGCL-v4",
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

# BSC Smoother spectral decay β₃
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ═══════════════════════════════════════════════════════════════════════════
# NLGCL Module (In-batch Heterogeneous Contrastive Learning)
# ═══════════════════════════════════════════════════════════════════════════
class NLGCL_Module(nn.Module):
    """
    Neighborhood-enriched Graph Contrastive Learning (NLGCL).
    
    Adapted from:
      - NLGCL-Plus/src/models/lgmrec_plus.py → neighbor_cl_loss(), InfoNCE()
      - NEGCL/NEGCL/models/negcl.py → ssl_triple_loss()
    
    Key adaptation for STAIR:
      - Uses FSC intermediate layer embeddings as "views" (zero-cost augmentation).
      - Contrasts across entity types (heterogeneous): User(L_g) ↔ Item(L_{g+1}).
      - In-batch negatives only (VRAM-safe: O(B²) instead of O(N²)).
    """

    def __init__(self, n_users: int, n_items: int,
                 G: int = 1, tau: float = 0.2, alpha: float = 0.5):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.G     = G
        self.tau   = tau
        self.alpha = alpha

    def info_nce_in_batch(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        In-batch InfoNCE contrastive loss.
        
        Math:
            L = -mean_i [ sim(a_i, p_i)/τ - logsumexp_j(sim(a_i, n_j)/τ) ]
        
        where sim(a, b) = (a/|a|) · (b/|b|) is cosine similarity.
        
        Uses torch.logsumexp for numerical stability (no exp overflow).
        
        Ref: NLGCL-Plus lgmrec_plus.py:InfoNCE() — adapted to in-batch variant.
        """
        anchor    = F.normalize(anchor,    p=2, dim=-1)  # (B, D)
        positive  = F.normalize(positive,  p=2, dim=-1)  # (B, D)
        negatives = F.normalize(negatives, p=2, dim=-1)  # (B, D)

        # Positive: (B,) — cosine similarity for each (anchor_i, positive_i) pair
        pos_sim = (anchor * positive).sum(dim=-1) / self.tau

        # Denominator: (B, B) — cosine similarity against all in-batch negatives
        neg_sim = torch.mm(anchor, negatives.t()) / self.tau

        # InfoNCE = pos - logsumexp(all_neg)
        # neg_sim already includes the positive at diagonal position
        loss = -pos_sim + torch.logsumexp(neg_sim, dim=-1)
        return loss.mean()

    def forward(
        self,
        layer_embeds: List[torch.Tensor],
        users: torch.Tensor,
        pos_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        In-batch Heterogeneous NLGCL loss from FSC layer intermediates.
        
        Follows the cross-entity contrastive pattern from NLGCL:
        
          For each gap g ∈ {0, ..., G-1}:
            L_u: Item(L_{g+1}) queries → User(L_g) keys  (in-batch users)
            L_i: User(L_{g+1}) queries → Item(L_g) keys  (in-batch items)
        
        Ref: NLGCL-Plus lgmrec_plus.py:neighbor_cl_loss()
        """
        users     = users.view(-1)
        pos_items = pos_items.view(-1)

        total_loss = torch.tensor(0.0, device=layer_embeds[0].device)
        num_gaps   = min(self.G, len(layer_embeds) - 1)

        for g in range(num_gaps):
            # Split joint embeddings [Users; Items] at each layer
            U_g, I_g   = torch.split(layer_embeds[g],     [self.n_users, self.n_items])
            U_g1, I_g1 = torch.split(layer_embeds[g + 1], [self.n_users, self.n_items])

            # User-side CL: propagated items query ego users
            # "Can the 1-hop item view find its interacting user in the batch?"
            cl_u = self.info_nce_in_batch(
                anchor    = I_g1[pos_items],
                positive  = U_g[users],
                negatives = U_g[users],
            )

            # Item-side CL: propagated users query ego items
            # "Can the 1-hop user view find its interacting item in the batch?"
            cl_i = self.info_nce_in_batch(
                anchor    = U_g1[users],
                positive  = I_g[pos_items],
                negatives = I_g[pos_items],
            )

            total_loss = total_loss + self.alpha * cl_u + (1.0 - self.alpha) * cl_i

        # Average over contrastive gaps (Eq. 6-7 in NLGCL / Eq. 12-13 in NLGCL+)
        # Ensures scaling G isolates architectural signal rather than scaling effective lambda.
        if num_gaps > 0:
            total_loss = total_loss / num_gaps

        return total_loss


# ═══════════════════════════════════════════════════════════════════════════
# STAIR-NLGCL Model
# ═══════════════════════════════════════════════════════════════════════════
class STAIR_NLGCL(freerec.models.GenRecArch):
    """
    STAIR-NLGCL v4: STAIR Baseline augmented with
    Neighborhood-enriched Graph Contrastive Learning (NLGCL).
    
    Architecture delta from STAIR baseline (main.py):
    ┌──────────────────────┬─────────────────────────────────────────────┐
    │ Method               │ Change                                      │
    ├──────────────────────┼─────────────────────────────────────────────┤
    │ __init__()           │ + NLGCL_Module instantiation                │
    │ encode()             │ + Captures intermediate layer_embeds        │
    │ fit()                │ + Adds λ·NLGCL_loss to BPR loss             │
    │ reset_ranking_buffers│ Uses encode_for_eval() (no layer_embeds)    │
    │ prepare()            │ IDENTICAL to STAIR baseline                 │
    │ marked_params()      │ IDENTICAL to STAIR baseline                 │
    │ whitening()          │ IDENTICAL to STAIR baseline                 │
    │ get_knn_graph()      │ IDENTICAL to STAIR baseline                 │
    │ recommend_*()        │ IDENTICAL to STAIR baseline                 │
    └──────────────────────┴─────────────────────────────────────────────┘
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

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

        # ── NLGCL Module ──
        self.nlgcl = NLGCL_Module(
            n_users = self.User.count,
            n_items = self.Item.count,
            G       = cfg.nlgcl_G,
            tau     = cfg.nlgcl_tau,
            alpha   = cfg.nlgcl_alpha,
        )

    # ─── Initialization (IDENTICAL to STAIR baseline) ───────────────────
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def marked_params(self):
        return [
            {'params': self.User.parameters(), 'smoother': None},
            {
                'params': self.Item.parameters(),
                'smoother': Smoother(
                    self.mAdj, beta=cfg.beta3,
                    L=cfg.num_layers, aggr='neumann'
                ),
            },
        ]

    def whitening(self, feats: torch.Tensor):
        """SVD Whitening — identical to STAIR baseline."""
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(
            self.Item.count / cfg.embedding_dim
        )

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        """kNN graph — identical to STAIR baseline."""
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        return edge_index

    def prepare(self, path: str):
        """Modality Initialization — identical to STAIR baseline."""
        from freerec.utils import import_pickle

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
        self.User.embeddings.weight.data.copy_(R @ mfeats_init)

    def sure_trainpipe(self, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_pairs_source()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

    # ═══════════════════════════════════════════════════════════════════
    # CORE CHANGE 1: encode() captures FSC intermediate layer embeddings
    # ═══════════════════════════════════════════════════════════════════
    def encode(self) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward Stepwise Convolution (FSC) with layer intermediate capture.
        
        Returns:
            userEmbds:    (N_u, D) — final aggregated user representations
            itemEmbds:    (N_i, D) — final aggregated item representations
            layer_embeds: [H⁰, H¹, H², H³] — per-layer intermediates for NLGCL
        
        The FSC computation (Neumann series aggregation with β₃ spectral decay)
        is IDENTICAL to STAIR baseline. The only addition is appending the
        intermediate `features` tensor to `layer_embeds` at each layer.
        """
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight),
            dim=0,
        )  # (N_u + N_i, D)

        # Layer 0: initial embeddings (ego representations)
        layer_embeds = [allEmbds]

        features = allEmbds
        smoothed = allEmbds

        # FSC: Neumann-series with dimension-wise spectral decay β₃
        beta = (1 - cfg.beta3).to(allEmbds.device)
        norm_correction = 1 - beta ** (self.num_layers + 1)

        raw_hop = getattr(cfg, 'nlgcl_raw_hop', False)
        if raw_hop:
            raw_features = allEmbds

        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features

            # Layer l: l-hop graph-propagated features (zero-cost "view")
            if raw_hop:
                raw_features = self.Adj @ raw_features
                layer_embeds.append(raw_features)
            else:
                layer_embeds.append(features)

        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        return userEmbds, itemEmbds, layer_embeds

    def encode_for_eval(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Baseline-compatible encode for evaluation (no layer_embeds overhead)."""
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight),
            dim=0,
        )
        features = allEmbds
        smoothed = allEmbds
        beta = (1 - cfg.beta3).to(allEmbds.device)
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        return torch.split(avgEmbds, (self.User.count, self.Item.count))

    # ═══════════════════════════════════════════════════════════════════
    # CORE CHANGE 2: fit() combines BPR + NLGCL losses
    # ═══════════════════════════════════════════════════════════════════
    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        """
        Training forward pass: L = L_BPR + λ_nlgcl · L_NLGCL
        
        BPR Loss: Identical to STAIR baseline (pairwise ranking loss).
        NLGCL Loss: Zero-cost contrastive learning from FSC intermediates.
        """
        userEmbds, itemEmbds, layer_embeds = self.encode()

        users     = data[self.User]
        positives = data[self.Item]
        negatives = data[self.INeg]

        # ── BPR Loss (identical to STAIR baseline) ──
        rec_loss = self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[positives]),
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[negatives]),
        )

        # ── NLGCL Loss (from FSC layer intermediates, zero additional cost) ──
        if self.training and cfg.lambda_nlgcl > 0.0:
            nlgcl_loss = self.nlgcl(layer_embeds, users, positives)
            return rec_loss + cfg.lambda_nlgcl * nlgcl_loss

        return rec_loss

    # ═══════════════════════════════════════════════════════════════════
    # Evaluation methods (IDENTICAL to STAIR baseline)
    # ═══════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════
# Coach (Training Loop — identical to STAIR baseline)
# ═══════════════════════════════════════════════════════════════════════════
class CoachForSTAIR_NLGCL(freerec.launcher.Coach):

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

    model = STAIR_NLGCL(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_NLGCL(
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
