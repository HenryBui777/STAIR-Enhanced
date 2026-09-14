"""
main_enhanced_v2.py — STAIR-Enhanced v2a (ResOnly): Residual-Whitening Projector
================================================================================
Chien luoc: Giu nguyen SVD Whitening (Structural Prior, frozen) + them nhanh
Residual Projector hoc phan correction phi tuyen truc tiep trong computational graph.

    e_i^{modal} = L2Norm(E_svd_i + lambda_res * Delta_i) * sqrt(N/D)
    e_i         = e_i^{ID} + e_i^{modal}

Dac diem ky thuat quan trong (Khac phuc triet de loi lambda_res bi ket):
  1. res_projector va lambda_res duoc goi TRUC TIEP trong encode() o moi forward pass
     -> BPR loss tinh gradient chuan xac cho lambda_res va cac layer MLP.
  2. E_svd duoc luu lam frozen buffer (Structural Prior goc khong bi pha vo).
  3. text_feat va visual_feat duoc luu lam buffer tren GPU/device.
  4. Tai epoch 0, e_i^{ID} ~ 0 va Delta ~ 0 (nho warm-start SVD)
     -> mo hinh bat dau CHINH XAC bang baseline SVD Whitening goc.
  5. Trong qua trinh train, optimizer AdamWSEvo tu dong hoc:
     - User/Item ID embeddings voi Smoother tren mAdj (lr=1e-3)
     - res_projector va lambda_res voi learning rate doc lap (lr_proj=1e-3 / 2e-3)
  6. Log chi tiet lambda_res, grad norm qua moi epoch de giam sat truc quan.
"""

from typing import Dict, Tuple

import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import freerec

from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother
from models.residual_projector_v2 import ResidualWhiteningProjector

freerec.declare(version='1.0.1')

# ============================================================================
# Config
# ============================================================================
cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim",  type=int,   default=64)
cfg.add_argument("--num-layers",     type=int,   default=3)
cfg.add_argument("--mfiles",         type=str,   default="textual_modality.pkl,visual_modality.pkl")
cfg.add_argument("--num-neighbors",  type=str,   default='5-1')
cfg.add_argument("--gamma",          type=float, default=0.2)
# V2 Hyperparameters:
cfg.add_argument("--lr-proj",        type=float, default=1e-3,
                 help="Learning rate cho Residual Projector (default: 1e-3 de on dinh)")
cfg.add_argument("--lambda-init",    type=float, default=0.1,
                 help="Gia tri khoi tao cua lambda_res (default: 0.1)")

cfg.set_defaults(
    description="EnhancedSTAIR-v2a",
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

# beta3: spectral decay parameter cho BSC Smoother (dong nhat voi main.py)
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ============================================================================
# Model: EnhancedSTAIR_v2a (Dynamic Residual-Whitening Architecture)
# ============================================================================
class EnhancedSTAIR_v2a(freerec.models.GenRecArch):
    """
    STAIR voi Dynamic Residual Projector (v2a — ResOnly).

    SVD Whitening (E_svd) dong vai tro Structural Prior bat bien (frozen buffer).
    Residual Projector tinh toan Delta_i va ket hop voi lambda_res trong computational graph.
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)
        self.num_layers = cfg.num_layers

        self.User.add_module('embeddings', nn.Embedding(self.User.count, cfg.embedding_dim))
        self.Item.add_module('embeddings', nn.Embedding(self.Item.count, cfg.embedding_dim))

        self.register_buffer(
            'Adj',
            self.dataset.train().to_normalized_adj(normalization='sym')
        )

        # Module v2a: Residual Projector (co chua nn.Parameter lambda_res)
        self.res_projector = ResidualWhiteningProjector(
            d_text=384,
            d_visual=4096,
            d_hidden=cfg.embedding_dim,
            lambda_init=cfg.lambda_init,
        )

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

    def reset_parameters(self):
        """Khoi tao nho cho Item ID embeddings, Kaiming cho projector."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)

    def marked_params(self):
        """
        Param groups cho AdamWSEvo:
          - User: standard lr, no smoother
          - Item ID: standard lr, with Smoother on mAdj
          - res_projector: custom lr (cfg.lr_proj), no smoother
        """
        return [
            {
                'params': self.User.parameters(),
                'smoother': None,
            },
            {
                'params': self.Item.parameters(),
                'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann'),
            },
            {
                'params': self.res_projector.text_proj.parameters(),
                'smoother': None,
                'lr': cfg.lr_proj,
                'weight_decay': 1e-4,
            },
            {
                'params': self.res_projector.vis_proj.parameters(),
                'smoother': None,
                'lr': cfg.lr_proj,
                'weight_decay': 1e-4,
            },
            {
                'params': [self.res_projector.lambda_raw],
                'smoother': None,
                'lr': cfg.lr_proj * 0.5,
                'weight_decay': 0.1,
            },
        ]

    def whitening(self, feats: torch.Tensor) -> torch.Tensor:
        """SVD Whitening goc cua STAIR."""
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        """Tinh kNN graph tu cosine similarity."""
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        return edge_index

    def prepare(self, path: str):
        """
        Khoi tao cac buffer va warm-start projector:
          1. Tinh mAdj (kNN graph)
          2. Tinh E_svd (SVD whitening goc, luu buffer bat bien)
          3. Warm-start projector W_t, W_v tu raw features
          4. Luu text_feat, visual_feat tren device lam buffer cho forward pass
          5. Khoi tao User embeddings qua R @ E_svd
        """
        from freerec.utils import import_pickle

        mfeats_raw = [
            import_pickle(os.path.join(path, mfile))
            for mfile in cfg.mfiles
        ]
        text_feat   = mfeats_raw[0].float()   # (N, 384)
        visual_feat = mfeats_raw[1].float()   # (N, 4096)

        # ---- 1. Xay mAdj (kNN graph) ----
        edge_index = torch.cat(
            [self.get_knn_graph(feats, k) for feats, k in zip(mfeats_raw, cfg.num_neighbors)],
            dim=1
        )
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce='sum')
        edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce='max')
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, edge_weight, normalization='sym')
        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.Item.count, self.Item.count)
        )
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        # ---- 2. Tinh E_svd (Structural Prior bat bien) ----
        e_svd_parts = [self.whitening(feat) * k for feat, k in zip(mfeats_raw, cfg.num_neighbors)]
        E_svd = sum(e_svd_parts) / sum(cfg.num_neighbors)   # (N, 64)
        self.register_buffer('E_svd', E_svd.to(cfg.device))
        print(f'[prepare] E_svd: shape={tuple(self.E_svd.shape)}, mean_norm={self.E_svd.norm(dim=-1).mean():.3f}')

        # ---- 3. Warm-start Projector tu raw features ----
        print('[prepare] Running warm-start SVD for residual projector...')
        self.res_projector.init_warm_start_weights(text_feat, visual_feat)
        self.res_projector.to(cfg.device)

        # ---- 4. Dang ky raw features len device de tinh dynamic forward pass ----
        self.register_buffer('text_feat', text_feat.to(cfg.device))
        self.register_buffer('visual_feat', visual_feat.to(cfg.device))

        # ---- 5. User init theo R @ E_svd (dong nhat voi STAIR goc) ----
        edge_index_ui = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index_ui, edge_weight_ui = freerec.graph.to_normalized(
            edge_index_ui, normalization='left'
        )
        R = torch.sparse_coo_tensor(
            edge_index_ui, edge_weight_ui,
            size=(self.User.count, self.Item.count)
        ).to_sparse_csr().to(cfg.device)
        self.User.embeddings.weight.data.copy_(R @ self.E_svd)

        # Item ID embeddings khoi tao bang 0 (de tai epoch 0, itemEmbds = E_svd hoan toan)
        self.Item.embeddings.weight.data.zero_()
        print(f'[prepare] User & Item embeddings initialized. Ready for dynamic residual training!')

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_pairs_source(
        ).gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def get_modal_item_embeddings(self) -> torch.Tensor:
        """
        Tinh dac trung modal cua Item theo co che Dynamic Residual v2a_bounded:
          - Magnitude Matching: Delta duoc scale cung do lon voi E_svd
          - Bounded Sigmoid: lambda = max_lambda * sigmoid(lambda_raw)
        """
        # Delta: (N, 64), L2-normalized = 1
        delta = self.res_projector(self.text_feat, self.visual_feat)

        # Ep scale cua Delta bang dung scale cua E_svd
        scale = math.sqrt(self.Item.count / cfg.embedding_dim)
        delta_scaled = delta * scale

        # Tinh lambda thuc te qua cong Sigmoid (max 0.3)
        actual_lambda = self.res_projector.max_lambda * torch.sigmoid(self.res_projector.lambda_raw)

        # Cong thuc Residual thuc su
        e_comb = self.E_svd + actual_lambda * delta_scaled

        # L2 Normalize lan cuoi va scale lai
        e_modal = F.normalize(e_comb, p=2, dim=-1, eps=1e-12) * scale
        return e_modal

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FSC/BSC smoothing:
          - Item representation = ID embeddings (smoothed) + Dynamic Modal residual
          - User representation = User embeddings (smoothed)
        """
        modal_items = self.get_modal_item_embeddings() # (N_I, 64)
        total_items = self.Item.embeddings.weight + modal_items

        allEmbds  = torch.cat((self.User.embeddings.weight, total_items), dim=0)
        features  = allEmbds
        smoothed  = allEmbds
        beta      = 1 - cfg.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(avgEmbds, (self.User.count, self.Item.count))
        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        """BPR loss voi gradient lan truyen ve ca User/Item embeddings va res_projector."""
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        userEmbds = userEmbds[users]
        iposEmbds = itemEmbds[positives]
        inegEmbds = itemEmbds[negatives]
        return self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds, iposEmbds),
            torch.einsum('BKD,BKD->BK', userEmbds, inegEmbds),
        )

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum('BKD,ND->BN', userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum('BKD,BKD->BK', userEmbds, itemEmbds)


# ============================================================================
# Coach: AdamWSEvo + Param Groups + lambda_res monitoring
# ============================================================================
class CoachForEnhancedSTAIR_v2a(freerec.launcher.Coach):
    """
    Coach cho v2a:
      - AdamWSEvo quan ly User, Item ID (smoothed) va res_projector (unsmoothed)
      - Theo doi su thay doi cua lambda_res va gradient norm sau moi epoch
    """

    def set_optimizer(self):
        self.optimizer = AdamWSEvo(
            self.model.marked_params(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )
        print(f'[Optimizer] AdamWSEvo ready (base lr={self.cfg.lr}, lr_proj={self.cfg.lr_proj})')

    def train_per_epoch(self, epoch: int):
        self.model.res_projector.train()
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            
            # Warm-up lambda_raw: dong bang trong 50 epoch dau
            if epoch <= 50:
                if self.model.res_projector.lambda_raw.grad is not None:
                    self.model.res_projector.lambda_raw.grad.zero_()
            
            self.optimizer.step()

            self.monitor(
                loss.item(), n=len(data[self.User]),
                reduction='mean', mode='train', pool=['LOSS'],
            )

        # Log chi tiet lambda_raw sau moi epoch de nguoi dung theo doi
        import torch
        actual_lambda = self.model.res_projector.max_lambda * torch.sigmoid(self.model.res_projector.lambda_raw).item()
        lam_raw_val = self.model.res_projector.lambda_raw.item()
        grad_val = self.model.res_projector.lambda_raw.grad
        grad_str = f'{grad_val.item():.6e}' if grad_val is not None else 'None'
        print(f'  [lambda_res @epoch {epoch:3d}]: actual={actual_lambda:.6f} | raw={lam_raw_val:.6f} | grad={grad_str}', flush=True)


# ============================================================================
# Entry Point
# ============================================================================
def main():
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

    model = EnhancedSTAIR_v2a(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForEnhancedSTAIR_v2a(
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
