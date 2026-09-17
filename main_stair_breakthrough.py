# -*- coding: utf-8 -*-
"""
main_stair_breakthrough.py
==========================
Runner Huấn Luyện Cho 3 Phương Pháp Đột Phá Mở Rộng:
1. --method dan_tans:        Degree-Aware Noise & Topology-Aware Negative Scheduling
2. --method dcd_gated:       Dual-Consensus Denoising & Gated Residuals
3. --method appnp_crossmodal: APPNP-Restart Propagation & Disentangled Cross-Modal CL
4. --method v5_plus:         STAIR-NE-NLGCL v5+ (Mốc đối chứng SOTA)

Tối ưu hóa chuyên biệt cho Amazon Baby & Amazon Sports.
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
            return cls
        return decorator
    dp.functional_datapipe = functional_datapipe

import freerec
from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

from models.stair_ne_nlgcl_v5_plus import STAIR_NE_NLGCL_v5_Plus
from models.stair_breakthrough import (
    STAIR_DAN_TANS_Module,
    STAIR_DCD_Gated_Module,
    STAIR_APPNP_CrossModal_Module,
)

freerec.declare(version='0.8.5')

# ═════════════════════════════════════════════════════════════════════════════
# Configuration Setup
# ═════════════════════════════════════════════════════════════════════════════
cfg = freerec.parser.Parser()

cfg.add_argument("--method", type=str, default="dan_tans",
                 choices=["dan_tans", "dcd_gated", "appnp_crossmodal", "v5_plus"],
                 help="Phương pháp thực nghiệm lựa chọn")
cfg.add_argument("--embedding-dim", type=int, default=256,
                 help="Latent vector embedding dimension D (default: 256)")
cfg.add_argument("--num-layers", type=int, default=3)
cfg.add_argument("--mfiles", type=str, default="textual_modality.pkl,visual_modality.pkl")
cfg.add_argument("--num-neighbors", type=str, default='5-1')
cfg.add_argument("--gamma", type=float, default=0.2)

# Cosine LR Warmup & Decay + Early Stopping
cfg.add_argument("--lr-warmup-epochs", type=int, default=15,
                 help="Warmup epochs cho Learning Rate (khuyên dùng 10-20 epoch, default: 15)")
cfg.add_argument("--min-lr", type=float, default=1e-6,
                 help="Tốc độ học tối thiểu sau khi Cosine Decay (default: 1e-6)")
cfg.add_argument("--patience", type=int, default=30,
                 help="Patience cho Early Stopping (default: 30)")
cfg.add_argument("--early-stop-patience", type=int, default=30,
                 help="Patience tích hợp trong FreeRec (default: 30)")
cfg.add_argument("--target-metric", type=str, default="NDCG@20",
                 help="Chỉ số mục tiêu để chọn best checkpoint và early stopping (default: NDCG@20)")

# CL Params
cfg.add_argument("--tau", type=float, default=0.20)
cfg.add_argument("--alpha-dir", type=float, default=0.50)
cfg.add_argument("--eps", type=float, default=0.08)
cfg.add_argument("--tau-thresh", type=float, default=0.85)
cfg.add_argument("--lambda-cl", type=float, default=0.010)
cfg.add_argument("--gamma-h", type=float, default=0.15)
cfg.add_argument("--warmup-epochs", type=int, default=50,
                 help="Warmup epochs cho Contrastive Loss Weight (default: 50)")

# Params riêng cho Hướng 3 (APPNP + CrossModal)
cfg.add_argument("--alpha-restart", type=float, default=0.15, help="Hệ số teleport APPNP")
cfg.add_argument("--lambda-cross", type=float, default=0.005, help="Trọng số cross-modal CL")

cfg.set_defaults(
    description="STAIR-Breakthrough-Runner",
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

cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ═════════════════════════════════════════════════════════════════════════════
# Model Architecture Integrating the 3 Breakthrough Strategies
# ═════════════════════════════════════════════════════════════════════════════
class STAIR_Breakthrough_Model(freerec.models.RecSysProxy):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.User = dataset.fields[freerec.data.fields.USER]
        self.Item = dataset.fields[freerec.data.fields.ITEM]
        self.INeg = dataset.fields[freerec.data.fields.INEG]

        self.num_layers = cfg.num_layers
        self.method = cfg.method

        # User and Item basic embeddings
        self.User.embeddings = nn.Embedding(self.User.count, cfg.embedding_dim)
        self.Item.embeddings = nn.Embedding(self.Item.count, cfg.embedding_dim)

        self.criterion = freerec.criterions.BPR()

        # Khởi tạo module tương ứng với method
        if self.method == 'dan_tans':
            self.cl_module = STAIR_DAN_TANS_Module(
                n_users       = self.User.count,
                n_items       = self.Item.count,
                tau           = cfg.tau,
                alpha_dir     = cfg.alpha_dir,
                eps_base      = cfg.eps,
                tau_thresh    = cfg.tau_thresh,
                lambda_cl     = cfg.lambda_cl,
                gamma_base    = cfg.gamma_h,
                warmup_epochs = cfg.warmup_epochs,
            )
        elif self.method == 'dcd_gated':
            self.cl_module = STAIR_DCD_Gated_Module(
                n_users       = self.User.count,
                n_items       = self.Item.count,
                embedding_dim = cfg.embedding_dim,
                tau           = cfg.tau,
                eps           = cfg.eps,
                tau_thresh    = cfg.tau_thresh,
                lambda_cl     = cfg.lambda_cl,
                gamma_h       = cfg.gamma_h,
                warmup_epochs = cfg.warmup_epochs,
            )
        elif self.method == 'appnp_crossmodal':
            self.cl_module = STAIR_APPNP_CrossModal_Module(
                n_users       = self.User.count,
                n_items       = self.Item.count,
                tau           = cfg.tau,
                eps           = cfg.eps,
                lambda_cl     = cfg.lambda_cl,
                lambda_cross  = cfg.lambda_cross,
                alpha_restart = cfg.alpha_restart,
                warmup_epochs = cfg.warmup_epochs,
            )
        else: # v5_plus baseline
            self.cl_module = STAIR_NE_NLGCL_v5_Plus(
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
        self.s_conf_sparse: Optional[torch.Tensor] = None

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)

    def marked_params(self):
        return [
            {'params': self.User.parameters(), 'smoother': None},
            {'params': self.Item.parameters(), 'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann')},
        ]

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
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
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
            [self.get_knn_graph(feats, k) for feats, k in zip(mfeats, cfg.num_neighbors)],
            dim=1
        )
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce='sum')
        edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce='max')
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, edge_weight, normalization='sym')
        mAdj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(self.Item.count, self.Item.count))
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        mfeats_w = [self.whitening(mfeat) * k for mfeat, k in zip(mfeats, cfg.num_neighbors)]
        mfeats_init = sum(mfeats_w).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats_init)

        edge_index_ui = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index_ui, edge_weight_ui = freerec.graph.to_normalized(edge_index_ui, normalization='left')
        R = torch.sparse_coo_tensor(edge_index_ui, edge_weight_ui, size=(self.User.count, self.Item.count)).to_sparse_csr()
        user_profiles_init = R @ mfeats_init
        self.User.embeddings.weight.data.copy_(user_profiles_init)

        self.register_buffer('item_modals_raw', mfeats_init.detach().clone())

        # ── Setup Bổ sung cho từng Method ──
        raw_edge_ui = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        u_deg_cpu = raw_edge_ui[0].bincount(minlength=self.User.count)
        i_deg_cpu = raw_edge_ui[1].bincount(minlength=self.Item.count)
        u_deg = u_deg_cpu.to(cfg.device)
        i_deg = i_deg_cpu.to(cfg.device)

        if self.method == 'dan_tans':
            self.cl_module.set_degrees(u_deg, i_deg)

        elif self.method == 'dcd_gated':
            with torch.no_grad():
                # ── True Zero-OOM Chunked Dual Consensus Engine (Thích ứng từ Baby -> Electronics 63K) ──
                # Xử lý theo từng khối chunk_size=2000 items: RAM đỉnh < 5GB, triệt tiêu 100% nguy cơ OOM / SIGKILL!
                i_norm = F.normalize(mfeats_init.cpu(), p=2, dim=-1)
                edge_weight_ui_ones = torch.ones_like(raw_edge_ui[0], dtype=torch.float)
                R_csr = torch.sparse_coo_tensor(
                    raw_edge_ui.cpu(), edge_weight_ui_ones.cpu(),
                    size=(self.User.count, self.Item.count)
                ).to_sparse_csr()

                chunk_size = 2000
                collected_rows = []
                collected_cols = []
                collected_vals = []

                raw_items = raw_edge_ui[1].cpu()
                raw_users = raw_edge_ui[0].cpu()
                item_sort_perm = torch.argsort(raw_items)
                sorted_items = raw_items[item_sort_perm]
                sorted_users = raw_users[item_sort_perm]
                item_ptrs = torch.searchsorted(sorted_items, torch.arange(self.Item.count + 1))

                for start_idx in range(0, self.Item.count, chunk_size):
                    end_idx = min(start_idx + chunk_size, self.Item.count)
                    cur_chunk_len = end_idx - start_idx
                    p_start = item_ptrs[start_idx].item()
                    p_end = item_ptrs[end_idx].item()

                    if p_end > p_start:
                        chunk_item_indices = sorted_items[p_start:p_end] - start_idx
                        chunk_user_indices = sorted_users[p_start:p_end]
                        Rt_chunk = torch.sparse_coo_tensor(
                            torch.stack([chunk_item_indices, chunk_user_indices]),
                            torch.ones(p_end - p_start, dtype=torch.float),
                            size=(cur_chunk_len, self.User.count)
                        ).to_sparse_csr()
                        co_chunk = torch.sparse.mm(Rt_chunk, R_csr).to_dense()
                    else:
                        co_chunk = torch.zeros((cur_chunk_len, self.Item.count), dtype=torch.float)

                    deg_row = i_deg_cpu[start_idx:end_idx].float().unsqueeze(1)
                    deg_col = i_deg_cpu.float().unsqueeze(0)
                    deg_norm_chunk = torch.sqrt(deg_row * deg_col).clamp(min=1.0)
                    ochiai_chunk = co_chunk / deg_norm_chunk

                    sim_m_chunk = torch.clamp(
                        torch.matmul(i_norm[start_idx:end_idx], i_norm.t()),
                        min=0.0
                    )

                    consensus_chunk = ochiai_chunk * sim_m_chunk
                    diag_rows = torch.arange(cur_chunk_len)
                    diag_cols = torch.arange(start_idx, end_idx)
                    consensus_chunk[diag_rows, diag_cols] = 0.0

                    topk_val, topk_idx = torch.topk(consensus_chunk, k=min(3, self.Item.count), dim=-1)
                    valid_mask = topk_val > 0.05

                    if valid_mask.any():
                        row_ids = torch.arange(start_idx, end_idx).unsqueeze(1).expand(-1, topk_val.size(1))[valid_mask]
                        col_ids = topk_idx[valid_mask]
                        val_ids = topk_val[valid_mask]
                        collected_rows.append(row_ids)
                        collected_cols.append(col_ids)
                        collected_vals.append(val_ids)

                if len(collected_rows) > 0:
                    all_rows = torch.cat(collected_rows, dim=0)
                    all_cols = torch.cat(collected_cols, dim=0)
                    all_vals = torch.cat(collected_vals, dim=0)

                    conf_idx = torch.stack([all_rows, all_cols], dim=0)
                    conf_idx, edge_val = freerec.graph.to_normalized(conf_idx, all_vals, normalization='sym')
                    self.s_conf_sparse = torch.sparse_coo_tensor(
                        conf_idx.to(cfg.device), edge_val.to(cfg.device),
                        size=(self.Item.count, self.Item.count),
                        device=cfg.device
                    ).coalesce()
                    print(f"[{cfg.dataset}] ✅ DCD-Gated: Khởi tạo thành công {len(edge_val)} cạnh ảo đồng thuận cao (Chunked Zero-OOM Engine, Device={cfg.device}).")
                else:
                    self.s_conf_sparse = None
                    print(f"[{cfg.dataset}] ℹ️ DCD-Gated: Không tìm thấy cạnh ảo nào vượt ngưỡng đồng thuận > 0.05.")

    def sure_trainpipe(self, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_pairs_source()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        allEmbds = torch.cat((self.User.embeddings.weight, self.Item.embeddings.weight), dim=0)
        layer_embeds = [allEmbds]
        features = allEmbds
        smoothed = allEmbds

        beta = (1.0 - self.beta3).to(allEmbds.device)
        norm_correction = 1.0 - beta ** (self.num_layers + 1)

        alpha_restart = getattr(self.cl_module, 'alpha_restart', 0.0) if self.method == 'appnp_crossmodal' else 0.0

        for _ in range(self.num_layers):
            if alpha_restart > 0.0:
                features = (1.0 - alpha_restart) * (self.Adj @ features * beta) + alpha_restart * allEmbds
            else:
                features = self.Adj @ features * beta
            smoothed = smoothed + features
            layer_embeds.append(features)

        avgEmbds = smoothed.mul(1.0 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(avgEmbds, (self.User.count, self.Item.count))

        # Nếu là DCD-Gated, áp dụng van gating cho item
        if self.method == 'dcd_gated' and self.s_conf_sparse is not None:
            U_0, I_0 = torch.split(layer_embeds[0], [self.User.count, self.Item.count])
            U_1, I_1 = torch.split(layer_embeds[1], [self.User.count, self.Item.count])
            I_1_refined = self.cl_module.forward_gated_items(I_0, I_1, self.s_conf_sparse)
            layer_embeds[1] = torch.cat([U_1, I_1_refined], dim=0)

        return userEmbds, itemEmbds, layer_embeds

    def encode_for_eval(self) -> Tuple[torch.Tensor, torch.Tensor]:
        allEmbds = torch.cat((self.User.embeddings.weight, self.Item.embeddings.weight), dim=0)
        features = allEmbds
        smoothed = allEmbds
        beta = (1.0 - self.beta3).to(allEmbds.device)
        norm_correction = 1.0 - beta ** (self.num_layers + 1)
        alpha_restart = getattr(self.cl_module, 'alpha_restart', 0.0) if self.method == 'appnp_crossmodal' else 0.0

        for _ in range(self.num_layers):
            if alpha_restart > 0.0:
                features = (1.0 - alpha_restart) * (self.Adj @ features * beta) + alpha_restart * allEmbds
            else:
                features = self.Adj @ features * beta
            smoothed = smoothed + features

        avgEmbds = smoothed.mul(1.0 - beta).div(norm_correction)
        return torch.split(avgEmbds, (self.User.count, self.Item.count))

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds, layer_embeds = self.encode()

        users     = data[self.User]
        positives = data[self.Item]
        negatives = data[self.INeg]

        rec_loss = self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[positives]),
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[negatives]),
        )

        if self.training:
            beta = (1.0 - self.beta3).to(userEmbds.device)
            i_mod = self.item_modals_raw if hasattr(self, 'item_modals_raw') else None

            if self.method == 'appnp_crossmodal':
                weighted_cl_loss, raw_cl_loss = self.cl_module(
                    layer_embeds = layer_embeds,
                    users        = users,
                    positives    = positives,
                    beta         = beta,
                    item_modals  = i_mod,
                ) if hasattr(self.cl_module, 'inject_spectral_noise') else (0.0, 0.0)
                cross_loss = self.cl_module.forward_cross_modal(userEmbds, i_mod, users, positives)
                return rec_loss + weighted_cl_loss + cross_loss

            weighted_cl_loss, raw_cl_loss = self.cl_module(
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
            self.Item: itemEmbds.detach().clone()
        }


# ═════════════════════════════════════════════════════════════════════════════
# Coach Class with Automatic Warmup Epoch Stepping
# ═════════════════════════════════════════════════════════════════════════════
class CoachForSTAIR_Breakthrough(freerec.launcher.Coach):
    """
    Coach Nâng Cấp:
    1. Cosine Warmup & Annealing Learning Rate Scheduler:
       - Warmup tuyến tính 15 epoch từ min_lr (1e-6) lên base_lr (1e-3).
       - Cosine decay từ base_lr về min_lr suốt các epoch còn lại.
    2. Điểm mục tiêu NDCG@20:
       - Theo dõi NDCG@20 trên tập Validation (which4best: NDCG@20).
       - Lưu checkpoint tối ưu tốt nhất 'best_model.pth' khi NDCG@20 tăng.
    3. Patience Early Stopping = 30 epochs:
       - Dừng huấn luyện nếu NDCG@20 không cải thiện sau 30 epochs liên tiếp.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_ndcg20 = -1.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.patience = getattr(self.cfg, 'patience', 30)
        self.lr_warmup_epochs = getattr(self.cfg, 'lr_warmup_epochs', 15)
        self.min_lr = getattr(self.cfg, 'min_lr', 1e-6)

    def adjust_learning_rate(self, epoch: int) -> float:
        base_lr = self.cfg.lr
        min_lr = self.min_lr
        warmup = self.lr_warmup_epochs
        total = self.cfg.epochs

        if epoch < warmup:
            current_lr = min_lr + (base_lr - min_lr) * float(epoch + 1) / float(max(1, warmup))
        else:
            progress = float(epoch + 1 - warmup) / float(max(1, total - warmup))
            current_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        return current_lr

    def train_per_epoch(self, epoch: int):
        current_lr = self.adjust_learning_rate(epoch)

        if hasattr(self.model, 'cl_module') and hasattr(self.model.cl_module, 'update_epoch'):
            self.model.cl_module.update_epoch(epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == self.cfg.epochs:
            gamma_h = getattr(self.model.cl_module, 'gamma_h', getattr(self.model.cl_module, 'gamma_base', 0.15))
            curr_lam = getattr(self.model.cl_module, 'current_lambda', 0.0)
            print(f"  [Epoch {epoch + 1:03d}] LR (Cosine): {current_lr:.6e} | Lambda_CL: {curr_lam:.5f} | Gamma_H: {gamma_h:.4f}")

        return super().train_per_epoch(epoch)

    def evaluate(self, epoch: int = 0, mode: str = 'valid'):
        super().evaluate(epoch, mode=mode)
        if mode == 'valid':
            try:
                meters = getattr(self, 'meters', None)
                if meters is None and hasattr(self, 'monitor') and hasattr(self.monitor, 'meters'):
                    meters = self.monitor.meters

                def get_val(name):
                    if meters is not None:
                        for k, v in meters.items():
                            if name.lower() == k.lower() or name.lower() in k.lower():
                                return getattr(v, 'avg', getattr(v, 'val', None))
                    return None

                r10 = get_val('Recall@10')
                r20 = get_val('Recall@20')
                n10 = get_val('NDCG@10')
                n20 = get_val('NDCG@20')

                if n20 is not None:
                    save_dir = getattr(self.cfg, 'CHECKPOINT_PATH', getattr(self.cfg, 'root_dir', '.'))
                    os.makedirs(save_dir, exist_ok=True)
                    best_ckpt_path = os.path.join(save_dir, "best_model.pth")

                    if n20 > self.best_ndcg20:
                        self.best_ndcg20 = n20
                        self.best_epoch = epoch
                        self.patience_counter = 0
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'best_ndcg20': n20,
                            'metrics': {'Recall@10': r10, 'Recall@20': r20, 'NDCG@10': n10, 'NDCG@20': n20}
                        }, best_ckpt_path)
                        r10_str = f"{r10:.4f}" if r10 is not None else "N/A"
                        r20_str = f"{r20:.4f}" if r20 is not None else "N/A"
                        n10_str = f"{n10:.4f}" if n10 is not None else "N/A"
                        print(
                            f"\n  🌟 [NEW BEST MODEL @Epoch {epoch:03d}] >>> NDCG@20: {n20:.4f} "
                            f"(R@10: {r10_str} | R@20: {r20_str} | N@10: {n10_str}) -> {best_ckpt_path}\n"
                        )
                    else:
                        self.patience_counter += 1
                        print(
                            f"  ⏳ [Patience: {self.patience_counter}/{self.patience}] "
                            f"Chưa có cải thiện NDCG@20 kể từ Epoch {self.best_epoch} (Best NDCG@20: {self.best_ndcg20:.4f})\n"
                        )
                        if self.patience_counter >= self.patience:
                            print(
                                f"\n🛑 [EARLY STOPPING TRIGGERED] Kích hoạt dừng sớm sau {self.patience} epochs "
                                f"không cải thiện NDCG@20 (Best Epoch: {self.best_epoch}, Best NDCG@20: {self.best_ndcg20:.4f}).\n"
                            )
                            self.cfg.epochs = epoch + 1
            except Exception as e:
                print(f"  ⚠️ Lưu ý: {e}")


def main():
    processed_dir = os.path.join(cfg.root, "Processed", cfg.dataset)
    if not os.path.exists(processed_dir) or (os.path.isdir(processed_dir) and not os.listdir(processed_dir)):
        for cand in [
            os.path.join(cfg.root, cfg.dataset),
            os.path.join("/kaggle/data", cfg.dataset),
            os.path.join("/kaggle/data/Processed", cfg.dataset),
            os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset),
            os.path.join("data", cfg.dataset),
        ]:
            if os.path.exists(cand) and os.path.isdir(cand) and len(os.listdir(cand)) > 0:
                os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
                try:
                    os.symlink(cand, processed_dir)
                except Exception:
                    import shutil
                    shutil.copytree(cand, processed_dir, dirs_exist_ok=True)
                break

    tasktag = getattr(cfg, 'tasktag', None) or getattr(freerec.data.tags, 'MATCHING', None)
    ds_cls = getattr(freerec.data.datasets, cfg.dataset, None)
    dataset = ds_cls(root=cfg.root) if isinstance(ds_cls, type) else freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=tasktag)
    dataset.TASK = tasktag

    model = STAIR_Breakthrough_Model(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_Breakthrough(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    coach.fit()


if __name__ == '__main__':
    main()
