# -*- coding: utf-8 -*-
"""
main_stair_sbn_bsc_v4_1_ssb.py
===============================
STAIR-BSC-Reweight v4.1-SSB: Safe Spectral Boost for Backward Stepwise Convolution
Synthesized from SIGE (AAAI 2026) and EVEN (AAAI 2025).

Entry point cho huấn luyện mô hình STAIR-BSC-Reweight v4.1-SSB trên 3 benchmark:
  - Amazon2014Baby_550_MMRec
  - Amazon2014Sports_550_MMRec
  - Amazon2014Electronics_550_MMRec

Khắc phục 5 tử huyệt của dự thảo v4.1:
  - Bảo tồn 100% tô-pô đồ thị kNN Baseline (0% cắt tỉa, không đói cấu trúc).
  - Tăng cường trọng số an toàn w_ij = 1.0 + alpha * q_modal + beta * q_behavior in [1.0, 1.8].
  - Đối xứng hóa tường minh W_sym = max(W, W^T) bảo toàn tính SPSD của Laplacian.
  - Zero learnable parameters trong preprocessor, không có rủi ro đóng băng tham số.
  - 100% sparse COO/CSR, 0 MB dense overhead, an toàn tuyệt đối trên Electronics 63K.

Usage:
  python main_stair_sbn_bsc_v4_1_ssb.py --config configs/Amazon2014Baby_550_MMRec.yaml --ssb-mode full_ssb
  python main_stair_sbn_bsc_v4_1_ssb.py --dataset Amazon2014Sports_550_MMRec --ssb-mode modal_only
  python main_stair_sbn_bsc_v4_1_ssb.py --dataset Amazon2014Sports_550_MMRec --ssb-mode behavior_only
"""

import math
import os
import sys
import time
from typing import Dict, Tuple, Optional

# Fix Windows console utf-8 encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# Compatibility fallback for torchdata.datapipes in PyTorch 2.x / Kaggle environments
try:
    import torchdata.datapipes as dp
except (ImportError, ModuleNotFoundError):
    import types
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

# Idempotent DataPipe registration patch for PyTorch 2.x / FreeRec reload safety
try:
    from torch.utils.data.datapipes.datapipe import IterDataPipe as _NativeIterDP, MapDataPipe as _NativeMapDP
    for _cls in [_NativeIterDP, _NativeMapDP]:
        if hasattr(_cls, 'register_datapipe_as_function'):
            _orig_reg = _cls.register_datapipe_as_function
            def _make_safe_reg(orig_fn):
                def _safe_reg(cls, function_name, cls_to_register, *args, **kwargs):
                    if hasattr(cls, 'functions') and function_name in cls.functions:
                        try:
                            del cls.functions[function_name]
                        except Exception:
                            pass
                    return orig_fn.__func__(cls, function_name, cls_to_register, *args, **kwargs)
                return _safe_reg
            _cls.register_datapipe_as_function = classmethod(_make_safe_reg(_orig_reg))
except Exception:
    pass

import freerec
from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

from models.stair_sbn_bsc_v4_1_ssb import STAIR_BSC_Reweight_Engine

freerec.declare(version='0.8.5')

# ============================================================================
# Configuration
# ============================================================================

cfg = freerec.parser.Parser()

# --- STAIR Baseline params ---
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3,
                 help="Số tầng convolution cho FSC và BSC")
cfg.add_argument("--mfiles", type=str,
                 default="textual_modality.pkl,visual_modality.pkl",
                 help="Các file lưu trữ đặc trưng đa phương thức")
cfg.add_argument("--num-neighbors", type=str, default='5-1',
                 help="Số láng giềng kNN cho từng modality: k_text-k_vis")
cfg.add_argument("--gamma", type=float, default=0.2)

# --- STAIR-BSC-Reweight v4.1-SSB params ---
cfg.add_argument("--ssb-mode", "--mode", dest="ssb_mode", type=str, default="full_ssb",
                 choices=["baseline", "modal_only", "behavior_only", "full_ssb"],
                 help="Chế độ thực nghiệm v4.1-SSB ['baseline', 'modal_only', 'behavior_only', 'full_ssb']")
cfg.add_argument("--ssb-alpha", "--alpha", dest="ssb_alpha", type=float, default=0.50,
                 help="Trọng số tăng cường đa phương thức alpha [0.0, 1.0] (mặc định: 0.50)")
cfg.add_argument("--ssb-beta", "--beta", dest="ssb_beta", type=float, default=0.30,
                 help="Trọng số tăng cường đồng mua hành vi beta [0.0, 1.0] (mặc định: 0.30)")
cfg.add_argument("--ssb-tau-text", "--tau-text", "--tau_text", dest="ssb_tau_text", type=float, default=0.10,
                 help="Ngưỡng lọc tương đồng văn bản [0.05, 0.25] (mặc định: 0.10)")
cfg.add_argument("--ssb-tau-visual", "--tau-visual", "--tau_visual", dest="ssb_tau_visual", type=float, default=0.10,
                 help="Ngưỡng lọc tương đồng hình ảnh [0.05, 0.25] (mặc định: 0.10)")
cfg.add_argument("--ssb-min-weight", dest="ssb_min_weight", type=float, default=1.00,
                 help="Cận dưới trọng số cạnh (mặc định: 1.00)")
cfg.add_argument("--ssb-max-weight", dest="ssb_max_weight", type=float, default=1.80,
                 help="Cận trên trọng số cạnh (mặc định: 1.80)")

cfg.set_defaults(
    description="STAIR-BSC-Reweight-v4.1-SSB",
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

# === YAML Configuration Loader & Normalizer ===
if getattr(cfg, 'config', None) is not None:
    config_path = cfg.config
    if not os.path.isabs(config_path):
        if not os.path.exists(config_path):
            repo_root = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(repo_root, config_path)
            if os.path.exists(alt_path):
                config_path = alt_path

    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_cfg = yaml.safe_load(f) or {}

        # Thu thập các tham số được truyền trực tiếp từ CLI để ưu tiên cao nhất (CLI > YAML > Defaults)
        cli_specified = set()
        for a in sys.argv[1:]:
            if a.startswith('--'):
                clean_a = a.lstrip('-').split('=')[0].replace('-', '_').lower()
                cli_specified.add(clean_a)
                if clean_a.startswith('ssb_'):
                    cli_specified.add(clean_a[4:])
                else:
                    cli_specified.add(f"ssb_{clean_a}")

        for raw_key, val in yaml_cfg.items():
            norm_key = raw_key.replace('-', '_').lower()

            if norm_key in cli_specified or f"ssb_{norm_key}" in cli_specified:
                print(f"[Config] Giữ nguyên giá trị CLI cho '{norm_key}' (không bị YAML ghi đè)")
                continue

            matched = False
            if hasattr(cfg, norm_key):
                setattr(cfg, norm_key, val)
                matched = True
            elif hasattr(cfg, f"ssb_{norm_key}"):
                setattr(cfg, f"ssb_{norm_key}", val)
                matched = True
            elif norm_key.startswith("ssb_") and hasattr(cfg, norm_key[4:]):
                setattr(cfg, norm_key[4:], val)
                matched = True
            elif hasattr(cfg, raw_key):
                setattr(cfg, raw_key, val)
                matched = True

            if matched:
                print(f"[Config] Nạp {norm_key} = {val} từ YAML")
            else:
                setattr(cfg, norm_key, val)
                print(f"[Config] Nạp {norm_key} = {val} từ YAML")
    else:
        print(f"[Config Cảnh báo] Không tìm thấy file config: {config_path}")

# Đảm bảo cfg.root luôn là đường dẫn tuyệt đối chuẩn xác
if hasattr(cfg, 'root') and cfg.root:
    cfg.root = os.path.abspath(cfg.root)

cfg.mfiles = cfg.mfiles.split(',')
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-')))

# beta3: Hệ số spectral filter cho Smoother trong BSC
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ============================================================================
# Model: STAIR-BSC-Reweight v4.1-SSB
# ============================================================================

class STAIR_SBN_BSC_v4_1_SSB(freerec.models.GenRecArch):
    """
    STAIR-BSC-Reweight v4.1-SSB (Safe Spectral Boost for Backward Stepwise Convolution).

    Kế thừa toàn bộ kiến trúc STAIR Baseline:
      - Forward: FSC trên đồ thị bipartite User-Item (giữ nguyên gốc)
      - Loss: BPR Ranking Loss thuần túy (không auxiliary loss, không xung đột gradient)
      - Backward: AdamWSEvo + Smoother(mAdj) với ma trận mAdj được tăng cường trọng số
        an toàn theo nguyên tắc Safe Spectral Boost (v4.1-SSB).
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
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
        """
        Đăng ký Smoother với ma trận mAdj đã được tăng cường trọng số an toàn.
        """
        params = [
            {
                'params': self.User.parameters(),
                'smoother': None
            },
            {
                'params': self.Item.parameters(),
                'smoother': Smoother(
                    self.mAdj, beta=cfg.beta3,
                    L=cfg.num_layers, aggr='neumann'
                )
            },
        ]
        return params

    def whitening(self, feats: torch.Tensor):
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(
            self.Item.count / cfg.embedding_dim
        )

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(
            sim, k, symmetric=False
        )
        return edge_index

    def prepare(self, path: str):
        """
        Khâu tiền xử lý (offline, 1 lần duy nhất trước Epoch 1):
          1. Nạp đặc trưng đa phương thức (text + vision).
          2. Xây dựng đồ thị kNN Baseline gốc (Bảo tồn 100% tô-pô).
          3. Tăng cường trọng số an toàn w_ij qua STAIR_BSC_Reweight_Engine (v4.1-SSB).
          4. Đăng ký ma trận mAdj vào buffer.
          5. SVD Whitening khởi tạo Item Embeddings & User Embeddings.
        """
        from freerec.utils import import_pickle

        # 1. Nạp modality features
        mfeats = [
            import_pickle(os.path.join(path, mfile))
            for mfile in cfg.mfiles
        ]

        # 2. Xây dựng đồ thị kNN Baseline gốc
        print(f"[v4.1-SSB] Đang xây dựng đồ thị kNN gốc: k_neighbors={cfg.num_neighbors}...")
        raw_edge_index = torch.cat(
            [self.get_knn_graph(feats, k)
             for feats, k in zip(mfeats, cfg.num_neighbors)],
            dim=1
        )
        num_items = self.Item.count
        num_users = self.User.count
        print(f"[v4.1-SSB] Baseline kNN hoàn tất: {raw_edge_index.size(1):,} cạnh thô.")

        # 3. Trích xuất ma trận tương tác người dùng - sản phẩm train_R dạng scipy CSR
        edge_index_ui = self.dataset.train().to_bigraph(
            edge_type='u2i'
        )['u2i'].edge_index

        row_ui = edge_index_ui[0].cpu().numpy()
        col_ui = edge_index_ui[1].cpu().numpy()
        data_ui = np.ones(len(row_ui), dtype=np.float32)
        train_R = sp.csr_matrix(
            (data_ui, (row_ui, col_ui)),
            shape=(num_users, num_items)
        )

        # 4. STAIR-BSC-Reweight Engine v4.1-SSB: Tính ma trận mAdj
        engine = STAIR_BSC_Reweight_Engine(
            mode=getattr(cfg, 'ssb_mode', 'full_ssb'),
            alpha=getattr(cfg, 'ssb_alpha', 0.50),
            beta=getattr(cfg, 'ssb_beta', 0.30),
            tau_t=getattr(cfg, 'ssb_tau_text', 0.10),
            tau_v=getattr(cfg, 'ssb_tau_visual', 0.10),
            min_weight=getattr(cfg, 'ssb_min_weight', 1.00),
            max_weight=getattr(cfg, 'ssb_max_weight', 1.80),
            verbose=True,
        )

        mAdj_boosted = engine.build_boosted_mAdj(
            text_feats=mfeats[0],
            vis_feats=mfeats[1],
            train_user_item_matrix=train_R,
            raw_knn_edge_index=raw_edge_index,
            num_items=num_items,
            target_device=cfg.device,
        )

        # Đăng ký mAdj buffer cho Smoother
        self.register_buffer('mAdj', mAdj_boosted)

        assert self.mAdj.is_sparse_csr, "Lỗi: mAdj phải ở định dạng torch.sparse_csr_tensor!"
        assert self.mAdj._nnz() > 0, "Lỗi: mAdj không được rỗng!"
        print(f"[v4.1-SSB] Đã đăng ký mAdj thành công: nnz={self.mAdj._nnz():,} trên device={self.mAdj.device}")

        # 5. SVD Whitening → Modality Initialization
        mfeats_whitened = [
            self.whitening(mfeat) * k
            for mfeat, k in zip(mfeats, cfg.num_neighbors)
        ]
        mfeats_combined = sum(mfeats_whitened).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats_combined)

        # 6. Khởi tạo User Embeddings via R @ mfeats
        edge_index_norm, edge_weight_norm = freerec.graph.to_normalized(
            edge_index_ui, normalization='left'
        )
        R_torch = torch.sparse_coo_tensor(
            edge_index_norm, edge_weight_norm,
            size=(num_users, num_items)
        ).to_sparse_csr()
        self.User.embeddings.weight.data.copy_(R_torch @ mfeats_combined)

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_pairs_source(
        ).gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        )
        features = allEmbds
        smoothed = allEmbds

        # FSC (Forward Stepwise Convolution)
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
        users, positives, negatives = (
            data[self.User], data[self.Item], data[self.INeg]
        )
        userEmbds = userEmbds[users]  # (B, 1, D)
        iposEmbds = itemEmbds[positives]  # (B, 1, D)
        inegEmbds = itemEmbds[negatives]  # (B, K, D)

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

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]  # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum("BKD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]  # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]  # (B, 101, D)
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)


# ============================================================================
# Coach (Training Loop)
# ============================================================================

class CoachForSTAIR_BSC_v4_1_SSB(freerec.launcher.Coach):

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


def main():
    print("=" * 80)
    print("🚀 KHỞI CHẠY HUẤN LUYỆN STAIR-BSC-REWEIGHT v4.1-SSB")
    print(f"   * Tập dữ liệu   : {cfg.dataset}")
    print(f"   * Chế độ SSB    : {cfg.ssb_mode}")
    print(f"   * Siêu tham số  : alpha={cfg.ssb_alpha}, beta={cfg.ssb_beta}, tau_t={cfg.ssb_tau_text}, tau_v={cfg.ssb_tau_visual}")
    print(f"   * Trọng số chặn : [{cfg.ssb_min_weight}, {cfg.ssb_max_weight}]")
    print("=" * 80)

    # Robust auto-bridge for FreeRec:
    # FreeRec expects dataset in os.path.join(cfg.root, "Processed", cfg.dataset).
    # If it is located in cfg.root/{cfg.dataset} or any candidate location,
    # symlink or copy it so RecDataSet finds it immediately without FileNotFoundError.
    processed_dir = os.path.join(cfg.root, "Processed", cfg.dataset)
    if not os.path.exists(processed_dir) or not os.listdir(processed_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(cfg.root, cfg.dataset),
            os.path.join("/kaggle/data", cfg.dataset),
            os.path.join("/kaggle/data/Processed", cfg.dataset),
            os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset),
            os.path.join("/kaggle/working/STAIR-Enhanced/data/Processed", cfg.dataset),
            os.path.join(script_dir, "data", cfg.dataset),
            os.path.join(script_dir, "data", "Processed", cfg.dataset),
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
                cfg.root, cfg.dataset, tasktag=getattr(cfg, 'tasktag', None)
            )
    else:
        dataset = freerec.data.datasets.RecDataSet(
            cfg.root, cfg.dataset, tasktag=getattr(cfg, 'tasktag', None)
        )

    model = STAIR_SBN_BSC_v4_1_SSB(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_BSC_v4_1_SSB(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()

