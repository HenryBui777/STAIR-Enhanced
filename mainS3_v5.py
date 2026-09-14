# -*- coding: utf-8 -*-
"""
mainS3_v5.py — STAIR-BSC-Reweight (STAIR-v5) Official Training Pipeline
======================================================================
Safe Topological Reweighting with Multiplicative Consensus Boosting and SPSD-Guaranteed BSC Smoother.

Khắc phục triệt để 5 tử huyệt kỹ thuật & bảo toàn 100% nguyên lý STAIR (AAAI 2025):
  1. 100% SPSD Guaranteed: Đối xứng hóa tường minh W_sym = max(W, W^T) và chuẩn hóa Symmetric Laplacian.
  2. 0% Edge Pruning: Bảo tồn 100% tô-pô đồ thị kNN Baseline, không gây đói cấu trúc cho các item đuôi dài.
  3. Multiplicative Consensus Boost: W_ij = W_base * (1 + alpha*q_m + beta*q_b) bảo toàn tính đơn điệu.
  4. BPR Loss Mịn: Bảo toàn gradient dày đặc (dense 100%) cho bộ làm mịn BSC trong AdamWSEvo.
  5. SVD Whitening Chuẩn tắc: Triệt tiêu hoàn toàn singular values S, đưa Covariance về 1/d * I_d.

Usage:
  # Chạy thực nghiệm với file config YAML của dataset:
  python mainS3_v5.py --config configs/Amazon2014Sports_550_MMRec.yaml --ssb-mode full_ssb
  python mainS3_v5.py --config configs/Amazon2014Baby_550_MMRec.yaml --ssb-mode modal_only

  # Chạy với file config siêu tham số v5 và chỉ định dataset:
  python mainS3_v5.py --config configs/stair_v5_hyperparams.yaml --dataset Amazon2014Sports_550_MMRec

  # Chạy với tham số dòng lệnh tùy biến:
  python mainS3_v5.py --dataset Amazon2014Sports_550_MMRec --ssb-mode full_ssb --alpha 0.40 --beta 0.20
"""

import os
import sys
import time
import math
from typing import Dict, Tuple, Optional

# Fix Windows console utf-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Tối ưu hóa bộ nhớ CUDA chống phân mảnh trên các đồ thị quy mô lớn (Electronics 63K items)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# Compatibility fallback for torchdata.datapipes in PyTorch 2.x / Kaggle
# =========================================================================
try:
    import torchdata.datapipes as dp
except (ImportError, ModuleNotFoundError):
    import types
    import torch.utils.data

    if "torchdata" not in sys.modules:
        td = types.ModuleType("torchdata")
        sys.modules["torchdata"] = td
    else:
        td = sys.modules["torchdata"]

    if "torchdata.datapipes" not in sys.modules:
        dp_mod = types.ModuleType("torchdata.datapipes")
        iter_mod = types.ModuleType("torchdata.datapipes.iter")
        map_mod = types.ModuleType("torchdata.datapipes.map")

        class IterDataPipe(torch.utils.data.IterableDataset):
            pass

        class MapDataPipe(torch.utils.data.Dataset):
            pass

        iter_mod.IterDataPipe = IterDataPipe
        map_mod.MapDataPipe = MapDataPipe

        dp_mod.iter = iter_mod
        dp_mod.map = map_mod
        td.datapipes = dp_mod

        sys.modules["torchdata.datapipes"] = dp_mod
        sys.modules["torchdata.datapipes.iter"] = iter_mod
        sys.modules["torchdata.datapipes.map"] = map_mod

# Idempotent DataPipe registration patch for FreeRec / Kaggle notebook reloads
try:
    from torch.utils.data.datapipes.datapipe import IterDataPipe as _NativeIterDP, MapDataPipe as _NativeMapDP
    for _cls in [_NativeIterDP, _NativeMapDP]:
        if hasattr(_cls, "register_datapipe_as_function"):
            _orig_reg = _cls.register_datapipe_as_function
            def _make_safe_reg(orig_fn):
                def _safe_reg(cls, function_name, cls_to_register, *args, **kwargs):
                    if hasattr(cls, "functions") and function_name in cls.functions:
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
from freerec.data.tags import USER, ITEM, ID, POSITIVE, NEGATIVE

# Chuẩn hóa import case-sensitive tương thích tuyệt đối môi trường Linux (Kaggle)
from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

from models.stair_sre_v5 import STAIR_BSC_Reweight_Engine

freerec.declare(version="0.8.5")

# =========================================================================
# Config Parser: STAIR Baseline Args + STAIR-v5 Reweight Args
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
cfg.add_argument("--num-neighbors", type=str, default="5-1",
                 help="kNN neighbor counts per modality, e.g. '5-1'")
cfg.add_argument("--gamma", type=float, default=0.2,
                 help="Spectral decay power gamma for beta3 (overridden by dataset YAML)")

# -- STAIR-v5 (STAIR-BSC-Reweight) Specific Parameters with Dual-Alias Support --
cfg.add_argument("--ssb-mode", "--mode", dest="ssb_mode", type=str, default="full_ssb",
                 choices=["full_ssb", "modal_only", "behavior_only", "baseline"],
                 help="Chế độ reweighting tô-pô ['baseline', 'modal_only', 'behavior_only', 'full_ssb']")
cfg.add_argument("--ssb-alpha", "--alpha", dest="ssb_alpha", type=float, default=0.40,
                 help="Trọng số tăng cường đồng thuận đa phương thức alpha (default: 0.40)")
cfg.add_argument("--ssb-beta", "--beta", dest="ssb_beta", type=float, default=0.20,
                 help="Trọng số tăng cường đồng mua Ochiai beta (default: 0.20)")
cfg.add_argument("--ssb-tau-text", "--tau-text", "--tau_text", dest="ssb_tau_text", type=float, default=0.10,
                 help="Ngưỡng lọc tương đồng văn bản tau_t (default: 0.10)")
cfg.add_argument("--ssb-tau-visual", "--tau-visual", "--tau_visual", dest="ssb_tau_visual", type=float, default=0.10,
                 help="Ngưỡng lọc tương đồng thị giác tau_v (default: 0.10)")
cfg.add_argument("--ssb-min-weight", "--min-weight", dest="ssb_min_weight", type=float, default=1.00,
                 help="Chặn dưới trọng số cạnh (default: 1.00)")
cfg.add_argument("--ssb-max-weight", "--max-weight", dest="ssb_max_weight", type=float, default=3.60,
                 help="Chặn trên trọng số cạnh multiplicative (default: 3.60)")

# Thiết lập giá trị mặc định chuẩn tắc kế thừa STAIR AAAI 2025
cfg.set_defaults(
    description="STAIR-BSC-Reweight (STAIR-v5)",
    root="../../data",
    dataset="Amazon2014Baby_550_MMRec",
    epochs=500,
    batch_size=1024,
    optimizer="adamwsevo",
    lr=1e-3,
    weight_decay=0.1,
    seed=1,
    monitors=["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
    which4best="NDCG@20",
)

# Parse CLI arguments ban đầu
cfg.compile()

# =========================================================================
# YAML Configuration Loader & Normalizer (CLI > YAML > Defaults)
# =========================================================================
if getattr(cfg, "config", None) is not None:
    config_path = cfg.config
    if not os.path.isabs(config_path):
        if not os.path.exists(config_path):
            repo_root = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(repo_root, config_path)
            if os.path.exists(alt_path):
                config_path = alt_path

    if os.path.exists(config_path):
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}

        # Ghi nhận các tham số được chỉ định tường minh trên CLI để ưu tiên tuyệt đối
        cli_specified = set()
        for a in sys.argv[1:]:
            if a.startswith("--"):
                clean_a = a.lstrip("-").split("=")[0].replace("-", "_").lower()
                cli_specified.add(clean_a)
                if clean_a.startswith("ssb_"):
                    cli_specified.add(clean_a[4:])
                else:
                    cli_specified.add(f"ssb_{clean_a}")

        for raw_key, val in yaml_cfg.items():
            norm_key = raw_key.replace("-", "_").lower()

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

# Đồng bộ các alias sang thuộc tính chuẩn
if hasattr(cfg, "ssb_alpha") and not hasattr(cfg, "alpha"):
    cfg.alpha = cfg.ssb_alpha
if hasattr(cfg, "alpha") and not hasattr(cfg, "ssb_alpha"):
    cfg.ssb_alpha = cfg.alpha

if hasattr(cfg, "ssb_beta") and not hasattr(cfg, "beta"):
    cfg.beta = cfg.ssb_beta
if hasattr(cfg, "beta") and not hasattr(cfg, "ssb_beta"):
    cfg.ssb_beta = cfg.beta

if hasattr(cfg, "ssb_mode") and not hasattr(cfg, "mode"):
    cfg.mode = cfg.ssb_mode
if hasattr(cfg, "mode") and not hasattr(cfg, "ssb_mode"):
    cfg.ssb_mode = cfg.mode

if hasattr(cfg, "ssb_tau_text") and not hasattr(cfg, "tau_text"):
    cfg.tau_text = cfg.ssb_tau_text
if hasattr(cfg, "tau_text") and not hasattr(cfg, "ssb_tau_text"):
    cfg.ssb_tau_text = cfg.tau_text

if hasattr(cfg, "ssb_tau_visual") and not hasattr(cfg, "tau_visual"):
    cfg.tau_visual = cfg.ssb_tau_visual
if hasattr(cfg, "tau_visual") and not hasattr(cfg, "ssb_tau_visual"):
    cfg.ssb_tau_visual = cfg.tau_visual

if hasattr(cfg, "ssb_min_weight") and not hasattr(cfg, "min_weight"):
    cfg.min_weight = cfg.ssb_min_weight
if hasattr(cfg, "min_weight") and not hasattr(cfg, "ssb_min_weight"):
    cfg.ssb_min_weight = cfg.min_weight

if hasattr(cfg, "ssb_max_weight") and not hasattr(cfg, "max_weight"):
    cfg.max_weight = cfg.ssb_max_weight
if hasattr(cfg, "max_weight") and not hasattr(cfg, "ssb_max_weight"):
    cfg.ssb_max_weight = cfg.max_weight

# Đảm bảo cfg.root là đường dẫn tuyệt đối chuẩn xác
if hasattr(cfg, "root") and cfg.root:
    cfg.root = os.path.abspath(cfg.root)

# Chuẩn hóa danh sách mfiles và num_neighbors
if isinstance(cfg.mfiles, str):
    cfg.mfiles = cfg.mfiles.split(",")
if isinstance(cfg.num_neighbors, str):
    cfg.num_neighbors = [int(k) for k in cfg.num_neighbors.split("-")]

if len(cfg.mfiles) != len(cfg.num_neighbors):
    raise ValueError(
        f"Length of mfiles ({len(cfg.mfiles)}) must match num_neighbors ({len(cfg.num_neighbors)})"
    )

# Tính toán vector phổ beta3 của BSC sau khi đã hoàn tất nạp gamma từ YAML
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / float(cfg.embedding_dim)).pow(cfg.gamma)
).to(cfg.device)


# =========================================================================
# STAIR-v5 Architecture Class (Kế thừa freerec.models.GenRecArch)
# =========================================================================
class STAIR_v5_Arch(freerec.models.GenRecArch):
    """
    STAIR-v5 Model Architecture (STAIR-BSC-Reweight):
      - Kế thừa GenRecArch (như AAAI 2025 STAIR gốc), bảo đảm tính tương thích với
        FreeRec trainpipe/validpipe/testpipe trên Kaggle.
      - 100% SPSD-guaranteed single-matrix Laplacian smoother via STAIR_BSC_Reweight_Engine.
      - SVD Whitening chuẩn tắc đại số tuyến tính (triệt tiêu hoàn toàn singular values S).
      - Modal fusion tỷ lệ 5:1 bảo toàn domain prior của STAIR.
      - BPR Loss mượt mà liên tục, bảo toàn gradient dày đặc (100% dense) cho Smoother.
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)
        self.num_layers = cfg.num_layers

        self.User.add_module(
            "embeddings", nn.Embedding(self.User.count, cfg.embedding_dim)
        )
        self.Item.add_module(
            "embeddings", nn.Embedding(self.Item.count, cfg.embedding_dim)
        )

        self.register_buffer(
            "Adj",
            self.dataset.train().to_normalized_adj(normalization="sym")
        )

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction="mean")

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.0e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def marked_params(self):
        """
        Đăng ký Smoother với ma trận mAdj đã được tăng cường trọng số an toàn SPSD.
        """
        params = [
            {
                "params": self.User.parameters(),
                "smoother": None,
            },
            {
                "params": self.Item.parameters(),
                "smoother": Smoother(
                    self.mAdj, beta=cfg.beta3,
                    L=cfg.num_layers, aggr="neumann"
                ),
            },
        ]
        return params

    def whitening(self, feats: torch.Tensor) -> torch.Tensor:
        """
        SVD Whitening chuẩn tắc đại số tuyến tính (Khắc phục triệt để Lỗi 5):
          feats = feats - mean
          U, S, Vh = svd(feats)
          output = U[:, :d] * sqrt(N / d)
        Đảm bảo Covariance = 1/d * I_d, loại bỏ hoàn toàn singular values S.
        """
        feats = feats - feats.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(feats, full_matrices=False)
        return U[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.0)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        return edge_index

    def prepare(self, path: str):
        """
        Khâu tiền xử lý (offline, 1 lần duy nhất trước Epoch 1):
          1. Nạp đặc trưng đa phương thức (text + vision).
          2. Xây dựng đồ thị kNN Baseline gốc (Bảo tồn 100% tô-pô).
          3. Tăng cường trọng số an toàn w_ij qua STAIR_BSC_Reweight_Engine (v5).
          4. Đăng ký ma trận mAdj dạng sparse CSR vào buffer.
          5. SVD Whitening độc lập từng phương thức, dung hợp 5:1 khởi tạo Item & User embeddings.
        """
        from freerec.utils import import_pickle

        # 1. Nạp modality features
        mfeats = [
            import_pickle(os.path.join(path, mfile))
            for mfile in cfg.mfiles
        ]

        # 2. Xây dựng đồ thị kNN Baseline gốc
        print(f"[STAIR-v5] Đang xây dựng đồ thị kNN gốc: k_neighbors={cfg.num_neighbors}...")
        raw_edge_index = torch.cat(
            [self.get_knn_graph(feats, k)
             for feats, k in zip(mfeats, cfg.num_neighbors)],
            dim=1
        )
        edge_weight = torch.ones_like(raw_edge_index[0], dtype=torch.float)
        raw_edge_index, raw_edge_weight = freerec.graph.coalesce(
            raw_edge_index, edge_weight, reduce="sum"
        )
        num_items = self.Item.count
        num_users = self.User.count
        print(f"[STAIR-v5] Baseline kNN hoàn tất: {raw_edge_index.size(1):,} cạnh gốc (Trọng số: {raw_edge_weight.min().item():.1f} - {raw_edge_weight.max().item():.1f}).")

        # 3. Trích xuất ma trận tương tác người dùng - sản phẩm train_R dạng scipy CSR
        edge_index_ui = self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index
        row_ui = edge_index_ui[0].cpu().numpy()
        col_ui = edge_index_ui[1].cpu().numpy()
        data_ui = np.ones(len(row_ui), dtype=np.float32)
        train_R = sp.csr_matrix(
            (data_ui, (row_ui, col_ui)),
            shape=(num_users, num_items)
        )

        # 4. STAIR-BSC-Reweight Engine: Xây dựng mAdj chuẩn SPSD
        ssb_mode = getattr(cfg, "ssb_mode", getattr(cfg, "mode", "full_ssb"))
        alpha_val = getattr(cfg, "ssb_alpha", getattr(cfg, "alpha", 0.40))
        beta_val = getattr(cfg, "ssb_beta", getattr(cfg, "beta", 0.20))
        tau_t_val = getattr(cfg, "ssb_tau_text", getattr(cfg, "tau_text", 0.10))
        tau_v_val = getattr(cfg, "ssb_tau_visual", getattr(cfg, "tau_visual", 0.10))
        min_w_val = getattr(cfg, "ssb_min_weight", getattr(cfg, "min_weight", 1.00))
        max_w_val = getattr(cfg, "ssb_max_weight", getattr(cfg, "max_weight", 3.60))

        engine = STAIR_BSC_Reweight_Engine(
            mode=ssb_mode,
            alpha=alpha_val,
            beta=beta_val,
            tau_t=tau_t_val,
            tau_v=tau_v_val,
            min_weight=min_w_val,
            max_weight=max_w_val,
            verbose=True,
        )

        mAdj_boosted = engine.build_boosted_mAdj(
            text_feats=mfeats[0],
            vis_feats=mfeats[1],
            train_user_item_matrix=train_R,
            raw_knn_adj=raw_edge_index,
            raw_edge_weight=raw_edge_weight,
            num_items=num_items,
            target_device=cfg.device,
        )

        # Đăng ký mAdj buffer cho Smoother
        self.register_buffer("mAdj", mAdj_boosted)
        assert self.mAdj.is_sparse_csr, "Lỗi: mAdj phải ở định dạng torch.sparse_csr_tensor!"
        assert self.mAdj._nnz() > 0, "Lỗi: mAdj không được rỗng!"
        print(f"[STAIR-v5] Đã đăng ký mAdj thành công: nnz={self.mAdj._nnz():,} trên device={self.mAdj.device}")

        del raw_edge_index, raw_edge_weight, edge_weight, mAdj_boosted
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5. SVD Whitening chuẩn xác (Khắc phục triệt để Lỗi 5)
        mfeats_whitened = [
            self.whitening(mfeat) * k
            for mfeat, k in zip(mfeats, cfg.num_neighbors)
        ]
        mfeats_combined = sum(mfeats_whitened).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats_combined)

        # 6. Khởi tạo User Embeddings qua R @ mfeats theo đúng STAIR gốc
        edge_index_norm, edge_weight_norm = freerec.graph.to_normalized(
            edge_index_ui, normalization="left"
        )
        R_torch = torch.sparse_coo_tensor(
            edge_index_norm, edge_weight_norm,
            size=(num_users, num_items)
        ).to_sparse_csr()
        self.User.embeddings.weight.data.copy_(R_torch @ mfeats_combined)
        print("[STAIR-v5] Khởi tạo Item & User embeddings từ SVD Whitening hoàn tất 100%.")

        del mfeats, mfeats_whitened, mfeats_combined, R_torch, edge_index_norm, edge_weight_norm, train_R
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_pairs_source().gen_train_sampling_neg_(
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
        userEmbds = userEmbds[users]
        iposEmbds = itemEmbds[positives]
        inegEmbds = itemEmbds[negatives]

        rec_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userEmbds, iposEmbds),
            torch.einsum("BKD,BKD->BK", userEmbds, inegEmbds)
        )
        return rec_loss

    def reset_ranking_buffers(self):
        """Executed before evaluation."""
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum("BKD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)


# =========================================================================
# Coach (Training Loop & Best Checkpoint Tracking)
# =========================================================================
class CoachForSTAIR_v5(freerec.launcher.Coach):

    def set_optimizer(self):
        opt_name = getattr(self.cfg, "optimizer", "adamwsevo").lower()
        if opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.marked_params(), lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay
            )
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif opt_name == "adamsevo":
            self.optimizer = AdamSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif opt_name == "adamwsevo":
            self.optimizer = AdamWSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer không hỗ trợ: {opt_name}")

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
                mode="train", pool=["LOSS"]
            )


# =========================================================================
# Main Execution Entrypoint
# =========================================================================
def main():
    ssb_mode = getattr(cfg, "ssb_mode", getattr(cfg, "mode", "full_ssb"))
    alpha_val = getattr(cfg, "ssb_alpha", getattr(cfg, "alpha", 0.40))
    beta_val = getattr(cfg, "ssb_beta", getattr(cfg, "beta", 0.20))
    tau_t_val = getattr(cfg, "ssb_tau_text", getattr(cfg, "tau_text", 0.10))
    tau_v_val = getattr(cfg, "ssb_tau_visual", getattr(cfg, "tau_visual", 0.10))
    min_w_val = getattr(cfg, "ssb_min_weight", getattr(cfg, "min_weight", 1.00))
    max_w_val = getattr(cfg, "ssb_max_weight", getattr(cfg, "max_weight", 3.60))

    print("=" * 85)
    print("🚀 KHỞI CHẠY HUẤN LUYỆN STAIR-BSC-REWEIGHT (STAIR-v5)")
    print(f"   * Tập dữ liệu      : {cfg.dataset}")
    print(f"   * Chế độ Reweight  : {ssb_mode}")
    print(f"   * Hệ số tăng cường : alpha={alpha_val}, beta={beta_val}")
    print(f"   * Ngưỡng tương đồng: tau_text={tau_t_val}, tau_visual={tau_v_val}")
    print(f"   * Trọng số chặn    : [{min_w_val}, {max_w_val}]")
    print(f"   * Cấu hình STAIR   : Layers={cfg.num_layers}, Dim={cfg.embedding_dim}, Gamma={cfg.gamma}")
    print(f"   * Tối ưu hóa       : LR={cfg.lr}, WeightDecay={cfg.weight_decay}, BatchSize={cfg.batch_size}")
    print("=" * 85)

    # Auto-bridge dataset cho Kaggle:
    processed_dir = os.path.join(cfg.root, "Processed", cfg.dataset)
    if os.path.islink(processed_dir) and not os.path.exists(processed_dir):
        try:
            os.unlink(processed_dir)
        except Exception:
            pass

    if not os.path.exists(processed_dir) or (os.path.isdir(processed_dir) and not os.listdir(processed_dir)):
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
                cfg.root, cfg.dataset, tasktag=getattr(cfg, "tasktag", None)
            )
    else:
        dataset = freerec.data.datasets.RecDataSet(
            cfg.root, cfg.dataset, tasktag=getattr(cfg, "tasktag", None)
        )

    model = STAIR_v5_Arch(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_v5(
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
