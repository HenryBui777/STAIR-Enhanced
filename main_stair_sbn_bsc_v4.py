# -*- coding: utf-8 -*-
"""
main_stair_sbn_bsc_v4.py
=========================
STAIR-SBN-BSC v4: Precomputed Structural Denoising for BSC Smoother.

Entry point cho huấn luyện mô hình STAIR-SBN-BSC v4 trên 3 benchmark:
  - Amazon2014Baby_550_MMRec
  - Amazon2014Sports_550_MMRec
  - Amazon2014Electronics_550_MMRec

Kiến trúc: STAIR Baseline + SBN-BSC Preprocessor (SIGE + EVEN)
  - Forward Pass: FSC trên đồ thị User-Item R (giữ nguyên gốc)
  - Loss: BPR Ranking thuần túy (KHÔNG auxiliary loss)
  - Backward Pass: AdamWSEvo + Smoother(mAdj_clean) — gradient được làm mịn
    theo đồ thị ĐỒNG MUA ĐÃ LỌC SẠCH

Usage:
    python main_stair_sbn_bsc_v4.py --config configs/Amazon2014Baby_550_MMRec.yaml
    python main_stair_sbn_bsc_v4.py --dataset Amazon2014Sports_550_MMRec
"""

from typing import Dict, Tuple, Optional

import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

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

from models.stair_sbn_bsc_v4 import SBN_BSC_Preprocessor
from models.stair_sbn_bsc_v4_utils import print_graph_stats

freerec.declare(version='0.8.5')

# ============================================================================
# Configuration
# ============================================================================

cfg = freerec.parser.Parser()

# --- STAIR Baseline params ---
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3,
                 help="the number of layers for FSC/BSC")
cfg.add_argument("--mfiles", type=str,
                 default="textual_modality.pkl,visual_modality.pkl",
                 help="the files saving modality")
cfg.add_argument("--num-neighbors", type=str, default='5-1',
                 help="for kNN graph")
cfg.add_argument("--gamma", type=float, default=0.2)

# --- SBN-BSC v4 params (support --sbn-*, --sbn_*, and short versions) ---
cfg.add_argument("--sbn-tau-text", "--tau-text", "--tau_text", "--sbn_tau_text", dest="sbn_tau_text", type=float, default=0.15,
                 help="Text similarity threshold for modal agreement [0.05, 0.30]")
cfg.add_argument("--sbn-tau-visual", "--tau-visual", "--tau_visual", "--sbn_tau_visual", dest="sbn_tau_visual", type=float, default=0.10,
                 help="Visual similarity threshold for modal agreement [0.05, 0.25]")
cfg.add_argument("--sbn-modal-discount", "--modal-discount", "--modal_discount", "--sbn_modal_discount", dest="sbn_modal_discount", type=float, default=0.50,
                 help="Modal discount factor rho [0.30, 0.80]")
cfg.add_argument("--sbn-prune-lambda", "--prune-lambda", "--prune_lambda", "--sbn_prune_lambda", dest="sbn_prune_lambda", type=float, default=0.50,
                 help="Adaptive pruning lambda multiplier [0.30, 0.90]")
cfg.add_argument("--sbn-min-edge-threshold", "--min-edge-threshold", "--min_edge_threshold", "--sbn_min_edge_threshold", dest="sbn_min_edge_threshold", type=float, default=0.05,
                 help="Minimum edge quality threshold tau_min [0.01, 0.10]")
cfg.add_argument("--sbn-ablation-config", "--ablation-config", "--ablation_config", "--sbn_ablation_config", dest="sbn_ablation_config", type=str, default="A6_full_sbn_bsc_v4",
                 help="Ablation configuration selector (A0_baseline -> A6_full_sbn_bsc_v4)")

cfg.set_defaults(
    description="STAIR-SBN-BSC-v4",
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

# === YAML Configuration Loader & Normalizer (Bug #1 & Bug #2 Fix) ===
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
        import sys
        print(f"[Config] Nạp cấu hình siêu tham số từ YAML: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_cfg = yaml.safe_load(f) or {}

        # Thu thập các tham số được truyền trực tiếp từ CLI để ưu tiên cao nhất (CLI > YAML > Defaults)
        cli_specified = set()
        for a in sys.argv[1:]:
            if a.startswith('--'):
                clean_a = a.lstrip('-').split('=')[0].replace('-', '_').lower()
                cli_specified.add(clean_a)
                if clean_a.startswith('sbn_'):
                    cli_specified.add(clean_a[4:])
                else:
                    cli_specified.add(f"sbn_{clean_a}")

        for raw_key, val in yaml_cfg.items():
            # Canonical normalization: convert kebab-case and uppercase to snake_case
            norm_key = raw_key.replace('-', '_').lower()

            if norm_key in cli_specified or f"sbn_{norm_key}" in cli_specified:
                print(f"[Config] Giữ nguyên giá trị CLI cho '{norm_key}' (không bị YAML ghi đè)")
                continue

            matched = False
            # 1. Exact canonical match (e.g. sbn_tau_text)
            if hasattr(cfg, norm_key):
                setattr(cfg, norm_key, val)
                matched = True
            # 2. Add sbn_ prefix (e.g. tau_text -> sbn_tau_text)
            elif hasattr(cfg, f"sbn_{norm_key}"):
                setattr(cfg, f"sbn_{norm_key}", val)
                matched = True
            # 3. Strip sbn_ prefix (e.g. sbn_dataset -> dataset)
            elif norm_key.startswith("sbn_") and hasattr(cfg, norm_key[4:]):
                setattr(cfg, norm_key[4:], val)
                matched = True
            elif hasattr(cfg, raw_key):
                setattr(cfg, raw_key, val)
                matched = True

            if matched:
                print(f"[Config] Override {norm_key} = {val} từ YAML")
            else:
                # Set dynamic attribute if not present in parser
                setattr(cfg, norm_key, val)
                print(f"[Config] Đặt giá trị mới {norm_key} = {val} từ YAML")
    else:
        print(f"[Config Cảnh báo] Không tìm thấy file config: {config_path}")

# Đảm bảo cfg.root luôn là đường dẫn tuyệt đối chuẩn xác
if hasattr(cfg, 'root') and cfg.root:
    cfg.root = os.path.abspath(cfg.root)

cfg.mfiles = cfg.mfiles.split(',')
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-')))

# beta3 here is the 1 - beta_j for BSC
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ============================================================================
# Model: STAIR-SBN-BSC v4
# ============================================================================

class STAIR_SBN_BSC_v4(freerec.models.GenRecArch):
    """
    STAIR-SBN-BSC v4 (Precomputed Structural Denoising for BSC).

    Kế thừa toàn bộ kiến trúc STAIR Baseline:
      - Forward: FSC trên đồ thị User-Item bipartite
      - Loss: BPR Ranking thuần túy
      - Backward: AdamWSEvo + Smoother(mAdj)

    Cải tiến duy nhất: mAdj được thay thế bằng ma trận kNN đã lọc sạch nhiễu
    qua SBN_BSC_Preprocessor (offline, 1 lần duy nhất trong prepare()).
    """

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
        r"""
        Compute the kNN graph.
        """
        features = F.normalize(features, dim=-1)  # (N, D)
        sim = features @ features.t()  # (N, N)
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(
            sim, k, symmetric=False
        )
        return edge_index

    def prepare(self, path: str):
        """
        Tiền xử lý dữ liệu:
        1. Load modality features (text + visual)
        2. Build raw kNN graph
        3. *** SBN-BSC v4: Build denoised mAdj thay thế kNN thô ***
        4. SVD Whitening → Item embedding initialization
        5. User embedding initialization via R @ mfeats
        """
        from freerec.utils import import_pickle

        # === Load modality features ===
        mfeats = [
            import_pickle(
                os.path.join(path, mfile)
            )
            for mfile in cfg.mfiles
        ]

        # === Build raw kNN graph (giữ lại edge_index trước khi coalesce) ===
        raw_edge_index = torch.cat(
            [self.get_knn_graph(feats, k)
             for feats, k in zip(mfeats, cfg.num_neighbors)],
            dim=1
        )

        # === Extract train user-item interaction matrix as scipy sparse ===
        edge_index_ui = self.dataset.train().to_bigraph(
            edge_type='u2i'
        )['u2i'].edge_index
        num_users = self.User.count
        num_items = self.Item.count

        # Build scipy CSR from edge_index
        row_ui = edge_index_ui[0].cpu().numpy()
        col_ui = edge_index_ui[1].cpu().numpy()
        data_ui = np.ones(len(row_ui), dtype=np.float32)
        train_R = sp.csr_matrix(
            (data_ui, (row_ui, col_ui)),
            shape=(num_users, num_items)
        )

        # === SBN-BSC v4: Build denoised mAdj ===
        preprocessor = SBN_BSC_Preprocessor(
            tau_text=getattr(cfg, 'sbn_tau_text', getattr(cfg, 'tau_text', 0.15)),
            tau_visual=getattr(cfg, 'sbn_tau_visual', getattr(cfg, 'tau_visual', 0.10)),
            modal_discount=getattr(cfg, 'sbn_modal_discount', getattr(cfg, 'modal_discount', 0.50)),
            prune_lambda=getattr(cfg, 'sbn_prune_lambda', getattr(cfg, 'prune_lambda', 0.50)),
            min_edge_threshold=getattr(cfg, 'sbn_min_edge_threshold', getattr(cfg, 'min_edge_threshold', 0.05)),
            ablation_config=getattr(cfg, 'sbn_ablation_config', None),
            verbose=True,
        )

        # text_feats = mfeats[0], vis_feats = mfeats[1]
        mAdj_clean = preprocessor.build_denoised_mAdj(
            text_feats=mfeats[0],
            vis_feats=mfeats[1],
            train_user_item_matrix=train_R,
            raw_knn_edge_index=raw_edge_index,
            num_items=num_items,
        )

        # Register denoised mAdj as buffer
        self.register_buffer(
            'mAdj',
            mAdj_clean.to(cfg.device)
        )

        # Safety assertions
        assert self.mAdj.is_sparse_csr, "mAdj must be in torch.sparse_csr format!"
        assert self.mAdj._nnz() > 0, "mAdj cannot be empty after pruning!"
        print(f"[STAIR-SBN-BSC v4] mAdj registered: nnz={self.mAdj._nnz():,}")

        # Print graph statistics
        print_graph_stats(self.mAdj, label="Denoised mAdj")

        # === SVD Whitening → MI (Modality Initialization) ===
        mfeats_whitened = [
            self.whitening(mfeat) * k
            for mfeat, k in zip(mfeats, cfg.num_neighbors)
        ]
        mfeats_combined = sum(mfeats_whitened).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats_combined)

        # === User embedding initialization via R @ mfeats ===
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
        )  # (N, D)

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

class CoachForSTAIR_SBN_BSC_v4(freerec.launcher.Coach):

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


# ============================================================================
# Main Entry Point
# ============================================================================

def main():

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

    model = STAIR_SBN_BSC_v4(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_SBN_BSC_v4(
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
