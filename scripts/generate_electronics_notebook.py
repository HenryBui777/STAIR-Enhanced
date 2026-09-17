# -*- coding: utf-8 -*-
"""
scripts/generate_electronics_notebook.py
=========================================
Tạo notebook chuyên biệt cho Amazon Electronics: notebook/stair_electronics.ipynb
- Cấu hình phương pháp DCD-Gated (Dim=256D, Cosine LR Warmup 15 eps, Patience=30, Max Epochs=500).
- Tự động quét và kết nối dataset Amazon2014Electronics_550_MMRec từ /kaggle/input.
- Đối sánh kết quả trực tiếp với STAIR Baseline (64D) trên toàn bộ 4 chỉ số khoa học.
"""
import copy
import json
import os

with open(r"d:\STAIR-Enhanced\models\stair_breakthrough.py", "r", encoding="utf-8") as f:
    breakthrough_model_code = f.read()

with open(r"d:\STAIR-Enhanced\models\stair_ne_nlgcl_v5_plus.py", "r", encoding="utf-8") as f:
    v5_plus_model_code = f.read()

with open(r"d:\STAIR-Enhanced\main_stair_ne_nlgcl_v5_plus.py", "r", encoding="utf-8") as f:
    runner_code = f.read()

template_path = r"d:\STAIR-Enhanced\notebook\P3\stair_ne_nlgcl_v5_plus.ipynb"
with open(template_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Clear outputs and execution count for clean notebook
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# Cell 0: Header
nb["cells"][0]["source"] = [
    "# 🚀 GIAI ĐOẠN 3: STAIR DCD-GATED TRÊN AMAZON ELECTRONICS\n",
    "## 🏆 Embedding Dim = 256 | Batch Size = 4096 | Cosine Warmup LR (15 eps) | Patience = 30 | Checkpoint: NDCG@20 (500 eps)\n",
    "*(Tập trung độc quyền trên tập dữ liệu quy mô lớn: **Amazon Electronics (~1.7M tương tác, 63K sản phẩm)**)*\n",
    "---\n",
    "### 🎯 PHƯƠNG PHÁP ĐỘT PHÁ TRIỂN KHAI:\n",
    "* **`dcd_gated` (Dual-Consensus Denoising & Gated Residuals):**\n",
    "  - Đã thiết lập kỷ lục nhảy vọt **+14.00% NDCG@20 (+18.27% NDCG@10)** trên Amazon Sports.\n",
    "  - Cơ chế làm dày cạnh ảo đồng thuận kép: Hành vi đồng mua (Ochiai) $\\odot$ Tương đồng đặc trưng Modal ảnh/văn bản.\n",
    "  - Van Gating an toàn tự động khép lại khi có tín hiệu nhiễu, chống suy thoái biểu diễn.\n",
    "---\n",
    "### 📌 CẤU HÌNH THAM SỐ:\n",
    "* **Dataset:** `Amazon2014Electronics_550_MMRec` (Batch size: 4096, Gamma: 0.4).\n",
    "* **Embedding Dimension:** `256` (gấp 4 lần Baseline 64D).\n",
    "* **Max Epochs:** `500` epochs với **Early Stopping Patience = 30 epochs** theo chỉ số trọng tâm **`NDCG@20`**.\n",
    "* **Cosine LR Warmup:** 15 epoch đầu tăng tuyến tính từ `1e-6` -> `1e-3`, sau đó decay theo Cosine về `1e-6`.\n"
]

# Cell 2: Environment & Sync (embedded code)
cell1_source = [
    "# Cell 1: Môi trường, Dependencies & Đồng bộ STAIR-Enhanced\n",
    "import os, shutil, subprocess, sys\n",
    "\n",
    "STAIR_DIR = '/kaggle/working/STAIR-Enhanced'\n",
    "os.chdir('/kaggle/working')\n",
    "\n",
    "# 1. Luôn clone hoặc đồng bộ cưỡng bức repository mới nhất từ origin/main\n",
    "if os.path.exists(STAIR_DIR):\n",
    "    print(\"Thư mục STAIR-Enhanced đã tồn tại. Đang đồng bộ cưỡng bức mã nguồn mới nhất...\")\n",
    "    try:\n",
    "        subprocess.run(['git', '-C', STAIR_DIR, 'fetch', 'origin', 'main'], check=True)\n",
    "        subprocess.run(['git', '-C', STAIR_DIR, 'reset', '--hard', 'origin/main'], check=True)\n",
    "        print(\"✅ Đã reset về commit mới nhất của origin/main.\")\n",
    "    except Exception as e:\n",
    "        print(f\"Lỗi git fetch/reset ({e}), đang làm sạch và clone lại từ đầu...\")\n",
    "        shutil.rmtree(STAIR_DIR, ignore_errors=True)\n",
    "\n",
    "if not os.path.exists(STAIR_DIR):\n",
    "    print(\"Cloning STAIR-Enhanced repository (branch main)...\" )\n",
    "    subprocess.run([\n",
    "        'git', 'clone', '--depth', '1',\n",
    "        'https://github.com/HenryBui777/STAIR-Enhanced.git', STAIR_DIR\n",
    "    ], check=True)\n",
    "\n",
    "for p in [STAIR_DIR, '/kaggle/working']:\n",
    "    if p not in sys.path:\n",
    "        sys.path.insert(0, p)\n",
    "\n",
    "if os.path.exists(STAIR_DIR):\n",
    "    os.chdir(STAIR_DIR)\n",
    "\n",
    "# Ghi đảm bảo module breakthrough, v5_plus và runner tồn tại trên đĩa\n",
    "os.makedirs(os.path.join(STAIR_DIR, 'models'), exist_ok=True)\n",
    "with open(os.path.join(STAIR_DIR, 'models', 'stair_breakthrough.py'), 'w', encoding='utf-8') as f:\n",
    f"    f.write({repr(breakthrough_model_code)})\n",
    "with open(os.path.join(STAIR_DIR, 'models', 'stair_ne_nlgcl_v5_plus.py'), 'w', encoding='utf-8') as f:\n",
    f"    f.write({repr(v5_plus_model_code)})\n",
    "with open(os.path.join(STAIR_DIR, 'main_stair_ne_nlgcl_v5_plus.py'), 'w', encoding='utf-8') as f:\n",
    f"    f.write({repr(runner_code)})\n",
    "\n",
    "# Xóa cache module để kernel luôn nạp phiên bản mới nhất từ đĩa\n",
    "for mod_name in list(sys.modules.keys()):\n",
    "    if 'stair_ne_nlgcl' in mod_name or 'stair_breakthrough' in mod_name:\n",
    "        sys.modules.pop(mod_name, None)\n",
    "\n",
    "# 2. Cài đặt các gói phụ thuộc bắt buộc\n",
    "print(\"📦 Cài đặt dependencies (torchdata, torch_geometric, freerec, nvidia-ml-py, prettytable)...\")\n",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'torchdata==0.7.1'], check=False)\n",
    "subprocess.run([\n",
    "    sys.executable, '-m', 'pip', 'install', '-q',\n",
    "    'torch_geometric', 'freerec==0.8.5', 'nvidia-ml-py', 'prettytable', 'matplotlib', 'pyyaml', 'seaborn'\n",
    "], check=True)\n",
    "\n",
    "# 3. Kaggle TorchData compatibility shims cho FreeRec (PyTorch 2.x & Python 3.10+)\n",
    "import types\n",
    "import torch.utils.data\n",
    "\n",
    "try:\n",
    "    import torchdata\n",
    "    import torchdata.datapipes as dp\n",
    "except Exception:\n",
    "    dp = None\n",
    "\n",
    "if dp is None or 'torchdata.datapipes' not in sys.modules:\n",
    "    if 'torchdata' not in sys.modules:\n",
    "        td = types.ModuleType('torchdata')\n",
    "        sys.modules['torchdata'] = td\n",
    "    else:\n",
    "        td = sys.modules['torchdata']\n",
    "    dp = types.ModuleType('torchdata.datapipes')\n",
    "    td.datapipes = dp\n",
    "    sys.modules['torchdata.datapipes'] = dp\n",
    "\n",
    "if not hasattr(dp, 'iter'):\n",
    "    iter_mod = types.ModuleType('torchdata.datapipes.iter')\n",
    "    dp.iter = iter_mod\n",
    "    sys.modules['torchdata.datapipes.iter'] = iter_mod\n",
    "if not hasattr(dp.iter, 'IterDataPipe'):\n",
    "    class IterDataPipe(torch.utils.data.IterableDataset):\n",
    "        def __iter__(self): return iter([])\n",
    "    dp.iter.IterDataPipe = IterDataPipe\n",
    "\n",
    "if not hasattr(dp, 'map'):\n",
    "    map_mod = types.ModuleType('torchdata.datapipes.map')\n",
    "    dp.map = map_mod\n",
    "    sys.modules['torchdata.datapipes.map'] = map_mod\n",
    "if not hasattr(dp.map, 'MapDataPipe'):\n",
    "    class MapDataPipe(torch.utils.data.Dataset):\n",
    "        def __getitem__(self, idx): raise NotImplementedError\n",
    "        def __len__(self): return 0\n",
    "    dp.map.MapDataPipe = MapDataPipe\n",
    "\n",
    "if not hasattr(dp, 'functional_datapipe'):\n",
    "    def functional_datapipe(name, enable_df_datapipes_support=False):\n",
    "        def decorator(cls):\n",
    "            def method(self, *args, **kwargs): return cls(self, *args, **kwargs)\n",
    "            if hasattr(dp, 'iter') and hasattr(dp.iter, 'IterDataPipe'): setattr(dp.iter.IterDataPipe, name, method)\n",
    "            if hasattr(dp, 'map') and hasattr(dp.map, 'MapDataPipe'): setattr(dp.map.MapDataPipe, name, method)\n",
    "            return cls\n",
    "        return decorator\n",
    "    dp.functional_datapipe = functional_datapipe\n",
    "\n",
    "print(\"✅ Môi trường STAIR-Enhanced & Dependencies đã hoàn tất sẵn sàng!\")\n"
]
nb["cells"][2]["source"] = cell1_source

# Cell 4: Data Preparation for Amazon Electronics
nb["cells"][3]["source"] = [
    "## Cell 2 📂 Chuẩn bị Dữ liệu Amazon Electronics từ Kaggle Input (Tự động quét & Đồng bộ)\n",
    "Tự động quét toàn bộ `/kaggle/input` để phát hiện thư mục dữ liệu `Amazon2014Electronics_550_MMRec` (hỗ trợ cả dạng nén zip, thư mục con, hoặc đặt tên dataset bất kỳ có chứa chữ `electronics`) và liên kết vào `/kaggle/data/Processed/Amazon2014Electronics_550_MMRec`."
]

cell2_source = [
    "# Cell 2: Chuẩn bị dữ liệu cho Amazon Electronics (Tự động đợi download xong)\n",
    "import os, shutil, glob, time\n",
    "\n",
    "DATA_ROOT = '/kaggle/data'\n",
    "PROCESSED_ROOT = os.path.join(DATA_ROOT, 'Processed')\n",
    "LOCAL_DATA = '/kaggle/working/STAIR-Enhanced/data'\n",
    "LOCAL_PROCESSED = os.path.join(LOCAL_DATA, 'Processed')\n",
    "\n",
    "for d in [DATA_ROOT, PROCESSED_ROOT, LOCAL_DATA, LOCAL_PROCESSED]:\n",
    "    os.makedirs(d, exist_ok=True)\n",
    "\n",
    "TARGET_DATASETS = {\n",
    "    'electronics': ('Amazon2014Electronics_550_MMRec', ['elect', 'electronic', 'electronics', 'amazon2014electronics']),\n",
    "}\n",
    "\n",
    "REQUIRED_EXTENSIONS = ('.npy', '.pkl', '.txt', '.inter', '.item', '.pt', '.csv', '.yaml')\n",
    "\n",
    "def bridge_directories(src_dir, target_folder):\n",
    "    destinations = [\n",
    "        os.path.join(DATA_ROOT, target_folder),\n",
    "        os.path.join(PROCESSED_ROOT, target_folder),\n",
    "        os.path.join(LOCAL_DATA, target_folder),\n",
    "        os.path.join(LOCAL_PROCESSED, target_folder),\n",
    "    ]\n",
    "    for dst in destinations:\n",
    "        if os.path.abspath(src_dir) == os.path.abspath(dst):\n",
    "            continue\n",
    "        os.makedirs(dst, exist_ok=True)\n",
    "        for item in os.listdir(src_dir):\n",
    "            s_item = os.path.join(src_dir, item)\n",
    "            d_item = os.path.join(dst, item)\n",
    "            if os.path.isfile(s_item) and not os.path.exists(d_item):\n",
    "                try:\n",
    "                    os.symlink(s_item, d_item)\n",
    "                except Exception:\n",
    "                    shutil.copy2(s_item, d_item)\n",
    "\n",
    "def scan_and_prepare_data():\n",
    "    input_base = '/kaggle/input'\n",
    "    print(\"🔍 Đang quét dữ liệu Amazon Electronics...\")\n",
    "    \n",
    "    for attempt in range(12):\n",
    "        found_datasets = {}\n",
    "        for key, (target_folder, keywords) in TARGET_DATASETS.items():\n",
    "            processed_dst = os.path.join(PROCESSED_ROOT, target_folder)\n",
    "            raw_dst = os.path.join(DATA_ROOT, target_folder)\n",
    "            \n",
    "            # 1. Đã có sẵn đủ file\n",
    "            for check_p in [processed_dst, raw_dst, os.path.join(LOCAL_DATA, target_folder)]:\n",
    "                if os.path.exists(check_p) and len(os.listdir(check_p)) >= 4:\n",
    "                    bridge_directories(check_p, target_folder)\n",
    "                    found_datasets[key] = processed_dst\n",
    "                    break\n",
    "            if key in found_datasets:\n",
    "                continue\n",
    "            \n",
    "            # 2. Tìm trong /kaggle/input\n",
    "            candidates = []\n",
    "            if os.path.exists(input_base):\n",
    "                for root, dirs, files in os.walk(input_base):\n",
    "                    if target_folder in dirs:\n",
    "                        cand = os.path.join(root, target_folder)\n",
    "                        if os.path.exists(cand) and len(os.listdir(cand)) >= 4:\n",
    "                            candidates.append(cand)\n",
    "                    elif any(f.endswith('.pkl') for f in files) and any(kw in root.lower() for kw in keywords):\n",
    "                        if len(files) >= 4:\n",
    "                            candidates.append(root)\n",
    "            \n",
    "            if candidates:\n",
    "                src = candidates[0]\n",
    "                os.makedirs(processed_dst, exist_ok=True)\n",
    "                for f in os.listdir(src):\n",
    "                    if f.endswith(REQUIRED_EXTENSIONS):\n",
    "                        shutil.copy2(os.path.join(src, f), os.path.join(processed_dst, f))\n",
    "                bridge_directories(processed_dst, target_folder)\n",
    "                print(f\"  [TÌM THẤY & KẾT NỐI] {key.upper()} -> {processed_dst} ({len(os.listdir(processed_dst))} tệp)\")\n",
    "                found_datasets[key] = processed_dst\n",
    "        \n",
    "        if len(found_datasets) == len(TARGET_DATASETS):\n",
    "            return found_datasets\n",
    "        else:\n",
    "            print(f\"⏳ Dataset đang được Kaggle tải về (Adding data...)... Đợi 5s (lần {attempt+1}/12)...\")\n",
    "            time.sleep(5)\n",
    "            \n",
    "    return found_datasets\n",
    "\n",
    "prepared_data = scan_and_prepare_data()\n",
    "print(\"=\" * 75)\n",
    "print(f\"TỔNG KẾT DỮ LIỆU: {len(prepared_data)} / {len(TARGET_DATASETS)} tập đã sẵn sàng trong Processed\")\n",
    "for k, (tf, _) in TARGET_DATASETS.items():\n",
    "    p_dir = os.path.join(PROCESSED_ROOT, tf)\n",
    "    status = f\"✅ {len(os.listdir(p_dir))} tệp sẵn sàng\" if (os.path.exists(p_dir) and len(os.listdir(p_dir)) >= 4) else \"❌ THIẾU\"\n",
    "    print(f\"  * {k.upper():12s} ({tf}): {status}\")\n",
    "print(\"=\" * 75)\n"
]
nb["cells"][4]["source"] = cell2_source

# Cell 10: Configs for Electronics
nb["cells"][9]["source"] = [
    "## Cell 5 📋 Cấu hình Siêu tham số DCD-Gated cho Amazon Electronics\n",
    "Thiết lập tham số tối ưu cho **Amazon Electronics (~1.7M tương tác)**:\n",
    "- `embedding_dim`: 256 (Tăng năng lực biểu diễn).\n",
    "- `batch_size`: 4096 (Tối ưu thông lượng và bộ nhớ trên GPU T4).\n",
    "- `lr_warmup_epochs`: 15 eps -> Cosine Annealing về 1e-6.\n",
    "- `patience`: 30 epochs dừng sớm khi không cải thiện `NDCG@20`.\n",
    "- `weight_decay`: 0.1, `tau`: 0.20, `eps`: 0.08, `tau_thresh`: 0.85, `lambda_cl`: 0.010, `gamma_h`: 0.15.\n"
]

cell5_source = [
    "# Cell 5: Cấu hình Siêu tham số cho Amazon Electronics (Dim 256, Cosine LR, Patience 30, Epochs 500)\n",
    "import os\n",
    "\n",
    "os.makedirs('/kaggle/working/logs/breakthrough', exist_ok=True)\n",
    "\n",
    "V5_PLUS_CONFIGS = {\n",
    "    'electronics': {\n",
    "        'yaml':              '/kaggle/working/STAIR-Enhanced/configs/Amazon2014Electronics_550_MMRec.yaml',\n",
    "        'embedding_dim':     256,\n",
    "        'epochs':            500,\n",
    "        'lr_warmup_epochs':  15,\n",
    "        'min_lr':            1e-6,\n",
    "        'patience':          30,\n",
    "        'weight_decay':      0.1,\n",
    "        'tau':               0.20,\n",
    "        'alpha_dir':         0.50,\n",
    "        'eps':               0.08,\n",
    "        'tau_thresh':        0.85,\n",
    "        'lambda_cl':         0.010,\n",
    "        'gamma_h':           0.15,\n",
    "        'warmup_epochs':     50,\n",
    "    },\n",
    "}\n",
    "\n",
    "print('=' * 85)\n",
    "print('✅ CẤU HÌNH THỰC NGHIỆM CHO AMAZON ELECTRONICS ĐÃ SẴN SÀNG (DIM=256, CHECKPOINT=NDCG@20):')\n",
    "for k, v in V5_PLUS_CONFIGS.items():\n",
    "    print(f\"  • [{k.upper()}]: Epochs={v['epochs']}, Dim={v['embedding_dim']}, LR Warmup={v['lr_warmup_epochs']} eps, Patience={v['patience']} eps, Lambda_CL={v['lambda_cl']}\")\n",
    "print('=' * 85)\n"
]
nb["cells"][10]["source"] = cell5_source

# Cell 12: Training Execution on Electronics
nb["cells"][11]["source"] = [
    "## Cell 6 🏋️ Huấn luyện DCD-Gated trên Amazon Electronics\n",
    "Chạy huấn luyện chính thức với phương pháp đột phá **DCD-Gated** (`SELECTED_METHOD = 'dcd_gated'`).\n",
    "Hệ thống sẽ tự động giám sát checkpoint theo `NDCG@20` cao nhất, kích hoạt dừng sớm sau 30 epochs nếu hội tụ, và đánh giá toàn diện trên tập kiểm thử độc lập (TEST)."
]

cell6_source = [
    "# Cell 6: Huấn luyện trên Amazon Electronics\n",
    "# Tùy chọn method: 'dcd_gated' (Đột phá), 'dan_tans' (Hướng 1), 'appnp_crossmodal' (Hướng 3), 'v5_plus' (gốc)\n",
    "SELECTED_METHOD = 'dcd_gated' # <--- Phương pháp đột phá đã lập kỷ lục trên Sports!\n",
    "\n",
    "DATA_ROOT = '/kaggle/data'\n",
    "\n",
    "if 'electronics' in prepared_data:\n",
    "    cfg_e = V5_PLUS_CONFIGS['electronics']\n",
    "    log_p = f\"/kaggle/working/logs/breakthrough/electronics_{SELECTED_METHOD}_dim256.log\"\n",
    "    run_training_v5_plus(\n",
    "        key='electronics',\n",
    "        yaml_cfg=cfg_e['yaml'],\n",
    "        data_root=DATA_ROOT,\n",
    "        log_path=log_p,\n",
    "        method=SELECTED_METHOD,\n",
    "        **{k: v for k, v in cfg_e.items() if k != 'yaml'}\n",
    "    )\n",
    "else:\n",
    "    print('⚠️ Bỏ qua Amazon Electronics do thiếu dữ liệu trong /kaggle/input.')\n"
]
nb["cells"][12]["source"] = cell6_source

# Cell 14: PrettyTable for Electronics
nb["cells"][13]["source"] = [
    "## Cell 7 📊 Bảng Tổng Kết Kết Quả Thực Nghiệm (Amazon Electronics)\n",
    "Bảng tổng hợp đối sánh 4 chỉ số khoa học: **Recall@10, Recall@20, NDCG@10, NDCG@20** và mức tăng trưởng trọng tâm **NDCG@20 (Δ vs Baseline N@20)** so với mốc STAIR Baseline gốc."
]

cell7_source = [
    "# Cell 7: Bảng tổng kết kết quả thực nghiệm Amazon Electronics\n",
    "import os, glob\n",
    "from prettytable import PrettyTable\n",
    "\n",
    "BASELINE = {\n",
    "    'electronics': {'Recall@10': 0.0442, 'Recall@20': 0.0665, 'NDCG@10': 0.0246, 'NDCG@20': 0.0303},\n",
    "}\n",
    "\n",
    "table = PrettyTable()\n",
    "table.field_names = ['Tập dữ liệu', 'Phương pháp', 'Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20', 'Δ vs Baseline N@20']\n",
    "\n",
    "for ds in ['electronics']:\n",
    "    bl = BASELINE[ds]\n",
    "    table.add_row([ds.upper(), 'STAIR Baseline (64D)', f\"{bl['Recall@10']:.4f}\", f\"{bl['Recall@20']:.4f}\", f\"{bl['NDCG@10']:.4f}\", f\"{bl['NDCG@20']:.4f}\", '-'])\n",
    "    \n",
    "    log_files = glob.glob(f\"/kaggle/working/logs/breakthrough/{ds}_*.log\")\n",
    "    for lf in log_files:\n",
    "        ep, m = extract_best_test(lf)\n",
    "        if m and len(m) >= 4:\n",
    "            tag = os.path.basename(lf).replace(f\"{ds}_\", \"\").replace(\".log\", \"\")\n",
    "            gain = (m['NDCG@20'] - bl['NDCG@20']) / bl['NDCG@20'] * 100\n",
    "            sign = '+' if gain >= 0 else ''\n",
    "            table.add_row([ds.upper(), f\"★ {tag} (@Ep{ep})\", f\"{m['Recall@10']:.4f}\", f\"{m['Recall@20']:.4f}\", f\"{m['NDCG@10']:.4f}\", f\"{m['NDCG@20']:.4f}\", f\"{sign}{gain:.2f}%\"])\n",
    "\n",
    "print(table)\n"
]
nb["cells"][14]["source"] = cell7_source

out_path = r"d:\STAIR-Enhanced\notebook\stair_electronics.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Successfully generated notebook for Electronics: {out_path}")
