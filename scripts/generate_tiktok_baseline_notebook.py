# -*- coding: utf-8 -*-
"""
scripts/generate_tiktok_baseline_notebook.py
=============================================
Tạo notebook tái lập nguyên bản STAIR Baseline (AAAI 2025) trên TikTok: notebook/stair_tiktok_baseline.ipynb
- Mô hình: STAIR Baseline gốc (Forward Stepwise Convolution + BSC Smoother).
- Cấu hình: Dim=64D, Batch Size=1024, Tri-modal (Vision, Text, Audio), Gamma=0.05, kNN='3-3-3', Epochs=500.
- Tự động quét và phát hiện dataset TikTok từ /kaggle/input (trnMat.pkl, audio_feat.npy, ...)
  hoặc giải nén tự động từ data/tiktok.zip.
- Tự động chuyển đổi định dạng raw sang chuẩn FreeRec/STAIR.
- Trích xuất bảng kết quả 4 chỉ số khoa học và so sánh trực tiếp với log mốc 14/09/2026.
"""
import copy
import json
import os

with open(r"d:\STAIR-Enhanced\main.py", "r", encoding="utf-8") as f:
    stair_main_code = f.read()

with open(r"d:\STAIR-Enhanced\configs\tiktok_MMRec.yaml", "r", encoding="utf-8") as f:
    tiktok_yaml_code = f.read()

nb = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# -------------------------------------------------------------
# Cell 0: Header
# -------------------------------------------------------------
nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🎯 TÁI LẬP STAIR BASELINE TRÊN TẬP DỮ LIỆU TIKTOK (TRI-MODAL MICRO-VIDEO)\n",
        "## 🏆 Mô hình STAIR Gốc (AAAI 2025) | Embedding Dim = 64D | Batch Size = 1024 | Tri-modal (Vision + Text + Audio) | Epochs = 500 | Checkpoint: NDCG@20\n",
        "*(Tái tạo chuẩn tắc mốc đối chuẩn nền tảng STAIR Baseline trên tập video ngắn đa giác quan: **TikTok (DiffMM / ACM MM 2024 - 9,308 Users, 6,710 Items)**)*\n",
        "---\n",
        "### 📌 CẤU HÌNH THAM SỐ CHUẨN TẮC (PAPER CONFIG):\n",
        "* **Dataset:** `tiktok` (Tương tác: 59,541 train / 3,051 valid / 6,130 test).\n",
        "* **Đa phương thức (Tri-modal):** `visual_modality.pkl` (128D), `textual_modality.pkl` (768D), `audio_modality.pkl` (128D).\n",
        "* **Đồ thị k-NN Tam phân:** `num_neighbors: '3-3-3'` (phân bổ đều $3$ láng giềng cho mỗi phương thức, tỷ lệ $33.3\\%$ mỗi kênh).\n",
        "* **Suy giảm phổ FSC:** $\\gamma = 0.05$ (bảo tồn thông tin đa phương tiện dải tần trung và cao trong video ngắn).\n",
        "* **Embedding Dim:** `64D` | **Số tầng tích chập:** `L = 3` | **Batch Size:** `1024`.\n",
        "* **Bộ tối ưu:** `AdamWSEvo` ($lr = 10^{-3}$, weight decay $= 0.1$, $\\beta_1 = 0.9, \\beta_2 = 0.999$).\n",
        "* **Checkpoint Selection:** Giám sát `NDCG@20` cao nhất trên tập Validation (`which4best: NDCG@20`), sau đó đánh giá chính thức trên tập TEST.\n"
    ]
})

# -------------------------------------------------------------
# Cell 1: Environment & Dependencies
# -------------------------------------------------------------
nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Cell 1 ⚙️ Thiết lập Môi trường, Dependencies & Đồng bộ STAIR-Enhanced\n",
        "Khởi tạo môi trường Kaggle, clone/đồng bộ mã nguồn mới nhất từ branch `main` của repository [STAIR-Enhanced](https://github.com/HenryBui777/STAIR-Enhanced.git), cài đặt các thư viện bắt buộc (`freerec==0.8.5`, `torchdata==0.7.1`, `torch_geometric`, `pynvml`, `prettytable`, `scipy`), và nhúng mã nguồn `main.py` cùng tệp cấu hình `tiktok_MMRec.yaml`."
    ]
})

cell1_code = [
    "# Cell 1: Môi trường, Dependencies & Đồng bộ STAIR-Enhanced\n",
    "import os, shutil, subprocess, sys\n",
    "\n",
    "STAIR_DIR = '/kaggle/working/STAIR-Enhanced'\n",
    "os.chdir('/kaggle/working')\n",
    "\n",
    "# 1. Đồng bộ repository mới nhất từ origin/main\n",
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
    "# Ghi đảm bảo main.py và configs/tiktok_MMRec.yaml tồn tại trên đĩa\n",
    "os.makedirs(os.path.join(STAIR_DIR, 'configs'), exist_ok=True)\n",
    "with open(os.path.join(STAIR_DIR, 'main.py'), 'w', encoding='utf-8') as f:\n",
    f"    f.write({repr(stair_main_code)})\n",
    "with open(os.path.join(STAIR_DIR, 'configs', 'tiktok_MMRec.yaml'), 'w', encoding='utf-8') as f:\n",
    f"    f.write({repr(tiktok_yaml_code)})\n",
    "\n",
    "# 2. Cài đặt các gói phụ thuộc bắt buộc\n",
    "print(\"📦 Cài đặt dependencies (torchdata, torch_geometric, freerec, nvidia-ml-py, prettytable, scipy)...\")\n",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'torchdata==0.7.1'], check=False)\n",
    "subprocess.run([\n",
    "    sys.executable, '-m', 'pip', 'install', '-q',\n",
    "    'torch_geometric', 'freerec==0.8.5', 'nvidia-ml-py', 'prettytable', 'matplotlib', 'pyyaml', 'seaborn', 'scipy'\n",
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
    "print(\"✅ Môi trường STAIR Baseline & Dependencies đã sẵn sàng 100%!\")\n"
]
nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell1_code
})

# -------------------------------------------------------------
# Cell 2: Data Preparation & Adapter
# -------------------------------------------------------------
nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Cell 2 📂 Chuẩn bị Dữ liệu TikTok từ Kaggle Input (Tự động Adapter sang FreeRec)\n",
        "Tự động quét toàn bộ `/kaggle/input` để phát hiện dữ liệu TikTok (chứa `trnMat.pkl`, `audio_feat.npy`, `image_feat.npy`, `text_feat.npy`, ...).\n",
        "- Nếu là dữ liệu thô (raw DiffMM / ACM MM 2024), hệ thống tự động chuyển đổi sang chuẩn FreeRec (`train.txt`, `valid.txt`, `test.txt`, `visual_modality.pkl`, `textual_modality.pkl`, `audio_modality.pkl`).\n",
        "- Tự động liên kết (symlink/bridge) vào `/kaggle/data/Processed/tiktok` và `/kaggle/data/tiktok` sẵn sàng cho huấn luyện."
    ]
})

cell2_code = [
    "# Cell 2: Chuẩn bị và Chuyển đổi Dữ liệu TikTok (Tự động quét & Adapter sang FreeRec)\n",
    "import os, shutil, glob, time, pickle, zipfile\n",
    "import numpy as np\n",
    "import torch\n",
    "import scipy.sparse as sp\n",
    "\n",
    "DATA_ROOT = '/kaggle/data'\n",
    "PROCESSED_ROOT = os.path.join(DATA_ROOT, 'Processed')\n",
    "LOCAL_DATA = '/kaggle/working/STAIR-Enhanced/data'\n",
    "LOCAL_PROCESSED = os.path.join(LOCAL_DATA, 'Processed')\n",
    "\n",
    "for d in [DATA_ROOT, PROCESSED_ROOT, LOCAL_DATA, LOCAL_PROCESSED]:\n",
    "    os.makedirs(d, exist_ok=True)\n",
    "\n",
    "TARGET_FOLDER = 'tiktok'\n",
    "REQUIRED_PROCESSED_FILES = [\n",
    "    'train.txt', 'valid.txt', 'test.txt',\n",
    "    'visual_modality.pkl', 'textual_modality.pkl', 'audio_modality.pkl'\n",
    "]\n",
    "RAW_FILES = [\n",
    "    'trnMat.pkl', 'valMat.pkl', 'tstMat.pkl',\n",
    "    'audio_feat.npy', 'image_feat.npy', 'text_feat.npy'\n",
    "]\n",
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
    "def convert_raw_tiktok(src_dir, dst_dir):\n",
    "    print(f\"🔄 Đang thực hiện chuyển đổi TikTok dataset từ {src_dir} sang chuẩn FreeRec/STAIR ({dst_dir})...\")\n",
    "    os.makedirs(dst_dir, exist_ok=True)\n",
    "    \n",
    "    for rf in RAW_FILES:\n",
    "        src_path = os.path.join(src_dir, rf)\n",
    "        dst_path = os.path.join(dst_dir, rf)\n",
    "        if os.path.exists(src_path) and not os.path.exists(dst_path):\n",
    "            shutil.copy2(src_path, dst_path)\n",
    "\n",
    "    trn_mat = pickle.load(open(os.path.join(src_dir, \"trnMat.pkl\"), \"rb\"))\n",
    "    val_mat = pickle.load(open(os.path.join(src_dir, \"valMat.pkl\"), \"rb\"))\n",
    "    tst_mat = pickle.load(open(os.path.join(src_dir, \"tstMat.pkl\"), \"rb\"))\n",
    "\n",
    "    num_users, num_items = trn_mat.shape\n",
    "    print(f\"  • Kích thước Ma trận tương tác: {num_users} Users x {num_items} Items\")\n",
    "    print(f\"  • Train NNZ: {trn_mat.nnz:,} | Valid NNZ: {val_mat.nnz:,} | Test NNZ: {tst_mat.nnz:,}\")\n",
    "\n",
    "    splits = [\n",
    "        (\"train.txt\", trn_mat),\n",
    "        (\"valid.txt\", val_mat),\n",
    "        (\"test.txt\", tst_mat)\n",
    "    ]\n",
    "\n",
    "    for fname, mat in splits:\n",
    "        out_path = os.path.join(dst_dir, fname)\n",
    "        if not hasattr(mat, 'row'):\n",
    "            mat = mat.tocoo()\n",
    "        rows = mat.row\n",
    "        cols = mat.col\n",
    "        order = np.lexsort((cols, rows))\n",
    "        sorted_rows = rows[order]\n",
    "        sorted_cols = cols[order]\n",
    "\n",
    "        with open(out_path, \"w\", encoding=\"utf-8\") as f:\n",
    "            f.write(\"USER\\tITEM\\tTIMESTAMP\\n\")\n",
    "            for u, i in zip(sorted_rows, sorted_cols):\n",
    "                f.write(f\"{u}\\t{i}\\t0\\n\")\n",
    "        print(f\"  [Đã sinh] {fname}: {len(sorted_rows):,} tương tác -> {out_path}\")\n",
    "\n",
    "    audio_feat = np.load(os.path.join(src_dir, \"audio_feat.npy\"))\n",
    "    image_feat = np.load(os.path.join(src_dir, \"image_feat.npy\"))\n",
    "    text_feat  = np.load(os.path.join(src_dir, \"text_feat.npy\"))\n",
    "\n",
    "    print(f\"  • Multimodal feature shapes:\")\n",
    "    print(f\"    - Audio: {audio_feat.shape} ({audio_feat.dtype})\")\n",
    "    print(f\"    - Image: {image_feat.shape} ({image_feat.dtype})\")\n",
    "    print(f\"    - Text : {text_feat.shape} ({text_feat.dtype})\")\n",
    "\n",
    "    audio_tensor = torch.from_numpy(audio_feat.astype(np.float32))\n",
    "    image_tensor = torch.from_numpy(image_feat.astype(np.float32))\n",
    "    text_tensor  = torch.from_numpy(text_feat.astype(np.float32))\n",
    "\n",
    "    concat_np = np.concatenate([audio_feat.astype(np.float32), image_feat.astype(np.float32), text_feat.astype(np.float32)], axis=1)\n",
    "    concat_tensor = torch.from_numpy(concat_np)\n",
    "\n",
    "    modality_maps = {\n",
    "        \"visual_modality.pkl\": image_tensor,\n",
    "        \"textual_modality.pkl\": text_tensor,\n",
    "        \"audio_modality.pkl\": audio_tensor,\n",
    "        \"multimodal_concat.pkl\": concat_tensor\n",
    "    }\n",
    "\n",
    "    for pkl_name, tensor_data in modality_maps.items():\n",
    "        out_path = os.path.join(dst_dir, pkl_name)\n",
    "        with open(out_path, \"wb\") as f:\n",
    "            pickle.dump(tensor_data, f, protocol=pickle.HIGHEST_PROTOCOL)\n",
    "        size_mb = os.path.getsize(out_path) / (1024 * 1024)\n",
    "        print(f\"  [Đã sinh] {pkl_name}: {tensor_data.shape} ({size_mb:.2f} MB)\")\n",
    "\n",
    "    print(f\"✅ Hoàn tất chuyển đổi TikTok dataset sang chuẩn STAIR/FreeRec!\")\n",
    "\n",
    "def scan_and_prepare_tiktok():\n",
    "    target_processed = os.path.join(PROCESSED_ROOT, TARGET_FOLDER)\n",
    "    \n",
    "    # 1. Kiểm tra nếu đã có sẵn đầy đủ các tệp processed\n",
    "    for cand in [target_processed, os.path.join(DATA_ROOT, TARGET_FOLDER), os.path.join(LOCAL_PROCESSED, TARGET_FOLDER)]:\n",
    "        if os.path.exists(cand) and all(os.path.exists(os.path.join(cand, f)) for f in REQUIRED_PROCESSED_FILES):\n",
    "            print(f\"✅ Dữ liệu TikTok chuẩn FreeRec đã tồn tại sẵn tại: {cand}\")\n",
    "            bridge_directories(cand, TARGET_FOLDER)\n",
    "            return {'tiktok': target_processed}\n",
    "\n",
    "    # 2. Tìm kiếm trong /kaggle/input\n",
    "    print(\"🔍 Đang quét /kaggle/input để tìm kiếm tập dữ liệu TikTok...\")\n",
    "    input_base = '/kaggle/input'\n",
    "    raw_src = None\n",
    "    \n",
    "    for attempt in range(12):\n",
    "        if os.path.exists(input_base):\n",
    "            for root, dirs, files in os.walk(input_base):\n",
    "                if all(rf in files for rf in REQUIRED_PROCESSED_FILES):\n",
    "                    print(f\"  [TÌM THẤY DỮ LIỆU ĐÃ XỬ LÝ] {root}\")\n",
    "                    os.makedirs(target_processed, exist_ok=True)\n",
    "                    for f in files:\n",
    "                        shutil.copy2(os.path.join(root, f), os.path.join(target_processed, f))\n",
    "                    bridge_directories(target_processed, TARGET_FOLDER)\n",
    "                    return {'tiktok': target_processed}\n",
    "\n",
    "                if all(rf in files for rf in RAW_FILES):\n",
    "                    raw_src = root\n",
    "                    print(f\"  [TÌM THẤY DỮ LIỆU RAW TIKTOK] {root}\")\n",
    "                    break\n",
    "            if raw_src:\n",
    "                break\n",
    "\n",
    "        # Kiểm tra dự phòng từ repo tiktok.zip\n",
    "        repo_zip = os.path.join(LOCAL_DATA, 'tiktok.zip')\n",
    "        if not raw_src and os.path.exists(repo_zip):\n",
    "            print(f\"📦 Phát hiện tệp nén {repo_zip} trong repository. Đang giải nén...\")\n",
    "            extract_dir = os.path.join(LOCAL_DATA, 'raw_tiktok')\n",
    "            with zipfile.ZipFile(repo_zip, 'r') as zf:\n",
    "                zf.extractall(extract_dir)\n",
    "            for root, dirs, files in os.walk(extract_dir):\n",
    "                if all(rf in files for rf in RAW_FILES):\n",
    "                    raw_src = root\n",
    "                    break\n",
    "\n",
    "        if raw_src:\n",
    "            break\n",
    "        print(f\"⏳ Dataset đang được tải về từ Kaggle... Đợi 5s (lần {attempt+1}/12)...\")\n",
    "        time.sleep(5)\n",
    "\n",
    "    if raw_src:\n",
    "        convert_raw_tiktok(raw_src, target_processed)\n",
    "        bridge_directories(target_processed, TARGET_FOLDER)\n",
    "        return {'tiktok': target_processed}\n",
    "    else:\n",
    "        print(\"❌ Không tìm thấy tập dữ liệu TikTok trong /kaggle/input hoặc trong repository!\")\n",
    "        return {}\n",
    "\n",
    "prepared_data = scan_and_prepare_tiktok()\n",
    "print(\"=\" * 75)\n",
    "p_dir = os.path.join(PROCESSED_ROOT, TARGET_FOLDER)\n",
    "ready_files = [f for f in REQUIRED_PROCESSED_FILES if os.path.exists(os.path.join(p_dir, f))]\n",
    "status = f\"✅ {len(ready_files)}/{len(REQUIRED_PROCESSED_FILES)} tệp chuẩn sẵn sàng\" if len(ready_files) == len(REQUIRED_PROCESSED_FILES) else \"❌ THIẾU\"\n",
    "print(f\"TỔNG KẾT DỮ LIỆU TIKTOK: {status} trong {p_dir}\")\n",
    "for rf in REQUIRED_PROCESSED_FILES:\n",
    "    fp = os.path.join(p_dir, rf)\n",
    "    if os.path.exists(fp):\n",
    "        sz = os.path.getsize(fp) / (1024 * 1024)\n",
    "        print(f\"  • {rf:25s}: {sz:.2f} MB\")\n",
    "print(\"=\" * 75)\n"
]
nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell2_code
})

# -------------------------------------------------------------
# Cell 3: Runner Engine & Metrics Extractor
# -------------------------------------------------------------
nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Cell 3 🛠️ Runner Engine: Huấn luyện STAIR Baseline & Trích xuất 4 Chỉ số Khoa học\n",
        "Xây dựng hàm thực thi huấn luyện STAIR Baseline nguyên bản qua `main.py`:\n",
        "- Giám sát bộ nhớ VRAM định kỳ (`pynvml`).\n",
        "- Luồng output trực tiếp thời gian thực ra màn hình và lưu toàn bộ log ra `/kaggle/working/logs/tiktok_baseline.log`.\n",
        "- Tự động trích xuất các chỉ số tại Checkpoint tốt nhất: **Recall@10, Recall@20, NDCG@10, NDCG@20**."
    ]
})

cell3_code = [
    "# Cell 3: Runner Engine cho STAIR Baseline\n",
    "import subprocess, sys, os, time, re, threading\n",
    "\n",
    "try:\n",
    "    import pynvml\n",
    "    pynvml.nvmlInit()\n",
    "    HAS_NVML = True\n",
    "except Exception:\n",
    "    HAS_NVML = False\n",
    "\n",
    "def get_gpu_memory_used():\n",
    "    if not HAS_NVML:\n",
    "        return 0.0\n",
    "    try:\n",
    "        handle = pynvml.nvmlDeviceGetHandleByIndex(0)\n",
    "        info = pynvml.nvmlDeviceGetMemoryInfo(handle)\n",
    "        return info.used / (1024 ** 2)\n",
    "    except Exception:\n",
    "        return 0.0\n",
    "\n",
    "TRACKED_METRICS = ['Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20']\n",
    "\n",
    "def extract_baseline_metrics(log_path):\n",
    "    if not os.path.exists(log_path):\n",
    "        return None, {}\n",
    "\n",
    "    best_epoch = None\n",
    "    best_metrics = {}\n",
    "    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:\n",
    "        lines = f.readlines()\n",
    "\n",
    "    # 1. Tìm best epoch từ dòng 'Load best model @Epoch'\n",
    "    for line in lines:\n",
    "        m = re.search(r'Load best model @Epoch\\s+(\\d+)', line)\n",
    "        if m:\n",
    "            best_epoch = int(m.group(1))\n",
    "\n",
    "    # 2. Tìm phần TEST tại best epoch ở cuối log\n",
    "    for line in reversed(lines):\n",
    "        if 'TEST' in line and any(k in line for k in ['NDCG', 'RECALL']):\n",
    "            for metric in TRACKED_METRICS:\n",
    "                pattern = rf'{re.escape(metric)}[\\sAvg:]+([0-9.]+)'\n",
    "                m = re.search(pattern, line, re.IGNORECASE)\n",
    "                if m and metric not in best_metrics:\n",
    "                    best_metrics[metric] = float(m.group(1))\n",
    "            if len(best_metrics) >= len(TRACKED_METRICS):\n",
    "                break\n",
    "\n",
    "    return best_epoch, best_metrics\n",
    "\n",
    "def run_stair_baseline(yaml_cfg, data_root, log_path):\n",
    "    print('=' * 85)\n",
    "    print('🚀 BẮT ĐẦU HUẤN LUYỆN: STAIR BASELINE (AAAI 2025) TRÊN TẬP TIKTOK')\n",
    "    print(f'  * Model Architecture : STAIR Baseline (Original)')\n",
    "    print(f'  * Config File        : {yaml_cfg}')\n",
    "    print(f'  * Data Root          : {data_root}')\n",
    "    print(f'  * Log Path           : {log_path}')\n",
    "    print('=' * 85)\n",
    "\n",
    "    os.makedirs(os.path.dirname(log_path), exist_ok=True)\n",
    "    t0 = time.time()\n",
    "    runner_py = '/kaggle/working/STAIR-Enhanced/main.py'\n",
    "    if not os.path.exists(runner_py):\n",
    "        runner_py = 'main.py'\n",
    "\n",
    "    cmd = [\n",
    "        sys.executable, runner_py,\n",
    "        '--config', yaml_cfg,\n",
    "        '--root',   data_root,\n",
    "    ]\n",
    "\n",
    "    with open(log_path, 'w', encoding='utf-8') as f:\n",
    "        proc = subprocess.Popen(\n",
    "            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n",
    "            text=True, bufsize=1, universal_newlines=True\n",
    "        )\n",
    "        for line in proc.stdout:\n",
    "            sys.stdout.write(line)\n",
    "            sys.stdout.flush()\n",
    "            f.write(line)\n",
    "            f.flush()\n",
    "        proc.wait()\n",
    "\n",
    "    elapsed = time.time() - t0\n",
    "    print('=' * 85)\n",
    "    if proc.returncode != 0:\n",
    "        print(f'❌ [THẤT BẠI] Quá trình huấn luyện STAIR Baseline gặp lỗi (Exit Code: {proc.returncode})!')\n",
    "    else:\n",
    "        print(f'✅ [HOÀN TẤT] Huấn luyện STAIR Baseline hoàn thành trong {elapsed/60:.2f} phút ({elapsed:.1f}s)!')\n",
    "\n",
    "    best_ep, metrics = extract_baseline_metrics(log_path)\n",
    "    print(f'  * Checkpoint tối ưu : Epoch {best_ep}')\n",
    "    for m, val in metrics.items():\n",
    "        print(f'  * {m:12s}: {val:.4f}')\n",
    "    print('=' * 85)\n"
]
nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell3_code
})

# -------------------------------------------------------------
# Cell 4: Config Preview
# -------------------------------------------------------------
nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Cell 4 📋 Cấu hình Siêu tham số STAIR Baseline cho TikTok\n",
        "Tệp cấu hình chính thức `configs/tiktok_MMRec.yaml`:\n",
        "- `embedding_dim`: 64 (Chuẩn baseline paper STAIR AAAI 2025).\n",
        "- `batch_size`: 1024 | `epochs`: 500 | `eval_freq`: 5 eps.\n",
        "- `gamma`: 0.05 (Hệ số suy giảm phổ FSC cho video ngắn).\n",
        "- `mfiles`: visual, textual, audio | `num_neighbors`: 3-3-3.\n",
        "- `optimizer`: adamwsevo ($lr = 10^{-3}$, weight decay $= 0.1$)."
    ]
})

cell4_code = [
    "# Cell 4: Xác nhận cấu hình STAIR Baseline TikTok\n",
    "import yaml\n",
    "\n",
    "yaml_path = '/kaggle/working/STAIR-Enhanced/configs/tiktok_MMRec.yaml'\n",
    "with open(yaml_path, 'r', encoding='utf-8') as f:\n",
    "    cfg_dict = yaml.safe_load(f)\n",
    "\n",
    "print('=' * 75)\n",
    "print('THIẾT LẬP THAM SỐ CHUẨN TẮC CHO STAIR BASELINE TIKTOK:')\n",
    "for k, v in cfg_dict.items():\n",
    "    print(f'  • {k:18s}: {v}')\n",
    "print('=' * 75)\n"
]
nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell4_code
})

# -------------------------------------------------------------
# Cell 5: Run STAIR Baseline Training
# -------------------------------------------------------------
nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Cell 5 🏋️ Thực thi Huấn luyện STAIR Baseline trên TikTok\n",
        "Khởi chạy huấn luyện mô hình STAIR Baseline nguyên bản.\n",
        "Tiến trình huấn luyện sẽ hiển thị trực tiếp và lưu vào `/kaggle/working/logs/tiktok_baseline.log`."
    ]
})

cell5_code = [
    "# Cell 5: Huấn luyện STAIR Baseline trên TikTok\n",
    "DATA_ROOT = '/kaggle/data'\n",
    "YAML_CFG = '/kaggle/working/STAIR-Enhanced/configs/tiktok_MMRec.yaml'\n",
    "LOG_PATH = '/kaggle/working/logs/tiktok_baseline.log'\n",
    "\n",
    "if 'tiktok' in prepared_data or (os.path.exists('/kaggle/data/Processed/tiktok') and len(os.listdir('/kaggle/data/Processed/tiktok')) >= 6):\n",
    "    run_stair_baseline(\n",
    "        yaml_cfg=YAML_CFG,\n",
    "        data_root=DATA_ROOT,\n",
    "        log_path=LOG_PATH,\n",
    "    )\n",
    "else:\n",
    "    print('⚠️ Bỏ qua huấn luyện do thiếu dữ liệu TikTok trong /kaggle/input.')\n"
]
nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell5_code
})

# -------------------------------------------------------------
# Cell 6: PrettyTable Summary & Comparison
# -------------------------------------------------------------
nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Cell 6 📊 Bảng Đối Soát Kết Quả với Log Mốc Ngày 14/09/2026\n",
        "Bảng tổng hợp đối sánh giữa kết quả vừa chạy và mốc chuẩn trong `logs/paper/tiktok.log` (Epoch 220) trên cả 4 chỉ số khoa học: **Recall@10, Recall@20, NDCG@10, NDCG@20**."
    ]
})

cell6_code = [
    "# Cell 6: Bảng đối soát kết quả STAIR Baseline TikTok\n",
    "from prettytable import PrettyTable\n",
    "\n",
    "# Mốc chuẩn chính thức trong logs/paper/tiktok.log (Best @Epoch 220)\n",
    "HISTORICAL_BASELINE = {\n",
    "    'epoch': 220,\n",
    "    'Recall@10': 0.0558,\n",
    "    'Recall@20': 0.0799,\n",
    "    'NDCG@10':   0.0292,\n",
    "    'NDCG@20':   0.0352,\n",
    "}\n",
    "\n",
    "log_file = '/kaggle/working/logs/tiktok_baseline.log'\n",
    "best_ep, current_metrics = extract_baseline_metrics(log_file)\n",
    "\n",
    "table = PrettyTable()\n",
    "table.field_names = ['Lần chạy', 'Checkpoint', 'Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20', 'Trạng thái']\n",
    "\n",
    "hb = HISTORICAL_BASELINE\n",
    "table.add_row([\n",
    "    'Log Mốc (14/09)', f'Epoch {hb[\"epoch\"]}',\n",
    "    f'{hb[\"Recall@10\"]:.4f}', f'{hb[\"Recall@20\"]:.4f}',\n",
    "    f'{hb[\"NDCG@10\"]:.4f}', f'{hb[\"NDCG@20\"]:.4f}',\n",
    "    'Mốc chuẩn paper'\n",
    "])\n",
    "\n",
    "if current_metrics and len(current_metrics) >= 4:\n",
    "    cm = current_metrics\n",
    "    diff_n20 = cm['NDCG@20'] - hb['NDCG@20']\n",
    "    status_str = f'Khớp chuẩn (Δ={diff_n20:+.4f})' if abs(diff_n20) < 0.002 else f'Chênh lệch (Δ={diff_n20:+.4f})'\n",
    "    table.add_row([\n",
    "        'Tái lập hiện tại', f'Epoch {best_ep}',\n",
    "        f'{cm[\"Recall@10\"]:.4f}', f'{cm[\"Recall@20\"]:.4f}',\n",
    "        f'{cm[\"NDCG@10\"]:.4f}', f'{cm[\"NDCG@20\"]:.4f}',\n",
    "        status_str\n",
    "    ])\n",
    "else:\n",
    "    table.add_row(['Tái lập hiện tại', 'Chưa hoàn tất', 'N/A', 'N/A', 'N/A', 'N/A', 'Đang chờ log'])\n",
    "\n",
    "print('=' * 85)\n",
    "print('BẢNG TỔNG KẾT ĐỐI SOÁT STAIR BASELINE TRÊN TẬP TIKTOK:')\n",
    "print(table)\n",
    "print('=' * 85)\n"
]
nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell6_code
})

out_path = r"d:\STAIR-Enhanced\notebook\stair_tiktok_baseline.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Successfully generated STAIR Baseline notebook for TikTok: {out_path}")
