# -*- coding: utf-8 -*-
import json

notebook_path = r"d:\STAIR-Enhanced\notebook\P3\stair_ne_nlgcl_v5_plus.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Set Cell 13 (Markdown)
nb["cells"][13]["source"] = [
    "## Cell 7 📊 Bảng Tổng Kết Kết Quả Thực Nghiệm (Baby & Sports)\n",
    "Bảng tổng hợp đối sánh 4 chỉ số khoa học: **Recall@10, Recall@20, NDCG@10, NDCG@20** và **Điểm Tổng Hợp** giữa Baseline, SOTA v5 và Phương pháp cải tiến mới."
]

# Set Cell 14 (Code - PrettyTable)
nb["cells"][14]["source"] = [
    "# Cell 7: Bảng tổng kết kết quả thực nghiệm Baby & Sports\n",
    "import os, glob\n",
    "from prettytable import PrettyTable\n",
    "\n",
    "BASELINE = {\n",
    "    'baby':   {'Recall@10': 0.0674, 'Recall@20': 0.1042, 'NDCG@10': 0.0359, 'NDCG@20': 0.0454},\n",
    "    'sports': {'Recall@10': 0.0743, 'Recall@20': 0.1111, 'NDCG@10': 0.0405, 'NDCG@20': 0.0500},\n",
    "}\n",
    "V5_RESULTS = {\n",
    "    'baby':   {'Recall@10': 0.0669, 'Recall@20': 0.1027, 'NDCG@10': 0.0362, 'NDCG@20': 0.0454},\n",
    "    'sports': {'Recall@10': 0.0753, 'Recall@20': 0.1113, 'NDCG@10': 0.0415, 'NDCG@20': 0.0508},\n",
    "}\n",
    "\n",
    "table = PrettyTable()\n",
    "table.field_names = ['Tập dữ liệu', 'Phương pháp', 'Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20', 'Điểm Tổng Hợp', 'Δ vs v5 R@20']\n",
    "\n",
    "for ds in ['baby', 'sports']:\n",
    "    # Mốc chuẩn Baseline\n",
    "    bl = BASELINE[ds]\n",
    "    bl_score = (bl['Recall@10'] + bl['Recall@20'] + bl['NDCG@10'] + bl['NDCG@20']) / 4.0\n",
    "    table.add_row([ds.upper(), 'STAIR Baseline (64D)', f\"{bl['Recall@10']:.4f}\", f\"{bl['Recall@20']:.4f}\", f\"{bl['NDCG@10']:.4f}\", f\"{bl['NDCG@20']:.4f}\", f\"{bl_score:.4f}\", '-'])\n",
    "    \n",
    "    # Mốc chuẩn v5\n",
    "    v5 = V5_RESULTS[ds]\n",
    "    v5_score = (v5['Recall@10'] + v5['Recall@20'] + v5['NDCG@10'] + v5['NDCG@20']) / 4.0\n",
    "    table.add_row([ds.upper(), 'STAIR v5 SOTA (64D)', f\"{v5['Recall@10']:.4f}\", f\"{v5['Recall@20']:.4f}\", f\"{v5['NDCG@10']:.4f}\", f\"{v5['NDCG@20']:.4f}\", f\"{v5_score:.4f}\", '0.00%'])\n",
    "    \n",
    "    # Các log thực nghiệm mới trong logs/breakthrough\n",
    "    log_files = glob.glob(f\"/kaggle/working/logs/breakthrough/{ds}_*.log\")\n",
    "    for lf in log_files:\n",
    "        ep, m = extract_best_test(lf)\n",
    "        if m and len(m) >= 4:\n",
    "            tag = os.path.basename(lf).replace(f\"{ds}_\", \"\").replace(\".log\", \"\")\n",
    "            score = (m['Recall@10'] + m['Recall@20'] + m['NDCG@10'] + m['NDCG@20']) / 4.0\n",
    "            gain = (m['Recall@20'] - v5['Recall@20']) / v5['Recall@20'] * 100\n",
    "            sign = '+' if gain >= 0 else ''\n",
    "            table.add_row([ds.upper(), f\"★ {tag} (@Ep{ep})\", f\"{m['Recall@10']:.4f}\", f\"{m['Recall@20']:.4f}\", f\"{m['NDCG@10']:.4f}\", f\"{m['NDCG@20']:.4f}\", f\"{score:.4f}\", f\"{sign}{gain:.2f}%\"])\n",
    "\n",
    "print(table)\n"
]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully updated PrettyTable in Cell 13 & 14!")
