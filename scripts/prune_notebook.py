# -*- coding: utf-8 -*-
import json

notebook_path = r"d:\STAIR-Enhanced\notebook\P3\stair_ne_nlgcl_v5_plus.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Keep only:
# 0: Header
# 1, 2: Cell 1 (Setup)
# 3, 4: Cell 2 (Data Bridge)
# 5, 6: Cell 3 (Unit Tests)
# 7, 8: Cell 4 (Runner)
# 9, 10: Cell 5 (Hyperparameters)
# 11, 12: Cell 6 (Train Baby & Sports)
# 17, 18: Summary Table (PrettyTable)

kept_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18]
pruned_cells = [nb["cells"][i] for i in kept_indices]

# Renumber Cell 17/18 markdown header to "## Cell 7 📊 Bảng Tổng Kết Kết Quả Thực Nghiệm"
pruned_cells[13]["source"] = [
    "## Cell 7 📊 Bảng Tổng Kết Kết Quả Thực Nghiệm (Baby & Sports)\n",
    "Bảng tổng hợp đối sánh 4 chỉ số khoa học: **Recall@10, Recall@20, NDCG@10, NDCG@20** và **Điểm Tổng Hợp** giữa Baseline, SOTA v5 và Phương pháp cải tiến mới."
]

nb["cells"] = pruned_cells

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully pruned notebook! Total cells now:", len(nb["cells"]))
