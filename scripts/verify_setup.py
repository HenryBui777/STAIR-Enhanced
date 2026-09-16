# -*- coding: utf-8 -*-
import json
import re

# 1. Check YAML configs
for yml in ["configs/Amazon2014Baby_550_MMRec.yaml", "configs/Amazon2014Sports_550_MMRec.yaml"]:
    with open(yml, "r", encoding="utf-8") as f:
        content = f.read()
    assert re.search(r"epochs:\s*500", content), f"Epochs not 500 in {yml}"
    assert "which4best: NDCG@20" in content, f"which4best not NDCG@20 in {yml}"
    print(f"[OK] YAML verified: {yml}")

# 2. Check main_stair_ne_nlgcl_v5_plus.py
with open("main_stair_ne_nlgcl_v5_plus.py", "r", encoding="utf-8") as f:
    code = f.read()
assert "epochs=500" in code, "epochs=500 missing in main_stair_ne_nlgcl_v5_plus.py"
assert "best_composite" not in code, "best_composite found in main_stair_ne_nlgcl_v5_plus.py"
assert "best_ndcg20" in code, "best_ndcg20 missing in main_stair_ne_nlgcl_v5_plus.py"
print("[OK] main_stair_ne_nlgcl_v5_plus.py verified")

# 3. Check main_stair_breakthrough.py
with open("main_stair_breakthrough.py", "r", encoding="utf-8") as f:
    code_bt = f.read()
assert "epochs=500" in code_bt, "epochs=500 missing in main_stair_breakthrough.py"
assert "best_composite" not in code_bt, "best_composite found in main_stair_breakthrough.py"
assert "best_ndcg20" in code_bt, "best_ndcg20 missing in main_stair_breakthrough.py"
print("[OK] main_stair_breakthrough.py verified")

# 4. Check notebook
with open("notebook/P3/stair_ne_nlgcl_v5_plus.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

assert len(nb["cells"]) == 15, f"Expected 15 cells, got {len(nb['cells'])}"

all_src = "".join("".join(c["source"]) for c in nb["cells"])
assert "Composite Metric (R10+R20+N10+N20)/4" not in all_src, "Old composite metric string found in notebook"
assert "Điểm Tổng Hợp" not in all_src, "Old 'Điểm Tổng Hợp' found in notebook"
assert "best_ndcg20" in all_src, "best_ndcg20 missing in notebook embedded code"
assert "Δ vs v5 N@20" in all_src, "Δ vs v5 N@20 missing in notebook table"

print(f"[OK] Notebook verified: 15 cells, epochs=500, NDCG@20 target metric")
print("ALL CHECKS PASSED!")
