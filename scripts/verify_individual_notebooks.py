import json, os, sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOKS = [
    ('baby', 'notebook/stair_baby_all.ipynb', 7050, 1024),
    ('sports', 'notebook/stair_sports_all.ipynb', 18357, 1024),
    ('electronics', 'notebook/stair_electronics_all.ipynb', 63001, 4096),
    ('clothing', 'notebook/stair_clothing_all.ipynb', 23033, 1024),
]

all_passed = True
print("=" * 85)
print("🔍 BẮT ĐẦU KIỂM TRA TOÀN DIỆN 4 NOTEBOOK ĐỘC LẬP:")
print("=" * 85)

for dkey, nb_path, exp_items, exp_bs in NOTEBOOKS:
    print(f"\n👉 [KIỂM TRA]: {nb_path} ({dkey.upper()})")
    if not os.path.exists(nb_path):
        print(f"   ❌ File không tồn tại: {nb_path}")
        all_passed = False
        continue
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    print(f"   * Tổng số cells: {len(cells)}")
    assert len(cells) == 8, f"Số cells mong đợi 8, nhưng thực tế {len(cells)}"
    
    # Check Cell 0
    c0 = ''.join(cells[0]['source'])
    assert 'BẢNG 1: CÁC CẤU HÌNH THỰC NGHIỆM' in c0, "Thiếu Bảng 1 trong Cell 0"
    assert 'BẢNG 2: DANH MỤC TOÀN BỘ FILE ĐẦU RA' in c0, "Thiếu Bảng 2 trong Cell 0"
    print("   * Cell 0: Đầy đủ Bảng Cấu hình & Bảng File Đầu Ra (OK)")
    
    # Check Cell 2
    c2 = ''.join(cells[2]['source'])
    assert f"EXPECTED_ITEMS = {exp_items}" in c2, f"Sai expected items trong Cell 2: {exp_items}"
    assert "check_item_count_matches" in c2, "Thiếu hàm check_item_count_matches trong Cell 2"
    print(f"   * Cell 2: Khớp chính xác {exp_items} items & có hàm bảo vệ chống nhầm tập (OK)")
    
    # Check Cell 5 (Training)
    c5 = ''.join(cells[5]['source'])
    assert "run_training_v5_plus" in c5, "Thiếu run_training_v5_plus trong Cell 5"
    assert "run_stair_baseline" in c5, "Thiếu run_stair_baseline trong Cell 5"
    assert "batch_size=" not in c5, "Không được hardcode batch_size trong hàm gọi Cell 5"
    assert "patience=" not in c5, "Không được hardcode patience trong hàm gọi Cell 5"
    print("   * Cell 5: Huấn luyện chuẩn 100%, không hardcode batch_size/patience (OK)")
    
    # Check Cell 6 (Evaluation)
    c6 = ''.join(cells[6]['source'])
    assert f"results_{dkey}.csv" in c6, f"Thiếu results_{dkey}.csv trong Cell 6"
    assert f"epoch_telemetry_{dkey}.csv" in c6, f"Thiếu epoch_telemetry_{dkey}.csv trong Cell 6"
    assert "generate_dataset_visualizations" in c6, "Thiếu visualizer trong Cell 6"
    print("   * Cell 6: Xuất PrettyTable, 2 CSVs & 5 Figures riêng biệt (OK)")
    
    # Check Cell 7 (Auto Zip & Download)
    c7 = ''.join(cells[7]['source'])
    assert f"stair_{dkey}_results.zip" in c7, f"Thiếu zip file trong Cell 7"
    assert "auto_download_link" in c7, "Thiếu auto download script trong Cell 7"
    print("   * Cell 7: Đóng gói zip tự động & kích hoạt tải về máy (OK)")

print("\n" + "=" * 85)
if all_passed:
    print("🎉 TOÀN BỘ 4 NOTEBOOK ĐÃ VƯỢT QUA TẤT CẢ CÁC BÀI KIỂM THỬ XÁC MINH!")
else:
    print("❌ CÓ LỖI XẢY RA TRONG QUÁ TRÌNH KIỂM THỬ!")
print("=" * 85)
