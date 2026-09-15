# TỔNG HỢP KIẾN TRÚC & HƯỚNG DẪN CẢI TIẾN: STAIR-NE-NLGCL v5+ (STAIR3-v3 REFINED)

> **Mục đích tài liệu:** Hướng dẫn kỹ thuật và tóm tắt bản chất cải tiến của mô hình Giai đoạn 3 dành cho việc tiếp tục tối ưu hóa, làm thực nghiệm và viết bài báo khoa học (Paper).

---

## 1. ĐỊNH VỊ TÊN GỌI & TỆP TIN DỰ ÁN

Trong quá trình nghiên cứu, có sự khác biệt giữa tên gọi trong tài liệu/trao đổi và tên tệp mã nguồn:




---

## 4. CÁC HƯỚNG CẢI TIẾN TIẾP THEO ĐỂ VIẾT BÀI BÁO (PAPER EXTENSIONS)

Để tối ưu thêm nữa trên nền tảng `v5+` nhằm gia tăng đóng góp học thuật cho bài báo, có 4 hướng khả thi cao:

### Hướng 1: Degree-Aware / Sparsity-Adaptive HANS (Điều phối mẫu âm theo bậc của node)
* **Vấn đề:** Hiện tại $\gamma_h = 0.15$ đang cố định cho toàn bộ nodes. Với node có bậc cao (popular items), mẫu âm dễ bị phạt quá tay; với node đuôi dài (cold-start / long-tail items), lực phạt chưa đủ mạnh.
* **Đề xuất:** Cho $\gamma_h$ phụ thuộc vào bậc $d_i$ của item:
  $$\gamma_h(i) = \gamma_0 \cdot \left( \frac{\log(1 + d_{\max}) - \log(1 + d_i)}{\log(1 + d_{\max})} \right)$$
  Giúp tăng lực đẩy mẫu âm cho các mặt hàng ít tương tác, cải thiện mạnh Recall trên tail items.

### Hướng 2: Modal-Consistency Guided Dynamic Margin (Lề tương phản tương thích ngữ nghĩa)
* **Vấn đề:** Hard MFNA dùng ngưỡng cắt cứng $0.85$. Các cặp có độ tương đồng từ $0.6 \sim 0.84$ vẫn chịu chung một mức phạt với các cặp có độ tương đồng $0.0$.
* **Đề xuất:** Áp dụng dynamic margin dựa trên khoảng cách ngữ nghĩa đa phương thức:
  $$\text{sim}_{\text{adjusted}}(u, i^-) = \text{sim}(u, i^-) - \alpha \cdot \max(0, S_{\text{modal}}(i^+, i^-) - \tau_{\text{base}})$$
  Ngăn việc phạt quá nặng các sản phẩm có tính bổ trợ hoặc có cùng phong cách.

### Hướng 3: Asymmetric Bi-directional Temperature ($\tau_u \ne \tau_i$)
* **Vấn đề:** Chiều User $\to$ Item và Item $\to$ User đang dùng chung nhiệt độ $\tau = 0.20$. Tuy nhiên, số lượng items thường lớn hơn users và có phân bố embedding thưa hơn.
* **Đề xuất:** Tách biệt nhiệt độ hiệu dụng theo hai chiều: $\tau_{u2i} = 0.20$, $\tau_{i2u} = 0.15$ để kiểm soát độ sắc nét của phân phối softmax phù hợp với từng không gian.

### Hướng 4: Tích hợp Modern Vision/Text Encoders (CLIP Features)
* **Vấn đề:** Đặc trưng gốc hiện tại được trích xuất từ các bộ trích xuất cổ điển (ResNet / Sentence-BERT cũ).
* **Đề xuất:** Thử nghiệm thay thế ma trận modal thô bằng embedding chuẩn hóa từ CLIP ViT-B/32, kết hợp chiếu SVD Whitening như đã chứng minh ở Section 3.3 của `STAIR3_v5_Report.md`.

---

## 5. HƯỚNG DẪN THỰC THI (QUICK RUN GUIDE)

### Chạy kiểm thử trên máy cục bộ:
```bash
# Huấn luyện trên Amazon Sports
python main_stair_ne_nlgcl_v5_plus.py --config configs/Amazon2014Sports_550_MMRec.yaml

# Huấn luyện trên Amazon Baby
python main_stair_ne_nlgcl_v5_plus.py --config configs/Amazon2014Baby_550_MMRec.yaml

# Huấn luyện trên Amazon Electronics
python main_stair_ne_nlgcl_v5_plus.py --config configs/Amazon2014Electronics_550_MMRec.yaml
```

### Chạy trên Kaggle:
1. Mở notebook: [`notebook/P3/stair_ne_nlgcl_v3_plus.ipynb`](file:///d:/STAIR-Enhanced/notebook/P3/stair_ne_nlgcl_v3_plus.ipynb).
2. Thêm input dataset: `stair-datasets-mmrec` (chứa các thư mục `Amazon2014Baby_550_MMRec`, `Amazon2014Sports_550_MMRec`, `Amazon2014Electronics_550_MMRec`).
3. Chạy các cell huấn luyện tương ứng với hàm `run_training_v5_plus`.
