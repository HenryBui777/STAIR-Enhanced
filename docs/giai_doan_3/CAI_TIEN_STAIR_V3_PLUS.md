# TỔNG HỢP KIẾN TRÚC & HƯỚNG DẪN CẢI TIẾN: STAIR-NE-NLGCL v5+ & BỘ 3 PHƯƠNG PHÁP ĐỘT PHÁ

> **Mục đích tài liệu:** Hướng dẫn kỹ thuật, rà soát mã nguồn và tóm tắt bản chất cải tiến của mô hình Giai đoạn 3 dành cho việc tiếp tục tối ưu hóa, làm thực nghiệm và viết bài báo khoa học (Paper).  
> **Tài liệu chi tiết chuyên sâu:** Tham khảo [`RA_SOAT_MA_NGUON_KIEN_TRUC_CAI_TIEN.md`](./RA_SOAT_MA_NGUON_KIEN_TRUC_CAI_TIEN.md).

---

## 1. ĐỊNH VỊ TÊN GỌI & TỆP TIN DỰ ÁN

Trong quá trình nghiên cứu, có sự khác biệt giữa tên gọi trong tài liệu/trao đổi và tên tệp mã nguồn:

| Tên gọi nghiên cứu | Tên tệp mã nguồn tương ứng | Vai trò trong hệ thống |
| :--- | :--- | :--- |
| **STAIR-NE-NLGCL v5+** | `models/stair_ne_nlgcl_v5_plus.py` | Lớp mô hình đối chứng cốt lõi (100% Direct Gradient Flow, Sign-preserving noise, Linear HANS, Hard MFNA). |
| **Method 1: DAN-TANS** | `models/stair_breakthrough.py` (`STAIR_DAN_TANS_Module`) | Mở rộng điều phối nhiễu theo bậc node và phạt âm thích ứng đuôi dài. |
| **Method 2: DCD-Gated** | `models/stair_breakthrough.py` (`STAIR_DCD_Gated_Module`) | **Đột phá SOTA (+14% NDCG@20):** Làm dày cạnh ảo đồng thuận kép (Ochiai x Modal) có cổng van Gating phi tuyến an toàn. |
| **Method 3: APPNP-CrossModal** | `models/stair_breakthrough.py` (`STAIR_APPNP_CrossModal_Module`) | Tích chập restart chống over-smoothing và căn chỉnh User <-> Modal feature. |
| **Runner Hệ Thống** | `main_stair_ne_nlgcl_v5_plus.py` | Backbone kết hợp, tích hợp **Chunked Zero-OOM Engine** và bộ điều phối huấn luyện. |
| **Notebook Electronics** | `notebook/stair_electronics.ipynb` | Notebook độc lập huấn luyện chuyên biệt trên tập dữ liệu lớn Amazon Electronics. |
| **Notebook P3 Tổng Hợp** | `notebook/P3/stair_ne_nlgcl_v5_plus.ipynb` | Notebook tổng hợp thực nghiệm trên Amazon Baby & Amazon Sports. |

---

## 2. BẢN CHẤT KIẾN TRÚC CỐT LÕI (5 TRỤ CỘT v5+)

1. **Direct Gradient Flow (No Projection Head):**
   - Loại bỏ hoàn toàn MLP Projection Head và Regularized Diagonal Spectral Projector.
   - 100% thông lượng gradient InfoNCE truyền thẳng vào bảng biểu diễn gốc $E_u, E_i$.
2. **True Sign-Preserving Spectral Perturbation ($|\eta| \ge 0$):**
   - $\tilde{\mathbf{h}} = \mathbf{h} + \epsilon \cdot \left(\boldsymbol{\beta} \odot \text{sign}(\mathbf{h}) \odot \frac{|\boldsymbol{\eta}|}{\||\boldsymbol{\eta}|\|_2}\right)$.
   - Bảo toàn 100% góc phần tư không gian, loại bỏ hoàn toàn hiện tượng đảo pha tọa độ.
3. **Linear HANS (Hardness-Aware Negative Scheduling):**
   - $\psi(s) = 1.0 + \gamma_h \cdot \max(0, s)$ với $\gamma_h = 0.15$.
   - Bảo toàn nhiệt độ hiệu dụng $\tau = 0.20$, không làm méo phân phối xác suất.
4. **Hard-Threshold MFNA (Modality False Negative Attenuation):**
   - Lọc bỏ triệt để các mẫu âm giả gần bản sao (near-duplicates) có $S_{\text{modal}} > 0.85$.
5. **Constant Contrastive Weight with Linear Warmup:**
   - $\lambda_{\text{cl}} = 0.010$ cố định suốt 500 epochs (Linear warmup trong 50 epochs đầu).

---

## 3. BỘ 3 PHƯƠNG PHÁP ĐỘT PHÁ MỞ RỘNG

### 3.1. DAN-TANS (Degree-Aware Noise & Topology-Aware Negative Scheduling)
* Head items (bậc cao) nhận biên độ nhiễu lớn hơn (lên tới $1.4\times$) để chống dồn ứ biểu diễn.
* Tail items (bậc thấp / cold-start) nhận hệ số phạt mẫu âm $\gamma_h$ lớn hơn (lên tới $2.0\times$) để định hình rõ nét ranh giới phân tách.

### 3.2. DCD-Gated (Dual-Consensus Denoising & Gated Residuals) — Kỷ Lục SOTA
* **Đồng thuận kép:** Cạnh ảo $S_{\text{conf}}$ chỉ được kết nối khi có sự đồng thuận giữa hành vi đồng mua (Ochiai) và độ tương đồng ngữ nghĩa đặc trưng (Modal Cosine).
* **Van Gating an toàn:** Cổng phi tuyến $\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{H}_1 \,\|\, S_{\text{conf}} \mathbf{H}_0])$ tự động đóng lại ($\mathbf{g} \to \mathbf{0}$) khi cạnh ảo tiềm ẩn nhiễu (như trên tập Baby), đảm bảo không bao giờ suy thoái dưới baseline.
* **Chunked Zero-OOM Engine:** Xử lý theo từng khối trượt `chunk_size = 2000` items, đưa mức tiêu thụ RAM đỉnh xuống dưới 5GB, triệt tiêu 100% lỗi tràn bộ nhớ / SIGKILL (-9) trên Amazon Electronics.

### 3.3. APPNP-CrossModal
* Tích chập restart với $\alpha_{\text{restart}} = 0.15$ giúp liên tục duy trì bản sắc đặc trưng gốc qua nhiều tầng tích chập sâu.
* Căn chỉnh trực tiếp User Embedding với Modal Feature của sản phẩm tích cực mà không nhồi nhét cạnh bẩn vào đồ thị.

---

## 4. CÁC HƯỚNG CẢI TIẾN TIẾP THEO ĐỂ VIẾT BÀI BÁO (PAPER EXTENSIONS)

Để tối ưu thêm nữa trên nền tảng `v5+` và `dcd_gated` nhằm gia tăng đóng góp học thuật cho bài báo:

### Hướng 1: Degree-Aware / Sparsity-Adaptive HANS (Đã triển khai trong DAN-TANS)
* Cho $\gamma_h$ phụ thuộc vào bậc $d_i$ của item:
  $$\gamma_h(i) = \gamma_0 \cdot \left(1.0 + \frac{\log(1 + d_{\max}) - \log(1 + d_i)}{\log(1 + d_{\max})}\right)$$
  Giúp tăng lực đẩy mẫu âm cho các mặt hàng ít tương tác, cải thiện mạnh Recall trên tail items.

### Hướng 2: Dual Consensus với Phân Cụm Ngữ Nghĩa Cấp Cụm (Cluster-level DCD)
* Kết hợp phân cụm k-Means trên không gian modal để tìm kiếm ứng viên đồng thuận ở cấp độ danh mục / chủ đề sản phẩm, mở rộng tầm nhìn của cạnh ảo ra ngoài phạm vi đồng mua trực tiếp.

### Hướng 3: Asymmetric Bi-directional Temperature ($\tau_u \ne \tau_i$)
* Chiều User $\to$ Item và Item $\to$ User có thể sử dụng nhiệt độ bất đối xứng: $\tau_{u2i} = 0.20$, $\tau_{i2u} = 0.15$ để kiểm soát độ sắc nét của phân phối softmax phù hợp với từng không gian biểu diễn.

### Hướng 4: Tích hợp Modern Vision/Text Encoders (CLIP Features)
* Thử nghiệm thay thế ma trận modal thô bằng embedding chuẩn hóa từ CLIP ViT-B/32, kết hợp chiếu SVD Whitening như đã chứng minh ở Section 3.3 của `STAIR3_v5_Report.md`.

---

## 5. HƯỚNG DẪN THỰC THI (QUICK RUN GUIDE)

### Chạy kiểm thử trên máy cục bộ:
```bash
# Huấn luyện DCD-Gated trên Amazon Sports
python main_stair_ne_nlgcl_v5_plus.py --config configs/Amazon2014Sports_550_MMRec.yaml --method dcd_gated

# Huấn luyện DCD-Gated trên Amazon Baby
python main_stair_ne_nlgcl_v5_plus.py --config configs/Amazon2014Baby_550_MMRec.yaml --method dcd_gated

# Huấn luyện DCD-Gated trên Amazon Electronics
python main_stair_ne_nlgcl_v5_plus.py --config configs/Amazon2014Electronics_550_MMRec.yaml --method dcd_gated
```

### Chạy trên Kaggle:
1. **Amazon Electronics (Khuyên dùng):**
   - Mở notebook: [`notebook/stair_electronics.ipynb`](file:///d:/STAIR-Enhanced/notebook/stair_electronics.ipynb).
   - Đã nhúng sẵn bản vá Chunked Zero-OOM Engine, tự động kết nối dataset `/kaggle/input/stair-datasets-mmrec`.
   - Nhấn **Save & Run All** hoặc chạy tuần tự từ Cell 1 đến Cell 7.
2. **Amazon Baby & Sports:**
   - Mở notebook: [`notebook/P3/stair_ne_nlgcl_v5_plus.ipynb`](file:///d:/STAIR-Enhanced/notebook/P3/stair_ne_nlgcl_v5_plus.ipynb).
   - Chọn phương pháp muốn chạy tại Cell 6 (`SELECTED_METHOD = 'dcd_gated'`).
