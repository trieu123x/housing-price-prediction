# Website Dự đoán Giá nhà Việt Nam (Vietnam Housing AI)

> **Môn học:** Intelligent System Development — TS. Trần Đình Quế  
> **Phần 2:** Regression — Dự đoán Giá nhà Việt Nam 2024  

---

## 📁 Cấu trúc thư mục dự án

```
website_dudoan_gia_nha/
├── index.html                ← Single Page Web App (Frontend + ML Engine)
├── model.json                ← Mô hình Gradient Boosting đã nén (113.9 KB)
├── vercel.json               ← Cấu hình deploy Vercel
├── notebook_part2.ipynb      ← Notebook huấn luyện & thí nghiệm (17 sections)
├── data/
│   └── vietnam_housing_dataset.csv ← Dataset giá nhà Việt Nam
└── README.md                 ← Hướng dẫn sử dụng & deploy
```

---

## ⚡ Deploy Vercel (Serverless / Static Site)

Mô hình Gradient Boosting Regressor được nhúng trực tiếp dạng `model.json` (113.9 KB). Thuật toán inference tích lũy đầu ra qua các cây (Tree Accumulation) được viết bằng **Pure JavaScript** chạy 100% trong trình duyệt client, không cần backend server.

### Deploy bằng Vercel CLI
```bash
cd website_dudoan_gia_nha
vercel --prod
```

### Deploy qua Vercel Dashboard
1. Push thư mục này lên repository GitHub.
2. Mở Vercel Dashboard → Import Project.
3. Chọn `Root Directory` là `website_dudoan_gia_nha`.
4. Nhấn **Deploy** — Vercel sẽ tự động deploy thành công!

---

## 🚀 Chạy Local

```bash
# Sử dụng Python HTTP server đơn giản:
py -3 -m http.server 8002
```
👉 Truy cập: `http://localhost:8002`

---

## 📊 Thông tin Mô hình Machine Learning

- **Bài toán:** Regression (Dự đoán giá bất động sản theo tỷ đồng)
- **Dataset:** Vietnam Housing Dataset 2024 (2000 mẫu, 6 đặc trưng BĐS)
- **Mô hình:** Gradient Boosting Regressor (80 cây, learning_rate=0.12, max_depth=4)
- **Hệ số xác định (R²):** `0.8655`
- **Sai số tuyệt đối trung bình (MAE):** `4.08 tỷ`
- **Thời gian Inference:** `< 2 ms` trên trình duyệt

---

# 🧠 Assignment 03 — Mạng nơ-ron sâu 5 tầng (NumPy from scratch)

Trang web nay chạy **song song hai mô hình** và cho phép đối sánh trực tiếp:

| Engine | Mô hình | Nguồn |
|---|---|---|
| 🧠 **Deep MLP-5** (mặc định) | `19 → 128 → 64 → 32 → 16 → 1` (Linear) | Assignment 03 |
| 🌳 Gradient Boosting | 80 cây, `learning_rate=0.12` | Assignment 01 |
| ⚖️ Đối sánh | Chạy cả hai, hiển thị chênh lệch | — |

## Hai quyết định thiết kế quan trọng

**1. One-Hot Encoding thay vì Label Encoding.** `Province` và `HouseType` là biến định
danh. Gán "Hà Nội = 3, TP.HCM = 7" áp một thứ tự giả lên mạng — với mô hình cây thì vô
hại (cây chỉ so ngưỡng), nhưng với mạng nơ-ron lấy tổ hợp tuyến tính thì sai hoàn toàn.
One-hot đưa đầu vào từ 6 lên **19 chiều**, và tầng 1 tự học một vector nhúng cho từng tỉnh.

**2. Học trên thang log.** Giá nhà lệch phải rất mạnh (skewness = **2.949**). Nếu tối
thiểu MSE trên giá gốc, một biệt thự 200 tỉ chi phối toàn bộ hàm mất mát. Sau `log1p`,
skewness rơi xuống **0.279** và mạng tối ưu **sai số tương đối** — đúng cách thị trường
định giá. Nhãn được `log1p` rồi chuẩn hoá z-score (`mean=2.7900`, `std=0.6870`); đầu ra
quy đổi ngược bằng `expm1(z * std + mean)`.

## Kiến trúc mạng

| Tầng | W shape | Kích hoạt | Tham số |
|---|---|---|---|
| Layer 1 | (19, 128) | ReLU | 2.560 |
| Layer 2 | (128, 64) | ReLU | 8.256 |
| Layer 3 | (64, 32) | ReLU | 2.080 |
| Layer 4 | (32, 16) | ReLU | 528 |
| Layer 5 | (16, 1) | **Linear** | 17 |
| **Tổng** | | | **13.441** |

Tầng 5 **bắt buộc** tuyến tính: Sigmoid chặn đầu ra trong `(0,1)`, ReLU chặn ở `[0,∞)` —
cả hai đều không biểu diễn được z-score có thể âm.

## Kết quả trên tập Test (300 mẫu)

Mọi mô hình đều học trên **cùng thang log** và quy đổi ngược bằng cùng công thức.

| Mô hình | R² (log) | MAE (tỷ) | RMSE (tỷ) | MAPE (%) |
|---|---|---|---|---|
| Deep MLP-5 (PyTorch) | **0.9332** | 3.594 | **5.998** | 17.69 |
| Gradient Boosting | 0.9299 | **3.578** | 6.279 | **17.22** |
| **Deep MLP-5 (NumPy)** | 0.9240 | 3.738 | 6.821 | 18.46 |
| Linear Regression | 0.9148 | 4.204 | 10.699 | 18.96 |
| Decision Tree | 0.8546 | 5.374 | 10.227 | 24.41 |

## Vì sao MAE nghe có vẻ tệ nhưng mô hình vẫn tốt

`R²(log) = 0.924` rất tốt, nhưng `MAE = 3.738 tỷ` nghe rất tệ. Hai con số này không mâu
thuẫn — chúng đo hai thứ khác nhau:

- **Trung vị sai số tương đối chỉ 14.82%** và gần như **không đổi** theo mức giá.
- Sai số *tuyệt đối* thì tăng theo giá: sai 15% trên căn 2 tỷ là 0.3 tỷ, sai 15% trên
  biệt thự 200 tỷ là **30 tỷ**.
- **14% số bất động sản đắt nhất tạo ra 41% tổng sai số tuyệt đối**; riêng nhóm > 100 tỷ
  chiếm 1% số mẫu nhưng gánh 9% tổng sai số.

Vì vậy trang web hiển thị khoảng giá tham khảo tính từ **trung vị sai số tương đối đo
trên tập test**, thay cho con số ±15% cố định trước đây. Xem phân tích đầy đủ trong
[`notebook_deep_part2.ipynb`](notebook_deep_part2.ipynb) và [`figures/`](figures/).

## Triển khai serverless

`model_deep.json` (128.6 KB) chứa trọng số, tham số StandardScaler, danh sách tỉnh/loại
hình cho one-hot, tham số biến đổi nhãn và các chỉ số đánh giá. Notebook tự động nhúng
bundle vào dòng `const DEEP_MODEL = ...;` trong `index.html`. Sai lệch giữa mô hình gốc
và bundle JSON: **3.4 × 10⁻⁶**.
