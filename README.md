# Website Dự đoán Giá nhà Việt Nam (Vietnam Housing AI)

> **Môn học:** Intelligent System Development — TS. Đinh Quế Trần  
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
