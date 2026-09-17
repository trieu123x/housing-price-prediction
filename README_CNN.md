# Assignment 04 — CNN 1D cho bài toán Dự đoán Giá nhà (Hồi quy)

> **Môn học:** Intelligent System Development — TS. Trần Đình Quế
> **Sinh viên:** Đinh Hải Triều — B23DCCN843 — Lớp 06

Bổ sung cho `README.md` (Assignment 01 và 03). Tài liệu này mô tả phần **mạng
nơ-ron tích chập 1D cho bài toán hồi quy** được thêm vào ở Assignment 04.

## Tệp mới

| File | Nội dung |
|---|---|
| `notebook_cnn_part2.ipynb` | Notebook huấn luyện CNN 1D bằng **NumPy from scratch**, PyTorch và TensorFlow |
| `model_cnn.json` | Trọng số CNN + tham số tiền xử lý và nghịch đảo nhãn (54 KB) |
| `model_cnn_samples.json` | 12 mẫu tham chiếu để kiểm tra parity JavaScript ↔ notebook |
| `figures/c2_fig*.png` | 5 biểu đồ do notebook sinh ra |

## Kiến trúc — điểm khác biệt nằm ở tầng ra

```
(1, 19) → Conv1D(16, K=3) → ReLU → Conv1D(16, K=3) → ReLU → MaxPool(2)
        → Conv1D(32, K=3) → ReLU → Flatten → Dropout(0.15)
        → Dense(16) → ReLU → Dense(1)  ← TUYẾN TÍNH, không Sigmoid
```

**5.009 tham số · 5 tầng huấn luyện được.** Độ dài chuỗi:
$19 \to 17 \to 15 \to 7 \to 5$.

Theo nguyên lý *"kiến trúc theo sát bài toán"*, nơ-ron đầu ra phải **tuyến tính**
và hàm mất mát là **MSE** — giữ Sigmoid sẽ chặn dự đoán trong $[0,1]$ và mô hình
không thể biểu diễn giá nhà hàng chục tỉ đồng. Gradient của head mới có hệ số
khác ($\partial L/\partial z = 2(\hat z - z)/n$) nên được kiểm chứng lại riêng:
sai số tương đối $9{,}4\times10^{-7}$.

Mạng học trên thang $\log(1+\text{Price})$ đã chuẩn hoá z-score, nên đầu ra phải
được nghịch đảo để về lại tỉ VNĐ:

$$\widehat{\text{Price}} = \exp(\hat{z}\cdot\sigma + \mu) - 1$$

## Triển khai serverless trên Vercel

Trọng số nằm trong `model_cnn.json`, **commit thẳng lên GitHub**; Vercel phục vụ
nó như một **tệp tĩnh** và frontend `fetch` về khi tải trang. Không có hàm
serverless, không có máy chủ suy luận — mọi phép tính chạy trên trình duyệt.

```javascript
const res = await fetch('model_cnn.json', { cache: 'no-cache' });
CNN_MODEL = await res.json();
```

Chuỗi đầu vào dài 19 được dựng trong JavaScript đúng thứ tự notebook:

```javascript
function buildCnnInput(province, houseType, area, bedrooms, bathrooms, floors) {
  const M = CNN_MODEL;
  const nums = [area, bedrooms, bathrooms, floors]
    .map((v, i) => (v - M.scaler.mean_[i]) / M.scaler.scale_[i]);
  const provOH = M.provinces.map(p => (p === province ? 1 : 0));
  const typeOH = M.house_types.map(t => (t === houseType ? 1 : 0));
  return [...nums, ...provOH, ...typeOH];      // 4 + 10 + 5 = 19
}
```

Nếu `fetch` thất bại, trang tự động lùi về Deep MLP (Assignment 03) và Gradient
Boosting (Assignment 01) vốn vẫn nhúng inline.

## Kết quả trên tập kiểm thử (300 bất động sản)

| Mô hình | R² (log) | MAE (tỉ) | RMSE (tỉ) | MAPE (%) |
|---|---|---|---|---|
| MLP-5 (Assignment 03) | **0,9333** | 3,580 | 6,445 | **16,93** |
| Gradient Boosting | 0,9300 | **3,549** | 6,259 | 17,28 |
| CNN-5 (TensorFlow) | 0,9283 | 3,562 | **6,238** | 17,64 |
| CNN-5 (PyTorch) | 0,9278 | 3,650 | 6,503 | 17,65 |
| **CNN-5 (NumPy from scratch)** | 0,9254 | 3,748 | 6,686 | 17,63 |
| Random Forest | 0,9246 | 3,601 | 6,572 | 17,01 |
| Linear Regression | 0,9148 | 4,204 | 10,699 | 18,96 |

CNN xếp thứ ba đến thứ năm — **thua sát nút** nhưng nhất quán theo một hướng.

## Vì sao CNN không thắng ở bài này?

Notebook phân tích **cấu trúc khối** của chuỗi đầu vào sau one-hot:

```
vị trí:  0    1    2    3  |  4 ............. 13  |  14 ......... 18
         Area Bed  Bath Flo|  one-hot Province     |  one-hot HouseType
         ─── 4 cột SỐ ────  ── 10 cột CHỈ BÁO ───   ── 5 cột CHỈ BÁO ──
```

Trong **17 cửa sổ** tích chập dài 3, có **13 cửa sổ nằm trọn trong khối one-hot**.
Ở đó nhiều nhất một giá trị bằng 1, nên phép tích chập rút gọn thành
$z = w_j + b$ — tức một **bảng tra cứu**, không phải bộ phát hiện mẫu cục bộ.

Tệ hơn: ba tỉnh nào bị gộp chung một cửa sổ hoàn toàn do **thứ tự bảng chữ cái**
quyết định. Không có lý do thực tế nào để "Đà Nẵng, Đồng Nai, Hà Nội" hợp thành
một nhóm cục bộ có ý nghĩa.

Đáng chú ý là Linear Regression đạt $R^2 = 0{,}9148$ — chỉ kém mô hình tốt nhất
0,019, cho thấy bài toán này **phần lớn là tuyến tính trên thang log**. Chi tiết
ở Chương III báo cáo `A04_06_trieu.843.PDF`.

## Kiểm chứng

```bash
node "../../tuan 4/lib/check_html_cnn.mjs" . house
```

Kết quả: 12/12 mẫu khớp, sai lệch giá tối đa **7,2 × 10⁻⁵ tỉ VNĐ** (khoảng 72
đồng) — đúng bằng ngân sách làm tròn 6 chữ số khi ghi JSON.

## Chạy local

```bash
py -3 -m http.server 8002
```

Mở `http://localhost:8002`. Cần chạy qua HTTP server (không mở trực tiếp file)
vì trang dùng `fetch` để nạp `model_cnn.json`.
