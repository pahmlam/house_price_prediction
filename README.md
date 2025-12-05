# 🏠 House Price Prediction System (AIO2025)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Gradio](https://img.shields.io/badge/Gradio-4.0-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-yellow)

Một hệ thống **End-to-End Machine Learning** hoàn chỉnh cho bài toán dự đoán giá nhà. Dự án tích hợp quy trình từ Khám phá dữ liệu (EDA), Tiền xử lý nâng cao, Huấn luyện mô hình đến Giao diện dự đoán tương tác trên Web.

Dự án được xây dựng dựa trên các yêu cầu và bài tập từ tài liệu **AIO2025**, mở rộng với kiến trúc FastAPI + Gradio.

---

## ✨ Tính năng chính

* **📂 Upload & Phân tích tự động:** Hỗ trợ tải file CSV, tự động phân tích và hiển thị biểu đồ EDA (Phân phối, Missing Values, Heatmap).
* **🧠 Pipeline Huấn luyện Nâng cao:**
    * Tự động xử lý dữ liệu thiếu bằng **KNN Imputer**.
    * Chuẩn hóa dữ liệu chống ngoại lai bằng **Robust Scaler**.
    * Xử lý biến thiên lệch (Skewed Target) bằng **Log Transformation**.
    * Tự động tạo đặc trưng phi tuyến với **Polynomial Features**.
* **🚀 Đa mô hình:** Huấn luyện song song 3 mô hình: **Linear Regression**, **Ridge**, **Lasso**.
* **🔮 Giao diện Dự đoán:** Nhập liệu trực quan thông qua Web UI để dự đoán giá nhà theo thời gian thực.
* **💾 Quản lý Mô hình:** Tự động lưu mô hình sau khi huấn luyện, hỗ trợ xóa/reset mô hình cũ.

---

## 🛠️ Cài đặt & Chạy dự án

### 1. Yêu cầu tiên quyết
* Python 3.8 trở lên.
* Git.

### 2. Cài đặt
Mở terminal và thực hiện các bước sau:

```bash
# Clone dự án (nếu bạn dùng git)
git clone https://github.com/pahmlam/house_price_prediction.git
cd house-price-prediction

# Tạo môi trường ảo (Khuyến nghị)
python -m venv venv

# Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
````

### 3\. Khởi chạy ứng dụng

```bash
python main.py
```

Sau khi chạy, truy cập trình duyệt tại địa chỉ: `http://localhost:8000`

-----

## 📊 Phương pháp & Thuật toán

Hệ thống áp dụng các kỹ thuật nâng cao để tối ưu hóa độ chính xác (RMSE/R2 Score) so với phương pháp cơ bản:

1.  **Xử lý dữ liệu thiếu (Imputation):** Sử dụng `KNNImputer` (K-Nearest Neighbors) thay vì điền trung bình (Mean), giúp giữ nguyên cấu trúc phân phối của dữ liệu.
2.  **Chuẩn hóa (Scaling):** Sử dụng `RobustScaler` dựa trên phân vị (Quantile) để giảm thiểu tác động của các giá trị ngoại lai (Outliers) thường gặp trong dữ liệu bất động sản.
3.  **Biến đổi biến mục tiêu (Target Transform):** Áp dụng `np.log1p` lên giá nhà (`SalePrice`) để đưa phân phối về dạng chuẩn (Normal Distribution), giúp các mô hình tuyến tính hoạt động hiệu quả hơn.
4.  **Feature Engineering:** Tạo các đặc trưng bậc 2 (`PolynomialFeatures degree=2`) để mô hình học được các mối quan hệ phi tuyến tính.

-----

## 📂 Cấu trúc dự án

```text
house_price_system/
├── data/                  # Thư mục chứa dữ liệu mẫu (nếu có)
├── models/                # Nơi lưu trữ các file mô hình (.pkl) sau khi train
├── static_images/         # Thư mục chứa ảnh biểu đồ EDA tạm thời
├── core.py                # Xử lý Logic chính: EDA, Preprocessing, Training
├── main.py                # App Server: FastAPI config & Gradio UI
├── requirements.txt       # Danh sách thư viện phụ thuộc
├── .gitignore             # File cấu hình bỏ qua của Git
└── README.md              # Tài liệu hướng dẫn
```

-----

## 📸 Hướng dẫn sử dụng

### Bước 1: Huấn luyện (Tab 1)

1.  Tải file dữ liệu `train.csv` lên hệ thống.
2.  Nhấn nút **"Tải lên & Phân tích EDA"** để xem biểu đồ dữ liệu.
3.  Nhấn nút **"Huấn luyện 3 Mô hình"**. Hệ thống sẽ chạy Pipeline và trả về kết quả RMSE/R2.

### Bước 2: Dự đoán (Tab 2)

1.  Chuyển sang tab **"Dự đoán giá nhà"**.
2.  Chọn loại mô hình muốn sử dụng (Linear, Ridge, hoặc Lasso).
3.  Nhập các thông số của ngôi nhà (Diện tích, Năm xây, Số phòng...).
4.  Nhấn **"Dự đoán ngay"** để xem giá trị ước tính.

-----

## ⚠️ Khắc phục sự cố thường gặp

**Lỗi: `ValueError: Path too long` trên Windows**

  * Dự án đã được xử lý để khắc phục lỗi này bằng cách lưu ảnh vào thư mục `static_images` thay vì encode Base64. Tuy nhiên, nếu vẫn gặp lỗi liên quan đến file hệ thống, hãy đảm bảo thư mục dự án không nằm quá sâu (Ví dụ: nên để ở `C:\Projects\HousePrice`).

**Lỗi: `Mô hình chưa được huấn luyện`**

  * Bạn cần quay lại Tab 1 và nhấn nút Huấn luyện trước khi thực hiện dự đoán.

-----

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh. Vui lòng tạo Pull Request hoặc mở Issue nếu bạn tìm thấy lỗi.

## 📜 License

Distributed under the MIT License. See LICENSE.txt for more information.

Copyright (c) 2025 Pham Tung Lam

```
```