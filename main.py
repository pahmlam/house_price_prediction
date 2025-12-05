import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles # Import để phục vụ file ảnh
import gradio as gr
from core import HousePriceManager
import os

# 1. Khởi tạo App & Manager
app = FastAPI()
manager = HousePriceManager()

app.mount("/static_images", StaticFiles(directory="static_images"), name="static_images")

# 2. Xây dựng Giao diện Gradio
def create_ui():
    with gr.Blocks(title="AIO2025 House Price Prediction") as interface:
        gr.Markdown("# 🏠 Hệ thống Dự đoán Giá nhà (Advanced)")
        gr.Markdown("""
        Hệ thống tích hợp các kỹ thuật nâng cao: **KNN Imputer**, **Robust Scaler**, **Log Target Transform** & **Polynomial Features**.
        """)

        # --- TAB 1: HUẤN LUYỆN ---
        with gr.Tab("🛠️ Quy trình Huấn luyện"):
            with gr.Row():
                file_input = gr.File(label="Bước 1: Tải lên Dataset (CSV)", file_types=[".csv"])
                upload_btn = gr.Button("🔍 Tải lên & Phân tích EDA", variant="secondary")
            
            status_text = gr.Textbox(label="Thông báo hệ thống", interactive=False)
            
            # Gallery hiển thị ảnh EDA
            with gr.Accordion("📊 Kết quả Phân tích Dữ liệu (EDA)", open=True):
                eda_gallery = gr.Gallery(label="Biểu đồ phân tích", columns=2, height="auto")
            
            gr.Markdown("---")
            train_btn = gr.Button("🚀 Bước 2: Huấn luyện 3 Mô hình (Linear, Ridge, Lasso)", variant="primary")
            
            # Bảng kết quả
            result_table = gr.Dataframe(label="Kết quả Đánh giá trên tập Test (RMSE & R2)", interactive=False)
            
            delete_btn = gr.Button("🗑️ Reset / Xóa mô hình cũ", variant="stop")

        # --- TAB 2: DỰ ĐOÁN ---
        with gr.Tab("🔮 Dự đoán Giá nhà"):
            gr.Markdown("### Nhập thông số ngôi nhà cần định giá")
            
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=["LinearRegression", "Ridge", "Lasso"], 
                    label="Chọn mô hình đã huấn luyện", 
                    value="Ridge"
                )
            
            # Các trường nhập liệu khớp với self.input_features trong core.py
            with gr.Row():
                with gr.Column():
                    inp_overall = gr.Slider(1, 10, value=7, step=1, label="Chất lượng tổng thể (OverallQual)")
                    inp_grliv = gr.Number(value=1500, label="Diện tích ở trên mặt đất (GrLivArea - sq ft)")
                    inp_cars = gr.Slider(0, 4, value=2, step=1, label="Sức chứa Gara (GarageCars)")
                    inp_garea = gr.Number(value=500, label="Diện tích Gara (GarageArea - sq ft)")
                with gr.Column():
                    inp_bsmt = gr.Number(value=1000, label="Diện tích hầm (TotalBsmtSF - sq ft)")
                    inp_1stflr = gr.Number(value=1000, label="Diện tích tầng 1 (1stFlrSF - sq ft)")
                    inp_bath = gr.Slider(0, 4, value=2, step=1, label="Số phòng tắm (FullBath)")
                    inp_year = gr.Number(value=2005, label="Năm xây dựng (YearBuilt)")

            predict_btn = gr.Button("💰 Dự đoán ngay", variant="primary")
            output_price = gr.Textbox(label="Giá trị ước tính", text_align="center", scale=2)

        # --- XỬ LÝ SỰ KIỆN ---
        
        # 1. Upload & EDA
        def on_upload(file):
            if file is None: return "Vui lòng chọn file.", None
            # Load dữ liệu
            msg = manager.load_data(file.name)
            # Vẽ biểu đồ (trả về danh sách đường dẫn file ảnh)
            plots = manager.perform_eda()
            return msg, plots

        upload_btn.click(on_upload, inputs=file_input, outputs=[status_text, eda_gallery])

        # 2. Huấn luyện
        def on_train():
            try:
                df_results = manager.train_models()
                return "✅ Huấn luyện hoàn tất! Đã lưu 3 mô hình.", df_results
            except Exception as e:
                return f"❌ Lỗi: {str(e)}", None

        train_btn.click(on_train, inputs=None, outputs=[status_text, result_table])

        # 3. Dự đoán
        def on_predict(model_name, q, grliv, cars, garea, bsmt, fst, bath, year):
            # Map input vào dictionary
            features = {
                "OverallQual": q, "GrLivArea": grliv, "GarageCars": cars,
                "GarageArea": garea, "TotalBsmtSF": bsmt, "1stFlrSF": fst,
                "FullBath": bath, "YearBuilt": year
            }
            return manager.predict_price(model_name, features)

        predict_btn.click(
            on_predict, 
            inputs=[model_selector, inp_overall, inp_grliv, inp_cars, inp_garea, inp_bsmt, inp_1stflr, inp_bath, inp_year], 
            outputs=output_price
        )

        # 4. Xóa mô hình
        delete_btn.click(manager.delete_models, inputs=None, outputs=status_text)

    return interface

# Mount ứng dụng
gradio_app = create_ui()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    print("Server đang chạy tại: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)