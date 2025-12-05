import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import joblib
import os
import uuid
import shutil

matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

MODEL_DIR = "models"
IMAGE_DIR = "static_images"

os.makedirs(MODEL_DIR, exist_ok=True)
if os.path.exists(IMAGE_DIR):
    shutil.rmtree(IMAGE_DIR)  
os.makedirs(IMAGE_DIR, exist_ok=True)

class HousePriceManager:
    def __init__(self):
        self.data = None
        self.input_features = [
            "OverallQual", "GrLivArea", "GarageCars", "GarageArea", 
            "TotalBsmtSF", "1stFlrSF", "FullBath", "YearBuilt"
        ]

    def load_data(self, file_path):
        """Tải dữ liệu từ file CSV tải lên"""
        try:
            self.data = pd.read_csv(file_path)
            return f"✅ Đã tải dữ liệu thành công! Kích thước: {self.data.shape}"
        except Exception as e:
            return f"❌ Lỗi tải file: {str(e)}"

    def _save_fig_to_file(self, fig):
        """
        FIX LỖI WINDOWS PATH TOO LONG:
        Thay vì trả về chuỗi Base64 dài, ta lưu ảnh vào thư mục và trả về đường dẫn file.
        """
        filename = f"eda_{uuid.uuid4().hex}.png"
        file_path = os.path.join(IMAGE_DIR, filename)

        fig.savefig(file_path, format='png', bbox_inches='tight')
        plt.close(fig)  # Giải phóng bộ nhớ RAM
        
        return os.path.abspath(file_path)

    def perform_eda(self):
        """
        Thực hiện EDA tối ưu hóa hiệu năng.
        Chỉ vẽ những đặc trưng quan trọng để tránh treo máy.
        """
        if self.data is None:
            return []
        
        plots = []
        
        # 1. Phân phối SalePrice 
        fig1 = plt.figure(figsize=(10, 6))
        sns.histplot(self.data['SalePrice'], kde=True, color="teal")
        plt.title("Phân phối giá nhà (SalePrice)")
        plots.append(self._save_fig_to_file(fig1))

        # 2. Dữ liệu bị thiếu (Chỉ Top 15) 
        missing = self.data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if not missing.empty:
            fig2 = plt.figure(figsize=(10, 8))
            missing.head(15).plot.barh(color="salmon", edgecolor="black")
            plt.title("Top 15 đặc trưng thiếu dữ liệu nhiều nhất")
            plt.xlabel("Số lượng mẫu bị thiếu")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plots.append(self._save_fig_to_file(fig2))

        # 3. Heatmap tương quan (Chỉ Top 15 features tương quan mạnh nhất) 
        # Giúp tránh lỗi render quá lâu khi vẽ ma trận 80x80
        numeric_df = self.data.select_dtypes(include=[np.number])
        
        if 'SalePrice' in numeric_df.columns:
            corr_with_target = numeric_df.corrwith(numeric_df['SalePrice']).abs()
            top_corr_features = corr_with_target.sort_values(ascending=False).head(15).index
            df_heatmap = numeric_df[top_corr_features]
        else:
            df_heatmap = numeric_df.iloc[:, :15] # Fallback

        fig3 = plt.figure(figsize=(12, 10))
        sns.heatmap(
            df_heatmap.corr(), 
            annot=True, fmt=".2f", 
            cmap="coolwarm", 
            linewidths=0.5, 
            cbar_kws={"shrink": .8}
        )
        plt.title(f"Heatmap tương quan (Top {len(df_heatmap.columns)} Features)")
        plt.tight_layout()
        plots.append(self._save_fig_to_file(fig3))

        return plots

    def train_models(self):
        """
        Huấn luyện mô hình với các cải tiến nâng cao theo mục 4 tài liệu:
        - Xử lý thiếu: KNNImputer 
        - Chuẩn hóa: RobustScaler (tốt cho ngoại lai)
        - Phi tuyến: PolynomialFeatures (degree=2) 
        - Target: Log Transformation 
        """
        if self.data is None:
            raise ValueError("Chưa có dữ liệu!")

        # Chỉ sử dụng các cột input_features để huấn luyện cho nhất quán với UI nhập liệu
        # (Trong thực tế nên dùng Feature Selection kỹ hơn, nhưng ở đây ta khớp với giao diện)
        X = self.data[self.input_features].copy()
        y = self.data['SalePrice']

        # LOG TRANSFORM TARGET: Giúp SalePrice có phân phối chuẩn hơn 
        y_log = np.log1p(y)

        # Chia train/test [cite: 201]
        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X, y_log, test_size=0.2, random_state=42
        )

        # Pipeline tiền xử lý
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),       # Điền khuyết thông minh 
            ('scaler', RobustScaler()),                   # Chống nhiễu ngoại lai
            ('poly', PolynomialFeatures(degree=2, include_bias=False)) # Quan hệ phi tuyến 
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.input_features)
            ])

        results = []
        models_map = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=10.0),   # Alpha cao hơn để giảm overfit do Poly features
            "Lasso": Lasso(alpha=0.0005)  # Alpha nhỏ do target scale log
        }

        # Vòng lặp huấn luyện [cite: 281]
        for name, model in models_map.items():
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            
            clf.fit(X_train, y_train_log)
            
            # Lưu model
            joblib.dump(clf, os.path.join(MODEL_DIR, f"{name}.pkl"))
            
            # Dự đoán trên tập Test
            y_pred_log = clf.predict(X_test)
            
            # Inverse Log Transform (Đưa về giá tiền thực tế)
            y_pred_real = np.expm1(y_pred_log)
            y_test_real = np.expm1(y_test_log)
            
            # Tính metrics
            rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
            r2 = r2_score(y_test_real, y_pred_real)
            
            results.append({
                "Mô hình": name,
                "RMSE (Test)": f"${rmse:,.0f}",
                "R2 Score": round(r2, 4)
            })

        return pd.DataFrame(results)

    def predict_price(self, model_name, inputs):
        """Dự đoán giá với input người dùng"""
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            return "⚠️ Cần huấn luyện mô hình trước!"
        
        try:
            model = joblib.load(model_path)
            
            # Tạo DataFrame input đúng thứ tự cột
            input_df = pd.DataFrame([inputs])[self.input_features]
            
            # Dự đoán (Log scale)
            log_pred = model.predict(input_df)
            
            # Chuyển về giá thực (Exp)
            real_pred = np.expm1(log_pred)[0]
            
            return f"${real_pred:,.2f}"
        except Exception as e:
            return f"Lỗi: {str(e)}"

    def delete_models(self):
        """Xóa mô hình cũ"""
        cnt = 0
        if os.path.exists(MODEL_DIR):
            for f in os.listdir(MODEL_DIR):
                if f.endswith(".pkl"):
                    os.remove(os.path.join(MODEL_DIR, f))
                    cnt += 1
        return f"Đã xóa {cnt} mô hình."