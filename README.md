
---

# 🏠 House Price Prediction System (AIO2025)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Gradio](https://img.shields.io/badge/Gradio-4.0-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-yellow)

An **end-to-end Machine Learning system** for house price prediction.
The project covers the full pipeline from **Exploratory Data Analysis (EDA)** and **advanced preprocessing** to **model training** and an **interactive web-based prediction interface**.

This project is built based on requirements and exercises from the **AIO2025** materials, and extended with a **FastAPI + Gradio** architecture.

---

## ✨ Key Features

* **📂 Upload & Automated Analysis:**
  Supports CSV file upload with automatic EDA visualization (distributions, missing values, correlation heatmaps).

* **🧠 Advanced Training Pipeline:**

  * Automatic missing value handling using **KNN Imputer**.
  * Outlier-robust normalization with **Robust Scaler**.
  * Skewed target handling via **Log Transformation**.
  * Automatic nonlinear feature generation using **Polynomial Features**.

* **🚀 Multi-Model Training:**
  Trains three models in parallel: **Linear Regression**, **Ridge**, and **Lasso**.

* **🔮 Prediction Interface:**
  Intuitive web UI for real-time house price prediction.

* **💾 Model Management:**
  Automatically saves trained models and supports deleting/resetting old models.

---

## 🛠️ Installation & Running the Project

### 1. Prerequisites

* Python 3.8 or later
* Git

### 2. Installation

Open a terminal and follow these steps:

```bash
# Clone the repository (if using Git)
git clone https://github.com/pahmlam/house_price_prediction.git
cd house-price-prediction

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Application

```bash
python main.py
```

After launching, open your browser and navigate to:
`http://localhost:8000`

---

## 📊 Methods & Algorithms

The system applies several advanced techniques to improve prediction accuracy (RMSE / R²) compared to baseline approaches:

1. **Missing Value Imputation:**
   Uses `KNNImputer` (K-Nearest Neighbors) instead of mean imputation, preserving the underlying data distribution.

2. **Feature Scaling:**
   Applies `RobustScaler` based on quantiles to reduce the impact of outliers, which are common in real estate data.

3. **Target Transformation:**
   Applies `np.log1p` to the house price (`SalePrice`) to approximate a normal distribution, improving the performance of linear models.

4. **Feature Engineering:**
   Generates second-order features using `PolynomialFeatures (degree=2)` to capture nonlinear relationships.

---

## 📂 Project Structure

```text
house_price_system/
├── data/                  # Sample datasets (if any)
├── models/                # Stored trained models (.pkl files)
├── static_images/         # Temporary storage for EDA plots
├── core.py                # Core logic: EDA, preprocessing, training
├── main.py                # App server: FastAPI config & Gradio UI
├── requirements.txt       # Dependency list
├── .gitignore             # Git ignore configuration
└── README.md              # Project documentation
```

---

## 📸 Usage Guide

### Step 1: Training (Tab 1)

1. Upload the `train.csv` dataset.
2. Click **“Upload & Analyze EDA”** to view data visualizations.
3. Click **“Train 3 Models”**.
   The system runs the full pipeline and returns RMSE / R² results.

### Step 2: Prediction (Tab 2)

1. Switch to the **“House Price Prediction”** tab.
2. Select the model to use (Linear, Ridge, or Lasso).
3. Enter house attributes (Area, Year Built, Number of Rooms, etc.).
4. Click **“Predict Now”** to view the estimated price.

---

## ⚠️ Common Troubleshooting

### Error: `ValueError: Path too long` on Windows

* This issue is mitigated by saving plots to the `static_images` directory instead of encoding them in Base64.
  If the error persists, ensure that the project directory path is not too deep
  (e.g., place it under `C:\Projects\HousePrice`).

### Error: `Model has not been trained`

* Return to **Tab 1** and train the models before attempting prediction.

---

## 🤝 Contributing

Contributions are welcome!
Please submit a Pull Request or open an Issue if you find a bug or have suggestions.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE.txt` for more information.

Copyright © 2025 **Pham Tung Lam**

---

