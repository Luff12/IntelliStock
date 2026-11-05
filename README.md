

# IntelliStock 📦
### AI-Powered Demand Forecasting & Inventory Intelligence

IntelliStock is an AI-powered inventory intelligence dashboard designed to help businesses analyze sales, forecast product demand, identify stock risks, and make data-driven replenishment decisions.

The application combines machine learning, data analytics, and interactive visualizations to support smarter inventory planning.

## ✨ Features

- **Sales Analytics Dashboard** — Explore sales trends, store performance, and item-level sales.
- **Demand Forecasting** — Generate future demand predictions using a trained XGBoost model.
- **Forecast Evaluation** — Evaluate predictions using MAE, RMSE, and WAPE.
- **Inventory Optimization** — Calculate reorder points, target stock levels, and recommended order quantities.
- **Forecast-Driven Reorder Recommendations** — Estimate replenishment needs using predicted demand and supplier lead time.
- **Model Explainability** — Understand feature contributions to model predictions using SHAP.
- **Inventory Anomaly Detection** — Identify unusual inventory patterns using anomaly detection techniques.
- **Interactive Dashboard** — Explore data with filters, charts, and downloadable CSV results.

## 🧰 Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Machine Learning | XGBoost, Scikit-learn |
| Explainable AI | SHAP |
| Model Storage | Joblib |
| Development | Jupyter Notebook, VS Code, Git |

## 📁 Project Structure

```text
IntelliStock/
├── app.py
├── requirements.txt
├── data/
│   └── training and inventory datasets
├── models/
│   └── trained forecasting model
├── notebooks/
│   └── analysis and model development
└── src/
    ├── forecast.py
    ├── explainability.py
    ├── anomaly.py
    └── other supporting modules
```

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Luff12/IntelliStock.git
cd IntelliStock
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install streamlit pandas numpy plotly scikit-learn xgboost shap joblib
```

If the project’s `requirements.txt` is maintained with the dependencies, you can install them with:

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

## 📊 Application Pages

| Page | Purpose |
|---|---|
| Overview | Summary metrics and sales trends |
| Sales Analysis | Analyze sales by date, store, and item |
| Forecast Evaluation | Measure forecasting performance |
| Inventory Optimization | Calculate stock targets and reorder quantities |
| Demand Forecast | Predict future demand for selected products |
| Forecast-Driven Reorder | Generate replenishment recommendations from forecast demand |
| Model Explainability | Inspect feature contributions |
| Inventory Anomaly Detection | Identify unusual inventory observations |

## 🧠 Machine Learning Workflow

1. Load and preprocess historical sales data.
2. Generate time-based and lag-based features.
3. Use the trained forecasting model to predict future demand.
4. Evaluate predictions against validation data.
5. Apply forecast results to inventory replenishment calculations.
6. Use explainability and anomaly detection to support analysis.

## 📂 Data Requirements

The application expects the historical sales dataset in one of these locations:

```text
data/train (1).csv
data/train.csv
```

The sales dataset should contain the following columns:

- `date`
- `store`
- `item`
- `sales`

The forecasting workflow also requires the trained model file in the expected `models/` directory.

Inventory recommendation features may use an inventory CSV containing store, item, and current stock information. If stock data is not available, current stock can be entered through the application.

> Keep private, sensitive, or large datasets and model files out of public repositories unless you have permission to share them.

## 📈 Forecast Evaluation Metrics

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted demand.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more strongly.
- **WAPE (Weighted Absolute Percentage Error):** Measures total absolute error relative to total actual demand.

## 🎯 Project Objective

IntelliStock demonstrates how machine learning and data analytics can be applied to inventory management to help reduce stockout risk, improve replenishment planning, and support data-driven operational decisions.

## 👨‍💻 Author

**Shibin Mathew**

GitHub: [Luff12](https://github.com/Luff12)

## 🔗 Repository

[https://github.com/Luff12/IntelliStock](https://github.com/Luff12/IntelliStock)
