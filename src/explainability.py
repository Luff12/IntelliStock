# src/explainability.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# ==================================================
# PATHS
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "train (1).csv"

if not DATA_PATH.exists():
    DATA_PATH = PROJECT_ROOT / "data" / "train.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"


# ==================================================
# FEATURES
# ==================================================
FEATURE_COLUMNS = [
    "store",
    "item",
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "week_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "sales_change_7",
]


# ==================================================
# LOAD DATA
# ==================================================
def load_explainability_data():
    """Load sales history and generate model features."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df.columns = [str(column).strip().lower() for column in df.columns]

    required_columns = [
        "date",
        "store",
        "item",
        "sales",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    df = df.dropna(subset=required_columns)

    df = df.sort_values(["store", "item", "date"]).copy()

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int64")

    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int64")

    # Lag features
    grouped_sales = df.groupby(["store", "item"])["sales"]

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = grouped_sales.shift(lag)

    # Rolling features using past sales only
    for window in [7, 14, 28]:

        df[f"rolling_mean_{window}"] = df.groupby(["store", "item"])["sales"].transform(
            lambda series: series.shift(1).rolling(window).mean()
        )

        df[f"rolling_std_{window}"] = df.groupby(["store", "item"])["sales"].transform(
            lambda series: series.shift(1).rolling(window).std()
        )

    # Sales change feature
    df["sales_change_7"] = df["lag_1"] - df["lag_7"]

    # Convert features to numeric
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows without sufficient history
    df = df.dropna(subset=FEATURE_COLUMNS).copy()

    return df


# ==================================================
# EXPLAIN MODEL PREDICTION
# ==================================================
def explain_item(store, item):
    """
    Generate a prediction and SHAP feature explanation.

    Returns a DataFrame compatible with app.py:
    feature, importance, shap_value, feature_value
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    df = load_explainability_data()

    # Select store and item
    item_data = df[(df["store"] == store) & (df["item"] == item)].sort_values("date")

    if item_data.empty:
        raise ValueError(
            f"No usable historical data found for " f"Store {store}, Item {item}."
        )

    # Use latest available feature row
    latest_row = item_data.iloc[-1]

    X = item_data.loc[[item_data.index[-1]], FEATURE_COLUMNS].copy()

    # Ensure numeric values
    X = X.apply(pd.to_numeric, errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = X.astype("float64")

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Generate prediction
    prediction = float(model.predict(X)[0])

    # SHAP explanation
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Handle SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 0]
    elif shap_values.ndim == 2:
        shap_values = shap_values[0]

    # Build feature explanation table
    explanation_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "shap_value": shap_values,
            "importance": np.abs(shap_values),
            "feature_value": X.iloc[0].values,
        }
    )

    explanation_df = explanation_df.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    # Add prediction details
    explanation_df["prediction"] = prediction
    explanation_df["date"] = latest_row["date"]

    return explanation_df
