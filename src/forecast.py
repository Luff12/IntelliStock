import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ==================================================
# PATHS
# ==================================================
# Automatically locate the IntelliStock project folder
# Assumes this file is inside IntelliStock/src/
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "train (1).csv"
MODEL_PATH = PROJECT_DIR / "models" / "xgboost_model.pkl"

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
# LOAD DATA AND MODEL
# ==================================================
def load_forecasting_data():
    """Load historical sales data and trained XGBoost model."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)

    df.columns = [str(column).strip().lower() for column in df.columns]

    required_columns = [
        "date",
        "store",
        "item",
        "sales",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    df = df.dropna(subset=required_columns)

    df = df.sort_values(["store", "item", "date"])

    try:
        model = joblib.load(MODEL_PATH)

    except ModuleNotFoundError as error:
        if "xgboost" in str(error).lower():
            raise ModuleNotFoundError(
                "XGBoost is required to load this model. "
                "Install it using the same Python environment "
                "that runs Streamlit:\n"
                "python -m pip install xgboost"
            ) from error

        raise

    return df, model


# ==================================================
# FEATURE ENGINEERING
# ==================================================
def create_features(history, forecast_date, store, item):
    """Create features for the next forecast date."""

    values = np.asarray(history, dtype=float)

    if len(values) < 28:
        raise ValueError("At least 28 days of historical sales are required.")

    lag_1 = values[-1]
    lag_7 = values[-7]
    lag_14 = values[-14]
    lag_28 = values[-28]

    row = {
        "store": store,
        "item": item,
        "day_of_week": forecast_date.dayofweek,
        "day_of_month": forecast_date.day,
        "month": forecast_date.month,
        "year": forecast_date.year,
        "week_of_year": int(forecast_date.isocalendar().week),
        "is_weekend": int(forecast_date.dayofweek >= 5),
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "rolling_mean_7": np.mean(values[-7:]),
        "rolling_mean_14": np.mean(values[-14:]),
        "rolling_mean_28": np.mean(values[-28:]),
        "rolling_std_7": np.std(values[-7:], ddof=1),
        "rolling_std_14": np.std(values[-14:], ddof=1),
        "rolling_std_28": np.std(values[-28:], ddof=1),
        "sales_change_7": lag_1 - lag_7,
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# ==================================================
# FORECAST GENERATION
# ==================================================
def forecast_item(df, model, store, item, horizon=14):
    """Generate recursive daily forecasts."""

    series = df[(df["store"] == store) & (df["item"] == item)].sort_values("date")

    if series.empty:
        raise ValueError(f"No sales history for Store {store}, " f"Item {item}.")

    # Aggregate daily sales and fill missing dates with zero
    history_series = series.set_index("date")["sales"].resample("D").sum().fillna(0)

    if len(history_series) < 28:
        raise ValueError("At least 28 days of history are needed.")

    history = history_series.tolist()

    last_date = history_series.index.max()

    forecast_rows = []

    for step in range(1, int(horizon) + 1):

        forecast_date = last_date + pd.Timedelta(days=step)

        features = create_features(
            history=history, forecast_date=forecast_date, store=store, item=item
        )

        prediction = float(model.predict(features)[0])

        prediction = max(0, int(round(prediction)))

        forecast_rows.append(
            {
                "date": forecast_date,
                "store": store,
                "item": item,
                "predicted_sales": prediction,
            }
        )

        # Use prediction for the next forecast day
        history.append(prediction)

    return pd.DataFrame(forecast_rows)
