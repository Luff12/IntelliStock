# src/uncertainty.py

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation_predictions.csv"


def get_error_margin(coverage=0.90):
    """Estimate an absolute forecast error margin from validation data."""

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Validation file not found: {VALIDATION_PATH}")

    df = pd.read_csv(VALIDATION_PATH)

    actual_col = next(
        (c for c in ["actual", "sales", "y_true"] if c in df.columns),
        None,
    )
    predicted_col = next(
        (c for c in ["predicted", "prediction", "y_pred"] if c in df.columns),
        None,
    )

    if actual_col is None or predicted_col is None:
        raise ValueError("Validation file must contain actual and predicted columns.")

    actual = pd.to_numeric(df[actual_col], errors="coerce")
    predicted = pd.to_numeric(df[predicted_col], errors="coerce")

    errors = (actual - predicted).abs().dropna()

    if errors.empty:
        raise ValueError("No valid validation errors found.")

    return float(errors.quantile(coverage))


def add_uncertainty_bounds(forecast_df, prediction_col, coverage=0.90):
    """Add lower and upper forecast bounds."""

    result = forecast_df.copy()
    margin = get_error_margin(coverage)

    result[prediction_col] = pd.to_numeric(result[prediction_col], errors="coerce")

    result["lower_bound"] = np.maximum(
        0,
        result[prediction_col] - margin,
    )

    result["upper_bound"] = result[prediction_col] + margin

    result["uncertainty_margin"] = margin

    return result
