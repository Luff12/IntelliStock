import pandas as pd
import numpy as np


def create_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store", "item", "date"])

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Group sales by store and item
    grouped = df.groupby(["store", "item"])["sales"]

    # Lag features: past demand
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # Rolling statistics (shift first to prevent leakage)
    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = df.groupby(["store", "item"])["sales"].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )

        df[f"rolling_std_{window}"] = df.groupby(["store", "item"])["sales"].transform(
            lambda x: x.shift(1).rolling(window).std()
        )

    # Trend feature
    df["sales_change_7"] = df["lag_7"] - df["lag_14"]

    return df


if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\Shib\IntelliStock\data\train (1).csv")
    df = create_features(df)

    print(df.head(35))
    print(df.isnull().sum())
