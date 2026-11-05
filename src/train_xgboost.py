import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os

from features import create_features

# Load data
df = pd.read_csv(r"C:\Users\Shib\IntelliStock\data\train (1).csv")
df["date"] = pd.to_datetime(df["date"])

# Feature engineering
df = create_features(df)

# Remove rows with missing lag/rolling values
df = df.dropna().copy()

# Encode store and item as categorical identifiers
store_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df["store"] = store_encoder.fit_transform(df["store"])
df["item"] = item_encoder.fit_transform(df["item"])

# Define features and target
features = [
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

target = "sales"

# Chronological split: last 90 days for validation
cutoff_date = df["date"].max() - pd.Timedelta(days=90)

train_df = df[df["date"] <= cutoff_date]
val_df = df[df["date"] > cutoff_date]

X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Validation period:", val_df["date"].min(), "to", val_df["date"].max())

# Train XGBoost
model = XGBRegressor(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="mae",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

# Predictions
predictions = model.predict(X_val)
predictions = np.maximum(predictions, 0)

# Evaluation metrics
mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))

wape = (np.sum(np.abs(y_val - predictions)) / np.sum(np.abs(y_val))) * 100

print("\nModel Performance")
print("----------------")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"WAPE: {wape:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost_model.pkl")

# Save validation predictions
results = val_df[["date", "store", "item"]].copy()
results["actual_sales"] = y_val.values
results["predicted_sales"] = predictions

results.to_csv("data/validation_predictions.csv", index=False)

# Plot total daily actual vs predicted sales
daily_results = results.groupby("date")[["actual_sales", "predicted_sales"]].sum()

plt.figure(figsize=(14, 5))
plt.plot(daily_results.index, daily_results["actual_sales"], label="Actual Sales")
plt.plot(daily_results.index, daily_results["predicted_sales"], label="Predicted Sales")

plt.title("XGBoost: Actual vs Predicted Daily Sales")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10))

importance.head(10).sort_values().plot(kind="barh", figsize=(8, 5))
plt.title("Top 10 XGBoost Feature Importances")
plt.tight_layout()
plt.show()
