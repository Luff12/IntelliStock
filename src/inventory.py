import pandas as pd
import numpy as np
from pathlib import Path

# ==================================================
# FILE PATHS
# ==================================================
PROJECT_DIR = Path(r"C:\Users\Shib\IntelliStock")
DATA_DIR = PROJECT_DIR / "data"

# Automatically find the dataset
possible_files = [DATA_DIR / "train (1).csv", DATA_DIR / "train.csv"]

DATA_PATH = next((file for file in possible_files if file.exists()), None)

if DATA_PATH is None:
    raise FileNotFoundError(
        f"Dataset not found in {DATA_DIR}\n"
        "Make sure train (1).csv or train.csv is inside the data folder."
    )

STOCK_PATH = DATA_DIR / "current_stock.csv"
OUTPUT_PATH = DATA_DIR / "inventory_recommendations.csv"

# ==================================================
# INVENTORY SETTINGS
# ==================================================
SERVICE_LEVEL_Z = 1.645  # 95% service level
REVIEW_PERIOD_DAYS = 7  # Review inventory every 7 days
DEFAULT_LEAD_TIME_DAYS = 7  # Assumed supplier lead time


# ==================================================
# MAIN FUNCTION
# ==================================================
def main():

    print("Loading sales dataset...")
    print(f"Dataset: {DATA_PATH}")

    # --------------------------------------------------
    # Load and validate sales data
    # --------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    required_sales_columns = ["date", "store", "item", "sales"]
    missing_columns = [col for col in required_sales_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "store", "item", "sales"])

    df = df.sort_values(["store", "item", "date"])

    # --------------------------------------------------
    # Calculate recent demand statistics
    # Use the most recent 28 days
    # --------------------------------------------------
    latest_date = df["date"].max()
    start_date = latest_date - pd.Timedelta(days=27)

    recent_data = df[(df["date"] >= start_date) & (df["date"] <= latest_date)]

    demand_stats = (
        recent_data.groupby(["store", "item"])["sales"]
        .agg(avg_daily_demand="mean", std_daily_demand="std")
        .reset_index()
    )

    demand_stats["std_daily_demand"] = demand_stats["std_daily_demand"].fillna(0)

    # --------------------------------------------------
    # Load current stock information
    # If unavailable, create demo inventory values
    # --------------------------------------------------
    if STOCK_PATH.exists():

        print("Loading current inventory data...")
        stock = pd.read_csv(STOCK_PATH)

    else:

        print("\ncurrent_stock.csv not found." "\nCreating DEMO inventory values.")

        stock = demand_stats[["store", "item"]].copy()

        # Demo stock = approximately 10 days of average demand
        stock["current_stock"] = np.ceil(demand_stats["avg_daily_demand"] * 10).astype(
            int
        )

        stock["lead_time_days"] = DEFAULT_LEAD_TIME_DAYS
        stock["on_order"] = 0
        stock["backorders"] = 0

        stock.to_csv(STOCK_PATH, index=False)

        print(f"Demo stock file created: {STOCK_PATH}")

    # --------------------------------------------------
    # Validate inventory columns
    # --------------------------------------------------
    required_stock_columns = [
        "store",
        "item",
        "current_stock",
        "lead_time_days",
        "on_order",
        "backorders",
    ]

    missing_stock_columns = [
        col for col in required_stock_columns if col not in stock.columns
    ]

    if missing_stock_columns:
        raise ValueError(
            "current_stock.csv is missing columns: " f"{missing_stock_columns}"
        )

    # --------------------------------------------------
    # Merge demand statistics and stock information
    # --------------------------------------------------
    inventory = demand_stats.merge(stock, on=["store", "item"], how="left")

    # Fill missing inventory values
    inventory["current_stock"] = inventory["current_stock"].fillna(0)

    inventory["lead_time_days"] = inventory["lead_time_days"].fillna(
        DEFAULT_LEAD_TIME_DAYS
    )

    inventory["on_order"] = inventory["on_order"].fillna(0)

    inventory["backorders"] = inventory["backorders"].fillna(0)

    # Ensure valid numeric values
    numeric_columns = [
        "current_stock",
        "lead_time_days",
        "on_order",
        "backorders",
        "avg_daily_demand",
        "std_daily_demand",
    ]

    for col in numeric_columns:
        inventory[col] = pd.to_numeric(inventory[col], errors="coerce").fillna(0)

    inventory["lead_time_days"] = inventory["lead_time_days"].clip(lower=1)

    # --------------------------------------------------
    # Inventory calculations
    # --------------------------------------------------

    # Expected demand during supplier lead time
    inventory["lead_time_demand"] = (
        inventory["avg_daily_demand"] * inventory["lead_time_days"]
    )

    # Safety stock based on demand variability
    inventory["safety_stock"] = (
        SERVICE_LEVEL_Z
        * inventory["std_daily_demand"]
        * np.sqrt(inventory["lead_time_days"])
    )

    # Reorder point
    inventory["reorder_point"] = (
        inventory["lead_time_demand"] + inventory["safety_stock"]
    )

    # Target stock for lead time + review period
    inventory["target_stock"] = (
        inventory["avg_daily_demand"]
        * (inventory["lead_time_days"] + REVIEW_PERIOD_DAYS)
        + inventory["safety_stock"]
    )

    # Inventory position includes incoming stock and backorders
    inventory["inventory_position"] = (
        inventory["current_stock"] + inventory["on_order"] - inventory["backorders"]
    )

    # Recommended quantity to order
    inventory["recommended_order_qty"] = np.ceil(
        np.maximum(0, inventory["target_stock"] - inventory["inventory_position"])
    ).astype(int)

    # --------------------------------------------------
    # Stockout risk classification
    # --------------------------------------------------
    inventory["stockout_risk"] = np.select(
        [
            inventory["inventory_position"] <= inventory["lead_time_demand"],
            inventory["inventory_position"] <= inventory["reorder_point"],
        ],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )

    # --------------------------------------------------
    # Reorder recommendation
    # --------------------------------------------------
    inventory["recommendation"] = np.where(
        inventory["recommended_order_qty"] > 0, "REORDER", "SUFFICIENT"
    )

    # --------------------------------------------------
    # Round values for readability
    # --------------------------------------------------
    columns_to_round = [
        "avg_daily_demand",
        "std_daily_demand",
        "lead_time_demand",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "inventory_position",
    ]

    inventory[columns_to_round] = inventory[columns_to_round].round(2)

    # --------------------------------------------------
    # Save output
    # --------------------------------------------------
    inventory.to_csv(OUTPUT_PATH, index=False)

    # --------------------------------------------------
    # Display summary
    # --------------------------------------------------
    print("\n======================================")
    print(" INVENTORY OPTIMIZATION COMPLETED")
    print("======================================")

    print(f"Sales dataset: {DATA_PATH}")
    print(f"Recommendations saved to: {OUTPUT_PATH}")

    print("\nStock Risk Summary:")
    print(inventory["stockout_risk"].value_counts())

    print("\nReorder Summary:")
    print(inventory["recommendation"].value_counts())

    print("\nTop 15 Reorder Recommendations:")

    reorder_items = inventory[inventory["recommended_order_qty"] > 0].sort_values(
        ["stockout_risk", "recommended_order_qty"], ascending=[True, False]
    )

    display_columns = [
        "store",
        "item",
        "current_stock",
        "inventory_position",
        "reorder_point",
        "recommended_order_qty",
        "stockout_risk",
        "recommendation",
    ]

    print(reorder_items[display_columns].head(15).to_string(index=False))


# ==================================================
# RUN SCRIPT
# ==================================================
if __name__ == "__main__":
    main()
