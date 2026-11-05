import os
import sys
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.anomaly import render_anomaly_detection

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="IntelliStock | AI Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_PATHS = [
    DATA_DIR / "train (1).csv",
    DATA_DIR / "train.csv",
    BASE_DIR / "train (1).csv",
    BASE_DIR / "train.csv",
]

VALIDATION_PATHS = [
    BASE_DIR / "validation_predictions.csv",
    DATA_DIR / "validation_predictions.csv",
]

INVENTORY_PATHS = [
    BASE_DIR / "inventory_recommendations.csv",
    DATA_DIR / "inventory_recommendations.csv",
]

# ---------------------------------------------------------
# OPTIONAL PROJECT MODULES
# ---------------------------------------------------------
try:
    from src.forecast import load_forecasting_data, forecast_item
except Exception:
    load_forecasting_data = None
    forecast_item = None

try:
    from src.explainability import load_explainability_data, explain_item
except Exception:
    load_explainability_data = None
    explain_item = None

try:
    from src.uncertainty import add_uncertainty_bounds
except Exception:
    add_uncertainty_bounds = None


# ---------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------
def find_existing_file(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def get_dataframe_from_result(result):
    """Extract a DataFrame from a DataFrame, tuple, list, or dict."""
    if isinstance(result, pd.DataFrame):
        return result

    if isinstance(result, dict):
        for value in result.values():
            if isinstance(value, pd.DataFrame):
                return value

    if isinstance(result, (tuple, list)):
        for value in result:
            if isinstance(value, pd.DataFrame):
                return value

    return None


@st.cache_data
def load_sales_data():
    file_path = find_existing_file(DATA_PATHS)

    if file_path is None:
        return None

    try:
        df = pd.read_csv(file_path)
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df
    except Exception as e:
        st.error(f"Unable to load sales data: {e}")
        return None


@st.cache_data
def load_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception:
        return None


def load_validation_data():
    path = find_existing_file(VALIDATION_PATHS)
    if path is None:
        return None
    return load_csv_file(path)


def load_inventory_data():
    path = find_existing_file(INVENTORY_PATHS)
    if path is None:
        return None
    return load_csv_file(path)


def get_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def format_number(value, decimals=2):
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "N/A"


def apply_filters(df):
    """Sidebar filters for store, item, and date."""
    filtered = df.copy()

    with st.sidebar:
        st.markdown("---")
        st.subheader("🔎 Data Filters")

        if "store" in filtered.columns:
            stores = sorted(filtered["store"].dropna().unique().tolist())
            selected_stores = st.multiselect(
                "Stores",
                stores,
                default=stores,
            )
            filtered = filtered[filtered["store"].isin(selected_stores)]

        if "item" in filtered.columns:
            items = sorted(filtered["item"].dropna().unique().tolist())
            selected_items = st.multiselect(
                "Items",
                items,
                default=items,
            )
            filtered = filtered[filtered["item"].isin(selected_items)]

        if "date" in filtered.columns:
            valid_dates = filtered["date"].dropna()

            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()

                selected_dates = st.date_input(
                    "Date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )

                if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                    start_date, end_date = selected_dates
                    filtered = filtered[
                        (filtered["date"].dt.date >= start_date)
                        & (filtered["date"].dt.date <= end_date)
                    ]

    return filtered


def add_uncertainty_if_available(forecast_df):
    if add_uncertainty_bounds is None:
        return forecast_df

    try:
        result = add_uncertainty_bounds(forecast_df)
        extracted = get_dataframe_from_result(result)
        return extracted if extracted is not None else forecast_df
    except Exception:
        return forecast_df


def call_function_safely(func, **kwargs):
    """Pass only arguments supported by the target function."""
    if func is None:
        return None

    try:
        signature = inspect.signature(func)
        accepted = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        return func(**accepted)
    except Exception:
        return None


def normalize_forecast_result(result):
    """Convert common forecast return formats into a DataFrame."""
    df = get_dataframe_from_result(result)

    if df is None and isinstance(result, (list, np.ndarray)):
        try:
            df = pd.DataFrame(result)
        except Exception:
            return None

    if df is None:
        return None

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    date_col = get_column(df, ["date", "ds", "forecast_date"])
    prediction_col = get_column(
        df,
        ["predicted_sales", "prediction", "predicted", "forecast", "yhat"],
    )

    if date_col and date_col != "date":
        df = df.rename(columns={date_col: "date"})

    if prediction_col and prediction_col != "predicted_sales":
        df = df.rename(columns={prediction_col: "predicted_sales"})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "predicted_sales" in df.columns:
        df["predicted_sales"] = safe_numeric(df["predicted_sales"])

    return df


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
sales_df = load_sales_data()

if sales_df is None:
    st.error(
        "Sales dataset not found. Please place your dataset in "
        "`data/train (1).csv` or `data/train.csv`."
    )
    st.stop()

if "sales" not in sales_df.columns:
    st.error("The sales dataset must contain a `sales` column.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("📦 IntelliStock")
st.sidebar.caption("AI Demand Forecasting & Inventory Optimization")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Sales Analysis",
        "Forecast Evaluation",
        "Inventory Optimization",
        "Demand Forecast",
        "Model Explainability",
        "Forecast-Driven Reorder",
        "Inventory Anomaly Detection",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("IntelliStock | Smart Inventory Intelligence")


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
if page == "Overview":
    st.title("📦 IntelliStock Dashboard")
    st.write(
        "AI-powered demand forecasting and inventory planning "
        "for smarter stock management."
    )

    filtered_df = apply_filters(sales_df)

    total_sales = filtered_df["sales"].sum()
    average_daily_sales = filtered_df["sales"].mean()
    total_records = len(filtered_df)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Units Sold", format_number(total_sales, 0))
    col2.metric("Average Sales", format_number(average_daily_sales))
    col3.metric("Records", f"{total_records:,}")

    st.markdown("### Sales Trend")

    if "date" in filtered_df.columns:
        daily_sales = (
            filtered_df.groupby("date", as_index=False)["sales"]
            .sum()
            .sort_values("date")
        )

        fig = px.line(
            daily_sales,
            x="date",
            y="sales",
            title="Daily Sales Trend",
            markers=False,
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Units Sold")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    if "store" in filtered_df.columns:
        store_sales = (
            filtered_df.groupby("store", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
        )

        with col1:
            fig = px.bar(
                store_sales,
                x="store",
                y="sales",
                title="Sales by Store",
            )
            st.plotly_chart(fig, use_container_width=True)

    if "item" in filtered_df.columns:
        item_sales = (
            filtered_df.groupby("item", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
            .head(15)
        )

        with col2:
            fig = px.bar(
                item_sales,
                x="item",
                y="sales",
                title="Top 15 Items by Sales",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Preview")
    st.dataframe(filtered_df.head(100), use_container_width=True)


# =========================================================
# PAGE 2: SALES ANALYSIS
# =========================================================
elif page == "Sales Analysis":
    st.title("📊 Sales Analysis")
    st.write("Explore historical sales patterns by date, store, and item.")

    filtered_df = apply_filters(sales_df)

    if "date" in filtered_df.columns:
        daily_sales = (
            filtered_df.groupby("date", as_index=False)["sales"]
            .sum()
            .sort_values("date")
        )

        fig = px.area(
            daily_sales,
            x="date",
            y="sales",
            title="Historical Daily Sales",
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    if "store" in filtered_df.columns:
        store_sales = (
            filtered_df.groupby("store", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
        )

        with col1:
            fig = px.bar(
                store_sales,
                x="store",
                y="sales",
                title="Store Performance",
                color="sales",
            )
            st.plotly_chart(fig, use_container_width=True)

    if "item" in filtered_df.columns:
        item_sales = (
            filtered_df.groupby("item", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
        )

        with col2:
            fig = px.bar(
                item_sales.head(20),
                x="item",
                y="sales",
                title="Top 20 Items",
                color="sales",
            )
            st.plotly_chart(fig, use_container_width=True)

    if "date" in filtered_df.columns:
        filtered_df["month"] = filtered_df["date"].dt.month
        monthly_sales = filtered_df.groupby("month", as_index=False)["sales"].sum()

        fig = px.line(
            monthly_sales,
            x="month",
            y="sales",
            markers=True,
            title="Monthly Sales Pattern",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sales Data")
    st.dataframe(filtered_df, use_container_width=True)


# =========================================================
# PAGE 3: FORECAST EVALUATION
# =========================================================
elif page == "Forecast Evaluation":
    st.title("📈 Forecast Evaluation")
    st.write("Evaluate the forecasting model using validation predictions.")

    validation = load_validation_data()

    if validation is None:
        st.warning(
            "Validation file not found. Expected "
            "`validation_predictions.csv` in the project root or data folder."
        )
        st.stop()

    validation.columns = [str(c).strip().lower() for c in validation.columns]

    # FIX: recognizes actual_sales and predicted_sales
    actual_col = get_column(
        validation,
        [
            "actual",
            "actual_sales",
            "sales",
            "y_true",
            "true_sales",
        ],
    )

    predicted_col = get_column(
        validation,
        [
            "predicted_sales",
            "prediction",
            "predicted",
            "y_pred",
            "forecast",
        ],
    )

    if actual_col is None or predicted_col is None:
        st.error(
            "Could not identify actual and predicted columns. "
            f"Available columns: {validation.columns.tolist()}"
        )
        st.stop()

    validation[actual_col] = safe_numeric(validation[actual_col])
    validation[predicted_col] = safe_numeric(validation[predicted_col])

    evaluation_df = validation.dropna(subset=[actual_col, predicted_col]).copy()

    if evaluation_df.empty:
        st.error("No valid numeric actual/predicted values found.")
        st.stop()

    actual = evaluation_df[actual_col]
    predicted = evaluation_df[predicted_col]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    total_actual = actual.abs().sum()
    wape = (
        (actual - predicted).abs().sum() / total_actual * 100
        if total_actual != 0
        else np.nan
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", format_number(mae))
    col2.metric("RMSE", format_number(rmse))
    col3.metric("WAPE", f"{format_number(wape)}%")

    st.markdown("### Actual vs Predicted Sales")

    plot_df = evaluation_df.copy()

    date_col = get_column(plot_df, ["date", "ds", "forecast_date"])

    if date_col:
        plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors="coerce")
        plot_df = plot_df.sort_values(date_col)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_df[date_col],
                y=plot_df[actual_col],
                mode="lines",
                name="Actual Sales",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df[date_col],
                y=plot_df[predicted_col],
                mode="lines",
                name="Predicted Sales",
            )
        )
        fig.update_layout(
            title="Actual vs Predicted Sales Over Time",
            xaxis_title="Date",
            yaxis_title="Sales",
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(plot_df)),
                y=plot_df[actual_col],
                mode="lines",
                name="Actual Sales",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(plot_df)),
                y=plot_df[predicted_col],
                mode="lines",
                name="Predicted Sales",
            )
        )
        fig.update_layout(
            title="Actual vs Predicted Sales",
            xaxis_title="Observation",
            yaxis_title="Sales",
        )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Validation Data")
    st.dataframe(evaluation_df, use_container_width=True)

    st.download_button(
        "Download Validation Results",
        data=evaluation_df.to_csv(index=False).encode("utf-8"),
        file_name="validation_evaluation.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE 4: INVENTORY OPTIMIZATION
# =========================================================
elif page == "Inventory Optimization":
    st.title("📦 Smart Inventory Reorder Recommendations")
    st.write(
        "Estimate replenishment needs from historical demand, adjust current stock, "
        "and identify items that need reordering."
    )

    if (
        "store" not in sales_df.columns
        or "item" not in sales_df.columns
        or "sales" not in sales_df.columns
    ):
        st.error("Dataset needs store, item, and sales columns for inventory planning.")
        st.stop()

    filtered_df = apply_filters(sales_df)
    if filtered_df.empty:
        st.warning("No sales data available for the selected filters.")
        st.stop()

    st.subheader("Replenishment Settings")
    p1, p2, p3 = st.columns(3)
    with p1:
        lead_time = st.number_input(
            "Supplier Lead Time (days)", min_value=1, max_value=90, value=7
        )
    with p2:
        safety_days = st.number_input(
            "Safety Stock (days)", min_value=0, max_value=60, value=3
        )
    with p3:
        review_days = st.number_input(
            "Order Review Period (days)", min_value=1, max_value=60, value=7
        )

    demand = (
        filtered_df.groupby(["store", "item"])["sales"]
        .agg(average_daily_demand="mean", demand_std="std")
        .reset_index()
    )
    demand["demand_std"] = demand["demand_std"].fillna(0)
    demand["reorder_point"] = np.ceil(
        demand["average_daily_demand"] * lead_time
        + demand["demand_std"] * np.sqrt(lead_time)
    ).astype(int)
    demand["target_stock"] = np.ceil(
        demand["average_daily_demand"] * (lead_time + safety_days + review_days)
    ).astype(int)

    # Merge optional inventory file values when available.
    inventory_df = load_inventory_data()
    if inventory_df is not None and not inventory_df.empty:
        inv = inventory_df.copy()
        inv.columns = [str(c).strip().lower() for c in inv.columns]
        inv_store = get_column(inv, ["store", "store_id"])
        inv_item = get_column(inv, ["item", "item_id"])
        inv_stock = get_column(
            inv, ["current_stock", "stock", "available_stock", "on_hand"]
        )
        if inv_store and inv_item and inv_stock:
            inv = inv[[inv_store, inv_item, inv_stock]].rename(
                columns={
                    inv_store: "store",
                    inv_item: "item",
                    inv_stock: "current_stock",
                }
            )
            demand = demand.merge(inv, on=["store", "item"], how="left")

    if "current_stock" not in demand.columns:
        demand["current_stock"] = 0
    demand["current_stock"] = pd.to_numeric(
        demand["current_stock"], errors="coerce"
    ).fillna(0)

    st.caption(
        "Edit the Current Stock column below. If no inventory file is available, stock starts at 0."
    )
    editable_cols = [
        "store",
        "item",
        "average_daily_demand",
        "demand_std",
        "current_stock",
        "reorder_point",
        "target_stock",
    ]
    editable = demand[editable_cols].copy()
    edited = st.data_editor(
        editable,
        use_container_width=True,
        hide_index=True,
        column_config={
            "store": st.column_config.NumberColumn("Store", disabled=True),
            "item": st.column_config.NumberColumn("Item", disabled=True),
            "average_daily_demand": st.column_config.NumberColumn(
                "Avg Daily Demand", format="%.2f", disabled=True
            ),
            "demand_std": st.column_config.NumberColumn(
                "Demand Std. Dev.", format="%.2f", disabled=True
            ),
            "current_stock": st.column_config.NumberColumn(
                "Current Stock", min_value=0, step=1
            ),
            "reorder_point": st.column_config.NumberColumn(
                "Reorder Point", disabled=True
            ),
            "target_stock": st.column_config.NumberColumn(
                "Target Stock", disabled=True
            ),
        },
        key="inventory_stock_editor",
    )

    edited["current_stock"] = (
        pd.to_numeric(edited["current_stock"], errors="coerce").fillna(0).clip(lower=0)
    )
    edited["recommended_order"] = np.ceil(
        (edited["target_stock"] - edited["current_stock"]).clip(lower=0)
    ).astype(int)
    edited["status"] = np.where(
        edited["current_stock"] <= edited["reorder_point"],
        "Reorder Now",
        "Stock Sufficient",
    )

    low_stock = edited[edited["status"] == "Reorder Now"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Items Requiring Reorder", f"{len(low_stock):,}")
    k2.metric("Total Current Stock", f"{edited['current_stock'].sum():,.0f}")
    k3.metric("Suggested Units to Order", f"{edited['recommended_order'].sum():,.0f}")

    st.subheader("Reorder Action List")
    action_list = edited.sort_values(
        ["status", "recommended_order"], ascending=[True, False]
    )
    st.dataframe(action_list, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Smart Inventory Plan (CSV)",
        data=action_list.to_csv(index=False).encode("utf-8"),
        file_name="smart_inventory_recommendations.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE 5: DEMAND FORECAST
# =========================================================
elif page == "Demand Forecast":
    st.title("🔮 Demand Forecast")
    st.write(
        "Generate future demand predictions for a selected store and item "
        "using the trained XGBoost model."
    )

    if load_forecasting_data is None or forecast_item is None:
        st.error(
            "Forecast module could not be imported. "
            "Check `src/forecast.py` and its dependencies."
        )
        st.stop()

    try:
        # Your src/forecast.py returns (historical_dataframe, trained_model)
        forecast_data, forecast_model = load_forecasting_data()

        if forecast_data is None or forecast_data.empty:
            st.error("The forecasting dataset is empty.")
            st.stop()

        forecast_data.columns = [
            str(column).strip().lower() for column in forecast_data.columns
        ]

        required_columns = {"store", "item", "date", "sales"}
        missing_columns = required_columns - set(forecast_data.columns)

        if missing_columns:
            st.error(
                "Forecast data is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
            st.stop()

        stores = sorted(forecast_data["store"].dropna().unique().tolist())

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_store = st.selectbox(
                "Select Store",
                stores,
                key="demand_forecast_store",
            )

        available_items = sorted(
            forecast_data.loc[forecast_data["store"] == selected_store, "item"]
            .dropna()
            .unique()
            .tolist()
        )

        with col2:
            selected_item = st.selectbox(
                "Select Item",
                available_items,
                key="demand_forecast_item",
            )

        with col3:
            forecast_days = st.number_input(
                "Forecast Horizon (days)",
                min_value=1,
                max_value=30,
                value=14,
                step=1,
            )

        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating demand forecast..."):
                # Correct signature:
                # forecast_item(df, model, store, item, horizon=14)
                forecast_result = forecast_item(
                    df=forecast_data,
                    model=forecast_model,
                    store=selected_store,
                    item=selected_item,
                    horizon=int(forecast_days),
                )

            if forecast_result is None or forecast_result.empty:
                st.warning("No forecast results were generated.")
            else:
                forecast_result = forecast_result.copy()
                forecast_result.columns = [
                    str(column).strip().lower() for column in forecast_result.columns
                ]

                if "date" in forecast_result.columns:
                    forecast_result["date"] = pd.to_datetime(
                        forecast_result["date"], errors="coerce"
                    )

                if "predicted_sales" not in forecast_result.columns:
                    st.error(
                        "The forecast output does not contain " "`predicted_sales`."
                    )
                    st.write("Returned columns:", forecast_result.columns.tolist())
                    st.stop()

                forecast_result["predicted_sales"] = pd.to_numeric(
                    forecast_result["predicted_sales"], errors="coerce"
                )
                forecast_result = forecast_result.dropna(subset=["predicted_sales"])

                if forecast_result.empty:
                    st.warning("The forecast contains no valid predictions.")
                else:
                    st.success("Forecast generated successfully.")

                    total_forecast = forecast_result["predicted_sales"].sum()
                    average_forecast = forecast_result["predicted_sales"].mean()
                    peak_row = forecast_result.loc[
                        forecast_result["predicted_sales"].idxmax()
                    ]

                    metric1, metric2, metric3 = st.columns(3)

                    metric1.metric(
                        "Total Forecast Demand",
                        format_number(total_forecast, 0),
                    )
                    metric2.metric(
                        "Average Daily Demand",
                        format_number(average_forecast),
                    )

                    peak_date = (
                        peak_row["date"].strftime("%d %b %Y")
                        if "date" in forecast_result.columns
                        and pd.notna(peak_row["date"])
                        else "N/A"
                    )

                    metric3.metric(
                        "Peak Forecast",
                        format_number(peak_row["predicted_sales"], 0),
                        peak_date,
                    )

                    st.subheader("Forecast Trend")

                    if "date" in forecast_result.columns:
                        chart_df = forecast_result.set_index("date")
                        st.line_chart(
                            chart_df["predicted_sales"],
                            use_container_width=True,
                        )
                    else:
                        st.line_chart(
                            forecast_result["predicted_sales"],
                            use_container_width=True,
                        )

                    st.subheader("Forecast Results")
                    st.dataframe(
                        forecast_result,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download Forecast CSV",
                        data=forecast_result.to_csv(index=False).encode("utf-8"),
                        file_name=(
                            f"forecast_store_{selected_store}_"
                            f"item_{selected_item}.csv"
                        ),
                        mime="text/csv",
                    )

    except Exception as error:
        st.error(f"Forecast could not be generated: {error}")


# =========================================================
# PAGE 6: MODEL EXPLAINABILITY
# =========================================================
elif page == "Model Explainability":
    st.title("🧠 Model Explainability")
    st.write("Understand which features influence the demand forecasting model.")

    if explain_item is None:
        st.error(
            "Explainability module could not be imported. "
            "Check `src/explainability.py` and its dependencies."
        )
        st.stop()

    stores = (
        sorted(sales_df["store"].dropna().unique().tolist())
        if "store" in sales_df.columns
        else []
    )
    items = (
        sorted(sales_df["item"].dropna().unique().tolist())
        if "item" in sales_df.columns
        else []
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_store = st.selectbox(
            "Store",
            stores,
            key="explain_store",
        )

    with col2:
        selected_item = st.selectbox(
            "Item",
            items,
            key="explain_item",
        )

    if st.button("Explain Prediction", type="primary"):
        with st.spinner("Generating explanation..."):
            explain_data = None

            if load_explainability_data is not None:
                try:
                    loaded_result = load_explainability_data()
                    explain_data = get_dataframe_from_result(loaded_result)
                except Exception:
                    explain_data = None

            result = call_function_safely(
                explain_item,
                store=selected_store,
                item=selected_item,
                data=explain_data,
                df=explain_data,
            )

            explanation_df = get_dataframe_from_result(result)

            if explanation_df is not None:
                explanation_df.columns = [
                    str(c).strip().lower() for c in explanation_df.columns
                ]

                st.dataframe(
                    explanation_df,
                    use_container_width=True,
                )

                feature_col = get_column(
                    explanation_df,
                    ["feature", "features", "name"],
                )
                importance_col = get_column(
                    explanation_df,
                    ["importance", "shap_value", "contribution", "value"],
                )

                if feature_col and importance_col:
                    explanation_df[importance_col] = safe_numeric(
                        explanation_df[importance_col]
                    )

                    explanation_df = explanation_df.dropna(subset=[importance_col])

                    explanation_df = explanation_df.sort_values(
                        importance_col,
                        key=lambda x: x.abs(),
                        ascending=True,
                    )

                    fig = px.bar(
                        explanation_df,
                        x=importance_col,
                        y=feature_col,
                        orientation="h",
                        title="Feature Importance / SHAP Contributions",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif result is not None:
                st.write(result)
            else:
                st.error(
                    "Could not generate an explanation. "
                    "Check the function signature in `src/explainability.py`."
                )


# =========================================================
# PAGE: FORECAST-DRIVEN REORDER RECOMMENDATIONS
# =========================================================
elif page == "Forecast-Driven Reorder":
    st.title("🧠 Forecast-Driven Reorder Recommendations")
    st.write(
        "Use the trained demand model to estimate stock needs over the "
        "supplier lead time and recommend replenishment."
    )

    if load_forecasting_data is None or forecast_item is None:
        st.error("Forecast module is unavailable. Check src/forecast.py.")
        st.stop()

    try:
        forecast_data, forecast_model = load_forecasting_data()
        if forecast_data is None or forecast_data.empty:
            st.error("Forecasting dataset is empty.")
            st.stop()

        forecast_data.columns = [str(c).strip().lower() for c in forecast_data.columns]
        required = {"store", "item", "date", "sales"}
        missing = required - set(forecast_data.columns)
        if missing:
            st.error("Forecast data is missing: " + ", ".join(sorted(missing)))
            st.stop()

        c1, c2 = st.columns(2)
        stores = sorted(forecast_data["store"].dropna().unique().tolist())
        with c1:
            reorder_store = st.selectbox("Store", stores, key="reorder_forecast_store")

        items = sorted(
            forecast_data.loc[forecast_data["store"] == reorder_store, "item"]
            .dropna()
            .unique()
            .tolist()
        )
        with c2:
            reorder_item = st.selectbox("Item", items, key="reorder_forecast_item")

        c3, c4, c5 = st.columns(3)
        with c3:
            supplier_lead = st.number_input(
                "Supplier Lead Time (days)",
                min_value=1,
                max_value=30,
                value=7,
                key="forecast_reorder_lead",
            )
        with c4:
            safety_days_input = st.number_input(
                "Safety Buffer (days)",
                min_value=0,
                max_value=30,
                value=3,
                key="forecast_reorder_safety",
            )
        with c5:
            review_period = st.number_input(
                "Review Period (days)",
                min_value=1,
                max_value=30,
                value=7,
                key="forecast_reorder_review",
            )

        current_stock = st.number_input(
            "Current Available Stock (units)",
            min_value=0,
            value=0,
            step=1,
            key="forecast_reorder_stock",
        )
        forecast_horizon = int(supplier_lead + safety_days_input + review_period)

        if st.button("Calculate Reorder Recommendation", type="primary"):
            with st.spinner("Forecasting demand and calculating stock needs..."):
                forecast_result = forecast_item(
                    df=forecast_data,
                    model=forecast_model,
                    store=reorder_store,
                    item=reorder_item,
                    horizon=forecast_horizon,
                )

            if forecast_result is None or forecast_result.empty:
                st.warning("No forecast was generated for this store/item.")
            else:
                forecast_result = forecast_result.copy()
                forecast_result.columns = [
                    str(c).strip().lower() for c in forecast_result.columns
                ]
                pred_col = get_column(
                    forecast_result,
                    ["predicted_sales", "prediction", "predicted", "forecast", "yhat"],
                )
                if pred_col is None:
                    st.error(
                        "Forecast output has no recognized prediction column. "
                        f"Columns returned: {forecast_result.columns.tolist()}"
                    )
                    st.stop()

                forecast_result[pred_col] = pd.to_numeric(
                    forecast_result[pred_col], errors="coerce"
                ).clip(lower=0)
                forecast_result = forecast_result.dropna(subset=[pred_col])
                if forecast_result.empty:
                    st.warning("Forecast output contains no valid numeric predictions.")
                else:
                    lead_demand = (
                        forecast_result[pred_col].head(int(supplier_lead)).sum()
                    )
                    protection_demand = (
                        forecast_result[pred_col].head(forecast_horizon).sum()
                    )
                    safety_stock_units = forecast_result[pred_col].head(
                        int(supplier_lead)
                    ).mean() * float(safety_days_input)
                    reorder_point = int(np.ceil(lead_demand + safety_stock_units))
                    target_stock = int(np.ceil(protection_demand))
                    recommended_order = int(
                        np.ceil(max(0, target_stock - float(current_stock)))
                    )
                    stockout_days = None
                    cumulative = forecast_result[pred_col].cumsum()
                    stockout_positions = np.flatnonzero(
                        cumulative.to_numpy() > float(current_stock)
                    )
                    if len(stockout_positions):
                        stockout_days = int(stockout_positions[0] + 1)

                    status = (
                        "REORDER NOW"
                        if float(current_stock) <= reorder_point
                        else "STOCK SUFFICIENT"
                    )
                    priority = (
                        "Critical"
                        if stockout_days is not None
                        and stockout_days <= int(supplier_lead)
                        else ("High" if status == "REORDER NOW" else "Normal")
                    )

                    st.subheader("Recommendation")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Status", status)
                    m2.metric("Reorder Point", f"{reorder_point:,} units")
                    m3.metric("Recommended Order", f"{recommended_order:,} units")
                    m4.metric(
                        "Estimated Stockout",
                        (
                            f"In {stockout_days} days"
                            if stockout_days
                            else "Not within forecast"
                        ),
                    )
                    st.write(f"**Priority:** {priority}")
                    st.caption(
                        "Recommendation is based on model forecasts for the selected "
                        "store/item and the chosen lead, safety, and review periods."
                    )

                    chart_df = forecast_result.copy()
                    if "date" in chart_df.columns:
                        chart_df["date"] = pd.to_datetime(
                            chart_df["date"], errors="coerce"
                        )
                        fig = px.line(
                            chart_df,
                            x="date",
                            y=pred_col,
                            title="Forecast Demand Used for Replenishment",
                        )
                    else:
                        fig = px.line(
                            chart_df,
                            y=pred_col,
                            title="Forecast Demand Used for Replenishment",
                        )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(forecast_result, use_container_width=True)

                    recommendation_df = pd.DataFrame(
                        [
                            {
                                "store": reorder_store,
                                "item": reorder_item,
                                "current_stock": int(current_stock),
                                "forecast_lead_demand": float(lead_demand),
                                "safety_stock": float(safety_stock_units),
                                "reorder_point": reorder_point,
                                "target_stock": target_stock,
                                "recommended_order": recommended_order,
                                "estimated_stockout_days": stockout_days,
                                "status": status,
                                "priority": priority,
                            }
                        ]
                    )
                    st.download_button(
                        "Download Reorder Recommendation (CSV)",
                        data=recommendation_df.to_csv(index=False).encode("utf-8"),
                        file_name="forecast_driven_reorder.csv",
                        mime="text/csv",
                    )

    except Exception as error:
        st.error(f"Could not calculate forecast-driven reorder: {error}")


# =========================================================
# PAGE 7: INVENTORY ANOMALY DETECTION
# =========================================================
elif page == "Inventory Anomaly Detection":
    render_anomaly_detection(sales_df)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, Python, XGBoost & LSTM")
