import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import IsolationForest


def render_anomaly_detection(data=None):

    st.title("🔍 Inventory Anomaly Detection")

    st.write(
        "Identify unusual inventory movements, abnormal sales patterns, "
        "and potential stock discrepancies using machine learning."
    )

    # -----------------------------------------
    # LOAD DATA
    # -----------------------------------------
    if data is None:
        uploaded_file = st.file_uploader(
            "Upload inventory or sales data", type=["csv"], key="anomaly_upload"
        )

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Unable to read file: {e}")
                return
        else:
            default_path = Path("data/current_stock.csv")

            if default_path.exists():
                data = pd.read_csv(default_path)
            else:
                st.info("Upload an inventory or sales CSV file to begin.")
                return

    if data is None or data.empty:
        st.warning("The dataset is empty.")
        return

    df = data.copy()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # -----------------------------------------
    # IDENTIFY NUMERIC COLUMNS
    # -----------------------------------------
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_columns:
        st.error(
            "No numeric columns found. Include sales, quantity, "
            "stock, price, or other numerical inventory data."
        )
        return

    # -----------------------------------------
    # USER CONFIGURATION
    # -----------------------------------------
    st.subheader("Configure Detection")

    default_features = []

    preferred_names = [
        "sales",
        "quantity",
        "qty",
        "stock",
        "inventory",
        "units",
        "price",
        "demand",
    ]

    for col in numeric_columns:
        if any(name in col.lower() for name in preferred_names):
            default_features.append(col)

    if not default_features:
        default_features = numeric_columns[: min(3, len(numeric_columns))]

    selected_features = st.multiselect(
        "Select numerical features for anomaly detection",
        options=numeric_columns,
        default=default_features,
    )

    contamination = st.slider(
        "Expected anomaly percentage",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        help="This controls the approximate proportion of records flagged as unusual.",
    )

    if len(selected_features) == 0:
        st.warning("Select at least one numerical feature.")
        return

    # -----------------------------------------
    # PREPARE DATA
    # -----------------------------------------
    working_df = df.copy()

    features = working_df[selected_features].copy()

    # Handle missing and infinite values
    features = features.replace([np.inf, -np.inf], np.nan)

    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")

        median_value = features[col].median()

        if pd.isna(median_value):
            median_value = 0

        features[col] = features[col].fillna(median_value)

    # Remove constant columns
    features = features.loc[:, features.nunique() > 1]

    if features.empty:
        st.error(
            "Selected features do not contain enough variation " "to detect anomalies."
        )
        return

    # -----------------------------------------
    # RUN ISOLATION FOREST
    # -----------------------------------------
    if st.button("Detect Inventory Anomalies", type="primary"):

        with st.spinner("Analyzing inventory patterns..."):

            model = IsolationForest(
                n_estimators=200, contamination=contamination, random_state=42
            )

            predictions = model.fit_predict(features)

            scores = model.decision_function(features)

            working_df["Anomaly_Score"] = scores

            working_df["Anomaly_Status"] = np.where(
                predictions == -1, "Anomaly", "Normal"
            )

            # -----------------------------------------
            # SUMMARY METRICS
            # -----------------------------------------
            total_records = len(working_df)

            anomaly_count = int((working_df["Anomaly_Status"] == "Anomaly").sum())

            normal_count = total_records - anomaly_count

            anomaly_percentage = (
                anomaly_count / total_records * 100 if total_records > 0 else 0
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Records", f"{total_records:,}")

            col2.metric("Anomalies Detected", f"{anomaly_count:,}")

            col3.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")

            # -----------------------------------------
            # VISUALIZATION
            # -----------------------------------------
            st.subheader("Anomaly Distribution")

            distribution = working_df["Anomaly_Status"].value_counts()

            st.bar_chart(distribution)

            # -----------------------------------------
            # ANOMALY RESULTS
            # -----------------------------------------
            st.subheader("Detected Anomalies")

            anomalies = working_df[
                working_df["Anomaly_Status"] == "Anomaly"
            ].sort_values(by="Anomaly_Score", ascending=True)

            if anomalies.empty:
                st.success("No anomalies were detected.")
            else:
                st.warning(f"{len(anomalies)} unusual records were identified.")

                st.dataframe(anomalies, use_container_width=True)

            # -----------------------------------------
            # COMPLETE RESULTS
            # -----------------------------------------
            st.subheader("Complete Analysis")

            st.dataframe(
                working_df.sort_values(by="Anomaly_Score", ascending=True),
                use_container_width=True,
            )

            # -----------------------------------------
            # DOWNLOAD RESULTS
            # -----------------------------------------
            csv_data = working_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Anomaly Report",
                data=csv_data,
                file_name="inventory_anomaly_report.csv",
                mime="text/csv",
            )

            st.caption(
                "Anomaly scores closer to or below zero indicate "
                "more unusual records. Review flagged records before "
                "taking inventory actions."
            )
