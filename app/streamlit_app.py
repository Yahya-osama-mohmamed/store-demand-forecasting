"""
Streamlit Web Dashboard — Interactive UI for the demand forecasting model.

Provides:
- Sidebar form for single (date, store, item) forecasts
- 90-day forecast chart with historical context and SHAP explanation
- Batch prediction via CSV upload
- Global model insights view
"""

import os
import json
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
from sklearn.pipeline import Pipeline

from src.config import (
    FINAL_PIPELINE_PATH, MODEL_METADATA_PATH, RAW_DATA_FILE, FIGURES_DIR,
    N_STORES, N_ITEMS,
)

# =============================================================================
# App Configuration
# =============================================================================
st.set_page_config(
    page_title="Demand Forecaster",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurable via environment so docker-compose / Render can point the
# dashboard at the API container; falls back to local dev default.
API_URL = os.getenv("API_URL", "http://localhost:8000")


# =============================================================================
# Helper Functions
# =============================================================================
# Caches are keyed by artifact mtime: a model trained AFTER the dashboard
# started (or a retrain) is picked up automatically. Naively caching the
# loader itself would permanently pin the "no model yet" state.

@st.cache_resource
def _load_model_cached(mtime: float):
    return joblib.load(FINAL_PIPELINE_PATH)


def load_local_model():
    """Load model directly for charts/SHAP inside Streamlit."""
    try:
        if FINAL_PIPELINE_PATH.exists():
            return _load_model_cached(FINAL_PIPELINE_PATH.stat().st_mtime)
    except Exception as e:
        st.sidebar.error(f"Failed to load local model: {e}")
    return None


@st.cache_resource
def _load_metadata_cached(mtime: float) -> dict:
    return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))


def load_metadata() -> dict:
    """Load model metadata (best model name, test SMAPE, ...)."""
    try:
        if MODEL_METADATA_PATH.exists():
            return _load_metadata_cached(MODEL_METADATA_PATH.stat().st_mtime)
    except Exception:
        pass
    return {}


@st.cache_data
def load_history(store: int, item: int, _mtime: float = 0.0):
    """Historical sales for one store-item series (for chart context)."""
    try:
        if RAW_DATA_FILE.exists():
            df = pd.read_csv(RAW_DATA_FILE, parse_dates=["date"])
            series = df[(df["store"] == store) & (df["item"] == item)]
            return series.sort_values("date")
    except Exception:
        pass
    return None


def demand_level_of(pred: float, baseline: float) -> str:
    if baseline <= 0:
        return "Unknown"
    ratio = pred / baseline
    return "Low" if ratio < 0.8 else ("High" if ratio > 1.2 else "Normal")


def predict_single_api(input_dict: dict):
    """Call the FastAPI endpoint for a single forecast."""
    try:
        response = requests.post(f"{API_URL}/predict", json=input_dict, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # Fallback to local prediction if API is down. The pipeline is
        # self-contained (feature engineering happens inside it).
        pipeline = load_local_model()
        if pipeline:
            df = pd.DataFrame([input_dict])
            pred = float(np.clip(pipeline.predict(df), 0, None)[0])
            try:
                fe = pipeline.named_steps["features"]
                lookup = fe.store_item_mean_
                row = lookup[(lookup["store"] == input_dict["store"]) &
                             (lookup["item"] == input_dict["item"])]
                baseline = float(row["store_item_mean"].iloc[0]) if len(row) else fe.global_mean_
            except Exception:
                baseline = 0.0
            return {
                "predicted_sales": round(pred, 2),
                "demand_level": demand_level_of(pred, baseline),
                "_source": "local_fallback",
            }
        else:
            st.error(f"API Error and no local model available: {e}")
            return None


def render_forecast_chart(store: int, item: int, start_date, horizon_days: int, pipeline):
    """Plot recent history plus a multi-day forecast for one series."""
    future_dates = pd.date_range(start=start_date, periods=horizon_days, freq="D")
    future_df = pd.DataFrame({
        "date": future_dates.strftime("%Y-%m-%d"),
        "store": store,
        "item": item,
    })
    preds = np.clip(pipeline.predict(future_df), 0, None)

    fig = go.Figure()
    history = load_history(store, item)
    if history is not None and len(history):
        recent = history.tail(365)
        fig.add_trace(go.Scatter(
            x=recent["date"], y=recent["sales"],
            mode="lines", name="History (last year)",
            line=dict(color="#90A4AE", width=1),
        ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=preds,
        mode="lines", name="Forecast",
        line=dict(color="#FF5722", width=2),
    ))
    fig.update_layout(
        title=f"Store {store} · Item {item} — {horizon_days}-Day Forecast",
        xaxis_title="Date", yaxis_title="Units sold",
        height=420, margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    return future_df, preds


def render_shap_waterfall(input_dict: dict, pipeline):
    """Render a SHAP waterfall plot for the given forecast."""
    import shap

    st.subheader("Forecast Explainability (SHAP)")
    st.write("How each feature pushed this forecast above or below the model's baseline.")

    try:
        df = pd.DataFrame([input_dict])
        regressor = pipeline.named_steps["regressor"]
        transformer = Pipeline(pipeline.steps[:-1])
        X_trans = np.asarray(transformer.transform(df))
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        elif getattr(shap_values, "ndim", 2) == 3:
            shap_values = shap_values[:, :, 0]

        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = np.atleast_1d(expected)[0]

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=expected,
            data=X_trans[0],
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False, max_display=10)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {e}")


# =============================================================================
# UI Layout
# =============================================================================

st.title("📦 Store Item Demand Forecasting")
metadata = load_metadata()
if metadata:
    st.caption(
        f"Model: **{metadata.get('best_model', '?')}** · "
        f"Test SMAPE: **{metadata.get('test_smape', '?')}** · "
        f"Trained: {str(metadata.get('trained_at_utc', ''))[:10]}"
    )
st.markdown("""
Forecast daily demand for any of the 50 items across 10 stores.
Use the sidebar for a single forecast, or the tabs for multi-day forecasts and batch scoring.
""")

tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📂 Batch Forecast", "📊 Model Insights"])

# --- SIDEBAR: Forecast parameters ---
st.sidebar.header("Forecast Parameters")

with st.sidebar.form("forecast_form"):
    forecast_date = st.date_input(
        "Forecast date",
        value=datetime.date(2018, 1, 15),
        min_value=datetime.date(2013, 1, 1),
        max_value=datetime.date(2020, 12, 31),
    )
    store = st.selectbox("Store", list(range(1, N_STORES + 1)))
    item = st.selectbox("Item", list(range(1, N_ITEMS + 1)))
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=90)
    submit_button = st.form_submit_button("Forecast Demand", type="primary",
                                          use_container_width=True)

input_data = {
    "date": forecast_date.isoformat(),
    "store": int(store),
    "item": int(item),
}


# --- TAB 1: Forecast ---
with tab1:
    if submit_button:
        with st.spinner("Forecasting..."):
            result = predict_single_api(input_data)

            if result:
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric(
                    f"Forecast — {forecast_date.isoformat()}",
                    f"{result['predicted_sales']:.1f} units",
                )
                col_res2.metric("Demand level", result["demand_level"])
                pipeline = load_local_model()
                if pipeline is not None:
                    try:
                        fe = pipeline.named_steps["features"]
                        lookup = fe.store_item_mean_
                        row = lookup[(lookup["store"] == input_data["store"]) &
                                     (lookup["item"] == input_data["item"])]
                        baseline = float(row["store_item_mean"].iloc[0]) if len(row) else fe.global_mean_
                        col_res3.metric("Historical daily average", f"{baseline:.1f} units")
                    except Exception:
                        pass

                if result.get("_source") == "local_fallback":
                    st.caption("ℹ️ Computed locally using fallback model (API unreachable).")

                if pipeline is not None:
                    st.divider()
                    future_df, preds = render_forecast_chart(
                        input_data["store"], input_data["item"],
                        forecast_date, horizon, pipeline,
                    )

                    csv = future_df.assign(predicted_sales=np.round(preds, 2)) \
                                   .to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download forecast as CSV", data=csv,
                        file_name=f"forecast_store{store}_item{item}.csv",
                        mime="text/csv",
                    )

                    st.divider()
                    render_shap_waterfall(input_data, pipeline)
    else:
        st.info("👈 Pick a date, store, and item in the sidebar, then click **Forecast Demand**.")


# --- TAB 2: Batch Forecast ---
with tab2:
    st.subheader("Batch Forecast from CSV")
    st.write("Upload a CSV with `date, store, item` columns to forecast in bulk.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df_upload)} records. Preview:")
            st.dataframe(df_upload.head(3), hide_index=True)

            if st.button("Generate Batch Forecasts", type="primary"):
                with st.spinner("Forecasting batch..."):
                    pipeline = load_local_model()
                    if pipeline:
                        preds = np.clip(pipeline.predict(df_upload), 0, None)

                        results_df = df_upload.copy()
                        results_df["Predicted_Sales"] = np.round(preds, 2)

                        st.success(f"Successfully forecast {len(results_df)} records!")

                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Rows forecast", len(results_df))
                        col_b.metric("Total predicted units", f"{preds.sum():,.0f}")
                        col_c.metric("Mean per row", f"{preds.mean():.1f}")

                        st.subheader("Results")
                        st.dataframe(results_df)

                        csv = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Forecasts as CSV",
                            data=csv,
                            file_name="demand_forecasts.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error("Model not available for local prediction.")
        except Exception as e:
            st.error(f"Error processing file: {e}")


# --- TAB 3: Model Insights ---
with tab3:
    st.subheader("Global Model Insights")
    st.write("Generated during training — overall patterns across all 500 series.")

    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.write("**Feature Importance (SHAP)**")
        shap_bar = FIGURES_DIR / "shap_bar.png"
        if shap_bar.exists():
            st.image(str(shap_bar), use_container_width=True)
        else:
            st.info("SHAP bar plot not found. Run the training pipeline first.")

        st.write("**Test Window — Forecast vs Actual**")
        fva = FIGURES_DIR / "forecast_vs_actual_test.png"
        if fva.exists():
            st.image(str(fva), use_container_width=True)

    with col_img2:
        st.write("**Overall Feature Impact (SHAP Summary)**")
        shap_summary = FIGURES_DIR / "shap_summary.png"
        if shap_summary.exists():
            st.image(str(shap_summary), use_container_width=True)
        else:
            st.info("SHAP summary plot not found.")

        st.write("**Yearly Seasonality**")
        seasonality = FIGURES_DIR / "eda_monthly_seasonality.png"
        if seasonality.exists():
            st.image(str(seasonality), use_container_width=True)
