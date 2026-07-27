"""
Main Pipeline — CLI entry point for the complete ML workflow.

Orchestrates the entire pipeline from data loading to model saving:
1. Download and load data
2. Clean
3. EDA (seasonality, trends, store/item structure)
4. Feature engineering (reference artifact)
5. Chronological train/val/test split
6. Feature-importance analysis
7. Build preprocessing pipeline
8. Train and tune models (LightGBM, XGBoost, HistGradientBoosting)
9. Evaluate and compare (selection by validation SMAPE)
10. SHAP explainability
11. MLflow tracking
12. Save the best model + metadata

Usage:
    python main.py              # Run full pipeline
    python main.py --skip-mlflow  # Skip MLflow tracking
"""

import json
import time
import argparse
import warnings
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress convergence and future warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import (
    RANDOM_STATE, TARGET, DATE_COLUMN, FIGURES_DIR, MODELS_DIR,
    FINAL_PIPELINE_PATH, BEST_MODEL_PATH, FEATURE_NAMES_PATH,
    PREPROCESSING_PIPELINE_PATH, MODEL_METADATA_PATH,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
)
from src.logger import get_logger
from src.data_loader import load_and_validate
from src.preprocessing import clean_data, split_data, get_default_preprocessor
from src.feature_engineering import engineer_features, FeatureEngineer
from src.feature_selection import select_features
from src.model_training import train_all_models
from src.model_evaluation import evaluate_all_models, smape
from src.explainability import run_explainability
from sklearn.pipeline import Pipeline

logger = get_logger(__name__)


# =============================================================================
# EDA Visualizations
# =============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """
    Generate all EDA visualizations and save to figures/.

    Args:
        df: Cleaned DataFrame with date, store, item, sales.
    """
    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    plt.style.use("seaborn-v0_8-whitegrid")
    dates = df[DATE_COLUMN]

    # 1. Total daily sales over time (trend + seasonality)
    daily = df.groupby(DATE_COLUMN)[TARGET].sum()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(daily.index, daily.values, color="#2196F3", linewidth=0.7)
    ax.plot(daily.rolling(30, center=True).mean(), color="#FF5722",
            linewidth=2, label="30-day rolling mean")
    ax.set_title("Total Daily Sales — All Stores & Items", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total sales")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_daily_sales.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Monthly seasonality
    monthly = df.groupby(dates.dt.month)[TARGET].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(monthly.index, monthly.values, color="#2196F3", edgecolor="white")
    ax.set_title("Average Sales by Month — Yearly Seasonality", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean sales per store-item-day")
    ax.set_xticks(range(1, 13))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_monthly_seasonality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Day-of-week pattern
    dow = df.groupby(dates.dt.dayofweek)[TARGET].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = ["#2196F3"] * 5 + ["#FF5722"] * 2
    ax.bar(labels, dow.values, color=colors, edgecolor="white")
    ax.set_title("Average Sales by Day of Week", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean sales per store-item-day")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_dayofweek.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Sales by store
    store_sales = df.groupby("store")[TARGET].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(store_sales.index.astype(str), store_sales.values,
           color="#4CAF50", edgecolor="white")
    ax.set_title("Average Sales by Store", fontsize=13, fontweight="bold")
    ax.set_xlabel("Store")
    ax.set_ylabel("Mean sales per item-day")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_sales_by_store.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Item demand spread (top & bottom 10 items)
    item_sales = df.groupby("item")[TARGET].mean().sort_values(ascending=False)
    top_bottom = pd.concat([item_sales.head(10), item_sales.tail(10)])
    fig, ax = plt.subplots(figsize=(11, 6))
    bar_colors = ["#2196F3"] * 10 + ["#FF5722"] * 10
    ax.bar(top_bottom.index.astype(str), top_bottom.values,
           color=bar_colors, edgecolor="white")
    ax.set_title("Item Demand Spread — Top 10 vs Bottom 10 Items",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Item")
    ax.set_ylabel("Mean sales per store-day")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_item_spread.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6. Year-over-year growth
    yearly = df.groupby(dates.dt.year)[TARGET].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(yearly.index, yearly.values, marker="o", color="#9C27B0", linewidth=2)
    for x, v in yearly.items():
        ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Year-over-Year Demand Growth", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean sales per store-item-day")
    ax.set_xticks(yearly.index)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_yearly_growth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 7. Sales distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df[TARGET], bins=60, color="#2196F3", edgecolor="white")
    ax.set_title("Distribution of Daily Store-Item Sales", fontsize=13, fontweight="bold")
    ax.set_xlabel("Units sold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_sales_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 8. Store × Month heatmap
    pivot = df.pivot_table(values=TARGET, index="store",
                           columns=dates.dt.month, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Mean sales"})
    ax.set_title("Mean Sales — Store × Month", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Store")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_store_month_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 9. Weekday vs weekend by month (interaction)
    df_tmp = df[[TARGET]].copy()
    df_tmp["month"] = dates.dt.month.values
    df_tmp["is_weekend"] = (dates.dt.dayofweek >= 5).values
    interaction = df_tmp.groupby(["month", "is_weekend"])[TARGET].mean().unstack()
    interaction.columns = ["Weekday", "Weekend"]
    fig, ax = plt.subplots(figsize=(10, 5))
    interaction.plot(kind="bar", ax=ax, color=["#2196F3", "#FF5722"], edgecolor="white")
    ax.set_title("Weekday vs Weekend Demand by Month", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean sales")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_weekend_by_month.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 10. Example series — one store-item over time
    example = df[(df["store"] == 1) & (df["item"] == 1)].set_index(DATE_COLUMN)[TARGET]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(example.index, example.values, color="#2196F3", linewidth=0.5, alpha=0.7)
    ax.plot(example.rolling(30, center=True).mean(), color="#FF5722",
            linewidth=2, label="30-day rolling mean")
    ax.set_title("Example Series — Store 1, Item 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("Units sold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_example_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        f"EDA complete. Saved {len(list(FIGURES_DIR.glob('eda_*.png')))} "
        f"visualizations to {FIGURES_DIR}"
    )


# =============================================================================
# MLflow Tracking
# =============================================================================

def log_to_mlflow(
    model_results: dict,
    comparison_df: pd.DataFrame,
    best_model_name: str,
    skip_mlflow: bool = False,
) -> None:
    """Log all experiment results to MLflow."""
    if skip_mlflow:
        logger.info("MLflow tracking skipped (--skip-mlflow flag).")
        return

    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        for name, result in model_results.items():
            with mlflow.start_run(run_name=name):
                for param_key, param_val in result["best_params"].items():
                    mlflow.log_param(param_key, param_val)

                model_row = comparison_df[comparison_df["Model"] == name].iloc[0]
                mlflow.log_metric("test_smape", model_row["Test SMAPE"])
                mlflow.log_metric("test_rmse", model_row["Test RMSE"])
                mlflow.log_metric("test_mae", model_row["Test MAE"])
                mlflow.log_metric("test_r2", model_row["Test R2"])
                mlflow.log_metric("val_smape", model_row["Val SMAPE"])
                mlflow.log_metric("train_smape", model_row["Train SMAPE"])
                mlflow.log_metric("overfit_gap", model_row["Overfit Gap"])
                mlflow.log_metric("training_time", result["train_time"])

                if name == best_model_name:
                    mlflow.set_tag("best_model", "true")

                mlflow.sklearn.log_model(result["pipeline"], "model")

                # Log figures only on the best model's run — shared EDA and
                # comparison plots attached to every run would misattribute
                # model-specific artifacts
                if name == best_model_name:
                    for fig_path in FIGURES_DIR.glob("*.png"):
                        mlflow.log_artifact(str(fig_path), "figures")

                logger.info(f"MLflow: Logged run for {name}")

        logger.info("MLflow tracking complete. Run 'mlflow ui' to view experiments.")

    except Exception as e:
        logger.warning(f"MLflow tracking failed: {e}. Continuing without MLflow.")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(skip_mlflow: bool = False) -> None:
    """Execute the complete ML pipeline end-to-end."""
    pipeline_start = time.time()

    logger.info("=" * 60)
    logger.info("STORE ITEM DEMAND FORECASTING PIPELINE")
    logger.info("=" * 60)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    logger.info("\n📥 Step 1: Loading data...")
    df_raw, profile = load_and_validate()
    logger.info(f"Dataset: {profile['n_rows']:,} rows, {profile['n_series']} series")
    logger.info(f"Coverage: {profile['date_min']} .. {profile['date_max']}")

    # =========================================================================
    # Step 2: Clean Data
    # =========================================================================
    logger.info("\n🧹 Step 2: Cleaning data...")
    df_clean = clean_data(df_raw)

    # =========================================================================
    # Step 3: EDA Visualizations
    # =========================================================================
    logger.info("\n📊 Step 3: Running EDA...")
    run_eda(df_clean)

    # =========================================================================
    # Step 4: Feature Engineering (reference artifact for EDA/notebooks)
    # =========================================================================
    logger.info("\n🔧 Step 4: Engineering features (reference artifact)...")
    # In the model pipelines, feature engineering happens INSIDE the sklearn
    # Pipeline (FeatureEngineer step) so aggregates are learned from training
    # folds only. This full-data version is saved purely for reference.
    df_featured = engineer_features(df_clean)
    df_featured.head(50000).to_csv(PROCESSED_DATA_DIR / "featured_data_sample.csv", index=False)
    logger.info(f"Saved featured-data sample: {df_featured.shape}")

    # =========================================================================
    # Step 5: Chronological Split
    # =========================================================================
    logger.info("\n✂️ Step 5: Splitting data (chronological)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    # =========================================================================
    # Step 6: Feature-Importance Analysis
    # =========================================================================
    logger.info("\n🎯 Step 6: Analyzing feature importance...")
    analysis_pipeline = Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", get_default_preprocessor()),
    ])
    X_train_transformed = analysis_pipeline.fit_transform(X_train, y_train)
    feature_names_out = list(
        analysis_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names_out)

    selected_features, rankings = select_features(
        X_train_df, y_train.reset_index(drop=True), top_n=15,
    )
    logger.info(
        f"Feature-importance analysis found {len(selected_features)} strong "
        f"features. Models are trained on the full feature set; see "
        f"figures/feature_selection_*.png for the rankings."
    )
    del X_train_transformed, X_train_df  # free ~1GB before training

    # =========================================================================
    # Step 7: Build Final Preprocessing Pipeline
    # =========================================================================
    logger.info("\n🏗️ Step 7: Building final preprocessing pipeline...")
    final_preprocessor = get_default_preprocessor()

    # =========================================================================
    # Step 8: Train Models
    # =========================================================================
    logger.info("\n🚀 Step 8: Training models...")
    model_results = train_all_models(final_preprocessor, X_train, y_train)

    # =========================================================================
    # Step 9: Evaluate & Compare Models
    # =========================================================================
    logger.info("\n📈 Step 9: Evaluating models...")
    comparison_df = evaluate_all_models(
        model_results, X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # Determine best model (comparison_df is ranked by Validation SMAPE)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = model_results[best_model_name]["pipeline"]
    logger.info(f"\n🏆 Best model: {best_model_name}")

    # =========================================================================
    # Step 10: SHAP Explainability
    # =========================================================================
    logger.info("\n🔍 Step 10: Running SHAP explainability...")
    shap_results = run_explainability(
        best_pipeline, X_train, X_test, model_name=best_model_name,
    )

    # =========================================================================
    # Step 11: MLflow Experiment Tracking
    # =========================================================================
    logger.info("\n📋 Step 11: Logging to MLflow...")
    log_to_mlflow(model_results, comparison_df, best_model_name, skip_mlflow)

    # =========================================================================
    # Step 12: Save Models
    # =========================================================================
    logger.info("\n💾 Step 12: Saving models...")

    joblib.dump(best_pipeline, FINAL_PIPELINE_PATH)
    logger.info(f"Saved final pipeline: {FINAL_PIPELINE_PATH}")

    best_regressor = best_pipeline.named_steps["regressor"]
    joblib.dump(best_regressor, BEST_MODEL_PATH)
    logger.info(f"Saved best model: {BEST_MODEL_PATH}")

    feature_names = list(X_train.columns)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    logger.info(f"Saved feature names: {FEATURE_NAMES_PATH}")

    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    joblib.dump(preprocessor_fitted, PREPROCESSING_PIPELINE_PATH)
    logger.info(f"Saved preprocessor: {PREPROCESSING_PIPELINE_PATH}")

    # Save model metadata (consumed by the API for versioning/context)
    best_row = comparison_df.iloc[0]
    metadata = {
        "best_model": best_model_name,
        "val_smape": round(float(best_row["Val SMAPE"]), 4),
        "test_smape": round(float(best_row["Test SMAPE"]), 4),
        "test_rmse": round(float(best_row["Test RMSE"]), 4),
        "test_mae": round(float(best_row["Test MAE"]), 4),
        "test_r2": round(float(best_row["Test R2"]), 4),
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_training_rows": int(len(X_train)),
        "train_end_date": str(pd.to_datetime(X_train[DATE_COLUMN]).max().date()),
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Saved model metadata: {MODEL_METADATA_PATH}")

    # Save processed splits (sampled train to keep repo size sane)
    X_train.assign(**{TARGET: y_train}).tail(200000).to_csv(
        PROCESSED_DATA_DIR / "train.csv", index=False)
    X_val.assign(**{TARGET: y_val}).to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    logger.info("Saved processed splits (train sampled to last 200k rows).")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Test SMAPE: {best_row['Test SMAPE']:.3f}")
    logger.info(f"Figures saved: {len(list(FIGURES_DIR.glob('*.png')))}")
    logger.info(f"Models saved: {len(list(MODELS_DIR.glob('*.joblib')))}")
    logger.info(f"\nProject artifacts in: {MODELS_DIR.parent}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Store Item Demand Forecasting Pipeline",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Skip MLflow experiment tracking.",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    run_pipeline(skip_mlflow=args.skip_mlflow)
