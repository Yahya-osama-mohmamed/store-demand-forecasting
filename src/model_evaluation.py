"""
Model Evaluation — Metrics, visualizations, and comparison for forecasting.

Evaluates each model on Train, Validation, and Test sets using:
- SMAPE (Symmetric Mean Absolute Percentage Error — the competition metric)
- RMSE, MAE, R²
- Predicted-vs-actual and residual diagnostics
- Aggregate forecast-vs-actual over the test window
- Overfitting analysis (validation-train SMAPE gap)
- Side-by-side comparison table (ranked by VALIDATION SMAPE)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, make_scorer,
)
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, REPORTS_DIR, DATE_COLUMN
from src.logger import get_logger

logger = get_logger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


# =============================================================================
# Metrics
# =============================================================================

def smape(y_true, y_pred) -> float:
    """
    Symmetric Mean Absolute Percentage Error (the competition metric).

        SMAPE = 100/n * sum( 2*|F - A| / (|A| + |F|) )

    Rows where both actual and forecast are 0 contribute 0 (not NaN).
    Lower is better; range [0, 200].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.where(denominator == 0, 0.0, 2.0 * np.abs(y_pred - y_true) / np.where(denominator == 0, 1.0, denominator))
    return float(100.0 * np.mean(diff))


# Scorer for RandomizedSearchCV (sklearn maximizes → negate)
SMAPE_SCORER = make_scorer(smape, greater_is_better=False)


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    dataset_name: str = "Test",
) -> Dict[str, float]:
    """
    Compute all regression metrics for a single dataset split.

    Args:
        y_true: Ground truth sales.
        y_pred: Predicted sales.
        dataset_name: Name of the dataset split (for logging).

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {
        "smape": smape(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    logger.info(
        f"{dataset_name} metrics — "
        f"SMAPE: {metrics['smape']:.3f}, "
        f"RMSE: {metrics['rmse']:.3f}, "
        f"MAE: {metrics['mae']:.3f}, "
        f"R²: {metrics['r2']:.4f}"
    )

    return metrics


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    model_name: str,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a single model on all three chronological splits.

    Returns:
        Nested dict: {split_name: {metric_name: value}}.
    """
    logger.info(f"\n--- Evaluating: {model_name} ---")
    results = {}

    for split_name, X, y in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = np.clip(pipeline.predict(X), 0, None)  # demand can't be negative
        results[split_name] = compute_metrics(y, y_pred, split_name.title())

    # Overfitting check: validation vs train SMAPE gap
    gap = results["validation"]["smape"] - results["train"]["smape"]
    if gap > 3.0:
        logger.warning(
            f"⚠ Possible overfitting: Val SMAPE ({results['validation']['smape']:.3f}) - "
            f"Train SMAPE ({results['train']['smape']:.3f}) = {gap:.3f}"
        )
    else:
        logger.info(f"✓ No significant overfitting: Val-Train SMAPE gap = {gap:.3f}")

    return results


# =============================================================================
# Visualizations
# =============================================================================

def plot_predicted_vs_actual(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    dataset_name: str = "Test",
    sample_n: int = 5000,
) -> None:
    """Scatter of predicted vs actual sales on a sample of the split."""
    if len(X) > sample_n:
        idx = X.sample(n=sample_n, random_state=42).index
        X_s, y_s = X.loc[idx], y.loc[idx]
    else:
        X_s, y_s = X, y
    y_pred = np.clip(pipeline.predict(X_s), 0, None)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_s, y_pred, s=6, alpha=0.25, color="#2196F3", edgecolors="none")
    lims = [0, max(float(y_s.max()), float(y_pred.max())) * 1.05]
    ax.plot(lims, lims, "k--", alpha=0.6, label="Perfect forecast")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual sales", fontsize=12)
    ax.set_ylabel("Predicted sales", fontsize=12)
    ax.set_title(f"Predicted vs Actual — {model_name} ({dataset_name})",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    filename = f"pred_vs_actual_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    sample_n: int = 20000,
) -> None:
    """Residual distribution on the test split."""
    if len(X) > sample_n:
        idx = X.sample(n=sample_n, random_state=42).index
        X_s, y_s = X.loc[idx], y.loc[idx]
    else:
        X_s, y_s = X, y
    residuals = y_s - np.clip(pipeline.predict(X_s), 0, None)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=60, color="#FF5722", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", alpha=0.6)
    ax.set_xlabel("Residual (actual − predicted)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Residual Distribution — {model_name} (Test)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    filename = f"residuals_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_vs_actual_timeline(
    model_results: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Aggregate daily forecast vs actual over the test window, all models.

    Summing across all 500 store-item series gives a readable one-line
    view of how well each model tracks level, trend, and weekly rhythm.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    dates = pd.to_datetime(X_test[DATE_COLUMN])

    actual_daily = y_test.groupby(dates.values).sum()
    ax.plot(actual_daily.index, actual_daily.values, color="black",
            linewidth=2, label="Actual")

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for idx, (name, result) in enumerate(model_results.items()):
        y_pred = np.clip(result["pipeline"].predict(X_test), 0, None)
        pred_daily = pd.Series(y_pred).groupby(dates.values).sum()
        ax.plot(pred_daily.index, pred_daily.values,
                color=colors[idx % len(colors)], linewidth=1.4,
                alpha=0.85, label=f"{name} forecast")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Total daily sales (all stores & items)", fontsize=12)
    ax.set_title("Test Window — Aggregate Forecast vs Actual",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_vs_actual_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved aggregate forecast-vs-actual plot.")


def plot_comparison_bar_chart(comparison_df: pd.DataFrame) -> None:
    """Grouped bar chart comparing error metrics across models."""
    metrics = ["Test SMAPE", "Test RMSE", "Test MAE"]
    models = comparison_df["Model"].tolist()

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    for ax, metric in zip(axes, metrics):
        values = comparison_df[metric].values
        ax.bar(models, values, color=colors[:len(models)], alpha=0.85, edgecolor="white")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("Model Performance Comparison (lower is better)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved model comparison bar chart.")


# =============================================================================
# Model Comparison Table
# =============================================================================

def create_comparison_table(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    training_times: Dict[str, float],
    descriptions: Dict[str, str],
) -> pd.DataFrame:
    """
    Create a comprehensive model comparison table.

    Returns:
        DataFrame with one row per model, sorted by validation SMAPE
        ascending (the selection metric; test columns are reported but
        never used to pick the winner).
    """
    rows = []
    for name in all_metrics:
        test = all_metrics[name].get("test", {})
        val = all_metrics[name].get("validation", {})
        train = all_metrics[name].get("train", {})

        rows.append({
            "Model": name,
            "Test SMAPE": test.get("smape", np.inf),
            "Test RMSE": test.get("rmse", np.inf),
            "Test MAE": test.get("mae", np.inf),
            "Test R2": test.get("r2", 0),
            "Val SMAPE": val.get("smape", np.inf),
            "Train SMAPE": train.get("smape", np.inf),
            "Overfit Gap": val.get("smape", np.inf) - train.get("smape", np.inf),
            "Training Time (s)": training_times.get(name, 0),
            "Description": descriptions.get(name, ""),
        })

    comparison_df = pd.DataFrame(rows)
    # Rank by VALIDATION SMAPE: choosing the winner on the test set would
    # leak test information into model selection and bias the reported score.
    comparison_df = comparison_df.sort_values("Val SMAPE", ascending=True)
    comparison_df = comparison_df.reset_index(drop=True)

    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    logger.info(f"Model comparison table saved to {REPORTS_DIR / 'model_comparison.csv'}")

    display_cols = [
        "Model", "Test SMAPE", "Test RMSE", "Test MAE", "Test R2",
        "Val SMAPE", "Overfit Gap", "Training Time (s)",
    ]
    logger.info("\n" + comparison_df[display_cols].to_string(index=False))

    return comparison_df


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================

def evaluate_all_models(
    model_results: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> pd.DataFrame:
    """
    Run the complete evaluation pipeline for all trained models.

    Steps:
    1. Compute metrics on all chronological splits for each model
    2. Predicted-vs-actual and residual diagnostics per model
    3. Aggregate forecast-vs-actual timeline over the test window
    4. Comparison table (ranked by validation SMAPE) and bar chart

    Returns:
        Comparison DataFrame.
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    all_metrics = {}
    training_times = {}
    descriptions = {}

    for name, result in model_results.items():
        pipeline = result["pipeline"]

        metrics = evaluate_model(
            pipeline,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            model_name=name,
        )
        all_metrics[name] = metrics
        training_times[name] = result["train_time"]
        descriptions[name] = result["description"]

        plot_predicted_vs_actual(pipeline, X_test, y_test, name, "Test")
        plot_residuals(pipeline, X_test, y_test, name)

    plot_forecast_vs_actual_timeline(model_results, X_test, y_test)

    comparison_df = create_comparison_table(all_metrics, training_times, descriptions)
    plot_comparison_bar_chart(comparison_df)

    best_model_name = comparison_df.iloc[0]["Model"]
    logger.info(f"\n🏆 Best model: {best_model_name} (lowest Validation SMAPE)")

    return comparison_df
