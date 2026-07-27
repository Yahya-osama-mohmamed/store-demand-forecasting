"""
Model Explainability — SHAP-based interpretation.

Uses SHAP (SHapley Additive exPlanations) to provide:
- Global feature importance (which features drive demand overall)
- Local explanations (why the model made a specific forecast)
- Dependence plots (how a feature's effect varies with its value)

All three candidate models are tree ensembles, so the fast, exact
TreeExplainer applies throughout.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any

from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """
    Run raw features through every pipeline step except the final regressor
    and return the result as a DataFrame with proper feature names.

    Args:
        pipeline: Fitted Pipeline (feature engineering + preprocessing + regressor).
        X: Raw input features (date, store, item).

    Returns:
        Transformed DataFrame in model-input space.
    """
    transformer = Pipeline(pipeline.steps[:-1])
    X_transformed = transformer.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        logger.warning("Could not extract feature names from preprocessor.")
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    return pd.DataFrame(np.asarray(X_transformed), columns=feature_names)


def get_shap_explainer(
    pipeline: Pipeline,
    model_name: str = "Model",
) -> shap.Explainer:
    """
    Create a SHAP TreeExplainer for the pipeline's regressor.

    All candidate models (LightGBM, XGBoost, HistGradientBoosting) are tree
    ensembles supported by the fast, exact TreeExplainer.
    """
    regressor = pipeline.named_steps["regressor"]
    logger.info(f"Using TreeExplainer for {model_name} ({type(regressor).__name__})")
    return shap.TreeExplainer(regressor)


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    explainer: shap.Explainer,
    max_samples: int = 2000,
) -> tuple:
    """
    Compute SHAP values for a (sampled) dataset.

    Args:
        pipeline: Fitted Pipeline.
        X: Raw features (before preprocessing).
        explainer: SHAP Explainer instance.
        max_samples: Maximum number of rows to explain (for speed).

    Returns:
        Tuple of (shap_values, X_transformed as DataFrame).
    """
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    X_df = transform_features(pipeline, X_sample)

    logger.info(f"Computing SHAP values for {len(X_df)} samples...")
    shap_values = explainer.shap_values(X_df)

    # Regression explainers return a 2D array; guard list/3D just in case
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    elif getattr(shap_values, "ndim", 2) == 3:
        shap_values = shap_values[:, :, 0]

    return shap_values, X_df


def plot_shap_summary(shap_values, X_df, model_name="Model") -> None:
    """SHAP beeswarm summary — direction and magnitude of every feature."""
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, plot_type="dot", show=False, max_display=15)
    plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP summary plot (beeswarm).")


def plot_shap_bar(shap_values, X_df, model_name="Model") -> None:
    """SHAP bar plot — global mean |SHAP| importance."""
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False, max_display=15)
    plt.title(f"Global Feature Importance (SHAP) — {model_name}",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP bar plot (global importance).")


def plot_shap_waterfall(
    explainer, shap_values, X_df, sample_idx: int = 0, model_name: str = "Model",
) -> None:
    """SHAP waterfall for a single forecast — the best stakeholder view."""
    try:
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = np.atleast_1d(expected)[0]

        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=expected,
            data=X_df.iloc[sample_idx].values,
            feature_names=X_df.columns.tolist(),
        )

        fig = plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False, max_display=12)
        plt.title(f"SHAP Waterfall — Single Forecast ({model_name})",
                  fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved SHAP waterfall plot.")
    except Exception as e:
        logger.warning(f"Could not create waterfall plot: {e}")


def plot_shap_dependence(
    shap_values, X_df, top_n: int = 3, model_name: str = "Model",
) -> None:
    """Dependence plots for the top-N most important features."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_features = [X_df.columns[i] for i in top_indices]

    for feature in top_features:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(feature, shap_values, X_df, show=False, ax=ax)
            ax.set_title(f"SHAP Dependence — {feature} ({model_name})",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            safe_name = feature.replace(" ", "_").replace("/", "_")
            fig.savefig(FIGURES_DIR / f"shap_dependence_{safe_name}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not create dependence plot for {feature}: {e}")

    logger.info(f"Saved SHAP dependence plots for top {top_n} features.")


def run_explainability(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str = "Best Model",
) -> Dict[str, Any]:
    """
    Run the full SHAP explainability pipeline.

    Args:
        pipeline: Fitted best-model Pipeline.
        X_train: Training features (unused for tree explainers; kept for parity).
        X_test: Test features (to explain).
        model_name: Model name for plot titles.

    Returns:
        Dictionary with SHAP values and feature importance ranking.
    """
    logger.info("=" * 60)
    logger.info(f"SHAP EXPLAINABILITY — {model_name}")
    logger.info("=" * 60)

    explainer = get_shap_explainer(pipeline, model_name)
    shap_values, X_df = compute_shap_values(pipeline, X_test, explainer)

    plot_shap_summary(shap_values, X_df, model_name)
    plot_shap_bar(shap_values, X_df, model_name)
    plot_shap_waterfall(explainer, shap_values, X_df, sample_idx=0, model_name=model_name)
    plot_shap_dependence(shap_values, X_df, top_n=3, model_name=model_name)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    logger.info("\nTop 10 features by SHAP importance:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

    return {
        "shap_values": shap_values,
        "feature_importance": importance_df,
        "explainer": explainer,
    }
