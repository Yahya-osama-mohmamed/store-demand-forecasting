"""
Feature Selection — Identify the most predictive features (analysis only).

Uses three complementary methods on the transformed training matrix:
1. Correlation filtering — flag highly correlated feature pairs
2. Mutual Information — rank features by information gain with the target
3. Model-based importance — Random Forest feature importances

The rankings are ANALYSIS ONLY: with a modest, well-regularized feature
set the models train on all features; these figures document where the
signal lives. Both rankers run on a random sample of the (913k-row)
training matrix for speed — importance rankings stabilize long before
full-data size.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict

from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

from src.config import RANDOM_STATE, FIGURES_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def correlation_filter(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> List[Tuple[str, str, float]]:
    """
    Report feature pairs with |correlation| above the threshold.

    Cyclical encodings and their source integers are expected to correlate;
    the report documents redundancy rather than dropping columns (tree
    models are unaffected by collinearity).

    Returns:
        List of (feature_a, feature_b, correlation) tuples.
    """
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    pairs = []
    for col in upper_tri.columns:
        for row in upper_tri.index[upper_tri[col] > threshold]:
            pairs.append((row, col, float(corr_matrix.loc[row, col])))
            logger.info(
                f"High correlation: {row} ~ {col} (r={corr_matrix.loc[row, col]:.3f})"
            )

    logger.info(f"Correlation filter: {len(pairs)} pairs above |r| > {threshold}.")
    return pairs


def mutual_information_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 15,
    sample_n: int = 50000,
) -> pd.DataFrame:
    """
    Rank features by Mutual Information with the target (regression).

    Runs on a random sample for speed — MI estimates stabilize well below
    full-dataset size.

    Returns:
        DataFrame with features ranked by MI score.
    """
    if len(X) > sample_n:
        idx = X.sample(n=sample_n, random_state=RANDOM_STATE).index
        X_s, y_s = X.loc[idx], y.loc[idx]
    else:
        X_s, y_s = X, y

    mi_scores = mutual_info_regression(X_s, y_s, random_state=RANDOM_STATE)

    mi_df = pd.DataFrame({
        "feature": X.columns,
        "mi_score": mi_scores,
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)

    logger.info(f"Top {top_n} features by Mutual Information:")
    for _, row in mi_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']}: {row['mi_score']:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=mi_df.head(top_n), x="mi_score", y="feature",
        hue="feature", palette="viridis", legend=False, ax=ax,
    )
    ax.set_title("Feature Ranking — Mutual Information", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mutual Information Score")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_selection_mi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return mi_df


def model_based_importance(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 15,
    sample_n: int = 100000,
) -> pd.DataFrame:
    """
    Rank features using Random Forest feature importances (on a sample).

    Returns:
        DataFrame with features ranked by importance.
    """
    if len(X) > sample_n:
        idx = X.sample(n=sample_n, random_state=RANDOM_STATE).index
        X_s, y_s = X.loc[idx], y.loc[idx]
    else:
        X_s, y_s = X, y

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_s, y_s)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(f"Top {top_n} features by Random Forest importance:")
    for _, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=importance_df.head(top_n), x="importance", y="feature",
        hue="feature", palette="magma", legend=False, ax=ax,
    )
    ax.set_title("Feature Ranking — Random Forest Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_selection_rf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return importance_df


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 15,
    corr_threshold: float = 0.95,
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Run the full feature-importance analysis.

    Returns:
        Tuple of (union of top features from MI + RF, dict of ranking DataFrames).
    """
    logger.info("=" * 60)
    logger.info("FEATURE-IMPORTANCE ANALYSIS")
    logger.info("=" * 60)

    correlation_filter(X, threshold=corr_threshold)
    mi_df = mutual_information_ranking(X, y, top_n=top_n)
    rf_df = model_based_importance(X, y, top_n=top_n)

    mi_top = set(mi_df.head(top_n)["feature"].tolist())
    rf_top = set(rf_df.head(top_n)["feature"].tolist())
    selected = sorted(mi_top | rf_top)

    logger.info(f"Strong features (union of MI + RF top-{top_n}): {len(selected)}")
    for feat in selected:
        logger.info(f"  ✓ {feat}")

    rankings = {"mutual_information": mi_df, "random_forest": rf_df}
    return selected, rankings
