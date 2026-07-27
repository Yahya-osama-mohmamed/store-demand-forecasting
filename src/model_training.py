"""
Model Training — Train, tune, and cross-validate forecasting models.

Models (the top performers for tabular demand forecasting — all
histogram-based gradient boosting, which dominates this competition's
public leaderboard):
1. LightGBM — leaf-wise gradient boosting, typically the strongest here
2. XGBoost — depth-wise gradient boosting with strong regularization
3. HistGradientBoosting — scikit-learn's native hist GBM (LightGBM-style)

All models are wrapped in sklearn Pipelines with the FeatureEngineer and
preprocessing ColumnTransformer, so every statistic is re-learned per
cross-validation fold — no leakage. Cross-validation uses TimeSeriesSplit:
each fold validates on data strictly AFTER its training window, mirroring
real forecasting.
"""

import time
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb

from src.config import (
    RANDOM_STATE, CV_FOLDS, N_ITER_SEARCH,
    LIGHTGBM_PARAMS, XGBOOST_PARAMS, HISTGB_PARAMS,
)
from src.feature_engineering import FeatureEngineer
from src.model_evaluation import SMAPE_SCORER
from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Model Definitions
# =============================================================================

def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Define all model configurations with their regressors and param grids.

    Returns:
        Dictionary mapping model names to config dicts containing:
        - 'regressor': sklearn-compatible regressor instance
        - 'params': hyperparameter search space
        - 'description': human-readable model description
    """
    configs = {
        "LightGBM": {
            "regressor": lgb.LGBMRegressor(
                random_state=RANDOM_STATE,
                verbosity=-1,
                n_jobs=1,  # search parallelizes across processes; nested
                           # thread pools can crash loky workers on Windows
            ),
            "params": LIGHTGBM_PARAMS,
            "description": (
                "Leaf-wise gradient boosting. Histogram-based splitting for "
                "speed; typically the strongest model on this competition."
            ),
        },
        "XGBoost": {
            "regressor": xgb.XGBRegressor(
                random_state=RANDOM_STATE,
                tree_method="hist",
                verbosity=0,
                n_jobs=1,  # see LightGBM note
            ),
            "params": XGBOOST_PARAMS,
            "description": (
                "Depth-wise gradient boosting with L1/L2 regularization "
                "and histogram tree construction."
            ),
        },
        "HistGradientBoosting": {
            "regressor": HistGradientBoostingRegressor(
                random_state=RANDOM_STATE,
            ),
            "params": HISTGB_PARAMS,
            "description": (
                "scikit-learn's native histogram gradient boosting "
                "(LightGBM-inspired), dependency-free and fast."
            ),
        },
    }

    return configs


# =============================================================================
# Training Pipeline
# =============================================================================

def build_model_pipeline(
    preprocessor: ColumnTransformer,
    regressor: Any,
) -> Pipeline:
    """
    Create an sklearn Pipeline: feature engineering → preprocessing → regressor.

    The pipeline ensures that:
    - Learned aggregates and all preprocessing are fit ONLY on training data
    - Time-series cross-validation properly re-fits every step per fold
    - The serialized pipeline is fully self-contained for deployment:
      it accepts raw (date, store, item) records

    Args:
        preprocessor: Unfitted ColumnTransformer (cloned per model).
        regressor: sklearn-compatible regressor instance.

    Returns:
        sklearn Pipeline.
    """
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", clone(preprocessor)),
        ("regressor", regressor),
    ])


def train_single_model(
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    n_iter: int = N_ITER_SEARCH,
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Train a single model with hyperparameter tuning via RandomizedSearchCV.

    Uses TimeSeriesSplit so every CV fold validates on data strictly after
    its training window (X_train must be sorted by date — split_data
    guarantees this). Optimizes SMAPE, the competition metric.

    Returns:
        Tuple of (best pipeline, best params, training time in seconds).
    """
    cv = TimeSeriesSplit(n_splits=cv_folds)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=min(n_iter, _count_param_combinations(param_grid)),
        cv=cv,
        scoring=SMAPE_SCORER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    train_time = time.time() - start_time

    best_params = search.best_params_
    best_score = -search.best_score_  # scorer is negated (lower SMAPE = better)

    logger.info(
        f"Best CV SMAPE: {best_score:.3f} (training time: {train_time:.1f}s)"
    )
    logger.info(f"Best params: {best_params}")

    return search.best_estimator_, best_params, train_time


def _count_param_combinations(param_grid: Dict[str, Any]) -> int:
    """Count the total number of parameter combinations in a grid."""
    total = 1
    for values in param_grid.values():
        if isinstance(values, list):
            total *= len(values)
    return total


def train_all_models(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all models with hyperparameter tuning.

    Args:
        preprocessor: ColumnTransformer for feature transformation.
        X_train: Training features (date-sorted).
        y_train: Training target (sales).

    Returns:
        Dictionary mapping model names to result dicts containing:
        - 'pipeline': Best fitted pipeline
        - 'best_params': Best hyperparameters
        - 'train_time': Training time in seconds
        - 'description': Model description
    """
    model_configs = get_model_configs()
    results = {}

    for name, config in model_configs.items():
        logger.info("=" * 60)
        logger.info(f"Training: {name}")
        logger.info("=" * 60)

        pipeline = build_model_pipeline(
            preprocessor=preprocessor,
            regressor=config["regressor"],
        )

        best_pipeline, best_params, train_time = train_single_model(
            pipeline=pipeline,
            param_grid=config["params"],
            X_train=X_train,
            y_train=y_train,
        )

        results[name] = {
            "pipeline": best_pipeline,
            "best_params": best_params,
            "train_time": train_time,
            "description": config["description"],
        }

    logger.info("=" * 60)
    logger.info("All models trained successfully.")
    logger.info("=" * 60)

    return results
