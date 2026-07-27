"""
Project Configuration — Central source of truth for all constants, paths, and settings.

This module defines:
- File paths (data, models, figures, reports)
- Random seed for reproducibility
- Time-based split boundaries
- Feature column definitions
- Hyperparameter search spaces
- Model configuration
"""

from pathlib import Path
from typing import Dict, List, Any


# =============================================================================
# Project Root & Directory Paths
# =============================================================================
# All paths are relative to project root for portability
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR,
             REPORTS_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Dataset Configuration
# =============================================================================
# Store Item Demand Forecasting Challenge (Kaggle).
# Direct download from a public GitHub mirror — no Kaggle credentials required.
DATASET_URL = (
    "https://raw.githubusercontent.com/DharitShah13/"
    "Kaggle-Store-Item-Demand-Forecasting-Challenge/master/train.csv"
)
RAW_DATA_FILE = RAW_DATA_DIR / "store-item-demand-train.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_VAL_FILE = PROCESSED_DATA_DIR / "validation.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"


# =============================================================================
# Reproducibility & Time-Based Split
# =============================================================================
RANDOM_STATE = 42

# The data spans 2013-01-01 .. 2017-12-31 (5 full years, 500 daily series).
# Chronological split — random splitting would leak the future into training:
#   Train      : 2013-01-01 .. 2017-06-30  (~90%)
#   Validation : 2017-07-01 .. 2017-09-30  (3 months — model selection)
#   Test       : 2017-10-01 .. 2017-12-31  (3 months — final estimate,
#                matching the competition's 3-month forecast horizon)
TRAIN_END_DATE = "2017-06-30"
VAL_END_DATE = "2017-09-30"


# =============================================================================
# Target Variable
# =============================================================================
TARGET = "sales"
DATE_COLUMN = "date"

# Valid entity ranges (used for validation)
N_STORES = 10
N_ITEMS = 50


# =============================================================================
# Feature Definitions
# =============================================================================
# Raw input columns the serving pipeline expects
RAW_FEATURES: List[str] = ["date", "store", "item"]

# Calendar features created by FeatureEngineer (numeric)
CALENDAR_NUMERIC: List[str] = [
    "year", "day", "dayofyear", "weekofyear",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
]

# Learned aggregate features (statistics fit on TRAINING data only)
AGGREGATE_FEATURES: List[str] = [
    "store_item_mean", "store_item_dow_mean",
    "item_month_mean", "store_month_mean",
]

# Calendar categoricals → one-hot encoded
CALENDAR_CATEGORICAL: List[str] = ["month", "dayofweek"]

# 0/1 flags passed through unscaled
FLAG_FEATURES: List[str] = ["is_weekend", "is_month_start", "is_month_end"]

# Entity IDs kept numeric — the learned aggregates carry the per-entity
# signal, and tree models can still split on the raw IDs
ENTITY_FEATURES: List[str] = ["store", "item"]

# All numeric columns going through the numeric branch
NUMERIC_FEATURES: List[str] = ENTITY_FEATURES + CALENDAR_NUMERIC + AGGREGATE_FEATURES


# =============================================================================
# Hyperparameter Search Spaces
# =============================================================================
LIGHTGBM_PARAMS: Dict[str, Any] = {
    "regressor__n_estimators": [300, 500, 800],
    "regressor__learning_rate": [0.03, 0.05, 0.1],
    "regressor__num_leaves": [31, 64, 128],
    "regressor__max_depth": [-1, 8, 12],
    "regressor__subsample": [0.8, 1.0],
    "regressor__colsample_bytree": [0.8, 1.0],
    "regressor__min_child_samples": [20, 50, 100],
    "regressor__reg_lambda": [0.0, 0.1, 1.0],
}

XGBOOST_PARAMS: Dict[str, Any] = {
    "regressor__n_estimators": [300, 500, 800],
    "regressor__learning_rate": [0.03, 0.05, 0.1],
    "regressor__max_depth": [6, 8, 10],
    "regressor__subsample": [0.8, 1.0],
    "regressor__colsample_bytree": [0.8, 1.0],
    "regressor__min_child_weight": [1, 5, 10],
    "regressor__reg_lambda": [0.5, 1.0, 2.0],
}

HISTGB_PARAMS: Dict[str, Any] = {
    "regressor__max_iter": [300, 500, 800],
    "regressor__learning_rate": [0.03, 0.05, 0.1],
    "regressor__max_leaf_nodes": [31, 63, 127],
    "regressor__min_samples_leaf": [20, 50, 100],
    "regressor__l2_regularization": [0.0, 0.1, 1.0],
}


# =============================================================================
# Model Training Configuration
# =============================================================================
# TimeSeriesSplit folds inside RandomizedSearchCV (chronological CV — each
# fold validates on data strictly after its training window)
CV_FOLDS = 2
N_ITER_SEARCH = 8          # RandomizedSearchCV iterations per model
# Primary metric: SMAPE (the competition metric) — lower is better


# =============================================================================
# MLflow Configuration
# =============================================================================
MLFLOW_EXPERIMENT_NAME = "store-demand-forecasting"
MLFLOW_TRACKING_URI = f"file:///{(PROJECT_ROOT / 'mlruns').as_posix()}"


# =============================================================================
# Model Persistence
# =============================================================================
FINAL_PIPELINE_PATH = MODELS_DIR / "final_pipeline.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
PREPROCESSING_PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
