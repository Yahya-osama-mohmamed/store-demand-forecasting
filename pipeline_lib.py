"""Shared definitions for the demand forecasting pipeline.

The analysis lives in `notebooks/demand_analysis.ipynb`. This module exists for
one reason: a pickled sklearn Pipeline stores its custom steps *by import path*,
so any class inside the saved model has to be importable at load time. If
`FeatureEngineer` were defined in the notebook it would pickle as
`__main__.FeatureEngineer` and neither the API nor the Lambda export could load
it.

So the transformer, the metric, and the column lists live here. Everything else
- loading, EDA, splitting, model search, evaluation, deployment export -
happens in the notebook.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        # The container runs as a non-root user on a filesystem it does not own,
        # so these directories cannot be created there - and do not need to be.
        # Serving only ever reads models/, which is baked into the image. The
        # notebook, which does write here, runs where the paths are writable.
        pass

DATASET_URL = (
    "https://raw.githubusercontent.com/DharitShah13/"
    "Kaggle-Store-Item-Demand-Forecasting-Challenge/master/train.csv"
)
RAW_DATA_FILE = RAW_DATA_DIR / "store-item-demand-train.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_VAL_FILE = PROCESSED_DATA_DIR / "validation.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

FINAL_PIPELINE_PATH = MODELS_DIR / "final_pipeline.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42
TARGET = "sales"
DATE_COLUMN = "date"
N_STORES, N_ITEMS = 10, 50

# The data spans 2013-01-01 .. 2017-12-31. Splits are chronological; a random
# split would put next December in training and last March in test.
TRAIN_END_DATE = "2017-06-30"
VAL_END_DATE = "2017-09-30"

CALENDAR_NUMERIC = [
    "year", "day", "dayofyear", "weekofyear",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
]
AGGREGATE_FEATURES = [
    "store_item_mean", "store_item_dow_mean", "item_month_mean", "store_month_mean",
]
CALENDAR_CATEGORICAL = ["month", "dayofweek"]
FLAG_FEATURES = ["is_weekend", "is_month_start", "is_month_end"]
ENTITY_FEATURES = ["store", "item"]
NUMERIC_FEATURES = ENTITY_FEATURES + CALENDAR_NUMERIC + AGGREGATE_FEATURES


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------

def smape(y_true, y_pred) -> float:
    """Symmetric Mean Absolute Percentage Error - the competition's metric.

        SMAPE = 100/n * sum( 2*|F - A| / (|A| + |F|) )

    Rows where actual and forecast are both zero contribute 0 rather than NaN.
    Lower is better; the range is [0, 200].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    diff = np.where(denom == 0, 0.0,
                    2.0 * np.abs(y_pred - y_true) / np.where(denom == 0, 1.0, denom))
    return float(100.0 * np.mean(diff))


# greater_is_better=False so sklearn's "higher score wins" convention still picks
# the lowest SMAPE; scores come back negated.
SMAPE_SCORER = make_scorer(smape, greater_is_better=False)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar and cyclical features from the date column.

    The sin/cos pairs let the models see that December neighbours January and
    Sunday neighbours Monday. Plain integers put them at opposite ends of the
    range, which forces a tree to spend splits rediscovering the wrap-around.
    """
    df = df.copy()
    dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df["year"] = dates.dt.year
    df["month"] = dates.dt.month
    df["day"] = dates.dt.day
    df["dayofweek"] = dates.dt.dayofweek
    df["dayofyear"] = dates.dt.dayofyear
    df["weekofyear"] = dates.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = dates.dt.is_month_start.astype(int)
    df["is_month_end"] = dates.dt.is_month_end.astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Calendar features plus demand aggregates learned from training data.

    `fit` needs y: the aggregates are mean sales per (store, item), per
    (store, item, weekday), and per seasonal slice. Learning them inside the
    pipeline is what keeps time-series CV honest - each fold computes its own
    means from its own past - and it is also what lets the API answer a request
    that carries nothing but (date, store, item).

    Unseen entity combinations fall back through coarser aggregates down to the
    global training mean, so a new store never produces NaN.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        for col in (DATE_COLUMN, "store", "item"):
            if col not in X.columns:
                raise ValueError(f"FeatureEngineer requires a '{col}' column.")
        if y is None:
            raise ValueError(
                "FeatureEngineer requires y (sales) during fit to learn aggregate "
                "demand statistics."
            )

        base = X[[DATE_COLUMN, "store", "item"]].copy()
        base["_sales"] = np.asarray(y, dtype=float)
        dates = pd.to_datetime(base[DATE_COLUMN], errors="coerce")
        base["_month"] = dates.dt.month
        base["_dow"] = dates.dt.dayofweek

        self.global_mean_ = float(base["_sales"].mean())
        self.store_item_mean_ = (
            base.groupby(["store", "item"])["_sales"].mean()
            .rename("store_item_mean").reset_index()
        )
        self.store_item_dow_mean_ = (
            base.groupby(["store", "item", "_dow"])["_sales"].mean()
            .rename("store_item_dow_mean").reset_index()
        )
        self.item_month_mean_ = (
            base.groupby(["item", "_month"])["_sales"].mean()
            .rename("item_month_mean").reset_index()
        )
        self.store_month_mean_ = (
            base.groupby(["store", "_month"])["_sales"].mean()
            .rename("store_month_mean").reset_index()
        )

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = create_calendar_features(X)
        df["store"] = pd.to_numeric(df["store"], errors="coerce")
        df["item"] = pd.to_numeric(df["item"], errors="coerce")

        # Left merges preserve row order; the fallback cascade ends at the
        # global training mean so unseen entities never yield NaN.
        df = df.merge(self.store_item_mean_, on=["store", "item"], how="left")
        df = df.merge(
            self.store_item_dow_mean_.rename(columns={"_dow": "dayofweek"}),
            on=["store", "item", "dayofweek"], how="left",
        )
        df = df.merge(
            self.item_month_mean_.rename(columns={"_month": "month"}),
            on=["item", "month"], how="left",
        )
        df = df.merge(
            self.store_month_mean_.rename(columns={"_month": "month"}),
            on=["store", "month"], how="left",
        )

        df["store_item_mean"] = df["store_item_mean"].fillna(self.global_mean_)
        df["store_item_dow_mean"] = df["store_item_dow_mean"].fillna(df["store_item_mean"])
        df["item_month_mean"] = df["item_month_mean"].fillna(self.global_mean_)
        df["store_month_mean"] = df["store_month_mean"].fillna(self.global_mean_)

        df.index = X.index
        return df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        # Must match the order transform() appends them in.
        engineered = [
            "year", "month", "day", "dayofweek", "dayofyear", "weekofyear",
            "is_weekend", "is_month_start", "is_month_end",
            "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
            "store_item_mean", "store_item_dow_mean",
            "item_month_mean", "store_month_mean",
        ]
        new = [f for f in engineered if f not in list(input_features)]
        return np.concatenate(
            [np.asarray(input_features, dtype=object), np.asarray(new, dtype=object)]
        )
