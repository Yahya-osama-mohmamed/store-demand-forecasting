"""
Feature Engineering — Calendar features and learned demand aggregates.

Demand for a (store, item) pair on a given day is driven by:
- Seasonality — month of year, day of week, position in the year
- The entity's own demand level — how much this store sells of this item

The `FeatureEngineer` transformer is the production entry point: it lives
INSIDE the sklearn Pipeline, so every statistic it needs (per-entity mean
sales, per-entity day-of-week means, seasonal means) is learned in `fit`
on training data only and re-applied identically at inference time. This
prevents both leakage during time-series cross-validation and train/serve
skew in the API: a request for any future date needs only (date, store,
item) — the aggregates come from the fitted transformer.

Features created:
- Calendar: year, month, day, dayofweek, dayofyear, weekofyear,
  is_weekend, is_month_start, is_month_end
- Cyclical encodings: sin/cos of month, day-of-week, and day-of-year
  (December is adjacent to January; Sunday is adjacent to Monday)
- Learned aggregates (fit on training sales only):
  store_item_mean, store_item_dow_mean, item_month_mean, store_month_mean
"""

import pandas as pd
import numpy as np
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin

from src.config import DATE_COLUMN
from src.logger import get_logger

logger = get_logger(__name__)


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive calendar and cyclical features from the date column.

    Cyclical sin/cos pairs let the models see that month 12 neighbours
    month 1 and Sunday neighbours Monday — plain integers put them at
    opposite ends of the range.
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
    """
    Sklearn transformer that adds calendar features and learned aggregates.

    Designed to be the FIRST step of the model pipeline:

        Pipeline([
            ("features", FeatureEngineer()),
            ("preprocessor", ColumnTransformer(...)),
            ("regressor", ...),
        ])

    `fit` learns per-entity and seasonal mean-sales statistics from the
    training fold only, so time-series cross-validation is leakage-free and
    single-row API requests get the exact same feature definitions as
    training data. Unseen (store, item) combinations fall back through
    coarser aggregates down to the global training mean.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """Learn aggregate demand statistics from training data."""
        for col in (DATE_COLUMN, "store", "item"):
            if col not in X.columns:
                raise ValueError(f"FeatureEngineer requires a '{col}' column.")
        if y is None:
            raise ValueError(
                "FeatureEngineer requires y (sales) during fit to learn "
                "aggregate demand statistics."
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
        """Add calendar features and learned aggregates (fit stats only)."""
        df = create_calendar_features(X)
        df["store"] = pd.to_numeric(df["store"], errors="coerce")
        df["item"] = pd.to_numeric(df["item"], errors="coerce")

        # Left-merges preserve row order; fallback cascade ends at the
        # global training mean so unseen entities never produce NaN
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
        """Return input feature names plus the engineered feature names."""
        if input_features is None:
            input_features = self.feature_names_in_
        # Same order in which transform() appends the new columns
        engineered = [
            "year", "month", "day", "dayofweek", "dayofyear", "weekofyear",
            "is_weekend", "is_month_start", "is_month_end",
            "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
            "store_item_mean", "store_item_dow_mean",
            "item_month_mean", "store_month_mean",
        ]
        new = [f for f in engineered if f not in list(input_features)]
        return np.concatenate([
            np.asarray(input_features, dtype=object),
            np.asarray(new, dtype=object),
        ])


def engineer_features(
    df: pd.DataFrame,
    target: Optional[str] = "sales",
) -> pd.DataFrame:
    """
    Apply all feature engineering for EDA/notebook exploration.

    Aggregates are computed from `df` itself here — acceptable only for
    full-dataset exploration. Production code must use the FeatureEngineer
    transformer inside the model pipeline so statistics come from training
    folds only.

    Args:
        df: Cleaned DataFrame including the target column.
        target: Target column name used for the aggregates.

    Returns:
        DataFrame with all engineered features added.
    """
    fe = FeatureEngineer()
    X = df.drop(columns=[target]) if target in df.columns else df
    y = df[target] if target in df.columns else None
    fe.fit(X, y)
    out = fe.transform(X)
    if target in df.columns:
        out[target] = df[target].values
    logger.info("Feature engineering complete (exploration mode).")
    return out
