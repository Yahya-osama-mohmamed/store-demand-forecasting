"""
Data Preprocessing Pipeline — Cleaning, Encoding, and Splitting.

Implements a scikit-learn Pipeline + ColumnTransformer approach to ensure:
- No data leakage (all statistics fit on training data only; splits are
  strictly chronological)
- Reproducible, consistent transformations
- Easy serialization for deployment

Steps:
1. Clean the raw data (types, duplicates, invalid values)
2. Chronological train / validation / test split
3. ColumnTransformer: numeric branch (impute), calendar one-hot branch,
   flag passthrough branch; everything else DROPPED so unexpected columns
   in inference data (e.g. `sales` or `id` in an uploaded CSV) are ignored
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import (
    TARGET, DATE_COLUMN, TRAIN_END_DATE, VAL_END_DATE,
    NUMERIC_FEATURES, CALENDAR_CATEGORICAL, FLAG_FEATURES,
    N_STORES, N_ITEMS,
)
from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Cleaning
# =============================================================================

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Initial data cleaning steps applied before the main pipeline.

    Handles:
    - Parsing the date column to datetime
    - Removing duplicate (date, store, item) keys (keep first)
    - Coercing store/item/sales to numeric
    - Clipping negative sales to 0 (returns are out of scope here)
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataCleaner":
        """No fitting required."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean the DataFrame."""
        df = X.copy()

        if DATE_COLUMN in df.columns:
            df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
            n_bad_dates = int(df[DATE_COLUMN].isna().sum())
            if n_bad_dates:
                logger.warning(f"Dropping {n_bad_dates} rows with unparseable dates.")
                df = df.dropna(subset=[DATE_COLUMN])

        for col in ("store", "item", TARGET):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Duplicate (date, store, item) keys would double-count sales
        key_cols = [c for c in (DATE_COLUMN, "store", "item") if c in df.columns]
        if len(key_cols) == 3:
            n_before = len(df)
            df = df.drop_duplicates(subset=key_cols, keep="first")
            n_dropped = n_before - len(df)
            if n_dropped:
                logger.info(f"Dropped {n_dropped} duplicate (date, store, item) rows.")

        if TARGET in df.columns:
            n_negative = int((df[TARGET] < 0).sum())
            if n_negative:
                logger.warning(f"Clipping {n_negative} negative sales values to 0.")
                df[TARGET] = df[TARGET].clip(lower=0)

        return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply initial cleaning to the raw DataFrame.

    Convenience wrapper around DataCleaner for use outside sklearn
    Pipelines (e.g., in EDA notebooks).

    Args:
        df: Raw DataFrame from data_loader.

    Returns:
        Cleaned DataFrame ready for splitting / feature engineering.
    """
    cleaner = DataCleaner()
    df_clean = cleaner.transform(df)
    logger.info(
        f"Cleaning complete: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} cols"
    )
    return df_clean


# =============================================================================
# Chronological Train / Validation / Test Split
# =============================================================================

def split_data(
    df: pd.DataFrame,
    target: str = TARGET,
    train_end: str = TRAIN_END_DATE,
    val_end: str = VAL_END_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Split data chronologically into Train / Validation / Test.

    A random split would leak future information into training (the model
    would learn from days it is later asked to predict). Instead:

        Train      : ..            .. train_end   (~90%)
        Validation : train_end+1   .. val_end     (3 months — model selection)
        Test       : val_end+1     .. end         (3 months — final estimate)

    The 3-month validation/test windows match the competition's forecast
    horizon. Output frames are sorted by date so TimeSeriesSplit inside
    the hyperparameter search sees chronological folds.

    Args:
        df: Cleaned DataFrame with date and target columns.
        target: Target column name.
        train_end: Last date (inclusive) of the training window.
        val_end: Last date (inclusive) of the validation window.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    train_mask = df[DATE_COLUMN] <= pd.Timestamp(train_end)
    val_mask = (df[DATE_COLUMN] > pd.Timestamp(train_end)) & \
               (df[DATE_COLUMN] <= pd.Timestamp(val_end))
    test_mask = df[DATE_COLUMN] > pd.Timestamp(val_end)

    feature_cols = [c for c in df.columns if c != target]
    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target]
    X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, target]
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, target]

    logger.info(
        f"Chronological split — "
        f"Train: {len(X_train):,} rows (.. {train_end}), "
        f"Val: {len(X_val):,} rows (.. {val_end}), "
        f"Test: {len(X_test):,} rows"
    )
    logger.info(
        f"Mean sales — Train: {y_train.mean():.2f}, "
        f"Val: {y_val.mean():.2f}, Test: {y_test.mean():.2f}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# Sklearn Preprocessing Pipeline
# =============================================================================

def build_preprocessing_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    flag_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles all feature transformations.

    Architecture:
    - Numeric features (entity IDs, calendar numerics, learned aggregates)
      → median imputation. No scaling: all three models are tree-based and
      split on thresholds, so scale is irrelevant.
    - Calendar categoricals (month, dayofweek) → OneHotEncoder
      (drop=None + handle_unknown="ignore" keeps unseen values
      distinguishable from every known category)
    - Flags (is_weekend, ...) → passthrough
    - Everything else → DROPPED. Deliberate safety net: extra columns in
      inference data (`sales`, `id`, arbitrary CSV columns) are ignored
      instead of crashing or leaking into the model.

    This transformer is fit ONLY on training data to prevent leakage.

    Returns:
        Configured ColumnTransformer.
    """
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = CALENDAR_CATEGORICAL if categorical_features is None else categorical_features
    flag_features = FLAG_FEATURES if flag_features is None else flag_features

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(
            drop=None,
            sparse_output=False,
            handle_unknown="ignore",
        )),
    ])

    transformers = [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
    if flag_features:
        transformers.append(("flag", "passthrough", flag_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_default_preprocessor() -> ColumnTransformer:
    """Build the default preprocessor using the standard feature definitions."""
    return build_preprocessing_pipeline()
