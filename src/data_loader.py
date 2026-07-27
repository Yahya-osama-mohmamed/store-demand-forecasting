"""
Data Loader — Download and load the Store Item Demand Forecasting dataset.

Handles:
- Downloading from a direct URL (no Kaggle credentials required)
- Loading and initial validation (schema checks, dtype verification)
- Basic data profiling (shape, date coverage, missing values, duplicates)

The dataset contains 913,000 daily sales records: 10 stores x 50 items
observed every day from 2013-01-01 to 2017-12-31 (500 daily time series).
"""

import pandas as pd
import requests
from pathlib import Path
from typing import Tuple, Dict, Any

from src.config import (
    DATASET_URL, RAW_DATA_FILE, TARGET, DATE_COLUMN, N_STORES, N_ITEMS,
)
from src.logger import get_logger

logger = get_logger(__name__)


# Expected columns in the raw dataset — used as a sanity check
EXPECTED_COLUMNS = ["date", "store", "item", "sales"]


def download_dataset(url: str = DATASET_URL, dest: Path = RAW_DATA_FILE) -> Path:
    """
    Download the dataset from a direct URL if not already present.

    Args:
        url: Direct download URL for the CSV file.
        dest: Local file path to save the downloaded CSV.

    Returns:
        Path to the downloaded (or existing) file.

    Raises:
        requests.HTTPError: If the download fails.
    """
    if dest.exists():
        logger.info(f"Dataset already exists at {dest} — skipping download.")
        return dest

    logger.info(f"Downloading dataset from {url}...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    dest.write_bytes(response.content)
    logger.info(f"Dataset saved to {dest} ({dest.stat().st_size / 1024**2:.1f} MB)")
    return dest


def load_raw_data(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load the raw CSV dataset with type parsing.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        DataFrame with a parsed datetime `date` column.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the schema doesn't match expectations.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Run download_dataset() first."
        )

    logger.info(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=[DATE_COLUMN])

    # --- Schema validation ---
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    logger.info(
        f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns "
        f"({df[DATE_COLUMN].min().date()} .. {df[DATE_COLUMN].max().date()})"
    )
    return df


def get_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data profile for EDA.

    Returns a dictionary with shape, date coverage, entity counts,
    missing values, duplicates, and target statistics.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary containing profile metrics.
    """
    n_rows, n_cols = df.shape

    missing = df.isnull().sum()
    n_duplicates = int(df.duplicated(subset=[DATE_COLUMN, "store", "item"]).sum())

    profile = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "date_min": str(df[DATE_COLUMN].min().date()),
        "date_max": str(df[DATE_COLUMN].max().date()),
        "n_stores": int(df["store"].nunique()),
        "n_items": int(df["item"].nunique()),
        "n_series": int(df.groupby(["store", "item"]).ngroups),
        "n_missing_total": int(missing.sum()),
        "n_duplicate_keys": n_duplicates,
        "sales_mean": round(float(df[TARGET].mean()), 2),
        "sales_median": float(df[TARGET].median()),
        "sales_min": float(df[TARGET].min()),
        "sales_max": float(df[TARGET].max()),
        "n_negative_sales": int((df[TARGET] < 0).sum()),
    }

    logger.info(
        f"Profile: {n_rows:,} rows, {profile['n_series']} store-item series, "
        f"{profile['n_missing_total']} missing values, "
        f"{n_duplicates} duplicate (date, store, item) keys"
    )

    return profile


def load_and_validate() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete data loading pipeline: download → load → profile.

    Returns:
        Tuple of (DataFrame, profile dictionary).
    """
    download_dataset()
    df = load_raw_data()
    profile = get_data_profile(df)
    return df, profile
