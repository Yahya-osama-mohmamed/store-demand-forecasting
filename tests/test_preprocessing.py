import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_data, split_data, build_preprocessing_pipeline
from src.feature_engineering import FeatureEngineer
from src.config import TARGET, DATE_COLUMN


@pytest.fixture
def sample_raw_data():
    """Small multi-year dataset spanning the split boundaries."""
    dates = pd.date_range("2016-01-01", "2017-12-31", freq="D")
    rows = []
    for store in (1, 2):
        for item in (1,):
            for d in dates:
                rows.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "store": store,
                    "item": item,
                    "sales": 10 + store + d.dayofweek,
                })
    return pd.DataFrame(rows)


def test_clean_data_removes_duplicates_and_negatives(sample_raw_data):
    """Duplicate (date, store, item) keys and negative sales must be handled."""
    df = sample_raw_data.copy()
    df.loc[0, "sales"] = -5
    df_with_dupes = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned = clean_data(df_with_dupes)

    assert len(cleaned) == len(sample_raw_data)
    assert (cleaned["sales"] >= 0).all()
    assert pd.api.types.is_datetime64_any_dtype(cleaned[DATE_COLUMN])


def test_split_data_is_chronological(sample_raw_data):
    """Splits must be strictly ordered in time — no future leaking into train."""
    cleaned = clean_data(sample_raw_data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(cleaned)

    assert X_train[DATE_COLUMN].max() < X_val[DATE_COLUMN].min()
    assert X_val[DATE_COLUMN].max() < X_test[DATE_COLUMN].min()

    # Boundaries from config: train ends 2017-06-30, val ends 2017-09-30
    assert X_train[DATE_COLUMN].max() == pd.Timestamp("2017-06-30")
    assert X_val[DATE_COLUMN].max() == pd.Timestamp("2017-09-30")
    assert len(X_train) + len(X_val) + len(X_test) == len(cleaned)

    # Sorted by date so TimeSeriesSplit sees chronological folds
    assert X_train[DATE_COLUMN].is_monotonic_increasing


def test_preprocessing_pipeline_output(sample_raw_data):
    """Full FeatureEngineer + ColumnTransformer output must be numeric, no NaN."""
    cleaned = clean_data(sample_raw_data)
    X = cleaned.drop(columns=[TARGET])
    y = cleaned[TARGET]

    pipeline = Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", build_preprocessing_pipeline()),
    ])
    X_transformed = pipeline.fit_transform(X, y)

    assert np.issubdtype(np.asarray(X_transformed).dtype, np.number)
    assert not np.isnan(np.asarray(X_transformed, dtype=float)).any()

    # get_feature_names_out must work
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    assert len(names) == X_transformed.shape[1]
    assert "num__store_item_mean" in names
    assert "flag__is_weekend" in names


def test_full_pipeline_single_row_inference(sample_raw_data):
    """
    Train/serve consistency: a pipeline fitted on training data must
    transform a single raw (date, store, item) row — as the API receives
    it — without error, even with extra columns like `sales` or `id`.
    """
    cleaned = clean_data(sample_raw_data)
    X = cleaned.drop(columns=[TARGET])
    y = cleaned[TARGET]

    pipeline = Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", build_preprocessing_pipeline()),
    ])
    pipeline.fit(X, y)

    # Single raw row with extra columns — must be ignored, not crash
    single = pd.DataFrame({
        "date": ["2018-03-01"], "store": [1], "item": [1],
        "sales": [999], "id": ["row-1"],
    })
    transformed = pipeline.transform(single)

    assert transformed.shape[0] == 1
    assert not np.isnan(np.asarray(transformed, dtype=float)).any()
