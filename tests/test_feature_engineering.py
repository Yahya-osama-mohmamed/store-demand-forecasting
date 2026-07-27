import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import create_calendar_features, FeatureEngineer


@pytest.fixture
def sample_sales_data():
    """Two stores x two items over 14 days with known sales."""
    dates = pd.date_range("2017-01-01", periods=14, freq="D")
    rows = []
    for store in (1, 2):
        for item in (1, 2):
            for i, d in enumerate(dates):
                rows.append({
                    "date": d,
                    "store": store,
                    "item": item,
                    # store 1 sells 10/day, store 2 sells 20/day (+item offset)
                    "sales": store * 10 + item + (5 if d.dayofweek >= 5 else 0),
                })
    return pd.DataFrame(rows)


def test_create_calendar_features():
    """Calendar features must be correct for a known date."""
    df = pd.DataFrame({"date": ["2017-12-25"], "store": [1], "item": [1]})
    result = create_calendar_features(df)

    # 2017-12-25 was a Monday
    assert result.loc[0, "year"] == 2017
    assert result.loc[0, "month"] == 12
    assert result.loc[0, "day"] == 25
    assert result.loc[0, "dayofweek"] == 0
    assert result.loc[0, "is_weekend"] == 0
    assert result.loc[0, "is_month_start"] == 0
    assert result.loc[0, "is_month_end"] == 0

    # Cyclical encodings must be on the unit circle
    assert result.loc[0, "month_sin"] ** 2 + result.loc[0, "month_cos"] ** 2 == pytest.approx(1.0)
    assert result.loc[0, "dow_sin"] ** 2 + result.loc[0, "dow_cos"] ** 2 == pytest.approx(1.0)


def test_calendar_weekend_flag():
    """Saturday and Sunday must be flagged as weekend."""
    df = pd.DataFrame({
        "date": ["2017-01-06", "2017-01-07", "2017-01-08"],  # Fri, Sat, Sun
        "store": [1, 1, 1], "item": [1, 1, 1],
    })
    result = create_calendar_features(df)
    assert list(result["is_weekend"]) == [0, 1, 1]


def test_feature_engineer_learns_aggregates_at_fit(sample_sales_data):
    """
    Train/serve consistency: aggregates must come from the TRAINING data
    seen in fit, not from the data being transformed.
    """
    X = sample_sales_data.drop(columns=["sales"])
    y = sample_sales_data["sales"]

    fe = FeatureEngineer()
    fe.fit(X, y)

    # Global mean must equal the training sales mean
    assert fe.global_mean_ == pytest.approx(y.mean())

    # Single future row for a known entity → gets that entity's training mean
    future = pd.DataFrame({"date": ["2018-06-15"], "store": [2], "item": [1]})
    out = fe.transform(future)
    expected_mean = sample_sales_data[
        (sample_sales_data["store"] == 2) & (sample_sales_data["item"] == 1)
    ]["sales"].mean()
    assert out.loc[0, "store_item_mean"] == pytest.approx(expected_mean)


def test_feature_engineer_unseen_entity_fallback(sample_sales_data):
    """Unseen (store, item) combinations must fall back to the global mean, not NaN."""
    X = sample_sales_data.drop(columns=["sales"])
    y = sample_sales_data["sales"]

    fe = FeatureEngineer().fit(X, y)
    unseen = pd.DataFrame({"date": ["2018-06-15"], "store": [9], "item": [42]})
    out = fe.transform(unseen)

    agg_cols = ["store_item_mean", "store_item_dow_mean",
                "item_month_mean", "store_month_mean"]
    assert not out[agg_cols].isna().any().any()
    assert out.loc[0, "store_item_mean"] == pytest.approx(fe.global_mean_)


def test_feature_engineer_requires_y(sample_sales_data):
    """fit without y cannot learn sales aggregates and must fail fast."""
    X = sample_sales_data.drop(columns=["sales"])
    with pytest.raises(ValueError, match="requires y"):
        FeatureEngineer().fit(X)


def test_feature_engineer_preserves_row_order(sample_sales_data):
    """Merges inside transform must not reorder or reindex rows."""
    X = sample_sales_data.drop(columns=["sales"])
    y = sample_sales_data["sales"]
    fe = FeatureEngineer().fit(X, y)

    shuffled = X.sample(frac=1.0, random_state=0)
    out = fe.transform(shuffled)

    assert list(out.index) == list(shuffled.index)
    assert list(out["store"]) == list(shuffled["store"])
    assert list(out["item"]) == list(shuffled["item"])
