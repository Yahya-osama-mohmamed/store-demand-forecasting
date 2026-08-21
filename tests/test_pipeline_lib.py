"""Tests for the transformer the notebook trains and the API serves.

The focus is the two things that would silently break a forecast: an aggregate
learned from data the model should not have seen, and an unseen store or item
producing NaN instead of a sensible fallback.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline_lib import (
    AGGREGATE_FEATURES,
    DATE_COLUMN,
    FeatureEngineer,
    create_calendar_features,
    smape,
)


def make_frame(days: int = 60, stores=(1, 2), items=(1, 2)) -> pd.DataFrame:
    """A small but structurally faithful panel: every (store, item) x every day."""
    dates = pd.date_range("2017-01-01", periods=days, freq="D")
    rows = []
    for d in dates:
        for s in stores:
            for i in items:
                # Level depends on the entity, shape depends on the weekday -
                # the same structure the real data has.
                rows.append({DATE_COLUMN: d, "store": s, "item": i,
                             "sales": 10 * s + i + (d.dayofweek >= 5) * 5})
    return pd.DataFrame(rows)


class TestCalendarFeatures:
    def test_creates_expected_columns(self):
        out = create_calendar_features(make_frame(7))
        for col in ("year", "month", "day", "dayofweek", "dayofyear", "weekofyear",
                    "is_weekend", "is_month_start", "is_month_end",
                    "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"):
            assert col in out.columns

    def test_cyclical_encoding_wraps_around(self):
        """December must sit next to January, not at the opposite end."""
        df = pd.DataFrame({DATE_COLUMN: pd.to_datetime(["2017-12-31", "2018-01-01"]),
                           "store": [1, 1], "item": [1, 1]})
        out = create_calendar_features(df)
        dec, jan = out.iloc[0], out.iloc[1]
        gap = np.hypot(dec["month_sin"] - jan["month_sin"], dec["month_cos"] - jan["month_cos"])
        # One month apart on the unit circle is 2*sin(pi/12) ~= 0.52
        assert gap < 0.6

    def test_weekend_flag(self):
        out = create_calendar_features(pd.DataFrame({
            DATE_COLUMN: pd.to_datetime(["2017-01-07", "2017-01-09"]),  # Sat, Mon
            "store": [1, 1], "item": [1, 1]}))
        assert out["is_weekend"].tolist() == [1, 0]


class TestFeatureEngineer:
    def test_requires_y_to_fit(self):
        """The aggregates are means of sales - without y there is nothing to learn."""
        df = make_frame()
        with pytest.raises(ValueError, match="requires y"):
            FeatureEngineer().fit(df[[DATE_COLUMN, "store", "item"]])

    def test_requires_entity_columns(self):
        df = make_frame()
        with pytest.raises(ValueError, match="store"):
            FeatureEngineer().fit(df[[DATE_COLUMN, "item"]], df["sales"])

    def test_aggregates_come_from_fit_data_only(self):
        """Fitting on early data then transforming later data must not update the
        statistics - that is the leak time-series CV exists to prevent."""
        df = make_frame(days=60)
        early = df[df[DATE_COLUMN] < "2017-02-01"]
        late = df[df[DATE_COLUMN] >= "2017-02-01"].copy()
        late["sales"] = late["sales"] * 100  # a demand explosion the model must not see

        fe = FeatureEngineer().fit(early[[DATE_COLUMN, "store", "item"]], early["sales"])
        learned = fe.global_mean_
        fe.transform(late[[DATE_COLUMN, "store", "item"]])
        assert fe.global_mean_ == learned
        assert learned < early["sales"].max() * 2

    def test_unseen_entity_falls_back_to_global_mean(self):
        df = make_frame()
        fe = FeatureEngineer().fit(df[[DATE_COLUMN, "store", "item"]], df["sales"])
        unseen = pd.DataFrame({DATE_COLUMN: pd.to_datetime(["2018-06-01"]),
                               "store": [99], "item": [999]})
        out = fe.transform(unseen)
        assert out[AGGREGATE_FEATURES].notna().all().all()
        assert out["store_item_mean"].iloc[0] == pytest.approx(fe.global_mean_)

    def test_row_order_and_index_preserved(self):
        """transform() merges four times; merges reorder rows unless handled."""
        df = make_frame(days=10)
        X = df[[DATE_COLUMN, "store", "item"]].sample(frac=1, random_state=0)
        fe = FeatureEngineer().fit(df[[DATE_COLUMN, "store", "item"]], df["sales"])
        out = fe.transform(X)
        assert list(out.index) == list(X.index)
        assert out["store"].tolist() == X["store"].tolist()

    def test_future_dates_work(self):
        """A forecast request is always for a date after training ended."""
        df = make_frame()
        fe = FeatureEngineer().fit(df[[DATE_COLUMN, "store", "item"]], df["sales"])
        future = pd.DataFrame({DATE_COLUMN: pd.to_datetime(["2030-12-25"]),
                               "store": [1], "item": [1]})
        out = fe.transform(future)
        assert out.notna().all().all()

    def test_feature_names_out_matches_transform(self):
        df = make_frame(days=5)
        X = df[[DATE_COLUMN, "store", "item"]]
        fe = FeatureEngineer().fit(X, df["sales"])
        assert list(fe.get_feature_names_out()) == list(fe.transform(X).columns)


class TestSmape:
    def test_perfect_forecast_is_zero(self):
        assert smape([10, 20, 30], [10, 20, 30]) == 0.0

    def test_both_zero_contributes_zero_not_nan(self):
        """0/0 would be NaN; the metric has to treat it as a perfect call."""
        assert smape([0, 10], [0, 10]) == 0.0

    def test_is_symmetric(self):
        assert smape([10], [20]) == pytest.approx(smape([20], [10]))

    def test_bounded_at_200(self):
        assert smape([10], [0]) == pytest.approx(200.0)
