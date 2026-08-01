"""Lambda handler: serverless demand-forecast inference.

Dependency-light by design: numpy + xgboost only (packaged as xgboost-cpu).
The feature engineering that the training pipeline does with pandas/sklearn
is rebuilt here from exported primitives (artifacts.json) — see
export_for_lambda.py for why, and parity_check.py for the proof that both
paths produce identical predictions.

Cold-start optimization: the model and artifacts are downloaded from S3 and
parsed ONCE per container (module-level cache), then reused across warm
invocations — subsequent calls skip S3 entirely.

Request body (POST /forecast):
    {"date": "2018-01-15", "store": 1, "item": 1}
    {"start_date": "2018-01-15", "periods": 14, "store": 1, "item": 1}

Response:
    {"store": 1, "item": 1, "forecast": [{"date": ..., "predicted_sales": ...}, ...]}
"""

import calendar
import datetime as dt
import json
import math
import os

import numpy as np
import xgboost as xgb

S3_BUCKET = os.environ.get("MODEL_BUCKET", "")
S3_PREFIX = os.environ.get("MODEL_PREFIX", "models")
LOCAL_ARTIFACTS = os.environ.get("LOCAL_ARTIFACTS", "")  # dir path for local tests

# Warm-invocation cache (survives across calls in the same Lambda container)
_CACHE = {"booster": None, "art": None}


def _load():
    if _CACHE["booster"] is not None:
        return _CACHE["booster"], _CACHE["art"]

    if LOCAL_ARTIFACTS:  # local testing / parity checks — no AWS needed
        model_path = os.path.join(LOCAL_ARTIFACTS, "model_xgb.json")
        art = json.load(open(os.path.join(LOCAL_ARTIFACTS, "artifacts.json"),
                             encoding="utf-8"))
    else:
        import boto3  # imported lazily so local tests don't require it
        s3 = boto3.client("s3")
        model_path = "/tmp/model_xgb.json"
        s3.download_file(S3_BUCKET, f"{S3_PREFIX}/model_xgb.json", model_path)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/artifacts.json")
        art = json.loads(obj["Body"].read())

    booster = xgb.Booster()
    booster.load_model(model_path)
    _CACHE["booster"], _CACHE["art"] = booster, art
    return booster, art


def _engineer(date: dt.date, store: int, item: int, art: dict) -> dict:
    """Mirror of pipeline_lib.FeatureEngineer.transform for one row."""
    dow = date.weekday()
    month = date.month
    doy = date.timetuple().tm_yday
    week = date.isocalendar()[1]
    last_day = calendar.monthrange(date.year, date.month)[1]

    g = art["global_mean"]
    si = art["store_item_mean"].get(f"{store}|{item}", g)
    si_dow = art["store_item_dow_mean"].get(f"{store}|{item}|{dow}", si)
    im = art["item_month_mean"].get(f"{item}|{month}", g)
    sm = art["store_month_mean"].get(f"{store}|{month}", g)

    return {
        "store": store, "item": item,
        "year": date.year, "month": month, "day": date.day,
        "dayofweek": dow, "dayofyear": doy, "weekofyear": week,
        "is_weekend": 1 if dow >= 5 else 0,
        "is_month_start": 1 if date.day == 1 else 0,
        "is_month_end": 1 if date.day == last_day else 0,
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "store_item_mean": si, "store_item_dow_mean": si_dow,
        "item_month_mean": im, "store_month_mean": sm,
    }


def _vectorize(rows: list, art: dict) -> np.ndarray:
    """Assemble the exact training-time feature matrix:
    [numeric block | one-hot blocks | flag block] in the fitted order."""
    out = np.zeros((len(rows), len(art["feature_order"])), dtype=np.float32)
    for i, f in enumerate(rows):
        j = 0
        for k, col in enumerate(art["num_cols"]):
            v = f.get(col)
            out[i, j] = art["num_medians"][k] if v is None else v
            j += 1
        for col, cats in zip(art["cat_cols"], art["cat_categories"]):
            v = f.get(col)
            for c in cats:
                out[i, j] = 1.0 if v == c else 0.0
                j += 1
        for col in art["flag_cols"]:
            out[i, j] = f.get(col, 0)
            j += 1
    return out


def predict(dates: list, store: int, item: int) -> list:
    booster, art = _load()
    rows = [_engineer(d, store, item, art) for d in dates]
    X = _vectorize(rows, art)
    preds = booster.predict(xgb.DMatrix(X))
    return [{"date": d.isoformat(), "predicted_sales": round(max(float(p), 0.0), 2)}
            for d, p in zip(dates, preds)]


def lambda_handler(event, context):
    try:
        body = event.get("body") or "{}"
        if isinstance(body, str):
            body = json.loads(body)

        store = int(body.get("store", 1))
        item = int(body.get("item", 1))
        if not (1 <= store <= 10 and 1 <= item <= 50):
            return _resp(400, {"error": "store must be 1-10, item 1-50"})

        if "start_date" in body:
            start = dt.date.fromisoformat(body["start_date"])
            periods = min(int(body.get("periods", 7)), 365)
            dates = [start + dt.timedelta(days=i) for i in range(periods)]
        else:
            dates = [dt.date.fromisoformat(body.get("date", "2018-01-01"))]

        forecast = predict(dates, store, item)
        _, art = _load()
        return _resp(200, {
            "store": store, "item": item, "forecast": forecast,
            "model": art["model_meta"]["best_model"],
            "expected_smape": art["model_meta"]["test_smape"],
        })
    except (ValueError, KeyError) as e:
        return _resp(400, {"error": f"bad request: {e}"})


def _resp(code, payload):
    return {"statusCode": code,
            "headers": {"Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"},
            "body": json.dumps(payload)}
