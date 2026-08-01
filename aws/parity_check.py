"""Prove the Lambda inference path == the real sklearn pipeline.

Compares predictions from aws/lambda_function.py (numpy + stdlib feature
rebuild) against pipeline.predict() on random (date, store, item) triples
spanning training years AND unseen future dates. Any drift means the export
or the feature rebuild is wrong — this must pass before deploying.

Run:  python -m aws.parity_check   (from the project root)
"""

import datetime as dt
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["LOCAL_ARTIFACTS"] = str(ROOT / "aws" / "build")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aws import lambda_function as lf  # noqa: E402
from pipeline_lib import FINAL_PIPELINE_PATH  # noqa: E402


def main(n: int = 500) -> None:
    rng = random.Random(42)
    pipe = joblib.load(FINAL_PIPELINE_PATH)

    dates = [dt.date(2013, 1, 1) + dt.timedelta(days=rng.randrange(0, 2920))
             for _ in range(n)]  # 2013..2020 → includes unseen future dates
    stores = [rng.randrange(1, 11) for _ in range(n)]
    items = [rng.randrange(1, 51) for _ in range(n)]

    # Lambda path
    lambda_preds = np.array([
        lf.predict([d], s, i)[0]["predicted_sales"]
        for d, s, i in zip(dates, stores, items)
    ])

    # Real pipeline path (predictions clipped at 0 like the serving layer)
    df = pd.DataFrame({"date": [d.isoformat() for d in dates],
                       "store": stores, "item": items})
    pipe_preds = np.clip(pipe.predict(df), 0, None)

    # Lambda rounds to 2dp for the API response — compare at that precision
    diff = np.abs(lambda_preds - np.round(pipe_preds, 2))
    print(f"samples: {n}")
    print(f"max abs diff : {diff.max():.6f}")
    print(f"mean abs diff: {diff.mean():.6f}")
    assert diff.max() <= 0.011, "PARITY FAILED — do not deploy"
    print("PARITY OK — Lambda inference path matches the trained pipeline")


if __name__ == "__main__":
    main()
