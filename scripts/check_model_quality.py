#!/usr/bin/env python
"""Fail the build if the shipped model no longer clears its floor.

A retrain that quietly degrades is the failure mode this catches: the notebook
still runs, the tests still pass, and a worse model reaches the image. The
floors below sit just under the numbers the README publishes, so genuine noise
between retrains passes while a real regression stops the pipeline.

Reads the metadata the notebook writes; the test set is never re-scored here.
"""

import json
import sys
from pathlib import Path

METADATA = Path(__file__).resolve().parent.parent / "models" / "model_metadata.json"

# SMAPE is an error metric: lower is better, so its bound is a ceiling.
# (metric, comparison, bound, why this number)
CHECKS = [
    ("test_smape", "max", 13.0, "published 12.33; the public leaderboard band is ~12.5-14"),
    ("val_smape", "max", 11.5, "published 10.95"),
    ("test_r2", "min", 0.90, "published 0.928"),
    ("test_bias_pct", "min", -10.0, "systematic under-forecasting starves inventory"),
    ("test_bias_pct", "max", 10.0, "systematic over-forecasting ties up capital"),
]


def main() -> int:
    if not METADATA.exists():
        print(f"FAIL: {METADATA} not found - run the notebook before shipping.")
        return 1

    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    print(f"model    : {meta.get('best_model')}")
    print(f"trained  : {meta.get('trained_at_utc')}")
    print()

    failures = []
    for metric, kind, bound, why in CHECKS:
        value = meta.get(metric)
        if value is None:
            failures.append(f"{metric} missing from metadata")
            continue
        ok = value >= bound if kind == "min" else value <= bound
        symbol = ">=" if kind == "min" else "<="
        print(f"{'PASS' if ok else 'FAIL'}  {metric:20} {value:<10} {symbol} {bound}   ({why})")
        if not ok:
            failures.append(f"{metric}={value} violates {symbol} {bound}")

    print()
    if failures:
        print("Model quality gate FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Model quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
