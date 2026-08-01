"""Export the trained pipeline to dependency-light Lambda artifacts.

Why not ship the pickled sklearn pipeline to Lambda?
----------------------------------------------------
The pickle needs scikit-learn + pandas + xgboost (~300MB unzipped — over
Lambda's 250MB zip limit) plus our custom FeatureEngineer class on the import
path. Instead we export the pipeline to primitives:

- the XGBoost Booster as portable JSON (loadable by xgboost alone)
- every learned statistic (aggregate lookup tables, imputer medians,
  one-hot category orders, output column order) as artifacts.json

The Lambda then rebuilds the exact feature vector with numpy + stdlib and
predicts with xgboost-cpu — a ~40MB package. `parity_check.py` proves the
two inference paths agree to float precision before anything is deployed.

Run:  python -m aws.export_for_lambda   (from the project root)
"""

import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_lib import FINAL_PIPELINE_PATH, MODEL_METADATA_PATH  # noqa: E402

BUILD_DIR = Path(__file__).resolve().parent / "build"


def main() -> None:
    BUILD_DIR.mkdir(exist_ok=True)
    pipe = joblib.load(FINAL_PIPELINE_PATH)
    fe = pipe.named_steps["features"]
    prep = pipe.named_steps["preprocessor"]
    regressor = pipe.named_steps["regressor"]

    # ---- 1. Booster as portable JSON ----
    booster = regressor.get_booster()
    booster.save_model(str(BUILD_DIR / "model_xgb.json"))

    # ---- 2. Learned statistics from FeatureEngineer ----
    def keyed(df, cols, value):
        return {"|".join(str(int(r[c])) for c in cols): float(r[value])
                for _, r in df.iterrows()}

    num_pipe = prep.named_transformers_["num"]
    imputer = num_pipe.named_steps["imputer"]
    encoder = prep.named_transformers_["cat"].named_steps["encoder"]

    # Column lists exactly as the fitted ColumnTransformer uses them
    num_cols = list(prep.transformers_[0][2])
    cat_cols = list(prep.transformers_[1][2])
    flag_cols = list(prep.transformers_[2][2])

    artifacts = {
        "model_meta": json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8")),
        "global_mean": float(fe.global_mean_),
        "store_item_mean": keyed(fe.store_item_mean_, ["store", "item"],
                                 "store_item_mean"),
        "store_item_dow_mean": keyed(fe.store_item_dow_mean_,
                                     ["store", "item", "_dow"],
                                     "store_item_dow_mean"),
        "item_month_mean": keyed(fe.item_month_mean_, ["item", "_month"],
                                 "item_month_mean"),
        "store_month_mean": keyed(fe.store_month_mean_, ["store", "_month"],
                                  "store_month_mean"),
        "num_cols": num_cols,
        "num_medians": [float(v) for v in imputer.statistics_],
        "cat_cols": cat_cols,
        "cat_categories": [[int(v) for v in cats] for cats in encoder.categories_],
        "flag_cols": flag_cols,
        "feature_order": [str(f) for f in prep.get_feature_names_out()],
    }
    (BUILD_DIR / "artifacts.json").write_text(json.dumps(artifacts),
                                              encoding="utf-8")

    n_feats = (len(num_cols) + sum(len(c) for c in artifacts["cat_categories"])
               + len(flag_cols))
    assert n_feats == len(artifacts["feature_order"]), \
        f"feature count mismatch: {n_feats} vs {len(artifacts['feature_order'])}"

    print(f"exported to {BUILD_DIR}")
    print(f"  model_xgb.json  : {(BUILD_DIR / 'model_xgb.json').stat().st_size / 1e6:.2f} MB")
    print(f"  artifacts.json  : {(BUILD_DIR / 'artifacts.json').stat().st_size / 1e6:.2f} MB")
    print(f"  features        : {n_feats} (order verified)")


if __name__ == "__main__":
    main()
