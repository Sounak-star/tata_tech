"""
balance_data.py — Pipeline A / Data & Feature Engineering Lead (Dev 1)

Step 2 of the fatigue data pipeline: clean the raw MediaPipe features and balance
the rare drowsiness/yawn classes with SMOTE, producing a training-ready CSV for
the XGBoost trainer (train_xgboost.py).

    raw_features.csv  →  clean (NaNs, dtypes)  →  SMOTE  →  training_ready_features.csv

Run:
    python balance_data.py --in raw_features_full.csv --out training_ready_features.csv
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURE_NAMES = os.path.join(HERE, "feature_names.json")
LABEL_COL = "label"


def load_feature_list() -> list[str]:
    with open(FEATURE_NAMES, "r") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean + SMOTE-balance fatigue features.")
    ap.add_argument("--in", dest="inp", default="raw_features_full.csv")
    ap.add_argument("--out", dest="out", default="training_ready_features.csv")
    ap.add_argument("--label", default=LABEL_COL)
    args = ap.parse_args()

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise SystemExit(f"imbalanced-learn required: pip install imbalanced-learn ({exc})")

    inp = args.inp if os.path.isabs(args.inp) else os.path.join(HERE, args.inp)
    df = pd.read_csv(inp)
    print(f"loaded {len(df):,} rows from {inp}")

    feats = [c for c in load_feature_list() if c in df.columns]
    if args.label not in df.columns:
        raise SystemExit(f"label column '{args.label}' not found. Columns: {list(df.columns)}")

    # Clean: coerce numeric, drop rows with missing features/labels.
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feats + [args.label]).reset_index(drop=True)

    X = df[feats].to_numpy(dtype=np.float32)
    y = df[args.label].astype(int).to_numpy()
    print("class counts (before):", dict(pd.Series(y).value_counts().sort_index()))

    # SMOTE — balance the minority fatigue/low-vigilance classes.
    k = max(1, min(5, int(pd.Series(y).value_counts().min()) - 1))
    X_res, y_res = SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
    print("class counts (after): ", dict(pd.Series(y_res).value_counts().sort_index()))

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    balanced = pd.DataFrame(X_res, columns=feats)
    balanced[args.label] = y_res
    balanced.to_csv(out, index=False)
    print(f"saved {len(balanced):,} balanced rows → {out}")


if __name__ == "__main__":
    main()
