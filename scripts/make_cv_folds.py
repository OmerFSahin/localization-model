#!/usr/bin/env python3
"""
Create a K-fold CSV from an existing localization index.

Input CSV should contain at least:
- case_id
- image
- meta

Output CSV will keep all existing columns and add:
- fold

Example:
    python scripts/make_cv_folds.py \
        --index-csv data/processed/localizer_index.csv \
        --out-csv data/processed/localizer_index_cv5.csv \
        --n-folds 5 \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.index_csv)

    required = ["case_id", "image", "meta"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.index_csv}")

    # One row per case expected
    if df["case_id"].duplicated().any():
        dupes = df[df["case_id"].duplicated(keep=False)]["case_id"].tolist()
        raise ValueError(f"Duplicate case_id rows found. Example duplicates: {dupes[:10]}")

    n = len(df)
    if args.n_folds < 2:
        raise ValueError("--n-folds must be >= 2")
    if n < args.n_folds:
        raise ValueError(f"Not enough cases ({n}) for {args.n_folds} folds")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)

    folds = np.empty(n, dtype=np.int64)
    for i, idx in enumerate(perm):
        folds[idx] = i % args.n_folds

    out_df = df.copy()
    out_df["fold"] = folds

    out_df = out_df.sort_values(["fold", "case_id"]).reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"Saved: {args.out_csv}")
    print(f"Cases: {len(out_df)}")
    print("Fold counts:")
    print(out_df["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    raise SystemExit(main())