#!/usr/bin/env python3
"""
scripts/make_index.py

Command-line tool to build the dataset index CSV.

Example:
    python scripts/make_index.py \
        --output-root output \
        --index-csv data/processed/localizer_index.csv \
        --test-frac 0.15 \
        --val-frac 0.15 \
        --seed 0 \
        --require-readable
"""

from __future__ import annotations

import argparse
from pathlib import Path

from localization.data.indexing import build_index, IndexConfig


def parse_args():
    ap = argparse.ArgumentParser(description="Build train/val/test index CSV for the localization dataset.")
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory that contains one subfolder per case (preprocessing output).",
    )
    ap.add_argument(
        "--index-csv",
        type=Path,
        default=Path("data/processed/localizer_index.csv"),
        help="Output CSV path to write (split,case_id,image,meta).",
    )
    ap.add_argument("--test-frac", type=float, default=0.15, help="Fraction of cases for test split.")
    ap.add_argument("--val-frac", type=float, default=0.15, help="Fraction of cases for val split.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed used for shuffling before splitting.")
    ap.add_argument(
        "--require-readable",
        action="store_true",
        help="If set, verify each scan is readable by SimpleITK (slower but safer).",
    )
    ap.add_argument(
        "--no-readable-check",
        action="store_true",
        help="Disable SimpleITK readability checks (faster). Overrides --require-readable.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    require_readable = True if args.require_readable else False
    if args.no_readable_check:
        require_readable = False

    cfg = IndexConfig(
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        require_readable=require_readable,
    )

    summary = build_index(
        output_root=args.output_root,
        index_csv=args.index_csv,
        cfg=cfg,
    )

    print("\n✅ Index built successfully")
    print(f"Index CSV: {summary['index_csv']}")
    print(f"Total usable: {summary['total']}")
    print(f"Train: {summary['train']} | Val: {summary['val']} | Test: {summary['test']}")


if __name__ == "__main__":
    raise SystemExit(main())