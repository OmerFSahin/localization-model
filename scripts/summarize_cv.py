#!/usr/bin/env python3
"""
Summarize K-fold CV results from evaluation JSON files.

Expected usage:
1) For each fold, save eval JSON, e.g.
   outputs/cv5_resnet_fold0/val_best_iou.json
   outputs/cv5_resnet_fold1/val_best_iou.json
   ...

2) Then run:
   python scripts/summarize_cv.py \
     --glob "outputs/cv5_resnet_fold*/val_best_iou.json"

This script prints:
- per-fold metrics
- mean ± std across folds
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean, pstdev


METRIC_KEYS = [
    "median_center_error_mm",
    "mean_center_error_mm",
    "p_at_thresh",
    "mean_iou",
    "n",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        type=str,
        required=True,
        help='Glob pattern for per-fold eval JSON files, e.g. "outputs/cv5_resnet_fold*/val_best_iou.json"',
    )
    return ap.parse_args()


def fmt_mean_std(vals, scale: float = 1.0, suffix: str = "") -> str:
    vals = [float(v) * scale for v in vals]
    if len(vals) == 1:
        return f"{vals[0]:.4f}{suffix}"
    return f"{mean(vals):.4f} ± {pstdev(vals):.4f}{suffix}"


def main():
    args = parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")

    rows = []
    for p in paths:
        with open(p, "r") as f:
            obj = json.load(f)

        metrics = obj.get("metrics", {})
        row = {
            "path": p,
            "split": obj.get("split"),
            "ckpt": obj.get("ckpt"),
        }
        for k in METRIC_KEYS:
            if k not in metrics:
                raise KeyError(f"Metric '{k}' missing in {p}")
            row[k] = metrics[k]
        rows.append(row)

    print("\nPer-fold results")
    print("-" * 100)
    for i, r in enumerate(rows):
        print(
            f"[{i}] {Path(r['path']).parent.name} | "
            f"medCE={r['median_center_error_mm']:.4f} mm | "
            f"meanCE={r['mean_center_error_mm']:.4f} mm | "
            f"P@T={100.0 * float(r['p_at_thresh']):.2f}% | "
            f"mIoU={r['mean_iou']:.4f} | "
            f"n={int(r['n'])}"
        )

    print("\nAggregate summary")
    print("-" * 100)
    medce_vals = [r["median_center_error_mm"] for r in rows]
    meance_vals = [r["mean_center_error_mm"] for r in rows]
    p_vals = [r["p_at_thresh"] for r in rows]
    miou_vals = [r["mean_iou"] for r in rows]
    n_vals = [r["n"] for r in rows]

    print(f"Folds: {len(rows)}")
    print(f"median_center_error_mm : {fmt_mean_std(medce_vals)}")
    print(f"mean_center_error_mm   : {fmt_mean_std(meance_vals)}")
    print(f"p_at_thresh            : {fmt_mean_std(p_vals, scale=100.0, suffix='%')}")
    print(f"mean_iou               : {fmt_mean_std(miou_vals)}")
    print(f"n per fold             : {fmt_mean_std(n_vals)}")

    best_fold_idx = max(range(len(rows)), key=lambda i: float(rows[i]["mean_iou"]))
    worst_fold_idx = min(range(len(rows)), key=lambda i: float(rows[i]["mean_iou"]))

    print("\nBest / worst by mean_iou")
    print("-" * 100)
    best = rows[best_fold_idx]
    worst = rows[worst_fold_idx]
    print(
        f"Best  : {Path(best['path']).parent.name} | "
        f"mIoU={best['mean_iou']:.4f} | "
        f"medCE={best['median_center_error_mm']:.4f} mm | "
        f"P@T={100.0 * float(best['p_at_thresh']):.2f}%"
    )
    print(
        f"Worst : {Path(worst['path']).parent.name} | "
        f"mIoU={worst['mean_iou']:.4f} | "
        f"medCE={worst['median_center_error_mm']:.4f} mm | "
        f"P@T={100.0 * float(worst['p_at_thresh']):.2f}%"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())