#!/usr/bin/env python3
"""
Run K-fold cross-validation end-to-end:
- train each fold
- evaluate best_iou.pt on validation split of each fold
- save per-fold JSON
- run summary

Example:
    python scripts/run_cv.py \
        --index-csv data/processed/localizer_index_cv5.csv \
        --out-root outputs/cv5_resnet_base16_logmm \
        --n-folds 5 \
        --model resnet3d_regressor \
        --base 16 \
        --epochs 50 \
        --device cuda \
        --positive-size \
        --size-target log_mm \
        --size-loss-w 1.0 \
        --num-workers 2 \
        --pin-memory
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index-csv", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fold-col", type=str, default="fold")

    # model / training
    ap.add_argument("--model", type=str, required=True,
                    choices=["unet3d", "cnn3d_regressor", "resnet3d_regressor"])
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--positive-size", action="store_true")

    # preprocessing / targets
    ap.add_argument("--target-spacing", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    ap.add_argument("--heat-sigma", type=float, default=3.0)
    ap.add_argument("--ct-clip", type=float, nargs=2, default=[-150.0, 350.0])
    ap.add_argument("--pad-multiple", type=int, default=8)
    ap.add_argument("--heatmap-method", type=str, default="separable",
                    choices=["separable", "meshgrid"])
    ap.add_argument("--size-target", type=str, default="mm", choices=["mm", "log_mm"])

    # loader / optimization
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--size-loss-w", type=float, default=0.1)
    ap.add_argument("--size-loss", type=str, default="mse", choices=["mse", "l1", "smooth_l1"])
    ap.add_argument("--log-every", type=int, default=1)

    # scheduler
    ap.add_argument("--scheduler", type=str, default=None, choices=["step"])
    ap.add_argument("--scheduler-step-size", type=int, default=15)
    ap.add_argument("--scheduler-gamma", type=float, default=0.5)

    # AMP
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    ap.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])

    # metrics / device
    ap.add_argument("--p-thresh-mm", type=float, default=20.0)
    ap.add_argument("--min-size-mm", type=float, default=10.0)
    ap.add_argument("--device", type=str, default=None)

    # cache options
    ap.add_argument("--use-cache", action="store_true", help="Use cached preprocessed samples.")
    ap.add_argument("--cache-index-csv", type=Path, default=None, help="Path to cache_index.csv when using cache.")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.use_cache and args.cache_index_csv is None:
        raise ValueError("--cache-index-csv must be provided when --use-cache is set")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    for fold in range(args.n_folds):
        fold_dir = out_root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            py, "scripts/train.py",
            "--index-csv", str(args.index_csv),
            "--outdir", str(fold_dir),
            "--cv-fold", str(fold),
            "--fold-col", str(args.fold_col),
            "--model", str(args.model),
            "--base", str(args.base),
            "--dropout", str(args.dropout),
            "--target-spacing", *map(str, args.target_spacing),
            "--heat-sigma", str(args.heat_sigma),
            "--ct-clip", *map(str, args.ct_clip),
            "--pad-multiple", str(args.pad_multiple),
            "--heatmap-method", str(args.heatmap_method),
            "--size-target", str(args.size_target),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--size-loss-w", str(args.size_loss_w),
            "--size-loss", str(args.size_loss),
            "--log-every", str(args.log_every),
            "--p-thresh-mm", str(args.p_thresh_mm),
            "--min-size-mm", str(args.min_size_mm),
        ]

        if args.pin_memory:
            train_cmd.append("--pin-memory")
        if args.positive_size:
            train_cmd.append("--positive-size")
        if args.device is not None:
            train_cmd += ["--device", str(args.device)]
        if args.scheduler is not None:
            train_cmd += [
                "--scheduler", str(args.scheduler),
                "--scheduler-step-size", str(args.scheduler_step_size),
                "--scheduler-gamma", str(args.scheduler_gamma),
            ]
        if args.amp:
            train_cmd.append("--amp")
            train_cmd += ["--amp-dtype", str(args.amp_dtype)]

        if args.use_cache:
            train_cmd.append("--use-cache")
            train_cmd += ["--cache-index-csv", str(args.cache_index_csv)]

        run_cmd(train_cmd)

        eval_json = fold_dir / "val_best_iou.json"
        eval_cmd = [
            py, "scripts/eval.py",
            "--index-csv", str(args.index_csv),
            "--ckpt", str(fold_dir / "best_iou.pt"),
            "--model", str(args.model),
            "--base", str(args.base),
            "--dropout", str(args.dropout),
            "--size-target", str(args.size_target),
            "--cv-fold", str(fold),
            "--cv-mode", "val",
            "--fold-col", str(args.fold_col),
            "--num-workers", "0",
            "--p-thresh-mm", str(args.p_thresh_mm),
            "--min-size-mm", str(args.min_size_mm),
            "--out-json", str(eval_json),
        ]

        if args.positive_size:
            eval_cmd.append("--positive-size")
        if args.device is not None:
            eval_cmd += ["--device", str(args.device)]

        run_cmd(eval_cmd)

    summary_cmd = [
        py, "scripts/summarize_cv.py",
        "--glob", str(out_root / "fold*" / "val_best_iou.json"),
    ]
    run_cmd(summary_cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())