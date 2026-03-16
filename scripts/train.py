#!/usr/bin/env python3
"""
scripts/train.py

Train the localization model.

Supported models:
- unet3d (default)
    Main U-Net style model with skip connections.
    Best choice when fine spatial localization is important.

- cnn3d_regressor
    Lightweight plain 3D CNN baseline.
    Useful as a simple baseline and for debugging / quick experiments.

- resnet3d_regressor
    Residual 3D CNN baseline.
    A stronger encoder-style alternative without a U-Net decoder.

Example:
    python scripts/train.py \
        --index-csv data/processed/localizer_index.csv \
        --outdir outputs/run01 \
        --model unet3d \
        --epochs 50 \
        --lr 1e-4 \
        --weight-decay 1e-4 \
        --base 16 \
        --batch-size 1 \
        --num-workers 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from localization.data.dataset import SampleConfig
from localization.data.dataloaders import build_loaders, LoaderConfig
from localization.train.losses import LossConfig
from localization.eval.metrics import ValConfig
from localization.train.trainer import train, TrainConfig
from localization.models.factory import build_model

def parse_args():
    ap = argparse.ArgumentParser(description="Train 3D localizer (heatmap + size).")

    # Paths
    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/localizer_index.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("outputs/run01"))

    # Model
    ap.add_argument(
    "--model",
    type=str,
    default="unet3d",
    choices=["unet3d", "cnn3d_regressor", "resnet3d_regressor"],
    help="Model architecture to train.",
)
    ap.add_argument("--base", type=int, default=16, help="Base channel count for the model.")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--positive-size", action="store_true", help="Use softplus to enforce size > 0.")

    # Dataset / targets
    ap.add_argument("--target-spacing", type=float, nargs=3, default=[2.0, 2.0, 2.0], help="Target spacing (x y z) mm.")
    ap.add_argument("--heat-sigma", type=float, default=3.0, help="Heatmap sigma in voxels on resampled grid.")
    ap.add_argument("--ct-clip", type=float, nargs=2, default=[-150.0, 350.0], help="CT clip window (min max).")
    ap.add_argument("--pad-multiple", type=int, default=8, help="Pad (Z,Y,X) to a multiple of this value.")
    ap.add_argument("--heatmap-method", type=str, default="separable", choices=["separable", "meshgrid"])
    ap.add_argument(
    "--size-target",
    type=str,
    default="mm",
    choices=["mm", "log_mm"],
    help="Target representation for bbox size regression.",
)

    # Loader
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")

    # Training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--size-loss-w", type=float, default=0.1, help="Weight for size regression loss.")
    ap.add_argument("--log-every", type=int, default=1)

    # Validation metrics
    ap.add_argument("--p-thresh-mm", type=float, default=20.0, help="Success threshold for P@T (mm).")
    ap.add_argument("--min-size-mm", type=float, default=10.0, help="Clamp predicted size (mm) for IoU metric.")

    # Device
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto).")

    return ap.parse_args()


def main():
    args = parse_args()

    # ---- build dataset config ----
    sample_cfg = SampleConfig(
        target_spacing_xyz=(args.target_spacing[0], args.target_spacing[1], args.target_spacing[2]),
        heat_sigma_vox=float(args.heat_sigma),
        ct_clip=(float(args.ct_clip[0]), float(args.ct_clip[1])),
        pad_multiple=int(args.pad_multiple),
        heatmap_method=str(args.heatmap_method),
        size_target=str(args.size_target),
    )

    # ---- build loader configs ----
    train_loader_cfg = LoaderConfig(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )

    val_loader_cfg = LoaderConfig(
        batch_size=1,  # keep val simple / stable
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )

    train_ds, val_ds, train_dl, val_dl = build_loaders(
        index_csv=args.index_csv,
        sample_cfg=sample_cfg,
        train_loader_cfg=train_loader_cfg,
        val_loader_cfg=val_loader_cfg,
    )

    print("Requested device:", args.device)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device 0:", torch.cuda.get_device_name(0))
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Model: {args.model} | base={args.base} | dropout={args.dropout} | positive_size={args.positive_size}")

    # ---- model ----
    net = build_model(
        name=args.model,
        base=int(args.base),
        dropout=float(args.dropout),
        positive_size=bool(args.positive_size),
    )
    # ---- training config ----
    loss_cfg = LossConfig(heat_loss="mse", size_weight=float(args.size_loss_w))
    val_cfg = ValConfig(clamp_min_size_mm=float(args.min_size_mm), success_thresh_mm=float(args.p_thresh_mm),     size_target=str(args.size_target),)

    train_cfg = TrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        loss_cfg=loss_cfg,
        val_cfg=val_cfg,
        best_metric="median_center_error_mm",
        maximize_best_metric=False,
        log_every=int(args.log_every),
    )

    # ---- run ----
    result = train(
        net=net,
        train_dl=train_dl,
        val_dl=val_dl,
        outdir=args.outdir,
        cfg=train_cfg,
        device=args.device,
        optimizer=None,  # uses AdamW by default
    )

    print("\n✅ Training finished.")
    print("Best checkpoint:", result["best_path"])
    print("Last checkpoint:", result["last_path"])
    print("History:", result["history_path"])
    print("Best epoch:", result["best_epoch"], "| Best metric:", result["best_metric_value"])


if __name__ == "__main__":
    raise SystemExit(main())