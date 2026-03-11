#!/usr/bin/env python3
"""
scripts/eval.py

Evaluate a trained localizer checkpoint on val or test split.

Examples:
    python scripts/eval.py --index-csv data/processed/localizer_index.csv --ckpt outputs/run01/best.pt --split val
    python scripts/eval.py --index-csv data/processed/localizer_index.csv --ckpt outputs/run01/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization.data.dataset import SampleConfig, LocalizerDataset
from localization.data.dataloaders import LoaderConfig
from localization.models.factory import build_model
from localization.eval.metrics import validate_epoch, ValConfig


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate localizer checkpoint on a dataset split.")

    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/localizer_index.csv"))
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint (.pt state_dict).")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--out-json", type=Path, default=None, help="Optional path to save metrics as JSON.")

    # Model
    ap.add_argument("--model", type=str, default="unet3d", choices=["unet3d", "cnn3d_regressor", "resnet3d_regressor"], help="Model architecture used by the checkpoint.",)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--positive-size", action="store_true")

    # Dataset / targets (must match training!)
    ap.add_argument("--target-spacing", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    ap.add_argument("--heat-sigma", type=float, default=3.0)
    ap.add_argument("--ct-clip", type=float, nargs=2, default=[-150.0, 350.0])
    ap.add_argument("--pad-multiple", type=int, default=8)
    ap.add_argument("--heatmap-method", type=str, default="separable", choices=["separable", "meshgrid"])

    # Loader
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")

    # Metrics behavior
    ap.add_argument("--p-thresh-mm", type=float, default=20.0)
    ap.add_argument("--min-size-mm", type=float, default=10.0)

    # Device
    ap.add_argument("--device", type=str, default=None)

    return ap.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset config (should match training) ---
    sample_cfg = SampleConfig(
        target_spacing_xyz=(args.target_spacing[0], args.target_spacing[1], args.target_spacing[2]),
        heat_sigma_vox=float(args.heat_sigma),
        ct_clip=(float(args.ct_clip[0]), float(args.ct_clip[1])),
        pad_multiple=int(args.pad_multiple),
        heatmap_method=str(args.heatmap_method),
    )

    ds = LocalizerDataset(args.index_csv, split=args.split, cfg=sample_cfg)

    loader_cfg = LoaderConfig(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )

    # Build DataLoader (shuffle False for evaluation)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=loader_cfg.drop_last,
        persistent_workers=loader_cfg.persistent_workers if loader_cfg.num_workers > 0 else False,
        prefetch_factor=loader_cfg.prefetch_factor if loader_cfg.num_workers > 0 and loader_cfg.prefetch_factor is not None else 2,
    )

    print(f"Evaluating split='{args.split}' with {len(ds)} samples on device='{device}'")

    # --- Model ---
    net = build_model(
        name=args.model,
        base=int(args.base),
        dropout=float(args.dropout),
        positive_size=bool(args.positive_size),
    )

    # --- Metrics config ---
    val_cfg = ValConfig(
        clamp_min_size_mm=float(args.min_size_mm),
        success_thresh_mm=float(args.p_thresh_mm),
    )

    metrics = validate_epoch(net, dl, device=device, cfg=val_cfg)

    # Print nicely
    med = metrics["median_center_error_mm"]
    mean = metrics["mean_center_error_mm"]
    pT = metrics["p_at_thresh"] * 100.0
    miou = metrics["mean_iou"]
    n = int(metrics["n"])

    print("\n✅ Evaluation results")
    print(f"n: {n}")
    print(f"median center error (mm): {med:.2f}")
    print(f"mean   center error (mm): {mean:.2f}")
    print(f"P@{val_cfg.success_thresh_mm:.0f}mm (%): {pT:.1f}")
    print(f"mean IoU: {miou:.3f}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": args.split,
            "ckpt": str(args.ckpt),
            "device": device,
            "metrics": metrics,
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved metrics JSON -> {args.out_json}")


if __name__ == "__main__":
    raise SystemExit(main())