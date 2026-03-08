#!/usr/bin/env python3
"""
scripts/debug_dataset.py

Quick dataset debugging script:
- Loads LocalizerDataset
- Fetches one or multiple samples
- Prints shapes and basic target info

Examples:
    python scripts/debug_dataset.py --index-csv data/processed/localizer_index.csv --split train --idx 0
    python scripts/debug_dataset.py --index-csv data/processed/localizer_index.csv --split val --num 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from localization.data.dataset import LocalizerDataset, SampleConfig


def parse_args():
    ap = argparse.ArgumentParser(description="Debug LocalizerDataset by loading a few samples.")
    ap.add_argument(
        "--index-csv",
        type=Path,
        default=Path("data/processed/localizer_index.csv"),
        help="Path to index CSV.",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to debug.",
    )
    ap.add_argument("--idx", type=int, default=0, help="Single sample index to load.")
    ap.add_argument("--num", type=int, default=1, help="How many consecutive samples to load starting at idx.")
    ap.add_argument("--target-spacing", type=float, nargs=3, default=[2.0, 2.0, 2.0], help="Target spacing (x y z).")
    ap.add_argument("--heat-sigma", type=float, default=3.0, help="Heatmap sigma in voxels.")
    ap.add_argument("--pad-multiple", type=int, default=8, help="Pad (Z,Y,X) to a multiple of this value.")
    ap.add_argument("--ct-clip", type=float, nargs=2, default=[-150.0, 350.0], help="CT clip window (min max).")
    return ap.parse_args()


def describe_sample(i: int, x: torch.Tensor, y: dict):
    case_id = y.get("case_id", None)
    print(f"\n--- Sample {i} ---")
    if case_id is not None:
        print("case_id:", case_id)

    print("x:", tuple(x.shape), x.dtype, "min/max:", float(x.min()), float(x.max()))
    print("heat:", tuple(y["heat"].shape), y["heat"].dtype, "min/max:", float(y["heat"].min()), float(y["heat"].max()))
    print("size:", y["size"].tolist())

    if "center_mm" in y:
        print("center_mm:", y["center_mm"].tolist())

    if "spacing" in y:
        print("spacing(xyz):", y["spacing"].tolist())

    if "pad_spec" in y:
        print("pad_spec:", y["pad_spec"])


def main():
    args = parse_args()

    cfg = SampleConfig(
        target_spacing_xyz=(args.target_spacing[0], args.target_spacing[1], args.target_spacing[2]),
        heat_sigma_vox=float(args.heat_sigma),
        ct_clip=(float(args.ct_clip[0]), float(args.ct_clip[1])),
        pad_multiple=int(args.pad_multiple),
    )

    ds = LocalizerDataset(args.index_csv, split=args.split, cfg=cfg)
    print(f"Loaded dataset split='{args.split}' with {len(ds)} samples from {args.index_csv}")

    start = args.idx
    end = min(len(ds), start + max(1, args.num))

    for i in range(start, end):
        x, y = ds[i]  # <-- if something is broken, error is raised here
        describe_sample(i, x, y)

    print("\n✅ Dataset debug finished successfully.")


if __name__ == "__main__":
    raise SystemExit(main())