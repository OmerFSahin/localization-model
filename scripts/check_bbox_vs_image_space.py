#!/usr/bin/env python3
"""
scripts/check_bbox_vs_image_space.py

Check whether bbox_mm annotations fall inside the physical extent of the image.

Usage:
    python scripts/check_bbox_vs_image_space.py --index-csv data/processed/localizer_index.csv
    python scripts/check_bbox_vs_image_space.py --index-csv data/processed/localizer_index.csv --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


def parse_args():
    ap = argparse.ArgumentParser(description="Check whether bbox_mm lies inside image physical extent.")
    ap.add_argument("--index-csv", type=Path, required=True, help="Path to localizer_index.csv")
    ap.add_argument("--split", type=str, default=None, choices=["train", "val", "test"], help="Optional split filter")
    return ap.parse_args()


def image_world_bounds(img: sitk.Image):
    """
    Compute world-coordinate bounds of the image from its 8 corners.
    Returns:
        mins_xyz, maxs_xyz
    """
    size_xyz = np.array(img.GetSize(), dtype=np.int64)  # (x,y,z)

    # valid corner indices go from 0 to size-1
    xs = [0, size_xyz[0] - 1]
    ys = [0, size_xyz[1] - 1]
    zs = [0, size_xyz[2] - 1]

    corners_mm = []
    for x in xs:
        for y in ys:
            for z in zs:
                p = img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
                corners_mm.append(p)

    corners_mm = np.asarray(corners_mm, dtype=np.float64)
    mins = corners_mm.min(axis=0)
    maxs = corners_mm.max(axis=0)
    return mins, maxs


def main():
    args = parse_args()

    df = pd.read_csv(args.index_csv)
    if args.split is not None:
        df = df[df["split"] == args.split].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No rows found for the given index/split.")

    bad = 0

    for i, row in df.iterrows():
        case_id = row["case_id"] if "case_id" in df.columns else f"row_{i}"
        img_path = Path(row["image"])
        meta_path = Path(row["meta"])

        img = sitk.ReadImage(str(img_path))

        with open(meta_path, "r") as f:
            meta = json.load(f)

        bbox = np.array(meta["bbox_mm"], dtype=np.float64)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox

        # make sure ordered
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        zmin, zmax = min(zmin, zmax), max(zmin, zmax)

        bbox_min = np.array([xmin, ymin, zmin], dtype=np.float64)
        bbox_max = np.array([xmax, ymax, zmax], dtype=np.float64)

        img_min, img_max = image_world_bounds(img)

        inside = np.all(bbox_min >= img_min) and np.all(bbox_max <= img_max)

        print(f"\nCase: {case_id}")
        print("Image path:", img_path)
        print("Image size xyz:", img.GetSize())
        print("Image spacing xyz:", img.GetSpacing())
        print("Image origin xyz:", img.GetOrigin())
        print("Image direction:", img.GetDirection())
        print("Image world min xyz:", img_min.tolist())
        print("Image world max xyz:", img_max.tolist())
        print("BBox min xyz:", bbox_min.tolist())
        print("BBox max xyz:", bbox_max.tolist())
        print("BBox inside image world extent:", inside)

        if not inside:
            bad += 1
            print(">>> WARNING: bbox_mm is outside image physical extent")

    print("\n===================================")
    print(f"Checked {len(df)} cases")
    print(f"Cases with bbox outside image extent: {bad}")
    print("===================================")


if __name__ == "__main__":
    raise SystemExit(main())