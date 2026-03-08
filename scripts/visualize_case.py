#!/usr/bin/env python3
"""
scripts/visualize_case.py

Visualize a scan with predicted and GT bounding boxes (3 views).

Examples:
    python scripts/visualize_case.py \
        --index-csv data/processed/localizer_index.csv \
        --ckpt outputs/run01/best.pt \
        --split val --idx 0 \
        --margin-mm 25

Notes:
- The model runs on the RESAMPLED image grid (target spacing).
- Boxes are drawn on the ORIGINAL image grid (full FOV).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk

from localization.data.dataset import SampleConfig
from localization.data.preprocess import normalize_ct
from localization.transforms.resample import sitk_resample_iso
from localization.geometry.coords import world_to_vox
from localization.inference.decode import DecodeConfig, decode_prediction, corners_from_bbox_mm
from localization.models.unet3d import LocalizerNet
from localization.viz.viewer import minmax_xyz_from_corners, plot_three_views, clamp_minmax_to_volume


def parse_args():
    ap = argparse.ArgumentParser(description="Visualize predicted vs GT bbox on a 3D scan (3 views).")

    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/localizer_index.csv"))
    ap.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path (.pt state_dict).")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--idx", type=int, default=0)

    # Model
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--positive-size", action="store_true")

    # Preprocess / resample (must match training)
    ap.add_argument("--target-spacing", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    ap.add_argument("--ct-clip", type=float, nargs=2, default=[-150.0, 350.0])

    # Decode options
    ap.add_argument("--min-size-mm", type=float, default=10.0, help="Clamp predicted size for visualization.")
    ap.add_argument("--margin-mm", type=float, default=25.0, help="Extra margin added around predicted bbox.")

    # Device
    ap.add_argument("--device", type=str, default=None)

    return ap.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load the selected row from index CSV ----
    df = pd.read_csv(args.index_csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(f"No rows for split='{args.split}' in {args.index_csv}")
    if args.idx < 0 or args.idx >= len(df):
        raise IndexError(f"idx out of range: {args.idx} (split '{args.split}' has {len(df)} rows)")

    row = df.iloc[args.idx]
    img_path = Path(row["image"])
    meta_path = Path(row["meta"])
    case_id = str(row["case_id"]) if "case_id" in df.columns else f"{args.split}:{args.idx}"

    # ---- Read ORIGINAL image (for drawing on full FOV) ----
    img0 = sitk.ReadImage(str(img_path))
    vol0 = sitk.GetArrayFromImage(img0).astype(np.float32)  # (Z,Y,X)

    # ---- Resample to target spacing (for model inference) ----
    target_spacing = (args.target_spacing[0], args.target_spacing[1], args.target_spacing[2])
    imgR = sitk_resample_iso(img0, out_spacing=target_spacing, interp=sitk.sitkLinear)
    volR = sitk.GetArrayFromImage(imgR).astype(np.float32)  # (Z,Y,X)

    # ---- Normalize like training ----
    volR = normalize_ct(volR, clip=(float(args.ct_clip[0]), float(args.ct_clip[1])))

    x = torch.from_numpy(volR[None, None]).to(device)  # (1,1,Z,Y,X)

    # ---- Load model + checkpoint ----
    net = LocalizerNet(base=int(args.base), dropout=float(args.dropout), positive_size=bool(args.positive_size)).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(sd)
    net.eval()

    with torch.no_grad():
        heat_p, size_p = net(x)

    heat_np = heat_p[0, 0].detach().cpu().numpy()  # (Z,Y,X)
    size_np = size_p[0].detach().cpu().numpy()     # (3,)

    # ---- Decode prediction on RESAMPLED grid (imgR) -> bbox in mm ----
    dec_cfg = DecodeConfig(clamp_min_size_mm=float(args.min_size_mm), margin_mm=float(args.margin_mm))
    pred_center_mm, pred_bbox_mm = decode_prediction(heat_np, size_np, imgR, cfg=dec_cfg)

    # ---- Load GT bbox in mm ----
    with open(meta_path, "r") as f:
        gt_bbox_mm = np.array(json.load(f)["bbox_mm"], dtype=np.float32)

    # ---- Convert both boxes to ORIGINAL voxel space for drawing ----
    pred_corners_mm = corners_from_bbox_mm(pred_bbox_mm)
    gt_corners_mm = corners_from_bbox_mm(gt_bbox_mm)

    pred_corners_vox0 = world_to_vox(pred_corners_mm, img0)  # (8,3) x,y,z
    gt_corners_vox0 = world_to_vox(gt_corners_mm, img0)

    pred_min, pred_max = minmax_xyz_from_corners(pred_corners_vox0)
    gt_min, gt_max = minmax_xyz_from_corners(gt_corners_vox0)

    pred_min, pred_max = clamp_minmax_to_volume(pred_min, pred_max, vol0.shape)
    gt_min, gt_max = clamp_minmax_to_volume(gt_min, gt_max, vol0.shape)

    # Choose slice center using predicted center converted to ORIGINAL voxel coords
    pred_center_vox0 = world_to_vox(pred_center_mm[None, :], img0)[0]  # (x,y,z)

    print(f"Case: {case_id}")
    print("Pred center mm:", pred_center_mm.tolist())
    print("Pred size mm:", np.maximum(size_np, args.min_size_mm).tolist())
    print("Pred bbox mm:", pred_bbox_mm.tolist())
    print("GT bbox mm:", gt_bbox_mm.tolist())

    # ---- Plot three views ----
    plot_three_views(
        vol0_zyx=vol0,
        pred_minmax_xyz=(pred_min, pred_max),
        gt_minmax_xyz=(gt_min, gt_max),
        center_xyz=pred_center_vox0,
        title_prefix=case_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())