#!/usr/bin/env python3
"""
scripts/visualize_case.py

Visualize a scan with predicted and GT bounding boxes (3 views).

Examples:
    python scripts/visualize_case.py \
        --index-csv data/processed/localizer_index.csv \
        --ckpt outputs/run01/best.pt \
        --model unet3d \
        --split val --idx 0 \
        --margin-mm 25 \
        --device cuda

Supported models:
- unet3d
- cnn3d_regressor
- resnet3d_regressor

Notes:
- The model runs on the RESAMPLED image grid (target spacing).
- Training uses padded inputs, so this script pads the resampled volume
  before inference and removes padding before geometric decode.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from localization.data.preprocess import (
    normalize_ct,
    pad_spec_for_shape,
    apply_pad,
)
from localization.transforms.resample import sitk_resample_iso
from localization.geometry.coords import world_to_vox
from localization.inference.decode import DecodeConfig, decode_prediction, corners_from_bbox_mm
from localization.models.factory import build_model
from localization.viz.viewer import minmax_xyz_from_corners, plot_three_views, clamp_minmax_to_volume

def unpad_zyx(arr: np.ndarray, pad_spec) -> np.ndarray:
    """
    Remove padding from a (Z,Y,X) array using pad_spec from pad_spec_for_shape.

    Expected pad_spec format:
        ((z0, z1), (y0, y1), (x0, x1))
    """
    (z0, z1), (y0, y1), (x0, x1) = pad_spec

    z_slice = slice(z0, arr.shape[0] - z1 if z1 > 0 else None)
    y_slice = slice(y0, arr.shape[1] - y1 if y1 > 0 else None)
    x_slice = slice(x0, arr.shape[2] - x1 if x1 > 0 else None)

    return arr[z_slice, y_slice, x_slice]




def parse_args():
    ap = argparse.ArgumentParser(description="Visualize predicted vs GT bbox on a 3D scan (3 views).")

    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/localizer_index.csv"))
    ap.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path (.pt state_dict).")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--idx", type=int, default=0)

    # Model
    ap.add_argument("--model", type=str, default="unet3d", choices=["unet3d", "cnn3d_regressor", "resnet3d_regressor"], help="Model architecture used by the checkpoint.",)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--positive-size", action="store_true")
    ap.add_argument("--size-target", type=str, default="mm", choices=["mm", "log_mm"], help="Target representation used during training.",)


    # Preprocess / resample (must match training)
    ap.add_argument("--target-spacing", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    ap.add_argument("--ct-clip", type=float, nargs=2, default=[-150.0, 350.0])
    ap.add_argument("--pad-multiple", type=int, default=8, help="Must match training pad_multiple.")

    # Decode options
    ap.add_argument("--min-size-mm", type=float, default=10.0, help="Clamp predicted size for visualization.")
    ap.add_argument("--margin-mm", type=float, default=25.0, help="Extra margin added around predicted bbox.")

    # Device
    ap.add_argument("--device", type=str, default=None)

    # Optional output
    ap.add_argument("--save-path", type=Path, default=None, help="Optional path to save the visualization as PNG.",)

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


    # ---- Pad like training ----
    pad_spec = pad_spec_for_shape(volR.shape, k=int(args.pad_multiple))
    volR_pad = apply_pad(volR, pad_spec, mode="constant", value=0.0)

    x = torch.from_numpy(volR[None, None]).to(device)  # (1,1,Z,Y,X)

    # ---- Load model + checkpoint ----
    net = build_model(name=args.model, base=int(args.base), dropout=float(args.dropout), positive_size=bool(args.positive_size),)
    sd = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(sd)
    net.to(device)
    net.eval()

    with torch.no_grad():
        heat_p, size_p = net(x)

    heat_np_pad = heat_p[0, 0].detach().cpu().numpy()  # padded (Z,Y,X)
    heat_np = unpad_zyx(heat_np_pad, pad_spec)  # (Z,Y,X)
    size_raw = size_p[0].detach().cpu().numpy()

    if args.size_target == "mm":
        size_np = size_raw
    elif args.size_target == "log_mm":
        size_np = np.exp(size_raw).astype(np.float32)
    else:
        raise ValueError(f"Unknown size_target: {args.size_target}")

    # ---- Decode prediction on RESAMPLED grid (imgR) -> bbox in mm ----
    dec_cfg = DecodeConfig(clamp_min_size_mm=float(args.min_size_mm), margin_mm=float(args.margin_mm))
    pred_center_mm, pred_bbox_mm = decode_prediction(heat_np, size_np, imgR, cfg=dec_cfg)

    # ---- Load GT bbox in mm ----
    with open(meta_path, "r") as f:
        gt_bbox_mm = np.array(json.load(f)["bbox_mm"], dtype=np.float32)

    # --- Debug: compare pred/gt boxes in both mm and voxel space ---
    pred_corners_mm = np.array([
        [pred_bbox_mm[0], pred_bbox_mm[1], pred_bbox_mm[2]],
        [pred_bbox_mm[0], pred_bbox_mm[1], pred_bbox_mm[5]],
        [pred_bbox_mm[0], pred_bbox_mm[4], pred_bbox_mm[2]],
        [pred_bbox_mm[0], pred_bbox_mm[4], pred_bbox_mm[5]],
        [pred_bbox_mm[3], pred_bbox_mm[1], pred_bbox_mm[2]],
        [pred_bbox_mm[3], pred_bbox_mm[1], pred_bbox_mm[5]],
        [pred_bbox_mm[3], pred_bbox_mm[4], pred_bbox_mm[2]],
        [pred_bbox_mm[3], pred_bbox_mm[4], pred_bbox_mm[5]],
    ], dtype=np.float32)

    gt_corners_mm = np.array([
        [gt_bbox_mm[0], gt_bbox_mm[1], gt_bbox_mm[2]],
        [gt_bbox_mm[0], gt_bbox_mm[1], gt_bbox_mm[5]],
        [gt_bbox_mm[0], gt_bbox_mm[4], gt_bbox_mm[2]],
        [gt_bbox_mm[0], gt_bbox_mm[4], gt_bbox_mm[5]],
        [gt_bbox_mm[3], gt_bbox_mm[1], gt_bbox_mm[2]],
        [gt_bbox_mm[3], gt_bbox_mm[1], gt_bbox_mm[5]],
        [gt_bbox_mm[3], gt_bbox_mm[4], gt_bbox_mm[2]],
        [gt_bbox_mm[3], gt_bbox_mm[4], gt_bbox_mm[5]],
    ], dtype=np.float32)

    pred_voxR = world_to_vox(pred_corners_mm, imgR)
    gt_voxR = world_to_vox(gt_corners_mm, imgR)

    print("pred_bbox_mm:", pred_bbox_mm.tolist())
    print("gt_bbox_mm:", gt_bbox_mm.tolist())

    print("pred_bbox_voxR min:", pred_voxR.min(axis=0).tolist(), "max:", pred_voxR.max(axis=0).tolist())
    print("gt_bbox_voxR   min:", gt_voxR.min(axis=0).tolist(), "max:", gt_voxR.max(axis=0).tolist())

    print("imgR size xyz:", imgR.GetSize())

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
    pred_center_vox0 = np.array(pred_center_vox0, dtype=np.float32)
    pred_center_vox0[0] = np.clip(pred_center_vox0[0], 0, vol0.shape[2] - 1)  # x
    pred_center_vox0[1] = np.clip(pred_center_vox0[1], 0, vol0.shape[1] - 1)  # y
    pred_center_vox0[2] = np.clip(pred_center_vox0[2], 0, vol0.shape[0] - 1)  # z
    raw_size_mm = size_np
    clamped_size_mm = np.maximum(size_np, args.min_size_mm)
    peak_zyx = np.unravel_index(np.argmax(heat_np), heat_np.shape)
    pred_peak_zyx = peak_zyx

    gt_center_mm = np.array([
        (gt_bbox_mm[0] + gt_bbox_mm[3]) / 2,
        (gt_bbox_mm[1] + gt_bbox_mm[4]) / 2,
        (gt_bbox_mm[2] + gt_bbox_mm[5]) / 2,
        ], dtype=np.float32)
    gt_center_voxR = world_to_vox(gt_center_mm[None, :], imgR)[0]
    pred_peak_xyz = np.array([pred_peak_zyx[2], pred_peak_zyx[1], pred_peak_zyx[0]], dtype=np.float32)

    print(f"Case: {case_id}")
    print("Raw pred size output:", size_raw.tolist())
    print("Decoded pred size mm:", size_np.tolist())
    print("Pred center mm:", pred_center_mm.tolist())
    print("Raw pred size mm:", raw_size_mm.tolist())
    print("Clamped pred size mm:", clamped_size_mm.tolist())
    print("Pred bbox mm:", pred_bbox_mm.tolist())
    print("GT bbox mm:", gt_bbox_mm.tolist())
    print("Pred heat peak zyx:", peak_zyx)
    print("Heat shape zyx:", heat_np.shape)
    print("Resampled image size xyz:", imgR.GetSize())
    print("GT center mm:", gt_center_mm.tolist())
    print("GT center vox xyz on resampled image:", gt_center_voxR.tolist())
    print("Pred peak zyx:", pred_peak_zyx)
    print("Pred peak xyz:", pred_peak_xyz.tolist())
    print("img0 size xyz:", img0.GetSize())
    print("img0 spacing xyz:", img0.GetSpacing())
    print("img0 origin xyz:", img0.GetOrigin())
    print("img0 direction:", img0.GetDirection()) 
    print("imgR size xyz:", imgR.GetSize())
    print("imgR spacing xyz:", imgR.GetSpacing())
    print("imgR origin xyz:", imgR.GetOrigin())
    print("imgR direction:", imgR.GetDirection())

    # ---- Plot three views ----
    plot_three_views(
        vol0_zyx=vol0,
        pred_minmax_xyz=(pred_min, pred_max),
        gt_minmax_xyz=(gt_min, gt_max),
        center_xyz=pred_center_vox0,
        title_prefix=case_id,
    )
    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save_path, dpi=600, bbox_inches="tight")
        print(f"Saved figure to: {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
