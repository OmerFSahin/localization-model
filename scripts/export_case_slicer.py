#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk

from localization.data.preprocess import normalize_ct, pad_spec_for_shape, apply_pad
from localization.transforms.resample import sitk_resample_iso
from localization.geometry.coords import world_to_vox
from localization.inference.decode import DecodeConfig, decode_prediction
from localization.models.factory import build_model


def unpad_zyx(arr: np.ndarray, pad_spec) -> np.ndarray:
    (z0, z1), (y0, y1), (x0, x1) = pad_spec
    z_slice = slice(z0, arr.shape[0] - z1 if z1 > 0 else None)
    y_slice = slice(y0, arr.shape[1] - y1 if y1 > 0 else None)
    x_slice = slice(x0, arr.shape[2] - x1 if x1 > 0 else None)
    return arr[z_slice, y_slice, x_slice]


def save_array_as_nrrd(arr_zyx: np.ndarray, ref_img: sitk.Image, out_path: Path, dtype=sitk.sitkFloat32):
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetSpacing(ref_img.GetSpacing())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    if dtype is not None:
        img = sitk.Cast(img, dtype)
    sitk.WriteImage(img, str(out_path))


def bbox_mm_to_mask_zyx(bbox_mm: np.ndarray, ref_img: sitk.Image) -> np.ndarray:
    shape_xyz = ref_img.GetSize()
    shape_zyx = (shape_xyz[2], shape_xyz[1], shape_xyz[0])
    mask = np.zeros(shape_zyx, dtype=np.uint8)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox_mm.astype(np.float32)

    corners_mm = np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
    ], dtype=np.float32)

    corners_vox_xyz = world_to_vox(corners_mm, ref_img)

    vmin = np.floor(corners_vox_xyz.min(axis=0)).astype(int)
    vmax = np.ceil(corners_vox_xyz.max(axis=0)).astype(int)

    x0, y0, z0 = vmin
    x1, y1, z1 = vmax

    x0 = np.clip(x0, 0, shape_xyz[0] - 1)
    y0 = np.clip(y0, 0, shape_xyz[1] - 1)
    z0 = np.clip(z0, 0, shape_xyz[2] - 1)

    x1 = np.clip(x1, 0, shape_xyz[0] - 1)
    y1 = np.clip(y1, 0, shape_xyz[1] - 1)
    z1 = np.clip(z1, 0, shape_xyz[2] - 1)

    mask[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = 1
    return mask

def union_bbox_mm(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    b1 = np.asarray(b1, dtype=np.float32)
    b2 = np.asarray(b2, dtype=np.float32)
    mins = np.minimum(b1[:3], b2[:3])
    maxs = np.maximum(b1[3:], b2[3:])
    return np.concatenate([mins, maxs]).astype(np.float32)


def expand_bbox_mm(bbox_mm: np.ndarray, margin_mm: float) -> np.ndarray:
    bbox_mm = np.asarray(bbox_mm, dtype=np.float32)
    mins = bbox_mm[:3] - float(margin_mm)
    maxs = bbox_mm[3:] + float(margin_mm)
    return np.concatenate([mins, maxs]).astype(np.float32)


def bbox_mm_to_crop_slices(bbox_mm: np.ndarray, ref_img: sitk.Image):
    corners_mm = np.array([
        [bbox_mm[0], bbox_mm[1], bbox_mm[2]],
        [bbox_mm[0], bbox_mm[1], bbox_mm[5]],
        [bbox_mm[0], bbox_mm[4], bbox_mm[2]],
        [bbox_mm[0], bbox_mm[4], bbox_mm[5]],
        [bbox_mm[3], bbox_mm[1], bbox_mm[2]],
        [bbox_mm[3], bbox_mm[1], bbox_mm[5]],
        [bbox_mm[3], bbox_mm[4], bbox_mm[2]],
        [bbox_mm[3], bbox_mm[4], bbox_mm[5]],
    ], dtype=np.float32)

    vox_xyz = world_to_vox(corners_mm, ref_img)
    vmin = np.floor(vox_xyz.min(axis=0)).astype(int)
    vmax = np.ceil(vox_xyz.max(axis=0)).astype(int)

    sx, sy, sz = ref_img.GetSize()

    x0 = int(np.clip(vmin[0], 0, sx - 1))
    y0 = int(np.clip(vmin[1], 0, sy - 1))
    z0 = int(np.clip(vmin[2], 0, sz - 1))
    x1 = int(np.clip(vmax[0], 0, sx - 1))
    y1 = int(np.clip(vmax[1], 0, sy - 1))
    z1 = int(np.clip(vmax[2], 0, sz - 1))

    return slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1), (x0, y0, z0)

def save_crop_as_nrrd(arr_crop_zyx: np.ndarray, ref_img: sitk.Image, xyz_start, out_path: Path, dtype=sitk.sitkFloat32):
    x0, y0, z0 = xyz_start

    spacing = np.array(ref_img.GetSpacing(), dtype=np.float32)
    origin = np.array(ref_img.GetOrigin(), dtype=np.float32)
    direction = np.array(ref_img.GetDirection(), dtype=np.float32).reshape(3, 3)

    A = direction @ np.diag(spacing)
    new_origin = origin + A @ np.array([x0, y0, z0], dtype=np.float32)

    img = sitk.GetImageFromArray(arr_crop_zyx)
    img.SetSpacing(tuple(spacing.tolist()))
    img.SetOrigin(tuple(new_origin.tolist()))
    img.SetDirection(tuple(direction.reshape(-1).tolist()))
    if dtype is not None:
        img = sitk.Cast(img, dtype)
    sitk.WriteImage(img, str(out_path))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--idx", type=int, default=0)

    ap.add_argument("--model", type=str, default="unet3d", choices=["unet3d", "cnn3d_regressor", "resnet3d_regressor"])
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--positive-size", action="store_true")

    ap.add_argument("--target-spacing", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    ap.add_argument("--ct-clip", type=float, nargs=2, default=[-150.0, 350.0])
    ap.add_argument("--pad-multiple", type=int, default=8)

    ap.add_argument("--min-size-mm", type=float, default=10.0)
    ap.add_argument("--margin-mm", type=float, default=25.0)
    ap.add_argument("--crop-margin-mm", type=float, default=20.0)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--outdir", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index_csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    row = df.iloc[args.idx]

    img_path = Path(row["image"])
    meta_path = Path(row["meta"])
    case_id = str(row["case_id"]) if "case_id" in df.columns else f"{args.split}_{args.idx}"

    img0 = sitk.ReadImage(str(img_path))
    vol0 = sitk.GetArrayFromImage(img0).astype(np.float32)
    imgR = sitk_resample_iso(
        img0,
        out_spacing=(float(args.target_spacing[0]), float(args.target_spacing[1]), float(args.target_spacing[2])),
        interp=sitk.sitkLinear,
    )

    volR = sitk.GetArrayFromImage(imgR).astype(np.float32)
    volR_norm = normalize_ct(volR, clip=(float(args.ct_clip[0]), float(args.ct_clip[1])))

    pad_spec = pad_spec_for_shape(volR_norm.shape, k=int(args.pad_multiple))
    volR_pad = apply_pad(volR_norm, pad_spec, mode="constant", value=0.0)

    x = torch.from_numpy(volR_pad[None, None]).to(args.device)

    net = build_model(
        name=args.model,
        base=int(args.base),
        dropout=float(args.dropout),
        positive_size=bool(args.positive_size),
    )
    sd = torch.load(args.ckpt, map_location=args.device)
    net.load_state_dict(sd)
    net.to(args.device)
    net.eval()

    with torch.no_grad():
        heat_p, size_p = net(x)

    heat_np_pad = heat_p[0, 0].detach().cpu().numpy()
    heat_np = unpad_zyx(heat_np_pad, pad_spec)
    size_raw = size_p[0].detach().cpu().numpy()

    if args.size_target == "mm":
        size_np = size_raw
    elif args.size_target == "log_mm":
        size_np = np.exp(size_raw).astype(np.float32)
    else:
        raise ValueError(f"Unknown size_target: {args.size_target}")

    dec_cfg = DecodeConfig(
        clamp_min_size_mm=float(args.min_size_mm),
        margin_mm=float(args.margin_mm),
    )
    pred_center_mm, pred_bbox_mm = decode_prediction(heat_np, size_np, imgR, cfg=dec_cfg)

    with open(meta_path, "r") as f:
        gt_bbox_mm = np.array(json.load(f)["bbox_mm"], dtype=np.float32)

    pred_mask = bbox_mm_to_mask_zyx(pred_bbox_mm, imgR)
    gt_mask = bbox_mm_to_mask_zyx(gt_bbox_mm, imgR)
    
    crop_bbox_mm = union_bbox_mm(pred_bbox_mm, gt_bbox_mm)
    crop_bbox_mm = expand_bbox_mm(crop_bbox_mm, margin_mm=float(args.crop_margin_mm))

    zsl, ysl, xsl, xyz_start = bbox_mm_to_crop_slices(crop_bbox_mm, imgR)

    scan_crop = volR[zsl, ysl, xsl]
    heat_crop = heat_np[zsl, ysl, xsl]
    pred_mask_crop = pred_mask[zsl, ysl, xsl]
    gt_mask_crop = gt_mask[zsl, ysl, xsl]

    zsl0, ysl0, xsl0, xyz_start0 = bbox_mm_to_crop_slices(crop_bbox_mm, img0)
    scan_crop_original = vol0[zsl0, ysl0, xsl0]

    save_array_as_nrrd(
        vol0,
        img0,
        args.outdir / f"{case_id}_scan_original.nrrd",
        dtype=sitk.sitkFloat32,
)

    save_array_as_nrrd(volR, imgR, args.outdir / f"{case_id}_scan_resampled.nrrd", dtype=sitk.sitkFloat32)
    save_array_as_nrrd(heat_np.astype(np.float32), imgR, args.outdir / f"{case_id}_pred_heatmap.nrrd", dtype=sitk.sitkFloat32)
    save_array_as_nrrd(pred_mask, imgR, args.outdir / f"{case_id}_pred_bbox_mask.nrrd", dtype=sitk.sitkUInt8)
    save_array_as_nrrd(gt_mask, imgR, args.outdir / f"{case_id}_gt_bbox_mask.nrrd", dtype=sitk.sitkUInt8)
    
    save_crop_as_nrrd(
        scan_crop.astype(np.float32),
        imgR,
        xyz_start,
        args.outdir / f"{case_id}_scan_crop.nrrd",
        dtype=sitk.sitkFloat32,
    )
    save_crop_as_nrrd(
        heat_crop.astype(np.float32),
        imgR,
        xyz_start,
        args.outdir / f"{case_id}_pred_heatmap_crop.nrrd",
        dtype=sitk.sitkFloat32,
    )
    save_crop_as_nrrd(
        pred_mask_crop.astype(np.uint8),
        imgR,
        xyz_start,
        args.outdir / f"{case_id}_pred_bbox_mask_crop.nrrd",
        dtype=sitk.sitkUInt8,
    )
    save_crop_as_nrrd(
        gt_mask_crop.astype(np.uint8),
        imgR,
        xyz_start,
        args.outdir / f"{case_id}_gt_bbox_mask_crop.nrrd",
        dtype=sitk.sitkUInt8,
    )

    save_crop_as_nrrd(
        scan_crop_original.astype(np.float32),
        img0,
        xyz_start0,
        args.outdir / f"{case_id}_scan_original_crop.nrrd",
        dtype=sitk.sitkFloat32,
    )

    info = {
        "case_id": case_id,
        "pred_center_mm": pred_center_mm.tolist(),
        "pred_bbox_mm": pred_bbox_mm.tolist(),
        "pred_size_mm_raw": size_np.tolist(),
        "gt_bbox_mm": gt_bbox_mm.tolist(),
        "crop_bbox_mm": crop_bbox_mm.tolist(),
    }
    with open(args.outdir / f"{case_id}_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved Slicer files to: {args.outdir}")


if __name__ == "__main__":
    raise SystemExit(main())