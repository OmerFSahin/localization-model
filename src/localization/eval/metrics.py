"""
localization.eval.metrics

Evaluation utilities for the localization model.

We compute:
- predicted center in mm (from heatmap argmax)
- center error in mm (L2 distance between predicted and GT center)
- bbox IoU (3D) using predicted center + predicted size

Conventions:
- heatmap arrays are (Z, Y, X) when converted to NumPy
- world coordinates are (x, y, z) in mm
- voxel coordinates are (x, y, z)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch


# -------------------------------------------------------
# Core geometry helpers
# -------------------------------------------------------
def center_mm_from_heatmap(
    hz_zyx: np.ndarray,
    spacing_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
) -> np.ndarray:
    """
    Decode center position in world mm from a heatmap by argmax.

    Args:
        hz_zyx: (Z,Y,X) heatmap (numpy)
        spacing_xyz: (3,) spacing (x,y,z) in mm
        origin_xyz: (3,) origin (x,y,z) in mm
        direction_xyz: (3,3) direction cosines matrix

    Returns:
        center_mm: (3,) in (x,y,z)
    """
    # Argmax in Z,Y,X array
    z, y, x = np.unravel_index(int(np.argmax(hz_zyx)), hz_zyx.shape)

    # Convert voxel (x,y,z) -> world (x,y,z)
    v_xyz = np.array([[x, y, z]], dtype=np.float32)  # (1,3)
    A = direction_xyz.reshape(3, 3) @ np.diag(spacing_xyz.astype(np.float32))
    center_mm = origin_xyz.astype(np.float32) + (A @ v_xyz.T).T[0]
    return center_mm.astype(np.float32)


def bbox_from_center_size(center_mm: np.ndarray, size_mm: np.ndarray) -> np.ndarray:
    """
    Build bbox [xmin,ymin,zmin,xmax,ymax,zmax] from center and size (both in mm).
    """
    center_mm = np.asarray(center_mm, dtype=np.float32)
    size_mm = np.asarray(size_mm, dtype=np.float32)
    half = size_mm / 2.0
    return np.concatenate([center_mm - half, center_mm + half]).astype(np.float32)


def iou3d(b1: np.ndarray, b2: np.ndarray, eps: float = 1e-8) -> float:
    """
    3D IoU of two axis-aligned boxes in mm coordinates.

    b format: [xmin,ymin,zmin,xmax,ymax,zmax]
    """
    b1 = np.asarray(b1, dtype=np.float32)
    b2 = np.asarray(b2, dtype=np.float32)

    a1, a2 = b1[:3], b1[3:]
    c1, c2 = b2[:3], b2[3:]

    inter_min = np.maximum(a1, c1)
    inter_max = np.minimum(a2, c2)
    inter_sz = np.maximum(0.0, inter_max - inter_min)
    inter = float(np.prod(inter_sz))

    vol_a = float(np.prod(np.maximum(0.0, a2 - a1)))
    vol_b = float(np.prod(np.maximum(0.0, c2 - c1)))

    return inter / (vol_a + vol_b - inter + eps)


# -------------------------------------------------------
# Aggregated validation
# -------------------------------------------------------
@dataclass(frozen=True)
class ValConfig:
    """
    Validation settings.
    """
    clamp_min_size_mm: float = 10.0
    success_thresh_mm: float = 20.0  # P@20mm


def validate_epoch(
    net: torch.nn.Module,
    val_dl,
    device: str = "cpu",
    cfg: ValConfig = ValConfig(),
) -> Dict[str, float]:
    """
    Run validation over a DataLoader.

    Expects the dataset to return:
        y["spacing"], y["origin"], y["direction"], y["center_mm"], y["size"]

    Returns dict:
        - median_center_error_mm
        - mean_center_error_mm
        - p_at_thresh
        - mean_iou
        - n
    """
    net.eval()

    center_errs = []
    ious = []

    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)

            heat_p, size_p = net(x)

            # heat_p: (B,1,Z,Y,X) -> take first in batch
            hz = heat_p[0, 0].detach().cpu().numpy()

            spacing = y["spacing"][0].numpy()
            origin = y["origin"][0].numpy()
            direction = y["direction"][0].numpy()

            gt_center = y["center_mm"][0].numpy()
            gt_size = y["size"][0].numpy()

            pred_center = center_mm_from_heatmap(hz, spacing, origin, direction)

            # Center error in mm
            center_err = float(np.linalg.norm(pred_center - gt_center))
            center_errs.append(center_err)

            # IoU using predicted size (clamped for stability early in training)
            pred_size = size_p[0].detach().cpu().numpy()
            pred_size = np.maximum(pred_size, float(cfg.clamp_min_size_mm))

            pred_box = bbox_from_center_size(pred_center, pred_size)
            gt_box = bbox_from_center_size(gt_center, gt_size)

            ious.append(iou3d(pred_box, gt_box))

    if len(center_errs) == 0:
        return {
            "median_center_error_mm": float("inf"),
            "mean_center_error_mm": float("inf"),
            "p_at_thresh": 0.0,
            "mean_iou": 0.0,
            "n": 0.0,
        }

    errs = np.asarray(center_errs, dtype=np.float32)
    ious = np.asarray(ious, dtype=np.float32)

    return {
        "median_center_error_mm": float(np.median(errs)),
        "mean_center_error_mm": float(np.mean(errs)),
        "p_at_thresh": float(np.mean(errs <= float(cfg.success_thresh_mm))),
        "mean_iou": float(np.mean(ious)),
        "n": float(len(errs)),
    }