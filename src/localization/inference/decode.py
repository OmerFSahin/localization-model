"""
localization.inference.decode

Decode utilities for converting model outputs (heatmap + size) into
physical coordinates and bounding boxes.

Conventions:
- Heatmap NumPy array is (Z, Y, X)
- Voxel coordinates are (x, y, z)
- World coordinates are (x, y, z) in millimeters (mm)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import SimpleITK as sitk


# -------------------------------------------------------
# Core decode: heatmap -> center in mm
# -------------------------------------------------------
def center_mm_from_heatmap(hz_zyx: np.ndarray, img: sitk.Image) -> np.ndarray:
    """
    Decode the predicted center (x,y,z) in mm using argmax on the heatmap.

    Args:
        hz_zyx: heatmap array with shape (Z,Y,X)
        img: SimpleITK image representing the *same grid* as hz_zyx
             (i.e., resampled image used for inference)

    Returns:
        center_mm: np.float32 (3,) in (x,y,z)
    """
    # Argmax in (Z,Y,X)
    z, y, x = np.unravel_index(int(np.argmax(hz_zyx)), hz_zyx.shape)

    # Voxel coordinate in (x,y,z)
    v_xyz = np.array([[x, y, z]], dtype=np.float32)

    # voxel -> world: origin + (direction @ diag(spacing)) @ voxel
    D = np.array(img.GetDirection(), dtype=np.float32).reshape(3, 3)
    S = np.diag(np.array(img.GetSpacing(), dtype=np.float32))
    A = D @ S
    o = np.array(img.GetOrigin(), dtype=np.float32)

    center_mm = o + (A @ v_xyz.T).T[0]
    return center_mm.astype(np.float32)


# -------------------------------------------------------
# Box building helpers
# -------------------------------------------------------
def bbox_from_center_size_mm(
    center_mm: np.ndarray,
    size_mm: np.ndarray,
    margin_mm: float = 0.0,
) -> np.ndarray:
    """
    Convert center + size into an axis-aligned bbox in mm.

    Args:
        center_mm: (3,) (x,y,z)
        size_mm: (3,) (wx,wy,wz)
        margin_mm: extra margin added on each side

    Returns:
        bbox_mm: (6,) [xmin,ymin,zmin,xmax,ymax,zmax]
    """
    center_mm = np.asarray(center_mm, dtype=np.float32).reshape(3)
    size_mm = np.asarray(size_mm, dtype=np.float32).reshape(3)

    half = size_mm / 2.0 + float(margin_mm)
    return np.concatenate([center_mm - half, center_mm + half]).astype(np.float32)


def corners_from_bbox_mm(bbox_mm: np.ndarray) -> np.ndarray:
    """
    Create the 8 corners of an axis-aligned bbox.

    Args:
        bbox_mm: (6,) [xmin,ymin,zmin,xmax,ymax,zmax]

    Returns:
        corners: (8,3) corners in (x,y,z)
    """
    xmin, ymin, zmin, xmax, ymax, zmax = [float(v) for v in np.asarray(bbox_mm).reshape(6)]
    xs = [xmin, xmax]
    ys = [ymin, ymax]
    zs = [zmin, zmax]
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)


# -------------------------------------------------------
# Optional: bundle decode outputs
# -------------------------------------------------------
@dataclass(frozen=True)
class DecodeConfig:
    """
    Common decode settings for inference.
    """
    clamp_min_size_mm: float = 10.0
    margin_mm: float = 0.0


def decode_prediction(
    heat_pred: np.ndarray,
    size_pred: np.ndarray,
    img_resampled: sitk.Image,
    cfg: DecodeConfig = DecodeConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full decode:
        heatmap -> center_mm
        clamp size -> bbox_mm

    Args:
        heat_pred: predicted heatmap (Z,Y,X) (NumPy)
        size_pred: predicted size (3,) in mm (NumPy)
        img_resampled: SimpleITK image of the inference grid
        cfg: DecodeConfig

    Returns:
        center_mm: (3,)
        bbox_mm: (6,)
    """
    center_mm = center_mm_from_heatmap(heat_pred, img_resampled)

    size_mm = np.asarray(size_pred, dtype=np.float32).reshape(3)
    size_mm = np.maximum(size_mm, float(cfg.clamp_min_size_mm))

    bbox_mm = bbox_from_center_size_mm(center_mm, size_mm, margin_mm=float(cfg.margin_mm))
    return center_mm, bbox_mm