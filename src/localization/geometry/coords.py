"""
localization.geometry.coords

Coordinate conversion helpers for SimpleITK images.

In medical imaging:
- "World"/physical coordinates are in millimeters (mm): (x, y, z)
- "Voxel"/index coordinates are in pixels/voxels: (x, y, z)
- SimpleITK spatial metadata uses (x, y, z) ordering:
    * img.GetSpacing()  -> (sx, sy, sz)
    * img.GetOrigin()   -> (ox, oy, oz)
    * img.GetDirection() -> 3x3 direction cosines (flattened row-major)

IMPORTANT:
- NumPy arrays from sitk.GetArrayFromImage are (z, y, x) — this is a different order.
  These functions operate ONLY on (x, y, z) world/voxel coordinates.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import SimpleITK as sitk


def affine_world_from_image(img: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the affine transform that maps voxel (x,y,z) -> world mm (x,y,z):

        world = origin + (Direction @ diag(Spacing)) @ voxel

    Returns:
        A: (3,3) matrix
        origin: (3,) vector

    Notes:
        - Direction is a rotation-like matrix (direction cosines)
        - Spacing scales axes in mm
    """
    direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)  # row-major
    spacing = np.array(img.GetSpacing(), dtype=np.float64)                    # (sx,sy,sz)
    A = direction @ np.diag(spacing)
    origin = np.array(img.GetOrigin(), dtype=np.float64)
    return A, origin


def vox_to_world(points_xyz_vox: np.ndarray, img: sitk.Image) -> np.ndarray:
    """
    Convert voxel coordinates (x,y,z) -> world coordinates (x,y,z) in mm.

    Args:
        points_xyz_vox: (N,3) voxel coords in (x,y,z) order
        img: SimpleITK image providing spacing/origin/direction

    Returns:
        points_xyz_mm: (N,3) world coords in mm
    """
    pts = np.asarray(points_xyz_vox, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz_vox must have shape (N,3), got {pts.shape}")

    A, origin = affine_world_from_image(img)
    mm = origin[None, :] + (pts @ A.T)
    return mm.astype(np.float32)


def world_to_vox(points_xyz_mm: np.ndarray, img: sitk.Image) -> np.ndarray:
    """
    Convert world coordinates (x,y,z) in mm -> voxel coordinates (x,y,z).

    Args:
        points_xyz_mm: (N,3) world coords in mm (x,y,z)
        img: SimpleITK image providing spacing/origin/direction

    Returns:
        points_xyz_vox: (N,3) voxel coords (x,y,z) in floating point
                        (not rounded; you can round/floor/ceil depending on use)
    """
    pts = np.asarray(points_xyz_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz_mm must have shape (N,3), got {pts.shape}")

    A, origin = affine_world_from_image(img)
    invA = np.linalg.inv(A)

    # (pts - origin) mapped back into voxel space
    vox = (invA @ (pts.T - origin[:, None])).T
    return vox.astype(np.float32)


def clamp_vox_to_image(points_xyz_vox: np.ndarray, img: sitk.Image) -> np.ndarray:
    """
    Clamp voxel coordinates to valid index ranges of the image.

    Useful before slicing or drawing boxes.

    Args:
        points_xyz_vox: (N,3) voxel coords (float or int)
        img: SimpleITK image (for size)

    Returns:
        clamped voxel coords as float32
    """
    pts = np.asarray(points_xyz_vox, dtype=np.float64)
    size_xyz = np.array(img.GetSize(), dtype=np.float64)  # (x,y,z)
    lo = np.zeros(3, dtype=np.float64)
    hi = size_xyz - 1.0
    pts = np.minimum(np.maximum(pts, lo[None, :]), hi[None, :])
    return pts.astype(np.float32)


def voxel_round(points_xyz_vox: np.ndarray) -> np.ndarray:
    """
    Round voxel coordinates to nearest integer indices.

    Returns int64 array with same shape (N,3).
    """
    pts = np.asarray(points_xyz_vox)
    return np.rint(pts).astype(np.int64)


def voxel_floor(points_xyz_vox: np.ndarray) -> np.ndarray:
    """
    Floor voxel coordinates to integer indices.

    Returns int64 array with same shape (N,3).
    """
    pts = np.asarray(points_xyz_vox)
    return np.floor(pts).astype(np.int64)


def voxel_ceil(points_xyz_vox: np.ndarray) -> np.ndarray:
    """
    Ceil voxel coordinates to integer indices.

    Returns int64 array with same shape (N,3).
    """
    pts = np.asarray(points_xyz_vox)
    return np.ceil(pts).astype(np.int64)