"""
localization.targets.heatmap

Heatmap target generation for localization.

Conventions used here:
- Volume arrays are (Z, Y, X) in NumPy (as returned by sitk.GetArrayFromImage)
- Center coordinates are given as voxel coordinates in (x, y, z)
  (this is the natural indexing order for SimpleITK geometry functions)

So:
- shape_zyx = (Z, Y, X)
- center_vox_xyz = (cx, cy, cz)

This module returns a float32 heatmap with values in [0,1].
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def make_heatmap_meshgrid(
    shape_zyx: Tuple[int, int, int],
    center_vox_xyz: np.ndarray,
    sigma_vox: float = 3.0,
) -> np.ndarray:
    """
    Meshgrid-based Gaussian heatmap (simple but memory heavy).

    Args:
        shape_zyx: (Z, Y, X)
        center_vox_xyz: (3,) center in voxel coords (x,y,z)
        sigma_vox: Gaussian sigma in voxels

    Returns:
        heatmap (Z,Y,X), float32
    """
    Z, Y, X = shape_zyx
    z, y, x = np.meshgrid(
        np.arange(Z, dtype=np.float32),
        np.arange(Y, dtype=np.float32),
        np.arange(X, dtype=np.float32),
        indexing="ij",
    )

    cx, cy, cz = [float(v) for v in center_vox_xyz]
    c_zyx = np.array([cz, cy, cx], dtype=np.float32)

    # grid shape: (3, Z, Y, X)
    grid = np.stack([z, y, x], axis=0)
    d2 = ((grid - c_zyx[:, None, None, None]) ** 2).sum(axis=0)

    heat = np.exp(-0.5 * d2 / (sigma_vox**2)).astype(np.float32)
    return heat


def make_heatmap_separable(
    shape_zyx: Tuple[int, int, int],
    center_vox_xyz: np.ndarray,
    sigma_vox: float = 3.0,
) -> np.ndarray:
    """
    Memory-friendly Gaussian heatmap using separability:

        heat[z,y,x] = gz[z] * gy[y] * gx[x]

    This avoids allocating large meshgrids.

    Args:
        shape_zyx: (Z, Y, X)
        center_vox_xyz: (3,) center in voxel coords (x,y,z)
        sigma_vox: Gaussian sigma in voxels

    Returns:
        heatmap (Z,Y,X), float32
    """
    Z, Y, X = shape_zyx
    cx, cy, cz = [float(v) for v in center_vox_xyz]

    z = np.arange(Z, dtype=np.float32)
    y = np.arange(Y, dtype=np.float32)
    x = np.arange(X, dtype=np.float32)

    # 1D Gaussians (note: cz corresponds to Z axis, etc.)
    gz = np.exp(-0.5 * ((z - cz) ** 2) / (sigma_vox**2)).astype(np.float32)  # (Z,)
    gy = np.exp(-0.5 * ((y - cy) ** 2) / (sigma_vox**2)).astype(np.float32)  # (Y,)
    gx = np.exp(-0.5 * ((x - cx) ** 2) / (sigma_vox**2)).astype(np.float32)  # (X,)

    heat = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    return heat.astype(np.float32)


def make_heatmap(
    shape_zyx: Tuple[int, int, int],
    center_vox_xyz: np.ndarray,
    sigma_vox: float = 3.0,
    method: str = "separable",
) -> np.ndarray:
    """
    Public API for heatmap generation.

    method:
        - "separable" (recommended)
        - "meshgrid"  (simple but can use a lot of RAM)

    Returns:
        heatmap (Z,Y,X), float32
    """
    if sigma_vox <= 0:
        raise ValueError(f"sigma_vox must be > 0, got {sigma_vox}")

    method = method.lower().strip()
    if method == "separable":
        return make_heatmap_separable(shape_zyx, center_vox_xyz, sigma_vox=sigma_vox)
    elif method == "meshgrid":
        return make_heatmap_meshgrid(shape_zyx, center_vox_xyz, sigma_vox=sigma_vox)
    else:
        raise ValueError(f"Unknown heatmap method '{method}'. Use 'separable' or 'meshgrid'.")