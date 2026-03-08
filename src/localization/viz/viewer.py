"""
localization.viz.viewer

Visualization helpers for 3D volumes and bounding boxes.

Conventions:
- Volume arrays are NumPy arrays in (Z, Y, X) order.
- Bounding boxes are provided in voxel coordinates of the SAME volume:
    min_xyz = (x_min, y_min, z_min)
    max_xyz = (x_max, y_max, z_max)

We plot three orthogonal views:
- Axial:    Z slice -> image is (Y, X)
- Coronal:  Y slice -> image is (Z, X)
- Sagittal: X slice -> image is (Z, Y) (we transpose for display)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


XYZ = Tuple[int, int, int]
MinMaxXYZ = Tuple[np.ndarray, np.ndarray]  # (min_xyz, max_xyz)


def draw_rect(ax, xmin: int, xmax: int, ymin: int, ymax: int, **kw) -> None:
    """
    Draw a rectangle on a matplotlib axis using (xmin,xmax,ymin,ymax).

    Note: matplotlib Rectangle takes (x,y,width,height) where x,y is top-left for imshow coords.
    """
    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        linewidth=2,
        **kw,
    )
    ax.add_patch(rect)


def minmax_xyz_from_corners(vox_corners_xyz: np.ndarray) -> MinMaxXYZ:
    """
    Convert 8 bbox corners (N,3) in voxel coords (x,y,z) to integer min/max.

    Args:
        vox_corners_xyz: array (N,3)

    Returns:
        (min_xyz, max_xyz) each shape (3,) ints
    """
    c = np.asarray(vox_corners_xyz, dtype=np.float32)
    mins = np.floor(c.min(axis=0)).astype(int)
    maxs = np.ceil(c.max(axis=0)).astype(int)
    return mins, maxs


def clamp_minmax_to_volume(min_xyz: np.ndarray, max_xyz: np.ndarray, vol_shape_zyx: Tuple[int, int, int]) -> MinMaxXYZ:
    """
    Clamp bbox coordinates to volume bounds.

    vol_shape_zyx is (Z,Y,X), but coords are (x,y,z).
    """
    Z, Y, X = vol_shape_zyx

    min_xyz = np.asarray(min_xyz, dtype=int).copy()
    max_xyz = np.asarray(max_xyz, dtype=int).copy()

    min_xyz[0] = int(np.clip(min_xyz[0], 0, X - 1))
    max_xyz[0] = int(np.clip(max_xyz[0], 0, X - 1))

    min_xyz[1] = int(np.clip(min_xyz[1], 0, Y - 1))
    max_xyz[1] = int(np.clip(max_xyz[1], 0, Y - 1))

    min_xyz[2] = int(np.clip(min_xyz[2], 0, Z - 1))
    max_xyz[2] = int(np.clip(max_xyz[2], 0, Z - 1))

    return min_xyz, max_xyz


def plot_three_views(
    vol0_zyx: np.ndarray,
    pred_minmax_xyz: MinMaxXYZ,
    gt_minmax_xyz: Optional[MinMaxXYZ] = None,
    center_xyz: Optional[np.ndarray] = None,
    title_prefix: str = "",
    cmap: str = "gray",
) -> None:
    """
    Plot axial/coronal/sagittal views with bbox overlays.

    Args:
        vol0_zyx: volume array (Z,Y,X)
        pred_minmax_xyz: (min_xyz, max_xyz) in voxel coords
        gt_minmax_xyz: optional GT bbox
        center_xyz: optional center voxel (x,y,z) to choose slices; if None, uses pred bbox center
        title_prefix: string prefix for plot titles
        cmap: matplotlib colormap
    """
    vol = np.asarray(vol0_zyx)
    if vol.ndim != 3:
        raise ValueError(f"Expected vol0_zyx to be 3D (Z,Y,X), got shape {vol.shape}")

    (pmin, pmax) = pred_minmax_xyz
    pmin, pmax = clamp_minmax_to_volume(pmin, pmax, vol.shape)

    if gt_minmax_xyz is not None:
        (gmin, gmax) = gt_minmax_xyz
        gmin, gmax = clamp_minmax_to_volume(gmin, gmax, vol.shape)

    Z, Y, X = vol.shape

    if center_xyz is None:
        center_xyz = ((pmin + pmax) / 2.0).astype(float)

    xx = int(np.clip(round(float(center_xyz[0])), 0, X - 1))
    yy = int(np.clip(round(float(center_xyz[1])), 0, Y - 1))
    zz = int(np.clip(round(float(center_xyz[2])), 0, Z - 1))

    # ---------------- Axial (Z slice): vol[z] is (Y,X) ----------------
    plt.figure(figsize=(6, 6))
    plt.imshow(vol[zz], cmap=cmap)
    ax = plt.gca()
    # rectangle: x range, y range
    draw_rect(ax, pmin[0], pmax[0], pmin[1], pmax[1])
    if gt_minmax_xyz is not None:
        draw_rect(ax, gmin[0], gmax[0], gmin[1], gmax[1], linestyle="--")
    plt.title(f"{title_prefix} | axial z={zz}".strip(" |"))
    plt.axis("off")
    plt.show()

    # --------------- Coronal (Y slice): vol[:,y,:] is (Z,X) ---------------
    plt.figure(figsize=(6, 6))
    cor = vol[:, yy, :]  # (Z,X)
    plt.imshow(cor, cmap=cmap)
    ax = plt.gca()
    # axes: x is X, y is Z
    draw_rect(ax, pmin[0], pmax[0], pmin[2], pmax[2])
    if gt_minmax_xyz is not None:
        draw_rect(ax, gmin[0], gmax[0], gmin[2], gmax[2], linestyle="--")
    plt.title(f"{title_prefix} | coronal y={yy}".strip(" |"))
    plt.axis("off")
    plt.show()

    # -------------- Sagittal (X slice): vol[:,:,x] is (Z,Y) --------------
    # We'll transpose to show as (Y,Z) with origin="lower" for nicer orientation.
    plt.figure(figsize=(6, 6))
    sag = vol[:, :, xx]  # (Z,Y)
    plt.imshow(sag.T, cmap=cmap, origin="lower")  # (Y,Z)
    ax = plt.gca()
    # after transpose: x-axis is Z, y-axis is Y
    draw_rect(ax, pmin[2], pmax[2], pmin[1], pmax[1])
    if gt_minmax_xyz is not None:
        draw_rect(ax, gmin[2], gmax[2], gmin[1], gmax[1], linestyle="--")
    plt.title(f"{title_prefix} | sagittal x={xx}".strip(" |"))
    plt.axis("off")
    plt.show()