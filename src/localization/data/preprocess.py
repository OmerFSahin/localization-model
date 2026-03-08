"""
localization.data.preprocess

Preprocessing utilities shared by:
- training dataset pipeline
- inference pipeline
- visualization pipeline

Main responsibilities:
1) Intensity normalization (CT-friendly defaults)
2) Shape padding to satisfy network stride constraints (e.g., multiple of 8)
3) Keeping padding consistent across inputs and targets (image + heatmap)

Conventions:
- Volume arrays are NumPy arrays in (Z, Y, X) order.
"""

from __future__ import annotations

from typing import Tuple, Union, Optional
import numpy as np


Array = np.ndarray
PadSpec = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]  # ((z0,z1),(y0,y1),(x0,x1))


# -------------------------------------------------------
# 1) CT normalization
# -------------------------------------------------------
def normalize_ct(
    arr_zyx: Array,
    clip: Tuple[float, float] = (-150.0, 350.0),
    eps: float = 1e-6,
) -> Array:
    """
    Normalize a CT volume using:
    - window clipping
    - per-volume z-score normalization

    Args:
        arr_zyx: input volume (Z,Y,X)
        clip: (min, max) HU window
        eps: numerical stability

    Returns:
        normalized volume (float32, Z,Y,X)
    """
    v = arr_zyx.astype(np.float32, copy=False)
    v = np.clip(v, clip[0], clip[1])

    mean = float(v.mean())
    std = float(v.std()) + float(eps)

    v = (v - mean) / std
    return v.astype(np.float32, copy=False)


# -------------------------------------------------------
# 2) Padding to a multiple of k
# -------------------------------------------------------
def pad_spec_for_shape(shape_zyx: Tuple[int, int, int], k: int = 8) -> PadSpec:
    """
    Compute a padding spec to make each dimension a multiple of k.

    We pad only at the end of each axis:
      Z: (0, pz)
      Y: (0, py)
      X: (0, px)

    Args:
        shape_zyx: (Z, Y, X)
        k: target multiple (e.g., 8 for 3 downsamplings)

    Returns:
        pad_spec: ((0,pz), (0,py), (0,px))
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    Z, Y, X = shape_zyx
    Z2 = ((Z + k - 1) // k) * k
    Y2 = ((Y + k - 1) // k) * k
    X2 = ((X + k - 1) // k) * k

    pz, py, px = Z2 - Z, Y2 - Y, X2 - X
    return ((0, int(pz)), (0, int(py)), (0, int(px)))


def apply_pad(arr_zyx: Array, pad_spec: PadSpec, mode: str = "constant", value: float = 0.0) -> Array:
    """
    Apply a pad spec (Z,Y,X) to an array.

    Args:
        arr_zyx: array (Z,Y,X)
        pad_spec: ((z0,z1),(y0,y1),(x0,x1))
        mode: np.pad mode
        value: constant pad value (only used if mode="constant")

    Returns:
        padded array
    """
    if pad_spec == ((0, 0), (0, 0), (0, 0)):
        return arr_zyx

    if mode == "constant":
        return np.pad(arr_zyx, pad_spec, mode=mode, constant_values=value)
    return np.pad(arr_zyx, pad_spec, mode=mode)


def pad_to_multiple(arr_zyx: Array, k: int = 8, mode: str = "constant", value: float = 0.0) -> tuple[Array, PadSpec]:
    """
    Convenience function:
    - compute pad spec for arr.shape
    - apply it

    Returns padded array and the pad spec.
    """
    spec = pad_spec_for_shape(arr_zyx.shape, k=k)
    return apply_pad(arr_zyx, spec, mode=mode, value=value), spec


# -------------------------------------------------------
# 3) Unpadding (useful at inference time)
# -------------------------------------------------------
def unpad(arr_zyx: Array, pad_spec: PadSpec) -> Array:
    """
    Remove padding added by apply_pad/pad_to_multiple.

    Args:
        arr_zyx: padded array (Z,Y,X)
        pad_spec: ((z0,z1),(y0,y1),(x0,x1))

    Returns:
        unpadded array
    """
    (z0, z1), (y0, y1), (x0, x1) = pad_spec

    Z, Y, X = arr_zyx.shape
    z_end = Z - z1
    y_end = Y - y1
    x_end = X - x1

    return arr_zyx[z0:z_end, y0:y_end, x0:x_end]