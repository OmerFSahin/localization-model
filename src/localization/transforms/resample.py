"""
localization.transforms.resample

Resampling utilities for 3D medical images (SimpleITK).

Why this module exists:
- Dataset training uses a fixed target spacing (e.g., 2mm isotropic)
- Inference and visualization must apply the *same* resampling
- Centralizing this logic avoids subtle inconsistencies

Key concepts:
- SimpleITK image metadata axis order is (x, y, z)
- NumPy arrays from sitk.GetArrayFromImage are (z, y, x)
"""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np
import SimpleITK as sitk


Spacing3 = Union[Tuple[float, float, float], Iterable[float]]


def _as_spacing3(spacing: Spacing3) -> Tuple[float, float, float]:
    """
    Normalize spacing input into a strict (x,y,z) float tuple.
    """
    s = tuple(float(v) for v in spacing)
    if len(s) != 3:
        raise ValueError(f"Expected spacing with 3 values (x,y,z), got: {spacing}")
    return (s[0], s[1], s[2])


def compute_out_size_xyz(img: sitk.Image, out_spacing_xyz: Spacing3) -> Tuple[int, int, int]:
    """
    Compute output voxel size (x,y,z) for a given target spacing.

    We preserve the physical extent approximately:
        out_size ≈ in_size * (in_spacing / out_spacing)

    Notes:
    - img.GetSize() returns (x,y,z)
    - img.GetSpacing() returns (x,y,z)
    """
    out_spacing_xyz = _as_spacing3(out_spacing_xyz)

    in_size = np.array(img.GetSize(), dtype=np.float64)      # (x,y,z)
    in_sp = np.array(img.GetSpacing(), dtype=np.float64)     # (x,y,z)
    out_sp = np.array(out_spacing_xyz, dtype=np.float64)     # (x,y,z)

    # floor keeps size conservative; you can also use round/ceil depending on preference.
    out_size = np.round(((in_size - 1) * in_sp) / out_sp).astype(np.int64) + 1

    # Safety: avoid zeros (can happen for tiny dimensions)
    out_size = np.maximum(out_size, 1)

    return tuple(int(v) for v in out_size)

def resample_to_spacing(
    img: sitk.Image,
    out_spacing_xyz: Spacing3 = (2.0, 2.0, 2.0),
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """
    Resample a 3D image to the desired spacing.

    Keeps:
    - origin
    - direction

    Sets:
    - output spacing (x,y,z)
    - output size (x,y,z) computed from physical extent

    Args:
        img: input SimpleITK image
        out_spacing_xyz: target spacing (x,y,z) in mm
        interpolator: sitk.sitkLinear for intensities (CT/MR),
                      sitk.sitkNearestNeighbor for labels/masks

    Returns:
        Resampled SimpleITK image
    """
    out_spacing_xyz = _as_spacing3(out_spacing_xyz)
    out_size_xyz = compute_out_size_xyz(img, out_spacing_xyz)

    f = sitk.ResampleImageFilter()
    f.SetInterpolator(interpolator)

    # Keep the same physical coordinate system
    f.SetOutputSpacing(out_spacing_xyz)
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())

    # Define the new voxel grid size
    f.SetSize(out_size_xyz)

    return f.Execute(img)


# Backwards-compatible alias (matches your notebook naming)
def sitk_resample_iso(
    img: sitk.Image,
    out_spacing: Spacing3 = (2.0, 2.0, 2.0),
    interp: int = sitk.sitkLinear,
) -> sitk.Image:
    """
    Alias for resample_to_spacing() to match earlier notebook code.

    out_spacing: (x,y,z)
    """
    return resample_to_spacing(img, out_spacing_xyz=out_spacing, interpolator=interp)