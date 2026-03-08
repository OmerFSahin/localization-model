"""
localization.data.io

This module centralizes all image I/O operations used across the project.

Responsibilities:
- Reading medical images using SimpleITK
- Verifying that image files are readable
- Converting between SimpleITK images and NumPy arrays
- Writing images to disk
- Extracting metadata for debugging and logging

Keeping all I/O logic in one module prevents duplication and ensures
consistent behavior across dataset loading, inference, and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk


PathLike = Union[str, Path]


# -------------------------------------------------------
# 1) Basic image loading
# -------------------------------------------------------
def read_sitk(path: PathLike) -> sitk.Image:
    """
    Read a medical image using SimpleITK.

    This wrapper adds safer error handling compared to calling
    sitk.ReadImage directly.

    Args:
        path: Path to the image file (str or Path).

    Returns:
        SimpleITK Image object.

    Raises:
        FileNotFoundError: if the file does not exist.
        RuntimeError: if SimpleITK fails to read the image.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")

    try:
        img = sitk.ReadImage(str(p))
    except Exception as e:
        raise RuntimeError(
            f"SimpleITK failed to read image: {p}\nReason: {e}"
        ) from e

    return img


def is_readable_sitk(path: PathLike) -> bool:
    """
    Check whether a file exists and can be read by SimpleITK.

    Useful for dataset sanity checks and index building.

    Args:
        path: image file path

    Returns:
        True if readable, False otherwise.
    """
    p = Path(path)

    if not p.exists():
        return False

    try:
        _ = sitk.ReadImage(str(p))
        return True
    except Exception:
        return False


# -------------------------------------------------------
# 2) Locate scan files inside a patient directory
# -------------------------------------------------------
def find_scan_file(
    patient_dir: PathLike,
    patterns: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """
    Search a directory for a likely scan file.

    Many preprocessing pipelines store scans with names such as:
        *(SCAN).nrrd
        *.nrrd
        *.nrrd.gz
        *.nhdr
        *.nii
        *.nii.gz

    This helper returns the first matching file.

    Args:
        patient_dir: directory containing patient data
        patterns: optional custom glob patterns

    Returns:
        Path to scan file or None if nothing found.
    """
    pdir = Path(patient_dir)

    if patterns is None:
        patterns = [
            "*(SCAN).nrrd",
            "*(SCAN).nrrd.gz",
            "*.nrrd",
            "*.nrrd.gz",
            "*.nhdr",
            "*.nii",
            "*.nii.gz",
        ]

    candidates: list[Path] = []

    for pat in patterns:
        candidates.extend(pdir.glob(pat))

    candidates = sorted(candidates)

    return candidates[0] if candidates else None


# -------------------------------------------------------
# 3) SimpleITK <-> NumPy conversions
# -------------------------------------------------------
def sitk_to_numpy(img: sitk.Image, dtype=np.float32) -> np.ndarray:
    """
    Convert SimpleITK image to NumPy array.

    IMPORTANT:
    SimpleITK returns arrays in (Z, Y, X) order.

    Meanwhile spacing/origin/direction follow (x, y, z) order.

    Keeping this distinction clear prevents coordinate bugs
    in localization pipelines.

    Args:
        img: SimpleITK image
        dtype: desired numpy dtype

    Returns:
        NumPy array with shape (Z, Y, X)
    """
    arr = sitk.GetArrayFromImage(img)
    return arr.astype(dtype, copy=False)


def numpy_to_sitk(
    arr_zyx: np.ndarray,
    reference: Optional[sitk.Image] = None
) -> sitk.Image:
    """
    Convert NumPy array (Z,Y,X) to SimpleITK image.

    If a reference image is provided, its spatial metadata
    (spacing, origin, direction) will be copied.

    Args:
        arr_zyx: numpy array with shape (Z,Y,X)
        reference: optional reference image

    Returns:
        SimpleITK image
    """
    img = sitk.GetImageFromArray(arr_zyx)

    if reference is not None:
        img.SetSpacing(reference.GetSpacing())
        img.SetOrigin(reference.GetOrigin())
        img.SetDirection(reference.GetDirection())

    return img


def write_sitk(
    img: sitk.Image,
    path: PathLike,
    use_compression: bool = True
) -> None:
    """
    Write a SimpleITK image to disk.

    Args:
        img: SimpleITK image
        path: output file path
        use_compression: enable compression if supported
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(img, str(p), useCompression=use_compression)


# -------------------------------------------------------
# 4) Image metadata container (useful for debugging)
# -------------------------------------------------------
@dataclass(frozen=True)
class ImageMeta:
    """
    Lightweight container for image metadata.

    This is useful for logging dataset properties or
    debugging coordinate issues.

    Attributes:
        size_xyz: image size in voxel space (x,y,z)
        spacing_xyz: voxel spacing in mm (x,y,z)
        origin_xyz: physical origin (x,y,z)
        direction_rowmajor: direction cosine matrix (flattened)
    """
    size_xyz: Tuple[int, int, int]
    spacing_xyz: Tuple[float, float, float]
    origin_xyz: Tuple[float, float, float]
    direction_rowmajor: Tuple[float, ...]


def get_image_meta(img: sitk.Image) -> ImageMeta:
    """
    Extract metadata from a SimpleITK image.

    Returns a structured object for easier logging
    and debugging.

    NOTE:
    - img.GetSize() returns (x,y,z)
    - NumPy arrays from the image are (z,y,x)

    Returns:
        ImageMeta object
    """
    size = tuple(int(v) for v in img.GetSize())
    spacing = tuple(float(v) for v in img.GetSpacing())
    origin = tuple(float(v) for v in img.GetOrigin())
    direction = tuple(float(v) for v in img.GetDirection())

    return ImageMeta(
        size_xyz=size,
        spacing_xyz=spacing,
        origin_xyz=origin,
        direction_rowmajor=direction,
    )