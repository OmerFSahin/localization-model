"""
tests/test_world_vox_roundtrip.py

Round-trip tests for coordinate transforms:
    voxel -> world -> voxel

This verifies that geometry conversions are consistent.

Run:
    pytest tests/test_world_vox_roundtrip.py
"""

import numpy as np
import SimpleITK as sitk

from localization.geometry.coords import vox_to_world, world_to_vox


def _make_test_image(
    size_xyz=(64, 80, 32),
    spacing_xyz=(1.5, 2.0, 3.0),
    origin_xyz=(10.0, -20.0, 5.0),
    direction_rowmajor=None,
):
    """
    Create a SimpleITK image with specified metadata.
    Content doesn't matter; only geometry does.
    """
    # Create dummy array in (Z,Y,X), then convert to sitk.Image
    Z = size_xyz[2]
    Y = size_xyz[1]
    X = size_xyz[0]
    arr = np.zeros((Z, Y, X), dtype=np.float32)

    img = sitk.GetImageFromArray(arr)  # creates image with size (X,Y,Z)
    img.SetSpacing(tuple(float(v) for v in spacing_xyz))
    img.SetOrigin(tuple(float(v) for v in origin_xyz))

    if direction_rowmajor is None:
        direction_rowmajor = (1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0)
    img.SetDirection(tuple(float(v) for v in direction_rowmajor))

    return img


def test_roundtrip_identity_direction():
    img = _make_test_image(
        size_xyz=(64, 80, 32),
        spacing_xyz=(1.25, 2.5, 3.0),
        origin_xyz=(100.0, -50.0, 10.0),
        direction_rowmajor=(1, 0, 0, 0, 1, 0, 0, 0, 1),
    )

    pts_vox = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.5, 20.25, 5.75],
            [63.0, 79.0, 31.0],
        ],
        dtype=np.float32,
    )

    mm = vox_to_world(pts_vox, img)
    vox2 = world_to_vox(mm, img)

    err = np.linalg.norm(vox2 - pts_vox, axis=1)
    assert np.all(err < 1e-4), f"Round-trip voxel error too large: {err}"


def test_roundtrip_rotated_direction_90deg_about_z():
    # 90-degree rotation about Z axis:
    # x' = -y, y' = x (in direction cosines matrix form)
    direction = (
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 1.0,
    )

    img = _make_test_image(
        size_xyz=(40, 50, 20),
        spacing_xyz=(2.0, 2.0, 2.0),
        origin_xyz=(0.0, 0.0, 0.0),
        direction_rowmajor=direction,
    )

    pts_vox = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 5.0, 0.0],
            [39.0, 49.0, 19.0],
        ],
        dtype=np.float32,
    )

    mm = vox_to_world(pts_vox, img)
    vox2 = world_to_vox(mm, img)

    err = np.linalg.norm(vox2 - pts_vox, axis=1)
    assert np.all(err < 1e-4), f"Round-trip voxel error too large: {err}"