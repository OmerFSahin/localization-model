"""
tests/test_dataset_shapes.py

Basic dataset shape test.

This test verifies:
- image tensor shape
- heatmap tensor shape
- size tensor shape
- channel dimensions

Run with:
    pytest tests/test_dataset_shapes.py
"""

from pathlib import Path

import torch

from localization.data.dataset import LocalizerDataset, SampleConfig


INDEX_CSV = Path("data/processed/localizer_index.csv")


def test_dataset_shapes():
    """
    Ensure dataset returns correctly shaped tensors.
    """

    cfg = SampleConfig(
        target_spacing_xyz=(2.0, 2.0, 2.0),
        heat_sigma_vox=3.0,
        ct_clip=(-150.0, 350.0),
        pad_multiple=8,
    )

    ds = LocalizerDataset(INDEX_CSV, split="train", cfg=cfg)

    assert len(ds) > 0, "Dataset is empty"

    x, y = ds[0]

    # ---- image tensor ----
    assert isinstance(x, torch.Tensor)
    assert x.ndim == 4  # (C,Z,Y,X)
    assert x.shape[0] == 1  # single channel

    # ---- heatmap tensor ----
    heat = y["heat"]
    assert isinstance(heat, torch.Tensor)
    assert heat.ndim == 4
    assert heat.shape[0] == 1

    # image and heatmap must match spatial dims
    assert x.shape[1:] == heat.shape[1:], "Image and heatmap shapes mismatch"

    # ---- size tensor ----
    size = y["size"]
    assert isinstance(size, torch.Tensor)
    assert size.shape == (3,), "Size must be (3,) -> (wx,wy,wz)"

    # ---- sanity checks ----
    assert torch.isfinite(x).all()
    assert torch.isfinite(heat).all()
    assert torch.isfinite(size).all()