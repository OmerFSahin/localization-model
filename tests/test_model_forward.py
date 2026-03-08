"""
tests/test_model_forward.py

Model forward-pass tests.

Verifies:
- forward works on odd spatial sizes (shape-safe skip connections)
- output shapes are correct
- outputs are finite

Run:
    pytest tests/test_model_forward.py
"""

import torch

from localization.models.unet3d import LocalizerNet


def test_model_forward_shapes_and_finiteness():
    net = LocalizerNet(base=16, dropout=0.0, positive_size=False)
    net.eval()

    # Deliberately choose odd sizes to stress skip connections.
    B = 2
    Z, Y, X = 65, 81, 97
    x = torch.randn(B, 1, Z, Y, X)

    with torch.no_grad():
        heat, size = net(x)

    # --- shapes ---
    assert heat.shape == (B, 1, Z, Y, X), f"Unexpected heat shape: {heat.shape}"
    assert size.shape == (B, 3), f"Unexpected size shape: {size.shape}"

    # --- finiteness ---
    assert torch.isfinite(heat).all(), "Heat output contains NaN/Inf"
    assert torch.isfinite(size).all(), "Size output contains NaN/Inf"


def test_model_forward_positive_size_option():
    net = LocalizerNet(base=8, dropout=0.0, positive_size=True)
    net.eval()

    x = torch.randn(1, 1, 33, 35, 37)

    with torch.no_grad():
        _, size = net(x)

    # softplus should enforce strictly positive values
    assert (size > 0).all(), "positive_size=True but size has non-positive values"