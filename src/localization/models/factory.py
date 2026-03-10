"""
localization.models.factory

Model factory for localization experiments.

This module provides a small helper to instantiate models by name, so the
training / evaluation scripts can stay clean and support multiple backbones.
"""

from __future__ import annotations

from localization.models.unet3d import LocalizerNet
from localization.models.cnn3d_regressor import CNN3DRegressor
from localization.models.resnet3d_regressor import ResNet3DRegressor


def build_model(
    name: str,
    base: int = 16,
    dropout: float = 0.0,
    positive_size: bool = False,
):
    """
    Build a localization model by name.

    Args:
        name: model name
            Supported:
            - "unet3d"
            - "cnn3d_regressor"
            - "resnet3d_regressor"
        base: base channel count
        dropout: dropout probability for size head
        positive_size: if True, enforce positive size predictions

    Returns:
        nn.Module

    Raises:
        ValueError: if model name is unknown
    """
    name = str(name).strip().lower()

    if name == "unet3d":
        return LocalizerNet(
            base=base,
            dropout=dropout,
            positive_size=positive_size,
        )

    if name == "cnn3d_regressor":
        return CNN3DRegressor(
            base=base,
            dropout=dropout,
            positive_size=positive_size,
        )

    if name == "resnet3d_regressor":
        return ResNet3DRegressor(
            base=base,
            dropout=dropout,
            positive_size=positive_size,
        )

    raise ValueError(
        f"Unknown model: {name}. "
        "Available models: unet3d, cnn3d_regressor, resnet3d_regressor"
    )