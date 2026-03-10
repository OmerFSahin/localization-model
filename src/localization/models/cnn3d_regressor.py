"""
localization.models.cnn3d_regressor

Simple 3D CNN baseline for localization.

Inputs:
    x: (B, 1, Z, Y, X)

Outputs:
    heat: (B, 1, Z, Y, X)
    size: (B, 3)

Notes:
- This is a lightweight alternative to U-Net / ResNet-style models.
- It uses a plain 3D CNN encoder without skip connections or residual blocks.
- The heatmap is predicted from low-resolution features and upsampled back
  to the input resolution.
- The size head uses global average pooled bottleneck features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    """
    Simple 3D conv block with InstanceNorm and LeakyReLU.
    """
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.LeakyReLU(0.1, inplace=True),

        nn.Conv3d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.LeakyReLU(0.1, inplace=True),
    )


class CNN3DRegressor(nn.Module):
    """
    Lightweight 3D CNN for heatmap localization + size regression.

    Args:
        base: base channel count (typical: 8, 16, 32)
        dropout: dropout probability for size head
        positive_size: if True, enforce size > 0 using softplus

    Forward:
        heat, size = net(x)
    """
    def __init__(self, base: int = 16, dropout: float = 0.0, positive_size: bool = False):
        super().__init__()

        self.base = int(base)
        self.positive_size = bool(positive_size)

        # Encoder
        self.enc1 = conv_block(1, base, stride=1)
        self.enc2 = conv_block(base, base * 2, stride=2)
        self.enc3 = conv_block(base * 2, base * 4, stride=2)
        self.enc4 = conv_block(base * 4, base * 8, stride=2)

        # Heat head
        self.heat_head = nn.Sequential(
            nn.Conv3d(base * 8, base * 4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base * 4, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(base * 4, 1, kernel_size=1),
        )

        # Size head
        self.gap = nn.AdaptiveAvgPool3d(1)   # -> (B, C, 1, 1, 1)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.head_size = nn.Linear(base * 8, 3)

    def forward(self, x: torch.Tensor):
        input_shape = x.shape[2:]  # (Z, Y, X)

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        feat = self.enc4(x3)

        # Heat head (low-res -> full-res)
        heat_lowres = self.heat_head(feat)
        heat = F.interpolate(
            heat_lowres,
            size=input_shape,
            mode="trilinear",
            align_corners=False,
        )

        # Size head
        pooled = self.gap(feat).flatten(1)   # (B, base*8)
        pooled = self.drop(pooled)
        size = self.head_size(pooled)        # (B, 3)

        if self.positive_size:
            size = F.softplus(size)

        return heat, size