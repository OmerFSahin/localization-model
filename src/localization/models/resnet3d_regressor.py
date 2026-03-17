"""
localization.models.resnet3d_regressor

Residual 3D CNN for localization.

Inputs:
    x: (B, 1, Z, Y, X)

Outputs:
    heat: (B, 1, Z, Y, X)
    size: (B, 3)

Notes:
- This is a ResNet-style encoder with a lightweight heatmap head.
- The heatmap is predicted from lower-resolution encoder features and then
  upsampled back to input size.
- The size head uses global average pooled bottleneck features.
- Compared with U-Net, this model has no skip-connected decoder, so it is a
  simpler alternative baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_norm_act(cin: int, cout: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    """
    3D conv block with InstanceNorm and LeakyReLU.
    """
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.LeakyReLU(0.1, inplace=True),
    )


class ResidualBlock3D(nn.Module):
    """
    Basic 3D residual block.

    If spatial size or channel count changes, the skip path is projected with
    a 1x1x1 convolution.
    """
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(cout, affine=True)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv3d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(cout, affine=True)

        if cin != cout or stride != 1:
            self.proj = nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(cout, affine=True),
            )
        else:
            self.proj = nn.Identity()

        self.act2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act2(out)
        return out


class ResNet3DRegressor(nn.Module):
    """
    Residual 3D CNN for heatmap localization + size regression.

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

        # Stem
        self.stem = conv_norm_act(1, base, kernel_size=3, stride=1)

        # Encoder
        self.layer1 = nn.Sequential(
            ResidualBlock3D(base, base, stride=1),
            ResidualBlock3D(base, base, stride=1),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock3D(base, base * 2, stride=2),
            ResidualBlock3D(base * 2, base * 2, stride=1),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock3D(base * 2, base * 4, stride=2),
            ResidualBlock3D(base * 4, base * 4, stride=1),
        )

        self.layer4 = nn.Sequential(
            ResidualBlock3D(base * 4, base * 8, stride=2),
            ResidualBlock3D(base * 8, base * 8, stride=1),
        )

        # Heat head from bottleneck features
        self.heat_head = nn.Sequential(
            conv_norm_act(base * 8, base * 4, kernel_size=3, stride=1),
            nn.Conv3d(base * 4, 1, kernel_size=1),
        )

        # Size head
        self.gap = nn.AdaptiveAvgPool3d(1)   # -> (B, C, 1, 1, 1)

        self.size_head = nn.Sequential(
            nn.Linear(base * 8, base * 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(base * 8, base * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(base * 4, 3),
        )

    def forward(self, x: torch.Tensor):
        input_shape = x.shape[2:]  # (Z, Y, X)

        # Encoder
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)

        # Heat head at low resolution, then upsample to full resolution
        heat_lowres = self.heat_head(feat)
        heat = F.interpolate(
            heat_lowres,
            size=input_shape,
            mode="trilinear",
            align_corners=False,
        )

        # Size head from global pooled bottleneck
        pooled = self.gap(feat).flatten(1)   # (B, base*8)
        size = self.size_head(pooled)        # (B, 3)

        if self.positive_size:
            size = F.softplus(size)

        return heat, size