"""
localization.models.unet3d

Shape-safe 3D U-Net style architecture for localization.

Inputs:
    x: (B, 1, Z, Y, X)

Outputs:
    heat: (B, 1, Z, Y, X)  (raw regression output; often trained with MSE vs. gaussian target)
    size: (B, 3)           (predicted box size in mm)

Notes:
- We use InstanceNorm3d, which is common in medical imaging.
- We include a spatial matching helper to prevent skip-connection concat failures
  when dimensions are not perfectly divisible by the downsampling factor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    """
    A typical U-Net conv block: conv-norm-act twice.

    Using InstanceNorm3d is common for 3D medical volumes.
    LeakyReLU provides stable gradients.
    """
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.LeakyReLU(0.1, inplace=True),

        nn.Conv3d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.LeakyReLU(0.1, inplace=True),
    )


class LocalizerNet(nn.Module):
    """
    A compact 3D U-Net for heatmap localization + size regression.

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
        self.enc1 = conv_block(1, base)
        self.down1 = nn.Conv3d(base, base * 2, kernel_size=2, stride=2)
        self.enc2 = conv_block(base * 2, base * 2)

        self.down2 = nn.Conv3d(base * 2, base * 4, kernel_size=2, stride=2)
        self.enc3 = conv_block(base * 4, base * 4)

        self.down3 = nn.Conv3d(base * 4, base * 8, kernel_size=2, stride=2)
        self.bott = conv_block(base * 8, base * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        # Heads
        self.head_heat = nn.Conv3d(base, 1, kernel_size=1)

        self.gap = nn.AdaptiveAvgPool3d(1)   # -> (B, C, 1, 1, 1)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.head_size = nn.Linear(base, 3)

    @staticmethod
    def _match_spatial(u: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Pad/crop tensor 'u' to match spatial shape of 'ref' (Z,Y,X).

        This prevents concat failures when shapes differ by 1 voxel due to odd sizes
        or because input was not perfectly divisible by 2^N.

        Strategy:
        - If u is smaller: pad symmetrically where possible
        - If u is larger: crop at the end
        """
        dz = ref.size(2) - u.size(2)
        dy = ref.size(3) - u.size(3)
        dx = ref.size(4) - u.size(4)

        # pad format for F.pad 3D: (x_left, x_right, y_left, y_right, z_left, z_right)
        pad = [
            max(0, dx // 2), max(0, dx - dx // 2),
            max(0, dy // 2), max(0, dy - dy // 2),
            max(0, dz // 2), max(0, dz - dz // 2),
        ]
        if any(pad):
            u = F.pad(u, pad)

        # crop if oversized
        if u.size(2) > ref.size(2):
            u = u[:, :, :ref.size(2), :, :]
        if u.size(3) > ref.size(3):
            u = u[:, :, :, :ref.size(3), :]
        if u.size(4) > ref.size(4):
            u = u[:, :, :, :, :ref.size(4)]

        return u

    def forward(self, x: torch.Tensor):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bott(self.down3(e3))

        # Decoder + skip connections
        u3 = self._match_spatial(self.up3(b), e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self._match_spatial(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self._match_spatial(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # Heads
        heat = self.head_heat(d1)  # (B,1,Z,Y,X)

        pooled = self.gap(d1).flatten(1)     # (B, base)
        pooled = self.drop(pooled)
        size = self.head_size(pooled)        # (B,3)

        if self.positive_size:
            # softplus ensures positive predictions (helps early training stability)
            size = F.softplus(size)

        return heat, size