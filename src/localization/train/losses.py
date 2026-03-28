"""
localization.train.losses

Loss functions for the localization model.

Default training objective (matches your notebook):
    loss = MSE(heat_pred, heat_tgt) + size_w * SmoothL1(size_pred, size_tgt)

Notes:
- Heatmap targets are Gaussian-like values in [0,1].
  MSE is a common choice for regression-to-heatmap.
- If you later change the heat head to output logits and want BCE,
  you can switch heat_loss_type to "bce".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict

import torch
import torch.nn.functional as F


HeatLossType = Literal["mse", "bce"]


@dataclass(frozen=True)
class LossConfig:
    """
    Configuration for the combined localization loss.
    """
    heat_loss: HeatLossType = "mse"
    size_weight: float = 0.1
    bce_pos_weight: float = 1.0
    size_loss: str = "mse"   # "mse", "l1", "smooth_l1"

def heatmap_loss(
    heat_pred: torch.Tensor,
    heat_tgt: torch.Tensor,
    loss_type: HeatLossType = "mse",
    bce_pos_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute heatmap loss.

    Args:
        heat_pred: (B,1,Z,Y,X)
        heat_tgt:  (B,1,Z,Y,X)
        loss_type: "mse" or "bce"
        bce_pos_weight: used only for BCEWithLogitsLoss (pos_weight)

    Returns:
        scalar loss tensor
    """
    if loss_type == "mse":
        return F.mse_loss(heat_pred, heat_tgt)

    if loss_type == "bce":

        pos_w = torch.tensor([float(bce_pos_weight)], device=heat_pred.device, dtype=heat_pred.dtype)
        return F.binary_cross_entropy_with_logits(heat_pred, heat_tgt, pos_weight=pos_w)

    raise ValueError(f"Unknown heat loss type: {loss_type}")


def size_loss(
    size_pred: torch.Tensor,
    size_tgt: torch.Tensor,
    loss_type: str = "mse",
    ) -> torch.Tensor:
    """
    Compute size regression loss.

    Args:
        size_pred: (B,3)
        size_tgt:  (B,3)
        loss_type: "mse", "l1", or "smooth_l1"

    Returns:
        scalar loss tensor
    """
    if loss_type == "mse":
        return F.mse_loss(size_pred, size_tgt)
    if loss_type == "l1":
        return F.l1_loss(size_pred, size_tgt)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(size_pred, size_tgt)

    raise ValueError(f"Unknown size loss type: {loss_type}")


def localizer_loss(
    heat_pred: torch.Tensor,
    heat_tgt: torch.Tensor,
    size_pred: torch.Tensor,
    size_tgt: torch.Tensor,
    cfg: LossConfig = LossConfig(),
) -> Dict[str, torch.Tensor]:
    """
    Combined loss for the localizer.

    Returns a dict with:
        - total
        - heat
        - size

    This is convenient for logging.
    """
    l_heat = heatmap_loss(
        heat_pred=heat_pred,
        heat_tgt=heat_tgt,
        loss_type=cfg.heat_loss,
        bce_pos_weight=cfg.bce_pos_weight,
    )
    l_size = size_loss(
        size_pred=size_pred,
        size_tgt=size_tgt,
        loss_type=cfg.size_loss,
    )

    total = l_heat + float(cfg.size_weight) * l_size

    return {"total": total, "heat": l_heat, "size": l_size}