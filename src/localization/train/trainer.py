"""
localization.train.trainer

Training loop utilities for the localization model.

This module:
- runs training epochs
- runs validation via localization.eval.metrics.validate_epoch
- saves "best.pt" based on a chosen metric
- writes a history JSON for plotting

Design goals:
- Keep scripts/train.py thin (only parses args/config and calls trainer)
- Keep logic testable and reusable
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Any

import json
import time

import numpy as np
import torch

from localization.train.losses import LossConfig, localizer_loss
from localization.eval.metrics import validate_epoch, ValConfig


# -------------------------------------------------------
# Config
# -------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # Loss weighting
    loss_cfg: LossConfig = LossConfig()

    # Validation behavior
    val_cfg: ValConfig = ValConfig()

    # Checkpoint selection
    best_metric: str = "median_center_error_mm"  # smaller is better
    maximize_best_metric: bool = False           # False -> minimize

    # Logging / saving
    log_every: int = 1

    # Scheduler
    scheduler_name: Optional[str] = None   # None or "step"
    scheduler_step_size: int = 15
    scheduler_gamma: float = 0.5
    scheduler_t_max: int = 50
    scheduler_eta_min: float = 1e-6

def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _is_better(new: float, best: float, maximize: bool) -> bool:
    if np.isnan(new):
        return False
    if best is None:
        return True
    return (new > best) if maximize else (new < best)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def validate_losses(
    net: torch.nn.Module,
    val_dl,
    loss_cfg: LossConfig,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute validation losses over the validation DataLoader.
    """
    net.eval()

    batch_total = []
    batch_heat = []
    batch_size = []

    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)
            heat_t = y["heat"].to(device)
            size_t = y["size"].to(device)

            heat_p, size_p = net(x)

            losses = localizer_loss(
                heat_pred=heat_p,
                heat_tgt=heat_t,
                size_pred=size_p,
                size_tgt=size_t,
                cfg=loss_cfg,
            )

            batch_total.append(float(losses["total"].detach().cpu()))
            batch_heat.append(float(losses["heat"].detach().cpu()))
            batch_size.append(float(losses["size"].detach().cpu()))

    return {
        "val_total_loss": float(np.mean(batch_total)) if batch_total else float("nan"),
        "val_heat_loss": float(np.mean(batch_heat)) if batch_heat else float("nan"),
        "val_size_loss": float(np.mean(batch_size)) if batch_size else float("nan"),
    }

# -------------------------------------------------------
# Main trainer
# -------------------------------------------------------
def train(
    net: torch.nn.Module,
    train_dl,
    val_dl,
    outdir: Path,
    cfg: TrainConfig = TrainConfig(),
    device: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Train the model and save artifacts.

    Args:
        net: model
        train_dl: training DataLoader
        val_dl: validation DataLoader
        outdir: output directory (checkpoints + history)
        cfg: TrainConfig
        device: "cuda" or "cpu" (auto if None)
        optimizer: optional optimizer (if None, uses torch.optim.AdamW)

    Returns:
        dict with:
            - history
            - best_metric_value
            - best_epoch
            - paths
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    net.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = None
    if cfg.scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.scheduler_step_size),
            gamma=float(cfg.scheduler_gamma),
        )
    elif cfg.scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.scheduler_t_max),
            eta_min=float(cfg.scheduler_eta_min),
        )

    history = {
        "started_at": _now_str(),
        "device": device,
        "cfg": {
            **asdict(cfg),
            # dataclasses inside dataclasses: convert explicitly
            "loss_cfg": asdict(cfg.loss_cfg),
            "val_cfg": asdict(cfg.val_cfg),
        },
        "epochs": [],
    }

    best_value: Optional[float] = None
    best_epoch: Optional[int] = None

    best_iou_value: Optional[float] = None
    best_iou_epoch: Optional[int] = None

    best_path = outdir / "best.pt"
    last_path = outdir / "last.pt"
    hist_path = outdir / "history.json"
    best_iou_path = outdir / "best_iou.pt"

    for epoch in range(1, int(cfg.epochs) + 1):
        net.train()
        batch_losses = []
        batch_heat = []
        batch_size = []

        for x, y in train_dl:
            x = x.to(device)
            heat_t = y["heat"].to(device)
            size_t = y["size"].to(device)

            optimizer.zero_grad(set_to_none=True)

            heat_p, size_p = net(x)

            losses = localizer_loss(
                heat_pred=heat_p,
                heat_tgt=heat_t,
                size_pred=size_p,
                size_tgt=size_t,
                cfg=cfg.loss_cfg,
            )

            losses["total"].backward()
            optimizer.step()


            batch_losses.append(float(losses["total"].detach().cpu()))
            batch_heat.append(float(losses["heat"].detach().cpu()))
            batch_size.append(float(losses["size"].detach().cpu()))

        train_total = float(np.mean(batch_losses)) if batch_losses else float("nan")
        train_heat = float(np.mean(batch_heat)) if batch_heat else float("nan")
        train_size = float(np.mean(batch_size)) if batch_size else float("nan")

        # Validation losses
        val_losses = validate_losses(net, val_dl, loss_cfg=cfg.loss_cfg, device=device)

        # Validation metrics
        val_metrics = validate_epoch(net, val_dl, device=device, cfg=cfg.val_cfg)

        current_iou = float(val_metrics.get("mean_iou", float("nan")))

        if not np.isnan(current_iou):
            if best_iou_value is None or current_iou > best_iou_value:
                best_iou_value = current_iou
                best_iou_epoch = epoch
                torch.save(net.state_dict(), best_iou_path)

        # Determine if best
        key = cfg.best_metric
        if key not in val_metrics:
            raise KeyError(f"best_metric '{key}' not in val_metrics keys: {list(val_metrics.keys())}")

        current_value = float(val_metrics[key])

        if _is_better(current_value, best_value, maximize=cfg.maximize_best_metric):
            best_value = current_value
            best_epoch = epoch
            torch.save(net.state_dict(), best_path)

        # Always save last
        torch.save(net.state_dict(), last_path)

        # Log epoch record
        rec = {
            "epoch": epoch,
            "train_total_loss": train_total,
            "train_heat_loss": train_heat,
            "train_size_loss": train_size,
            **val_losses,
            **val_metrics,
        }
        history["epochs"].append(rec)

        # Print
        if (epoch % cfg.log_every) == 0:
            val_total = val_losses.get("val_total_loss", float("nan"))
            val_heat = val_losses.get("val_heat_loss", float("nan"))
            val_size = val_losses.get("val_size_loss", float("nan"))

            med = val_metrics.get("median_center_error_mm", float("nan"))
            p20 = val_metrics.get("p_at_thresh", float("nan")) * 100.0
            miou = val_metrics.get("mean_iou", float("nan"))

            print(
                f"Ep {epoch:03d} | "
                f"train {train_total:.4f} (heat {train_heat:.4f}, size {train_size:.4f}) | "
                f"val {val_total:.4f} (heat {val_heat:.4f}, size {val_size:.4f}) | "
                f"medCE {med:.2f} mm | P@{cfg.val_cfg.success_thresh_mm:.0f} {p20:.1f}% | mIoU {miou:.3f}"
            )

        # Write history each epoch (safe if training crashes)
        _save_json(hist_path, history)

        if scheduler is not None:
            scheduler.step()

    history["finished_at"] = _now_str()
    _save_json(hist_path, history)

    result = {
        "history_path": str(hist_path),
        "best_path": str(best_path),
        "last_path": str(last_path),
        "best_metric": cfg.best_metric,
        "best_metric_value": best_value,
        "best_epoch": best_epoch,
        "outdir": str(outdir),
        "best_iou_path": str(best_iou_path),
        "best_iou_value": best_iou_value,
        "best_iou_epoch": best_iou_epoch,
    }
    return result