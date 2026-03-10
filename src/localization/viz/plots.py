"""
localization.viz.plots

Utilities for plotting training history for localization experiments.

This module is designed to work with the history.json written by
localization.train.trainer.

Typical usage:
    from localization.viz.plots import load_history, plot_all_history

    history = load_history("outputs/run01/history.json")
    figs = plot_all_history(history)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


HistoryDict = Dict[str, Any]


def load_history(path: str | Path) -> HistoryDict:
    """
    Load a history.json file.

    Args:
        path: path to history.json

    Returns:
        Parsed history dictionary.
    """
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def moving_average(values: Sequence[float], window: int = 1) -> List[float]:
    """
    Simple causal moving average.

    Args:
        values: sequence of values
        window: smoothing window size (1 = no smoothing)

    Returns:
        Smoothed list of values.
    """
    vals = [float(v) for v in values]
    if window <= 1 or len(vals) == 0:
        return vals

    out: List[float] = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start:i + 1]
        chunk = [v for v in chunk if np.isfinite(v)]
        out.append(float(np.mean(chunk)) if len(chunk) > 0 else float("nan"))
    return out


def get_epoch_records(history: HistoryDict) -> List[Dict[str, Any]]:
    """
    Return epoch records from history.
    """
    return list(history.get("epochs", []))


def get_epoch_numbers(history: HistoryDict) -> List[int]:
    """
    Return epoch numbers from history.
    """
    records = get_epoch_records(history)
    return [int(rec.get("epoch", i + 1)) for i, rec in enumerate(records)]


def extract_series(history: HistoryDict, key: str, default: float = float("nan")) -> List[float]:
    """
    Extract one metric series from history.

    Args:
        history: parsed history dict
        key: metric key
        default: fallback if key is missing

    Returns:
        List of float values.
    """
    values: List[float] = []
    for rec in get_epoch_records(history):
        v = rec.get(key, default)
        try:
            values.append(float(v))
        except Exception:
            values.append(float("nan"))
    return values


def best_epoch_for_metric(
    history: HistoryDict,
    metric_key: str,
    maximize: bool = False,
) -> Optional[int]:
    """
    Find best epoch for a metric.

    Args:
        history: parsed history dict
        metric_key: metric name
        maximize: whether higher is better

    Returns:
        Best epoch number or None.
    """
    best_epoch: Optional[int] = None
    best_value: Optional[float] = None

    for rec in get_epoch_records(history):
        if metric_key not in rec:
            continue

        try:
            value = float(rec[metric_key])
        except Exception:
            continue

        if not np.isfinite(value):
            continue

        if best_value is None:
            best_value = value
            best_epoch = int(rec.get("epoch", 0))
            continue

        better = (value > best_value) if maximize else (value < best_value)
        if better:
            best_value = value
            best_epoch = int(rec.get("epoch", 0))

    return best_epoch


def history_to_table(history: HistoryDict) -> List[Dict[str, Any]]:
    """
    Return epoch records as a plain list-of-dicts table.

    Useful in notebooks for quick inspection.

    Args:
        history: parsed history dict

    Returns:
        List of per-epoch records.
    """
    return get_epoch_records(history)


def _make_title(title: str, title_prefix: str = "") -> str:
    if title_prefix:
        return f"{title_prefix} | {title}"
    return title


def _plot_single_series_group(
    history: HistoryDict,
    keys: Sequence[str],
    title: str,
    ylabel: str,
    smooth: int = 1,
    title_prefix: str = "",
    mark_best_metric: Optional[str] = "median_center_error_mm",
    maximize_best_metric: bool = False,
    percent_keys: Optional[Sequence[str]] = None,
):
    """
    Internal helper for plotting a group of metrics from one history.
    """
    percent_keys = set(percent_keys or [])
    xs = get_epoch_numbers(history)

    fig = plt.figure(figsize=(10, 6))

    for key in keys:
        ys = extract_series(history, key)
        if key in percent_keys:
            ys = [100.0 * y if np.isfinite(y) else y for y in ys]
        ys = moving_average(ys, window=smooth)
        plt.plot(xs, ys, label=key)

    if mark_best_metric is not None:
        be = best_epoch_for_metric(history, mark_best_metric, maximize=maximize_best_metric)
        if be is not None:
            plt.axvline(be, linestyle="--", alpha=0.4, label=f"best epoch ({be})")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(_make_title(title, title_prefix))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_losses(
    history: HistoryDict,
    smooth: int = 1,
    title_prefix: str = "",
):
    """
    Plot total training / validation losses.

    Returns:
        matplotlib figure
    """
    return _plot_single_series_group(
        history=history,
        keys=["train_total_loss", "val_total_loss"],
        title="Total Loss",
        ylabel="Loss",
        smooth=smooth,
        title_prefix=title_prefix,
        mark_best_metric="median_center_error_mm",
        maximize_best_metric=False,
    )


def plot_component_losses(
    history: HistoryDict,
    smooth: int = 1,
    title_prefix: str = "",
):
    """
    Plot heat and size losses for train/val.

    Returns:
        matplotlib figure
    """
    return _plot_single_series_group(
        history=history,
        keys=["train_heat_loss", "val_heat_loss", "train_size_loss", "val_size_loss"],
        title="Heat / Size Loss",
        ylabel="Loss",
        smooth=smooth,
        title_prefix=title_prefix,
        mark_best_metric="median_center_error_mm",
        maximize_best_metric=False,
    )


def plot_center_error(
    history: HistoryDict,
    smooth: int = 1,
    title_prefix: str = "",
):
    """
    Plot center error metrics.

    Returns:
        matplotlib figure
    """
    return _plot_single_series_group(
        history=history,
        keys=["median_center_error_mm", "mean_center_error_mm"],
        title="Center Error",
        ylabel="mm",
        smooth=smooth,
        title_prefix=title_prefix,
        mark_best_metric="median_center_error_mm",
        maximize_best_metric=False,
    )


def plot_success_rate(
    history: HistoryDict,
    smooth: int = 1,
    title_prefix: str = "",
):
    """
    Plot success rate metric p_at_thresh in percent.

    Returns:
        matplotlib figure
    """
    return _plot_single_series_group(
        history=history,
        keys=["p_at_thresh"],
        title="Success Rate",
        ylabel="Percent (%)",
        smooth=smooth,
        title_prefix=title_prefix,
        mark_best_metric="median_center_error_mm",
        maximize_best_metric=False,
        percent_keys=["p_at_thresh"],
    )


def plot_mean_iou(
    history: HistoryDict,
    smooth: int = 1,
    title_prefix: str = "",
):
    """
    Plot mean IoU.

    Returns:
        matplotlib figure
    """
    return _plot_single_series_group(
        history=history,
        keys=["mean_iou"],
        title="Mean IoU",
        ylabel="IoU",
        smooth=smooth,
        title_prefix=title_prefix,
        mark_best_metric="median_center_error_mm",
        maximize_best_metric=False,
    )


def plot_all_history(
    history: HistoryDict,
    smooth: int = 1,
    title_prefix: str = "",
) -> List[Tuple[str, Any]]:
    """
    Plot the standard full set of training figures.

    Args:
        history: parsed history dict
        smooth: moving average window
        title_prefix: optional plot title prefix

    Returns:
        List of (name, fig) tuples.
    """
    figs = [
        ("loss_total", plot_losses(history, smooth=smooth, title_prefix=title_prefix)),
        ("loss_components", plot_component_losses(history, smooth=smooth, title_prefix=title_prefix)),
        ("center_error", plot_center_error(history, smooth=smooth, title_prefix=title_prefix)),
        ("success_rate", plot_success_rate(history, smooth=smooth, title_prefix=title_prefix)),
        ("mean_iou", plot_mean_iou(history, smooth=smooth, title_prefix=title_prefix)),
    ]
    return figs


def save_figures(
    figs: Sequence[Tuple[str, Any]],
    outdir: str | Path,
    dpi: int = 150,
) -> List[Path]:
    """
    Save figures to a directory.

    Args:
        figs: list of (name, fig) tuples
        outdir: output directory
        dpi: save dpi

    Returns:
        List of saved file paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for name, fig in figs:
        path = outdir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(path)
    return saved


def close_figures(figs: Sequence[Tuple[str, Any]]) -> None:
    """
    Close matplotlib figures created by this module.
    """
    for _, fig in figs:
        plt.close(fig)