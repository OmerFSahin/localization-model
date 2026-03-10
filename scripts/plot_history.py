#!/usr/bin/env python3
"""
scripts/plot_history.py

Plot training history from a history.json file written by localization.train.trainer.

Example:
    python scripts/plot_history.py --history outputs/run01/history.json

Save figures instead of showing:
    python scripts/plot_history.py \
        --history outputs/run01/history.json \
        --save-dir outputs/run01/plots

Compare multiple runs:
    python scripts/plot_history.py \
        --history outputs/run01/history.json outputs/run02/history.json \
        --labels unet resnet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="Plot training history for localization experiments.")
    ap.add_argument(
        "--history",
        type=Path,
        nargs="+",
        required=True,
        help="One or more history.json files.",
    )
    ap.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels for the runs. Must match number of history files.",
    )
    ap.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save PNG plots. If omitted, plots are shown interactively.",
    )
    ap.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help="Optional prefix added to plot titles.",
    )
    ap.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window size for smoothing (1 = no smoothing).",
    )
    return ap.parse_args()


def load_history(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) == 0:
        return values[:]

    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        chunk = [v for v in chunk if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def extract_series(history: Dict[str, Any], key: str) -> List[float]:
    epochs = history.get("epochs", [])
    vals: List[float] = []
    for rec in epochs:
        v = rec.get(key, float("nan"))
        try:
            vals.append(float(v))
        except Exception:
            vals.append(float("nan"))
    return vals


def get_epoch_numbers(history: Dict[str, Any]) -> List[int]:
    epochs = history.get("epochs", [])
    return [int(rec.get("epoch", i + 1)) for i, rec in enumerate(epochs)]


def best_epoch_for_metric(
    history: Dict[str, Any],
    metric_key: str,
    maximize: bool = False,
) -> Optional[int]:
    epochs = history.get("epochs", [])
    best_epoch = None
    best_value = None

    for rec in epochs:
        if metric_key not in rec:
            continue
        try:
            value = float(rec[metric_key])
        except Exception:
            continue

        if best_value is None:
            best_value = value
            best_epoch = int(rec.get("epoch", len(epochs)))
            continue

        is_better = (value > best_value) if maximize else (value < best_value)
        if is_better:
            best_value = value
            best_epoch = int(rec.get("epoch", len(epochs)))

    return best_epoch


def maybe_prefix(title_prefix: str, title: str) -> str:
    return f"{title_prefix} | {title}" if title_prefix else title


def plot_metric_group(
    histories: List[Dict[str, Any]],
    labels: List[str],
    keys: List[str],
    title: str,
    ylabel: str,
    smooth: int = 1,
    best_metric_key: Optional[str] = None,
    best_metric_maximize: bool = False,
):
    fig = plt.figure(figsize=(10, 6))

    for history, label in zip(histories, labels):
        xs = get_epoch_numbers(history)

        for key in keys:
            ys = extract_series(history, key)
            ys = moving_average(ys, smooth)
            plt.plot(xs, ys, label=f"{label} | {key}")

        if best_metric_key is not None:
            be = best_epoch_for_metric(history, best_metric_key, maximize=best_metric_maximize)
            if be is not None:
                plt.axvline(be, linestyle="--", alpha=0.4)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return fig


def main():
    args = parse_args()

    if args.labels is not None and len(args.labels) > 0 and len(args.labels) != len(args.history):
        raise ValueError("Number of --labels must match number of --history files.")

    histories = [load_history(p) for p in args.history]

    if args.labels is None or len(args.labels) == 0:
        labels = [p.parent.name if p.parent.name else p.stem for p in args.history]
    else:
        labels = args.labels

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: total loss
    fig1 = plot_metric_group(
        histories=histories,
        labels=labels,
        keys=["train_total_loss", "val_total_loss"],
        title=maybe_prefix(args.title_prefix, "Total Loss"),
        ylabel="Loss",
        smooth=args.smooth,
        best_metric_key="median_center_error_mm",
        best_metric_maximize=False,
    )

    # Figure 2: component losses
    fig2 = plot_metric_group(
        histories=histories,
        labels=labels,
        keys=["train_heat_loss", "val_heat_loss", "train_size_loss", "val_size_loss"],
        title=maybe_prefix(args.title_prefix, "Heat / Size Loss"),
        ylabel="Loss",
        smooth=args.smooth,
        best_metric_key="median_center_error_mm",
        best_metric_maximize=False,
    )

    # Figure 3: center errors
    fig3 = plot_metric_group(
        histories=histories,
        labels=labels,
        keys=["median_center_error_mm", "mean_center_error_mm"],
        title=maybe_prefix(args.title_prefix, "Center Error"),
        ylabel="mm",
        smooth=args.smooth,
        best_metric_key="median_center_error_mm",
        best_metric_maximize=False,
    )

    # Figure 4: success rate
    fig4 = plt.figure(figsize=(10, 6))
    for history, label in zip(histories, labels):
        xs = get_epoch_numbers(history)
        ys = extract_series(history, "p_at_thresh")
        ys = [100.0 * y for y in ys]
        ys = moving_average(ys, args.smooth)
        plt.plot(xs, ys, label=f"{label} | p_at_thresh")

        be = best_epoch_for_metric(history, "median_center_error_mm", maximize=False)
        if be is not None:
            plt.axvline(be, linestyle="--", alpha=0.4)

    plt.xlabel("Epoch")
    plt.ylabel("Percent (%)")
    plt.title(maybe_prefix(args.title_prefix, "Success Rate"))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Figure 5: IoU
    fig5 = plot_metric_group(
        histories=histories,
        labels=labels,
        keys=["mean_iou"],
        title=maybe_prefix(args.title_prefix, "Mean IoU"),
        ylabel="IoU",
        smooth=args.smooth,
        best_metric_key="median_center_error_mm",
        best_metric_maximize=False,
    )

    figs = [
        ("loss_total.png", fig1),
        ("loss_components.png", fig2),
        ("center_error.png", fig3),
        ("success_rate.png", fig4),
        ("mean_iou.png", fig5),
    ]

    if args.save_dir is not None:
        for name, fig in figs:
            out = args.save_dir / name
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved: {out}")
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    raise SystemExit(main())