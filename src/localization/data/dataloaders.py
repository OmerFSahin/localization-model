"""
localization.data.dataloaders

DataLoader builders for the localization project.

Why this module exists:
- Avoid repeating DataLoader boilerplate across scripts
- Keep dataset config and loader config in one place
- Make it easy to switch between notebook-friendly and speed-friendly settings

Usage pattern:
    from localization.data.dataloaders import build_loaders
    train_ds, val_ds, train_dl, val_dl = build_loaders(index_csv, cfg=sample_cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import torch
from torch.utils.data import DataLoader

from localization.data.dataset import LocalizerDataset, SampleConfig


PathLike = Union[str, Path]


@dataclass(frozen=True)
class LoaderConfig:
    """
    DataLoader configuration.

    Notes:
    - For Jupyter/Windows stability, num_workers=0 is safest.
    - For speed on Linux, num_workers=2..8 is typical.
    """
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None  # only used if num_workers > 0
    drop_last: bool = False


def _to_loader_kwargs(cfg: LoaderConfig, shuffle: bool) -> Dict[str, Any]:
    """
    Convert LoaderConfig to DataLoader kwargs safely.

    PyTorch rules:
    - persistent_workers and prefetch_factor require num_workers > 0
    """
    kwargs: Dict[str, Any] = dict(
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )

    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        if cfg.prefetch_factor is not None:
            kwargs["prefetch_factor"] = cfg.prefetch_factor

    return kwargs


def build_loaders(
    index_csv: PathLike,
    sample_cfg: Optional[SampleConfig] = None,
    train_loader_cfg: Optional[LoaderConfig] = None,
    val_loader_cfg: Optional[LoaderConfig] = None,
    cv_fold: Optional[int] = None,
    fold_col: str = "fold",
) -> Tuple[LocalizerDataset, LocalizerDataset, DataLoader, DataLoader]:
    """
    Build train/val datasets and dataloaders.

    Args:
        index_csv: path to the index CSV
        sample_cfg: SampleConfig for dataset preprocessing/targets
        train_loader_cfg: DataLoader config for training loader
        val_loader_cfg: DataLoader config for validation loader
        cv_fold: if not None, use cross-validation mode with the given validation fold
        fold_col: fold column name in the CSV

    Returns:
        (train_ds, val_ds, train_dl, val_dl)
    """
    sample_cfg = sample_cfg or SampleConfig()
    train_loader_cfg = train_loader_cfg or LoaderConfig()
    val_loader_cfg = val_loader_cfg or LoaderConfig()

    if cv_fold is None:
        train_ds = LocalizerDataset(index_csv=index_csv, split="train", cfg=sample_cfg)
        val_ds = LocalizerDataset(index_csv=index_csv, split="val", cfg=sample_cfg)
    else:
        train_ds = LocalizerDataset(
            index_csv=index_csv,
            cfg=sample_cfg,
            cv_fold=int(cv_fold),
            cv_mode="train",
            fold_col=fold_col,
        )
        val_ds = LocalizerDataset(
            index_csv=index_csv,
            cfg=sample_cfg,
            cv_fold=int(cv_fold),
            cv_mode="val",
            fold_col=fold_col,
        )

    train_dl = DataLoader(train_ds, **_to_loader_kwargs(train_loader_cfg, shuffle=True))
    val_dl = DataLoader(val_ds, **_to_loader_kwargs(val_loader_cfg, shuffle=False))

    return train_ds, val_ds, train_dl, val_dl


def build_test_loader(
    index_csv: PathLike,
    sample_cfg: Optional[SampleConfig] = None,
    test_loader_cfg: Optional[LoaderConfig] = None,
) -> Tuple[LocalizerDataset, DataLoader]:
    """
    Build test dataset and loader.

    Returns:
        (test_ds, test_dl)
    """
    sample_cfg = sample_cfg or SampleConfig()
    test_loader_cfg = test_loader_cfg or LoaderConfig()

    test_ds = LocalizerDataset(index_csv=index_csv, split="test", cfg=sample_cfg)
    test_dl = DataLoader(test_ds, **_to_loader_kwargs(test_loader_cfg, shuffle=False))
    return test_ds, test_dl