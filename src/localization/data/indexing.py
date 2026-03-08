"""
localization.data.indexing

Index building utilities for the localization dataset.

Given an OUTPUT_ROOT directory containing subfolders per case, we:
- locate a scan file (nrrd/nhdr/nii etc.)
- verify meta.json exists
- optionally verify the scan file is readable by SimpleITK
- create train/val/test splits
- write a CSV index with columns:
    split, case_id, image, meta

This CSV becomes the single source of truth for Dataset/Dataloader code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import csv
import random

import SimpleITK as sitk


PathLike = Union[str, Path]


# -----------------------------
# Configuration container
# -----------------------------
@dataclass(frozen=True)
class IndexConfig:
    """
    Settings for building the index.
    """
    test_frac: float = 0.15
    val_frac: float = 0.15
    seed: int = 0
    require_readable: bool = True  # run sitk.ReadImage check


# -----------------------------
# Scan discovery + readability
# -----------------------------
DEFAULT_SCAN_PATTERNS = (
    "*(SCAN).nrrd",
    "*(SCAN).nrrd.gz",
    "*.nrrd",
    "*.nrrd.gz",
    "*.nhdr",
    "*.nii",
    "*.nii.gz",
)


def find_scan_file(case_dir: PathLike, patterns: Optional[Iterable[str]] = None) -> Optional[Path]:
    """
    Find a likely scan file inside a case directory.

    Returns:
        Path if found, else None
    """
    cdir = Path(case_dir)
    pats = tuple(patterns) if patterns is not None else DEFAULT_SCAN_PATTERNS

    candidates: List[Path] = []
    for pat in pats:
        candidates.extend(cdir.glob(pat))

    candidates = sorted(candidates)
    return candidates[0] if candidates else None


def is_readable_sitk(path: PathLike) -> bool:
    """
    Check if a file can be read by SimpleITK.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        _ = sitk.ReadImage(str(p))
        return True
    except Exception:
        return False


# -----------------------------
# Collect usable cases
# -----------------------------
def collect_cases(
    output_root: PathLike,
    require_readable: bool = True,
    meta_name: str = "meta.json",
    patterns: Optional[Iterable[str]] = None,
) -> List[Tuple[str, Path, Path]]:
    """
    Scan output_root and collect usable (case_id, scan_path, meta_path) tuples.

    A case is usable if:
    - it is a directory
    - meta.json exists
    - a scan file exists
    - optionally scan file is readable by SimpleITK

    Returns:
        list of (case_id, scan_path, meta_path)
    """
    root = Path(output_root)
    if not root.exists():
        raise FileNotFoundError(f"output_root does not exist: {root}")

    pairs: List[Tuple[str, Path, Path]] = []

    for case_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        meta_path = case_dir / meta_name
        scan_path = find_scan_file(case_dir, patterns=patterns)

        if scan_path is None:
            continue
        if not meta_path.exists():
            continue
        if require_readable and not is_readable_sitk(scan_path):
            continue

        pairs.append((case_dir.name, scan_path, meta_path))

    return pairs


# -----------------------------
# Split logic
# -----------------------------
def split_cases(
    cases: Sequence[Tuple[str, Path, Path]],
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[List[Tuple[str, Path, Path]], List[Tuple[str, Path, Path]], List[Tuple[str, Path, Path]]]:
    """
    Split collected cases into train/val/test.

    Ensures that tiny datasets still have a non-empty train split when possible.

    Returns:
        (train, val, test)
    """
    cases = list(cases)
    n = len(cases)
    if n == 0:
        raise RuntimeError("No usable cases found. Check folder structure and files.")

    rng = random.Random(seed)
    rng.shuffle(cases)

    n_test = max(1 if n >= 3 else 0, int(test_frac * n))
    n_val = max(1 if n >= 4 else 0, int(val_frac * n))

    test = cases[:n_test]
    val = cases[n_test:n_test + n_val]
    train = cases[n_test + n_val:]

    # Ensure train isn't empty (for tiny datasets)
    if len(train) == 0 and len(val) > 0:
        train.append(val.pop())
    if len(train) == 0 and len(test) > 0:
        train.append(test.pop())

    return train, val, test


# -----------------------------
# Write CSV
# -----------------------------
def write_index_csv(
    index_csv: PathLike,
    train: Sequence[Tuple[str, Path, Path]],
    val: Sequence[Tuple[str, Path, Path]],
    test: Sequence[Tuple[str, Path, Path]],
) -> None:
    """
    Write the index CSV with columns: split, case_id, image, meta
    """
    out = Path(index_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "case_id", "image", "meta"])

        for split_name, grp in (("train", train), ("val", val), ("test", test)):
            for case_id, img_path, meta_path in grp:
                w.writerow([split_name, case_id, str(img_path), str(meta_path)])


# -----------------------------
# High-level convenience
# -----------------------------
def build_index(
    output_root: PathLike,
    index_csv: PathLike,
    cfg: Optional[IndexConfig] = None,
    meta_name: str = "meta.json",
    patterns: Optional[Iterable[str]] = None,
) -> dict:
    """
    Full pipeline:
    - collect cases
    - split cases
    - write CSV

    Returns:
        summary dict with counts
    """
    cfg = cfg or IndexConfig()

    cases = collect_cases(
        output_root=output_root,
        require_readable=cfg.require_readable,
        meta_name=meta_name,
        patterns=patterns,
    )

    train, val, test = split_cases(
        cases,
        test_frac=cfg.test_frac,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
    )

    write_index_csv(index_csv, train, val, test)

    return {
        "total": len(cases),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "index_csv": str(index_csv),
    }