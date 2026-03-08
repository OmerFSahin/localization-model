"""
localization.data.sanity

Sanity checks for the localization dataset index and files.

These utilities verify that:
- index CSV exists and has required columns
- referenced files exist
- scans are readable with SimpleITK
- meta.json contains valid bbox_mm

Running these checks before training prevents runtime crashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Dict, List
import json
import pandas as pd
import SimpleITK as sitk


PathLike = Union[str, Path]


REQUIRED_COLUMNS = ["split", "case_id", "image", "meta"]


# -------------------------------------------------------
# CSV structure check
# -------------------------------------------------------
def check_index_columns(index_csv: PathLike) -> None:
    """
    Ensure the index CSV has the required columns.
    """
    df = pd.read_csv(index_csv)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Index CSV missing columns: {missing}")

    print(f"Index CSV columns OK: {list(df.columns)}")


# -------------------------------------------------------
# File existence check
# -------------------------------------------------------
def check_file_paths(index_csv: PathLike) -> List[int]:
    """
    Check whether image and meta files exist.

    Returns:
        list of row indices with missing files
    """
    df = pd.read_csv(index_csv)
    bad_rows: List[int] = []

    for i, row in df.iterrows():
        img = Path(row["image"])
        meta = Path(row["meta"])

        if not img.exists() or not meta.exists():
            bad_rows.append(i)

    if bad_rows:
        print(f"Missing files in {len(bad_rows)} rows")
    else:
        print("All file paths exist")

    return bad_rows


# -------------------------------------------------------
# Scan readability check
# -------------------------------------------------------
def check_scan_readable(index_csv: PathLike) -> List[int]:
    """
    Verify that scan files can be opened by SimpleITK.
    """
    df = pd.read_csv(index_csv)
    bad_rows: List[int] = []

    for i, row in df.iterrows():
        img_path = Path(row["image"])

        try:
            _ = sitk.ReadImage(str(img_path))
        except Exception:
            bad_rows.append(i)

    if bad_rows:
        print(f"{len(bad_rows)} scans are not readable by SimpleITK")
    else:
        print("All scans readable")

    return bad_rows


# -------------------------------------------------------
# meta.json schema check
# -------------------------------------------------------
def check_meta_bbox(index_csv: PathLike) -> List[int]:
    """
    Ensure meta.json contains valid bbox_mm entries.
    """
    df = pd.read_csv(index_csv)
    bad_rows: List[int] = []

    for i, row in df.iterrows():
        meta_path = Path(row["meta"])

        try:
            with open(meta_path) as f:
                data = json.load(f)

            bbox = data.get("bbox_mm", None)
            if bbox is None or len(bbox) != 6:
                bad_rows.append(i)

        except Exception:
            bad_rows.append(i)

    if bad_rows:
        print(f"{len(bad_rows)} meta.json files missing or invalid bbox_mm")
    else:
        print("All meta.json files contain valid bbox_mm")

    return bad_rows


# -------------------------------------------------------
# Full sanity check
# -------------------------------------------------------
def run_full_sanity(index_csv: PathLike) -> Dict[str, int]:
    """
    Run all dataset checks and return summary stats.
    """
    index_csv = Path(index_csv)

    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    print("Running dataset sanity checks...\n")

    check_index_columns(index_csv)

    missing_files = check_file_paths(index_csv)
    unreadable = check_scan_readable(index_csv)
    bad_meta = check_meta_bbox(index_csv)

    summary = {
        "missing_files": len(missing_files),
        "unreadable_scans": len(unreadable),
        "bad_meta": len(bad_meta),
    }

    print("\nSanity summary:", summary)

    return summary