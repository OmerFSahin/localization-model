#!/usr/bin/env python3
"""
scripts/check_dataset.py

Command-line sanity checks for the dataset index and referenced files.

Example:
    python scripts/check_dataset.py --index-csv data/processed/localizer_index.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from localization.data.sanity import run_full_sanity


def parse_args():
    ap = argparse.ArgumentParser(description="Run dataset sanity checks (paths, readability, bbox schema).")
    ap.add_argument(
        "--index-csv",
        type=Path,
        default=Path("data/processed/localizer_index.csv"),
        help="Path to index CSV (split,case_id,image,meta).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    summary = run_full_sanity(args.index_csv)

    # Exit with non-zero status if any problems were found
    problems = summary["missing_files"] + summary["unreadable_scans"] + summary["bad_meta"]
    if problems > 0:
        print("\n❌ Dataset sanity checks failed.")
        raise SystemExit(2)

    print("\n✅ Dataset sanity checks passed.")


if __name__ == "__main__":
    raise SystemExit(main())