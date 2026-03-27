#!/usr/bin/env python3
"""
scripts/build_cache.py

Build cached preprocessed samples for the localization project.

What this script does:
- reads the dataset index CSV
- runs the standard preprocessing pipeline once per sample
- saves each processed sample as a .pt file
- writes a cache_index.csv pointing to cached files

Typical use:
    python scripts/build_cache.py \
        --index-csv data/processed/localizer_index_cv5.csv \
        --outdir data/cache/localizer_sp2_clip_m150_350_sigma3_logmm \
        --size-target log_mm \
        --target-spacing 2.0 2.0 2.0 \
        --heat-sigma 3.0 \
        --ct-clip -150 350 \
        --pad-multiple 8 \
        --heatmap-method separable
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch

from localization.data.dataset import LocalizerDataset, SampleConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build cached preprocessed samples for localization.")

    ap.add_argument("--index-csv", type=Path, required=True, help="Input dataset index CSV.")
    ap.add_argument("--outdir", type=Path, required=True, help="Output cache directory.")

    ap.add_argument(
        "--size-target",
        type=str,
        default="log_mm",
        choices=["mm", "log_mm"],
        help="Target representation for bbox size.",
    )
    ap.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=[2.0, 2.0, 2.0],
        help="Target spacing (x y z) in mm.",
    )
    ap.add_argument(
        "--heat-sigma",
        type=float,
        default=3.0,
        help="Heatmap sigma in voxels on the resampled grid.",
    )
    ap.add_argument(
        "--ct-clip",
        type=float,
        nargs=2,
        default=[-150.0, 350.0],
        help="CT clipping range (min max).",
    )
    ap.add_argument(
        "--pad-multiple",
        type=int,
        default=8,
        help="Pad Z/Y/X to a multiple of this value.",
    )
    ap.add_argument(
        "--heatmap-method",
        type=str,
        default="separable",
        choices=["separable", "meshgrid"],
        help="Heatmap generation method.",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached .pt files.",
    )

    return ap.parse_args()


def _sample_cfg_from_args(args: argparse.Namespace) -> SampleConfig:
    return SampleConfig(
        size_target=str(args.size_target),
        target_spacing_xyz=(
            float(args.target_spacing[0]),
            float(args.target_spacing[1]),
            float(args.target_spacing[2]),
        ),
        heat_sigma_vox=float(args.heat_sigma),
        ct_clip=(float(args.ct_clip[0]), float(args.ct_clip[1])),
        pad_multiple=int(args.pad_multiple),
        heatmap_method=str(args.heatmap_method),
    )


def _safe_case_id(row: pd.Series, index: int) -> str:
    if "case_id" in row and pd.notna(row["case_id"]):
        return str(row["case_id"])
    return f"sample_{index:05d}"


def _tensor_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _tensor_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tensor_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_tensor_to_cpu(v) for v in obj)
    return obj


def build_cache(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_dir = outdir / "samples"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = _sample_cfg_from_args(args)

    src_df = pd.read_csv(args.index_csv)
    rows_out: List[Dict[str, Any]] = []

    print(f"Building cache from: {args.index_csv}")
    print(f"Output directory    : {outdir}")
    print(
        "Config              : "
        f"size_target={cfg.size_target}, "
        f"target_spacing_xyz={cfg.target_spacing_xyz}, "
        f"heat_sigma_vox={cfg.heat_sigma_vox}, "
        f"ct_clip={cfg.ct_clip}, "
        f"pad_multiple={cfg.pad_multiple}, "
        f"heatmap_method={cfg.heatmap_method}"
    )
    print(f"Total rows          : {len(src_df)}")

    for idx, row in src_df.iterrows():
        case_id = _safe_case_id(row, idx)
        cache_path = cache_dir / f"{case_id}.pt"

        if cache_path.exists() and not args.overwrite:
            print(f"[{idx + 1:04d}/{len(src_df):04d}] skip  {case_id} -> {cache_path.name}")
        else:
            # Use split-based loading only to access the same preprocessing pipeline.
            # We build a tiny one-row temporary dataset by writing this row to a temp CSV-like DataFrame slice.
            one_df = pd.DataFrame([row])

            # Reuse dataset logic by creating a temporary CSV on disk.
            tmp_csv = outdir / f".tmp_{case_id}.csv"
            one_df.to_csv(tmp_csv, index=False)

            try:
                ds = LocalizerDataset(index_csv=tmp_csv, split=row.get("split", None), cfg=cfg)
                x, y = ds[0]

                record = {
                    "x": _tensor_to_cpu(x),                  # (1, Z, Y, X)
                    "heat": _tensor_to_cpu(y["heat"]),       # (1, Z, Y, X)
                    "size": _tensor_to_cpu(y["size"]),       # (3,)
                    "center_mm": _tensor_to_cpu(y["center_mm"]),
                    "spacing": _tensor_to_cpu(y["spacing"]),
                    "origin": _tensor_to_cpu(y["origin"]),
                    "direction": _tensor_to_cpu(y["direction"]),
                    "case_id": y.get("case_id", case_id),
                    "pad_spec": y.get("pad_spec"),
                    "meta": {
                        "source_index_csv": str(args.index_csv),
                        "source_image": str(row["image"]) if "image" in row else None,
                        "source_meta": str(row["meta"]) if "meta" in row else None,
                        "split": str(row["split"]) if "split" in row and pd.notna(row["split"]) else None,
                        "fold": int(row["fold"]) if "fold" in row and pd.notna(row["fold"]) else None,
                        "cfg": {
                            "size_target": cfg.size_target,
                            "target_spacing_xyz": list(cfg.target_spacing_xyz),
                            "heat_sigma_vox": cfg.heat_sigma_vox,
                            "ct_clip": list(cfg.ct_clip),
                            "pad_multiple": cfg.pad_multiple,
                            "heatmap_method": cfg.heatmap_method,
                        },
                    },
                }

                torch.save(record, cache_path)
                print(f"[{idx + 1:04d}/{len(src_df):04d}] write {case_id} -> {cache_path.name}")

            finally:
                if tmp_csv.exists():
                    tmp_csv.unlink()

        out_row: Dict[str, Any] = dict(row)
        out_row["case_id"] = case_id
        out_row["cache_path"] = str(cache_path)
        rows_out.append(out_row)

    cache_index_csv = outdir / "cache_index.csv"
    pd.DataFrame(rows_out).to_csv(cache_index_csv, index=False)

    print("\nDone.")
    print(f"Cache index: {cache_index_csv}")
    print(f"Sample dir : {cache_dir}")


def main() -> int:
    args = parse_args()
    build_cache(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())