"""
localization.data.dataset

PyTorch Dataset for the localization model.

It reads samples from an index CSV with columns:
- split: "train" / "val" / "test"
- case_id: unique identifier (optional but recommended)
- image: path to scan file
- meta:  path to meta.json containing bbox_mm

meta.json expected schema (minimum):
{
  "bbox_mm": [xmin, ymin, zmin, xmax, ymax, zmax]
}

Conventions:
- SimpleITK world coordinates are (x,y,z) in mm
- NumPy arrays from sitk.GetArrayFromImage are (Z,Y,X)
- Heatmap targets are (Z,Y,X)
- Returned image tensor is (1,Z,Y,X)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import json
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset

from localization.data.io import read_sitk
from localization.transforms.resample import sitk_resample_iso
from localization.geometry.coords import world_to_vox
from localization.targets.heatmap import make_heatmap
from localization.data.preprocess import normalize_ct, pad_spec_for_shape, apply_pad


PathLike = Union[str, Path]


@dataclass(frozen=True)
class SampleConfig:
    """
    Dataset configuration.

    Keep all dataset knobs here so they can be set from configs later.
    """
    size_target: str = "mm"   # "mm" or "log_mm"
    target_spacing_xyz: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    heat_sigma_vox: float = 3.0
    ct_clip: Tuple[float, float] = (-150.0, 350.0)
    pad_multiple: int = 8
    heatmap_method: str = "separable"  # "separable" or "meshgrid"


class LocalizerDataset(Dataset):
    """
    Returns:
        x: torch.FloatTensor (1, Z, Y, X)
        y: dict with:
            - heat: (1, Z, Y, X) float
            - size: (3,) float (mm) [wx, wy, wz]
            - center_mm: (3,) float (x,y,z) in mm
            - spacing: (3,) float (x,y,z) in mm  (for metrics)
            - origin: (3,) float (x,y,z)
            - direction: (3,3) float
            - case_id: str (if present)
            - pad_spec: pad specification (for debugging/unpadding)
    """

    def __init__(
        self,
        index_csv: PathLike,
        split: str,
        cfg: Optional[SampleConfig] = None,
    ):
        self.index_csv = Path(index_csv)
        self.split = str(split)
        self.cfg = cfg or SampleConfig()

        df = pd.read_csv(self.index_csv)

        # Filter by split
        if "split" not in df.columns:
            raise ValueError(f"Index CSV must contain a 'split' column: {self.index_csv}")

        df = df[df["split"] == self.split].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError(f"No rows for split='{self.split}' in {self.index_csv}")

        # Basic required columns
        for col in ("image", "meta"):
            if col not in df.columns:
                raise ValueError(f"Index CSV missing required column '{col}': {self.index_csv}")

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def _load_bbox_mm(self, meta_path: Path) -> np.ndarray:
        """
        Read bbox_mm from a meta.json file.

        Returns:
            bbox_mm: np.float32 shape (6,) [xmin,ymin,zmin,xmax,ymax,zmax]
        """
        with open(meta_path, "r") as f:
            obj = json.load(f)

        if "bbox_mm" not in obj:
            raise KeyError(f"'bbox_mm' not found in meta file: {meta_path}")

        bbox = np.array(obj["bbox_mm"], dtype=np.float32).reshape(-1)
        if bbox.size != 6:
            raise ValueError(f"bbox_mm must have 6 numbers, got {bbox.size} in {meta_path}")

        xmin, ymin, zmin, xmax, ymax, zmax = [float(v) for v in bbox]

        # Ensure min/max are ordered (robust to swapped values)
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        zmin, zmax = min(zmin, zmax), max(zmin, zmax)

        return np.array([xmin, ymin, zmin, xmax, ymax, zmax], dtype=np.float32)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]

        img_path = Path(row["image"])
        meta_path = Path(row["meta"])

        # 1) Read image (SimpleITK) and resample to target spacing
        img0 = read_sitk(img_path)
        img = sitk_resample_iso(img0, out_spacing=self.cfg.target_spacing_xyz, interp=sitk.sitkLinear)

        # 2) Convert to NumPy (Z,Y,X) and normalize intensities
        arr_zyx = sitk.GetArrayFromImage(img).astype(np.float32)
        arr_zyx = normalize_ct(arr_zyx, clip=self.cfg.ct_clip)

        # 3) Load bbox in mm -> center and size in mm
        bbox_mm = self._load_bbox_mm(meta_path)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox_mm

        center_mm = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2], dtype=np.float32)
        size_mm = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float32)  # (wx,wy,wz)

        if self.cfg.size_target == "mm":
            size_target = size_mm
        elif self.cfg.size_target == "log_mm":
            size_target = np.log(np.maximum(size_mm, 1e-6)).astype(np.float32)
        else:
            raise ValueError(f"Unknown size_target: {self.cfg.size_target}")

        # 4) Convert center from world(mm) to voxel(x,y,z) on the resampled grid
        center_vox_xyz = world_to_vox(center_mm[None, :], img)[0]  # float voxel coords (x,y,z)

        # 5) Generate heatmap target in (Z,Y,X)
        heat_zyx = make_heatmap(
            shape_zyx=arr_zyx.shape,
            center_vox_xyz=center_vox_xyz,
            sigma_vox=self.cfg.heat_sigma_vox,
            method=self.cfg.heatmap_method,
        ).astype(np.float32)

        # 6) Pad BOTH image and heatmap with the SAME pad spec
        pad_spec = pad_spec_for_shape(arr_zyx.shape, k=self.cfg.pad_multiple)
        arr_zyx = apply_pad(arr_zyx, pad_spec, mode="constant", value=0.0)
        heat_zyx = apply_pad(heat_zyx, pad_spec, mode="constant", value=0.0)

        # 7) Convert to tensors (channels first)
        x = torch.from_numpy(arr_zyx[None])         # (1,Z,Y,X)
        y_heat = torch.from_numpy(heat_zyx[None])   # (1,Z,Y,X)
        y_size = torch.from_numpy(size_target)      # (3,)

        # Extra geometry metadata for validation metrics (mm-level errors)
        spacing_xyz = np.array(img.GetSpacing(), dtype=np.float32)              # (x,y,z)
        origin_xyz = np.array(img.GetOrigin(), dtype=np.float32)                # (x,y,z)
        direction = np.array(img.GetDirection(), dtype=np.float32).reshape(3, 3)

        case_id = None
        if "case_id" in self.df.columns:
            case_id = str(row["case_id"])

        return x, {
            "heat": y_heat,
            "size": y_size,
            "center_mm": torch.from_numpy(center_mm),
            "spacing": torch.from_numpy(spacing_xyz),
            "origin": torch.from_numpy(origin_xyz),
            "direction": torch.from_numpy(direction),
            "case_id": case_id,
            "pad_spec": pad_spec,
        }