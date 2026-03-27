from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]


class CachedLocalizerDataset(Dataset):
    """
    Dataset that reads preprocessed cached .pt samples.

    Expected cache index columns:
    - cache_path
    - split (optional for split mode)
    - fold  (optional for CV mode)
    """

    def __init__(
        self,
        cache_index_csv: PathLike,
        split: Optional[str] = None,
        cv_fold: Optional[int] = None,
        cv_mode: Optional[str] = None,   # "train" or "val"
        fold_col: str = "fold",
    ):
        self.cache_index_csv = Path(cache_index_csv)
        df = pd.read_csv(self.cache_index_csv)

        if "cache_path" not in df.columns:
            raise ValueError(f"Cache index missing required column 'cache_path': {self.cache_index_csv}")

        if cv_fold is not None:
            if cv_mode not in {"train", "val"}:
                raise ValueError("When cv_fold is used, cv_mode must be 'train' or 'val'")
            if fold_col not in df.columns:
                raise ValueError(f"Cache index must contain fold column '{fold_col}'")

            self.split = cv_mode
            self.cv_fold = int(cv_fold)
            self.cv_mode = str(cv_mode)
            self.fold_col = str(fold_col)

            if self.cv_mode == "train":
                df = df[df[fold_col] != self.cv_fold].reset_index(drop=True)
            else:
                df = df[df[fold_col] == self.cv_fold].reset_index(drop=True)
        else:
            if split is None:
                raise ValueError("Either split must be provided, or cv_fold + cv_mode must be provided")
            if "split" not in df.columns:
                raise ValueError(f"Cache index must contain 'split' column: {self.cache_index_csv}")

            self.split = str(split)
            df = df[df["split"] == self.split].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError("No rows found for requested split/CV selection")

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        cache_path = Path(row["cache_path"])

        obj = torch.load(cache_path, map_location="cpu")

        x = obj["x"]
        y = {
            "heat": obj["heat"],
            "size": obj["size"],
            "center_mm": obj["center_mm"],
            "spacing": obj["spacing"],
            "origin": obj["origin"],
            "direction": obj["direction"],
            "case_id": obj.get("case_id"),
            "pad_spec": obj.get("pad_spec"),
        }
        return x, y