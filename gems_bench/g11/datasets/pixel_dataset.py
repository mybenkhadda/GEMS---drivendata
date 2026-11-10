"""Pixel samples (100 m) for gradient boosting: features x targets with split / block provenance."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..spatial_splits import CODES
from ..training.sampling import category_masks, mixture_sample


@dataclass
class GeoPixelDataset:
    X: np.ndarray                 # (n, f) float32
    y: np.ndarray                 # (n,) target (1 = mapped / buffered fault, 0 = unlabeled - NOT a confirmed negative)
    weight: np.ndarray            # (n,) sample weights (unlabeled down-weighted, PU-style)
    rows: np.ndarray
    cols: np.ndarray
    block: np.ndarray             # geographic block id (for grouped CV / bootstrap)
    split: str
    category: np.ndarray
    feature_names: list
    target_name: str
    provenance: dict = field(default_factory=dict)

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.X)

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["y"], df["w"], df["row"], df["col"], df["block"] = self.y, self.weight, self.rows, self.cols, self.block
        return df

    @classmethod
    def build(cls, store, feature_names, target: np.ndarray, target_name: str, split_code: np.ndarray, split: str,
              fault: np.ndarray, dist_m: np.ndarray, cfg: dict, hard_geo=None, hard_terrain=None, disagreement=None,
              n: int | None = None, seed: int | None = None) -> "GeoPixelDataset":
        region = split_code == CODES[split]
        cats = category_masks(region, fault, dist_m, cfg["labels"]["near_band_m"], hard_geo, hard_terrain,
                              disagreement)
        bpx = max(1, int(round(cfg["sampling"]["balance_block_km"] * 1000 / 100.0)))
        idx, code, info = mixture_sample(cats, cfg["sampling"]["mix"], int(n or cfg["sampling"]["pixel_samples_per_split"]),
                                         bpx, int(seed if seed is not None else cfg["reproducibility"]["seed"]))
        rows, cols = np.unravel_index(idx, split_code.shape)
        X = store.sample(feature_names, rows, cols) if len(idx) else np.zeros((0, len(feature_names)), np.float32)
        y = np.nan_to_num(target[rows, cols], nan=0.0).astype(np.float32)
        wu = float(cfg["models"]["xgb"]["pu_unlabeled_weight"])
        w = np.where(y > 0.5, 1.0, wu).astype(np.float32)
        nbc = -(-split_code.shape[1] // bpx)
        block = (rows // bpx) * nbc + cols // bpx
        prov = {"split": split, "target": target_name, "sampling": info, "pu_unlabeled_weight": wu,
                "label_semantics": "unlabeled = background, not verified negative"}
        return cls(X, y, w, rows, cols, block, split, code, list(feature_names), target_name, prov)
