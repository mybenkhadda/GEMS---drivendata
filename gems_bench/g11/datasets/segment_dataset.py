"""Candidate-segment table for the ranker: features, label (overlap with mapped faults), group, split."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..spatial_splits import CODES

INV = {v: k for k, v in CODES.items()}
NON_FEATURES = {"segment_id", "label", "overlap_fraction", "group", "split", "geometry", "dem_tiles", "dem_error",
                "source_model", "generator"}


@dataclass
class FaultSegmentDataset:
    table: pd.DataFrame
    feature_names: list
    provenance: dict = field(default_factory=dict)

    @classmethod
    def build(cls, features: pd.DataFrame, gdf, overlap: pd.DataFrame, split_code: np.ndarray, groups: np.ndarray,
              grid, label_threshold: float = 0.5) -> "FaultSegmentDataset":
        """Each segment takes the split / group of the pixel under its midpoint; segments whose vertices fall in
        more than one split are dropped (they would straddle the embargo)."""
        from ..grid import inside, world_to_pixel

        rows = []
        for i, g in enumerate(gdf.geometry):
            pts = np.asarray([g.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, 9)])
            r, c = world_to_pixel(pts[:, 0], pts[:, 1], grid)
            ok = inside(r, c, grid)
            codes = set(np.unique(split_code[r[ok], c[ok]])) - {0, CODES["embargo"]}
            mid = len(pts) // 2
            rows.append({"segment_id": i, "split": INV[codes.pop()] if len(codes) == 1 else "mixed",
                         "group": int(groups[r[mid], c[mid]]) if ok[mid] else -1})
        meta = pd.DataFrame(rows)
        t = pd.concat([features.reset_index(drop=True), meta], axis=1)
        t["overlap_fraction"] = overlap.overlap_fraction.to_numpy()
        t["label"] = (t.overlap_fraction >= label_threshold).astype(int)
        t = t[t.split != "mixed"].reset_index(drop=True)
        names = [c for c in t.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(t[c])]
        return cls(t, names, {"label": f"overlap_fraction >= {label_threshold}", "dropped_mixed": int(
            (meta.split == "mixed").sum())})

    def split(self, name: str) -> pd.DataFrame:
        return self.table[self.table.split == name]
