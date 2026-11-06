"""Central feature registry (``03_features/feature_registry/feature_registry.parquet``) and the writer that saves
every feature as a single-band tiled GeoTIFF on the analysis grid together with its registry row.

Registry columns: feature_id, family, parent_source, unit, scale_m, operator, aggregation, valid_fraction,
nodata_policy, orientation_convention, normalization, output_path, fingerprint, status (+ meaning, stage).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cache import atomic_write_csv, atomic_write_geotiff, atomic_write_parquet
from .config import stable_hash

COLUMNS = ["feature_id", "family", "parent_source", "unit", "scale_m", "operator", "aggregation", "valid_fraction",
           "nodata_policy", "orientation_convention", "normalization", "output_path", "fingerprint", "status",
           "meaning", "stage"]
AXIAL = "axial strike azimuth a in [0,180) deg clockwise from north, stored as (cos 2a, sin 2a)"
FAMILY_DIR = {"raw": "raw_bands", "magnetic": "magnetic", "gravity": "gravity", "terrain": "terrain_100m",
              "dem": "dem_multiscale", "structural": "structural", "cross_modal": "cross_modal"}


class Registry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.df = pd.read_parquet(self.path) if self.path.exists() else pd.DataFrame(columns=COLUMNS)

    def upsert(self, rows: list[dict]):
        if not rows:
            return
        new = pd.DataFrame(rows).reindex(columns=COLUMNS)
        self.df = pd.concat([self.df[~self.df.feature_id.isin(new.feature_id)], new], ignore_index=True)

    def drop_stage(self, stage: str):
        self.df = self.df[self.df.stage != stage].reset_index(drop=True)

    def save(self):
        df = self.df.copy()
        df["scale_m"] = pd.to_numeric(df["scale_m"], errors="coerce")
        df["valid_fraction"] = pd.to_numeric(df["valid_fraction"], errors="coerce")
        atomic_write_parquet(self.path, df)
        atomic_write_csv(self.path.with_suffix(".csv"), df)

    def family(self, fam: str) -> pd.DataFrame:
        return self.df[self.df.family == fam]


class FeatureWriter:
    """Writes features of one stage; ``rows`` collects the registry entries (upserted by the stage at the end)."""

    def __init__(self, atlas_root: Path, grid, footprint: np.ndarray, stage: str, stage_fp: str):
        self.root, self.grid, self.foot = Path(atlas_root), grid, footprint
        self.stage, self.stage_fp = stage, stage_fp
        self.rows: list[dict] = []
        self.paths: list[Path] = []

    def path_for(self, family: str, feature_id: str, sub: str | None = None) -> Path:
        d = self.root / "03_features" / FAMILY_DIR[family]
        return (d / sub if sub else d) / f"{feature_id}.tif"

    def put(self, feature_id: str, arr: np.ndarray, *, family: str, parent: str, unit: str, scale_m=None,
            operator: str, aggregation: str = "none (100 m grid)", nodata: str = "NaN outside footprint / edge guard",
            orientation: str = "", normalization: str = "none (physical units)", meaning: str = "",
            status: str = "COMPLETE", sub: str | None = None, mask: np.ndarray | None = None) -> np.ndarray:
        a = np.asarray(arr, np.float32)
        a = np.where(self.foot if mask is None else mask, a, np.nan).astype(np.float32)
        p = self.path_for(family, feature_id, sub)
        atomic_write_geotiff(p, a, self.grid.profile(), [feature_id], tags={"unit": unit, "operator": operator})
        self.paths.append(p)
        vf = float(np.isfinite(a[self.foot]).mean()) if self.foot.any() else 0.0
        self.rows.append({"feature_id": feature_id, "family": family, "parent_source": parent, "unit": unit,
                          "scale_m": scale_m, "operator": operator, "aggregation": aggregation, "valid_fraction": vf,
                          "nodata_policy": nodata, "orientation_convention": orientation,
                          "normalization": normalization, "output_path": p.relative_to(self.root).as_posix(),
                          "fingerprint": stable_hash([self.stage_fp, feature_id]), "status": status,
                          "meaning": meaning, "stage": self.stage})
        return a

    def alias(self, feature_id: str, *, target_path: str, family: str, parent: str, unit: str, operator: str,
              meaning: str, status: str, scale_m=None):
        """Registry row for a feature that is NOT re-materialised because a delivered band already carries it."""
        self.rows.append({"feature_id": feature_id, "family": family, "parent_source": parent, "unit": unit,
                          "scale_m": scale_m, "operator": operator, "aggregation": "none (100 m grid)",
                          "valid_fraction": np.nan, "nodata_policy": "as delivered band", "orientation_convention": "",
                          "normalization": "none", "output_path": target_path,
                          "fingerprint": stable_hash([self.stage_fp, feature_id]), "status": status,
                          "meaning": meaning, "stage": self.stage})


def pct_rank(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] of the finite values inside ``mask`` (label-free normalisation); NaN elsewhere."""
    out = np.full(a.shape, np.nan, np.float32)
    m = mask & np.isfinite(a)
    v = a[m]
    if v.size:
        order = np.argsort(v, kind="stable")
        r = np.empty(v.size, np.float32)
        r[order] = (np.arange(v.size, dtype=np.float32) + 0.5) / v.size
        out[m] = r
    return out
