"""Reload API - read every atlas product without recomputation.

    atlas = GeologicalAtlas.open("/content/drive/MyDrive/GEMS_2/GEMS_GEOLOGICAL_DATA_ATLAS_V1")
    features = atlas.load_features(families=["magnetic", "gravity", "terrain", "dem"])   # FeatureSet
    coverage = atlas.load_coverage()                                                    # dict of masks
    lineaments = atlas.load_lineaments()                                                # GeoDataFrame
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import read_geotiff, read_json
from .geospatial import Grid


@dataclass
class FeatureSet:
    names: list[str]
    arrays: dict
    meta: pd.DataFrame
    grid: Grid

    def __getitem__(self, k):
        return self.arrays[k]

    def __len__(self):
        return len(self.names)

    def stack(self, names: list[str] | None = None) -> np.ndarray:
        return np.stack([self.arrays[n] for n in (names or self.names)])

    def table(self, mask: np.ndarray, names: list[str] | None = None) -> pd.DataFrame:
        """Pixel table (rows of ``mask``) - use a sampled mask for large areas."""
        rr, cc = np.nonzero(mask)
        d = {"row": rr, "col": cc}
        d.update({n: self.arrays[n][rr, cc] for n in (names or self.names)})
        return pd.DataFrame(d)


class GeologicalAtlas:
    def __init__(self, root: Path):
        self.root = Path(root)
        g = read_json(self.root / "02_geospatial" / "grids" / "analysis_grid.json")
        if g is None:
            raise FileNotFoundError(f"{self.root}: no 02_geospatial/grids/analysis_grid.json - run the geospatial stage")
        self.grid = Grid.from_dict(g)
        reg = self.root / "03_features" / "feature_registry" / "feature_registry.parquet"
        self.registry = pd.read_parquet(reg) if reg.exists() else pd.DataFrame()

    @classmethod
    def open(cls, root: str | Path) -> "GeologicalAtlas":
        return cls(Path(root))

    def families(self) -> list[str]:
        return sorted(self.registry.family.unique()) if len(self.registry) else []

    def load_feature(self, feature_id: str) -> np.ndarray:
        row = self.registry[self.registry.feature_id == feature_id]
        if row.empty:
            raise KeyError(f"{feature_id} not in the registry")
        p = self.root / row.output_path.iloc[0]
        if not p.exists():
            raise FileNotFoundError(f"{feature_id}: {p} missing (status {row.status.iloc[0]}) - re-run its stage")
        return read_geotiff(p, 1)

    def load_features(self, families=None, feature_ids=None, include_aliases: bool = True) -> FeatureSet:
        r = self.registry
        if families is not None:
            r = r[r.family.isin(families)]
        if feature_ids is not None:
            r = r[r.feature_id.isin(feature_ids)]
        ok = ["COMPLETE"] + (["alias_of_delivered"] if include_aliases else [])
        r = r[r.status.isin(ok)]
        arrays = {fid: self.load_feature(fid) for fid in r.feature_id}
        return FeatureSet(list(arrays), arrays, r.reset_index(drop=True), self.grid)

    def load_coverage(self, mode: str | None = None) -> dict:
        m = self.root / "02_geospatial" / "masks"
        c = self.root / "02_geospatial" / "coverage"
        out = {}
        for n in ("valid_numerical", "fault_labels", "label_footprint", "template_valid", "multimodal_coverage"):
            p = m / f"{n}.tif"
            if p.exists():
                out[n] = read_geotiff(p, 1)
        for n in ("dem_coverage_catalog_tiles", "dem_overlap_count", "dem_project_id", "dem_seam_mask"):
            p = c / f"{n}.tif"
            if p.exists():
                out[n] = read_geotiff(p, 1)
        modes = [mode] if mode else ["FAST_10M", "DETAILED_2M"]
        for md in modes:
            p = c / f"dem_coverage_{md}.tif"
            if p.exists():
                out[f"dem_coverage_{md}"] = read_geotiff(p, 1)
        return out

    def load_lineaments(self):
        import geopandas as gpd

        p = self.root / "04_eda" / "candidate_lineaments" / "candidate_lineaments.parquet"
        if not p.exists():
            raise FileNotFoundError("no candidate lineaments - run the EDA lineament stage")
        return gpd.read_parquet(p)

    def load_table(self, name: str) -> pd.DataFrame:
        for d in ("04_eda/tables", "04_eda/profiles", "04_eda/geological_evidence", "01_inventory"):
            for ext in (".parquet", ".csv"):
                p = self.root / d / f"{name}{ext}"
                if p.exists():
                    return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
        raise FileNotFoundError(name)

    def ml_ready(self) -> dict:
        s = read_json(self.root / "06_ml_ready" / "schemas" / "feature_stack_schema.json")
        if s is None:
            raise FileNotFoundError("06_ml_ready not built - run section 14")
        return s
