"""Deterministic layout of ``<drive_root>/GEMS_GEOLOGICAL_DATA_ATLAS_V1/`` (independent of every other namespace)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LAYOUT = {
    "00_raw_snapshot": ["geodawn", "faults", "dem_1m", "source_metadata", "official_template"],
    "01_inventory": ["manifests"],
    "02_geospatial": ["grids", "masks", "footprints", "dem_mosaic", "coverage"],
    "03_features": ["raw_bands", "magnetic", "gravity", "terrain_100m", "dem_multiscale", "structural", "cross_modal",
                    "feature_registry"],
    "04_eda": ["tables", "profiles", "geological_evidence", "candidate_lineaments", "reports"],
    "05_figures": ["overview", "geophysics", "terrain", "structures", "cross_modal", "fault_discovery"],
    "06_ml_ready": ["feature_stack", "coverage_masks", "normalization", "schemas"],
    "07_cache": ["download_state", "copy_state", "fft", "dem_windows", "aggregations", "manifests"],
    "08_reports": [],
}


@dataclass(frozen=True)
class AtlasPaths:
    drive_root: Path
    atlas_name: str
    scratch_root: Path

    @classmethod
    def from_config(cls, cfg) -> "AtlasPaths":
        return cls(Path(cfg["drive_root"]), cfg["atlas_root"], Path(cfg["scratch_root"]))

    @classmethod
    def from_root(cls, root: str | Path, scratch: str | Path = "/content/atlas_scratch") -> "AtlasPaths":
        root = Path(root)
        return cls(root.parent, root.name, Path(scratch))

    @property
    def root(self) -> Path:
        return self.drive_root / self.atlas_name

    def d(self, top: str, sub: str | None = None) -> Path:
        if top not in LAYOUT or (sub is not None and sub not in LAYOUT[top]):
            raise KeyError(f"{top}/{sub} is not part of the atlas layout")
        return self.root / top if sub is None else self.root / top / sub

    # frequently used locations
    @property
    def snapshot(self) -> Path:
        return self.d("00_raw_snapshot")

    @property
    def inventory(self) -> Path:
        return self.d("01_inventory")

    @property
    def registry_file(self) -> Path:
        return self.d("03_features", "feature_registry") / "feature_registry.parquet"

    @property
    def stage_dir(self) -> Path:
        return self.d("07_cache", "manifests")

    @property
    def reports(self) -> Path:
        return self.d("08_reports")

    def ensure(self) -> "AtlasPaths":
        for top, subs in LAYOUT.items():
            (self.root / top).mkdir(parents=True, exist_ok=True)
            for s in subs:
                (self.root / top / s).mkdir(parents=True, exist_ok=True)
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        return self

    def rel(self, p: Path) -> str:
        try:
            return Path(p).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(p)

    def abs(self, rel: str) -> Path:
        return self.root / rel
