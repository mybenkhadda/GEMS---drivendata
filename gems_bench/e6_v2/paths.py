"""The single Drive namespace ``<drive_root>/<namespace>/`` and every folder the pipeline writes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TOP_DIRS = ("manifests", "cache", "datasets", "checkpoints", "predictions", "metrics", "figures", "submissions",
            "reports")
DEM_DIRS = ("source_inventory", "mosaic", "dem_2m", "windows", "features_2m", "features_100m", "coverage_100m",
            "manifests")


@dataclass(frozen=True)
class E6V2Paths:
    drive_root: Path
    namespace: str
    scratch_root: Path

    @classmethod
    def from_config(cls, cfg: dict) -> "E6V2Paths":
        p = cfg["paths"]
        return cls(Path(p["drive_root"]), p["namespace"], Path(p["scratch_root"]))

    # ---- namespace ------------------------------------------------------
    @property
    def root(self) -> Path:
        return self.drive_root / self.namespace

    def __getattr__(self, name: str) -> Path:
        if name in TOP_DIRS:
            return self.root / name
        raise AttributeError(name)

    @property
    def terrain(self) -> Path:
        return self.root / "cache" / "terrain"

    @property
    def potential(self) -> Path:
        return self.root / "cache" / "potential_fields"

    @property
    def base(self) -> Path:
        return self.root / "cache" / "base"

    @property
    def dem(self) -> Path:
        return self.root / "cache" / "dem"

    def dem_dir(self, name: str) -> Path:
        if name not in DEM_DIRS:
            raise KeyError(name)
        return self.dem / name

    @property
    def splits(self) -> Path:
        return self.root / "datasets" / "splits"

    def unit_dir(self, unit: str) -> Path:
        return self.root / "checkpoints" / unit

    def ensure(self) -> "E6V2Paths":
        for d in TOP_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        for d in (self.terrain, self.potential, self.base, self.splits):
            d.mkdir(parents=True, exist_ok=True)
        for d in DEM_DIRS:
            (self.dem / d).mkdir(parents=True, exist_ok=True)
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        return self

    def rel(self, path: Path) -> str:
        """Path relative to the namespace (stored in manifests so a moved Drive root keeps them valid)."""
        try:
            return Path(path).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)

    def resolve(self, rel: str) -> Path:
        return self.root / rel
