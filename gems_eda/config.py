"""Project configuration for the GEMS geological EDA.

Band names and modality groups follow the GDAL metadata embedded in
``gems-geodawn-numerical-features.tif`` (19 bands). Names are re-read from the
raster at load time; anything not listed here falls into the ``other`` group.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

RAW_FILES = {
    "features": "gems-geodawn-numerical-features.tif",
    "faults": "existing_faults.tif",
    "submission": "example_submission.tif",
}

# Geophysical modality -> raw band names (from the raster's GDAL metadata).
MODALITIES: dict[str, tuple[str, ...]] = {
    "magnetic": ("mag_anom", "rtp", "tmi", "tmi_hg", "tmi_vg", "tc"),
    "gravity": ("iso_grav_anom", "iso_grav_anom_hg", "iso_grav_anom_vg", "iso_grav_anom_slope"),
    "terrain": ("det_elev", "det_elev_slope"),
    "basement": ("depth_to_base_surf",),
    "conductivity": ("cond_surf",),
    "strain": ("geod_2ndinv", "geod_shearrate", "geod_dilaterate"),
    "seismicity": ("deq_n100a15", "ieq_n100a15"),
}

# The potential field / surface from which derivatives are computed per modality.
PRIMARY_FIELD = {
    "magnetic": "rtp",
    "gravity": "iso_grav_anom",
    "terrain": "det_elev",
    "basement": "depth_to_base_surf",
    "conductivity": "cond_surf",
    "strain": "geod_shearrate",
}

# Modalities with enough spatial resolution to carry fault-scale evidence.
# Strain and seismicity are regional (tens of km) and are kept as context only.
EVIDENCE_GROUPS = ("magnetic", "gravity", "terrain", "basement", "conductivity")
DETECTION_GROUPS = ("magnetic", "gravity", "terrain", "basement", "conductivity")

MODALITY_COLORS = {
    "magnetic": "#b03a2e",
    "gravity": "#1f618d",
    "terrain": "#9a7d0a",
    "basement": "#6c3483",
    "conductivity": "#117a65",
    "strain": "#ca6f1e",
    "seismicity": "#5d6d7e",
    "faults": "#111111",
    "other": "#7f8c8d",
}

# Physical scales (km) used for multi-scale transforms; sigma = scale / 2.
SCALES_KM = (2.0, 5.0, 10.0, 20.0)


def band_modality(name: str) -> str:
    """Modality of a raw or derived signal name (derived names are '<modality>:<transform>')."""
    if ":" in name:
        return name.split(":", 1)[0]
    for modality, bands in MODALITIES.items():
        if name in bands:
            return modality
    return "other"


# Accepted raw-file locations relative to a project root: the original flat layout (files next to
# outputs/) and the GEMS 2.0 data layout (data/raw/<source>/), used on Google Drive / Colab.
RAW_SUBDIRS = {
    "features": ("", "data/raw/geodawn"),
    "faults": ("", "data/raw/faults", "data/raw/geodawn"),
    "submission": ("", "data/raw/geodawn", "data/raw"),
}


def resolve_raw(root: Path, key: str) -> Path:
    """First existing location of raw file ``key`` under ``root`` (flat layout first); flat path if none exists."""
    root = Path(root)
    for sub in RAW_SUBDIRS.get(key, ("",)):
        p = root / sub / RAW_FILES[key]
        if p.exists():
            return p
    return root / RAW_FILES[key]


def find_project_root(start: Path | None = None) -> Path:
    """Project root: ``start`` if it holds the data, else ``$GEMS_ROOT``, else walk up from ``start`` / cwd.

    A root is any directory where the features GeoTIFF resolves (``resolve_raw``), i.e. either the flat
    layout or ``data/raw/geodawn/``.
    """
    import os

    def holds_data(p: Path) -> bool:
        return resolve_raw(p, "features").exists()

    if start is not None and holds_data(Path(start)):
        return Path(start).resolve()
    env = os.environ.get("GEMS_ROOT")
    if start is None and env and holds_data(Path(env)):
        return Path(env).resolve()
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if holds_data(candidate):
            return candidate
    raise FileNotFoundError(f"{RAW_FILES['features']} not found above {here} (set GEMS_ROOT or pass root)")


@dataclass(frozen=True)
class EDAPaths:
    """All output locations, relative to the project root."""

    root: Path

    @property
    def out(self) -> Path:
        return self.root / "outputs" / "eda"

    @property
    def figures(self) -> Path:
        return self.out / "figures"

    @property
    def tables(self) -> Path:
        return self.out / "tables"

    @property
    def features(self) -> Path:
        return self.out / "features"

    @property
    def datasets(self) -> Path:
        return self.out / "datasets"

    @property
    def findings(self) -> Path:
        return self.out / "EDA_FINDINGS.md"

    @property
    def manifest_file(self) -> Path:
        return self.out / "manifest.json"

    def raw(self, key: str) -> Path:
        return resolve_raw(self.root, key)

    def ensure(self) -> "EDAPaths":
        for p in (self.figures, self.tables, self.features, self.datasets):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class Manifest:
    """Registry of every artifact written by the EDA, with reload paths relative to the root."""

    paths: EDAPaths
    entries: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.paths.manifest_file.exists():
            self.entries = json.loads(self.paths.manifest_file.read_text(encoding="utf-8"))

    def add(self, name: str, path: Path, kind: str, description: str = "") -> Path:
        rel = Path(path).resolve().relative_to(self.paths.root.resolve()).as_posix()
        self.entries[name] = {"path": rel, "kind": kind, "description": description}
        self.paths.manifest_file.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")
        return Path(path)

    def path(self, name: str) -> Path:
        return self.paths.root / self.entries[name]["path"]


@dataclass
class EDAConfig:
    """Tunable analysis parameters. Distances are in km unless suffixed _px."""

    max_dim: int | None = 2048          # working-resolution cap (None = native grid)
    region_grid: tuple[int, int] = (3, 3)
    zoom_km: float = 60.0               # side length of the named zoom windows
    density_radius_km: float = 10.0
    short_fault_km: float = 3.0
    gap_km: float = 5.0
    gap_misalign_deg: float = 25.0
    profile_half_km: float = 12.0       # widened to >= 20 px at load time
    station_spacing_km: float = 2.0
    tangent_km: float = 2.0
    max_stations: int = 6000
    core_km: float = 1.0                # |offset| treated as the fault core
    background_min_km: float = 15.0     # background stations at least this far from labels
    near_tol_px: int = 1
    far_km: float = 5.0
    strong_evidence: float = 0.90
    weak_evidence: float = 0.60
    candidate_scales_km: tuple[float, ...] = (2.0, 5.0, 10.0)
    candidate_high_q: float = 0.95
    candidate_low_q: float = 0.85
    candidate_min_km: float = 6.0
    candidate_match_km: float = 2.0
    seed: int = 7
