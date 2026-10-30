"""Streamlit-free API for downstream experiments (P6 / M9): request saved features by name, computed by exactly the
same engine and formulas as the app.

    import sys; sys.path.insert(0, "/content/GEMS/streamlit/gems_layer_lab")
    from lab.integration import FeatureStore
    fs = FeatureStore("/content/drive/MyDrive/GEMS_2")
    fs.list()                                   # saved features (name, version, status, hash, inputs)
    a = fs.compute("mag_grav_ratio")            # full native grid (float32, NaN = invalid), tiled with halos
    w = fs.compute("mag_grav_ratio", window=(row0, col0, height, width))
    fs.export(["mag_grav_ratio"])               # native GeoTIFF + manifest (never overwrites)
    fs.exported_path("mag_grav_ratio")          # latest export of the current version, if any

No label is ever read by these calls; features flagged AOI-dependent are refused for full-grid computation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.windows import Window

from .engine import Engine
from .library import FeatureLibrary
from .store import OutputStore

FEATURES_REL = "data/raw/geodawn/gems-geodawn-numerical-features.tif"
FAULTS_REL = "data/raw/faults/existing_faults.tif"
TEMPLATE_REL = "data/raw/example_submission.tif"


class FeatureStore:
    def __init__(self, gems_root: str | Path, extras: list | None = None, plugin_dirs: list | None = None,
                 features_tif: str | None = None, template_tif: str | None = None):
        """``features_tif`` / ``template_tif`` override the GEMS_2 layout (e.g. the TIFs next to app.py); the registry
        and outputs stay under ``<gems_root>/outputs/geological_layer_lab``."""
        self.root = Path(gems_root)
        self.store = OutputStore(self.root)
        self.library = FeatureLibrary(self.store.registry_path)
        tmpl = template_tif or str(self.root / TEMPLATE_REL)
        self.engine = Engine(features_tif or str(self.root / FEATURES_REL), None, extras or [], self.library,
                             plugin_dirs, footprint_tif=tmpl if Path(tmpl).is_file() else None)

    def list(self) -> list[dict]:
        return [{k: e.get(k) for k in ("name", "version", "status", "formula_hash", "formula", "raw_inputs",
                                       "depends_on", "aoi_dependent", "halo_px")} for e in
                self.library.features.values()]

    def compute(self, name: str, window: tuple | None = None, tile: int = 512) -> np.ndarray:
        """Saved feature on a window (row0, col0, height, width) or the full grid (tiled)."""
        if name not in self.library.features:
            raise KeyError(f"no saved feature {name!r}")
        if self.engine.library_meta().get(name, (0, True))[1]:
            raise ValueError(f"{name} is AOI-dependent (unfrozen normalisation): not computable as a grid feature")
        if window is not None:
            r0, c0, h, w = window
            return self.engine.evaluate(None, Window(c0, r0, w, h), library_targets=[name]).arrays[name]
        H, W = self.engine.H, self.engine.W
        out = np.full((H, W), np.nan, np.float32)
        for r in range(0, H, tile):
            for c in range(0, W, tile):
                w = Window(c, r, min(tile, W - c), min(tile, H - r))
                out[r:r + int(w.height), c:c + int(w.width)] = self.engine.evaluate(
                    None, w, library_targets=[name]).arrays[name]
                self.engine.clear_cache()
        return out

    def export(self, names: list, tile: int = 512, footprint_mask: bool = False, progress=None) -> dict:
        from .export_native import export_features

        return export_features(self.engine, names, self.store.root / "exports", tile, footprint_mask,
                               progress=progress)

    def exported_path(self, name: str) -> str | None:
        e = self.library.get(name)
        ok = [x for x in e.get("exports", []) if x.get("formula_hash") == e["formula_hash"] and Path(x["path"]).exists()]
        return ok[-1]["path"] if ok else None
