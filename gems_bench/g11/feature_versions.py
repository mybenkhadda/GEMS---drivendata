"""Feature identity: dependency classes, per-class specification hashes, layer ids and the per-layer completion
manifest of the feature pool (``03_features/layer_manifest.parquet``).

A feature layer depends on exactly one *dependency class*:

* ``geophysical`` - GeoDAWN bands and everything derived from them (raw, geo_*, G06 indices, geo_edge_rank);
* ``dem``         - everything derived from the DEM aggregation (dem100_*, geo x DEM multimodal, terrain_scarp_rank);
* ``label``       - targets derived from the mapped faults (Y_*). Label layers are never predictors.

Each class has a specification hash built ONLY from the inputs and config keys that change its values
(:func:`spec_hashes`). Split, loss, training, sampling, model and evaluation settings are not part of any feature spec,
so changing them never invalidates a feature layer. The DEM processing resolution is part of the ``dem`` spec and
of the DEM layer id (``dem100_slope_mean@r10m``): layers computed at 2 m and 10 m never share an id, a file or a
manifest row.

The pool manifest stores one row per layer that was verified complete, with the size / mtime / sha256 recorded at
creation (or at verified recovery), so later launches check ``stat`` only and re-hash a sample (never 30 GB).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import stable_hash

DEP_CLASSES = ("geophysical", "dem", "label")
DEM_FAMILIES = ("dem_100m", "multimodal")
DEM_DERIVED_NAMES = ("terrain_scarp_rank",)
DEM_SPEC_KEYS = ("stats_resolution_m", "context_scales_m", "allow_unknown_vertical", "seam_tolerance_m", "priority")
LAYER_COLUMNS = ["layer_id", "name", "family", "dep_class", "processing_resolution_m", "relpath", "size", "mtime_ns",
                 "sha256", "finite_fraction", "nodata_outside_valid", "spec_hash", "status", "provenance", "created_by",
                 "verified_utc", "unit", "scale_m", "source", "label_derived"]


def dep_class(family: str, name: str) -> str:
    if family == "target" or str(name).startswith("Y_"):
        return "label"
    if family in DEM_FAMILIES or name in DEM_DERIVED_NAMES or str(name).startswith("dem100_"):
        return "dem"
    return "geophysical"


def res_tag(res_m: float) -> str:
    return f"r{float(res_m):g}m"


def layer_id(name: str, dclass: str, dem_res_m: float | None) -> str:
    """Pool id of a layer: DEM-class layers carry their processing resolution (``name@r10m``)."""
    if dclass == "dem":
        if dem_res_m is None:
            raise ValueError(f"{name}: a DEM-class layer needs its processing resolution")
        return f"{name}@{res_tag(dem_res_m)}"
    return name


def layer_relpath(name: str, family: str, dclass: str, dem_res_m: float | None, version_private: str | None = None) -> str:
    """Relative path (under ``03_features``) for a NEWLY computed layer. Existing (legacy) layers keep their path."""
    from .features import family_dir

    sub = family_dir(family)
    if version_private:                                     # FORCE_NEW_VERSION: nothing shared with the pool
        return f"versions/{version_private}/{sub}/{name}.npy"
    if dclass == "dem":
        return f"{sub}/{res_tag(dem_res_m)}/{name}.npy"
    return f"{sub}/{name}.npy"


def grid_signature(grid) -> dict:
    return {"crs": str(grid.crs), "shape": [int(x) for x in grid.shape],
            "transform": [round(float(v), 6) for v in tuple(grid.transform)[:6]]}


def spec_hashes(cfg: dict, source_hashes: dict, grid, dem_catalog_sha: str | None = None,
                g06_sources: list | None = None) -> dict:
    """Specification hash per dependency class (only the inputs that change the layer values)."""
    gg = cfg["geophysical_grid"]
    geo = stable_hash({"geodawn": source_hashes.get("geodawn"), "grid": grid_signature(grid),
                       "valid_min_band_share": gg.get("valid_min_band_share"), "nodata_abs": float(gg.get("nodata_abs")),
                       "geophysics": cfg["feature_engineering"]["geophysics"], "g06": sorted(g06_sources or [])})
    dem = stable_hash({"geophysical": geo, "dem_catalog": dem_catalog_sha,
                       "dem": {k: cfg["dem"].get(k) for k in DEM_SPEC_KEYS}})
    lab = cfg["labels"]
    label = stable_hash({"fault_raster": source_hashes.get("fault_raster"), "grid": grid_signature(grid),
                         "faults": cfg["faults"], "buffers_m": lab["buffers_m"], "near_band_m": lab.get("near_band_m")})
    return {"geophysical": geo, "dem": dem, "label": label,
            "dem_statistics_resolution_m": float(cfg["dem"]["stats_resolution_m"])}


def spec_diff(a: dict, b: dict) -> list[str]:
    return [f"{k}: {a.get(k)} != {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


# ------------------------------------------------------------------------------------------ expected layers
class _NameStore:
    """Minimal in-memory store used to enumerate the layers a generator writes (dry run on a tiny grid)."""

    def __init__(self, grid):
        self.grid = grid
        self.arrays, self.rows = {}, []

    def put(self, name, arr, family, scale_m=None, unit="", source="", label_derived=False, fingerprint=""):
        self.arrays[name] = np.asarray(arr, np.float32)
        self.rows.append({"name": name, "family": family, "scale_m": scale_m, "unit": unit, "source": source,
                          "label_derived": bool(label_derived)})

    def get(self, name, mmap=True):
        return self.arrays[name]

    def flush(self):
        return None


def _tiny_grid(n: int = 96):
    from .grid import GridSpec

    g = GridSpec("EPSG:32611", (100.0, 0.0, 500000.0, 0.0, -100.0, 4000000.0), (n, n))
    g.valid = np.ones((n, n), bool)
    return g


def expected_layers(cfg: dict, raw_band_names: list, g06_names: list | None = None, dem_available: bool = True,
                    dem_res_m: float | None = None) -> pd.DataFrame:
    """Every layer the CURRENT configuration produces (name, family, dep_class, layer_id, unit, scale_m, label_derived).

    Generated by running the real generators on a tiny synthetic grid (exact names, no hand-maintained list);
    G06 layer names come from the attached G06 registry (``g06_names``), raw names from the GeoDAWN band list.
    """
    from .features import geophysics as G
    from .features.multiscale import compute_multimodal_alignment

    grid = _tiny_grid()
    st = _NameStore(grid)
    rng = np.random.default_rng(0)
    bands = {k: rng.normal(0, 1, grid.shape).astype(np.float32) for k in cfg["feature_engineering"]["geophysics"]["bands"]}
    for b in raw_band_names:
        st.put(f"raw_{b}", np.zeros(grid.shape, np.float32), "raw", 100.0, "as delivered", "geodawn")
    G.compute_geophysical_to_store(bands, grid, cfg, st, "")
    for n in g06_names or []:
        st.put(n, np.zeros(grid.shape, np.float32), "g06", None, "", "g06 store")
    st.put("geo_edge_rank", np.zeros(grid.shape, np.float32), "evidence_rank", None, "rank", "label-free")
    if dem_available:
        from .dem_reader import dem_stat_names

        for n in dem_stat_names(cfg):
            st.put(f"dem100_{n}", np.zeros(grid.shape, np.float32), "dem_100m", 100.0, "", "dem_1m aggregation")
        st.put("terrain_scarp_rank", np.zeros(grid.shape, np.float32), "evidence_rank", 100.0, "rank", "label-free")
        for s in multimodal_scales(cfg):
            k = f"{int(s)}m"
            geo = {f"{f}_{q}_{k}": rng.uniform(0, 1, grid.shape).astype(np.float32)
                   for f in ("magnetic", "gravity") for q in ("sc2", "ss2", "coh", "gm")}
            terr = {"sc2": geo[f"magnetic_sc2_{k}"], "ss2": geo[f"magnetic_ss2_{k}"], "coh": geo[f"magnetic_coh_{k}"],
                    "scarp": geo[f"magnetic_gm_{k}"]}
            mm = compute_multimodal_alignment(geo, terr, grid.valid, int(s))
            for r in mm.meta.itertuples():
                st.put(r.name, mm[r.name], "multimodal", r.scale_m, r.unit, "geo x dem")
    t = pd.DataFrame(st.rows).drop_duplicates("name", keep="last")
    for k in labels_expected(cfg):
        t = pd.concat([t, pd.DataFrame([{"name": k, "family": "target", "scale_m": None,
                                         "unit": "m" if k == "Y_DISTANCE" else "1", "source": "fault labels",
                                         "label_derived": True}])], ignore_index=True)
    t["dep_class"] = [dep_class(f, n) for f, n in zip(t.family, t.name)]
    res = float(dem_res_m if dem_res_m is not None else cfg["dem"]["stats_resolution_m"])
    t["processing_resolution_m"] = np.where(t.dep_class == "dem", res, np.nan)
    t["layer_id"] = [layer_id(n, c, res) for n, c in zip(t.name, t.dep_class)]
    return t.reset_index(drop=True)


def labels_expected(cfg: dict) -> list[str]:
    return ["Y_TRACE_100M"] + [f"Y_BUFFER_{int(b)}M" for b in cfg["labels"]["buffers_m"]] + ["Y_DISTANCE"]


def multimodal_scales(cfg: dict) -> list:
    sc = cfg["feature_engineering"]["geophysics"]["scales_m"]
    return [s for s in sc if s in (400, 800)] or sc[:1]


# ------------------------------------------------------------------------------------------ pool manifest
class LayerManifest:
    """``03_features/layer_manifest.parquet``: one row per layer verified complete (the completion manifest).

    A layer is reusable only through a row here: status VERIFIED_COMPLETE, spec_hash equal to the requested spec,
    and the file's current size / mtime equal to the recorded ones (sha256 re-checked for a sample or on demand).
    """

    def __init__(self, features_root):
        self.root = Path(features_root)
        self.path = self.root / "layer_manifest.parquet"
        self.df = pd.read_parquet(self.path) if self.path.is_file() else pd.DataFrame(columns=LAYER_COLUMNS)
        for c in LAYER_COLUMNS:
            if c not in self.df.columns:
                self.df[c] = None
        self._dirty = False

    def __len__(self):
        return len(self.df)

    def row(self, lid: str) -> dict | None:
        r = self.df[self.df.layer_id == lid]
        return None if r.empty else r.iloc[-1].to_dict()

    def upsert(self, rows) -> None:
        rows = [dict(r) for r in rows]
        if not rows:
            return
        for r in rows:
            r.setdefault("verified_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        new = pd.DataFrame(rows)
        keep = self.df[~self.df.layer_id.isin(new.layer_id)]
        self.df = new if keep.empty else pd.concat([keep, new], ignore_index=True)
        self._dirty = True

    def flush(self) -> Path:
        from .registry import atomic_write_parquet

        out = self.df.copy()
        out["label_derived"] = out["label_derived"].map(lambda v: bool(v) if v is not None and v == v else False)
        atomic_write_parquet(self.path, out[LAYER_COLUMNS])
        self._dirty = False
        return self.path

    def stat_ok(self, lid: str) -> tuple[bool, str]:
        r = self.row(lid)
        if r is None:
            return False, "no completion manifest"
        if r.get("status") != "VERIFIED_COMPLETE":
            return False, f"status {r.get('status')}"
        p = self.root / str(r["relpath"])
        if not p.is_file():
            return False, "file missing"
        st = p.stat()
        if int(st.st_size) != int(r["size"]):
            return False, f"size {st.st_size} != recorded {r['size']}"
        if int(st.st_mtime_ns) != int(r["mtime_ns"]):
            return False, "modified after verification (mtime changed)"
        return True, "stat matches completion manifest"

    def reusable(self, lid: str, spec_hash: str) -> tuple[bool, str]:
        ok, why = self.stat_ok(lid)
        if not ok:
            return ok, why
        r = self.row(lid)
        if r.get("spec_hash") != spec_hash:
            return False, f"spec {r.get('spec_hash')} != requested {spec_hash}"
        return True, why
