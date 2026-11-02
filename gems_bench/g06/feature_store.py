"""The central, reusable geological feature store.

Layout (inside the run directory)::

    features/
        footprint.npz                    data footprint (>= 50 % of bands finite: the NativeData rule)
        raw/raw_physical.tif             phys_<band>  (observed values; QC / EDA reference, never a model input)
        raw/raw_model.tif                raw_<band>, miss_<band>, raw_n_missing
        magnetic/magnetic.tif            mag_<rtp>_<op>_s<m>, mag_<rtp>_persist
        magnetic/magnetic_spectral.tif   tilt, analytic signal, tilt-zero closeness, upward-continuation residuals
        gravity/gravity.tif, gravity/gravity_spectral.tif
        multiscale/multiscale.tif        extra-field gradients (terrain, basement) + DoG band energies
        texture/texture.tif              tex_*
        structural/structural.tif        struct_* (structure tensor, ridges, continuity, cross-modal, discontinuity)
        context/context.tif              ctx_* regional (label-free)
        context/context_global.tif       ctx_dist_nodata_km, ctx_domain (unsupervised)
        <dir>/<group>.json               group manifest (fingerprint, names, coverage)
        feature_registry.parquet         one row per channel: FeatureSpec + group / band index / coverage / usable
        feature_schema.json              ordered ids, group files, schema hash, grid, source band indices, halo
        feature_coverage.csv             finite share / moments per channel inside the footprint

One tiled pass computes every TILED group from one halo read of the 19 bands; each tile is cached (``TileCache``)
and only then marked complete, so a Colab disconnect resumes at the first incomplete tile. Each group is assembled
into one compressed GeoTIFF (``GroupWriter``) and verified (alignment + a tile read-back). GLOBAL groups (FFT,
distance to nodata, domains) are computed afterwards from full single bands / the assembled context group.

Nothing here is fitted on labels. Normalisation that must be training-only is fitted inside the folds by the models.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import feature_context as FC
from . import feature_physics as FP
from . import feature_raw as FR
from . import feature_structure as FS
from . import feature_texture as FT
from .contracts import ContractError, FeatureSpec, GridSpec, SchemaError

GROUPS = {   # name: (directory, file, mode, family)
    "raw_physical": ("raw", "raw_physical.tif", "tiled", "qc"),
    "raw": ("raw", "raw_model.tif", "tiled", "raw"),
    "magnetic": ("magnetic", "magnetic.tif", "tiled", "physics"),
    "gravity": ("gravity", "gravity.tif", "tiled", "physics"),
    "multiscale": ("multiscale", "multiscale.tif", "tiled", "physics"),
    "texture": ("texture", "texture.tif", "tiled", "texture"),
    "structural": ("structural", "structural.tif", "tiled", "structure"),
    "context": ("context", "context.tif", "tiled", "context"),
    "magnetic_spectral": ("magnetic", "magnetic_spectral.tif", "global", "physics"),
    "gravity_spectral": ("gravity", "gravity_spectral.tif", "global", "physics"),
    "context_global": ("context", "context_global.tif", "global", "context"),
}
GROUP_ORDER = tuple(GROUPS)
SCHEMA_VERSION = "g06.schema.1"


# =============================================================================== specs / names
def _enabled(cfg: dict, g: str) -> bool:
    gc = (cfg.get("groups") or {}).get(g)
    if g in ("raw_physical", "raw"):
        gc = (cfg.get("groups") or {}).get("raw")
    return bool(gc) and bool(gc.get("enabled", True))


def _gcfg(cfg: dict, g: str) -> dict:
    return dict((cfg.get("groups") or {}).get("raw" if g == "raw_physical" else g) or {})


def group_names(cfg: dict, missing_bands) -> dict:
    """{group: ordered feature ids} of every ENABLED group (deterministic from the configuration)."""
    bands = cfg["bands"]
    out = {}
    if _enabled(cfg, "raw"):
        phys, model = FR.raw_names(bands, missing_bands)
        out["raw_physical"], out["raw"] = phys, model
    for g, pre in (("magnetic", "mag"), ("gravity", "grav")):
        if _enabled(cfg, g):
            c = _gcfg(cfg, g)
            out[g] = FP.physics_names(pre, c["field"], c["scales_m"], tuple(c.get("ops", FP.PHYS_OPS)), True)
    if _enabled(cfg, "multiscale"):
        c = _gcfg(cfg, "multiscale")
        n = []
        for f in c.get("extra_fields", []):
            n += FP.physics_names("ms", f, c.get("extra_scales_m", []), tuple(c.get("extra_ops", ("hgm",))),
                                  "edge" in c.get("extra_ops", ()) and len(c.get("extra_scales_m", [])) > 0)
        dog = c.get("dog") or {}
        for f in dog.get("fields", []):
            n += FP.frequency_names(f, dog["bands_m"])
        out["multiscale"] = n
    if _enabled(cfg, "texture"):
        out["texture"] = [n for n, _, _ in FT.texture_names(_gcfg(cfg, "texture"))]
    if _enabled(cfg, "structural"):
        out["structural"] = FS.structure_names(_gcfg(cfg, "structural"))
    if _enabled(cfg, "context"):
        out["context"] = FC.context_names(_gcfg(cfg, "context"))
    for g, pre in (("magnetic_spectral", "mag"), ("gravity_spectral", "grav")):
        if _enabled(cfg, g):
            c = _gcfg(cfg, g)
            out[g] = FP.spectral_names(pre, c["field"], c.get("upward_km", []))
    if _enabled(cfg, "context_global"):
        out["context_global"] = FC.global_context_names(_gcfg(cfg, "context_global"))
    for g, names in out.items():
        if len(set(names)) != len(names):
            raise ContractError(f"group {g}: duplicate feature ids")
    allids = [n for g in out for n in out[g]]
    if len(set(allids)) != len(allids):
        raise ContractError("a feature id appears in two groups")
    return out


def _scale_of(name: str) -> float | None:
    import re

    m = re.search(r"_s(\d+)$", name) or re.search(r"_w(\d+)$", name)
    if m is None:
        return None
    v = float(m.group(1))
    return v if name.endswith(f"_s{m.group(1)}") else v * 100.0


def build_specs(cfg: dict, names: dict, band_table: pd.DataFrame | None, stats: pd.DataFrame | None,
                missing_bands, version: str) -> list[FeatureSpec]:
    """FeatureSpec of every channel (descriptions generated from the operator vocabulary)."""
    specs = []
    if "raw" in names:
        specs += FR.build_raw_feature_specs(cfg["bands"], band_table, missing_bands, stats, version)
    units = {b: str((s or {}).get("unit", "")) for b, s in cfg["bands"].items()}
    for g, ids in names.items():
        if g in ("raw", "raw_physical"):
            continue
        d, f, mode, fam = GROUPS[g]
        for n in ids:
            parts = n.split("_")
            op = next((p for p in reversed(parts) if p in FP.PHYS_UNITS), None)
            src = next((b for b in sorted(cfg["bands"], key=len, reverse=True) if f"_{b}_" in f"_{n}_"), "")
            if g in ("magnetic", "gravity") or n.startswith("ms_"):
                desc = FP.PHYS_DESC.get(op, n)
                unit = FP.PHYS_UNITS.get(op, "").format(u=units.get(src, "")) if op else ""
                tr = f"masked Gaussian (normalised convolution) + {op or 'operator'}"
                if "_dog" in n or "_specslope" in n:
                    desc, unit, tr = "log10 local band energy / spectral slope", "log10", "difference of Gaussians"
                if n.endswith("_persist"):
                    desc, unit, tr = "share of scales with a thin gradient-maximum edge", "share", "NMS across scales"
            elif g.endswith("_spectral"):
                desc = {"tilt": "tilt derivative atan(VDR/THDR)", "asig": "analytic-signal amplitude",
                        "tiltzero": "closeness to the tilt zero contour"}.get(parts[-1], "residual vs upward "
                                                                                     "continuation")
                unit, tr = ("rad" if parts[-1] == "tilt" else ""), "FFT (whole raster, nearest-filled, masked)"
            elif g == "texture":
                desc, unit, tr = f"window statistic {parts[-2]}", "", "masked box filter"
            elif g == "structural":
                desc, unit, tr = f"structural descriptor {'_'.join(parts[1:])}", "", "structure tensor / Hessian"
            else:
                desc, unit, tr = f"regional context {n}", "", "masked regional filter"
            fit_scope = "full_footprint_unsupervised" if n in ("ctx_domain",) or n.endswith("_asinh") else "none"
            specs.append(FeatureSpec(n, g, fam, desc, unit, src or "derived", tr, _scale_of(n), None,
                                     "nan_outside_support", True, ("A", "B"), fit_scope, version, True,
                                     {"group_mode": mode}))
    return specs


def required_halo_px(cfg: dict) -> int:
    """Largest operator support of the enabled tiled groups (pixels)."""
    h = 2
    for g, pre in (("magnetic", "mag"), ("gravity", "grav")):
        if _enabled(cfg, g):
            h = max(h, FP.required_halo_px(_gcfg(cfg, g)["scales_m"]))
    if _enabled(cfg, "multiscale"):
        c = _gcfg(cfg, "multiscale")
        h = max(h, FP.required_halo_px(c.get("extra_scales_m", []) or [100]))
        dog = c.get("dog") or {}
        for (lo, hi), im in zip(dog.get("bands_m", []), dog.get("integration_m", [])):
            h = max(h, FP.gaussian_radius(FP.sigma_px(hi)) + FP.gaussian_radius(FP.sigma_px(im)) + 2)
    if _enabled(cfg, "texture"):
        h = max(h, FT.required_halo_px(_gcfg(cfg, "texture")))
    if _enabled(cfg, "structural"):
        h = max(h, FS.required_halo_px(_gcfg(cfg, "structural")))
    if _enabled(cfg, "context"):
        h = max(h, FC.required_halo_px(_gcfg(cfg, "context")))
    return int(h)


def effective_halo(cfg: dict) -> int:
    return max(int((cfg.get("tiling") or {}).get("halo_px", 0)), required_halo_px(cfg))


# =============================================================================== tile computation
def footprint_window(fields: dict, n_bands: int, min_share: float = 0.5) -> np.ndarray:
    cnt = sum(np.isfinite(a).astype(np.int16) for a in fields.values())
    return cnt >= int(np.ceil(min_share * n_bands))


def compute_tile(fields: dict, cfg: dict, names: dict, missing_bands, aux: dict, win=None) -> dict:
    """{group: (C, h, w) float32 stack in ``names[group]`` order} for one padded window. ``aux``:
    {"entropy_edges": {band: edges}, "median_abs": {band: value}}. The core is cropped when ``win`` is given."""
    bands = cfg["bands"]
    fp = footprint_window(fields, len(bands), float((cfg.get("nodata") or {}).get("footprint_min_share", 0.5)))
    out: dict = {}
    cache: dict = {}
    if "raw" in names:
        phys, model = FR.transform_raw_window(fields, bands, missing_bands, fp)
        out["raw_physical"], out["raw"] = phys, model
    if "magnetic" in names:
        c = _gcfg(cfg, "magnetic")
        out["magnetic"] = FP.magnetic_features(fields[c["field"]], c["scales_m"], tuple(c.get("ops", FP.PHYS_OPS)),
                                               cache=cache, field=c["field"])
    if "gravity" in names:
        c = _gcfg(cfg, "gravity")
        out["gravity"] = FP.gravity_features(fields[c["field"]], c["scales_m"], tuple(c.get("ops", FP.PHYS_OPS)),
                                             cache=cache, field=c["field"])
    if "multiscale" in names:
        c = _gcfg(cfg, "multiscale")
        d = {}
        ops = tuple(c.get("extra_ops", ("hgm",)))
        for f in c.get("extra_fields", []):
            o = ops + (("persist",) if "edge" in ops else ())
            d.update(FP.multiscale_gradients(fields[f], c.get("extra_scales_m", []), "ms", f, o, cache=cache))
        dog = c.get("dog") or {}
        for f in dog.get("fields", []):
            d.update(FP.frequency_features(fields[f], f, dog["bands_m"], dog["integration_m"]))
        out["multiscale"] = d
    if "texture" in names:
        out["texture"] = FT.texture_feature_bank(fields, _gcfg(cfg, "texture"), aux.get("entropy_edges") or {})
    if "structural" in names:
        out["structural"] = FS.structural_feature_bank(fields, _gcfg(cfg, "structural"), cache)
    if "context" in names:
        out["context"] = FC.context_feature_bank(fields, _gcfg(cfg, "context"), aux.get("median_abs") or {})
    cache.clear()
    stacks = {}
    for g, d in out.items():
        want = names[g]
        miss, extra = [n for n in want if n not in d], [n for n in d if n not in want]
        if miss or extra:
            raise SchemaError(f"group {g}: computed channels differ from the registry (missing {miss[:5]}, "
                              f"unexpected {extra[:5]})")
        st = np.stack([np.where(fp, np.asarray(d[n], np.float32), np.nan) for n in want])
        if win is not None:
            rs, cs = win.core
            st = st[:, rs, cs]
        stacks[g] = np.ascontiguousarray(st, np.float32)
    return stacks


# =============================================================================== build
class _Coverage:
    def __init__(self, names):
        n = len(names)
        self.names = names
        self.cnt, self.s1, self.s2 = np.zeros(n), np.zeros(n), np.zeros(n)
        self.mn, self.mx = np.full(n, np.inf), np.full(n, -np.inf)
        self.n_fp = 0

    def add(self, stack: np.ndarray, fp: np.ndarray) -> None:
        v = stack[:, fp].astype(np.float64)
        self.n_fp += int(fp.sum())
        ok = np.isfinite(v)
        self.cnt += ok.sum(1)
        z = np.where(ok, v, 0.0)
        self.s1 += z.sum(1)
        self.s2 += (z * z).sum(1)
        if v.shape[1]:
            self.mn = np.minimum(self.mn, np.where(ok, v, np.inf).min(1))
            self.mx = np.maximum(self.mx, np.where(ok, v, -np.inf).max(1))

    def table(self, group: str) -> pd.DataFrame:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self.s1 / np.maximum(self.cnt, 1)
            std = np.sqrt(np.maximum(self.s2 / np.maximum(self.cnt, 1) - mean ** 2, 0))
        return pd.DataFrame({"feature": self.names, "group": group, "n_footprint": self.n_fp,
                             "frac_finite": self.cnt / max(self.n_fp, 1),
                             "min": np.where(np.isfinite(self.mn), self.mn, np.nan),
                             "max": np.where(np.isfinite(self.mx), self.mx, np.nan), "mean": mean, "std": std})


def _group_path(root: Path, g: str) -> Path:
    d, f, _, _ = GROUPS[g]
    return Path(root) / "features" / d / f


def _group_manifest(root: Path, g: str) -> Path:
    d, _, _, _ = GROUPS[g]
    return Path(root) / "features" / d / f"{g}.json"


def group_reusable(root: Path, g: str, names: list[str], fingerprint: str, grid: GridSpec) -> bool:
    from .persistence import read_json
    from .raster_io import verify_raster_alignment

    m = read_json(_group_manifest(root, g))
    p = _group_path(root, g)
    if not isinstance(m, dict) or m.get("fingerprint") != fingerprint or m.get("names") != names or not p.exists():
        return False
    try:
        return not verify_raster_alignment(p, grid, names)
    except Exception:
        return False


def build_feature_store(src_path, band_index: dict, grid: GridSpec, cfg: dict, root, footprint: np.ndarray,
                        stats: pd.DataFrame, entropy_edges: dict, band_table: pd.DataFrame | None, fingerprint: str,
                        version: str, reuse: bool = True, progress=print, groups_only=None) -> dict:
    """Compute, assemble, register and validate every enabled group. Returns a summary dict."""
    import rasterio

    from .persistence import write_json, write_table
    from .raster_io import GroupWriter, TileCache, iter_windows, read_aligned_window, verify_raster_alignment

    root = Path(root)
    t0 = time.perf_counter()
    tiling = cfg.get("tiling") or {}
    tile_px, block_px = int(tiling.get("tile_px", 512)), int(tiling.get("block_px", 256))
    halo = effective_halo(cfg)
    missing_bands = FR.missing_indicator_bands(stats, float(_gcfg(cfg, "raw").get("missing_min_frac", 0.001)))
    names = group_names(cfg, missing_bands)
    if groups_only:
        names = {g: v for g, v in names.items() if g in groups_only}
    aux = {"entropy_edges": entropy_edges,
           "median_abs": dict(zip(stats.band, stats.get("median_abs", pd.Series(1.0, index=stats.index))))}
    fp_full = np.asarray(footprint, bool)
    if fp_full.shape != grid.shape:
        raise ContractError(f"footprint {fp_full.shape} != grid {grid.shape}")
    (root / "features").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "features" / "footprint.npz", footprint=np.packbits(fp_full),
                        shape=np.asarray(fp_full.shape))
    tiled = [g for g in names if GROUPS[g][2] == "tiled"]
    todo = [g for g in tiled if not (reuse and group_reusable(root, g, names[g], fingerprint, grid))]
    wins = list(iter_windows(grid.shape, tile_px, halo))
    caches = {g: TileCache(root / "tiles" / fingerprint[:10] / g, names[g]) for g in todo}
    order = list(cfg["bands"])
    idx = [int(band_index[b]) for b in order]
    n_done = n_reused = 0
    if todo:
        with rasterio.open(src_path) as ds:
            for i, w in enumerate(wins):
                if reuse and all(caches[g].complete(w) for g in todo):
                    n_reused += 1
                    continue
                if not fp_full[w.grid_slice].any():
                    for g in todo:
                        caches[g].save(w, np.full((len(names[g]), *w.core_shape), np.nan, np.float32))
                    n_done += 1
                    continue
                arr = read_aligned_window(ds, idx, w)
                fields = {b: arr[j].astype(np.float64) for j, b in enumerate(order)}
                stacks = compute_tile(fields, cfg, {g: names[g] for g in todo}, missing_bands, aux, w)
                core_fp = fp_full[w.grid_slice]
                for g in todo:
                    caches[g].save(w, np.where(core_fp[None], stacks[g], np.nan).astype(np.float32))
                n_done += 1
                if progress and (i % 8 == 0 or i == len(wins) - 1):
                    progress(f"  tile {i + 1}/{len(wins)} {w.tile_id} ({time.perf_counter() - t0:.0f} s)")
    coverage = []
    for g in tiled:
        p = _group_path(root, g)
        if g not in todo:
            m = json.loads(_group_manifest(root, g).read_text(encoding="utf-8"))
            coverage.append(pd.DataFrame(m["coverage"]))
            continue
        cov = _Coverage(names[g])
        gw = GroupWriter(p, names[g], grid, {"G06_GROUP": g, "G06_FINGERPRINT": fingerprint}, block_px,
                         str(tiling.get("compress", "deflate")))
        try:
            for w in wins:
                st = caches[g].load(w)
                gw.write(w, st)
                cov.add(st, fp_full[w.grid_slice])
            gw.close()
        except Exception:
            gw.abort()
            raise
        probs = verify_raster_alignment(p, grid, names[g], int(cfg["grid"]["epsg"]))
        mid = wins[len(wins) // 2]
        from .raster_io import read_window_bands

        back = read_window_bands(p, list(range(1, len(names[g]) + 1)), (mid.row0, mid.row1, mid.col0, mid.col1))
        ref = caches[g].load(mid)
        if not np.array_equal(np.isnan(back), np.isnan(ref)) or not np.allclose(back[np.isfinite(back)],
                                                                                ref[np.isfinite(ref)]):
            probs.append(f"tile read-back {mid.tile_id} differs from the cached tile")
        if probs:
            raise ContractError(f"group {g} failed verification: {probs}")
        T = cov.table(g)
        write_json(_group_manifest(root, g), {"group": g, "fingerprint": fingerprint, "names": names[g],
                                              "halo_px": halo, "tile_px": tile_px, "n_tiles": len(wins),
                                              "coverage": T.to_dict("list"), "written": time.strftime(
                                                  "%Y-%m-%dT%H:%M:%S")})
        coverage.append(T)
        if not bool(tiling.get("keep_tile_cache", False)):
            caches[g].clear()
        progress and progress(f"  group {g}: {len(names[g])} channels assembled + verified")
    # global groups
    for g in [x for x in names if GROUPS[x][2] == "global"]:
        if reuse and group_reusable(root, g, names[g], fingerprint, grid):
            m = json.loads(_group_manifest(root, g).read_text(encoding="utf-8"))
            coverage.append(pd.DataFrame(m["coverage"]))
            continue
        arrays = _global_group(g, cfg, src_path, band_index, fp_full, root, names, grid)
        arrays = {n: np.where(fp_full, np.asarray(arrays[n], np.float32), np.nan) for n in names[g]}
        from .raster_io import write_feature_raster

        p = write_feature_raster(_group_path(root, g), arrays, grid, {"G06_GROUP": g,
                                                                     "G06_FINGERPRINT": fingerprint}, block_px)
        probs = verify_raster_alignment(p, grid, names[g], int(cfg["grid"]["epsg"]))
        if probs:
            raise ContractError(f"group {g} failed verification: {probs}")
        cov = _Coverage(names[g])
        cov.add(np.stack([arrays[n] for n in names[g]]), fp_full)
        T = cov.table(g)
        write_json(_group_manifest(root, g), {"group": g, "fingerprint": fingerprint, "names": names[g],
                                              "coverage": T.to_dict("list"),
                                              "written": time.strftime("%Y-%m-%dT%H:%M:%S")})
        coverage.append(T)
        del arrays
        progress and progress(f"  global group {g}: {len(names[g])} channels")
    COV = pd.concat(coverage, ignore_index=True) if coverage else pd.DataFrame()
    specs = build_specs(cfg, names, band_table, stats, missing_bands, version)
    REG = registry_frame(specs, names, COV, float((cfg.get("registry") or {}).get("min_coverage", 0.01)))
    write_table(root / "features" / "feature_registry.parquet", REG)
    write_table(root / "features" / "feature_coverage.csv", COV)
    schema = {"schema_version": SCHEMA_VERSION, "feature_ids": list(REG.feature_id),
              "groups": {g: {"file": f"{GROUPS[g][0]}/{GROUPS[g][1]}", "mode": GROUPS[g][2], "features": names[g]}
                         for g in names},
              "grid": grid.to_dict(), "band_index": {k: int(v) for k, v in band_index.items()},
              "missing_indicator_bands": missing_bands, "halo_effective_px": halo, "tile_px": tile_px,
              "store_fingerprint": fingerprint, "version": version}
    schema["schema_hash"] = schema_hash_of(schema)
    write_json(root / "features" / "feature_schema.json", schema)
    return {"n_features": len(REG), "n_usable": int(REG.usable.sum()), "groups": {g: len(v) for g, v in
                                                                                  names.items()},
            "halo_effective_px": halo, "tiles": len(wins), "tiles_computed": n_done, "tiles_reused": n_reused,
            "schema_hash": schema["schema_hash"], "seconds": round(time.perf_counter() - t0, 1)}


def schema_hash_of(schema: dict) -> str:
    from .persistence import stable_hash

    return stable_hash(schema["schema_version"], schema["feature_ids"],
                       {g: v["file"] for g, v in schema["groups"].items()}, schema["grid"], schema["band_index"])


def registry_frame(specs: list[FeatureSpec], names: dict, COV: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    rows = []
    cov = COV.set_index("feature") if len(COV) else pd.DataFrame()
    for s in specs:
        r = s.to_row()
        g = s.group
        r["band_in_group"] = names[g].index(s.feature_id) + 1
        r["file"] = f"{GROUPS[g][0]}/{GROUPS[g][1]}"
        if len(cov) and s.feature_id in cov.index:
            c = cov.loc[s.feature_id]
            for k in ("frac_finite", "min", "max", "mean", "std"):
                r[k] = float(c[k])
            const = bool(np.isfinite(c["std"]) and float(c["std"]) == 0.0)
            r["constant"] = const
            r["usable"] = bool(s.usable and not const and float(c["frac_finite"]) >= min_coverage)
        rows.append(r)
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    T = pd.DataFrame(rows)
    T["_g"] = T.group.map(order)
    return T.sort_values(["_g", "band_in_group"]).drop(columns="_g").reset_index(drop=True)


def _global_group(g: str, cfg: dict, src_path, band_index: dict, fp: np.ndarray, root: Path, names: dict,
                  grid: GridSpec) -> dict:
    from .raster_io import read_band_full

    c = _gcfg(cfg, g)
    if g in ("magnetic_spectral", "gravity_spectral"):
        a = read_band_full(src_path, band_index[c["field"]])
        a = np.where(fp, a, np.nan)
        return FP.spectral_features(a, "mag" if g.startswith("magnetic") else "grav", c["field"],
                                    float(c.get("smooth_m", 100)), c.get("upward_km", []))
    if g == "context_global":
        out = {"ctx_dist_nodata_km": FC.distance_to_nodata_km(fp, float(c.get("dist_cap_km", 20.0)))}
        dc = c.get("domains") or {}
        if dc.get("enabled", True):
            out["ctx_domain"] = _domains(root, dc, fp, grid)
        return out
    raise KeyError(g)


def _domains(root: Path, dc: dict, fp: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Unsupervised regional domains (k-means on a footprint sample of regional features; label-free)."""
    import joblib

    store = open_feature_store(root, partial=True)
    feats = [f for f in dc.get("features", []) if f in store.names]
    if len(feats) < 2:
        raise ContractError(f"domains need >= 2 regional features in the store, found {feats}")
    rng = np.random.default_rng(int(dc.get("seed", 2026)))
    ir = np.flatnonzero(fp.ravel())
    pick = np.sort(rng.choice(ir, min(int(dc.get("sample", 200000)), ir.size), replace=False))
    r, cc = np.unravel_index(pick, fp.shape)
    X = store.sample(feats, r, cc)
    sc, km = FC.geological_domain_features(X, int(dc.get("k", 6)), int(dc.get("seed", 2026)))
    out = np.full(fp.shape, np.nan, np.float32)
    step = 512
    for r0 in range(0, fp.shape[0], step):
        r1 = min(r0 + step, fp.shape[0])
        A = store.load(feats, (r0, r1, 0, fp.shape[1]))
        Xw = A.reshape(len(feats), -1).T
        out[r0:r1] = FC.predict_domains(sc, km, Xw).reshape(r1 - r0, fp.shape[1])
    p = Path(root) / "features" / "context" / "domain_model.joblib"
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": sc, "kmeans": km, "features": feats}, p)
    return out


# =============================================================================== reading
class FeatureStore:
    """Read-only view of a built store: ordered ids, registry, lazy window / point access."""

    def __init__(self, root, schema: dict, registry: pd.DataFrame):
        self.root = Path(root)
        self.schema = schema
        self.registry = registry
        self.names = list(registry.feature_id) if len(registry) else []
        self._where = {}
        for g, v in schema.get("groups", {}).items():
            for i, n in enumerate(v["features"], 1):
                self._where[n] = (g, i)

    @property
    def schema_hash(self) -> str:
        return str(self.schema.get("schema_hash", ""))

    @property
    def grid(self) -> GridSpec:
        g = self.schema["grid"]
        return GridSpec(int(g["height"]), int(g["width"]), str(g["crs"]), tuple(g["transform"]), float(g["pixel_m"]))

    def path(self, group: str) -> Path:
        return self.root / "features" / self.schema["groups"][group]["file"]

    def index_of(self, name: str) -> tuple[str, int]:
        if name not in self._where:
            raise SchemaError(f"feature {name!r} is not in the store (schema {self.schema_hash})")
        return self._where[name]

    def usable(self, names=None) -> list[str]:
        R = self.registry
        ok = set(R.loc[R.usable.astype(bool), "feature_id"]) if "usable" in R else set(self.names)
        return [n for n in (names or self.names) if n in ok]

    def by_family(self, families, track: str = "A", usable_only: bool = True) -> list[str]:
        R = self.registry
        m = R.family.isin(list(families)) & R.permitted_tracks.astype(str).str.contains(track)
        out = list(R.loc[m, "feature_id"])
        return self.usable(out) if usable_only else out

    def footprint(self) -> np.ndarray:
        with np.load(self.root / "features" / "footprint.npz") as z:
            shape = tuple(int(x) for x in z["shape"])
            return np.unpackbits(z["footprint"])[: shape[0] * shape[1]].reshape(shape).astype(bool)

    def _groups(self, names):
        by: dict = {}
        for j, n in enumerate(names):
            g, i = self.index_of(n)
            by.setdefault(g, []).append((j, i))
        return by

    def load(self, names, window=None) -> np.ndarray:
        """(C, h, w) float32 in ``names`` order; ``window`` = (r0, r1, c0, c1), default the full grid."""
        from .raster_io import read_window_bands

        names = list(names)
        h, w = self.grid.shape
        win = window or (0, h, 0, w)
        out = np.empty((len(names), int(win[1] - win[0]), int(win[3] - win[2])), np.float32)
        for g, pairs in self._groups(names).items():
            a = read_window_bands(self.path(g), [i for _, i in pairs], win)
            for k, (j, _) in enumerate(pairs):
                out[j] = a[k]
        return out

    def sample(self, names, rows, cols) -> np.ndarray:
        """(N, C) float32 at integer pixels, ``names`` order."""
        from .raster_io import read_points

        names = list(names)
        out = np.empty((len(rows), len(names)), np.float32)
        for g, pairs in self._groups(names).items():
            a = read_points(self.path(g), [i for _, i in pairs], rows, cols)
            for k, (j, _) in enumerate(pairs):
                out[:, j] = a[:, k]
        return out


def open_feature_store(root, partial: bool = False) -> FeatureStore:
    """Open a built store. ``partial=True`` (used while building) reads the group manifests instead of the final
    schema / registry, so global groups can consume the already assembled tiled groups."""
    from .persistence import read_json

    root = Path(root)
    if not partial:
        schema = read_json(root / "features" / "feature_schema.json")
        reg = root / "features" / "feature_registry.parquet"
        if not isinstance(schema, dict) or not reg.exists():
            raise FileNotFoundError(f"no feature store under {root / 'features'} (run the feature stage first)")
        return FeatureStore(root, schema, pd.read_parquet(reg))
    groups, ids = {}, []
    grid = None
    for g, (d, f, _, _) in GROUPS.items():
        m = read_json(root / "features" / d / f"{g}.json")
        if isinstance(m, dict) and (root / "features" / d / f).exists():
            groups[g] = {"file": f"{d}/{f}", "features": m["names"]}
            ids += m["names"]
            if grid is None:
                import rasterio

                with rasterio.open(root / "features" / d / f) as ds:
                    grid = GridSpec.from_dataset(ds).to_dict()
    schema = {"groups": groups, "grid": grid, "schema_hash": "partial"}
    return FeatureStore(root, schema, pd.DataFrame({"feature_id": ids}))


def load_features(store: FeatureStore, names, window=None) -> np.ndarray:
    return store.load(names, window)


def validate_feature_store(store: FeatureStore, grid: GridSpec, epsg: int, expected_band_index: dict | None = None
                           ) -> list[str]:
    """Problems of a built store (empty = valid): files, alignment, schema hash, registry / schema agreement."""
    from .contracts import validate_band_identity
    from .raster_io import verify_raster_alignment

    probs = []
    if store.schema.get("schema_hash") != schema_hash_of(store.schema):
        probs.append("schema hash does not match the schema content")
    if list(store.registry.feature_id) != list(store.schema.get("feature_ids", [])):
        probs.append("registry order differs from the schema")
    for g, v in store.schema.get("groups", {}).items():
        p = store.root / "features" / v["file"]
        if not p.exists():
            probs.append(f"{g}: file {v['file']} missing")
            continue
        probs += [f"{g}: {x}" for x in verify_raster_alignment(p, grid, v["features"], epsg)]
    if expected_band_index:
        try:
            validate_band_identity(store.schema.get("band_index", {}), expected_band_index)
        except ContractError as e:
            probs.append(str(e))
    return probs
