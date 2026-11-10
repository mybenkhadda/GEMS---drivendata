"""100 m geophysical features (label-free) at physical scales 100 m - 3.2 km.

Built on the G06 masked operators (``g06.feature_physics``: normalised convolution, never zero filling). Gaussian
sigma = scale / 2 (a scale is a smoothing width, not a source depth). Per field ``<f>`` (magnetic, gravity,
elevation) and scale ``<s>m``:

  gm     Gaussian gradient magnitude (field unit / km)       dx, dy  derivatives east / north (unit / km)
  lap    Laplacian (unit / km^2)                             hl1, hl2  Hessian eigenvalues (unit / km^2)
  coh    structure-tensor coherence (1)                      sc2, ss2  axial strike cos 2a / sin 2a (1)
  ridge  max(-hl2, 0)   valley  max(hl1, 0)                  aniso  (l1-l2)/(l1+l2) of the tensor
Cross-scale: ``<f>_persist`` = share of scales with gm above its footprint ``persistence_quantile``;
``<f>_wav_<a>_<b>`` = log10 energy of the difference of Gaussians between scales a and b (wavelet-like band);
``magrav_orient_agree_<s>m`` = coherence-weighted cos 2(a_mag - a_grav) (axial, 180-degree symmetric).
Existing G06 store features are attached by :func:`attach_g06_store` when present (never recomputed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from gems_bench.g06.feature_physics import masked_smooth

from . import FeatureCube


def _deriv_km(S: np.ndarray, px_km: float) -> dict:
    dr, dc = np.gradient(S, px_km)
    dx, dy = dc, -dr
    dxr, dxx = np.gradient(dx, px_km)
    dyr, _ = np.gradient(dy, px_km)
    return {"dx": dx, "dy": dy, "dxx": dxx, "dyy": -dyr, "dxy": -dxr}


def axial_from_gradient(dx, dy, rho_px: float) -> dict:
    gx, gy = np.nan_to_num(dx), np.nan_to_num(dy)
    jxx = ndimage.gaussian_filter(gx * gx, rho_px)
    jyy = ndimage.gaussian_filter(gy * gy, rho_px)
    jxy = ndimage.gaussian_filter(gx * gy, rho_px)
    l_sum = jxx + jyy + 1e-30
    l_dif = np.hypot(jxx - jyy, 2 * jxy)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)               # gradient axis, math angle from east
    az = np.radians((90.0 - np.degrees(theta + np.pi / 2)) % 180.0)   # strike azimuth clockwise from north
    return {"coh": l_dif / l_sum, "sc2": np.cos(2 * az), "ss2": np.sin(2 * az), "aniso": l_dif / l_sum}


def axial_agreement(c2a, s2a, c2b, s2b) -> np.ndarray:
    """cos 2(a - b) from axial vectors: +1 parallel, -1 perpendicular (180-degree symmetric)."""
    return c2a * c2b + s2a * s2b


def field_features(a: np.ndarray, field: str, scales_m, px_m: float, rho_factor: float, valid: np.ndarray,
                   persistence_q: float, wavelet_bands_m) -> tuple[dict, list]:
    px_km = px_m / 1000.0
    out, meta = {}, []
    gms = []
    smooth = {}
    for s in scales_m:
        sig = max(s / px_m / 2.0, 0.5)
        S, _ = masked_smooth(a, sig, 0.25)
        smooth[s] = S
        d = _deriv_km(S, px_km)
        gm = np.hypot(d["dx"], d["dy"])
        tr, disc = d["dxx"] + d["dyy"], np.sqrt(np.maximum(0.25 * (d["dxx"] - d["dyy"]) ** 2 + d["dxy"] ** 2, 0))
        l1, l2 = 0.5 * tr + disc, 0.5 * tr - disc
        ax = axial_from_gradient(d["dx"], d["dy"], rho_factor * sig)
        k = f"{field}_{{}}_{int(s)}m"
        for nm, arr, unit in (("gm", gm, "u/km"), ("dx", d["dx"], "u/km"), ("dy", d["dy"], "u/km"),
                              ("lap", tr, "u/km^2"), ("hl1", l1, "u/km^2"), ("hl2", l2, "u/km^2"),
                              ("ridge", np.maximum(-l2, 0), "u/km^2"), ("valley", np.maximum(l1, 0), "u/km^2"),
                              ("coh", ax["coh"], "1"), ("sc2", ax["sc2"], "1"), ("ss2", ax["ss2"], "1"),
                              ("aniso", ax["aniso"], "1")):
            out[k.format(nm)] = np.where(valid & np.isfinite(S), arr, np.nan).astype(np.float32)
            meta.append({"name": k.format(nm), "family": f"geo_{field}", "scale_m": float(s), "unit": unit})
        gms.append(out[k.format("gm")])
    thr = [np.nanquantile(g[valid], persistence_q) if np.isfinite(g[valid]).any() else np.inf for g in gms]
    pers = np.nanmean(np.stack([(g >= t).astype(np.float32) for g, t in zip(gms, thr)]), 0)
    out[f"{field}_persist"] = np.where(valid, pers, np.nan).astype(np.float32)
    meta.append({"name": f"{field}_persist", "family": f"geo_{field}", "scale_m": None, "unit": "fraction"})
    for lo, hi in wavelet_bands_m:
        Sa, _ = masked_smooth(a, max(lo / px_m / 2, 0.5), 0.25)
        Sb, _ = masked_smooth(a, max(hi / px_m / 2, 0.5), 0.25)
        e, _ = masked_smooth((Sa - Sb) ** 2, max(hi / px_m / 2, 0.5), 0.25)
        nm = f"{field}_wav_{int(lo)}_{int(hi)}m"
        out[nm] = np.where(valid, np.log10(np.maximum(e, 1e-30)), np.nan).astype(np.float32)
        meta.append({"name": nm, "family": f"geo_{field}", "scale_m": float(hi), "unit": "log10(u^2)"})
    return out, meta


def compute_geophysical_features(bands: dict, grid, cfg: dict) -> FeatureCube:
    """``bands``: {'magnetic': array, 'gravity': array, 'elevation': array, ...} on ``grid`` (NaN = nodata)."""
    g = cfg["feature_engineering"]["geophysics"]
    valid = grid.valid if grid.valid is not None else np.ones(grid.shape, bool)
    arrays, meta, notes = {}, [], []
    for field, a in bands.items():
        if a is None:
            notes.append(f"{field}: band absent - skipped")
            continue
        o, m = field_features(np.asarray(a, np.float64), field, g["scales_m"], grid.res[0], g["structure_rho_factor"],
                              valid, g["persistence_quantile"], g["wavelet_bands_m"])
        arrays.update(o)
        meta += m
    if "magnetic" in bands and "gravity" in bands and bands["magnetic"] is not None and bands["gravity"] is not None:
        for s in g["scales_m"]:
            k = int(s)
            ag = axial_agreement(arrays[f"magnetic_sc2_{k}m"], arrays[f"magnetic_ss2_{k}m"],
                                 arrays[f"gravity_sc2_{k}m"], arrays[f"gravity_ss2_{k}m"])
            w = arrays[f"magnetic_coh_{k}m"] * arrays[f"gravity_coh_{k}m"]
            arrays[f"magrav_orient_agree_{k}m"] = (ag * np.sqrt(np.clip(w, 0, 1))).astype(np.float32)
            arrays[f"magrav_coh_product_{k}m"] = w.astype(np.float32)
            meta += [{"name": f"magrav_orient_agree_{k}m", "family": "geo_cross", "scale_m": float(s), "unit": "1"},
                     {"name": f"magrav_coh_product_{k}m", "family": "geo_cross", "scale_m": float(s), "unit": "1"}]
    return FeatureCube(arrays, pd.DataFrame(meta), grid, notes)


def compute_geophysical_to_store(bands: dict, grid, cfg: dict, store, fingerprint: str = "") -> list[str]:
    """Memory-bounded variant of :func:`compute_geophysical_features`: one (field, scale) at a time is computed and
    written to the memmapped ``store``; cross-scale persistence, wavelet energy and mag-grav agreement are then
    derived from the stored layers. Peak RAM ~ 15 full-grid float64 arrays instead of all scales at once."""
    g = cfg["feature_engineering"]["geophysics"]
    valid = grid.valid if grid.valid is not None else np.ones(grid.shape, bool)
    names = []
    for field, a in bands.items():
        if a is None:
            continue
        a = np.asarray(a, np.float64)
        for s in g["scales_m"]:
            o, m = field_features(a, field, [s], grid.res[0], g["structure_rho_factor"], valid, g["persistence_quantile"], [])
            for r in m:
                if r["name"].endswith("_persist"):
                    continue
                store.put(r["name"], o[r["name"]], r["family"], r["scale_m"], r["unit"], "geodawn", False, fingerprint)
                names.append(r["name"])
            del o
        gms = [np.asarray(store.get(f"{field}_gm_{int(s)}m"), np.float32) for s in g["scales_m"]]
        thr = [np.nanquantile(x[valid], g["persistence_quantile"]) for x in gms]
        pers = np.nanmean(np.stack([(x >= t).astype(np.float32) for x, t in zip(gms, thr)]), 0)
        store.put(f"{field}_persist", np.where(valid, pers, np.nan), f"geo_{field}", None, "fraction", "geodawn", False,
                  fingerprint)
        names.append(f"{field}_persist")
        del gms, pers
        for lo, hi in g["wavelet_bands_m"]:
            Sa, _ = masked_smooth(a, max(lo / grid.res[0] / 2, 0.5), 0.25)
            Sb, _ = masked_smooth(a, max(hi / grid.res[0] / 2, 0.5), 0.25)
            e, _ = masked_smooth((Sa - Sb) ** 2, max(hi / grid.res[0] / 2, 0.5), 0.25)
            nm = f"{field}_wav_{int(lo)}_{int(hi)}m"
            store.put(nm, np.where(valid, np.log10(np.maximum(e, 1e-30)), np.nan), f"geo_{field}", float(hi),
                      "log10(u^2)", "geodawn", False, fingerprint)
            names.append(nm)
    if bands.get("magnetic") is not None and bands.get("gravity") is not None:
        for s in g["scales_m"]:
            k = int(s)
            get = lambda n: np.asarray(store.get(n), np.float32)        # noqa: E731
            ag = axial_agreement(get(f"magnetic_sc2_{k}m"), get(f"magnetic_ss2_{k}m"), get(f"gravity_sc2_{k}m"),
                                 get(f"gravity_ss2_{k}m"))
            w = get(f"magnetic_coh_{k}m") * get(f"gravity_coh_{k}m")
            store.put(f"magrav_orient_agree_{k}m", ag * np.sqrt(np.clip(w, 0, 1)), "geo_cross", float(s), "1", "geodawn",
                      False, fingerprint)
            store.put(f"magrav_coh_product_{k}m", w, "geo_cross", float(s), "1", "geodawn", False, fingerprint)
            names += [f"magrav_orient_agree_{k}m", f"magrav_coh_product_{k}m"]
    store.flush()
    return names


def _win(ds, grid):
    from ..grid import raster_window_on_grid

    return None if grid is None else raster_window_on_grid(ds, grid)


def read_all_bands(path, grid=None) -> dict:
    """Every GeoDAWN band by its exact identifier (``g06.raster_io.band_identifiers``), nodata -> NaN.
    With ``grid`` only the window covering it is read (SMOKE sub-grid)."""
    import rasterio

    from gems_bench.g06.raster_io import band_identifiers

    out = {}
    with rasterio.open(path) as ds:
        win = _win(ds, grid)
        for i, name in enumerate(band_identifiers(ds), 1):
            a = ds.read(i, window=win, masked=True).astype(np.float32).filled(np.nan)
            a[np.abs(a) >= 1e30] = np.nan
            out[name] = a
    return out


def band_registry(path) -> pd.DataFrame:
    """Name, description, unit, dtype, nodata and decimated statistics of every band (19 expected for GeoDAWN)."""
    import rasterio

    from gems_bench.g06.raster_io import band_identifiers

    rows = []
    with rasterio.open(path) as ds:
        f = max(1, int(np.ceil(np.sqrt(ds.width * ds.height / 1_000_000))))
        for i, name in enumerate(band_identifiers(ds), 1):
            a = ds.read(i, out_shape=(ds.height // f, ds.width // f), masked=True).astype(np.float32).filled(np.nan)
            a[np.abs(a) >= 1e30] = np.nan
            v = a[np.isfinite(a)]
            rows.append({"band": i, "name": name, "description": ds.descriptions[i - 1] or "",
                         "unit": (ds.units[i - 1] if ds.units else "") or ds.tags(i).get("units", ""),
                         "dtype": ds.dtypes[i - 1], "nodata": ds.nodata, "valid_fraction": float(np.isfinite(a).mean()),
                         "min": float(v.min()) if v.size else None, "max": float(v.max()) if v.size else None,
                         "mean": float(v.mean()) if v.size else None, "std": float(v.std()) if v.size else None})
    return pd.DataFrame(rows)


def read_geodawn_bands(path, names: dict, grid=None) -> dict:
    """Bands by exact description/tag name (``g06.raster_io.resolve_exact_band``); missing -> None (recorded)."""
    import rasterio

    from gems_bench.g06.raster_io import resolve_exact_band

    out = {}
    with rasterio.open(path) as ds:
        win = _win(ds, grid)
        for field, bn in names.items():
            try:
                i = resolve_exact_band(ds, bn)
            except (KeyError, ValueError):
                out[field] = None
                continue
            a = ds.read(i, window=win, masked=True).astype(np.float32).filled(np.nan)
            a[np.abs(a) >= 1e30] = np.nan
            out[field] = a
    return out


def attach_g06_store(registry_path, store, grid) -> list[str]:
    """Register existing G06 feature rasters (same grid only) in the G11 store without recomputation."""
    import rasterio

    reg = pd.read_parquet(registry_path)
    col = next((c for c in ("path", "file", "output_path") if c in reg.columns), None)
    name_col = next((c for c in ("name", "feature", "feature_id") if c in reg.columns), None)
    if col is None or name_col is None:
        return []
    added = []
    from pathlib import Path

    base = Path(registry_path).parent
    for r in reg.itertuples():
        p = Path(getattr(r, col))
        p = p if p.is_absolute() else base / p
        if not p.is_file() or p.suffix.lower() not in (".tif", ".tiff"):
            continue
        with rasterio.open(p) as ds:
            if tuple(ds.shape) != tuple(grid.shape) or not np.allclose(tuple(ds.transform)[:6], grid.transform[:6]):
                continue
            a = ds.read(1, masked=True).astype(np.float32).filled(np.nan)
        nm = f"g06_{getattr(r, name_col)}"
        store.put(nm, a, family="g06", source=str(p))
        added.append(nm)
    return added
