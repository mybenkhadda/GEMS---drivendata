"""Original physical observations prepared for ML.

Two groups, never mixed:

* ``raw_physical`` (``phys_<band>``): the 19 bands exactly as observed (float32, NaN nodata) - the physical
  reference for EDA, profiles and QC. Not a model family.
* ``raw`` (``raw_<band>``): model inputs after a FIXED, physically motivated transform declared per band in
  ``feature_store.yaml`` (identity, asinh(x / scale), signed log1p(|x| / scale), log1p, log10 of a positive
  quantity), plus missingness indicators ``miss_<band>`` for bands whose gaps differ from the footprint and the
  per-pixel count ``raw_n_missing``. No empirical normalisation is fitted here: robust scaling, when a model needs
  it, is fitted on the training rows of each fold (``models_patch.FoldScaler``).

The band-statistics pass (``band_stats``) reads one band at a time and records footprint coverage, moments,
quantiles and the constant-channel flag; its quantiles also give the fixed histogram edges of the texture entropy.
These are label-free footprint statistics (no fault label is read).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import FeatureSpec


def raw_transform(a: np.ndarray, kind: str, scale: float = 1.0) -> np.ndarray:
    a = np.asarray(a, np.float64)
    s = float(scale) if scale else 1.0
    if kind == "identity":
        return a
    if kind == "asinh":
        return np.arcsinh(a / s)
    if kind == "slog1p":
        return np.sign(a) * np.log1p(np.abs(a) / s)
    if kind == "log1p":
        return np.log1p(np.maximum(a, 0.0) / s)
    if kind == "log10":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(a > 0, np.log10(a / s), np.nan)
    raise KeyError(f"unknown raw transform {kind!r}")


def raw_names(bands: dict, missing_bands) -> tuple[list[str], list[str]]:
    """(physical ids, model ids) in band-declaration order."""
    phys = [f"phys_{b}" for b in bands]
    model = [f"raw_{b}" for b in bands] + [f"miss_{b}" for b in missing_bands] + ["raw_n_missing"]
    return phys, model


def build_raw_feature_specs(bands: dict, band_table: pd.DataFrame | None, missing_bands, stats: pd.DataFrame | None,
                            version: str) -> list[FeatureSpec]:
    """FeatureSpecs of both raw groups. Constant / empty bands are kept but ``usable=False``."""
    st = stats.set_index("band") if stats is not None and len(stats) else None
    desc = dict(zip(band_table.identifier, band_table.description)) if band_table is not None and len(band_table) \
        else {}
    out = []
    for b, spec in bands.items():
        spec = spec or {}
        usable = True
        if st is not None and b in st.index:
            usable = not bool(st.loc[b, "constant"]) and float(st.loc[b, "frac_finite"]) > 0
        unit = str(spec.get("unit", ""))
        out.append(FeatureSpec(f"phys_{b}", "raw_physical", "qc", f"observed {b}: {desc.get(b, '')}", unit, b,
                               "identity", None, None, "nan_nodata", True, (), "none", version, False))
        tr = str(spec.get("transform", "identity"))
        out.append(FeatureSpec(f"raw_{b}", "raw", "raw", f"{b} ({spec.get('modality', '')}) model input",
                               unit if tr == "identity" else "transformed", b,
                               f"{tr}(x / {spec.get('scale', 1.0)})" if tr != "identity" else "identity", None, None,
                               "nan_nodata", True, ("A", "B"), "none", version, usable,
                               {"transform": tr, "scale": spec.get("scale", 1.0)}))
    for b in missing_bands:
        out.append(FeatureSpec(f"miss_{b}", "raw", "raw", f"1 where {b} is missing inside the footprint", "flag", b,
                               "isnan", None, (0, 1), "never_missing", True, ("A", "B"), "none", version, True))
    out.append(FeatureSpec("raw_n_missing", "raw", "raw", "number of missing bands at the pixel", "count",
                           ",".join(bands), "sum(isnan)", None, (0, len(bands)), "never_missing", True, ("A", "B"),
                           "none", version, True))
    return out


def transform_raw_window(fields: dict, bands: dict, missing_bands, footprint: np.ndarray) -> tuple[dict, dict]:
    """(physical, model) dicts for one window. ``footprint`` = data footprint of the window (bool)."""
    phys, model = {}, {}
    nmiss = np.zeros(footprint.shape, np.float32)
    for b, spec in bands.items():
        spec = spec or {}
        a = np.asarray(fields[b], np.float64)
        phys[f"phys_{b}"] = a.astype(np.float32)
        model[f"raw_{b}"] = raw_transform(a, str(spec.get("transform", "identity")),
                                          float(spec.get("scale", 1.0))).astype(np.float32)
        nmiss += ~np.isfinite(a)
    for b in missing_bands:
        model[f"miss_{b}"] = np.where(footprint, (~np.isfinite(fields[b])).astype(np.float32), np.nan)
    model["raw_n_missing"] = np.where(footprint, nmiss, np.nan).astype(np.float32)
    return phys, model


def validate_raw_channels(arrays: dict, specs: list[FeatureSpec], valid: np.ndarray) -> list[dict]:
    """Per channel: finite coverage inside the footprint, constant flag, expected-range violations."""
    rows = []
    by = {s.feature_id: s for s in specs}
    for k, a in arrays.items():
        a = np.asarray(a)
        v = a[np.asarray(valid, bool)]
        fin = v[np.isfinite(v)]
        sp = by.get(k)
        rng_bad = 0
        if sp is not None and sp.expected_range and fin.size:
            lo, hi = sp.expected_range
            rng_bad = int(((fin < lo) | (fin > hi)).sum())
        rows.append({"feature": k, "frac_finite": float(fin.size / max(v.size, 1)),
                     "constant": bool(fin.size and float(fin.max()) == float(fin.min())),
                     "out_of_range": rng_bad, "status": "OK" if fin.size and not rng_bad else
                     ("EMPTY" if not fin.size else "RANGE_VIOLATION")})
    return rows


def footprint_of(path, n_bands: int, min_share: float = 0.5, abs_max: float = 1e30) -> np.ndarray:
    """Data footprint = pixels where >= ``min_share`` of the bands are finite (the NativeData rule), band by band."""
    import rasterio

    with rasterio.open(path) as ds:
        cnt = np.zeros((ds.height, ds.width), np.int16)
        for i in range(1, ds.count + 1):
            a = ds.read(i)
            cnt += (np.isfinite(a) & (np.abs(a) < abs_max)).astype(np.int16)
    return cnt >= int(np.ceil(min_share * n_bands))


def band_stats(path, band_index: dict, footprint: np.ndarray, entropy_levels: int = 16,
               quantiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> tuple[pd.DataFrame, dict]:
    """(per-band statistics table, {band: entropy bin edges}). One band in memory at a time."""
    from .raster_io import read_band_full

    rows, edges = [], {}
    fp = np.asarray(footprint, bool)
    n_fp = int(fp.sum())
    for b, idx in band_index.items():
        a = read_band_full(path, idx)
        v = a[fp]
        fin = v[np.isfinite(v)].astype(np.float64)
        r = {"band": b, "index": int(idx), "n_footprint": n_fp, "n_finite": int(fin.size),
             "frac_finite": float(fin.size / max(n_fp, 1)), "missing_in_footprint": int(n_fp - fin.size),
             "finite_outside_footprint": int(np.isfinite(a[~fp]).sum())}
        if fin.size:
            qv = np.quantile(fin, quantiles)
            r.update({"min": float(fin.min()), "max": float(fin.max()), "mean": float(fin.mean()),
                      "std": float(fin.std()), "median_abs": float(np.median(np.abs(fin))),
                      **{f"q{int(round(q * 100)):02d}": float(x) for q, x in zip(quantiles, qv)},
                      "constant": bool(fin.max() == fin.min())})
            e = np.unique(np.quantile(fin, np.linspace(0, 1, entropy_levels + 1)[1:-1]))
            edges[b] = [float(x) for x in e]
        else:
            r.update({"constant": True})
            edges[b] = []
        rows.append(r)
        del a
    return pd.DataFrame(rows), edges


def missing_indicator_bands(stats: pd.DataFrame, min_frac: float) -> list[str]:
    """Bands whose missing share INSIDE the footprint exceeds ``min_frac`` get an explicit indicator channel."""
    if stats is None or not len(stats):
        return []
    s = stats.assign(miss=1.0 - stats.frac_finite)
    return [str(b) for b in s.loc[s.miss > float(min_frac), "band"]]
