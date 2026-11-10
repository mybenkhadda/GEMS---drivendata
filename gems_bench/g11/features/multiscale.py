"""Joint geophysics x terrain alignment on the 100 m grid (axial orientation geometry, 180-degree symmetric).

Inputs are rank-normalised (percentile over the footprint, label-free) so modalities with different units can be
combined. Missing terrain (no DEM) stays NaN - never treated as "no terrain expression".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import FeatureCube
from .geophysics import axial_agreement


def pct_rank(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(a.shape, np.nan, np.float32)
    m = mask & np.isfinite(a)
    v = a[m]
    if v.size:
        order = np.argsort(v, kind="stable")
        r = np.empty(v.size, np.float32)
        r[order] = (np.arange(v.size, dtype=np.float32) + 0.5) / v.size
        out[m] = r
    return out


def compute_multimodal_alignment(geo: dict, terrain: dict, valid: np.ndarray, scale_m: int,
                                 strong: float = 0.9) -> FeatureCube:
    """``geo`` needs magnetic / gravity ``*_gm/coh/sc2/ss2_<s>m`` (and optionally a conductivity field);
    ``terrain`` needs 100 m terrain summaries ``coh``, ``sc2``, ``ss2``, ``scarp`` (NaN without DEM)."""
    k = f"{int(scale_m)}m"
    out, meta = {}, []

    def put(n, a, unit="1"):
        out[n] = np.where(valid, a, np.nan).astype(np.float32)
        meta.append({"name": n, "family": "multimodal", "scale_m": float(scale_m), "unit": unit})

    tc, t2, ts2 = terrain.get("coh"), terrain.get("sc2"), terrain.get("ss2")
    for fld in ("magnetic", "gravity", "conductivity"):
        if f"{fld}_sc2_{k}" not in geo or tc is None:
            continue
        ag = axial_agreement(geo[f"{fld}_sc2_{k}"], geo[f"{fld}_ss2_{k}"], t2, ts2)
        put(f"{fld}_terrain_orient_agree_{k}", ag)
        put(f"{fld}_terrain_coh_product_{k}", geo[f"{fld}_coh_{k}"] * tc)
    gr = {f: pct_rank(geo[f"{f}_gm_{k}"], valid) for f in ("magnetic", "gravity", "conductivity") if f"{f}_gm_{k}" in geo}
    tr = pct_rank(terrain["scarp"], valid) if terrain.get("scarp") is not None else None
    if gr:
        geo_best = np.nanmax(np.stack(list(gr.values())), 0)
        put(f"geo_edge_rank_max_{k}", geo_best)
        if tr is not None:
            put(f"cross_modal_coincidence_{k}", np.sqrt(np.clip(geo_best, 0, 1) * np.clip(tr, 0, 1)))
            have_t = np.isfinite(tr)
            put(f"geo_without_terrain_{k}", np.where(have_t, (geo_best >= strong) & (tr < 0.5), np.nan))
            put(f"terrain_without_geo_{k}", np.where(have_t, (tr >= strong) & (geo_best < 0.5), np.nan))
    return FeatureCube(out, pd.DataFrame(meta))


def structural_persistence(axials: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Resultant length of coherence-weighted axial vectors across scales/modalities (1 = same strike)."""
    c = sum(w * a for a, _, w in axials)
    s = sum(w * b for _, b, w in axials)
    W = sum(w for _, _, w in axials)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.hypot(c, s) / np.maximum(W, 1e-12)).astype(np.float32)


def terrain_100m_summaries(store, grid, scale_m: int, rho_factor: float = 2.0) -> dict:
    """100 m terrain orientation / edge layers derived from the aggregated 1 m DEM (``dem100_elev_mean``).

    Lineament strike comes from the structure tensor of that surface (axial, like the geophysics) - NOT from the
    aspect, which is a 360-degree slope direction. ``scarp`` = rank-free cell p95 slope. NaN without DEM coverage.
    """
    from .geophysics import field_features

    try:
        z = np.asarray(store.get("dem100_elev_mean"), np.float64)
        slope95 = np.asarray(store.get("dem100_slope_p95"), np.float32)
    except KeyError:
        return {"coh": None, "sc2": None, "ss2": None, "scarp": None}
    valid = np.isfinite(z)
    o, _ = field_features(z, "dem", [scale_m], grid.res[0], rho_factor, valid, 0.9, [])
    k = f"{int(scale_m)}m"
    return {"coh": o[f"dem_coh_{k}"], "sc2": o[f"dem_sc2_{k}"], "ss2": o[f"dem_ss2_{k}"], "scarp": slope95}
