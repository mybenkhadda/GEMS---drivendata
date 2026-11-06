"""Exact area-weighted aggregation of work-resolution DEM features into 100 m statistics.

A 100 m cell is the (100/res)^2 block of aligned work pixels. A work pixel supports a feature when it has DEM
data, is farther than the feature's guard from missing data and farther than ``dem.seam_guard_m`` from a seam
between DEM projects. ``support_<base>`` = share of supporting pixels; a statistic is NaN where its support is
below ``dem.min_cell_support``. ``coverage`` = share of pixels with data that pass the seam and slope guards.
"""
from __future__ import annotations

import warnings

import numpy as np


def channels(p: dict, detailed: bool) -> list[tuple[str, str, str]]:
    """(summary name, base feature, statistic)."""
    s1, s2 = p["structure_scales_m"]
    r1, r2 = p["tpi_radii_m"]
    ch = [("edge_p50", "edge", "p50"), ("edge_p90", "edge", "p90"), ("edge_max", "edge", "max"),
          ("edge_anisotropy", "edge_anisotropy", "mean"), (f"coherence_p90_{s1}m", f"coherence_{s1}m", "p90"),
          (f"coherence_p90_{s2}m", f"coherence_{s2}m", "p90"), ("orient_c2", "strike_c2", "wmean_c"),
          ("orient_s2", "strike_s2", "wmean_s"), ("orient_R", "strike_c2", "wR"), ("slope_p50", "slope", "p50"),
          ("slope_p90", "slope", "p90"), ("slope_std", "slope", "std"),
          ("profile_curvature_min", "profile_curvature", "p05"), ("profile_curvature_max", "profile_curvature", "p95"),
          ("plan_curvature_absp90", "plan_curvature", "absp90"), (f"tpi_{r1}m_absp90", f"tpi_{r1}m", "absp90"),
          (f"tpi_{r2}m_absp90", f"tpi_{r2}m", "absp90"), ("relief_mean", "relief", "mean"),
          ("ridge_p90", "ridge", "p90"), ("valley_p90", "valley", "p90"), ("scarp_p90", "scarp", "p90"),
          ("roughness_mean", "roughness", "mean"), ("edge_fraction", "edge_pixel", "mean"),
          ("elevation_mean", "elevation", "mean")]
    if detailed:
        ch += [("crest_p90", "crest", "p90"), ("toe_p90", "toe", "p90")]
    return ch


def bases(p: dict, detailed: bool) -> list[str]:
    return list(dict.fromkeys(b for _, b, _ in channels(p, detailed)))


def summarise(F: dict, G: dict, z, d_miss, d_seam, sl, nr: int, nc: int, f: int, p: dict, detailed: bool,
              coh_weight: str):
    have = np.isfinite(z) & (d_seam > p["seam_guard_m"])
    minsup = float(p["min_cell_support"])

    def blocks(a, m):
        x = np.where(m, a, np.nan)[sl].astype(np.float32)
        return x.reshape(nr, f, nc, f).transpose(0, 2, 1, 3).reshape(nr, nc, f * f)

    use = {b: have & (d_miss > G[b]) for b in G}
    support = {b: use[b][sl].reshape(nr, f, nc, f).mean(axis=(1, 3)).astype(np.float32) for b in bases(p, detailed)}
    cov = (have & (d_miss > G["slope"]))[sl].reshape(nr, f, nc, f).mean(axis=(1, 3)).astype(np.float32)
    out = []
    cache = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wb = blocks(F[coh_weight], use["strike_c2"])
        cb, sb = blocks(F["strike_c2"], use["strike_c2"]), blocks(F["strike_s2"], use["strike_c2"])
        W = np.nansum(wb, -1)
        mc = np.nansum(wb * cb, -1) / np.maximum(W, 1e-12)
        ms = np.nansum(wb * sb, -1) / np.maximum(W, 1e-12)
        for name, b, stat in channels(p, detailed):
            if stat in ("wmean_c", "wmean_s", "wR"):
                v = {"wmean_c": mc, "wmean_s": ms, "wR": np.hypot(mc, ms)}[stat]
                v = np.where(W > 0, v, np.nan)
            else:
                if b not in cache:
                    cache[b] = blocks(F[b], use[b])
                x = cache[b]
                if stat == "mean":
                    v = np.nanmean(x, -1)
                elif stat == "std":
                    v = np.nanstd(x, -1)
                elif stat == "max":
                    v = np.nanmax(x, -1)
                elif stat == "absp90":
                    v = np.nanpercentile(np.abs(x), 90, -1)
                else:
                    v = np.nanpercentile(x, float(stat[1:]), -1)
            out.append(np.where(support[b] >= minsup, v, np.nan).astype(np.float32))
    return np.stack(out), cov, np.stack([support[b] for b in bases(p, detailed)])
