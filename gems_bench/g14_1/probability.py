"""Raw probability-map diagnostics (Phase G1): is there usable ranking before any post-processing?

All functions are label-light: they use the map, the valid footprint and the (test-time input) known-fault raster.
Held-out truth enters only ``truth_ranking`` (AUC / average precision of the raw score on a held-out unit), which is a
measurement on an evaluation zone, never a fitted quantity.

UNTESTED: written without execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

DIST_BINS_KM = (0.0, 0.05, 0.15, 0.25, 0.35, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, np.inf)


def _pix_km(transform) -> float:
    return abs(float(tuple(transform)[0])) / 1000.0


def summary(p: np.ndarray, valid: np.ndarray, known: np.ndarray, transform, top_fracs=(0.001, 0.01, 0.05, 0.08)) -> dict:
    """Distribution, entropy, high-confidence area, mass placement relative to the footprint and known faults."""
    a = np.asarray(p, np.float32)
    v = valid & np.isfinite(a)
    x = np.clip(a[v].astype(np.float64), 1e-7, 1 - 1e-7)
    q = np.quantile(x, [0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1]) if x.size else [np.nan] * 11
    ent = -(x * np.log2(x) + (1 - x) * np.log2(1 - x))
    kd = ndimage.distance_transform_edt(~known) if known.any() else np.full(a.shape, np.inf)
    mass = float(x.sum())
    out = {"valid_px": int(v.sum()), "finite_outside_valid_px": int((np.isfinite(a) & ~valid).sum()),
           "mass_outside_valid": float(np.nansum(a[~valid])) if (~valid).any() else 0.0,
           **{f"q{k}": float(val) for k, val in zip(("0", "0.1", "1", "5", "25", "50", "75", "95", "99", "99.9", "100"), q)},
           "mean": float(x.mean()) if x.size else np.nan, "mean_binary_entropy_bits": float(ent.mean()) if x.size else np.nan,
           "area_p_gt_0.5": float((x > 0.5).mean()) if x.size else np.nan,
           "area_p_gt_0.9": float((x > 0.9).mean()) if x.size else np.nan, "mass": mass}
    for r in (0, 1, 2, 3, 5, 10):
        m = v & (kd <= r)
        out[f"mass_share_within_{r}px_of_known"] = float(np.nansum(a[m]) / max(mass, 1e-12))
    out["mass_removed_by_pixel_exact_mask"] = out["mass_share_within_0px_of_known"]
    s = np.where(v, np.nan_to_num(a), -np.inf)
    vals = s[v]
    for f in top_fracs:
        k = max(int(np.ceil(f * vals.size)), 1)
        thr = np.partition(vals, vals.size - k)[vals.size - k]
        top = v & (s >= thr)
        topm = top & ~known
        out[f"top{f}_share_on_known"] = float((top & known).sum() / max(top.sum(), 1))
        out[f"top{f}_share_within_3px_known"] = float((top & (kd <= 3)).sum() / max(top.sum(), 1))
        out[f"top{f}_eligible_share_within_3px_known"] = float((topm & (kd <= 3)).sum() / max(topm.sum(), 1))
    out["pixel_km"] = _pix_km(transform)
    return out


def histogram(p: np.ndarray, valid: np.ndarray, known: np.ndarray, n_bins: int = 60) -> pd.DataFrame:
    """Logit-spaced histogram, separately on known-fault pixels, within 3 px of them, and elsewhere."""
    a = np.asarray(p, np.float32)
    kd = ndimage.distance_transform_edt(~known) if known.any() else np.full(a.shape, np.inf)
    edges = 1 / (1 + np.exp(-np.linspace(-12, 12, n_bins + 1)))
    rows = []
    for name, m in (("on_known", valid & known), ("near_known_1_3px", valid & (kd > 0) & (kd <= 3)),
                    ("far_gt_3px", valid & (kd > 3))):
        h, _ = np.histogram(np.clip(np.nan_to_num(a[m]), 1e-7, 1 - 1e-7), edges)
        for i in range(n_bins):
            rows.append({"group": name, "lo": edges[i], "hi": edges[i + 1], "count": int(h[i]),
                         "share": float(h[i] / max(m.sum(), 1))})
    return pd.DataFrame(rows)


def distance_curve(p: np.ndarray, valid: np.ndarray, known: np.ndarray, transform, bins_km=DIST_BINS_KM,
                   top_frac: float = 0.01) -> pd.DataFrame:
    """Probability as a function of distance to known faults (Phase D4 dependence curve)."""
    a = np.asarray(p, np.float32)
    d = ndimage.distance_transform_edt(~known) * _pix_km(transform) if known.any() else np.full(a.shape, np.inf)
    v = valid & np.isfinite(a)
    vals = np.where(v, a, -np.inf)
    k = max(int(np.ceil(top_frac * v.sum())), 1)
    thr = np.partition(vals[v], v.sum() - k)[v.sum() - k]
    top = v & (vals >= thr)
    rows = []
    for lo, hi in zip(bins_km[:-1], bins_km[1:]):
        m = v & (d >= lo) & (d < hi)
        rows.append({"dist_lo_km": lo, "dist_hi_km": hi, "px": int(m.sum()),
                     "mean_p": float(a[m].mean()) if m.any() else np.nan,
                     "median_p": float(np.median(a[m])) if m.any() else np.nan,
                     "p90": float(np.quantile(a[m], 0.9)) if m.any() else np.nan,
                     "share_of_mass": float(a[m].sum() / max(a[v].sum(), 1e-12)),
                     f"share_of_top{top_frac}": float((top & m).sum() / max(top.sum(), 1)),
                     "share_of_area": float(m.sum() / max(v.sum(), 1))})
    return pd.DataFrame(rows)


def component_stats(p: np.ndarray, valid: np.ndarray, known: np.ndarray, transform, frac: float = 0.01,
                    exclude_known: bool = True) -> dict:
    """Connected components of the top-``frac`` eligible pixels: count, length distribution, ridge thickness,
    fragmentation (components per 100 km of skeleton)."""
    from ..geology.skeleton import skeletonize

    a = np.where(valid, np.nan_to_num(np.asarray(p, np.float32)), -np.inf)
    dom = valid & (~known if exclude_known else np.ones_like(valid))
    vals = a[dom]
    k = max(int(np.ceil(frac * vals.size)), 1)
    thr = np.partition(vals, vals.size - k)[vals.size - k]
    m = dom & (a >= thr)
    lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    sk = skeletonize(m)
    km = _pix_km(transform)
    if n == 0:
        return {"frac": frac, "n_components": 0}
    idx = np.arange(1, n + 1)
    area = ndimage.sum(np.ones_like(lab), lab, idx)
    sklen = ndimage.sum(sk.astype(np.float32), lab, idx) * km
    return {"frac": frac, "n_components": int(n), "area_px": int(m.sum()), "skeleton_km": float(sk.sum() * km),
            "median_component_km": float(np.median(sklen)), "p90_component_km": float(np.quantile(sklen, 0.9)),
            "share_components_lt_1km": float((sklen < 1).mean()),
            "ridge_thickness_px": float(m.sum() / max(sk.sum(), 1)),
            "fragmentation_per_100km": float(n / max(sk.sum() * km, 1e-9) * 100),
            "largest_component_px": int(area.max())}


def spatial_concentration(p: np.ndarray, valid: np.ndarray, block_px: int = 100) -> dict:
    """Gini coefficient of probability mass over square blocks (0 = uniform, 1 = all mass in one block)."""
    a = np.where(valid, np.nan_to_num(np.asarray(p, np.float64)), 0.0)
    H, W = a.shape
    hh, ww = -(-H // block_px), -(-W // block_px)
    pad = np.zeros((hh * block_px, ww * block_px))
    pad[:H, :W] = a
    vv = np.zeros_like(pad, bool)
    vv[:H, :W] = valid
    m = pad.reshape(hh, block_px, ww, block_px).sum((1, 3)).ravel()
    n = vv.reshape(hh, block_px, ww, block_px).sum((1, 3)).ravel()
    m = np.sort(m[n > 0.5 * block_px * block_px])
    if m.sum() <= 0:
        return {"gini": np.nan, "n_blocks": int(m.size)}
    c = np.cumsum(m) / m.sum()
    lorenz_area = (c.sum() - 0.5 * c[-1]) / m.size          # trapezoid rule with c(0) = 0
    gini = 1 - 2 * lorenz_area
    return {"gini": float(gini), "n_blocks": int(m.size), "top10pct_blocks_mass_share": float(m[-max(1, m.size // 10):].sum() / m.sum())}


def regional_calibration(p: np.ndarray, valid: np.ndarray, regimes: np.ndarray | None) -> pd.DataFrame:
    """Mean probability and high-confidence area per label-free regime (regional calibration drift)."""
    if regimes is None:
        return pd.DataFrame()
    a = np.asarray(p, np.float32)
    rows = []
    for r in np.unique(regimes[valid]):
        m = valid & (regimes == r) & np.isfinite(a)
        rows.append({"regime": int(r), "px": int(m.sum()), "mean_p": float(a[m].mean()) if m.any() else np.nan,
                     "area_p_gt_0.5": float((a[m] > 0.5).mean()) if m.any() else np.nan,
                     "q99": float(np.quantile(a[m], 0.99)) if m.any() else np.nan})
    return pd.DataFrame(rows)


def truth_ranking(score_crop: np.ndarray, truth_crop: np.ndarray, scored_crop: np.ndarray, tol_px: int = 3,
                  max_n: int = 2_000_000, seed: int = 0) -> dict:
    """AUC and average precision of the RAW score for 'within tol_px of held-out truth' on one evaluation unit
    (measurement only). Shows whether the ranking carries signal that a representation might destroy."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if not truth_crop.any():
        return {"auc": np.nan, "ap": np.nan, "n": 0}
    near = ndimage.distance_transform_edt(~truth_crop) <= tol_px
    m = scored_crop & np.isfinite(score_crop)
    y, s = near[m], np.asarray(score_crop, np.float64)[m]
    if y.size > max_n:
        i = np.random.default_rng(seed).choice(y.size, max_n, replace=False)
        y, s = y[i], s[i]
    if y.all() or not y.any():
        return {"auc": np.nan, "ap": np.nan, "n": int(y.size)}
    return {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
            "base_rate": float(y.mean()), "n": int(y.size)}
