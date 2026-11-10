"""Feature distributions by label state on block-sampled pixels (one pixel per block).

Groups: fault, near (<= near band), sampled background, distant unlabeled (> 3 km), with DEM, without DEM.
Robust effect size = (median_a - median_b) / pooled MAD; overlap = histogram intersection (0 = separated,
1 = identical); missingness and outlier rate (|robust z| > 5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan


def robust_effect(a: np.ndarray, b: np.ndarray) -> float:
    if not a.size or not b.size:
        return np.nan
    s = np.sqrt((_mad(a) ** 2 + _mad(b) ** 2) / 2) + 1e-12
    return float((np.median(a) - np.median(b)) / s)


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    if not a.size or not b.size:
        return np.nan
    lo, hi = np.nanquantile(np.r_[a, b], [0.005, 0.995])
    if hi <= lo:
        return 1.0
    ha, e = np.histogram(np.clip(a, lo, hi), bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(np.clip(b, lo, hi), bins=bins, range=(lo, hi), density=True)
    return float(np.minimum(ha, hb).sum() * (e[1] - e[0]))


def group_masks(fault, dist_m, valid, dem_covered, near_m: float) -> dict:
    return {"fault": valid & fault, "near": valid & ~fault & (dist_m <= near_m),
            "background": valid & (dist_m > near_m), "distant_unlabeled": valid & (dist_m > 3000),
            "with_dem": valid & dem_covered, "without_dem": valid & ~dem_covered}


def distribution_table(store, names, groups: dict, block_px: int, seed: int, max_n: int) -> pd.DataFrame:
    from .spatial import block_sample

    samples = {g: block_sample(m, max(1, block_px // (4 if g in ("fault", "near") else 1)), seed, max_n)
               for g, m in groups.items() if m.any()}
    rows = []
    for n in names:
        a = store.get(n)
        vals = {g: np.asarray(a[r, c], np.float64) for g, (r, c, _) in samples.items()}
        for g, v in vals.items():
            fin = v[np.isfinite(v)]
            z = (fin - np.median(fin)) / (_mad(fin) + 1e-12) if fin.size else fin
            q = np.quantile(fin, [0.05, 0.25, 0.5, 0.75, 0.95]) if fin.size else [np.nan] * 5
            rows.append({"feature": n, "group": g, "n": int(v.size), "missing": float(1 - fin.size / max(v.size, 1)),
                         "mean": float(fin.mean()) if fin.size else np.nan, "std": float(fin.std()) if fin.size else np.nan,
                         "q05": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q95": q[4],
                         "outlier_rate": float((np.abs(z) > 5).mean()) if fin.size else np.nan})
        if "fault" in vals and "background" in vals:
            a1, b1 = vals["fault"][np.isfinite(vals["fault"])], vals["background"][np.isfinite(vals["background"])]
            rows.append({"feature": n, "group": "fault_vs_background", "effect_size": robust_effect(a1, b1),
                         "overlap": overlap_coefficient(a1, b1)})
    return pd.DataFrame(rows)
