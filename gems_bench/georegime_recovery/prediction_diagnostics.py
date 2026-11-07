"""Section 03 - confidence distribution, concentration and nodata reliability of the V1 full-model raster.

Population: eligible pixels outside mapped faults (``bank.valid & ~bank.labels``), the pixels the official
scorer can penalise; mapped-fault pixels are reported separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute(prob: np.ndarray, valid: np.ndarray, labels: np.ndarray, cfg: dict) -> dict:
    elig = valid & ~labels
    fin = np.isfinite(prob)
    p = np.clip(prob[elig & fin], 0, 1).astype(np.float64)
    pm = np.clip(prob[valid & labels & fin], 0, 1)
    total = float(p.sum())
    s = np.sort(p)[::-1]
    cum = np.cumsum(s)
    n = len(s)

    def share_top(f):
        k = max(int(round(f * n)), 1)
        return float(cum[k - 1] / total) if total > 0 else np.nan

    gini = float(1 - 2 * np.sum(np.cumsum(np.sort(p)) / max(total, 1e-12)) / n + 1 / n) if n and total > 0 else np.nan
    q = np.quantile(p, [0.5, 0.9, 0.99, 0.999]) if n else [np.nan] * 4
    rows = [
        ("eligible_pixels_outside_mapped", int(elig.sum()), "px", "valid & ~mapped"),
        ("finite_eligible_pixels", int(n), "px", ""),
        ("nan_inside_eligible", int((elig & ~fin).sum()), "px", "should be 0"),
        ("finite_outside_valid", int((~valid & fin).sum()), "px", "should be 0 (V1 writes NaN outside)"),
        ("out_of_range_values", int(((prob < 0) | (prob > 1))[fin].sum()), "px", "should be 0"),
        ("total_mass_outside_mapped", total, "sum p", "probability mass the FP term can penalise"),
        ("mass_on_mapped_pixels", float(pm.sum()), "sum p", "excluded pixel-exactly by the official convention"),
        ("mean_confidence", float(p.mean()) if n else np.nan, "p", ""),
        ("median", float(q[0]), "p", ""), ("q90", float(q[1]), "p", ""), ("q99", float(q[2]), "p", ""),
        ("q999", float(q[3]), "p", ""), ("max", float(s[0]) if n else np.nan, "p", ""),
        ("mean_on_mapped_pixels", float(pm.mean()) if pm.size else np.nan, "p", ""),
        ("share_mass_top_1pct_pixels", share_top(0.01), "fraction", "Lorenz"),
        ("share_mass_top_10pct_pixels", share_top(0.10), "fraction", "Lorenz"),
        ("gini", gini, "", "0 = uniform, 1 = all mass on one pixel"),
        ("effective_area_fraction", float(total ** 2 / max(np.sum(p ** 2), 1e-12) / n) if n else np.nan, "fraction",
         "(sum p)^2 / (n sum p^2): participation ratio"),
    ]
    summary = pd.DataFrame(rows, columns=["quantity", "value", "unit", "note"])
    thr = []
    for t in cfg["diagnostics"]["thresholds"]:
        above = p >= t
        thr.append({"threshold": t, "eligible_fraction_above": float(above.mean()) if n else np.nan,
                    "mass_fraction_above": float(p[above].sum() / total) if total > 0 else np.nan,
                    "n_pixels_above": int(above.sum())})
    fr = np.unique(np.r_[np.logspace(-5, 0, 60), [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]])
    lorenz = pd.DataFrame({"top_pixel_fraction": fr, "mass_fraction": [share_top(f) for f in fr]})
    edges = np.r_[0, np.logspace(-6, 0, 61)]
    counts, _ = np.histogram(p, bins=edges)
    hist = pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "count": counts})
    ex_t = np.logspace(-6, 0, 80)
    asc = s[::-1]
    exceed = pd.DataFrame({"threshold": ex_t, "eligible_fraction_above":
                           (n - np.searchsorted(asc, ex_t, side="left")) / n if n else np.full(len(ex_t), np.nan)})
    stats = {r[0]: r[1] for r in rows}
    return {"summary": summary, "thresholds": pd.DataFrame(thr), "lorenz": lorenz, "histogram": hist,
            "exceedance": exceed, "stats": stats}
