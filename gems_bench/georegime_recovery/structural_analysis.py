"""Section 12 - performance by held-fault length, linearity, strike agreement and region.

* held components (8-connected, as in V1) get PCA length, linearity l1/(l1+l2), axial strike and weighted
  recall = sum M(g) / n over their pixels (full-area discovery protocol, primary exclusion);
* local strike agreement: structure-tensor strike of the smoothed held trace vs of the smoothed prediction
  at held pixels with support M >= ``recall_missed`` (axial difference 0-90 degrees);
* regional recall per 10 km block with coverage-aware denominators (blocks with fewer than
  ``min_truth_per_block`` truth pixels are not reported).
Strike convention: azimuth clockwise from north, axial (0-180 degrees).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .split_audit import _component_geometry


def tensor_strike(img: np.ndarray, sigma_grad: float = 1.0, sigma_int: float = 2.0):
    """Axial strike azimuth (deg, clockwise from north) and coherence of linear features in ``img``."""
    s = ndimage.gaussian_filter(np.nan_to_num(img, nan=0.0).astype(np.float32), sigma_grad)
    gr, gc = np.gradient(s)
    ge, gn = gc, -gr                                       # east, north components (rows point south)
    jee = ndimage.gaussian_filter(ge * ge, sigma_int)
    jnn = ndimage.gaussian_filter(gn * gn, sigma_int)
    jen = ndimage.gaussian_filter(ge * gn, sigma_int)
    theta = 0.5 * np.arctan2(2 * jen, jee - jnn)            # dominant gradient direction, math angle from east
    coh = np.hypot(jee - jnn, 2 * jen) / (jee + jnn + 1e-12)
    strike_math = theta + np.pi / 2                          # features run perpendicular to the gradient
    az = (90.0 - np.degrees(strike_math)) % 180.0
    return az.astype(np.float32), coh.astype(np.float32)


def axial_diff(a, b):
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def components_table(unit, U, maps, transform, cfg) -> pd.DataFrame:
    D = cfg["diagnostics"]
    G = maps["truth"]
    lab, n = ndimage.label(G, structure=np.ones((3, 3), np.uint8))
    if not n:
        return pd.DataFrame()
    ids = np.arange(1, n + 1)
    tp = ndimage.sum(maps["tp_support"], lab, index=ids)
    npx = ndimage.sum(G.astype(float), lab, index=ids)
    geo = _component_geometry(lab, ids, transform)
    s, m_ = D["length_classes_km"]
    rows = []
    for i in ids:
        g = geo.get(int(i), {})
        L = g.get("length_km", np.nan)
        cls = "short (<%g km)" % s if L < s else ("medium (%g-%g km)" % (s, m_) if L < m_ else "long (>=%g km)" % m_)
        rows.append({"unit": unit, "component": int(i), "n_px": int(npx[i - 1]), "length_km": L, "length_class": cls,
                     "linearity": g.get("linearity"), "strike_deg": g.get("strike_deg"),
                     "recall_w": float(tp[i - 1] / max(npx[i - 1], 1)), "tp_w": float(tp[i - 1]),
                     "centroid_e": g.get("centroid_e"), "centroid_n": g.get("centroid_n")})
    return pd.DataFrame(rows)


def class_summary(comp: pd.DataFrame, by: str) -> pd.DataFrame:
    if comp.empty:
        return pd.DataFrame()
    g = comp.groupby(["unit", by], observed=True)
    out = g.agg(n_components=("component", "size"), n_truth=("n_px", "sum"), tp_w=("tp_w", "sum")).reset_index()
    out["recall_w"] = out.tp_w / out.n_truth.clip(lower=1)
    pooled = comp.groupby(by, observed=True).agg(n_components=("component", "size"), n_truth=("n_px", "sum"),
                                                 tp_w=("tp_w", "sum")).reset_index()
    pooled["unit"] = "pooled"
    pooled["recall_w"] = pooled.tp_w / pooled.n_truth.clip(lower=1)
    return pd.concat([out, pooled], ignore_index=True)


def strike_agreement(unit, prob, maps, cfg) -> tuple[pd.DataFrame, dict]:
    G = maps["truth"]
    az_t, coh_t = tensor_strike(G.astype(np.float32))
    az_p, coh_p = tensor_strike(np.clip(np.nan_to_num(prob, nan=0.0), 0, 1))
    m = G & (maps["tp_support"] >= cfg["diagnostics"]["recall_missed"]) & (coh_t > 0.2) & (coh_p > 0.2)
    d = axial_diff(az_t[m], az_p[m])
    edges = np.arange(0, 91, 10)
    h, _ = np.histogram(d, bins=edges)
    tab = pd.DataFrame({"unit": unit, "diff_lo_deg": edges[:-1], "diff_hi_deg": edges[1:], "n_px": h})
    st = {"unit": unit, "n_px": int(m.sum()), "median_diff_deg": float(np.median(d)) if d.size else np.nan,
          "share_within_20deg": float((d <= 20).mean()) if d.size else np.nan,
          "mapped_strikes": az_t[G & (coh_t > 0.2)], "predicted_strikes": az_p[m]}
    return tab, st


def regional_recall(unit, maps, valid, block_px, min_truth) -> pd.DataFrame:
    h, w = valid.shape
    nbc = (w + block_px - 1) // block_px
    ri, ci = np.indices(valid.shape, dtype=np.int32)
    bid = ((ri // block_px) * nbc + ci // block_px).ravel()
    nb = int(bid.max()) + 1
    n = np.bincount(bid, weights=maps["truth"].ravel().astype(float), minlength=nb)
    tp = np.bincount(bid, weights=maps["tp_support"].ravel(), minlength=nb)
    fp = np.bincount(bid, weights=maps["fp_mass"].ravel(), minlength=nb)
    vf = np.bincount(bid, weights=valid.ravel().astype(float), minlength=nb) / (block_px * block_px)
    b = np.arange(nb)
    df = pd.DataFrame({"unit": unit, "block": b, "row0": (b // nbc) * block_px, "col0": (b % nbc) * block_px,
                       "n_truth": n, "tp_w": tp, "fp_w": fp, "valid_fraction": vf})
    df["recall_w"] = np.where(n >= min_truth, tp / np.maximum(n, 1), np.nan)
    return df[(df.n_truth > 0) | (df.fp_w > 0)].reset_index(drop=True)
