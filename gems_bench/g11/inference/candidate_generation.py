"""Fault-candidate generators on a 100 m probability (or evidence) map -> LineString segments (metres).

Generators: threshold (quantile), hysteresis (low/high quantiles), ridge_nms (Hessian ridge strength at
``ridge_sigma_m`` + non-maximum suppression across the ridge), skeleton (thinned threshold mask). Every binary
result is skeletonised, split at junctions and traced into polylines; segments shorter than ``min_length_m`` are
dropped. Thresholds are quantiles of the map itself (label-free); they are never tuned on test labels.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _skeleton_segments(mask: np.ndarray, grid, min_length_m: float, generator: str, simplify_m: float = 50.0):
    from ..fault_labels import segments_from_raster

    g = segments_from_raster(mask, grid, min_length_m, simplify_m)
    if len(g):
        g["generator"] = generator
        g = g.drop(columns=[c for c in ("orig_fid", "source") if c in g.columns])
    return g


def threshold_mask(p: np.ndarray, valid: np.ndarray, q: float) -> np.ndarray:
    v = p[valid & np.isfinite(p)]
    return valid & (np.nan_to_num(p, nan=-np.inf) >= np.quantile(v, q)) if v.size else np.zeros_like(valid)


def hysteresis_mask(p: np.ndarray, valid: np.ndarray, q_lo: float, q_hi: float) -> np.ndarray:
    """8-connected hysteresis: keep components of p >= q_lo that contain a pixel >= q_hi.

    (skimage's ``apply_hysteresis_threshold`` labels with 4-connectivity, which fragments thin DIAGONAL traces.)
    """
    v = p[valid & np.isfinite(p)]
    if not v.size:
        return np.zeros_like(valid)
    lo, hi = np.quantile(v, [q_lo, q_hi])
    a = np.nan_to_num(p, nan=-np.inf)
    low = valid & (a >= lo)
    lab, n = ndimage.label(low, structure=np.ones((3, 3), bool))
    if not n:
        return low
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[low & (a >= hi)])] = True
    keep[0] = False
    return keep[lab]


def ridge_nms_mask(p: np.ndarray, valid: np.ndarray, sigma_px: float, q: float) -> np.ndarray:
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    a = np.nan_to_num(p, nan=0.0).astype(np.float64)
    H = hessian_matrix(a, sigma=sigma_px, order="rc", use_gaussian_derivatives=True)
    l1, l2 = hessian_matrix_eigvals(H)
    ridge = np.maximum(-np.minimum(l1, l2), 0)                   # bright line -> strongly negative eigenvalue
    hrr, hrc, hcc = H
    theta = 0.5 * np.arctan2(2 * hrc, hrr - hcc)                   # principal (across-ridge) direction
    ur, uc = np.cos(theta), np.sin(theta)
    rr, cc = np.indices(a.shape, dtype=np.float64)
    f = ndimage.map_coordinates(ridge, [rr + ur, cc + uc], order=1, mode="nearest")
    b = ndimage.map_coordinates(ridge, [rr - ur, cc - uc], order=1, mode="nearest")
    nms = (ridge >= f) & (ridge >= b)
    v = ridge[valid]
    thr = np.quantile(v, q) if v.size else np.inf
    return valid & nms & (ridge >= thr)


def generate_candidates(p: np.ndarray, valid: np.ndarray, grid, cfg: dict, source_model: str = "model"):
    """All configured generators; returns one GeoDataFrame with columns generator, source_model, geometry."""
    import geopandas as gpd
    import pandas as pd

    cg = cfg["candidate_generation"]
    sig = cg["ridge_sigma_m"] / grid.res[0]
    masks = {}
    for gname in cg["generators"]:
        if gname == "threshold":
            masks[gname] = threshold_mask(p, valid, cg["threshold_quantile"])
        elif gname == "hysteresis":
            masks[gname] = hysteresis_mask(p, valid, *cg["hysteresis_quantiles"])
        elif gname == "ridge_nms":
            masks[gname] = ridge_nms_mask(p, valid, sig, cg["threshold_quantile"])
        elif gname == "skeleton":
            from skimage.morphology import skeletonize

            masks[gname] = skeletonize(threshold_mask(p, valid, cg["threshold_quantile"]))
        else:
            raise ValueError(f"unknown generator {gname}")
    parts = []
    for gname, m in masks.items():
        g = _skeleton_segments(m, grid, cg["min_length_m"], gname)
        if len(g):
            g["source_model"] = source_model
            parts.append(g)
    if not parts:
        return gpd.GeoDataFrame({"generator": [], "source_model": []}, geometry=[], crs=grid.crs)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=grid.crs)
