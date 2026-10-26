"""Per-candidate descriptors (one row per fused candidate of a unit) + distance-aware labels.

Geometry: length, chord, sinuosity, strike (as cos 2θ / sin 2θ), curvature, pixel count.
Provenance: kind, number of fused sources, cross-modal agreement (``agree_*``, ``n_agree``).
Known faults (the unit's *visible* faults only: disc = retained, full = all mapped, geo = none -> NaN): median /
minimum distance and distance from the candidate's ends to the nearest known endpoint.
Evidence along the line: per-unit rank of each member's honest score (mean / max / p90); mean |z| of every
candidate feature layer; orientation consistency of magnetic / gravity edges with the candidate strike;
fault-normal asymmetry of elevation (|mean z(+n) - mean z(-n)| at 300 m).

Labels (``label``): 1 if mean k(dist to scored truth) >= ``positive_kd``; 0 if the median distance is >=
``negative_dist_km``; otherwise -1 = unlabeled (mapped faults are incomplete; ambiguous rows are not negatives).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .candidates import CandidateSet, geometry_table
from .official import Unit
from .representation import rank01

ID_COLS = ("cand_id", "task", "zone", "source", "kind", "sources", "label", "kd_mean", "kd_max", "near300",
           "dist_median_km", "r_mid", "c_mid", "r_full", "c_full", "strike_deg", "score_evidence", "score_lgbm")
# kd_mean / kd_max / near300 / dist_median_km read the unit's truth: labels only, never features


def _along(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.asarray(a)[p[:, 0], p[:, 1]]


def edge_strike(layer: np.ndarray) -> np.ndarray:
    """Strike (azimuth deg from north, clockwise) of the edges of a layer: perpendicular to its gradient."""
    a = np.nan_to_num(layer, nan=0.0)
    gr, gc = np.gradient(ndimage.gaussian_filter(a, 1.0))
    grad_az = np.degrees(np.arctan2(gc, -gr))                # gradient azimuth (north = -row)
    return ((grad_az + 90.0) % 180.0).astype(np.float32)


def describe(u: Unit, cs: CandidateSet, *, scores: dict, layers: dict, elevation: np.ndarray | None,
             truth_stats: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    from ..geology.skeleton import endpoints, skeletonize

    if not len(cs):
        return pd.DataFrame()
    df = pd.concat([cs.table.reset_index(drop=True), geometry_table(cs)], axis=1)
    df["task"], df["zone"] = u.task, u.zone
    df["r_full"], df["c_full"] = df.r_mid + u.sl[0].start, df.c_mid + u.sl[1].start
    th = np.radians(2 * df.strike_deg.to_numpy())
    df["strike_cos2"], df["strike_sin2"] = np.cos(th), np.sin(th)
    for k in ("feature", "model", "tip"):
        df[f"is_{k}"] = (df.kind == k).astype(int)
    paths = cs.paths
    if u.known is not None and u.known.any():
        dk = ndimage.distance_transform_edt(~u.known, sampling=(0.1, 0.1))
        ep = endpoints(skeletonize(u.known))
        de = ndimage.distance_transform_edt(~ep, sampling=(0.1, 0.1)) if ep.any() else np.full(u.known.shape, np.inf)
        df["dist_known_median_km"] = [float(np.median(_along(dk, p))) for p in paths]
        df["dist_known_min_km"] = [float(_along(dk, p).min()) for p in paths]
        df["dist_known_endpoint_km"] = [float(min(de[p[0, 0], p[0, 1]], de[p[-1, 0], p[-1, 1]])) for p in paths]
    else:
        df["dist_known_median_km"] = df["dist_known_min_km"] = df["dist_known_endpoint_km"] = np.nan
    for name, s in scores.items():
        r = rank01(s, u.ev)
        vals = [_along(r, p) for p in paths]
        df[f"{name}_mean"] = [float(v.mean()) for v in vals]
        df[f"{name}_max"] = [float(v.max()) for v in vals]
        df[f"{name}_p90"] = [float(np.quantile(v, 0.9)) for v in vals]
    for name, a in layers.items():
        df[f"ev_{name}"] = [float(np.nanmean(np.abs(_along(a, p)))) for p in paths]
    for name in [n for n in layers if n.startswith(("mag", "grav"))][:2]:
        st = edge_strike(layers[name])
        cand = df.strike_deg.to_numpy()
        df[f"orient_consistency_{name}"] = [
            float(np.abs(np.mean(np.cos(np.radians(2 * (_along(st, p) - c)))))) for p, c in zip(paths, cand)]
    if elevation is not None:
        az = np.radians(df.strike_deg.to_numpy())
        h, w = elevation.shape
        asym = []
        for p, a in zip(paths, az):
            n = np.array([np.sin(a), np.cos(a)]) * 3.0      # (row, col) normal at 300 m
            vals = []
            for sgn in (1, -1):
                q = np.rint(p + sgn * n).astype(int)
                q[:, 0], q[:, 1] = np.clip(q[:, 0], 0, h - 1), np.clip(q[:, 1], 0, w - 1)
                vals.append(np.nanmean(_along(elevation, q)))
            asym.append(float(abs(vals[0] - vals[1])))
        df["normal_asymmetry_elev"] = asym
    if truth_stats is not None and len(truth_stats) == len(df):
        rc = cfg["ranker"]
        ts = truth_stats.reset_index(drop=True)
        for c in ("kd_mean", "kd_max", "near300", "dist_median_km"):
            df[c] = ts[c].to_numpy()
        df["label"] = np.where(df.kd_mean >= rc["positive_kd"], 1,
                               np.where(df.dist_median_km >= rc["negative_dist_km"], 0, -1))
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]
