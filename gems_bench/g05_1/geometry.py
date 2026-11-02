"""Candidate geometry (registry family D3, graph endpoints, displacement profiles). Pure numpy / scipy.

A candidate is a set of skeleton pixels (one label of the GeoMoE inventory raster). Its trace is the pixel-graph
diameter (8-connectivity, two BFS passes), so lengths are measured along the trace (1 or sqrt(2) px per step), never
as a pixel count. Strike is axial, degrees clockwise from grid north (row axis points south), in [0, 180).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy import ndimage

PX_KM = 0.1
NB8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
GEOMETRY_COLUMNS = ("n_px", "chain_length_km", "endpoint_dist_km", "tortuosity", "linearity", "strike_deg",
                    "strike_dispersion_deg", "curvature_deg_per_km", "line_r2", "width_px", "n_branch_px",
                    "n_endpoints", "end_r0", "end_c0", "end_r1", "end_c1", "dir0_deg", "dir1_deg")


def segment_pixels(lab: np.ndarray) -> dict:
    """{label: (rows, cols)} in crop coordinates (labels > 0)."""
    out = {}
    for i, o in enumerate(ndimage.find_objects(np.asarray(lab)), start=1):
        if o is None:
            continue
        rr, cc = np.nonzero(lab[o] == i)
        out[i] = (rr + o[0].start, cc + o[1].start)
    return out


def _neighbours(rr, cc) -> list[list[int]]:
    idx = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(rr, cc))}
    return [[idx[(int(r) + a, int(c) + b)] for a, b in NB8 if (int(r) + a, int(c) + b) in idx]
            for r, c in zip(rr, cc)]


def _bfs(nb, start: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(nb)
    dist = np.full(n, -1, int)
    par = np.full(n, -1, int)
    dist[start] = 0
    q = deque([start])
    while q:
        k = q.popleft()
        for j in nb[k]:
            if dist[j] < 0:
                dist[j], par[j] = dist[k] + 1, k
                q.append(j)
    return dist, par


def path_order(rr, cc) -> np.ndarray:
    """Indices (into rr / cc) of the longest shortest path of the pixel graph (trace order, end to end). A
    disconnected pixel set uses its largest component."""
    rr, cc = np.asarray(rr), np.asarray(cc)
    if len(rr) <= 1:
        return np.arange(len(rr))
    nb = _neighbours(rr, cc)
    d0, _ = _bfs(nb, 0)
    reach = np.flatnonzero(d0 >= 0)
    if len(reach) < len(rr) / 2:                      # start in the larger component
        alt = int(np.flatnonzero(d0 < 0)[0])
        d0, _ = _bfs(nb, alt)
    a = int(np.argmax(d0))
    da, par = _bfs(nb, a)
    b = int(np.argmax(da))
    path = [b]
    while path[-1] != a:
        path.append(int(par[path[-1]]))
    return np.asarray(path[::-1])


def chain_length_px(rr, cc, order) -> float:
    if len(order) < 2:
        return float(len(order))
    d = np.hypot(np.diff(np.asarray(rr)[order]), np.diff(np.asarray(cc)[order]))
    return float(d.sum())


def azimuth_deg(dr, dc) -> np.ndarray:
    """Axial strike in [0, 180): clockwise from north (row axis points south)."""
    return np.degrees(np.arctan2(np.asarray(dc, float), -np.asarray(dr, float))) % 180.0


def axial_dispersion_deg(theta_deg) -> float:
    """Circular std of axial angles (doubled-angle method), degrees in [0, 90]."""
    t = np.radians(2.0 * np.asarray(theta_deg, float))
    if not t.size:
        return float("nan")
    R = float(np.hypot(np.cos(t).mean(), np.sin(t).mean()))
    return float(np.degrees(np.sqrt(max(-2.0 * np.log(max(R, 1e-12)), 0.0)) / 2.0))


def axial_diff_deg(a, b) -> np.ndarray:
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 180.0
    return np.minimum(d, 180.0 - d)


def end_direction_deg(rr, cc, order, at_end: bool, tail_px: int = 5) -> float:
    """Outward direction (full circle, degrees clockwise from north) at one end of the trace: from the point
    ``tail_px`` inside the trace to the end pixel."""
    o = order[::-1] if at_end else order
    if len(o) < 2:
        return float("nan")
    k = min(tail_px, len(o) - 1)
    dr = float(rr[o[0]] - rr[o[k]])
    dc = float(cc[o[0]] - cc[o[k]])
    return float(np.degrees(np.arctan2(dc, -dr)) % 360.0)


def features(rr, cc, px_km: float = PX_KM, local_px: int = 3) -> dict:
    """Label-free geometry of one candidate (see GEOMETRY_COLUMNS)."""
    rr, cc = np.asarray(rr, float), np.asarray(cc, float)
    n = len(rr)
    nan = float("nan")
    out = dict.fromkeys(GEOMETRY_COLUMNS, nan)
    out["n_px"] = n
    if n == 0:
        return out
    order = path_order(rr.astype(int), cc.astype(int))
    L = chain_length_px(rr, cc, order)
    e0, e1 = order[0], order[-1]
    de = float(np.hypot(rr[e1] - rr[e0], cc[e1] - cc[e0]))
    out.update(chain_length_km=L * px_km, endpoint_dist_km=de * px_km,
               tortuosity=(L / de) if de > 0 else nan, width_px=n / max(L, 1.0),
               end_r0=rr[e0], end_c0=cc[e0], end_r1=rr[e1], end_c1=cc[e1])
    if n >= 2:
        X = np.c_[rr - rr.mean(), cc - cc.mean()]
        w, V = np.linalg.eigh(X.T @ X / n)
        l1, l2 = float(max(w[1], 0.0)), float(max(w[0], 0.0))
        out["linearity"] = l1 / (l1 + l2) if (l1 + l2) > 0 else nan
        out["line_r2"] = 1.0 - l2 / l1 if l1 > 0 else nan
        out["strike_deg"] = float(azimuth_deg(V[0, 1], V[1, 1]))
    if len(order) > local_px:
        pr, pc = rr[order], cc[order]
        dr, dc = pr[local_px:] - pr[:-local_px], pc[local_px:] - pc[:-local_px]
        loc = azimuth_deg(dr, dc)
        out["strike_dispersion_deg"] = axial_dispersion_deg(loc)
        turn = axial_diff_deg(loc[1:], loc[:-1])
        out["curvature_deg_per_km"] = float(turn.sum() / max(L * px_km, 1e-9))
    nb = _neighbours(rr.astype(int), cc.astype(int))
    deg = np.asarray([len(x) for x in nb])
    out["n_branch_px"] = int((deg >= 3).sum())
    out["n_endpoints"] = int((deg <= 1).sum())
    out["dir0_deg"] = end_direction_deg(rr, cc, order, at_end=False)
    out["dir1_deg"] = end_direction_deg(rr, cc, order, at_end=True)
    return out


def geometry_table(lab: np.ndarray, labels=None, px_km: float = PX_KM) -> pd.DataFrame:
    """One row per label (crop coordinates for the endpoint columns)."""
    P = segment_pixels(lab)
    keys = sorted(P) if labels is None else [int(x) for x in labels]
    rows = []
    for k in keys:
        rr, cc = P.get(k, (np.zeros(0, int), np.zeros(0, int)))
        rows.append({"label": k, **features(rr, cc, px_km)})
    return pd.DataFrame(rows, columns=["label", *GEOMETRY_COLUMNS])


def survey_artifact(strike_deg, axes_deg, tol_deg: float) -> np.ndarray:
    """True where the strike lies within ``tol_deg`` of a survey / grid axis (artifact indicator, never a label)."""
    s = np.asarray(strike_deg, float)
    hit = np.zeros(s.shape, bool)
    for a in axes_deg:
        hit |= axial_diff_deg(s, float(a)) <= float(tol_deg)
    return hit & np.isfinite(s)
