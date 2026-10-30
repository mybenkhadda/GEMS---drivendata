"""Cross-section explorer: sample one or more layers along a straight line inside the evaluated AOI.

Coordinates are AOI pixel (row, col) floats; spacing in pixels (1 px = 100 m). Sampling is bilinear and NaN-aware
(a sample is NaN unless all four neighbours are valid). Known-fault crossings are DIAGNOSTIC markers only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


def line_coords(start, end, spacing_px=1.0):
    (r0, c0), (r1, c1) = start, end
    L = float(np.hypot(r1 - r0, c1 - c0))
    n = max(2, int(np.floor(L / float(spacing_px))) + 1)
    t = np.linspace(0, 1, n)
    return r0 + t * (r1 - r0), c0 + t * (c1 - c0), t * L


def sample(arr, rr, cc):
    a = np.asarray(arr, np.float32)
    ok = np.isfinite(a)
    v = map_coordinates(np.where(ok, a, 0), [rr, cc], order=1, mode="constant", cval=np.nan)
    w = map_coordinates(ok.astype(np.float32), [rr, cc], order=1, mode="constant", cval=0)
    return np.where(w > 0.999, v, np.nan)


def transect(layers: dict, start, end, spacing_px=1.0, pixel_m=100.0, faults=None) -> tuple[pd.DataFrame, list]:
    """(table: distance_km + one column per layer [+ fault_hit], fault crossing distances in km)."""
    rr, cc, d = line_coords(start, end, spacing_px)
    T = pd.DataFrame({"distance_km": d * pixel_m / 1000.0, "row": rr, "col": cc})
    for k, a in layers.items():
        T[k] = sample(a, rr, cc)
    marks = []
    if faults is not None:
        f = np.asarray(faults, bool)
        ri, ci = np.clip(np.round(rr).astype(int), 0, f.shape[0] - 1), np.clip(np.round(cc).astype(int), 0,
                                                                                 f.shape[1] - 1)
        hit = f[ri, ci]
        T["fault_hit"] = hit
        starts = np.flatnonzero(hit & ~np.r_[False, hit[:-1]])
        marks = T.distance_km.to_numpy()[starts].tolist()
    return T, marks


def display_normalize(T: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Robust z of each profile FOR DISPLAY ONLY (stored values are not changed)."""
    D = T.copy()
    for c in cols:
        v = D[c].to_numpy(float)
        med = np.nanmedian(v)
        mad = 1.4826 * np.nanmedian(np.abs(v - med))
        D[c] = (v - med) / mad if mad > 0 else np.nan
    return D
