"""Axial orientation statistics (doubled angles; 0-180 degrees, clockwise from north)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def axial_mean(az_deg, w=None) -> tuple[float, float]:
    a = np.radians(np.asarray(az_deg, float)) * 2
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    C, S = np.nansum(w * np.cos(a)), np.nansum(w * np.sin(a))
    R = float(np.hypot(C, S) / max(np.nansum(w), 1e-12))
    return float(np.degrees(np.arctan2(S, C) / 2) % 180.0), R


def rayleigh_axial(az_deg) -> float:
    """p-value of the Rayleigh test on doubled angles (uniform vs unimodal axial)."""
    n = np.isfinite(np.asarray(az_deg, float)).sum()
    _, R = axial_mean(az_deg)
    z = n * R * R
    return float(np.exp(-z) * (1 + (2 * z - z * z) / (4 * n) - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4)
                               / (288 * n * n))) if n else np.nan


def rose(az_deg, w=None, bins: int = 18) -> pd.DataFrame:
    e = np.linspace(0, 180, bins + 1)
    h, _ = np.histogram(np.asarray(az_deg, float) % 180, bins=e, weights=w)
    return pd.DataFrame({"az_lo": e[:-1], "az_hi": e[1:], "weight": h / max(h.sum(), 1e-12)})


def orientation_summary(groups: dict) -> pd.DataFrame:
    """``groups``: name -> (azimuths, weights or None)."""
    rows = []
    for k, (az, w) in groups.items():
        m, R = axial_mean(az, w)
        rows.append({"group": k, "n": int(np.isfinite(np.asarray(az, float)).sum()), "mean_strike_deg": m,
                     "resultant": R, "rayleigh_p": rayleigh_axial(az)})
    return pd.DataFrame(rows)
