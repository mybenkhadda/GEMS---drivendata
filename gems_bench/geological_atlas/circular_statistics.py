"""Axial (180-degree periodic) statistics for strikes. Angles in degrees clockwise from north, in [0, 180).

All averages use doubled angles: mean strike = 0.5 atan2(sum w sin 2a, sum w cos 2a), resultant length
R = |sum w e^{2ia}| / sum w (1 = all parallel, 0 = no preferred strike). Never an arithmetic mean of strikes.
"""
from __future__ import annotations

import numpy as np


def to_axial(az_deg) -> np.ndarray:
    return np.mod(np.asarray(az_deg, float), 180.0)


def vectors(az_deg) -> tuple[np.ndarray, np.ndarray]:
    a = np.radians(to_axial(az_deg))
    return np.cos(2 * a), np.sin(2 * a)


def from_vectors(c2, s2) -> np.ndarray:
    return np.mod(np.degrees(0.5 * np.arctan2(s2, c2)), 180.0)


def axial_mean(az_deg, w=None) -> tuple[float, float]:
    c, s = vectors(az_deg)
    w = np.ones_like(c) if w is None else np.asarray(w, float)
    ok = np.isfinite(c) & np.isfinite(w)
    if not ok.any() or w[ok].sum() <= 0:
        return float("nan"), float("nan")
    C, S, W = (w[ok] * c[ok]).sum(), (w[ok] * s[ok]).sum(), w[ok].sum()
    return float(from_vectors(C, S)), float(np.hypot(C, S) / W)


def axial_difference(a_deg, b_deg) -> np.ndarray:
    """Smallest angle between two axial directions, in [0, 90]."""
    d = np.abs(to_axial(a_deg) - to_axial(b_deg))
    return np.minimum(d, 180.0 - d)


def agreement(c2a, s2a, c2b, s2b) -> np.ndarray:
    """cos(2 (a - b)) for unit axial vectors: +1 parallel, -1 perpendicular, 0 at 45 degrees."""
    return (c2a * c2b + s2a * s2b).astype(np.float32)


def rayleigh_axial(az_deg, w=None) -> dict:
    """Rayleigh test on doubled angles (unweighted n for the p-value, weights for R)."""
    az = to_axial(az_deg)
    az = az[np.isfinite(az)]
    n = az.size
    if n < 2:
        return {"n": n, "R": np.nan, "z": np.nan, "p": np.nan}
    _, R = axial_mean(az, None if w is None else np.asarray(w)[np.isfinite(to_axial(az_deg))])
    z = n * R ** 2
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n)))
    return {"n": int(n), "R": float(R), "z": float(z), "p": min(max(p, 0.0), 1.0)}


def rose(az_deg, w=None, bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """(bin edges over 0..180, weighted counts normalised to sum 1)."""
    az = to_axial(az_deg)
    w = np.ones_like(az) if w is None else np.asarray(w, float)
    ok = np.isfinite(az) & np.isfinite(w)
    edges = np.linspace(0, 180, bins + 1)
    h, _ = np.histogram(az[ok], bins=edges, weights=w[ok])
    return edges, h / max(h.sum(), 1e-12)


def circular_std_deg(R: float) -> float:
    """Axial circular standard deviation (degrees) from the doubled-angle resultant."""
    if not np.isfinite(R) or R <= 0:
        return float("nan")
    return float(np.degrees(np.sqrt(-2 * np.log(R))) / 2)
