"""Axial (strike) orientation algebra and label-free orientation fields.

Conventions (identical to ``gems_eda.enhance.Enhancer.structure_tensor`` and ``rs_features.t_normal_contrast``)
------------------------------------------------------------------------------------------------------------
* strike ``θ`` in degrees clockwise from north, axial: θ and θ + 180° are the same fault direction.
* along-strike unit vector in (row, col) pixel space: ``(-cos θ, sin θ)`` (rows grow southwards);
  fault-normal unit vector: ``(sin θ, cos θ)``.
* the doubled-angle encoding ``(cos 2θ, sin 2θ)`` is continuous across the 0/180° wrap, so it is used for
  regression targets, averaging and error metrics; ``axial_angle`` inverts it.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def axial_vector(theta_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.radians(np.asarray(theta_deg, np.float64))
    return np.cos(2 * t).astype(np.float32), np.sin(2 * t).astype(np.float32)


def axial_angle(c2: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Strike in [0, 180) degrees from the doubled-angle vector."""
    return (np.degrees(0.5 * np.arctan2(s2, c2)) % 180.0).astype(np.float32)


def axial_difference(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Smallest angle between two axial directions, in [0, 90] degrees."""
    d = np.abs((np.asarray(a_deg) - np.asarray(b_deg)) % 180.0)
    return np.minimum(d, 180.0 - d)


def along_normal_offsets(theta_deg: np.ndarray, step_px: float):
    """(row, col) offsets of one ``step_px`` along strike and along the fault normal."""
    t = np.radians(np.nan_to_num(np.asarray(theta_deg, np.float64)))
    tr, tc = (-np.cos(t) * step_px).astype(np.float32), (np.sin(t) * step_px).astype(np.float32)
    nr, nc = (np.sin(t) * step_px).astype(np.float32), (np.cos(t) * step_px).astype(np.float32)
    return (tr, tc), (nr, nc)


def strike_field(field: np.ndarray, grid, scale_km: float):
    """Label-free (strike deg, coherence) of a measured field (structure tensor at ``scale_km``)."""
    from gems_eda.enhance import Enhancer

    st, coh, _ = Enhancer(grid).structure_tensor(np.asarray(field, np.float32), scale_km)
    return st, coh


def orientation_entropy(theta_deg: np.ndarray, weight: np.ndarray | None, window_px: int,
                        n_bins: int = 8) -> np.ndarray:
    """Normalised Shannon entropy (0 = one strike, 1 = isotropic) of weighted strikes in a moving window."""
    t = np.nan_to_num(np.asarray(theta_deg, np.float64)) % 180.0
    w = np.ones_like(t) if weight is None else np.nan_to_num(np.asarray(weight, np.float64))
    b = np.minimum((t / (180.0 / n_bins)).astype(int), n_bins - 1)
    hist = np.stack([ndimage.uniform_filter(np.where(b == i, w, 0.0), window_px, mode="nearest")
                     for i in range(n_bins)])
    p = hist / np.maximum(hist.sum(0, keepdims=True), 1e-12)
    h = -(p * np.log(np.maximum(p, 1e-12))).sum(0) / np.log(n_bins)
    return h.astype(np.float32)


def mean_axial(theta_deg: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Weighted circular mean of axial directions (degrees)."""
    c, s = axial_vector(theta_deg)
    w = np.ones_like(c) if weight is None else weight
    m = np.isfinite(c) & np.isfinite(w)
    return float(axial_angle(np.sum(c[m] * w[m]), np.sum(s[m] * w[m])))
