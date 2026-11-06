"""Structure tensor, axial orientation and multiscale edge persistence for any 2-D field.

Gradients are geographic: gE = d/d(east), gN = d/d(north) = -d/d(row). The tensor J = G_s * [gE^2, gE gN, gN^2]
gives coherence ((l1 - l2) / (l1 + l2))^2, anisotropy (l1 - l2) / (l1 + l2), energy l1 + l2 and the dominant
gradient direction phi (counter-clockwise from east). The STRUCTURE (strike) azimuth, clockwise from north, is
a = -phi (mod 180 deg); it is stored as the axial vector (cos 2a, sin 2a) so that 0 and 180 deg are identical.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .fft_operators import nearest_fill


def filled(z: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    v = np.isfinite(z) if valid is None else (valid & np.isfinite(z))
    return (nearest_fill(np.where(v, z, 0.0), v) if not v.all() else z.astype(np.float64)), v


def gradients(z: np.ndarray, px: float, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    gE = ndimage.gaussian_filter(z, sigma_px, order=(0, 1)) / px
    gN = -ndimage.gaussian_filter(z, sigma_px, order=(1, 0)) / px
    return gE, gN


def tensor(gE, gN, sigma_px: float):
    return (ndimage.gaussian_filter(gE * gE, sigma_px), ndimage.gaussian_filter(gE * gN, sigma_px),
            ndimage.gaussian_filter(gN * gN, sigma_px))


def describe(jee, jen, jnn) -> dict:
    r = np.sqrt((jee - jnn) ** 2 + 4 * jen ** 2)
    tr = jee + jnn
    aniso = r / (tr + 1e-30)
    phi = 0.5 * np.arctan2(2 * jen, jee - jnn)
    a = np.mod(-phi, np.pi)                       # strike azimuth (rad, clockwise from north)
    return {"coherence": (aniso ** 2).astype(np.float32), "anisotropy": aniso.astype(np.float32),
            "energy": tr.astype(np.float32), "strike_deg": np.degrees(a).astype(np.float32),
            "c2": np.cos(2 * a).astype(np.float32), "s2": np.sin(2 * a).astype(np.float32)}


def structure(z: np.ndarray, px: float, grad_sigma_m: float, scale_m: float, valid=None) -> dict:
    zf, _ = filled(z, valid)
    gE, gN = gradients(zf, px, max(grad_sigma_m / px, 0.5))
    return describe(*tensor(gE, gN, max(scale_m / 2.0 / px, 0.5)))


def orientation_consistency(c2s: list, s2s: list, weights: list) -> np.ndarray:
    """Weighted resultant length of per-scale axial vectors (1 = the same strike at every scale)."""
    C = sum(w * c for w, c in zip(weights, c2s))
    S = sum(w * s for w, s in zip(weights, s2s))
    W = sum(weights)
    return (np.sqrt(C ** 2 + S ** 2) / np.maximum(W, 1e-12)).astype(np.float32)


def persistence(ranks: list[np.ndarray], thr: float) -> np.ndarray:
    """Share of scales at which a pixel's edge strength rank is >= ``thr`` (label-free, per-scale ranks)."""
    st = np.stack(ranks)
    fin = np.isfinite(st)
    hit = np.where(fin, st >= thr, False).sum(0)
    n = fin.sum(0)
    return np.where(n > 0, hit / np.maximum(n, 1), np.nan).astype(np.float32)


def cross_scale_stability(ranks: list[np.ndarray]) -> np.ndarray:
    """Mean over consecutive scale pairs of min(rank_a, rank_b): high only where an edge survives scale changes."""
    vals = [np.fmin(a, b) for a, b in zip(ranks[:-1], ranks[1:])]
    return np.nanmean(np.stack(vals), 0).astype(np.float32) if vals else ranks[0]


def edge_density(edge: np.ndarray, valid: np.ndarray, window_px: int) -> np.ndarray:
    e = ndimage.uniform_filter(np.where(valid, edge, 0).astype(np.float32), window_px)
    n = ndimage.uniform_filter(valid.astype(np.float32), window_px)
    return np.where(n > 0.2, e / np.maximum(n, 1e-6), np.nan).astype(np.float32)
