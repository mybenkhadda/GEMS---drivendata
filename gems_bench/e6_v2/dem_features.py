"""2 m DEM operators. Every operator is a Gaussian (derivative) filter with ``sigma = scale / 2`` and an explicit
support (``guard_m``); a 2 m value is usable only where it is farther than its guard from missing data and
farther than ``dem.seam_guard_m`` from a seam between DEM projects.

Angles live in array coordinates (x = column, y = row). Orientation is axial and stored as the doubled-angle
vector of the structure STRIKE, (cos 2t, sin 2t): a 90-degree rotation negates both components, a flip of
either axis negates sin 2t - the rule ``augmentation.py`` applies.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .fft_features import nearest_fill

N_AZIMUTH = 8


def compute(z: np.ndarray, res: float, cfg: dict) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """{name: 2 m array} and {name: guard_m}. ``z`` NaN = missing (filled by nearest value, then guarded)."""
    d = cfg["dem"]
    ok = np.isfinite(z)
    zf = nearest_fill(np.where(ok, z, 0.0), ok).astype(np.float32) if not ok.all() else z.astype(np.float32)
    gs = d["gradient_sigma_m"] / res
    g0 = d["edge_guard_sigma"] * d["gradient_sigma_m"]
    out, guard = {}, {}

    gx = ndimage.gaussian_filter(zf, gs, order=(0, 1)) / res      # d/dcol, m/m
    gy = ndimage.gaussian_filter(zf, gs, order=(1, 0)) / res      # d/drow
    out["slope"] = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    guard["slope"] = g0

    # 8-azimuth directional derivatives -> edge energy / anisotropy at the 5 m structure scale
    s5 = d["structure_scales_m"][0] / 2.0
    e = []
    for i in range(N_AZIMUTH):
        a = np.pi * i / N_AZIMUTH
        e.append(ndimage.gaussian_filter(np.abs(gx * np.cos(a) + gy * np.sin(a)), s5 / res))
    e = np.stack(e)
    emax, emin = e.max(0), e.min(0)
    out["edge_energy"] = e.mean(0).astype(np.float32)
    out["edge_anisotropy"] = ((emax - emin) / (emax + emin + 1e-6)).astype(np.float32)
    guard["edge_energy"] = guard["edge_anisotropy"] = g0 + d["edge_guard_sigma"] * s5
    del e, emax, emin

    # structure tensor at each scale: coherence; strike orientation at the largest scale
    jxx_l = jxy_l = jyy_l = None
    for s in d["structure_scales_m"]:
        si = s / 2.0 / res
        jxx = ndimage.gaussian_filter(gx * gx, si)
        jxy = ndimage.gaussian_filter(gx * gy, si)
        jyy = ndimage.gaussian_filter(gy * gy, si)
        rr = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
        out[f"coherence_{s}m"] = ((rr / (jxx + jyy + 1e-12)) ** 2).astype(np.float32)
        guard[f"coherence_{s}m"] = g0 + d["edge_guard_sigma"] * s / 2.0
        jxx_l, jxy_l, jyy_l, rr_l, s_l = jxx, jxy, jyy, rr, s
    inv = 1.0 / np.maximum(rr_l, 1e-12)
    out["strike_cos2"] = (-(jxx_l - jyy_l) * inv).astype(np.float32)
    out["strike_sin2"] = (-2.0 * jxy_l * inv).astype(np.float32)
    guard["strike_cos2"] = guard["strike_sin2"] = guard[f"coherence_{s_l}m"]
    phi = 0.5 * np.arctan2(2 * jxy_l, jxx_l - jyy_l)                 # dominant gradient axis
    del jxx_l, jxy_l, jyy_l, rr_l, inv

    # scarp: height step across the structure over +- scarp_offset along the gradient axis
    zs = ndimage.gaussian_filter(zf, gs)
    o = d["scarp_offset_m"] / res
    rr_i, cc_i = np.indices(zf.shape, dtype=np.float32)
    dr, dc = (o * np.sin(phi)).astype(np.float32), (o * np.cos(phi)).astype(np.float32)
    up = ndimage.map_coordinates(zs, [rr_i + dr, cc_i + dc], order=1, mode="nearest")
    dn = ndimage.map_coordinates(zs, [rr_i - dr, cc_i - dc], order=1, mode="nearest")
    out["scarp"] = np.abs(up - dn).reshape(zf.shape).astype(np.float32)
    guard["scarp"] = guard["strike_cos2"] + d["scarp_offset_m"] + g0
    del rr_i, cc_i, dr, dc, up, dn, phi, zs, gx, gy

    # curvature at curvature_sigma: principal curvatures and the second derivative along the gradient
    sc = d["curvature_sigma_m"] / res
    zx = ndimage.gaussian_filter(zf, sc, order=(0, 1)) / res
    zy = ndimage.gaussian_filter(zf, sc, order=(1, 0)) / res
    zxx = ndimage.gaussian_filter(zf, sc, order=(0, 2)) / res ** 2
    zyy = ndimage.gaussian_filter(zf, sc, order=(2, 0)) / res ** 2
    zxy = ndimage.gaussian_filter(zf, sc, order=(1, 1)) / res ** 2
    half = np.sqrt(((zxx - zyy) / 2) ** 2 + zxy ** 2)
    mean = (zxx + zyy) / 2
    out["kmin"], out["kmax"] = (mean - half).astype(np.float32), (mean + half).astype(np.float32)
    kp = (zxx * zx ** 2 + 2 * zxy * zx * zy + zyy * zy ** 2) / (zx ** 2 + zy ** 2 + 1e-12)
    out["crest"] = np.maximum(-kp, 0).astype(np.float32)          # convex break: top of a scarp
    out["toe"] = np.maximum(kp, 0).astype(np.float32)             # concave break: foot of a scarp
    for k in ("kmin", "kmax", "crest", "toe"):
        guard[k] = d["edge_guard_sigma"] * d["curvature_sigma_m"]
    del zx, zy, zxx, zyy, zxy, half, mean, kp

    for r in d["tpi_radii_m"]:
        out[f"tpi_{r}m"] = (zf - ndimage.gaussian_filter(zf, r / 2.0 / res)).astype(np.float32)
        guard[f"tpi_{r}m"] = d["edge_guard_sigma"] * r / 2.0

    sr = d["roughness_sigma_m"] / res
    resid = zf - ndimage.gaussian_filter(zf, sr)
    out["roughness"] = np.sqrt(ndimage.gaussian_filter(resid * resid, sr)).astype(np.float32)
    guard["roughness"] = 2 * d["edge_guard_sigma"] * d["roughness_sigma_m"]
    del resid

    s_small = d["structure_scales_m"][0]
    out["edge"] = ((out["slope"] >= d["edge_slope_deg"]) & (out[f"coherence_{s_small}m"] >= d["edge_coherence"])
                   ).astype(np.float32)
    guard["edge"] = max(guard["slope"], guard[f"coherence_{s_small}m"])
    return out, guard
