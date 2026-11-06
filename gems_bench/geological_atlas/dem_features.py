"""Resolution-appropriate DEM operators on a work-resolution window (metres, NaN = no data).

Common to both modes (scales from the mode's parameters): slope, profile / plan curvature, TPI at two radii,
local relief, directional edge responses (4 azimuths FAST, 8 DETAILED) -> edge energy / anisotropy, structure
tensor coherence at two scales and the axial strike at the larger one, scarp step across the strike, ridge /
valley strength (Hessian), roughness. DETAILED only: crest / toe responses (fine slope breaks). FAST computes
NO fine-scale product. Every operator has an explicit guard (its support); values closer than the guard to
missing data are excluded downstream, so missing DEM is never treated as zero elevation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .fft_operators import nearest_fill


def compute(z: np.ndarray, res: float, p: dict, detailed: bool) -> tuple[dict, dict]:
    ok = np.isfinite(z)
    zf = (nearest_fill(np.where(ok, z, 0.0), ok) if not ok.all() else z).astype(np.float32)
    k = float(p["edge_guard_sigma"])
    gs = p["gradient_sigma_m"] / res
    g0 = k * p["gradient_sigma_m"]
    F, G = {}, {}
    gE = ndimage.gaussian_filter(zf, gs, order=(0, 1)) / res
    gN = -ndimage.gaussian_filter(zf, gs, order=(1, 0)) / res
    F["slope"] = np.degrees(np.arctan(np.hypot(gE, gN))).astype(np.float32)
    G["slope"] = g0
    s_small, s_large = p["structure_scales_m"]
    n_az = 8 if detailed else 4
    e = np.stack([ndimage.gaussian_filter(np.abs(gE * np.sin(a) + gN * np.cos(a)), s_small / 2 / res)
                  for a in np.pi * np.arange(n_az) / n_az])          # derivative along azimuth a (from north)
    emax, emin = e.max(0), e.min(0)
    F["edge"] = e.mean(0).astype(np.float32)
    F["edge_anisotropy"] = ((emax - emin) / (emax + emin + 1e-9)).astype(np.float32)
    G["edge"] = G["edge_anisotropy"] = g0 + k * s_small / 2
    del e, emax, emin
    for s in (s_small, s_large):
        si = s / 2 / res
        jee, jen, jnn = (ndimage.gaussian_filter(x, si) for x in (gE * gE, gE * gN, gN * gN))
        r = np.sqrt((jee - jnn) ** 2 + 4 * jen ** 2)
        F[f"coherence_{s}m"] = ((r / (jee + jnn + 1e-12)) ** 2).astype(np.float32)
        G[f"coherence_{s}m"] = g0 + k * s / 2
        if s == s_large:
            phi = 0.5 * np.arctan2(2 * jen, jee - jnn)               # dominant gradient direction (ccw from east)
            a = np.mod(-phi, np.pi)                                  # strike azimuth clockwise from north
            F["strike_c2"], F["strike_s2"] = np.cos(2 * a).astype(np.float32), np.sin(2 * a).astype(np.float32)
            G["strike_c2"] = G["strike_s2"] = G[f"coherence_{s}m"]
    zs = ndimage.gaussian_filter(zf, gs)
    o = p["scarp_offset_m"] / res
    rr, cc = np.indices(zf.shape, dtype=np.float32)
    dr = (-o * np.sin(phi)).astype(np.float32)                     # row offset (rows run south)
    dc = (o * np.cos(phi)).astype(np.float32)
    up = ndimage.map_coordinates(zs, [rr + dr, cc + dc], order=1, mode="nearest")
    dn = ndimage.map_coordinates(zs, [rr - dr, cc - dc], order=1, mode="nearest")
    F["scarp"] = np.abs(up - dn).astype(np.float32)
    G["scarp"] = G["strike_c2"] + p["scarp_offset_m"] + g0
    del rr, cc, dr, dc, up, dn, phi, gE, gN
    sc = p["curvature_sigma_m"] / res
    zx = ndimage.gaussian_filter(zf, sc, order=(0, 1)) / res
    zy = -ndimage.gaussian_filter(zf, sc, order=(1, 0)) / res
    zxx = ndimage.gaussian_filter(zf, sc, order=(0, 2)) / res ** 2
    zyy = ndimage.gaussian_filter(zf, sc, order=(2, 0)) / res ** 2
    zxy = -ndimage.gaussian_filter(zf, sc, order=(1, 1)) / res ** 2
    p2 = zx ** 2 + zy ** 2
    F["profile_curvature"] = (-(zxx * zx ** 2 + 2 * zxy * zx * zy + zyy * zy ** 2) / ((p2 + 1e-12) * (1 + p2) ** 1.5)
                              ).astype(np.float32)
    F["plan_curvature"] = np.clip(-(zxx * zy ** 2 - 2 * zxy * zx * zy + zyy * zx ** 2) / ((p2 + 1e-12) ** 1.5), -1, 1
                                  ).astype(np.float32)
    half = np.sqrt(((zxx - zyy) / 2) ** 2 + zxy ** 2)
    F["ridge"] = np.maximum(-((zxx + zyy) / 2 - half), 0).astype(np.float32)
    F["valley"] = np.maximum((zxx + zyy) / 2 + half, 0).astype(np.float32)
    for n in ("profile_curvature", "plan_curvature", "ridge", "valley"):
        G[n] = k * p["curvature_sigma_m"]
    if detailed:
        kp = (zxx * zx ** 2 + 2 * zxy * zx * zy + zyy * zy ** 2) / (p2 + 1e-12)
        F["crest"] = np.maximum(-kp, 0).astype(np.float32)
        F["toe"] = np.maximum(kp, 0).astype(np.float32)
        G["crest"] = G["toe"] = k * p["curvature_sigma_m"]
        del kp
    del zx, zy, zxx, zyy, zxy, p2, half
    for r_ in p["tpi_radii_m"]:
        F[f"tpi_{r_}m"] = (zf - ndimage.gaussian_filter(zf, r_ / 2 / res)).astype(np.float32)
        G[f"tpi_{r_}m"] = k * r_ / 2
    n = max(int(round(p["relief_window_m"] / res)), 3)
    F["relief"] = (ndimage.maximum_filter(zf, n) - ndimage.minimum_filter(zf, n)).astype(np.float32)
    G["relief"] = p["relief_window_m"] / 2
    sr = p["roughness_sigma_m"] / res
    resid = zf - ndimage.gaussian_filter(zf, sr)
    F["roughness"] = np.sqrt(ndimage.gaussian_filter(resid * resid, sr)).astype(np.float32)
    G["roughness"] = 2 * k * p["roughness_sigma_m"]
    F["edge_pixel"] = ((F["slope"] >= p["edge_slope_deg"]) & (F[f"coherence_{s_small}m"] >= p["edge_coherence"])
                       ).astype(np.float32)
    G["edge_pixel"] = max(G["slope"], G[f"coherence_{s_small}m"])
    F["elevation"] = np.where(ok, z, np.nan).astype(np.float32)
    G["elevation"] = 0.0
    return F, G
