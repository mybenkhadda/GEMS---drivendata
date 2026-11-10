"""Native-resolution terrain morphology (mask-aware, physical scales in metres).

Conventions (rows point south, columns east; spacing ``res`` m):
* gradients ``zx = dz/dEast``, ``zy = dz/dNorth`` (m/m); slope = atan(|grad|) in degrees;
* aspect = azimuth (deg clockwise from north) of the DOWNSLOPE direction;
* curvatures (Zevenbergen-Thorne / Evans form, 1/m, p = zx, q = zy, r = zxx, t = zyy, s = zxy):
  profile = -(r p^2 + 2 s p q + t q^2) / ((p^2+q^2) (1+p^2+q^2)^1.5)  (negative = convex in the slope direction),
  plan = -(t p^2 - 2 s p q + r q^2) / (p^2+q^2)^1.5,  mean H, Gaussian K;
* smoothing is normalised convolution over valid pixels (``g06.feature_physics.masked_smooth``); outputs are NaN
  within ``edge_guard_sigma * sigma`` of nodata, so a nodata edge cannot create a fault-like gradient.
A scale s is computed only if s >= ``min_scale_res_ratio * res`` and the kernel fits in the window.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from gems_bench.g06.feature_physics import masked_smooth


@dataclass
class TerrainFeatureCube:
    arrays: dict
    meta: pd.DataFrame
    valid: np.ndarray
    res: float
    transform: tuple | None = None
    crs: str | None = None
    skipped_scales: list = field(default_factory=list)

    def __getitem__(self, k):
        return self.arrays[k]


def supported_scales(scales_m, res: float, shape, min_ratio: float = 2.0, truncate: float = 4.0) -> tuple[list, list]:
    ok, skipped = [], []
    for s in scales_m:
        sig = s / res / 2.0
        if s < min_ratio * res:
            skipped.append((s, f"below {min_ratio} x resolution"))
        elif truncate * sig * 2 + 1 > min(shape):
            skipped.append((s, "kernel larger than the window"))
        else:
            ok.append(float(s))
    return ok, skipped


def _guard(valid: np.ndarray, sigma_px: float, k: float) -> np.ndarray:
    r = int(np.ceil(k * max(sigma_px, 1.0)))
    return ndimage.binary_erosion(valid, iterations=r, border_value=0) if r > 0 else valid


def derivatives_m(S: np.ndarray, res: float) -> dict:
    dr, dc = np.gradient(S, res)
    zx, zy = dc, -dr
    zxr, zxx = np.gradient(zx, res)
    zyr, _ = np.gradient(zy, res)
    return {"zx": zx, "zy": zy, "zxx": zxx, "zyy": -zyr, "zxy": -zxr}


def slope_aspect(d: dict) -> tuple[np.ndarray, np.ndarray]:
    g = np.hypot(d["zx"], d["zy"])
    slope = np.degrees(np.arctan(g))
    aspect = np.degrees(np.arctan2(-d["zx"], -d["zy"])) % 360.0
    return slope, aspect


def hillshade(d: dict, azimuth: float, altitude: float) -> np.ndarray:
    az, alt = np.radians(azimuth), np.radians(altitude)
    slope = np.arctan(np.hypot(d["zx"], d["zy"]))
    aspect = np.arctan2(-d["zx"], -d["zy"])
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def curvatures(d: dict) -> dict:
    p, q, r, t, s = d["zx"], d["zy"], d["zxx"], d["zyy"], d["zxy"]
    g2 = p * p + q * q
    w = 1 + g2
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = -(r * p * p + 2 * s * p * q + t * q * q) / (np.maximum(g2, 1e-12) * w ** 1.5)
        plan = -(t * p * p - 2 * s * p * q + r * q * q) / np.maximum(g2, 1e-12) ** 1.5
    prof = np.where(g2 > 1e-10, prof, 0.0)
    plan = np.where(g2 > 1e-10, plan, 0.0)
    mean = ((1 + q * q) * r - 2 * p * q * s + (1 + p * p) * t) / (2 * w ** 1.5)
    gauss = (r * t - s * s) / (w * w)
    tr, disc = r + t, np.sqrt(np.maximum(0.25 * (r - t) ** 2 + s * s, 0))
    return {"profile_curvature": prof, "plan_curvature": plan, "mean_curvature": mean, "gaussian_curvature": gauss,
            "hessian_l1": 0.5 * tr + disc, "hessian_l2": 0.5 * tr - disc}


def masked_extreme(z: np.ndarray, valid: np.ndarray, size: int, min_support: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    zmax = ndimage.maximum_filter(np.where(valid, z, -np.inf), size=size, mode="constant", cval=-np.inf)
    zmin = ndimage.minimum_filter(np.where(valid, z, np.inf), size=size, mode="constant", cval=np.inf)
    sup = ndimage.uniform_filter(valid.astype(np.float32), size=size, mode="constant")
    bad = sup < min_support
    return np.where(bad, np.nan, zmax), np.where(bad, np.nan, zmin)


def structure_orientation(d: dict, sigma_int_px: float) -> dict:
    """Structure tensor of (zx, zy): coherence and LINEAMENT strike (perpendicular to the dominant gradient),
    stored as the axial vector (cos 2a, sin 2a) with a = azimuth clockwise from north."""
    gx, gy = np.nan_to_num(d["zx"]), np.nan_to_num(d["zy"])
    jxx = ndimage.gaussian_filter(gx * gx, sigma_int_px)
    jyy = ndimage.gaussian_filter(gy * gy, sigma_int_px)
    jxy = ndimage.gaussian_filter(gx * gy, sigma_int_px)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)          # gradient direction, math angle from east
    coh = np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-12)
    az = np.radians((90.0 - np.degrees(theta + np.pi / 2)) % 180.0)
    return {"coherence": coh, "strike_c2": np.cos(2 * az), "strike_s2": np.sin(2 * az), "strike_az": az}


def nms_edge(g: np.ndarray, zx: np.ndarray, zy: np.ndarray) -> np.ndarray:
    """1 where |grad| is a local maximum across the gradient direction (thin scarp crest)."""
    n = np.hypot(zx, zy) + 1e-12
    ur, uc = -zy / n, zx / n                               # (row, col) step along the gradient
    rr, cc = np.indices(g.shape, dtype=np.float32)
    h = np.where(np.isfinite(g), g, -1.0)
    f = ndimage.map_coordinates(h, [rr + ur, cc + uc], order=1, mode="nearest")
    b = ndimage.map_coordinates(h, [rr - ur, cc - uc], order=1, mode="nearest")
    return ((h >= f) & (h >= b) & (h > 0)).astype(np.float32)


def along_strike_mean(a: np.ndarray, az: np.ndarray, half_px: int) -> np.ndarray:
    dr, dc = -np.cos(az), np.sin(az)
    rr, cc = np.indices(a.shape, dtype=np.float32)
    A = np.nan_to_num(a, nan=0.0).astype(np.float32)
    acc = np.zeros_like(A)
    for k in range(-half_px, half_px + 1):
        acc += ndimage.map_coordinates(A, [rr + k * dr, cc + k * dc], order=1, mode="constant", cval=0.0)
    return acc / (2 * half_px + 1)


def compute_terrain_features(z: np.ndarray, valid: np.ndarray, res: float, scales_m, cfg: dict,
                             transform=None, crs=None, families=None) -> TerrainFeatureCube:
    """All terrain features of one DEM window.

    ``z`` elevation (m, NaN = nodata), ``valid`` its mask, ``res`` pixel size (m). Names are
    ``<feature>_<scale>m``; ``meta`` columns: name, family, scale_m, unit.
    ``families`` limits the computation (first_order, second_order, structural, hillshade).
    """
    tc = cfg["feature_engineering"]["terrain"]
    z = np.where(valid, np.asarray(z, np.float64), np.nan)
    fam = set(families or ("first_order", "second_order", "structural", "hillshade"))
    scales, skipped = supported_scales(scales_m, res, z.shape, tc["min_scale_res_ratio"])
    out, meta = {}, []

    def put(name, arr, family, scale, unit, guard):
        out[name] = np.where(guard, arr, np.nan).astype(np.float32)
        meta.append({"name": name, "family": family, "scale_m": scale, "unit": unit})

    persist_c, persist_s, persist_w = 0.0, 0.0, 0.0
    for s in scales:
        sig = s / res / 2.0
        S, _ = masked_smooth(z, sig, 0.5)
        d = derivatives_m(S, res)
        gd = _guard(valid, sig, tc["edge_guard_sigma"]) & np.isfinite(S)
        k = f"{int(s)}m"
        if "first_order" in fam:
            slope, aspect = slope_aspect(d)
            put(f"slope_{k}", slope, "first_order", s, "deg", gd)
            put(f"aspect_c_{k}", np.cos(np.radians(aspect)), "first_order", s, "1", gd)
            put(f"aspect_s_{k}", np.sin(np.radians(aspect)), "first_order", s, "1", gd)
            put(f"gradient_{k}", np.hypot(d["zx"], d["zy"]), "first_order", s, "m/m", gd)
            size = max(3, int(round(s / res)) | 1)
            zmax, zmin = masked_extreme(z, valid, size)
            put(f"relief_{k}", zmax - zmin, "first_order", s, "m", gd)
            put(f"tpi_{k}", z - S, "first_order", s, "m", gd)
            R2, _ = masked_smooth((z - S) ** 2, sig, 0.5)
            put(f"roughness_{k}", np.sqrt(np.maximum(R2, 0)), "first_order", s, "m", gd)
            TRI, _ = masked_smooth(np.abs(z - S), sig, 0.5)
            put(f"tri_{k}", TRI, "first_order", s, "m", gd)
            sl_d = derivatives_m(slope, res)
            put(f"slope_break_{k}", np.hypot(sl_d["zx"], sl_d["zy"]), "first_order", s, "deg/m", gd)
        if "second_order" in fam:
            for n, a in curvatures(d).items():
                put(f"{n}_{k}", a, "second_order", s, "1/m", gd)
        if "structural" in fam:
            c = curvatures(d)
            put(f"ridge_{k}", np.maximum(-c["hessian_l2"], 0), "structural", s, "1/m", gd)
            put(f"valley_{k}", np.maximum(c["hessian_l1"], 0), "structural", s, "1/m", gd)
            so = structure_orientation(d, 2.0 * sig)
            put(f"coherence_{k}", so["coherence"], "structural", s, "1", gd)
            put(f"strike_c2_{k}", so["strike_c2"], "structural", s, "1", gd)
            put(f"strike_s2_{k}", so["strike_s2"], "structural", s, "1", gd)
            g = np.hypot(d["zx"], d["zy"])
            edge = nms_edge(g, d["zx"], d["zy"])
            slope_n = np.clip(np.degrees(np.arctan(g)) / 30.0, 0, 1)
            scarp = edge * slope_n * so["coherence"]
            put(f"scarp_likelihood_{k}", scarp, "structural", s, "1", gd)
            put(f"linearity_{k}", so["coherence"] * edge, "structural", s, "1", gd)
            put(f"lineament_continuity_{k}", along_strike_mean(scarp, so["strike_az"], max(2, int(round(2 * s / res)))),
                "structural", s, "1", gd)
            w = np.nan_to_num(so["coherence"])
            persist_c = persist_c + w * so["strike_c2"]
            persist_s = persist_s + w * so["strike_s2"]
            persist_w = persist_w + w
        if "hillshade" in fam and s == min(scales):
            for azi in tc["hillshade_azimuths"]:
                put(f"hillshade_{int(azi)}_{k}", hillshade(d, azi, tc["hillshade_altitude"]), "hillshade", s, "1", gd)
    if "structural" in fam and scales:
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.hypot(persist_c, persist_s) / np.maximum(persist_w, 1e-12)
        gd = _guard(valid, max(scales) / res / 2.0, tc["edge_guard_sigma"])
        put("orientation_persistence", R, "structural", float(max(scales)), "1", gd)
    return TerrainFeatureCube(out, pd.DataFrame(meta), valid, res, transform, crs, skipped)
