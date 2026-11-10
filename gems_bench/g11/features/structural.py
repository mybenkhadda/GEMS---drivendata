"""Cross-fault elevation profiles (priority feature family).

For a segment, stations are placed every ``spacing_m`` along it; at each station the local tangent (central
difference of the line) gives a normal; elevations are sampled at signed ``offsets_m`` along the normal (positive =
right-hand side looking along the segment). Per station a robust line (Theil-Sen) is fitted to each side
(|offset| > ``inner_m``); the step is the difference of the two side fits at offset 0, the slope contrast the
difference of their slopes. Uncertainty = MAD of residuals; along-strike consistency = |median step| / MAD(step).

Absence of a scarp is NOT evidence against a fault (many faults have no surface expression); the features are
returned with ``profile_valid_fraction`` and are NaN when the DEM does not cover the station.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProfileSet:
    offsets_m: np.ndarray            # (n_off,)
    elevations: np.ndarray           # (n_station, n_off) metres, NaN = no DEM
    stations_xy: np.ndarray          # (n_station, 2)
    normals: np.ndarray              # (n_station, 2) unit normal (dx, dy)
    tile_ids: list


def _stations(line, spacing_m: float):
    L = line.length
    n = max(int(L // spacing_m) + 1, 2)
    ts = np.linspace(0, L, n)
    pts = np.array([line.interpolate(t).coords[0] for t in ts])
    eps = min(spacing_m / 2, L / 4)
    a = np.array([line.interpolate(max(t - eps, 0)).coords[0] for t in ts])
    b = np.array([line.interpolate(min(t + eps, L)).coords[0] for t in ts])
    tan = b - a
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
    normal = np.c_[tan[:, 1], -tan[:, 0]]            # right-hand normal
    return pts, normal


def sample_cross_fault_profiles(segment, dem_reader, offsets_m, spacing_m: float, crs, resolution: float = 1.0
                                ) -> ProfileSet:
    """Read one DEM window around the segment (halo = max |offset|) and bilinearly sample the profiles."""
    from scipy import ndimage

    offs = np.asarray(offsets_m, float)
    pts, nrm = _stations(segment, spacing_m)
    halo = float(np.max(np.abs(offs))) + 5 * resolution
    minx, miny, maxx, maxy = segment.bounds
    win = dem_reader.read_dem_window((minx, miny, maxx, maxy), crs, resolution=resolution, halo_m=halo)
    a, _, c, _, e, f = win.transform
    X = pts[:, None, 0] + offs[None] * nrm[:, None, 0]
    Y = pts[:, None, 1] + offs[None] * nrm[:, None, 1]
    col, row = (X - c) / a - 0.5, (Y - f) / e - 0.5
    z = win.elevation
    zz = ndimage.map_coordinates(np.nan_to_num(z, nan=0.0), [row, col], order=1, mode="constant", cval=np.nan)
    vv = ndimage.map_coordinates(win.valid.astype(np.float32), [row, col], order=1, mode="constant", cval=0.0)
    zz = np.where(vv > 0.99, zz, np.nan)
    return ProfileSet(offs, zz.astype(np.float32), pts, nrm, list(win.source_tiles))


def theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return np.nan, np.nan
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    m = dx != 0
    if not m.any():
        return 0.0, float(np.median(y))
    slope = float(np.median((y[j] - y[i])[m] / dx[m]))
    return slope, float(np.median(y - slope * x))


def extract_cross_fault_features(p: ProfileSet, inner_m: float = 10.0) -> dict:
    """Robust cross-fault statistics over all stations (see module docstring for definitions)."""
    o = p.offsets_m
    L, R = o < -inner_m, o > inner_m
    steps, contrasts, asym, widths, resid = [], [], [], [], []
    valid_station = []
    for z in p.elevations:
        okL, okR = np.isfinite(z[L]).sum() >= 2, np.isfinite(z[R]).sum() >= 2
        valid_station.append(okL and okR)
        if not (okL and okR):
            continue
        mL, bL = theil_sen(o[L], z[L])
        mR, bR = theil_sen(o[R], z[R])
        steps.append(bR - bL)
        contrasts.append(np.degrees(np.arctan(mR)) - np.degrees(np.arctan(mL)))
        rl = z[L] - (mL * o[L] + bL)
        rr = z[R] - (mR * o[R] + bR)
        r = np.r_[rl[np.isfinite(rl)], rr[np.isfinite(rr)]]
        resid.append(np.median(np.abs(r)) if r.size else np.nan)
        asym.append((np.nanstd(z[R]) - np.nanstd(z[L])) / (np.nanstd(z[R]) + np.nanstd(z[L]) + 1e-6))
        # scarp width: span of offsets where the profile departs from both side trends by > 25 % of |step|
        full = np.where(o < 0, mL * o + bL, mR * o + bR)
        dev = np.abs(z - full)
        thr = 0.25 * abs(bR - bL)
        w = o[np.isfinite(dev) & (dev > thr)]
        widths.append(float(np.ptp(w)) if w.size >= 2 and thr > 0 else 0.0)
    vf = float(np.mean(valid_station)) if valid_station else 0.0
    if not steps:
        return {"elevation_step_m": np.nan, "slope_contrast": np.nan, "profile_asymmetry": np.nan,
                "scarp_width_m": np.nan, "scarp_strength": np.nan, "along_strike_consistency": np.nan,
                "cross_fault_variability": np.nan, "profile_valid_fraction": vf, "n_stations": len(p.elevations)}
    steps = np.asarray(steps)
    mad = float(np.median(np.abs(steps - np.median(steps)))) + 1e-6
    noise = float(np.nanmedian(resid)) + 1e-6
    return {"elevation_step_m": float(np.median(steps)), "slope_contrast": float(np.median(contrasts)),
            "profile_asymmetry": float(np.median(asym)), "scarp_width_m": float(np.median(widths)),
            "scarp_strength": float(abs(np.median(steps)) / noise),
            "along_strike_consistency": float(abs(np.median(steps)) / mad),
            "cross_fault_variability": noise, "profile_valid_fraction": vf, "n_stations": len(p.elevations)}
