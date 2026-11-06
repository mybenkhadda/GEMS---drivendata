"""Regional terrain feature bank from the 100 m elevation band (``terrain.elevation_band``).

Scales s in ``terrain.scales_m`` (300, 1000, 2000, 3000 m): Gaussian operators with sigma = s / 2. At 100 m the
smallest resolvable structure is a few hundred metres - nothing here resolves metre-scale scarps (that is the
high-resolution DEM bank). Features are NaN within ``edge_guard_sigma`` x sigma of the footprint edge.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import structure_tensor as st
from .feature_registry import AXIAL


def build(gd, footprint: np.ndarray, cfg, writer) -> dict:
    t = cfg["terrain"]
    px = gd.grid.pixel_m
    band = t["elevation_band"]
    z_raw = gd.read(band)
    z, ok = st.filled(z_raw, footprint)
    dist = ndimage.distance_transform_edt(footprint) * px
    gsig = float(t["edge_guard_sigma"])
    parent = f"geodawn:{band}"
    smooth = {}

    def put(fid, arr, unit, op, meaning, scale=None, guard_m=0.0, orient=""):
        return writer.put(fid, arr, family="terrain", parent=parent, unit=unit, scale_m=scale, operator=op,
                          meaning=meaning, mask=footprint & (dist > guard_m), orientation=orient)

    put("terrain_elevation", z_raw, "m", "raw band", f"{band} as delivered")
    if t["slope_band"] in gd.names:
        put("terrain_slope_delivered", gd.read(t["slope_band"]), "as delivered", "raw band",
            f"{t['slope_band']} as delivered (operator undocumented)")
    for s in t["scales_m"]:
        sg = s / 2.0 / px
        g = gsig * s / 2.0
        zs = ndimage.gaussian_filter(z, sg)
        smooth[s] = zs
        zx = ndimage.gaussian_filter(z, sg, order=(0, 1)) / px
        zy = -ndimage.gaussian_filter(z, sg, order=(1, 0)) / px
        zxx = ndimage.gaussian_filter(z, sg, order=(0, 2)) / px ** 2
        zyy = ndimage.gaussian_filter(z, sg, order=(2, 0)) / px ** 2
        zxy = -ndimage.gaussian_filter(z, sg, order=(1, 1)) / px ** 2
        p2 = zx ** 2 + zy ** 2
        slope = np.degrees(np.arctan(np.sqrt(p2)))
        put(f"terrain_slope_{s}m", slope, "degree", f"gaussian gradient, sigma {s / 2:g} m", "slope", s, g)
        prof = -(zxx * zx ** 2 + 2 * zxy * zx * zy + zyy * zy ** 2) / ((p2 + 1e-12) * (1 + p2) ** 1.5)
        plan = -(zxx * zy ** 2 - 2 * zxy * zx * zy + zyy * zx ** 2) / ((p2 + 1e-12) ** 1.5 + 1e-12)
        put(f"terrain_profile_curvature_{s}m", prof, "1/m", "profile curvature (Zevenbergen-Thorne form)",
            "curvature along the slope direction (slope breaks)", s, g)
        put(f"terrain_plan_curvature_{s}m", np.clip(plan, -1, 1), "1/m", "plan (contour) curvature, clipped +-1",
            "convergence / divergence across the slope", s, g)
        half = np.sqrt(((zxx - zyy) / 2) ** 2 + zxy ** 2)
        lmin, lmax = (zxx + zyy) / 2 - half, (zxx + zyy) / 2 + half
        put(f"terrain_ridge_{s}m", np.maximum(-lmin, 0), "1/m", "max(-lambda_min of the Hessian, 0)",
            "ridge strength (convex-up crests)", s, g)
        put(f"terrain_valley_{s}m", np.maximum(lmax, 0), "1/m", "max(lambda_max of the Hessian, 0)",
            "valley strength (concave linear lows)", s, g)
        put(f"terrain_tpi_{s}m", z - zs, "m", f"z - gaussian mean (sigma {s / 2:g} m)", "topographic position index", s, g)
        n = max(int(round(s / px)), 3)
        put(f"terrain_relief_{s}m", ndimage.maximum_filter(z, n) - ndimage.minimum_filter(z, n), "m",
            f"max - min in {n} x {n} px", "local relief", s, s / 2.0)
        res = z - zs
        put(f"terrain_roughness_{s}m", np.sqrt(ndimage.gaussian_filter(res * res, sg)), "m",
            "local std of z - smooth(z)", "terrain roughness", s, 2 * g)
        sl = ndimage.gaussian_gradient_magnitude(slope, max(sg / 2, 0.5)) / px
        put(f"terrain_slope_break_{s}m", sl, "degree/m", "|grad(slope)|", "multiscale slope break", s, 1.5 * g)
        d = st.describe(*st.tensor(zx, zy, max(sg, 0.5)))
        put(f"terrain_coherence_{s}m", d["coherence"], "1", "structure tensor", "topographic gradient coherence", s, 2 * g)
        put(f"terrain_strike_c2_{s}m", d["c2"], "1", "structure tensor", "cos 2a terrain lineament strike", s, 2 * g,
            AXIAL)
        put(f"terrain_strike_s2_{s}m", d["s2"], "1", "structure tensor", "sin 2a terrain lineament strike", s, 2 * g,
            AXIAL)
        if s == 1000:
            put("terrain_dz_east_1km", zx, "m/m", "gaussian d/dEast, sigma 500 m", "directional gradient east", s, g)
            put("terrain_dz_north_1km", zy, "m/m", "gaussian d/dNorth, sigma 500 m", "directional gradient north", s, g)
    sc = list(t["scales_m"])
    for a, b in zip(sc[:-1], sc[1:]):
        put(f"terrain_dog_{a}_{b}m", smooth[a] - smooth[b], "m", f"G(sigma {a / 2:g}) - G(sigma {b / 2:g})",
            f"band-pass relief between {a} and {b} m", b, gsig * b / 2.0)
    return {"scales": sc}
