"""Magnetic feature family from the RAW reduced-to-pole band (``potential.magnetic_band`` = ``rtp``) and the shared
potential-field builder used by ``gravity.py``.

Only the RTP field is differentiated: total-field products (``tmi``, ``mag_anom`` ...) are never relabelled as
RTP. Delivered derivative bands are compared with the recomputed ones (Spearman on a block sample); a recomputed
feature that matches a delivered band with rho >= ``potential.duplicate_rho`` is not materialised again - its
registry row points at the delivered band (status ``alias_of_delivered``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr

from . import structure_tensor as st
from .feature_registry import AXIAL, pct_rank
from .fft_operators import FieldSpectrum


def block_sample(mask: np.ndarray, block: int, rng: np.random.Generator, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    """One random valid pixel per ``block`` x ``block`` block (reduces spatial autocorrelation)."""
    H, W = mask.shape
    rr, cc = np.nonzero(mask)
    bid = (rr // block) * ((W + block - 1) // block) + cc // block
    order = rng.permutation(len(rr))
    _, first = np.unique(bid[order], return_index=True)
    pick = order[first]
    if len(pick) > max_n:
        pick = rng.choice(pick, max_n, replace=False)
    return rr[pick], cc[pick]


def build_potential_family(field: np.ndarray, footprint: np.ndarray, *, prefix: str, family: str, band: str,
                           unit: str, cfg, writer, px: float, delivered: dict | None = None, rng=None) -> dict:
    """Derivatives, continuation, residuals, multiscale edges and structural features of one potential field.

    Returns {"spectrum": DataFrame, "band_energy": DataFrame, "duplicates": DataFrame, "edge_ranks": {...},
    "orientation": (c2, s2, coherence) at the middle structural scale}.
    """
    p = cfg["potential"]
    rng = rng or np.random.default_rng(cfg["seed"])
    spec = FieldSpectrum(field, footprint, px, p["pad_frac"], p["taper_px"])
    dist = ndimage.distance_transform_edt(footprint) * px
    guard = float(p["edge_guard_m"])
    gmask = footprint & (dist > guard)
    parent = f"geodawn:{band}"
    out = {"edge_ranks": {}}
    dup_rows = []

    def put(fid, arr, u, op, meaning, scale=None, mask=gmask, orient=""):
        return writer.put(fid, arr, family=family, parent=parent, unit=u, scale_m=scale, operator=op,
                          meaning=meaning, mask=mask, orientation=orient)

    put(f"{prefix}_field", field, unit, "raw band (nodata -> NaN)", f"{band} as delivered", mask=footprint)
    dx, dy, dz = spec.dx(), spec.dy(), spec.dz()
    thd = np.hypot(dx, dy)
    computed = {"thd": thd, "vd": dz}
    for key, arr in computed.items():
        dname = (delivered or {}).get(key)
        if dname is not None:
            rr, cc = block_sample(gmask & np.isfinite(dname[1]), 10, rng, 50000)
            rho = float(spearmanr(arr[rr, cc], dname[1][rr, cc]).statistic) if len(rr) > 10 else np.nan
            dup_rows.append({"computed": f"{prefix}_{key}", "delivered_band": dname[0], "spearman": rho,
                             "n_blocks": int(len(rr))})
            if np.isfinite(rho) and rho >= float(p["duplicate_rho"]):
                writer.alias(f"{prefix}_{key}", target_path=f"03_features/raw_bands/{dname[0]}.tif", family=family,
                             parent=parent, unit=f"{unit}/m", operator=f"FFT {key}",
                             meaning=f"recomputed {key} matches delivered {dname[0]} (rho={rho:.3f}); delivered band used",
                             status="alias_of_delivered")
                continue
        put(f"{prefix}_{key}", arr, f"{unit}/m", "FFT ikx/iky (THD)" if key == "thd" else "FFT |k| F",
            "total horizontal derivative" if key == "thd" else "first vertical derivative")
    put(f"{prefix}_tilt", np.arctan2(dz, thd + 1e-12 * np.nanmax(thd)), "rad", "atan2(fz, THD + eps)",
        "tilt derivative: sign-balanced edge locator, ~0 over contacts")
    put(f"{prefix}_analytic_signal", np.sqrt(dx ** 2 + dy ** 2 + dz ** 2), f"{unit}/m", "sqrt(fx^2+fy^2+fz^2)",
        "analytic-signal amplitude")
    fxx, fyy, fxy = spec.second()
    put(f"{prefix}_dxx", fxx, f"{unit}/m^2", "FFT -kx^2 F", "second horizontal derivative east-east")
    put(f"{prefix}_dyy", fyy, f"{unit}/m^2", "FFT -ky^2 F", "second horizontal derivative north-north")
    put(f"{prefix}_dxy", fxy, f"{unit}/m^2", "FFT -kx ky F", "mixed second horizontal derivative")
    # boundary strike from the horizontal gradient: the boundary runs perpendicular to grad(f)
    phi = np.arctan2(dy, dx)
    az = np.mod(-phi, np.pi)
    put(f"{prefix}_boundary_c2", np.cos(2 * az), "1", "axial vector of the strike perpendicular to grad f",
        "cos 2a of local boundary strike", orient=AXIAL)
    put(f"{prefix}_boundary_s2", np.sin(2 * az), "1", "axial vector of the strike perpendicular to grad f",
        "sin 2a of local boundary strike", orient=AXIAL)
    del fxx, fyy, fxy, phi, az, dz
    base = spec.apply(1.0)
    thd_scales = {0: thd}
    for h in p["continuation_m"]:
        g = min(float(h), float(p["continuation_guard_cap_m"]))
        m = footprint & (dist > max(guard, g))
        up = spec.upward(h)
        uc = spec.apply(up) + spec.mean
        put(f"{prefix}_uc{h}", uc, unit, f"upward continuation exp(-|k| {h})",
            f"field continued upward {h} m (deeper / broader sources)", scale=h, mask=m)
        put(f"{prefix}_residual{h}", base + spec.mean - uc, unit, f"field - uc{h}",
            f"residual: sources shallower than ~{h} m continuation", scale=h, mask=m)
        thd_h = np.hypot(spec.dx(up), spec.dy(up))
        put(f"{prefix}_thd_uc{h}", thd_h, f"{unit}/m", f"THD of uc{h}", f"boundaries of sources after {h} m continuation",
            scale=h, mask=m)
        thd_scales[h] = np.where(m, thd_h, np.nan)
    del base
    thr = float(cfg["evidence"]["strong_rank"])
    ranks = [pct_rank(np.where(gmask, a, np.nan), gmask) for a in thd_scales.values()]
    out["edge_ranks"] = dict(zip(thd_scales, ranks))
    put(f"{prefix}_edge_persistence", st.persistence(ranks, thr), "fraction",
        f"share of scales (0,{','.join(map(str, p['continuation_m']))} m) with THD rank >= {thr}",
        "multiscale boundary persistence")
    put(f"{prefix}_edge_stability", st.cross_scale_stability(ranks), "rank",
        "mean of min(rank_a, rank_b) over consecutive continuation levels", "cross-scale edge stability")
    put(f"{prefix}_boundary_strength", np.nanmean(np.stack(ranks), 0), "rank", "mean THD percentile rank over scales",
        "multiscale boundary strength")
    put(f"{prefix}_edge_density_2km", st.edge_density(ranks[0] >= thr, gmask, int(round(2000 / px))), "fraction",
        f"share of pixels with THD rank >= {thr} in a 2 km window", "edge density")
    c2s, s2s, ws = [], [], []
    mid = None
    for s in p["structure_scales_m"]:
        d = st.structure(field, px, max(px, s / 4.0), s, footprint)
        put(f"{prefix}_coherence_{s}m", d["coherence"], "1", f"structure tensor, integration sigma {s / 2:g} m",
            "gradient coherence (linear, organised boundaries)", scale=s)
        put(f"{prefix}_anisotropy_{s}m", d["anisotropy"], "1", "(l1-l2)/(l1+l2)", "structure-tensor anisotropy", scale=s)
        put(f"{prefix}_strike_c2_{s}m", d["c2"], "1", "structure tensor strike", "cos 2a dominant strike", scale=s,
            orient=AXIAL)
        put(f"{prefix}_strike_s2_{s}m", d["s2"], "1", "structure tensor strike", "sin 2a dominant strike", scale=s,
            orient=AXIAL)
        c2s.append(d["c2"])
        s2s.append(d["s2"])
        ws.append(d["coherence"])
        if mid is None or s == p["structure_scales_m"][len(p["structure_scales_m"]) // 2]:
            mid = (d["c2"], d["s2"], d["coherence"])
    put(f"{prefix}_orientation_consistency", st.orientation_consistency(c2s, s2s, ws), "1",
        "coherence-weighted resultant of per-scale strike vectors", "orientation consistency across scales")
    out["orientation"] = mid
    out["spectrum"] = spec.radial_spectrum().assign(field=prefix)
    out["band_energy"] = spec.band_energy().assign(field=prefix)
    out["duplicates"] = pd.DataFrame(dup_rows)
    return out


def build(gd, footprint: np.ndarray, cfg, writer, rng=None) -> dict:
    band = cfg["potential"]["magnetic_band"]
    if band not in gd.names:
        raise KeyError(f"magnetic band {band!r} not in the GeoDAWN raster ({gd.names}); the TMI bands are not RTP")
    return build_potential_family(gd.read(band), footprint, prefix="mag", family="magnetic", band=band,
                                  unit=cfg["potential"]["magnetic_unit"], cfg=cfg, writer=writer,
                                  px=gd.grid.pixel_m, delivered=None, rng=rng)
