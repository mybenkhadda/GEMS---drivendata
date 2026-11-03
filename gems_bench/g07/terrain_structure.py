"""Structural landforms from a DEM window: scarps, lineaments, truncations, drainage alignment, persistence.

These are OBSERVATIONS of the terrain, not interpretations: a scarp may be tectonic, erosional or anthropogenic;
a drainage lineament is not a fault. The features are built so a model can weigh competing interpretations
(continuity along strike, coherence, persistence across scales, agreement with the geophysical orientation).

Per scale ``s`` (work-resolution window, metres)::

    dem_scarp_s / dem_scarp_max_s      m      height step |S(x + s n) - S(x - s n)| across the local gradient
                                              direction n, times structure-tensor coherence (mean / max)
    dem_scarpmask_s                    share  slope >= scarp_slope_deg & coherence >= scarp_coherence & step in
                                              [scarp_step_m]
    dem_lin_s                          share  thin lineaments: ridge / valley strength maximal across strike, coherent
    dem_lin_orient_s                   orientation (doubled strike) of the lineament pixels
    dem_along_s / dem_trunc_s          continuity along strike / abrupt termination (truncation proxy)
Per tier::

    dem_scarp_persist_<tier>           share of the tier's scales at which the pixel is a scarp
    dem_dircoh_<tier>                  orientation persistence across scales
    dem_valleyline_<tier> / dem_valley_orient_<tier>   valley-line density / orientation (drainage alignment)

100 m terrain-versus-geophysics orientation agreement (``orientation_agreement``) uses the G06 structure-tensor
orientation of the magnetic / gravity fields: A = (1 + cos 2 (strike_dem - strike_geo)) / 2.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..g06.feature_structure import along_cross_features, axial_agreement, directional_coherence
from .dem_io import MAX, MEAN, ORIENT


def _t(s) -> int:
    return int(round(float(s)))


def structure_kinds(cfg: dict, res: float, scales, fine: bool = True) -> dict:
    k = {}
    for s in scales:
        t = _t(s)
        k.update({f"dem_scarp_s{t}": MEAN, f"dem_scarp_max_s{t}": MAX, f"dem_scarpmask_s{t}": MEAN,
                  f"dem_lin_s{t}": MEAN, f"dem_lin_orient_s{t}": ORIENT, f"dem_along_s{t}": MEAN,
                  f"dem_trunc_s{t}": MEAN})
    tier = "fine" if fine else "coarse"
    if scales:
        k.update({f"dem_scarp_persist_{tier}": MEAN, f"dem_valleyline_{tier}": MEAN,
                  f"dem_valley_orient_{tier}": ORIENT})
        if len(scales) > 1:
            k[f"dem_dircoh_{tier}"] = MEAN
    return k


def required_extra_px(cfg: dict, res: float, scales) -> int:
    smax = max([float(s) for s in scales] or [2 * res])
    return int(np.ceil(4 * max(1.0, smax / res / 2) + smax / res)) + 2


def _sample(a, rr, cc, dr, dc):
    return ndimage.map_coordinates(a, [rr + dr, cc + dc], order=1, mode="nearest")


def scarp_candidates(d: dict, cfg: dict) -> dict:
    """Scarp step height (m), strength (step x coherence) and the thresholded scarp mask at the scale of ``d``."""
    S, th, coh = d["S"], d["theta_grad"], d["aniso"]
    res, s = d["res"], d["scale"]
    rr, cc = np.indices(S.shape, dtype=np.float64)
    nc, nr = np.cos(th), -np.sin(th)
    off = s / res
    step = np.abs(_sample(S, rr, cc, off * nr, off * nc) - _sample(S, rr, cc, -off * nr, -off * nc))
    slope = np.degrees(np.arctan(np.hypot(d["zx"], d["zy"])))
    lo, hi = (float(x) for x in cfg.get("scarp_step_m", [0.5, 30.0]))
    mask = (slope >= float(cfg.get("scarp_slope_deg", 12.0))) & (coh >= float(cfg.get("scarp_coherence", 0.55))) & \
        (step >= lo) & (step <= hi)
    return {"step": step, "strength": step * coh, "mask": mask.astype(np.float32)}


def terrain_lineaments(d: dict, cfg: dict) -> dict:
    """Thin ridge / valley lines: line strength maximal across strike (non-maximum suppression) and coherent."""
    from .dem_features import ridge_valley

    rv = ridge_valley(d)
    L = np.maximum(rv["ridge"], rv["valley"])
    th = d["theta_grad"]
    rr, cc = np.indices(L.shape, dtype=np.float64)
    nc, nr = np.cos(th), -np.sin(th)
    Lf = np.where(np.isfinite(L), L, -1.0)
    fwd, bwd = _sample(Lf, rr, cc, nr, nc), _sample(Lf, rr, cc, -nr, -nc)
    thr = float(cfg.get("lineament_min_m", 0.05))
    lin = (Lf >= fwd) & (Lf >= bwd) & (Lf > thr) & (d["aniso"] >= float(cfg.get("lineament_coherence", 0.5)))
    valley_line = lin & (rv["valley"] >= rv["ridge"])
    return {"lin": lin.astype(np.float32), "valley_line": valley_line.astype(np.float32), "strength": L}


def directional_coherence_dem(thetas: list, weights: list) -> np.ndarray:
    return directional_coherence(thetas, weights)


def terrain_truncation(d: dict) -> dict:
    """Along-strike continuity and truncation of the slope field (``g06.feature_structure.along_cross_features``)."""
    hg = np.hypot(d["zx"], d["zy"])
    ac = along_cross_features({"hgm": hg, "S": d["S"], "sigma_px": d["scale"] / d["res"]}, d["theta_grad"])
    return {"along": ac["along"], "trunc": ac["trunc"]}


def drainage_alignment(lin: dict, d: dict) -> dict:
    """Valley-line density and its doubled strike angle (weight = valley-line mask): drainage that follows a
    structural grain shows up as a coherent valley orientation; an offset drainage is NOT inferred here."""
    strike2 = 2 * (d["theta_grad"] + np.pi / 2)
    return {"valleyline": lin["valley_line"], "valley_orient": strike2, "valley_orient__w": lin["valley_line"]}


def multiscale_scarp_persistence(masks: list) -> np.ndarray:
    return np.mean(np.stack(masks), 0) if masks else None


def structure_bank(z: np.ndarray, res: float, ds: dict, cfg: dict, fine: bool = True) -> dict:
    """Every feature of ``structure_kinds`` from the per-scale derivative dicts ``ds`` (from ``dem_features``)."""
    out, masks, thetas, wts = {}, [], [], []
    tier = "fine" if fine else "coarse"
    scales = sorted(ds)
    mid = scales[len(scales) // 2] if scales else None
    for s in scales:
        d = ds[s]
        t = _t(s)
        sc = scarp_candidates(d, cfg)
        out[f"dem_scarp_s{t}"], out[f"dem_scarp_max_s{t}"] = sc["strength"], sc["strength"]
        out[f"dem_scarpmask_s{t}"] = sc["mask"]
        masks.append(sc["mask"])
        ln = terrain_lineaments(d, cfg)
        out[f"dem_lin_s{t}"] = ln["lin"]
        out[f"dem_lin_orient_s{t}"] = 2 * (d["theta_grad"] + np.pi / 2)
        out[f"dem_lin_orient_s{t}__w"] = ln["lin"]
        tr = terrain_truncation(d)
        out[f"dem_along_s{t}"], out[f"dem_trunc_s{t}"] = tr["along"], tr["trunc"]
        thetas.append(d["theta_grad"])
        wts.append(d["aniso"])
        if s == mid:
            dr = drainage_alignment(ln, d)
            out[f"dem_valleyline_{tier}"] = dr["valleyline"]
            out[f"dem_valley_orient_{tier}"], out[f"dem_valley_orient_{tier}__w"] = dr["valley_orient"], \
                dr["valley_orient__w"]
    if masks:
        out[f"dem_scarp_persist_{tier}"] = multiscale_scarp_persistence(masks)
    if len(scales) > 1:
        out[f"dem_dircoh_{tier}"] = directional_coherence_dem(thetas, wts)
    return out


def orientation_agreement(dem_cos2: np.ndarray, dem_sin2: np.ndarray, geo_ocos: np.ndarray, geo_osin: np.ndarray
                          ) -> np.ndarray:
    """Axial agreement of the DEM strike (doubled-angle unit vector) with a G06 structure-tensor GRADIENT axis
    (``struct_<f>_ocos/osin``; strike = axis + 90 deg, i.e. the doubled vector flips sign)."""
    with np.errstate(invalid="ignore"):
        cos_d = dem_cos2 * (-geo_ocos) + dem_sin2 * (-geo_osin)      # cos 2(strike_dem - strike_geo)
    return 0.5 * (1.0 + cos_d)


def axial(t1, t2):
    return axial_agreement(t1, t2)
