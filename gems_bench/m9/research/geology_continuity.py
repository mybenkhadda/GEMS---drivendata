"""G02-C multiscale structural continuity of ridges (reuses G01 tolerances, percentiles and doubled-angle algebra).

Per field and wavelength lambda (ridge skeletons from ``geology_truncation.ridge_skeleton``):
    near_lam(x)      exp(-d^2 / (2 tol^2)), d = distance to the lambda ridge, tol = G01 ``tolerance_m(lambda)``
    along_lam(x)     share of the line through x along the LOCAL strike (|u| <= L, L = 2 lambda, >= 800 m) occupied by
                     the ridge (dilated by lambda / 8): a lineament that continues, not a blob
    straight_lam(x)  axial agreement of the strike at x - L' t and x + L' t (L' = 1.5 lambda): 1 straight, 0 bent 90 deg
Across scales:
    P_ridge          mean_lam [d_lam <= tol_lam]: the ridge sits at the SAME position at several scales (G01's P is
                     energy persistence and ignores position)
    strike_stab      |sum_lam w e^{2 i theta_lam}| / sum_lam w (w = energy percentile): one orientation at all scales
Bands (``split_km``): narrow (lambda <= split) and broad (lambda > split) are scored SEPARATELY
    C_band = near_band * mean(P_ridge, strike_stab, along_band, straight_band),  G02-C = max(C_narrow, C_broad)
so a short fault expressed only at fine scales is never averaged away by broad smoothing. Fields: max over mag / grav
(a structure continuous in one field is not penalised for the other). Magnetic-gravity tolerant agreement is G01's J
and is deliberately NOT inside G02-C (it would duplicate G01 in the incremental test); it is kept as ``comp:J``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geology_displacement import RotFrame, azimuths, int_shift, nan_running_mean, select_by_azimuth
from .geology_wavelet import combine, doubled_agreement, strike_from_doubled, tolerance_m, unit_doubled

CONT_KEYS = ("score", "P_ridge", "strike_stability", "along", "straightness", "narrow", "broad")


def _along_and_straight(skel: np.ndarray, cu: np.ndarray, su: np.ndarray, strike: np.ndarray, valid: np.ndarray,
                        lam_km: float, pixel_m, cc: dict, az: list[float]) -> tuple[np.ndarray, np.ndarray]:
    lam = float(lam_km) * 1000.0
    py, px = float(pixel_m[0]), float(pixel_m[1])
    step = max(min(py, px), lam / 8.0)
    dil = float(cc["ridge_dilate_frac"]) * lam
    D = (ndimage.distance_transform_edt(~np.asarray(skel, bool), sampling=(py, px)) <= max(dil, 0.5 * min(py, px)))
    D = np.where(valid, D.astype(np.float32), np.nan)
    L = max(float(cc["along_half_length_frac"]) * lam, float(cc["min_along_half_length_m"]))
    half = max(1, int(round(L / step)))
    kc = max(1, int(round(float(cc["curvature_offset_frac"]) * lam / step)))
    al, sr = {}, {}
    for phi in az:
        fr = RotFrame(D.shape, pixel_m, phi, step)
        al[phi] = fr.from_rot(nan_running_mean(fr.to_rot(D), half, 0, 0.5))
        Rc, Rs = fr.to_rot(cu), fr.to_rot(su)
        A = doubled_agreement(int_shift(Rc, -kc, 0), int_shift(Rs, -kc, 0), int_shift(Rc, kc, 0), int_shift(Rs, kc, 0))
        sr[phi] = fr.from_rot(A)
    return select_by_azimuth(al, strike), select_by_azimuth(sr, strike)


def continuity_field(skels: dict, energy_pct: dict, c2s2: dict, valid: np.ndarray, pixel_m, cfg: dict,
                     tol_cfg: dict) -> dict:
    """G02-C maps of ONE field. ``skels`` / ``energy_pct``: {lam: array}; ``c2s2``: {lam: (strike_c2, strike_s2)}
    (M9 coherence-weighted doubled-angle vectors); ``tol_cfg``: G01 ``tolerance`` section."""
    cc = cfg["continuity"]
    az = azimuths(cfg["orientation"]["n_azimuths"])
    v = np.asarray(valid, bool)
    shape = v.shape
    py, px = float(pixel_m[0]), float(pixel_m[1])
    lams = sorted(skels)
    if not lams:
        return {k: np.full(shape, np.nan, np.float32) for k in CONT_KEYS}
    hits, n = np.zeros(shape, np.float32), 0
    sc, ss, sw = np.zeros(shape, np.float32), np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    per = {}
    for lam in lams:
        skel = np.asarray(skels[lam], bool)
        tol = tolerance_m(lam, tol_cfg)
        d = ndimage.distance_transform_edt(~skel, sampling=(py, px)) if skel.any() else np.full(shape, np.inf)
        hits += (d <= tol)
        n += 1
        c2, s2 = c2s2[lam]
        cu, su, _ = unit_doubled(c2, s2)
        w = np.nan_to_num(np.asarray(energy_pct[lam], np.float32), nan=0.0)
        sc += np.nan_to_num(cu) * w
        ss += np.nan_to_num(su) * w
        sw += w * np.isfinite(cu)
        strike = strike_from_doubled(c2, s2)
        strike = np.where(np.isfinite(cu), strike, np.nan)
        al, st = _along_and_straight(skel, cu, su, strike, v, lam, pixel_m, cc, az)
        per[lam] = {"near": np.exp(-0.5 * (np.minimum(d, 1e9) / tol) ** 2).astype(np.float32), "along": al,
                    "straight": st}
    P = (hits / max(n, 1)).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(sw > 1e-6, np.hypot(sc, ss) / np.maximum(sw, 1e-6), np.nan).astype(np.float32)
    split = float(cc["split_km"])
    bands = {"narrow": [lam for lam in lams if lam <= split], "broad": [lam for lam in lams if lam > split]}
    out = {"P_ridge": P, "strike_stability": R}
    along_all, straight_all, score = None, None, None
    for b, ls in bands.items():
        if not ls:
            out[b] = np.full(shape, np.nan, np.float32)
            continue
        near = np.max([per[lam]["near"] for lam in ls], 0)
        along = np.fmax.reduce(np.stack([per[lam]["along"] for lam in ls]), axis=0)
        wsum = np.sum([per[lam]["near"] for lam in ls], 0)
        straight = np.sum([np.nan_to_num(per[lam]["straight"]) * per[lam]["near"] for lam in ls], 0) / \
            np.maximum(wsum, 1e-6)
        anyf = np.isfinite(np.fmax.reduce(np.stack([per[lam]["straight"] for lam in ls]), axis=0))
        straight = np.where(anyf, straight, np.nan)
        cb = near * combine([P, R, along, straight])
        out[b] = cb.astype(np.float32)
        along_all = along if along_all is None else np.fmax(along_all, along)
        straight_all = straight if straight_all is None else np.fmax(straight_all, straight)
        score = cb if score is None else np.fmax(score, cb)
    out.update(score=score, along=along_all, straightness=straight_all)
    return {k: np.where(v, np.asarray(out[k], np.float32), np.nan).astype(np.float32) for k in CONT_KEYS}


def combine_fields(per_field: dict) -> dict:
    """Max over fields of every continuity map (NaN-aware)."""
    maps = list(per_field.values())
    if not maps:
        return {}
    out = {}
    for k in CONT_KEYS:
        arrs = [m[k] for m in maps if m.get(k) is not None]
        a = arrs[0]
        for b in arrs[1:]:
            a = np.fmax(a, b)
        out[k] = a
    return out
