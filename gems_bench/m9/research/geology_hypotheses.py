"""G02-E competing geological explanations of a geophysical lineament (hypothesis SCORES, not probabilities).

Measurements (label-free, [0, 1]; computed on the same window)
    L            lineament strength = max(M, G) (G01 multiscale energy percentiles)
    theta_line   local lineament strike: energy-weighted doubled-angle mean of the M9 wavelet strike over scales and
                 fields; theta_fabric / R_fabric: the same field smoothed at ``fabric_sigma_m`` (regional grain)
    discord      R_fabric (1 - A(theta_line, theta_fabric)): the lineament CUTS the regional grain
    parallel     R_fabric A + (1 - R_fabric) / 2: the lineament FOLLOWS the grain (neutral where there is no grain)
    step         |P(-n side) - P(+n side)| of the raw-field percentile (training-geography reference) at +-1 km across
                 theta_line, max over fields: juxtaposition of units with different susceptibility / density
    blob_near    scale-normalised |sigma^2 LoG| of the raw fields / IQR(reference), max over scales, max-filtered over
                 ``blob_radius_m``: a compact anomalous body nearby
    disruption   noisy-or(G02-A displacement, G02-B truncation, discord x straightness)
    straight, persist   G02-C straightness and ridge-position persistence
    terrain_align  A(theta_line, terrain strike) x terrain coherence (structure tensor of detrended elevation)
    artifact     max of: proximity to invalid data, survey-resolution seam (fine-minus-coarse energy step across the
                 line), [grid-axis-parallel straight lineament: measured, weight 0 by default]
Supports
    fault       = L * disruption * (0.5 + 0.5 persist)                 deformation that offsets / cuts structure
    contact     = L * step * parallel * (1 - disruption)              juxtaposition along the grain, no disruption
    intrusive   = L * blob_near * (1 - straight) * (1 - disruption / 2) curved margin of a compact body
    terrain     = S * terrain_align * straight                          structurally controlled surface expression
                                                                         (corroborates; NOT exclusive of the others)
    ambiguous   = max(L - max(fault, contact, intrusive), L * artifact) strong lineament the evidence cannot explain
    dominant    argmax of (fault, contact, intrusive, ambiguous); margin = top1 - top2; uncertainty = 1 - margin / top1
    G02-E       = clip(fault + ambiguous_weight * ambiguous, 0, 1): fault-explained lineaments first, unexplained ones
                 next, contact / intrusion-explained ones last.
The formulas are fixed choices made before any evaluation; a negative G02-E result refutes this rule, not the
geological distinction itself. Without displacement or truncation evidence a real fault falls into ``ambiguous``:
by construction the map says "cannot tell", not "not a fault".
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geology_displacement import azimuths, noisy_or
from .geology_truncation import across_step
from .geology_wavelet import axial_agreement, masked_gaussian, strike_from_doubled, unit_doubled

HYPOTHESES = ("fault", "contact", "intrusive", "ambiguous")
HYP_KEYS = ("hyp:fault", "hyp:contact", "hyp:intrusive", "hyp:terrain", "hyp:ambiguous", "hyp:dominant",
            "hyp:margin", "hyp:uncertainty", "meas:discord", "meas:parallel", "meas:step", "meas:blob_near",
            "meas:terrain_align", "meas:artifact", "meas:seam", "meas:theta_line", "G02-E")


def line_strike(vectors: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(strike deg, summed c2, summed s2) of ``vectors`` = [(strike_c2, strike_s2, weight), ...]."""
    C = S = None
    for c2, s2, w in vectors:
        w = np.nan_to_num(np.asarray(w, np.float32), nan=0.0)
        c = np.nan_to_num(np.asarray(c2, np.float32)) * w
        s = np.nan_to_num(np.asarray(s2, np.float32)) * w
        C = c if C is None else C + c
        S = s if S is None else S + s
    th = strike_from_doubled(C, S)
    return np.where(np.hypot(C, S) > 1e-6, th, np.nan).astype(np.float32), C, S


def fabric(C: np.ndarray, S: np.ndarray, valid, sigma_m: float, pixel_m) -> tuple[np.ndarray, np.ndarray]:
    """(regional fabric strike, coherence R in [0, 1]) from UNIT doubled vectors smoothed at ``sigma_m``."""
    cu, su, _ = unit_doubled(C, S)
    sig = (sigma_m / float(pixel_m[0]), sigma_m / float(pixel_m[1]))
    cs = masked_gaussian(np.where(valid, cu, np.nan), sig)
    ss = masked_gaussian(np.where(valid, su, np.nan), sig)
    return strike_from_doubled(cs, ss), np.clip(np.hypot(cs, ss), 0.0, 1.0).astype(np.float32)


def blob_response(raw: np.ndarray, valid, sigmas_m, pixel_m, iqr: float) -> np.ndarray:
    """max over scales of |sigma^2 LoG(raw)| / IQR (raw nearest-filled; NaN on invalid pixels)."""
    from gems_eda.enhance import fill_nan

    a = np.asarray(raw, np.float32)
    m = np.isfinite(a) & np.asarray(valid, bool)
    if not m.any() or not np.isfinite(iqr) or iqr <= 0:
        return np.full(a.shape, np.nan, np.float32)
    g = fill_nan(np.where(m, a, np.nan)).astype(np.float32)
    out = np.zeros(a.shape, np.float32)
    py, px = float(pixel_m[0]), float(pixel_m[1])
    for s in sigmas_m:
        sp = (float(s) / py, float(s) / px)
        r = np.abs(ndimage.gaussian_laplace(g, sp, mode="nearest")) * (sp[0] * sp[1])
        out = np.maximum(out, r / float(iqr))
    return np.where(m, out, np.nan).astype(np.float32)


def terrain_strike(elev: np.ndarray, valid, sigma_m: float, pixel_m) -> tuple[np.ndarray, np.ndarray]:
    """(strike of the dominant elevation gradient + 90 deg, coherence) from the smoothed structure tensor."""
    from gems_eda.enhance import fill_nan

    a = np.asarray(elev, np.float32)
    m = np.isfinite(a) & np.asarray(valid, bool)
    if not m.any():
        nan = np.full(a.shape, np.nan, np.float32)
        return nan, nan.copy()
    g = fill_nan(np.where(m, a, np.nan)).astype(np.float32)
    py, px = float(pixel_m[0]), float(pixel_m[1])
    sp = (sigma_m / py, sigma_m / px)
    gr = ndimage.gaussian_filter(g, sp, order=(1, 0), mode="nearest") / py     # d/d row (towards south)
    gc = ndimage.gaussian_filter(g, sp, order=(0, 1), mode="nearest") / px     # d/d col (towards east)
    gE, gN = gc, -gr
    s2 = (2 * sp[0], 2 * sp[1])
    Jee = ndimage.gaussian_filter(gE * gE, s2)
    Jnn = ndimage.gaussian_filter(gN * gN, s2)
    Jen = ndimage.gaussian_filter(gE * gN, s2)
    # doubled azimuth of the gradient (azimuth clockwise from north: x = N component, y = E component)
    c2, s2_ = Jnn - Jee, 2.0 * Jen
    tr = Jee + Jnn
    coh = np.where(tr > 1e-20, np.hypot(c2, s2_) / np.maximum(tr, 1e-20), 0.0).astype(np.float32)
    grad_az = np.degrees(0.5 * np.arctan2(s2_, c2)) % 180.0
    strike = (grad_az + 90.0) % 180.0
    return np.where(m, strike, np.nan).astype(np.float32), np.where(m, coh, np.nan).astype(np.float32)


def hypotheses(inp: dict, valid: np.ndarray, pixel_m, cfg: dict) -> dict:
    """G02-E maps of one window. ``inp`` keys: L, disp, trunc, straight, persist, vectors [(c2, s2, w)], raw_pct
    {field: percentile map}, raw {field: raw field}, raw_iqr {field: IQR}, fine_coarse (list of fine-minus-coarse
    percentile maps, optional), S (terrain percentile, optional), elev (detrended elevation, optional)."""
    hc = cfg["hypotheses"]
    v = np.asarray(valid, bool)
    shape = v.shape
    az = azimuths(cfg["orientation"]["n_azimuths"])
    nanm = np.full(shape, np.nan, np.float32)
    L = np.asarray(inp["L"], np.float32)
    th, C, S = line_strike(inp["vectors"])
    thf, Rf = fabric(C, S, v, float(hc["fabric_sigma_m"]), pixel_m)
    A = axial_agreement(th, thf)
    discord = np.where(np.isfinite(A), Rf * (1.0 - A), np.nan)
    parallel = np.where(np.isfinite(A), Rf * A + (1.0 - Rf) * 0.5, np.nan)
    h, half = float(hc["side_offset_m"]), float(hc["side_half_length_m"])
    rp = {f: a for f, a in (inp.get("raw_pct") or {}).items() if a is not None}
    steps = across_step(rp, th, pixel_m, h, half, az) if rp else {}
    step = None
    for a in steps.values():
        step = a if step is None else np.fmax(step, a)
    step = nanm if step is None else step
    blob = None
    for f, a in (inp.get("raw") or {}).items():
        b = blob_response(a, v, hc["blob_sigma_m"], pixel_m, (inp.get("raw_iqr") or {}).get(f, np.nan))
        blob = b if blob is None else np.fmax(blob, b)
    if blob is not None:
        rad = (float(hc["blob_radius_m"]) / float(pixel_m[0]), float(hc["blob_radius_m"]) / float(pixel_m[1]))
        size = (2 * int(round(rad[0])) + 1, 2 * int(round(rad[1])) + 1)
        bn = ndimage.maximum_filter(np.nan_to_num(np.clip(blob / float(hc["blob_ref"]), 0, 1)), size=size)
        blob_near = np.where(v, bn, np.nan).astype(np.float32)
    else:
        blob_near = nanm
    straight = np.asarray(inp["straight"], np.float32)
    persist = np.asarray(inp["persist"], np.float32)
    disruption = noisy_or(inp.get("disp"), inp.get("trunc"),
                          np.where(np.isfinite(discord * straight), discord * straight, np.nan))
    disruption = nanm if disruption is None else disruption
    d0 = np.nan_to_num(disruption, nan=0.0)
    fault = L * d0 * (0.5 + 0.5 * np.nan_to_num(persist, nan=0.0))
    contact = L * np.nan_to_num(step, nan=0.0) * np.nan_to_num(parallel, nan=0.5) * (1.0 - d0)
    intrusive = L * np.nan_to_num(blob_near, nan=0.0) * (1.0 - np.nan_to_num(straight, nan=0.0)) * (1.0 - 0.5 * d0)
    # ---- terrain (corroborating) ----------------------------------------------------------------------------
    if inp.get("elev") is not None and inp.get("S") is not None:
        tst, tcoh = terrain_strike(inp["elev"], v, float(hc["terrain_sigma_m"]), pixel_m)
        talign = axial_agreement(th, tst) * tcoh
        terrain = np.asarray(inp["S"], np.float32) * talign * np.nan_to_num(straight, nan=0.0)
    else:
        talign, terrain = nanm, nanm
    # ---- artefacts ---------------------------------------------------------------------------------------------
    dist = ndimage.distance_transform_edt(np.pad(v, 1, constant_values=False),
                                          sampling=(float(pixel_m[0]), float(pixel_m[1])))[1:-1, 1:-1]
    inv = np.clip(1.0 - dist / float(hc["invalid_m"]), 0.0, 1.0)
    seam = nanm
    fc = [a for a in (inp.get("fine_coarse") or []) if a is not None]
    if fc:
        st = across_step({f"fc{i}": a for i, a in enumerate(fc)}, th, pixel_m, h, half, az)
        seam = np.clip(np.fmax.reduce(np.stack(list(st.values())), axis=0) / 0.5, 0.0, 1.0)
    dx = np.minimum(np.abs((np.nan_to_num(th) + 90.0) % 180.0 - 90.0), np.abs((np.nan_to_num(th) % 180.0) - 90.0))
    axis = (dx <= float(hc["axis_tol_deg"])) * np.nan_to_num(straight, nan=0.0)
    artifact = np.maximum.reduce([float(hc["invalid_weight"]) * inv, float(hc["seam_weight"]) * np.nan_to_num(seam),
                                  float(hc["axis_weight"]) * axis])
    best3 = np.maximum.reduce([fault, contact, intrusive])
    ambiguous = np.maximum(np.clip(L - best3, 0.0, 1.0), L * np.clip(artifact, 0.0, 1.0))
    stack = np.stack([fault, contact, intrusive, ambiguous]).astype(np.float32)
    fin = np.isfinite(L) & v
    srt = np.sort(np.nan_to_num(stack, nan=-1.0), axis=0)
    top1, top2 = srt[-1], srt[-2]
    dominant = np.where(fin, np.argmax(np.nan_to_num(stack, nan=-1.0), 0), -1).astype(np.int8)
    margin = np.where(fin, top1 - top2, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        unc = np.where(fin & (top1 > 1e-6), 1.0 - margin / np.maximum(top1, 1e-6), np.nan)
    e = np.clip(fault + float(hc["ambiguous_weight"]) * ambiguous, 0.0, 1.0)
    out = {"hyp:fault": fault, "hyp:contact": contact, "hyp:intrusive": intrusive, "hyp:terrain": terrain,
           "hyp:ambiguous": ambiguous, "hyp:margin": margin, "hyp:uncertainty": unc, "meas:discord": discord,
           "meas:parallel": parallel, "meas:step": step, "meas:blob_near": blob_near, "meas:terrain_align": talign,
           "meas:artifact": artifact, "meas:seam": seam, "meas:theta_line": th, "G02-E": e}
    out = {k: np.where(fin, np.asarray(a, np.float32), np.nan).astype(np.float32) for k, a in out.items()}
    out["hyp:dominant"] = dominant
    return out
