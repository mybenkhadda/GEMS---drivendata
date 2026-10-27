"""G02-B truncation signatures: persistent fabric ridges that END along a candidate line.

Ridges (per field, per wavelength lambda; label-free)
    ridge = G01 energy percentile >= ``ridge_q`` AND a local maximum across the M9 wavelet strike (non-maximum
    suppression at +-1 pixel along the strike normal), thinned to a one-pixel skeleton; components shorter than
    ``min_ridge_len_m`` are dropped. Ridges mark the crests of band-limited energy: anomaly edges / bands at lambda.

Endpoints
    skeleton pixels with crossing number 1 (one run of skeleton neighbours in the 8-ring; robust to the staircase
    corners that defeat neighbour counts). Tip direction = from the centroid of the skeleton within ``tip_radius_px``
    towards the endpoint. Each endpoint gets a weight
        w = abruptness = clip((E_end - E_beyond) / abrupt_drop, 0, 1),  E_beyond = energy 0.75 lambda past the tip
    so a ridge that FADES (weak signal reaching the percentile threshold) carries little weight, and w = 0 when the tip
    lies within ``invalid_margin_m`` of invalid data or E_beyond is outside the data (data edges are not geology; these
    endpoints are counted separately in ``edge_endpoints``).

Alignment along a candidate line (azimuth phi, normal n)
    A truncating line collects endpoints whose tips POINT AT it from either side: endpoints on the -n side facing +n
    and endpoints on the +n side facing -n, at least ``min_facing_deg`` oblique to the line. Their weighted count in the
    two one-sided strips (|u| <= L, 0 < |v| <= w) is compared with the count expected from the local isotropic density
    of the same endpoints (Gaussian background): z = (obs - exp) / sqrt(exp + 1). score = clip(z / z_ref, 0, 1) when
    obs >= ``min_count``. A swarm of endpoints with no common line (fragmented fabric, noise) gives z ~ 0.

Also measured (inputs to G02-E): the energy-percentile step across the local strike (|E_-n - E_+n|: juxtaposition or
termination of an anomaly band) and the fine-minus-coarse energy step across it (a survey-resolution seam changes
the fine scales only).

Limitations: truncation by a fault and termination of a body at an intrusive or unconformable contact look the same
here; G02-E weighs them against competing explanations. Ridges that end because the fabric bends out of the energy
band are counted as endpoints (abruptness only partly controls this).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geology_displacement import RotFrame, azimuths, frame_vectors, int_shift, offset_px, select_by_azimuth, \
    side_means

TRUNC_KEYS = ("score", "count", "azimuth", "lambda_km", "energy_step", "edge_endpoints")
RING = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))   # circular 8-neighbour order


# =============================================================================== ridges
def ridge_skeleton(E_pct: np.ndarray, strike_deg: np.ndarray, valid: np.ndarray, pixel_m, q: float,
                   min_len_m: float) -> np.ndarray:
    """One-pixel skeleton of across-strike energy maxima above the percentile ``q`` (bool)."""
    from skimage.morphology import skeletonize

    E = np.asarray(E_pct, np.float32)
    th = np.asarray(strike_deg, np.float32)
    ok = np.isfinite(E) & np.isfinite(th) & np.asarray(valid, bool)
    if not ok.any():
        return np.zeros(E.shape, bool)
    d = float(min(pixel_m))
    dr, dc = offset_px(d, np.where(ok, th, 0.0) + 90.0, pixel_m)
    rr, cc = np.indices(E.shape, dtype=np.float32)
    Ez = np.where(np.isfinite(E), E, -1.0).astype(np.float32)
    kw = {"order": 1, "mode": "constant", "cval": -1.0, "prefilter": False}
    Ep = ndimage.map_coordinates(Ez, [rr + dr.astype(np.float32), cc + dc.astype(np.float32)], **kw)
    Em = ndimage.map_coordinates(Ez, [rr - dr.astype(np.float32), cc - dc.astype(np.float32)], **kw)
    ridge = ok & (E >= float(q)) & (Ez >= Ep) & (Ez >= Em)
    skel = skeletonize(ridge)
    return remove_short(skel, pixel_m, min_len_m)


def remove_short(skel: np.ndarray, pixel_m, min_len_m: float) -> np.ndarray:
    skel = np.asarray(skel, bool)
    if not skel.any() or min_len_m <= 0:
        return skel
    lab, n = ndimage.label(skel, structure=np.ones((3, 3), bool))
    size = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64) * float(np.mean(pixel_m))
    keep = size >= float(min_len_m)
    keep[0] = False
    return keep[lab]


def crossing(skel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(crossing number = 0->1 transitions around the 8-ring, number of skeleton neighbours) per pixel."""
    s = np.asarray(skel, bool).astype(np.float32)
    nbs = [int_shift(s, r, c, fill=0.0) > 0.5 for r, c in RING]
    tr = np.zeros(s.shape, np.int8)
    for k in range(8):
        tr += (~nbs[k]) & nbs[(k + 1) % 8]
    cnt = np.sum(nbs, axis=0).astype(np.int8)
    return tr, cnt


def skeleton_nodes(skel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(endpoints, junctions) of a one-pixel skeleton."""
    skel = np.asarray(skel, bool)
    tr, cnt = crossing(skel)
    return skel & (tr == 1) & (cnt <= 3), skel & (tr >= 3)


def tip_azimuth(skel: np.ndarray, rows: np.ndarray, cols: np.ndarray, radius_px: int, pixel_m) -> np.ndarray:
    """Full-circle azimuth (deg) pointing from the local skeleton centroid to each endpoint (NaN if undefined)."""
    R = int(radius_px)
    y, x = np.mgrid[-R:R + 1, -R:R + 1]
    disk = (y ** 2 + x ** 2) <= R * R
    s = np.asarray(skel, np.float32)
    kw = {"mode": "constant", "cval": 0.0}
    n = ndimage.correlate(s, disk.astype(np.float32), **kw)[rows, cols]
    sy = ndimage.correlate(s, (y * disk).astype(np.float32), **kw)[rows, cols]
    sx = ndimage.correlate(s, (x * disk).astype(np.float32), **kw)[rows, cols]
    with np.errstate(invalid="ignore", divide="ignore"):
        tr, tc = -sy / n, -sx / n                                   # centroid -> endpoint, pixel coordinates
    tE, tN = tc * float(pixel_m[1]), -tr * float(pixel_m[0])
    ok = np.hypot(tE, tN) > 1e-6
    return np.where(ok, np.degrees(np.arctan2(tE, tN)) % 360.0, np.nan).astype(np.float32)


def endpoint_table(skel: np.ndarray, E_pct: np.ndarray, valid: np.ndarray, lam_km: float, pixel_m, tc: dict) -> dict:
    """Endpoints of one skeleton: rows, cols, tip azimuth, abruptness weight, edge flag."""
    ep, _ = skeleton_nodes(skel)
    rows, cols = np.nonzero(ep)
    out = {"rows": rows, "cols": cols, "tip_az": np.zeros(0, np.float32), "w": np.zeros(0, np.float32),
           "edge": np.zeros(0, bool)}
    if not rows.size:
        return out
    taz = tip_azimuth(skel, rows, cols, int(tc["tip_radius_px"]), pixel_m)
    E = np.asarray(E_pct, np.float32)
    e_end = E[rows, cols]
    b = float(tc["beyond_frac"]) * float(lam_km) * 1000.0
    dr, dc = offset_px(b, np.nan_to_num(taz), pixel_m)
    e_bey = ndimage.map_coordinates(E, [rows + dr, cols + dc], order=1, mode="constant", cval=np.nan, prefilter=False)
    v = np.asarray(valid, bool)
    dist_inv = ndimage.distance_transform_edt(np.pad(v, 1, constant_values=False),
                                              sampling=(float(pixel_m[0]), float(pixel_m[1])))[1:-1, 1:-1]
    edge = (dist_inv[rows, cols] < float(tc["invalid_margin_m"])) | ~np.isfinite(e_bey) | ~np.isfinite(taz)
    w = np.clip((e_end - e_bey) / float(tc["abrupt_drop"]), 0.0, 1.0)
    w = np.where(edge | ~np.isfinite(w), 0.0, w).astype(np.float32)
    out.update(tip_az=taz, w=w, edge=edge)
    return out


# =============================================================================== strip statistics
def window_sum(R: np.ndarray, lo_off: int, hi_off: int, axis: int) -> np.ndarray:
    """S[i] = sum_{k = i + lo_off}^{i + hi_off} R[k] along ``axis`` (zero outside; cumulative sums)."""
    R = np.moveaxis(np.asarray(R, np.float64), axis, 0)
    n = R.shape[0]
    C = np.zeros((n + 1,) + R.shape[1:], np.float64)
    np.cumsum(R, axis=0, out=C[1:])
    i = np.arange(n)
    lo = np.clip(i + int(lo_off), 0, n)
    hi = np.clip(i + int(hi_off) + 1, 0, n)
    S = np.where((hi > lo)[(slice(None),) + (None,) * (R.ndim - 1)], C[hi] - C[lo], 0.0)
    return np.moveaxis(S, 0, axis).astype(np.float32)


def aligned_endpoint_counts(eps: dict, shape, pixel_m, phi: float, half_m: float, width_m: float,
                            min_facing_deg: float, bg_sigma_m: float) -> tuple[np.ndarray, np.ndarray]:
    """(observed, expected) weighted counts of endpoints facing a line of azimuth ``phi`` through each pixel."""
    rows, cols, taz, w = eps["rows"], eps["cols"], eps["tip_az"], eps["w"]
    if not rows.size:
        z = np.zeros(shape, np.float32)
        return z, z.copy()
    (_, _), (nE, nN) = frame_vectors(phi)
    a = np.radians(np.nan_to_num(taz))
    facing = np.sin(a) * nE + np.cos(a) * nN
    s = np.sin(np.radians(float(min_facing_deg)))
    wp = np.where(facing >= s, w, 0.0).astype(np.float32)        # tip points towards +n: must lie on the -n side
    wm = np.where(facing <= -s, w, 0.0).astype(np.float32)       # tip points towards -n: must lie on the +n side
    step = float(min(pixel_m))
    fr = RotFrame(shape, pixel_m, phi, step)
    half = max(1, int(round(half_m / step)))
    kw = max(1, int(round(width_m / step)))
    Sp = window_sum(window_sum(fr.splat(rows, cols, wp), -half, half, 0), -kw, -1, 1)
    Sm = window_sum(window_sum(fr.splat(rows, cols, wm), -half, half, 0), 1, kw, 1)
    obs = np.nan_to_num(fr.from_rot(Sp + Sm), nan=0.0)
    D = np.zeros(shape, np.float32)
    np.add.at(D, (rows, cols), wp + wm)
    sig = (bg_sigma_m / float(pixel_m[0]), bg_sigma_m / float(pixel_m[1]))
    dens = ndimage.gaussian_filter(D, sig, mode="constant")       # weighted endpoints per pixel (isotropic)
    area = (2 * half + 1) * kw * (step * step) / (float(pixel_m[0]) * float(pixel_m[1]))
    return obs.astype(np.float32), (dens * area).astype(np.float32)


# =============================================================================== evidence map
def truncation_evidence(skels: dict, energy_pct: dict, strike: dict, valid: np.ndarray, pixel_m, cfg: dict) -> dict:
    """G02-B maps of ONE field: {score, count, azimuth, lambda_km, energy_step, edge_endpoints}.

    ``skels`` / ``energy_pct`` / ``strike``: {lam_km: ridge skeleton / G01 energy percentile / strike (deg)}."""
    tc = cfg["truncation"]
    az = azimuths(cfg["orientation"]["n_azimuths"])
    shape = np.asarray(valid).shape
    best = np.full(shape, -np.inf, np.float32)
    out = {k: np.full(shape, np.nan, np.float32) for k in TRUNC_KEYS if k != "score"}
    edge = np.zeros(shape, np.float32)
    estep = None
    for lam in sorted(skels):
        lam_m = float(lam) * 1000.0
        eps = endpoint_table(skels[lam], energy_pct[lam], valid, lam, pixel_m, tc)
        if eps["rows"].size:
            np.add.at(edge, (eps["rows"][eps["edge"]], eps["cols"][eps["edge"]]), 1.0)
        half = max(float(tc["strip_half_length_frac"]) * lam_m, float(tc["min_strip_half_length_m"]))
        width = float(tc["strip_width_frac"]) * lam_m
        bg = min(float(tc["background_sigma_frac"]) * lam_m, float(tc["max_background_sigma_m"]))
        for phi in az:
            obs, exp = aligned_endpoint_counts(eps, shape, pixel_m, phi, half, width, float(tc["min_facing_deg"]), bg)
            z = (obs - exp) / np.sqrt(exp + 1.0)
            s = np.clip(z / float(tc["z_ref"]), 0.0, 1.0) * (obs >= float(tc["min_count"]))
            better = s > best
            best = np.where(better, s, best)
            out["count"] = np.where(better, obs, out["count"])
            out["azimuth"] = np.where(better, np.float32(phi), out["azimuth"])
            out["lambda_km"] = np.where(better, np.float32(lam), out["lambda_km"])
        st = across_step({"e": energy_pct[lam]}, strike[lam], pixel_m, float(tc["side_offset_frac"]) * lam_m,
                         half, az)["e"]
        estep = st if estep is None else np.fmax(estep, st)
    v = np.asarray(valid, bool)
    out["score"] = np.where(np.isfinite(best), best, np.nan).astype(np.float32)
    out["energy_step"] = estep if estep is not None else np.full(shape, np.nan, np.float32)
    sig = 2000.0 / float(pixel_m[0]), 2000.0 / float(pixel_m[1])
    out["edge_endpoints"] = ndimage.gaussian_filter(edge, sig, mode="constant")   # diagnostic density (per pixel)
    return {k: np.where(v, a, np.nan).astype(np.float32) for k, a in out.items()}


def across_step(arrays: dict, strike_deg: np.ndarray, pixel_m, h_m: float, half_m: float, az: list[float]) -> dict:
    """{name: |mean on the -n side - mean on the +n side|} across the LOCAL strike (nearest tested azimuth)."""
    step = max(float(min(pixel_m)), h_m / 4.0)
    per = {name: {} for name in arrays}
    for phi in az:
        sm = side_means(arrays, pixel_m, phi, h_m, half_m, step)
        for name, (lm, rm) in sm.items():
            per[name][phi] = np.abs(lm - rm)
    return {name: select_by_azimuth(per[name], strike_deg) for name in arrays}
