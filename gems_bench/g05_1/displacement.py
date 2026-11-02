"""Stage E: displacement-style evidence from shifted profile correspondence. Pure numpy / scipy.

For one candidate trace and one potential field (e.g. ``rtp``, ``iso_grav_anom``):
  1. two profiles parallel to the trace at +d and -d px along the local normal (``side_px``);
  2. the along-strike shift k in [-K, K] that maximises the Pearson correlation of A(s) and B(s + k);
  3. gain = corr(best k) - corr(k = 0): a lateral offset of anomalies across the trace raises it;
  4. stability: the best shift on each half of the trace and at each smoothing scale must agree in sign and within
     2 px (``consistent``);
  5. negative control: the same computation on two profiles at the SAME separation but both on one side of the trace
     (centred ``control_offset_px`` away); ``excess = gain - control_gain`` is the reported evidence.
A trace shorter than ``min_length_px`` is UNRESOLVABLE (reported, never forced to a number). These are observations
of profile correspondence, not measured fault slip: the G05 ``offset_proxy`` column was a curvature measure and is
kept under that name (``curvature_deg_per_km``), never presented as displacement.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import path_order

STATUSES = ("MEASURED", "UNRESOLVABLE_SHORT", "UNRESOLVABLE_NODATA")


def trace_frame(rr, cc, order, smooth_px: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(row, col, unit normal row, unit normal col) along the ordered trace; tangent from the smoothed path."""
    r = np.asarray(rr, float)[order]
    c = np.asarray(cc, float)[order]
    if smooth_px > 0 and len(r) > 3:
        rs, cs = ndimage.gaussian_filter1d(r, smooth_px, mode="nearest"), ndimage.gaussian_filter1d(c, smooth_px,
                                                                                                    mode="nearest")
    else:
        rs, cs = r, c
    tr, tc = np.gradient(rs), np.gradient(cs)
    n = np.hypot(tr, tc)
    n[n == 0] = 1.0
    return r, c, tc / n, -tr / n


def prepare(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(NaN-filled float64 field, float64 NaN indicator) for repeated bilinear sampling."""
    f = np.asarray(field, np.float64)
    bad = ~np.isfinite(f)
    return np.where(bad, 0.0, f), bad.astype(np.float64)


def sample(prep, r, c) -> np.ndarray:
    """Bilinear samples of a ``prepare``d field (or a raw array); NaN outside the array or where a neighbouring pixel
    is NaN."""
    f, bad = prep if isinstance(prep, tuple) else prepare(prep)
    r, c = np.asarray(r, float), np.asarray(c, float)
    v = ndimage.map_coordinates(f, [r, c], order=1, mode="constant", cval=0.0)
    b = ndimage.map_coordinates(bad, [r, c], order=1, mode="constant", cval=1.0)
    out = np.where(b > 1e-6, np.nan, v)
    out[(r < 0) | (c < 0) | (r > f.shape[0] - 1) | (c > f.shape[1] - 1)] = np.nan
    return out


def pearson(a, b, min_n: int = 8) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n:
        return float("nan")
    x, y = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def best_shift(A, B, max_shift: int, min_n: int = 8) -> tuple[int | None, float, float]:
    """(best shift k, corr at k, corr at 0): corr(A[s], B[s + k])."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    n = len(A)
    c0 = pearson(A, B, min_n)
    best_k, best_c = None, -np.inf
    for k in range(-int(max_shift), int(max_shift) + 1):
        if k >= 0:
            c = pearson(A[:n - k], B[k:], min_n)
        else:
            c = pearson(A[-k:], B[:n + k], min_n)
        if np.isfinite(c) and c > best_c:
            best_k, best_c = k, c
    return best_k, (float(best_c) if best_k is not None else float("nan")), c0


def _pair(field, r, c, nr, nc, d1: float, d2: float):
    return sample(field, r + d1 * nr, c + d1 * nc), sample(field, r + d2 * nr, c + d2 * nc)


def smoothed(field: np.ndarray, sigmas) -> dict:
    """{sigma: ``prepare``d field smoothed at sigma px} (NaN kept where the input is NaN); computed once per unit and
    field, then sampled for every segment."""
    f = np.asarray(field, np.float32)
    fin = np.isfinite(f)
    out = {}
    for s in sigmas:
        if float(s) <= 0 or not fin.any():
            out[float(s)] = prepare(f)
        else:
            g = ndimage.gaussian_filter(np.where(fin, f, np.nanmedian(f[fin])), float(s))
            out[float(s)] = prepare(np.where(fin, g, np.nan))
    return out


def segment_evidence(fields: dict, rr, cc, dc: dict, order=None) -> dict:
    """Displacement-style evidence of one trace on one field (``fields`` = ``smoothed(field, sigmas)``, ``dc`` =
    config ``displacement``)."""
    rr, cc = np.asarray(rr), np.asarray(cc)
    order = path_order(rr, cc) if order is None else order
    out = {"status": "UNRESOLVABLE_SHORT", "n_samples": int(len(order))}
    if len(order) < int(dc["min_length_px"]):
        return out
    K = int(dc["max_shift_px"])
    r, c, nr, nc = trace_frame(rr, cc, order)
    shifts, gains, ctrl = [], [], []
    for sig in dc["smoothing_sigma_px"]:
        f = fields[float(sig)]
        for d in dc["side_px"]:
            A, B = _pair(f, r, c, nr, nc, +float(d), -float(d))
            k, cb, c0 = best_shift(A, B, K)
            if k is None:
                continue
            shifts.append(k)
            gains.append(cb - c0 if np.isfinite(c0) else np.nan)
            m = float(dc["control_offset_px"])
            Ac, Bc = _pair(f, r, c, nr, nc, m + float(d), m - float(d))
            kc, cbc, c0c = best_shift(Ac, Bc, K)
            ctrl.append(cbc - c0c if kc is not None and np.isfinite(c0c) else np.nan)
            if float(sig) == float(dc["smoothing_sigma_px"][0]) and d == dc["side_px"][0]:
                h = len(A) // 2
                kh1 = best_shift(A[:h], B[:h], K)[0]
                kh2 = best_shift(A[h:], B[h:], K)[0]
                out.update(shift_half1_px=kh1, shift_half2_px=kh2, corr_best=cb, corr_zero=c0)
    if not shifts:
        out["status"] = "UNRESOLVABLE_NODATA"
        return out
    s = np.asarray(shifts, float)
    halves = [out.get("shift_half1_px"), out.get("shift_half2_px")]
    sign_ok = len({int(np.sign(x)) for x in s if x != 0}) <= 1
    spread_ok = float(s.max() - s.min()) <= 2.0
    half_ok = None not in halves and abs(halves[0] - halves[1]) <= 2 and np.sign(halves[0]) == np.sign(halves[1])
    g = float(np.nanmean(gains)) if np.isfinite(gains).any() else np.nan
    gc_ = float(np.nanmean(ctrl)) if np.isfinite(ctrl).any() else np.nan
    out.update(status="MEASURED", shift_px=float(np.median(s)), shift_abs_px=float(abs(np.median(s))), gain=g,
               control_gain=gc_, excess=(g - gc_) if np.isfinite(g) and np.isfinite(gc_) else np.nan,
               consistent=bool(sign_ok and spread_ok and half_ok and np.median(s) != 0))
    return out
