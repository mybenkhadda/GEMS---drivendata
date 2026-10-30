"""Transformation library of the GEMS Geological Feature Discovery Studio.

Conventions (apply to every function)
-------------------------------------
* Arrays are 2-D float32 on the native grid; **NaN = invalid / nodata**. Every operation propagates invalidity:
  a pixel that is NaN in an input is NaN in the output (neighbourhood filters use NaN-normalised convolution for
  valid pixels, so a valid pixel next to a gap is estimated from its valid neighbours only; the gap itself stays NaN).
  Nothing is ever silently replaced by 0 or another "meaningful" value.
* ``sigma`` / ``size`` arguments are in **pixels** (GEMS: 1 px = 100 m, so sigma=10 is a 1 km Gaussian sigma).
* Gradients are **per pixel**: ``gradient_x`` = d/d(east), ``gradient_y`` = d/d(north) (rows run southward, so the
  row derivative is negated). Multiply by 10 for "per km" at 100 m.
* Angles are **degrees, azimuth clockwise from north, axial in [0, 180)**: 0 and 180 denote the same structural axis.
  ``orientation`` returns the local STRIKE (the line direction, perpendicular to the dominant gradient).
* Normalisations that use statistics of the evaluated area (``zscore`` without center / scale, ``percentile_rank``,
  ``minmax`` without lo / hi, ...) are **AOI-dependent**: valid for interactive exploration, refused by full-footprint
  export until frozen (``featurelang.freeze_normalization`` writes the development statistics into the formula).

Each public function is described by an ``OpSpec`` in ``OPS``: which arguments are arrays, the halo (pixels of
context needed so a windowed result equals the full-grid result) and whether it is AOI-dependent.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import ndimage

OPS_VERSION = "gll-ops.1"
MAX_SIGMA_PX = 80.0
MAX_SIZE_PX = 61
TRUNCATE = 4.0


class OpError(ValueError):
    """Invalid argument of a transformation (raised before computing)."""


# =============================================================================== recording (diagnostics / freezing)
class Recorder:
    """Per-evaluation notes: division instability counts and the statistics an AOI-dependent normalisation used
    (``site`` = formula location), so they can be reported and frozen."""

    def __init__(self):
        self.division: list[dict] = []
        self.normalization: dict[str, dict] = {}
        self.site: str | None = None


_REC = Recorder()


def set_recorder(r: Recorder | None) -> Recorder:
    global _REC
    _REC = r or Recorder()
    return _REC


# =============================================================================== helpers
def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _sigma(s, name="sigma", allow_zero=False) -> float:
    s = float(s)
    lo = 0.0 if allow_zero else 0.25
    if not math.isfinite(s) or s < lo or s > MAX_SIGMA_PX:
        raise OpError(f"{name} must be in [{lo}, {MAX_SIGMA_PX}] pixels (got {s})")
    return s


def _size(n, name="size") -> int:
    n = int(n)
    if n < 3 or n > MAX_SIZE_PX or n % 2 == 0:
        raise OpError(f"{name} must be an odd integer in [3, {MAX_SIZE_PX}] pixels (got {n})")
    return n


def _eps(e, name="eps") -> float:
    e = float(e)
    if not math.isfinite(e) or e <= 0:
        raise OpError(f"{name} must be positive and finite (got {e})")
    return e


def _keep_nan(out, *inputs):
    bad = np.zeros(np.shape(out), bool)
    for x in inputs:
        if np.ndim(x) == 2:
            bad |= ~np.isfinite(x)
    out = np.asarray(out, np.float32)
    out = np.where(np.isfinite(out) & ~bad, out, np.nan).astype(np.float32)
    return out


def _valid(x) -> np.ndarray:
    v = _f32(x)
    return v[np.isfinite(v)]


# =============================================================================== A. mathematics
def f_abs(x):
    return _keep_nan(np.abs(_f32(x)), x)


def f_sqrt(x):
    """sqrt; negative inputs -> NaN (never a clipped value)."""
    with np.errstate(invalid="ignore"):
        return _keep_nan(np.sqrt(_f32(x)), x)


def f_log(x):
    """natural log; x <= 0 -> NaN."""
    a = _f32(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _keep_nan(np.where(a > 0, np.log(np.where(a > 0, a, 1)), np.nan), x)


def f_log1p(x):
    a = _f32(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _keep_nan(np.where(a > -1, np.log1p(np.where(a > -1, a, 0)), np.nan), x)


def f_exp(x):
    """exp; overflow -> NaN (reported as non-finite)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return _keep_nan(np.exp(_f32(x)), x)


def f_clip(x, lo, hi):
    lo, hi = float(lo), float(hi)
    if lo > hi:
        raise OpError("clip: lo must be <= hi")
    return _keep_nan(np.clip(_f32(x), lo, hi), x)


def f_power(x, p):
    p = float(p)
    if abs(p) > 5:
        raise OpError("power exponent must be within [-5, 5]")
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return _keep_nan(np.power(_f32(x), p), x)


def f_where(cond, a, b):
    """where(cond, a, b); NaN where ``cond`` is NaN (comparisons with an invalid pixel are invalid, not False)."""
    c = _f32(cond)
    A = np.broadcast_to(_f32(a), c.shape) if np.ndim(a) else np.full(c.shape, float(a), np.float32)
    B = np.broadcast_to(_f32(b), c.shape) if np.ndim(b) else np.full(c.shape, float(b), np.float32)
    out = np.where(c > 0.5, A, B).astype(np.float32)
    return np.where(np.isfinite(c), out, np.nan).astype(np.float32)


def f_minimum(a, b):
    return _keep_nan(np.minimum(_f32(a), _f32(b)), a, b)


def f_maximum(a, b):
    return _keep_nan(np.maximum(_f32(a), _f32(b)), a, b)


def ratio(a, b, eps=1e-5):
    """a / b with the denominator floored to +-eps (sign kept; b == 0 -> +eps). Floored pixels are counted."""
    eps = _eps(eps)
    den = _f32(b)
    small = np.abs(den) < eps
    _REC.division.append({"site": _REC.site, "op": "ratio", "floored_share": float(np.nanmean(small))
                          if den.size else 0.0})
    safe = np.where(small, np.where(den < 0, -eps, eps), den)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _keep_nan(_f32(a) / safe, a, b)


def safe_divide(a, b, eps=1e-3):
    """Tikhonov-regularised division a*b / (b^2 + eps^2): equals a/b when |b| >> eps, bounded by |a| / (2 eps), smooth
    through b = 0 (no explosion, no sign flip)."""
    eps = _eps(eps)
    A, B = _f32(a), _f32(b)
    _REC.division.append({"site": _REC.site, "op": "safe_divide",
                          "floored_share": float(np.nanmean(np.abs(B) < eps)) if B.size else 0.0})
    return _keep_nan(A * B / (B * B + eps * eps), a, b)


def normdiff(a, b, eps=1e-5):
    """(a - b) / (|a| + |b| + eps), in [-1, 1]."""
    eps = _eps(eps)
    A, B = _f32(a), _f32(b)
    return _keep_nan((A - B) / (np.abs(A) + np.abs(B) + eps), a, b)


def product(a, b):
    return _keep_nan(_f32(a) * _f32(b), a, b)


# =============================================================================== B. normalisation
def _record(kind, **stats):
    if _REC.site is not None:
        _REC.normalization[_REC.site] = {"kind": kind, **{k: float(v) for k, v in stats.items()}}


def zscore(x, center=None, scale=None):
    """(x - center) / scale. Without center / scale: mean / std of the evaluated area (AOI-dependent)."""
    a = _f32(x)
    v = _valid(a)
    if center is None or scale is None:
        if not v.size:
            return np.full(a.shape, np.nan, np.float32)
        center, scale = float(v.mean()), float(v.std())
        _record("zscore", center=center, scale=scale)
    if not scale or float(scale) <= 1e-12:
        return np.full(a.shape, np.nan, np.float32)
    return _keep_nan((a - float(center)) / float(scale), x)


def robust_z(x, center=None, scale=None):
    """(x - median) / (1.4826 MAD). Without center / scale: AOI statistics (AOI-dependent)."""
    a = _f32(x)
    v = _valid(a)
    if center is None or scale is None:
        if not v.size:
            return np.full(a.shape, np.nan, np.float32)
        center = float(np.median(v))
        scale = float(1.4826 * np.median(np.abs(v - center)))
        _record("robust_z", center=center, scale=scale)
    if not scale or float(scale) <= 1e-12:
        return np.full(a.shape, np.nan, np.float32)
    return _keep_nan((a - float(center)) / float(scale), x)


def minmax(x, lo=None, hi=None):
    """(x - lo) / (hi - lo); without lo / hi: AOI min / max (AOI-dependent). Not clipped."""
    a = _f32(x)
    v = _valid(a)
    if lo is None or hi is None:
        if not v.size:
            return np.full(a.shape, np.nan, np.float32)
        lo, hi = float(v.min()), float(v.max())
        _record("minmax", lo=lo, hi=hi)
    if float(hi) - float(lo) <= 1e-12:
        return np.full(a.shape, np.nan, np.float32)
    return _keep_nan((a - float(lo)) / (float(hi) - float(lo)), x)


def percentile_rank(x):
    """Rank of each valid pixel among the valid pixels of the evaluated area, in [0, 1] (always AOI-dependent)."""
    from scipy.stats import rankdata

    a = _f32(x)
    out = np.full(a.shape, np.nan, np.float32)
    ok = np.isfinite(a)
    if ok.sum() > 1:
        out[ok] = ((rankdata(a[ok]) - 1) / (ok.sum() - 1)).astype(np.float32)
    return out


def winsorize(x, lower=1.0, upper=99.0, lo=None, hi=None):
    """Clip to percentiles [lower, upper] of the evaluated area, or to frozen lo / hi values."""
    a = _f32(x)
    if lo is None or hi is None:
        v = _valid(a)
        if not v.size:
            return np.full(a.shape, np.nan, np.float32)
        if not 0 <= float(lower) < float(upper) <= 100:
            raise OpError("winsorize: 0 <= lower < upper <= 100")
        lo, hi = (float(q) for q in np.percentile(v, [float(lower), float(upper)]))
        _record("winsorize", lo=lo, hi=hi)
    return _keep_nan(np.clip(a, float(lo), float(hi)), x)


def quantile_clip(x, q_lo=0.01, q_hi=0.99, lo=None, hi=None):
    """winsorize with quantiles in [0, 1]."""
    return winsorize(x, 100 * float(q_lo), 100 * float(q_hi), lo, hi)


# =============================================================================== C. spatial filters
def gaussian(x, sigma=3.0):
    """NaN-normalised Gaussian smoothing (sigma px); NaN input pixels stay NaN."""
    s = _sigma(sigma)
    a = _f32(x)
    m = np.isfinite(a)
    if not m.any():
        return np.full(a.shape, np.nan, np.float32)
    w = ndimage.gaussian_filter(m.astype(np.float32), s, mode="reflect", truncate=TRUNCATE)
    v = ndimage.gaussian_filter(np.where(m, a, 0).astype(np.float32), s, mode="reflect", truncate=TRUNCATE)
    out = np.divide(v, w, out=np.full(a.shape, np.nan, np.float32), where=w > 1e-6)
    return np.where(m, out, np.nan).astype(np.float32)


def lowpass(x, sigma=5.0):
    return gaussian(x, sigma)


def local_mean(x, sigma=3.0):
    return gaussian(x, sigma)


def highpass(x, sigma=5.0):
    """x - gaussian(x, sigma): residual of structures smaller than ~sigma px."""
    return _keep_nan(_f32(x) - gaussian(x, sigma), x)


def bandpass(x, sigma_small=2.0, sigma_large=8.0):
    """Difference of Gaussians gaussian(x, small) - gaussian(x, large)."""
    if float(sigma_small) >= float(sigma_large):
        raise OpError("bandpass: sigma_small must be < sigma_large")
    return _keep_nan(gaussian(x, sigma_small) - gaussian(x, sigma_large), x)


def local_var(x, sigma=3.0):
    a = _f32(x)
    m1 = gaussian(a, sigma)
    m2 = gaussian(a * a, sigma)
    return _keep_nan(np.maximum(m2 - m1 * m1, 0.0), x)


def local_std(x, sigma=3.0):
    return _keep_nan(np.sqrt(local_var(x, sigma)), x)


def local_zscore(x, sigma=10.0):
    """(x - local mean) / local std over a Gaussian neighbourhood (not AOI-dependent)."""
    a = _f32(x)
    sd = local_std(a, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _keep_nan(np.where(sd > 1e-12, (a - gaussian(a, sigma)) / sd, np.nan), x)


def _fill_local(a, sigma=3.0):
    """Valid-neighbour estimate inside gaps (used ONLY as filter padding; gaps are restored to NaN afterwards)."""
    m = np.isfinite(a)
    if m.all():
        return a
    w = ndimage.gaussian_filter(m.astype(np.float32), sigma)
    v = ndimage.gaussian_filter(np.where(m, a, 0).astype(np.float32), sigma)
    est = np.divide(v, w, out=np.zeros(a.shape, np.float32), where=w > 1e-6)
    med = float(np.median(a[m])) if m.any() else 0.0
    return np.where(m, a, np.where(w > 1e-6, est, med)).astype(np.float32)


def median_filter(x, size=5):
    """Median over a size x size window (px). Gaps are padded with a local estimate for the filter only and
    restored to NaN; valid pixels near large gaps are approximate (documented)."""
    n = _size(size)
    a = _f32(x)
    if not np.isfinite(a).any():
        return np.full(a.shape, np.nan, np.float32)
    return _keep_nan(ndimage.median_filter(_fill_local(a, n / 2), size=n, mode="reflect"), x)


def local_robust_z(x, size=15):
    """(x - local median) / (1.4826 local MAD), size x size px window."""
    a = _f32(x)
    med = median_filter(a, size)
    mad = 1.4826 * median_filter(np.abs(a - med), size)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _keep_nan(np.where(mad > 1e-12, (a - med) / mad, np.nan), x)


def _grad(x, sigma):
    a = gaussian(x, sigma) if float(sigma) > 0 else _f32(x)
    gr, gc = np.gradient(a)
    return gc.astype(np.float32), (-gr).astype(np.float32)          # (d/d east, d/d north) per pixel


def gradient_x(x, sigma=1.0):
    """d/d(east) per pixel after Gaussian smoothing (sigma px; 0 = none)."""
    _sigma(sigma, allow_zero=True)
    return _keep_nan(_grad(x, sigma)[0], x)


def gradient_y(x, sigma=1.0):
    """d/d(north) per pixel after Gaussian smoothing."""
    _sigma(sigma, allow_zero=True)
    return _keep_nan(_grad(x, sigma)[1], x)


def gradmag(x, sigma=1.0):
    """|grad x| per pixel after Gaussian smoothing (sigma px)."""
    _sigma(sigma, allow_zero=True)
    gx, gy = _grad(x, sigma)
    return _keep_nan(np.hypot(gx, gy), x)


def directional_gradient(x, azimuth=0.0, sigma=1.0):
    """Signed derivative along the azimuth (deg clockwise from north) per pixel."""
    _sigma(sigma, allow_zero=True)
    gx, gy = _grad(x, sigma)
    t = math.radians(float(azimuth))
    return _keep_nan(gx * math.sin(t) + gy * math.cos(t), x)


def sobel(x, sigma=0.0):
    """Sobel gradient magnitude (per pixel, kernel normalised by 8); optional pre-smoothing."""
    _sigma(sigma, allow_zero=True)
    a = gaussian(x, sigma) if float(sigma) > 0 else _f32(x)
    return _keep_nan(np.hypot(ndimage.sobel(a, 1), ndimage.sobel(a, 0)) / 8.0, x)


def laplacian(x, sigma=1.0):
    """Laplacian (per pixel^2) of the Gaussian-smoothed field."""
    _sigma(sigma, allow_zero=True)
    a = gaussian(x, sigma) if float(sigma) > 0 else _f32(x)
    return _keep_nan(ndimage.laplace(a), x)


def _hess(x, sigma):
    a = gaussian(x, sigma)
    gr, gc = np.gradient(a)
    hrr, hrc = np.gradient(gr)
    _, hcc = np.gradient(gc)
    # east / north frame: d2/de2 = hcc, d2/dn2 = hrr, d2/de dn = -hrc
    return hcc, hrr, -hrc


def _eig2(a, b, c):
    """eigenvalues (l1 >= l2) of [[a, c], [c, b]]."""
    tr, d = (a + b) / 2, np.sqrt(((a - b) / 2) ** 2 + c * c)
    return tr + d, tr - d


def hessian(x, sigma=2.0, which="max"):
    """Eigenvalue of the Hessian of the smoothed field (per pixel^2): which = 'max' | 'min' | 'det'."""
    _sigma(sigma)
    hee, hnn, hen = _hess(x, sigma)
    l1, l2 = _eig2(hee, hnn, hen)
    if which == "max":
        out = l1
    elif which == "min":
        out = l2
    elif which == "det":
        out = l1 * l2
    else:
        raise OpError("hessian: which must be 'max', 'min' or 'det'")
    return _keep_nan(out, x)


def ridge_strength(x, sigma=3.0, polarity="both"):
    """Scale-normalised line strength sigma^2 * |curvature across the ridge|: polarity 'bright' (ridges, max of
    -lambda_min), 'dark' (valleys, max of lambda_max) or 'both'."""
    _sigma(sigma)
    hee, hnn, hen = _hess(x, sigma)
    l1, l2 = _eig2(hee, hnn, hen)
    s2 = float(sigma) ** 2
    if polarity == "bright":
        out = np.maximum(-l2, 0) * s2
    elif polarity == "dark":
        out = np.maximum(l1, 0) * s2
    elif polarity == "both":
        out = np.maximum(np.abs(l1), np.abs(l2)) * s2
    else:
        raise OpError("ridge_strength: polarity must be 'bright', 'dark' or 'both'")
    return _keep_nan(out, x)


def ridge_sharpness(x, sigma=2.0):
    """ridge_strength(sigma) / ridge_strength(2 sigma): > 1 for narrow lines, < 1 for broad flexures."""
    a, b = ridge_strength(x, sigma), ridge_strength(x, 2 * float(sigma))
    scale = float(np.nanmedian(b)) if np.isfinite(b).any() else 0.0
    eps = max(scale * 1e-3, 1e-12)
    return _keep_nan(a / (b + eps), x)


# =============================================================================== D. geological transforms
def _tensor(x, sigma, grad_sigma=1.0):
    gx, gy = _grad(x, grad_sigma)
    ok = np.isfinite(gx) & np.isfinite(gy)
    gx, gy = np.where(ok, gx, np.nan), np.where(ok, gy, np.nan)
    jee, jnn, jen = gaussian(gx * gx, sigma), gaussian(gy * gy, sigma), gaussian(gx * gy, sigma)
    return jee, jnn, jen


def orientation(x, sigma=5.0, grad_sigma=1.0):
    """Local STRIKE azimuth (deg, clockwise from north, axial [0, 180)) from the structure tensor: perpendicular to
    the dominant gradient direction averaged over a Gaussian window of sigma px (doubled-angle mean)."""
    _sigma(sigma)
    jee, jnn, jen = _tensor(x, sigma, grad_sigma)
    grad_az = 0.5 * np.degrees(np.arctan2(2 * jen, jnn - jee))      # dominant gradient azimuth (axial)
    strike = np.mod(grad_az + 90.0, 180.0)
    strike = np.where(strike >= 180.0 - 1e-9, 0.0, strike)
    return _keep_nan(strike, x)


def coherence(x, sigma=5.0, grad_sigma=1.0):
    """Structure-tensor coherence ((l1 - l2) / (l1 + l2))^2 in [0, 1]: 1 = one dominant direction."""
    _sigma(sigma)
    jee, jnn, jen = _tensor(x, sigma, grad_sigma)
    tr = jee + jnn
    d = np.sqrt((jnn - jee) ** 2 + 4 * jen * jen)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _keep_nan(np.where(tr > 1e-20, (d / tr) ** 2, np.nan), x)


def gradient_anisotropy(x, sigma=5.0, grad_sigma=1.0):
    """(l1 - l2) / (l1 + l2) in [0, 1] (square root of coherence)."""
    return _keep_nan(np.sqrt(coherence(x, sigma, grad_sigma)), x)


def axial_agreement(theta_a, theta_b):
    """cos^2 of the axial angle difference, in [0, 1]: 1 = parallel structures, 0 = perpendicular. Inputs in deg."""
    d = np.radians(_f32(theta_a) - _f32(theta_b))
    return _keep_nan(0.5 * (1 + np.cos(2 * d)), theta_a, theta_b)


def axial_difference(theta_a, theta_b):
    """Smallest axial angle between two orientations, deg in [0, 90]."""
    d = np.abs(np.mod(_f32(theta_a) - _f32(theta_b), 180.0))
    return _keep_nan(np.minimum(d, 180.0 - d), theta_a, theta_b)


def curvature(x, sigma=2.0, kind="mean", pixel_m=100.0):
    """Surface curvature of the smoothed field in 1/m (field units per metre): kind 'mean' (Laplacian / 2),
    'profile' or 'plan' (Zevenbergen-Thorne). pixel_m converts pixel derivatives to metres."""
    _sigma(sigma)
    px = float(pixel_m)
    gx, gy = (g / px for g in _grad(x, sigma))
    hee, hnn, hen = (h / px ** 2 for h in _hess(x, sigma))
    p2q2 = gx * gx + gy * gy
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "mean":
            out = 0.5 * (hee + hnn)
        elif kind == "profile":
            out = -(hee * gx * gx + 2 * hen * gx * gy + hnn * gy * gy) / (p2q2 * (1 + p2q2) ** 1.5)
            out = np.where(p2q2 > 1e-20, out, np.nan)
        elif kind == "plan":
            out = -(hnn * gx * gx - 2 * hen * gx * gy + hee * gy * gy) / (p2q2 ** 1.5)
            out = np.where(p2q2 > 1e-20, out, np.nan)
        else:
            raise OpError("curvature: kind must be 'mean', 'profile' or 'plan'")
    return _keep_nan(out, x)


def multiscale_persistence(x, sigma_min=1.0, sigma_max=8.0, n=4):
    """min / max over n geometric scales of the scale-normalised gradient sigma * |grad| (in [0, 1]): high where an
    edge persists across scales (fault-like), low for edges present at one scale only."""
    s0, s1 = _sigma(sigma_min), _sigma(sigma_max)
    n = int(n)
    if s0 >= s1 or not 2 <= n <= 8:
        raise OpError("multiscale_persistence: sigma_min < sigma_max and 2 <= n <= 8")
    stack = np.stack([gradmag(x, s) * s for s in np.geomspace(s0, s1, n)])
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = np.nanmax(stack, 0) if np.isfinite(stack).any() else np.full(stack.shape[1:], np.nan)
        out = np.where(mx > 1e-20, np.nanmin(stack, 0) / mx, np.nan)
    return _keep_nan(out, x)


def texture_entropy(x, sigma=3.0, bins=16, lo=None, hi=None):
    """Local Shannon entropy (nats) of the value histogram in a Gaussian window. Bin edges from lo / hi (frozen) or
    the 1st / 99th percentiles of the evaluated area (AOI-dependent)."""
    _sigma(sigma)
    bins = int(bins)
    if not 4 <= bins <= 64:
        raise OpError("texture_entropy: bins in [4, 64]")
    a = _f32(x)
    if lo is None or hi is None:
        v = _valid(a)
        if not v.size:
            return np.full(a.shape, np.nan, np.float32)
        lo, hi = (float(q) for q in np.percentile(v, [1, 99]))
        _record("texture_entropy", lo=lo, hi=hi)
    if float(hi) <= float(lo):
        return np.full(a.shape, np.nan, np.float32)
    k = np.clip(((a - float(lo)) / (float(hi) - float(lo)) * bins).astype(np.int32), 0, bins - 1)
    ok = np.isfinite(a)
    H = np.zeros(a.shape, np.float32)
    w = gaussian(np.where(ok, 1.0, np.nan), sigma)
    for b in range(bins):
        p = gaussian(np.where(ok, (k == b).astype(np.float32), np.nan), sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(w > 0, p / w, 0)
            H -= np.where(p > 0, p * np.log(p), 0).astype(np.float32)
    return _keep_nan(H, x)


def grad_ratio(a, b, sigma=3.0, eps=1e-3):
    """|grad a| / |grad b| (floored denominator) after smoothing both at sigma px."""
    return ratio(gradmag(a, sigma), gradmag(b, sigma), eps)


def grad_diff(a, b, sigma=3.0, signed=True):
    """|grad a| - |grad b| (signed) or its absolute value. Normalise inputs first when units differ."""
    d = gradmag(a, sigma) - gradmag(b, sigma)
    return _keep_nan(d if signed else np.abs(d), a, b)


def contrast(a, b, eps=1e-5):
    """Cross-layer normalised contrast (|a| - |b|) / (|a| + |b| + eps)."""
    return normdiff(f_abs(a), f_abs(b), eps)


# =============================================================================== registry
def _halo_sigma(*names, mult=1.0, extra=2):
    def h(kw):
        s = max([float(kw.get(n, 0) or 0) for n in names] + [0.0])
        return int(math.ceil(TRUNCATE * s * mult)) + extra if s > 0 else extra
    return h


def _halo_size(kw):
    return int(kw.get("size", 5)) // 2 + 1


def _halo_median_fill(kw):
    n = int(kw.get("size", 5))
    return n // 2 + int(math.ceil(TRUNCATE * n / 2)) + 1


def _aoi_if_missing(*pairs):
    def d(kw):
        return any(kw.get(p) is None for p in pairs)
    return d


@dataclass(frozen=True)
class OpSpec:
    fn: Callable
    arrays: tuple                        # parameter names that take arrays
    halo: Callable = field(default=lambda kw: 0)
    aoi_dependent: Callable = field(default=lambda kw: False)
    group: str = "math"
    doc: str = ""

    def signature(self) -> inspect.Signature:
        return inspect.signature(self.fn)


def _spec(fn, arrays, halo=None, aoi=None, group="math"):
    return OpSpec(fn, tuple(arrays), halo or (lambda kw: 0), aoi or (lambda kw: False), group,
                  (inspect.getdoc(fn) or "").split("\n")[0])


OPS: dict[str, OpSpec] = {
    # A. mathematics
    "abs": _spec(f_abs, ["x"]), "sqrt": _spec(f_sqrt, ["x"]), "log": _spec(f_log, ["x"]),
    "log1p": _spec(f_log1p, ["x"]), "exp": _spec(f_exp, ["x"]), "clip": _spec(f_clip, ["x"]),
    "power": _spec(f_power, ["x"]), "where": _spec(f_where, ["cond", "a", "b"]),
    "minimum": _spec(f_minimum, ["a", "b"]), "maximum": _spec(f_maximum, ["a", "b"]),
    "ratio": _spec(ratio, ["a", "b"]), "safe_divide": _spec(safe_divide, ["a", "b"]),
    "normdiff": _spec(normdiff, ["a", "b"]), "product": _spec(product, ["a", "b"]),
    # B. normalisation
    "zscore": _spec(zscore, ["x"], aoi=_aoi_if_missing("center", "scale"), group="normalization"),
    "robust_z": _spec(robust_z, ["x"], aoi=_aoi_if_missing("center", "scale"), group="normalization"),
    "minmax": _spec(minmax, ["x"], aoi=_aoi_if_missing("lo", "hi"), group="normalization"),
    "percentile_rank": _spec(percentile_rank, ["x"], aoi=lambda kw: True, group="normalization"),
    "winsorize": _spec(winsorize, ["x"], aoi=_aoi_if_missing("lo", "hi"), group="normalization"),
    "quantile_clip": _spec(quantile_clip, ["x"], aoi=_aoi_if_missing("lo", "hi"), group="normalization"),
    "local_zscore": _spec(local_zscore, ["x"], _halo_sigma("sigma", mult=1.0, extra=3), group="normalization"),
    "local_robust_z": _spec(local_robust_z, ["x"], lambda kw: 2 * _halo_median_fill(kw), group="normalization"),
    # C. spatial filters
    "gaussian": _spec(gaussian, ["x"], _halo_sigma("sigma"), group="filter"),
    "lowpass": _spec(lowpass, ["x"], _halo_sigma("sigma"), group="filter"),
    "local_mean": _spec(local_mean, ["x"], _halo_sigma("sigma"), group="filter"),
    "highpass": _spec(highpass, ["x"], _halo_sigma("sigma"), group="filter"),
    "bandpass": _spec(bandpass, ["x"], _halo_sigma("sigma_small", "sigma_large"), group="filter"),
    "local_var": _spec(local_var, ["x"], _halo_sigma("sigma"), group="filter"),
    "local_std": _spec(local_std, ["x"], _halo_sigma("sigma"), group="filter"),
    "median_filter": _spec(median_filter, ["x"], _halo_median_fill, group="filter"),
    "laplacian": _spec(laplacian, ["x"], _halo_sigma("sigma", extra=3), group="filter"),
    "sobel": _spec(sobel, ["x"], _halo_sigma("sigma", extra=3), group="filter"),
    "gradmag": _spec(gradmag, ["x"], _halo_sigma("sigma", extra=3), group="filter"),
    "gradient_x": _spec(gradient_x, ["x"], _halo_sigma("sigma", extra=3), group="filter"),
    "gradient_y": _spec(gradient_y, ["x"], _halo_sigma("sigma", extra=3), group="filter"),
    "directional_gradient": _spec(directional_gradient, ["x"], _halo_sigma("sigma", extra=3), group="filter"),
    "hessian": _spec(hessian, ["x"], _halo_sigma("sigma", extra=4), group="filter"),
    "ridge_strength": _spec(ridge_strength, ["x"], _halo_sigma("sigma", extra=4), group="filter"),
    "ridge_sharpness": _spec(ridge_sharpness, ["x"], _halo_sigma("sigma", mult=2.0, extra=4),
                             aoi=lambda kw: False, group="geology"),
    # D. geological transforms
    "orientation": _spec(orientation, ["x"], lambda kw: _halo_sigma("sigma")(kw) + _halo_sigma(
        "grad_sigma", extra=3)({"grad_sigma": kw.get("grad_sigma", 1.0)}), group="geology"),
    "coherence": _spec(coherence, ["x"], lambda kw: _halo_sigma("sigma")(kw) + _halo_sigma(
        "grad_sigma", extra=3)({"grad_sigma": kw.get("grad_sigma", 1.0)}), group="geology"),
    "gradient_anisotropy": _spec(gradient_anisotropy, ["x"], lambda kw: _halo_sigma("sigma")(kw) + _halo_sigma(
        "grad_sigma", extra=3)({"grad_sigma": kw.get("grad_sigma", 1.0)}), group="geology"),
    "axial_agreement": _spec(axial_agreement, ["theta_a", "theta_b"], group="geology"),
    "axial_difference": _spec(axial_difference, ["theta_a", "theta_b"], group="geology"),
    "curvature": _spec(curvature, ["x"], _halo_sigma("sigma", extra=4), group="geology"),
    "multiscale_persistence": _spec(multiscale_persistence, ["x"], _halo_sigma("sigma_max", extra=3),
                                    group="geology"),
    "texture_entropy": _spec(texture_entropy, ["x"], _halo_sigma("sigma"), aoi=_aoi_if_missing("lo", "hi"),
                             group="geology"),
    "grad_ratio": _spec(grad_ratio, ["a", "b"], _halo_sigma("sigma", extra=3), group="geology"),
    "grad_diff": _spec(grad_diff, ["a", "b"], _halo_sigma("sigma", extra=3), group="geology"),
    "contrast": _spec(contrast, ["a", "b"], group="geology"),
}


def bind(name: str, args: list, kwargs: dict) -> dict:
    """Map positional / keyword arguments of op ``name`` to parameter names (signature defaults applied)."""
    spec = OPS[name]
    try:
        b = spec.signature().bind(*args, **kwargs)
    except TypeError as e:
        raise OpError(f"{name}{spec.signature()}: {e}") from e
    b.apply_defaults()
    return dict(b.arguments)


def reference_table() -> list[dict]:
    """Documentation rows for the UI (name, group, signature, array args, first doc line)."""
    return [{"function": k, "group": s.group, "signature": f"{k}{s.signature()}", "array_args": ", ".join(s.arrays),
             "description": s.doc} for k, s in sorted(OPS.items(), key=lambda kv: (kv[1].group, kv[0]))]
