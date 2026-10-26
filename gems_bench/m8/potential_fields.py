"""Physics-guided potential-field operators (pure numpy / scipy: no labels, no pipeline context).

Conventions
-----------
* Inputs are float32 rasters with NaN nodata. Derivatives are taken on a nearest-filled float64 copy (so nodata never
  propagates through a filter as NaN) and NaN is restored on the source footprint afterwards.
* ``sigma_px`` is the Gaussian sigma in pixels (``scale_m / pixel_m``, floored at 0.7 px). A 100 m output pixel does
  not imply 100 m information content: callers skip scales below a fraction of each survey's effective resolution.
* Horizontal derivatives: Gaussian derivative filters, in field units per km, as (east, north) components.
* Vertical derivative (VDR): wavenumber-domain |k| filter of the smoothed field (per km) with reflect padding
  (the standard potential-field relation dz F = |k| F for a source-free half-space). Valid for potential fields only.

Operators (competing definitions; none is declared "the" fault detector)::

    thdr             total horizontal derivative |grad_h f|
    analytic_signal  3-D analytic-signal amplitude sqrt(fx^2 + fy^2 + fz^2)       (potential fields, units consistent)
    tilt_edge        1 - |atan2(VDR, THDR)| / 90 deg      -> 1 on the tilt zero contour (source edges)
    tdx              atan2(THDR, |VDR|) / 90 deg          -> 1 on edges (TDX, Cooper & Cowan 2006)
    theta            THDR / AS (cos of the theta map)      -> 1 on edges, amplitude-balanced
    lcn              THDR / local RMS(THDR)                 local contrast normalisation (balances weak / strong areas)
    *_band           the same from PROVIDED derivative bands (hgrad / vgrad), only when verified compatible

Guards: tilt / TDX / theta are NaN where the analytic signal is below a low quantile of the footprint (the angle of a
near-zero vector is noise); LCN uses a denominator floor; every ratio has an explicit epsilon.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

EPS = 1e-12


# =============================================================================== basics
def fill(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(nearest-filled float64 copy, finite mask)."""
    from gems_eda.enhance import fill_nan

    m = np.isfinite(a)
    return fill_nan(np.asarray(a, np.float32)), m


def restore(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = np.asarray(a, np.float32).copy()
    out[~m] = np.nan
    return out


def sigma_px_of(scale_m: float, pixel_m: float = 100.0) -> float:
    return max(float(scale_m) / float(pixel_m), 0.7)


def smooth(f: np.ndarray, sigma_px: float) -> np.ndarray:
    return ndimage.gaussian_filter(f, sigma_px, mode="nearest")


def hgrad(f: np.ndarray, sigma_px: float, px_km: tuple[float, float] = (0.1, 0.1)) -> tuple[np.ndarray, np.ndarray]:
    """(east, north) Gaussian derivatives per km of a FILLED field; px_km = (row size, col size) in km."""
    d_r = ndimage.gaussian_filter(f, sigma_px, order=(1, 0), mode="nearest")
    d_c = ndimage.gaussian_filter(f, sigma_px, order=(0, 1), mode="nearest")
    return d_c / px_km[1], -d_r / px_km[0]


def _wavenumber(shape, px_km):
    ky = 2 * np.pi * np.fft.fftfreq(shape[0], d=px_km[0])
    kx = 2 * np.pi * np.fft.rfftfreq(shape[1], d=px_km[1])
    return np.hypot(ky[:, None], kx[None, :])


def _spectral(f: np.ndarray, px_km, mult) -> np.ndarray:
    """Apply a radial wavenumber multiplier ``mult(k)`` with reflect padding (1/4 of each side)."""
    h, w = f.shape
    ph, pw = max(h // 4, 1), max(w // 4, 1)
    g = f - float(f.mean())
    F = np.pad(g, ((ph, ph), (pw, pw)), mode="reflect")
    k = _wavenumber(F.shape, px_km)
    out = np.fft.irfft2(np.fft.rfft2(F) * mult(k), s=F.shape)[ph:ph + h, pw:pw + w]
    return out


def vdr_fft(f_smooth: np.ndarray, px_km=(0.1, 0.1)) -> np.ndarray:
    """First vertical derivative (per km) of a filled, already smoothed potential field."""
    return _spectral(f_smooth, px_km, lambda k: k)


def upward(f: np.ndarray, height_km: float, px_km=(0.1, 0.1)) -> np.ndarray:
    """Upward continuation by ``height_km`` (physical scale-space of a potential field); adds the mean back."""
    return _spectral(f, px_km, lambda k: np.exp(-k * float(height_km))) + float(f.mean())


def axial_diff(a, b) -> np.ndarray:
    """Smallest difference between two axial angles (deg), in [0, 90]."""
    return np.abs(((np.asarray(a, np.float64) - np.asarray(b, np.float64)) + 90.0) % 180.0 - 90.0)


def strike_from_gradient(ge: np.ndarray, gn: np.ndarray) -> np.ndarray:
    """Edge strike azimuth (deg from north, clockwise, [0, 180)): perpendicular to the gradient direction."""
    az = np.degrees(np.arctan2(ge, gn))
    return ((az + 90.0) % 180.0).astype(np.float32)


def coherence(ge: np.ndarray, gn: np.ndarray, sigma_int_px: float) -> np.ndarray:
    """Structure-tensor coherence in [0, 1] (1 = one dominant orientation: a lineament, not a blob)."""
    jee = ndimage.gaussian_filter(ge * ge, sigma_int_px, mode="nearest")
    jnn = ndimage.gaussian_filter(gn * gn, sigma_int_px, mode="nearest")
    jen = ndimage.gaussian_filter(ge * gn, sigma_int_px, mode="nearest")
    tr = jee + jnn
    lam = np.sqrt((jee - jnn) ** 2 + 4 * jen ** 2)
    return np.where(tr > EPS, lam / (tr + EPS), 0.0).astype(np.float32)


# =============================================================================== operators
OPERATOR_INFO = {
    "thdr": "total horizontal derivative |grad_h f| (field units / km)",
    "analytic_signal": "sqrt(fx^2 + fy^2 + fz^2), fz = |k| VDR (field units / km)",
    "tilt_edge": "1 - |atan2(VDR, THDR)| / 90 deg (dimensionless; 1 on tilt zero contour)",
    "tdx": "atan2(THDR, |VDR|) / 90 deg (dimensionless; 1 on edges)",
    "theta": "THDR / AS (dimensionless)",
    "lcn": "THDR / local RMS(THDR) (dimensionless)",
    "tilt_edge_band": "tilt edge from provided hgrad / vgrad bands (dimensionless)",
    "analytic_signal_band": "sqrt(hgrad^2 + vgrad^2) from provided bands (band units)",
    "tdx_band": "TDX from provided hgrad / vgrad bands (dimensionless)",
}
NEEDS_VDR = ("analytic_signal", "tilt_edge", "tdx", "theta")


def operator(op: str, field: np.ndarray, sigma_px: float, *, px_km=(0.1, 0.1), amp_floor_q: float = 0.02,
             lcn_sigma_px: float = 25.0, hg: np.ndarray | None = None, vg: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """One operator at one scale on a full raster. Returns (float32 layer with NaN outside the source footprint, info).

    ``hg`` / ``vg`` are the provided horizontal-gradient-amplitude / vertical-gradient bands (only for *_band)."""
    info = {"operator": op, "sigma_px": float(sigma_px), "definition": OPERATOR_INFO.get(op, "")}
    if op.endswith("_band"):
        if hg is None or vg is None:
            raise ValueError(f"{op}: provided derivative bands missing")
        h, mh = fill(hg)
        v, mv = fill(vg)
        m = mh & mv
        th, vz = np.abs(smooth(h, sigma_px)), smooth(v, sigma_px)
        return _angles(op.replace("_band", ""), th, vz, m, amp_floor_q, info)
    f, m = fill(field)
    if op == "lcn":
        ge, gn = hgrad(f, sigma_px, px_km)
        th = np.hypot(ge, gn)
        rms = np.sqrt(ndimage.gaussian_filter(th * th, lcn_sigma_px, mode="nearest"))
        floor = max(float(np.quantile(rms[m], 0.05)) if m.any() else 0.0, EPS)
        info["denominator_floor"] = floor
        return restore(th / np.maximum(rms, floor), m), info
    ge, gn = hgrad(f, sigma_px, px_km)
    th = np.hypot(ge, gn)
    if op == "thdr":
        return restore(th, m), info
    if op not in NEEDS_VDR:
        raise KeyError(op)
    vz = vdr_fft(smooth(f, sigma_px), px_km)
    return _angles(op, th, vz, m, amp_floor_q, info)


def _angles(op: str, th: np.ndarray, vz: np.ndarray, m: np.ndarray, q: float, info: dict) -> tuple[np.ndarray, dict]:
    asig = np.sqrt(th * th + vz * vz)
    if op == "analytic_signal":
        return restore(asig, m), info
    floor = max(float(np.quantile(asig[m], q)) if m.any() else 0.0, EPS)
    low = asig < floor
    info.update(amplitude_floor=floor, undefined_share=float(low[m].mean()) if m.any() else np.nan)
    if op == "tilt_edge":
        out = 1.0 - np.abs(np.arctan2(vz, th + EPS)) / (np.pi / 2)
    elif op == "tdx":
        out = np.arctan2(th, np.abs(vz) + EPS) / (np.pi / 2)
    elif op == "theta":
        out = th / (asig + EPS)
    else:
        raise KeyError(op)
    out = np.where(low, np.nan, out)
    return restore(out, m & ~low), info


def gradient_parts(field: np.ndarray, sigma_px: float, px_km=(0.1, 0.1)) -> dict:
    """(thdr, strike, coherence) of a field at one scale (label-free; used by worms, profiles, the walker)."""
    f, m = fill(field)
    ge, gn = hgrad(f, sigma_px, px_km)
    return {"thdr": restore(np.hypot(ge, gn), m), "strike": restore(strike_from_gradient(ge, gn), m),
            "coherence": restore(coherence(ge, gn, max(2.0 * sigma_px, 1.0)), m), "ge": ge, "gn": gn, "mask": m}


# =============================================================================== local thresholds
def block_quantile_map(a: np.ndarray, mask: np.ndarray, block_px: int, q: float, min_count: int = 200) -> np.ndarray:
    """Smooth map of the ``q`` quantile of ``a`` over ``mask`` in blocks of ``block_px`` (bilinear between block
    centres; blocks with too few samples take the global quantile). Used for local percentile thresholds."""
    h, w = a.shape
    nb_r, nb_c = max(int(np.ceil(h / block_px)), 1), max(int(np.ceil(w / block_px)), 1)
    v_all = a[mask & np.isfinite(a)]
    g = float(np.quantile(v_all, q)) if v_all.size else 0.0
    tab = np.full((nb_r, nb_c), g, np.float64)
    for i in range(nb_r):
        for j in range(nb_c):
            sl = (slice(i * block_px, min((i + 1) * block_px, h)), slice(j * block_px, min((j + 1) * block_px, w)))
            v = a[sl][mask[sl] & np.isfinite(a[sl])]
            if v.size >= min_count:
                tab[i, j] = float(np.quantile(v, q))
    if nb_r == 1 and nb_c == 1:
        return np.full((h, w), tab[0, 0], np.float32)
    rr = (np.arange(h) + 0.5) / block_px - 0.5
    cc = (np.arange(w) + 0.5) / block_px - 0.5
    R, C = np.meshgrid(np.clip(rr, 0, nb_r - 1), np.clip(cc, 0, nb_c - 1), indexing="ij")
    return ndimage.map_coordinates(tab, [R, C], order=1, mode="nearest").astype(np.float32)


def local_normalize(a: np.ndarray, valid: np.ndarray, block_px: int, q: float = 0.9) -> np.ndarray:
    """a / (local q-quantile): a global top-frac threshold on the result is a local percentile threshold on ``a``."""
    if block_px <= 0:
        return a
    t = block_quantile_map(a, valid, block_px, q)
    floor = max(float(np.nanquantile(t, 0.05)), EPS)
    out = a / np.maximum(t, floor)
    return np.where(valid, out, np.nan).astype(np.float32)


# =============================================================================== spectra / resolution
def radial_psd(field: np.ndarray, px_km=(0.1, 0.1), n_bins: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """(wavenumber cycles/km, mean power) of the demeaned, Hann-tapered, nearest-filled field."""
    f, m = fill(field)
    rows, cols = np.nonzero(m)
    if not len(rows):
        return np.zeros(0), np.zeros(0)
    f = f[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    f = f - f.mean()
    f = f * np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1]))
    P = np.abs(np.fft.rfft2(f)) ** 2
    ky = np.fft.fftfreq(f.shape[0], d=px_km[0])
    kx = np.fft.rfftfreq(f.shape[1], d=px_km[1])
    k = np.hypot(ky[:, None], kx[None, :]).ravel()
    P = P.ravel()
    edges = np.linspace(0, k.max(), n_bins + 1)
    idx = np.clip(np.digitize(k, edges) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    pw = np.bincount(idx, P, minlength=n_bins) / np.maximum(cnt, 1)
    kc = 0.5 * (edges[:-1] + edges[1:])
    ok = (cnt > 0) & (kc > 0)
    return kc[ok], pw[ok]


def effective_resolution_m(field: np.ndarray, px_km=(0.1, 0.1), snr: float = 3.0) -> dict:
    """Empirical resolution: the shortest wavelength whose radial power still exceeds ``snr`` x the high-wavenumber
    noise floor (median power of the top 15 % wavenumbers). A diagnostic, not a survey specification."""
    k, p = radial_psd(field, px_km)
    if len(k) < 8:
        return {"effective_resolution_m": np.nan, "noise_floor": np.nan}
    floor = float(np.median(p[k >= np.quantile(k, 0.85)]))
    above = np.flatnonzero(p > snr * max(floor, EPS))
    if not len(above):
        return {"effective_resolution_m": np.nan, "noise_floor": floor}
    kmax = float(k[above.max()])
    return {"effective_resolution_m": 1000.0 / max(kmax, EPS) / 2.0, "noise_floor": floor,
            "k_max_cycles_per_km": kmax, "note": "half-wavelength at the last wavenumber above snr x noise floor"}
