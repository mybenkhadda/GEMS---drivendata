"""Multiscale geophysical features of potential fields and surfaces (label-free).

Masked operators: a field with gaps is smoothed by NORMALISED convolution, S = G(a m) / G(m) (m = finite mask,
``mode='constant'``), never by zero filling, so a missing value cannot pull a neighbour's gradient towards zero and
the result is identical whether a raster is processed whole or tile by tile (outside the raster counts as missing in
both cases). S is NaN where the kernel support G(m) is below ``min_support``; derivatives are central differences
of S (``np.gradient``: NaN propagates, nothing is invented across a gap). Outputs are NaN wherever the source pixel
itself is missing.

Scales are Gaussian sigmas in metres (100 / 300 / 700 / 1500 m = 1 / 3 / 7 / 15 px). A Gaussian scale is a
smoothing width, NOT a source depth.

Feature families per field ``<p>`` and scale ``s<m>`` (units: field unit per km, per km^2, or dimensionless)::

    hgm        horizontal gradient magnitude |grad S|
    ocos/osin  doubled-angle encoding cos(2 phi), sin(2 phi) of the gradient axis (period 180 deg, continuous)
    lap        Laplacian S_rr + S_cc
    hl1/hl2    Hessian eigenvalues (hl1 >= hl2)
    d2n/d2t    second directional derivative across (along the gradient) / along strike
    curv       iso-line curvature, compressed: atan(sigma_km d2t / |grad S|) / (pi / 2) in [-1, 1]
    edge       1 where |grad S| is a local maximum across the gradient direction (Canny-type thin edge)
    persist    share of scales at which the pixel is such an edge (multiscale boundary persistence)

Frequency (local, tileable): log10 band energy of difference-of-Gaussians d = S_a - S_b, integrated over a window,
and the slope between the finest and coarsest band. Global spectral group (whole raster, one band at a time - FFT is
inherently global): tilt derivative atan(VDR / THDR), analytic-signal amplitude, closeness to the tilt zero contour
and the residual against upward continuation (regional vs residual anomaly); these use the M8 operators
(``m8.potential_fields``) on a nearest-filled field and are masked back to the source footprint.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

TRUNCATE = 4.0
PIXEL_M = 100.0


# =============================================================================== primitives
def sigma_px(scale_m: float, pixel_m: float = PIXEL_M) -> float:
    return max(float(scale_m) / float(pixel_m), 0.5)


def gaussian_radius(sigma: float, truncate: float = TRUNCATE) -> int:
    return int(truncate * float(sigma) + 0.5)


def masked_smooth(a: np.ndarray, sigma: float, min_support: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """(S, support): normalised Gaussian convolution of ``a`` over its finite pixels (float64)."""
    a = np.asarray(a, np.float64)
    m = np.isfinite(a)
    a0 = np.where(m, a, 0.0)
    if sigma <= 0:
        return np.where(m, a0, np.nan), m.astype(np.float64)
    num = ndimage.gaussian_filter(a0, sigma, mode="constant", cval=0.0, truncate=TRUNCATE)
    den = ndimage.gaussian_filter(m.astype(np.float64), sigma, mode="constant", cval=0.0, truncate=TRUNCATE)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(den >= min_support, num / np.maximum(den, 1e-12), np.nan)
    return S, den


def derivatives(S: np.ndarray, px_km: float = 0.1) -> dict:
    """First and second derivatives of S per km (rows = southward, columns = eastward); NaN propagates."""
    sr, sc = np.gradient(S, px_km)
    srr, src = np.gradient(sr, px_km)
    _, scc = np.gradient(sc, px_km)
    return {"sr": sr, "sc": sc, "srr": srr, "scc": scc, "src": src}


def gradient_axis(sr: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """Gradient direction angle (rad, from east, counter-clockwise with north up): east = +col, north = -row."""
    return np.arctan2(-sr, sc)


def hessian_eigen(d: dict) -> tuple[np.ndarray, np.ndarray]:
    tr = d["srr"] + d["scc"]
    disc = np.sqrt(np.maximum(0.25 * (d["srr"] - d["scc"]) ** 2 + d["src"] ** 2, 0.0))
    return 0.5 * tr + disc, 0.5 * tr - disc


def directional_second(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """(d2 across = along the gradient, d2 along strike) per km^2."""
    sr, sc = d["sr"], d["sc"]
    g2 = sr * sr + sc * sc + 1e-30
    d2n = (sc * sc * d["scc"] + 2 * sc * sr * d["src"] + sr * sr * d["srr"]) / g2
    d2t = (sr * sr * d["scc"] - 2 * sr * sc * d["src"] + sc * sc * d["srr"]) / g2
    return d2n, d2t


def nms_edges(hgm: np.ndarray, sr: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """1.0 where |grad| is >= both neighbours one pixel along +/- the gradient direction; NaN where undefined."""
    g = np.hypot(sr, sc)
    with np.errstate(invalid="ignore", divide="ignore"):
        ur, uc = np.where(g > 0, sr / g, 0.0), np.where(g > 0, sc / g, 0.0)
    rr, cc = np.indices(hgm.shape, dtype=np.float64)
    h = np.where(np.isfinite(hgm), hgm, -1.0)
    fwd = ndimage.map_coordinates(h, [rr + ur, cc + uc], order=1, mode="nearest")
    bwd = ndimage.map_coordinates(h, [rr - ur, cc - uc], order=1, mode="nearest")
    e = ((h >= fwd) & (h >= bwd) & (h > 0)).astype(np.float32)
    return np.where(np.isfinite(hgm), e, np.nan).astype(np.float32)


def field_at_scale(a: np.ndarray, scale_m: float, px_km: float = 0.1, min_support: float = 0.25) -> dict:
    """Smoothed field + derivatives + gradient magnitude at one scale (shared by physics / texture / structure)."""
    s = sigma_px(scale_m, px_km * 1000.0)
    S, _ = masked_smooth(a, s, min_support)
    d = derivatives(S, px_km)
    d["S"] = S
    d["hgm"] = np.hypot(d["sr"], d["sc"])
    d["sigma_px"] = s
    d["sigma_km"] = s * px_km
    return d


# =============================================================================== feature names
PHYS_OPS = ("hgm", "ocos", "osin", "lap", "hl1", "hl2", "d2n", "d2t", "curv", "edge")
PHYS_UNITS = {"hgm": "{u}/km", "ocos": "1", "osin": "1", "lap": "{u}/km^2", "hl1": "{u}/km^2", "hl2": "{u}/km^2",
              "d2n": "{u}/km^2", "d2t": "{u}/km^2", "curv": "1", "edge": "flag"}
PHYS_DESC = {"hgm": "horizontal gradient magnitude", "ocos": "cos(2 x gradient axis angle)",
             "osin": "sin(2 x gradient axis angle)", "lap": "Laplacian", "hl1": "larger Hessian eigenvalue",
             "hl2": "smaller Hessian eigenvalue", "d2n": "second derivative across strike (gradient direction)",
             "d2t": "second derivative along strike", "curv": "compressed iso-line curvature",
             "edge": "thin gradient-maximum edge flag"}


def phys_name(prefix: str, field: str, op: str, scale_m: float) -> str:
    return f"{prefix}_{field}_{op}_s{int(round(scale_m))}"


def multiscale_gradients(a: np.ndarray, scales_m, prefix: str, field: str, ops=PHYS_OPS, px_km: float = 0.1,
                         cache: dict | None = None) -> dict:
    """Every op of ``ops`` at every scale; ``cache`` (optional) receives the per-scale derivative dicts."""
    out = {}
    valid = np.isfinite(a)
    edges = []
    for sm in scales_m:
        d = field_at_scale(a, sm, px_km)
        if cache is not None:
            cache[(field, float(sm))] = d
        need = set(ops)
        if "hgm" in need:
            out[phys_name(prefix, field, "hgm", sm)] = d["hgm"]
        if need & {"ocos", "osin"}:
            phi = gradient_axis(d["sr"], d["sc"])
            out[phys_name(prefix, field, "ocos", sm)] = np.cos(2 * phi)
            out[phys_name(prefix, field, "osin", sm)] = np.sin(2 * phi)
        if "lap" in need:
            out[phys_name(prefix, field, "lap", sm)] = d["srr"] + d["scc"]
        if need & {"hl1", "hl2"}:
            l1, l2 = hessian_eigen(d)
            out[phys_name(prefix, field, "hl1", sm)] = l1
            out[phys_name(prefix, field, "hl2", sm)] = l2
        if need & {"d2n", "d2t", "curv"}:
            d2n, d2t = directional_second(d)
            if "d2n" in need:
                out[phys_name(prefix, field, "d2n", sm)] = d2n
            if "d2t" in need:
                out[phys_name(prefix, field, "d2t", sm)] = d2t
            if "curv" in need:
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[phys_name(prefix, field, "curv", sm)] = np.arctan(
                        d["sigma_km"] * d2t / (d["hgm"] + 1e-12)) / (np.pi / 2)
        if "edge" in need or "persist" in need:
            e = nms_edges(d["hgm"], d["sr"], d["sc"])
            edges.append(e)
            if "edge" in need:
                out[phys_name(prefix, field, "edge", sm)] = e
    if edges:
        import warnings

        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"{prefix}_{field}_persist"] = np.nanmean(np.stack(edges), 0) if len(edges) > 1 else edges[0]
    return {k: np.where(valid, v, np.nan).astype(np.float32) for k, v in out.items()}


def physics_names(prefix: str, field: str, scales_m, ops=PHYS_OPS, persist: bool = True) -> list[str]:
    """Feature ids produced by ``multiscale_gradients`` (same order as its output dict)."""
    out = []
    for sm in scales_m:
        for op in ("hgm", "ocos", "osin", "lap", "hl1", "hl2", "d2n", "d2t", "curv", "edge"):
            if op in ops:
                out.append(phys_name(prefix, field, op, sm))
    if persist:
        out.append(f"{prefix}_{field}_persist")
    return out


def potential_field_boundaries(a: np.ndarray, scales_m, prefix: str, field: str, px_km: float = 0.1) -> dict:
    """Boundary evidence: thin edges per scale and their persistence across scales."""
    return multiscale_gradients(a, scales_m, prefix, field, ops=("edge", "persist"), px_km=px_km)


def magnetic_features(rtp: np.ndarray, scales_m, ops=PHYS_OPS, px_km: float = 0.1, cache: dict | None = None,
                      field: str = "rtp") -> dict:
    """Multiscale gradient / curvature / boundary features of the (reduced-to-pole) magnetic field."""
    return multiscale_gradients(rtp, scales_m, "mag", field, tuple(ops) + ("persist",), px_km, cache)


def gravity_features(grav: np.ndarray, scales_m, ops=PHYS_OPS, px_km: float = 0.1, cache: dict | None = None,
                     field: str = "iso_grav_anom") -> dict:
    """Multiscale gradient / curvature / boundary features of the isostatic gravity anomaly."""
    return multiscale_gradients(grav, scales_m, "grav", field, tuple(ops) + ("persist",), px_km, cache)


def frequency_features(a: np.ndarray, field: str, bands_m, integration_m, px_km: float = 0.1,
                       prefix: str = "ms") -> dict:
    """Local spectral characteristics: log10 energy of DoG bands (S_a - S_b)^2 integrated over a Gaussian window,
    and the energy slope finest - coarsest band (positive = rough / short-wavelength dominated)."""
    valid = np.isfinite(a)
    out, logs = {}, []
    for (lo, hi), im in zip(bands_m, integration_m):
        Sa, _ = masked_smooth(a, sigma_px(lo, px_km * 1000))
        Sb, _ = masked_smooth(a, sigma_px(hi, px_km * 1000))
        E, _ = masked_smooth((Sa - Sb) ** 2, sigma_px(im, px_km * 1000))
        le = np.log10(np.maximum(E, 0.0) + 1e-12)
        out[f"{prefix}_{field}_dog{int(lo)}_{int(hi)}"] = le
        logs.append(le)
    if len(logs) >= 2:
        out[f"{prefix}_{field}_specslope"] = logs[0] - logs[-1]
    return {k: np.where(valid, v, np.nan).astype(np.float32) for k, v in out.items()}


def frequency_names(field: str, bands_m, prefix: str = "ms") -> list[str]:
    n = [f"{prefix}_{field}_dog{int(lo)}_{int(hi)}" for lo, hi in bands_m]
    return n + ([f"{prefix}_{field}_specslope"] if len(bands_m) >= 2 else [])


# =============================================================================== global spectral group
def spectral_names(prefix: str, field: str, upward_km) -> list[str]:
    return [f"{prefix}_{field}_tilt", f"{prefix}_{field}_asig", f"{prefix}_{field}_tiltzero"] + \
        [f"{prefix}_{field}_resid_uc{float(h):g}km" for h in upward_km]


def spectral_features(a: np.ndarray, prefix: str, field: str, smooth_m: float, upward_km,
                      px_km: float = 0.1) -> dict:
    """Whole-raster FFT features of one potential field (memory: a few float64 copies of one band)."""
    from ..m8 import potential_fields as PFm

    f, m = PFm.fill(np.asarray(a, np.float32))
    s = sigma_px(smooth_m, px_km * 1000)
    fs = PFm.smooth(f, s)
    ge, gn = PFm.hgrad(f, s, (px_km, px_km))
    th = np.hypot(ge, gn)
    vz = PFm.vdr_fft(fs, (px_km, px_km))
    tilt = np.arctan2(vz, th + 1e-12)
    out = {f"{prefix}_{field}_tilt": tilt, f"{prefix}_{field}_asig": np.sqrt(th * th + vz * vz),
           f"{prefix}_{field}_tiltzero": 1.0 - np.abs(tilt) / (np.pi / 2)}
    for h in upward_km:
        out[f"{prefix}_{field}_resid_uc{float(h):g}km"] = f - PFm.upward(f, float(h), (px_km, px_km))
    return {k: PFm.restore(v, m) for k, v in out.items()}


def required_halo_px(scales_m, extra_px: int = 2) -> int:
    """Support of the derivative stack at the largest scale (Gaussian radius + central-difference stencil)."""
    return max((gaussian_radius(sigma_px(s)) for s in scales_m), default=0) + 2 + int(extra_px)
