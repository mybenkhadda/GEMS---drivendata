"""G02-A displacement signatures: apparent lateral offset of coherent fabric across a candidate line.

Map frame and conventions (identical in every G02 module)
    pixel (r, c) -> E = c * px, N = -r * py (metres; rows run south). Azimuths are clockwise from grid north, axial
    (phi and phi + 180 are the same line), the convention of M9's wavelet strike (``spectral.wavelet_layers``:
    strike = wavevector azimuth + 90).
        t = (sin phi, cos phi)     along the candidate line
        n = (cos phi, -sin phi)    normal (azimuth phi + 90); the "+n side" is on the right looking along t
    ``RotFrame`` resamples a window onto a (t, n) grid of spacing ``step`` (rows along +t, columns along +n) with
    explicit map <-> pixel algebra, so oriented profiles, shifts and running means become axis-aligned array operations
    (O(1) per pixel in the profile length). Pixels must be square within ``max_pixel_aspect`` (checked).

Measurement (one field, one wavelength lambda, one azimuth phi; label-free)
    f      = difference-of-Gaussians band-pass of the raw field peaking at lambda (signed: band polarity is kept)
    P_v(u) = f(x + v n + u t), profiles parallel to the line at normal offsets v in {-h2, -h1, +h1, +h2}
    rho_ab(s) = NCC over |u| <= L of P_a(u) and P_b(u + s), s in a bounded search |s| <= S_max
    across pair (-h1, +h1) -> shift s_X;  same-side pairs (-h2, -h1), (+h1, +h2) -> s_L, s_R
    Continuous fabric oblique to the line gives shifts LINEAR in the normal separation (s = dv cot alpha) and no
    intercept; a lateral offset D at the line adds a STEP to the across shift only. Hence
        slope     b = (s_L + s_R) / (2 (h2 - h1))           (oblique-fabric prediction, per metre of separation)
        offset    a = s_X - 2 h1 b                          (apparent lateral offset at the line, metres)
        gain      = rho_X(s_X) - rho_X(2 h1 b)              (how much better a step fits than oblique continuity)
    a > 0: the +n block appears displaced towards +t (apparent SINISTRAL sense); a < 0: apparent DEXTRAL. The sign is
    invariant to phi -> phi + 180 (both t and n flip).

What it is NOT
    ``a`` is an APPARENT offset of a potential-field pattern. Potential fields cannot separate strike slip from dip
    slip on a dipping contact, from a change of source depth, or from a pre-existing bend of the fabric; true slip
    magnitude is not identifiable without further constraints. Offsets larger than the bounded search, offsets of
    periodic fabric by a multiple of its period, and fabric nearly parallel to the line are undetectable (score 0, not
    negative evidence). Many faults produce no detectable offset: G02-A is a one-sided evidence family.

Score (bounded [0, 1]; NaN = no support: invalid data or too few finite samples)
    score = rho_X+ * min(rho_L, rho_R)+ * gain_n * resolvable(a) * unambiguous * side_consistency * energy_gate
    max over the tested azimuths and wavelengths; the argmax azimuth, offset, wavelength, the periodicity ambiguity
    (second NCC peak / first) and the azimuth ambiguity (second-best azimuth score / best) are saved separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

DISP_KEYS = ("score", "offset_m", "azimuth", "lambda_km", "rho_x", "ambiguity", "azimuth_ambiguity")


# =============================================================================== frame / primitives
def azimuths(n: int) -> list[float]:
    return [180.0 * k / int(n) for k in range(int(n))]


def check_square(pixel_m, max_aspect: float = 1.02) -> None:
    py, px = float(pixel_m[0]), float(pixel_m[1])
    if py <= 0 or px <= 0:
        raise ValueError(f"pixel size must be positive, got {pixel_m}")
    if max(py, px) / min(py, px) > float(max_aspect):
        raise ValueError(f"G02 rotated-frame sampling needs near-square pixels: {py:.2f} x {px:.2f} m exceeds aspect "
                         f"{max_aspect} (resample the grid first)")


def frame_vectors(phi_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """((tE, tN), (nE, nN)) unit vectors along and normal to azimuth ``phi_deg``."""
    a = np.radians(float(phi_deg))
    return (float(np.sin(a)), float(np.cos(a))), (float(np.cos(a)), float(-np.sin(a)))


def offset_px(dist_m, azimuth_deg, pixel_m) -> tuple[np.ndarray, np.ndarray]:
    """(d_row, d_col) of a displacement of ``dist_m`` metres towards ``azimuth_deg`` (scalars or arrays)."""
    a = np.radians(np.asarray(azimuth_deg, np.float64))
    d = np.asarray(dist_m, np.float64)
    return -d * np.cos(a) / float(pixel_m[0]), d * np.sin(a) / float(pixel_m[1])


def int_shift(a: np.ndarray, r: int, c: int, fill=np.nan) -> np.ndarray:
    """out[i, j] = a[i + r, j + c] (``fill`` outside)."""
    a = np.asarray(a)
    H, W = a.shape
    r, c = int(r), int(c)
    out = np.full(a.shape, fill, dtype=a.dtype if np.issubdtype(a.dtype, np.floating) else np.float32)
    rs0, rs1 = max(0, -r), min(H, H - r)
    cs0, cs1 = max(0, -c), min(W, W - c)
    if rs0 < rs1 and cs0 < cs1:
        out[rs0:rs1, cs0:cs1] = a[rs0 + r:rs1 + r, cs0 + c:cs1 + c]
    return out


def nan_running_mean(a: np.ndarray, half: int, axis: int = 0, min_share: float = 0.6) -> np.ndarray:
    """NaN-aware running mean over 2 half + 1 samples along ``axis``; NaN where < ``min_share`` are finite."""
    a = np.asarray(a, np.float32)
    size = 2 * max(int(half), 0) + 1
    f = np.isfinite(a)
    num = ndimage.uniform_filter1d(np.where(f, a, 0.0).astype(np.float32), size, axis=axis, mode="constant", cval=0.0)
    den = ndimage.uniform_filter1d(f.astype(np.float32), size, axis=axis, mode="constant", cval=0.0)
    return np.where(den >= float(min_share), num / np.maximum(den, 1e-6), np.nan).astype(np.float32)


class RotFrame:
    """Resampling of a (H, W) window onto a grid aligned with azimuth ``phi`` (rows along +t, columns along +n)."""

    def __init__(self, shape, pixel_m, phi_deg: float, step_m: float, margin: int = 2):
        self.shape = (int(shape[0]), int(shape[1]))
        self.py, self.px = float(pixel_m[0]), float(pixel_m[1])
        self.phi = float(phi_deg)
        self.step = float(step_m)
        (self.tE, self.tN), (self.nE, self.nN) = frame_vectors(self.phi)
        H, W = self.shape
        rr = np.array([0.0, 0.0, H - 1.0, H - 1.0])
        cc = np.array([0.0, W - 1.0, 0.0, W - 1.0])
        u, v = self._uv(rr, cc)
        self.u0 = float(u.min()) - margin * self.step
        self.v0 = float(v.min()) - margin * self.step
        self.rshape = (int(np.ceil((u.max() - self.u0) / self.step)) + margin + 1,
                       int(np.ceil((v.max() - self.v0) / self.step)) + margin + 1)
        self._back = None

    def _uv(self, r, c):
        E = np.asarray(c, np.float64) * self.px
        N = -np.asarray(r, np.float64) * self.py
        return E * self.tE + N * self.tN, E * self.nE + N * self.nN

    def rc_of_ij(self, i, j):
        u = self.u0 + np.asarray(i, np.float64) * self.step
        v = self.v0 + np.asarray(j, np.float64) * self.step
        E = u * self.tE + v * self.nE
        N = u * self.tN + v * self.nN
        return -N / self.py, E / self.px

    def ij_of_rc(self, r, c):
        u, v = self._uv(r, c)
        return (u - self.u0) / self.step, (v - self.v0) / self.step

    def to_rot(self, a: np.ndarray) -> np.ndarray:
        """Bilinear resampling onto the rotated grid (NaN outside the window and next to NaN samples)."""
        ii, jj = np.indices(self.rshape, dtype=np.float32)
        r, c = self.rc_of_ij(ii, jj)
        return ndimage.map_coordinates(np.asarray(a, np.float32), [r.astype(np.float32), c.astype(np.float32)],
                                       order=1, mode="constant", cval=np.nan, prefilter=False).astype(np.float32)

    def from_rot(self, R: np.ndarray) -> np.ndarray:
        """Bilinear resampling of a rotated-grid array back onto the window's pixels."""
        if self._back is None:
            r, c = np.indices(self.shape, dtype=np.float32)
            i, j = self.ij_of_rc(r, c)
            self._back = (i.astype(np.float32), j.astype(np.float32))
        return ndimage.map_coordinates(np.asarray(R, np.float32), list(self._back), order=1, mode="constant",
                                       cval=np.nan, prefilter=False).astype(np.float32)

    def splat(self, rows, cols, weights) -> np.ndarray:
        """Point weights summed into their nearest rotated cell (mass preserving; for sparse point maps)."""
        i, j = self.ij_of_rc(rows, cols)
        ii, jj = np.rint(i).astype(np.int64), np.rint(j).astype(np.int64)
        w = np.asarray(weights, np.float32)
        ok = (ii >= 0) & (ii < self.rshape[0]) & (jj >= 0) & (jj < self.rshape[1]) & np.isfinite(w)
        R = np.zeros(self.rshape, np.float32)
        np.add.at(R, (ii[ok], jj[ok]), w[ok])
        return R


def select_by_azimuth(maps: dict, theta_deg: np.ndarray) -> np.ndarray:
    """Per pixel, the map of the tested azimuth closest (axially) to ``theta_deg`` (NaN where theta is NaN)."""
    phis = np.array(sorted(maps), np.float64)
    th = np.asarray(theta_deg, np.float64)
    d = np.abs((np.nan_to_num(th)[None] - phis[:, None, None] + 90.0) % 180.0 - 90.0)
    idx = np.argmin(d, 0)
    stack = np.stack([np.asarray(maps[p], np.float32) for p in phis])
    out = np.take_along_axis(stack, idx[None], 0)[0]
    return np.where(np.isfinite(th), out, np.nan).astype(np.float32)


def side_means(arrays: dict, pixel_m, phi_deg: float, h_m: float, half_m: float, step_m: float,
               min_share: float = 0.6) -> dict:
    """{name: (mean on the -n side, mean on the +n side)}: running means along the line at normal offsets -+h."""
    first = next(iter(arrays.values()))
    fr = RotFrame(first.shape, pixel_m, phi_deg, step_m)
    k = max(1, int(round(float(h_m) / float(step_m))))
    half = max(1, int(round(float(half_m) / float(step_m))))
    out = {}
    for name, a in arrays.items():
        R = fr.to_rot(a)
        lm = nan_running_mean(int_shift(R, 0, -k), half, 0, min_share)
        rm = nan_running_mean(int_shift(R, 0, k), half, 0, min_share)
        out[name] = (fr.from_rot(lm), fr.from_rot(rm))
    return out


def noisy_or(*maps) -> np.ndarray | None:
    """1 - prod(1 - m) over the FINITE members (a missing family never lowers the others); NaN where none is finite."""
    ms = [np.clip(np.asarray(m, np.float32), 0.0, 1.0) for m in maps if m is not None]
    if not ms:
        return None
    keep = np.ones(ms[0].shape, np.float32)
    anyf = np.zeros(ms[0].shape, bool)
    for m in ms:
        f = np.isfinite(m)
        keep *= np.where(f, 1.0 - m, 1.0)
        anyf |= f
    return np.where(anyf, 1.0 - keep, np.nan).astype(np.float32)


# =============================================================================== band-pass
def dog_sigmas_m(lam_m: float, ratio: float = 2.0) -> tuple[float, float]:
    """(sigma1, sigma2) of a DoG whose amplitude response peaks at wavelength ``lam_m``:
    k_peak^2 = 2 ln(r^2) / (sigma1^2 (r^2 - 1)), r = sigma2 / sigma1."""
    r2 = float(ratio) ** 2
    s1 = np.sqrt(2.0 * np.log(r2) / (r2 - 1.0)) * float(lam_m) / (2.0 * np.pi)
    return float(s1), float(ratio) * float(s1)


def bandpass(field: np.ndarray, valid: np.ndarray, lam_km: float, pixel_m, ratio: float = 2.0) -> np.ndarray:
    """Signed DoG band-pass at ``lam_km`` (nearest-filled before filtering; NaN restored on invalid pixels)."""
    from gems_eda.enhance import fill_nan

    a = np.asarray(field, np.float32)
    m = np.isfinite(a) & np.asarray(valid, bool)
    if not m.any():
        return np.full(a.shape, np.nan, np.float32)
    g = fill_nan(np.where(m, a, np.nan)).astype(np.float32)
    s1, s2 = dog_sigmas_m(float(lam_km) * 1000.0, ratio)
    py, px = float(pixel_m[0]), float(pixel_m[1])
    out = ndimage.gaussian_filter(g, (s1 / py, s1 / px), mode="nearest") - \
        ndimage.gaussian_filter(g, (s2 / py, s2 / px), mode="nearest")
    return np.where(m, out, np.nan).astype(np.float32)


# =============================================================================== scale parameters
@dataclass(frozen=True)
class DispScale:
    lam_km: float
    step_m: float
    k1: int                 # inner profile offset (steps)
    k2: int                 # outer profile offset (steps)
    half: int               # running-mean half length along the line (steps)
    M: int                  # bounded shift search |m| <= M (steps)
    a0_m: float             # offset resolution
    excl: int               # second-peak exclusion (steps)
    side_tol: float         # side-shift agreement tolerance (steps)

    @property
    def halo_m(self) -> float:
        return (self.k2 + self.half + self.M + 3) * self.step_m


def disp_scale(lam_km: float, pixel_m, dc: dict) -> DispScale:
    lam = float(lam_km) * 1000.0
    step = max(min(float(pixel_m[0]), float(pixel_m[1])), lam / float(dc["samples_per_wavelength"]))
    k1 = max(1, int(round(float(dc["h1_frac"]) * lam / step)))
    k2 = max(k1 + 1, int(round(float(dc["h2_frac"]) * lam / step)))
    half = max(1, int(round(max(float(dc["half_length_frac"]) * lam, float(dc["min_half_length_m"])) / step)))
    smax = min(float(dc["max_offset_m"]), float(dc["max_offset_frac_of_length"]) * half * step)
    M = max(1, int(np.floor(smax / step + 1e-9)))
    return DispScale(float(lam_km), step, k1, k2, half, M, max(step, float(dc["offset_resolution_frac"]) * lam),
                     max(1, int(np.ceil(float(dc["peak_exclusion_frac"]) * lam / step))),
                     max(float(dc["side_consistency_frac"]) * lam / step, 0.5))


def disp_halo_m(lams_km, pixel_m, cfg: dict) -> float:
    """Context a window needs so every profile, shift and band-pass kernel sees real neighbours."""
    dc = cfg["displacement"]
    out = 0.0
    for lam in lams_km:
        sc = disp_scale(lam, pixel_m, dc)
        _, s2 = dog_sigmas_m(float(lam) * 1000.0, float(cfg["bandpass"]["sigma_ratio"]))
        out = max(out, sc.halo_m * np.sqrt(2.0) + 3.0 * s2)
    return float(out)


# =============================================================================== NCC machinery (rotated frame)
def ncc_stack(PA: np.ndarray, PB: np.ndarray, half: int, shifts, min_share: float = 0.6) -> np.ndarray:
    """rho[k](x) = NCC over the window |u| <= half of PA(u) and PB(u + shifts[k]) (rows = along-line axis)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mA = nan_running_mean(PA, half, 0, min_share)
        vA = nan_running_mean(PA * PA, half, 0, min_share) - mA * mA
        mB = nan_running_mean(PB, half, 0, min_share)
        vB = nan_running_mean(PB * PB, half, 0, min_share) - mB * mB
        fin = vA[np.isfinite(vA)]
        eps = 1e-9 * (float(np.mean(np.abs(fin))) if fin.size else 1.0) + 1e-30
        out = np.full((len(shifts),) + PA.shape, np.nan, np.float32)
        for k, m in enumerate(shifts):
            m = int(m)
            cross = nan_running_mean(PA * int_shift(PB, m, 0), half, 0, min_share)
            mBs = int_shift(mB, m, 0)
            vBs = int_shift(vB, m, 0)
            den = np.sqrt(np.clip(vA, 0.0, None) * np.clip(vBs, 0.0, None))
            rho = (cross - mA * mBs) / np.where(den > eps, den, np.nan)
            out[k] = np.clip(rho, -1.0, 1.0)
    return out


def stack_peak(S: np.ndarray, shifts, excl: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(best value, refined best shift in steps (parabolic), best value >= ``excl`` steps away from the peak)."""
    shifts = np.asarray(shifts, np.float32)
    n = S.shape[0]
    fin = np.isfinite(S).any(0)
    Sz = np.where(np.isfinite(S), S, -np.inf)
    ib = np.argmax(Sz, 0)
    best = np.take_along_axis(Sz, ib[None], 0)[0]
    im, ip = np.clip(ib - 1, 0, n - 1), np.clip(ib + 1, 0, n - 1)
    rm = np.take_along_axis(Sz, im[None], 0)[0]
    rp = np.take_along_axis(Sz, ip[None], 0)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        den = rm - 2.0 * best + rp
        ok = (ib > 0) & (ib < n - 1) & np.isfinite(rm) & np.isfinite(rp) & (den < 0)
        delta = np.where(ok, np.clip(0.5 * (rm - rp) / np.where(ok, den, -1.0), -0.5, 0.5), 0.0)
    s = shifts[ib] + delta.astype(np.float32)
    far = np.abs(np.arange(n)[:, None, None] - ib[None]) >= int(excl)
    second = np.where(far, Sz, -np.inf).max(0) if n > 1 else np.full(best.shape, -np.inf, np.float32)
    return (np.where(fin, best, np.nan).astype(np.float32), np.where(fin, s, np.nan).astype(np.float32),
            second.astype(np.float32))


def interp_stack(S: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``S`` along axis 0 at fractional indices ``x`` (NaN outside [0, n - 1])."""
    n = S.shape[0]
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x) & (x >= 0) & (x <= n - 1)
    xc = np.clip(np.nan_to_num(x), 0, n - 1)
    x0 = np.floor(xc).astype(np.int64)
    x1 = np.minimum(x0 + 1, n - 1)
    w = (xc - x0).astype(np.float32)
    v = (1 - w) * np.take_along_axis(S, x0[None], 0)[0] + w * np.take_along_axis(S, x1[None], 0)[0]
    return np.where(ok, v, np.nan).astype(np.float32)


def displacement_rot(R: np.ndarray, E: np.ndarray | None, sc: DispScale, dc: dict) -> dict:
    """G02-A measurements of one band-passed field on one rotated grid (see module docstring)."""
    ms = float(dc["min_tap_share"])
    shifts = np.arange(-sc.M, sc.M + 1)
    P = {k: int_shift(R, 0, k) for k in (-sc.k2, -sc.k1, sc.k1, sc.k2)}       # P[k](i, j) = R(i, j + k)
    rX = ncc_stack(P[-sc.k1], P[sc.k1], sc.half, shifts, ms)
    rL = ncc_stack(P[-sc.k2], P[-sc.k1], sc.half, shifts, ms)
    rR = ncc_stack(P[sc.k1], P[sc.k2], sc.half, shifts, ms)
    bX, sX, secX = stack_peak(rX, shifts, sc.excl)
    bL, sL, _ = stack_peak(rL, shifts, sc.excl)
    bR, sR, _ = stack_peak(rR, shifts, sc.excl)
    del rL, rR
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        slope = (sL + sR) / (2.0 * (sc.k2 - sc.k1))
        s_pred = slope * 2.0 * sc.k1
        rho_pred = interp_stack(rX, s_pred + sc.M)
        a_steps = sX - s_pred
        off_m = (a_steps * sc.step_m).astype(np.float32)
        gain_raw = bX - rho_pred
        gain = np.clip(np.nan_to_num(gain_raw, nan=0.0) / float(dc["gain_ref"]), 0.0, 1.0)
        sig = 1.0 - np.exp(-(np.nan_to_num(off_m) / sc.a0_m) ** 2)
        unamb = np.clip(1.0 - np.clip(secX, 0.0, None) / np.maximum(bX, 1e-6), 0.0, 1.0)
        cside = np.exp(-((sL - sR) / sc.side_tol) ** 2)
        inside = (np.abs(sL) < sc.M - 0.5) & (np.abs(sR) < sc.M - 0.5)       # side shifts not clipped by the search
        if E is not None:
            lo, hi = (float(x) for x in dc["energy_gate"])
            es = np.sqrt(np.clip(int_shift(E, 0, -sc.k1), 0, 1) * np.clip(int_shift(E, 0, sc.k1), 0, 1))
            eg = np.nan_to_num(np.clip((es - lo) / (hi - lo), 0.0, 1.0), nan=0.0)
        else:
            eg = 1.0
        score = np.clip(bX, 0, 1) * np.clip(np.minimum(bL, bR), 0, 1) * gain * sig * unamb * \
            np.nan_to_num(cside, nan=0.0) * inside * eg
    support = np.isfinite(bX) & np.isfinite(bL) & np.isfinite(bR)
    nan = np.float32(np.nan)
    return {"score": np.where(support, score, nan).astype(np.float32),
            "offset_m": np.where(support, off_m, nan).astype(np.float32),
            "rho_x": np.where(support, bX, nan).astype(np.float32),
            "ambiguity": np.where(support, 1.0 - unamb, nan).astype(np.float32)}


# =============================================================================== evidence map
def displacement_evidence(field: np.ndarray, valid: np.ndarray, energy_pct: dict, lams_km, pixel_m, cfg: dict) -> dict:
    """G02-A maps of ONE field on one window: {score, offset_m, azimuth, lambda_km, rho_x, ambiguity,
    azimuth_ambiguity}. ``energy_pct``: {lam_km: G01 energy percentile at that scale} (the energy gate)."""
    dc = cfg["displacement"]
    az = azimuths(cfg["orientation"]["n_azimuths"])
    shape = np.asarray(field).shape
    out = {k: np.full(shape, np.nan, np.float32) for k in DISP_KEYS if k != "score"}
    best = np.full(shape, -np.inf, np.float32)
    for lam in lams_km:
        f = bandpass(field, valid, lam, pixel_m, float(cfg["bandpass"]["sigma_ratio"]))
        if not np.isfinite(f).any():
            continue
        sc = disp_scale(lam, pixel_m, dc)
        e = energy_pct.get(float(lam)) if energy_pct else None
        lb = np.full(shape, -np.inf, np.float32)
        l2 = np.full(shape, -np.inf, np.float32)
        lm = {k: np.full(shape, np.nan, np.float32) for k in ("offset_m", "rho_x", "ambiguity", "azimuth")}
        for phi in az:
            fr = RotFrame(shape, pixel_m, phi, sc.step_m)
            res = displacement_rot(fr.to_rot(f), fr.to_rot(e) if e is not None else None, sc, dc)
            s = fr.from_rot(res["score"])
            s = np.where(np.isfinite(s), s, -np.inf)
            better = s > lb
            l2 = np.where(better, lb, np.maximum(l2, s))
            lb = np.where(better, s, lb)
            for k in ("offset_m", "rho_x", "ambiguity"):
                lm[k] = np.where(better, fr.from_rot(res[k]), lm[k])
            lm["azimuth"] = np.where(better, np.float32(phi), lm["azimuth"])
        with np.errstate(invalid="ignore", divide="ignore"):
            az_amb = np.where(lb > 0, np.clip(l2, 0.0, None) / np.where(lb > 0, lb, 1.0), np.nan).astype(np.float32)
        better = lb > best
        best = np.where(better, lb, best)
        for k in ("offset_m", "rho_x", "ambiguity", "azimuth"):
            out[k] = np.where(better, lm[k], out[k])
        out["lambda_km"] = np.where(better, np.float32(lam), out["lambda_km"])
        out["azimuth_ambiguity"] = np.where(better, az_amb, out["azimuth_ambiguity"])
    out["score"] = np.where(np.isfinite(best), best, np.nan).astype(np.float32)
    v = np.asarray(valid, bool)
    return {k: np.where(v, a, np.nan).astype(np.float32) for k, a in out.items()}
