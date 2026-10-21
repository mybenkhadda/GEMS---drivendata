"""Geophysical enhancement filters expressed at physical (km) scales.

Conventions
-----------
* Arrays are (row, col) with rows increasing southward; gradients are returned
  as (east, north) components in field-units per km.
* Orientations are *strikes*: azimuth of the structure, clockwise from north, in [0, 180).
* A physical scale ``s`` km maps to a Gaussian ``sigma = s / 2`` km (FWHM ~ 1.2 s).
* NaNs are nearest-filled before filtering and restored afterwards, so filters
  never smear nodata inward; edges of the valid area should still be read with care.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.stats import skew

from .io import Grid


def fill_nan(a: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill of NaNs (float64 copy)."""
    m = ~np.isfinite(a)
    out = np.asarray(a, dtype=np.float64).copy()
    if not m.any():
        return out
    if m.all():
        return np.zeros_like(out)
    idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
    return out[tuple(idx)]


def nan_gaussian(a: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian smoothing by normalised convolution (ignores NaNs)."""
    m = np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(m, a, 0.0).astype(np.float64), sigma_px, mode="nearest")
    den = ndimage.gaussian_filter(m.astype(np.float64), sigma_px, mode="nearest")
    out = num / np.maximum(den, 1e-6)
    out[~m] = np.nan
    return out.astype(np.float32)


def robust_standardize(a: np.ndarray) -> np.ndarray:
    """(a - median) / (1.4826 MAD); falls back to std when MAD is zero."""
    v = a[np.isfinite(a)]
    if v.size == 0:
        return a.astype(np.float32)
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    scale = mad if mad > 0 else (np.std(v) or 1.0)
    return ((a - med) / scale).astype(np.float32)


def standardize_signal(a: np.ndarray, skew_threshold: float = 3.0) -> tuple[np.ndarray, str]:
    """Robust z-score, with signed-log compression for heavily skewed bands."""
    z = robust_standardize(a)
    v = z[np.isfinite(z)]
    if v.size > 200_000:
        v = np.random.default_rng(0).choice(v, 200_000, replace=False)
    if v.size and abs(skew(v)) > skew_threshold:
        return (np.sign(z) * np.log1p(np.abs(z))).astype(np.float32), "robust-z + signed log"
    return z, "robust-z"


def _restore(out: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = out.astype(np.float32)
    out[~valid] = np.nan
    return out


class Enhancer:
    """Multi-scale derivative, curvature, ridge, wavelet and orientation filters."""

    def __init__(self, grid: Grid):
        self.grid = grid

    # -- scale handling -------------------------------------------------
    def sigma_px(self, scale_km: float) -> float:
        return max(scale_km / 2.0 / self.grid.pixel_km, 0.7)

    def _prep(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return fill_nan(a), np.isfinite(a)

    # -- first derivatives ----------------------------------------------
    def gradient(self, a: np.ndarray, scale_km: float) -> tuple[np.ndarray, np.ndarray]:
        f, m = self._prep(a)
        s = self.sigma_px(scale_km)
        d_r = ndimage.gaussian_filter(f, s, order=(1, 0), mode="nearest")
        d_c = ndimage.gaussian_filter(f, s, order=(0, 1), mode="nearest")
        return _restore(d_c / self.grid.pixel_km_x, m), _restore(-d_r / self.grid.pixel_km_y, m)

    def thdr(self, a: np.ndarray, scale_km: float) -> np.ndarray:
        """Total horizontal derivative |grad f| (edge strength)."""
        ge, gn = self.gradient(a, scale_km)
        return np.hypot(ge, gn)

    def directional_derivative(self, a: np.ndarray, scale_km: float, azimuth_deg: float) -> np.ndarray:
        """Derivative along ``azimuth_deg``; use strike + 90 to enhance structures of that strike."""
        ge, gn = self.gradient(a, scale_km)
        t = np.radians(azimuth_deg)
        return ge * np.sin(t) + gn * np.cos(t)

    def vertical_derivative(self, a: np.ndarray) -> np.ndarray:
        """First vertical derivative of a potential field via |k| in the wavenumber domain."""
        f, m = self._prep(a)
        f = f - f[m].mean() if m.any() else f
        h, w = f.shape
        ph, pw = max(h // 4, 1), max(w // 4, 1)
        F = np.pad(f, ((ph, ph), (pw, pw)), mode="reflect")
        ky = 2 * np.pi * np.fft.fftfreq(F.shape[0], d=self.grid.pixel_km_y)
        kx = 2 * np.pi * np.fft.rfftfreq(F.shape[1], d=self.grid.pixel_km_x)
        k = np.hypot(ky[:, None], kx[None, :])
        out = np.fft.irfft2(np.fft.rfft2(F) * k, s=F.shape)[ph:ph + h, pw:pw + w]
        return _restore(out, m)

    def tilt(self, a: np.ndarray, scale_km: float) -> np.ndarray:
        """Tilt angle (deg) = atan(VDR / THDR); zero contour tracks source edges."""
        smooth = nan_gaussian(a, self.sigma_px(scale_km))
        vdr = self.vertical_derivative(smooth)
        thdr = self.thdr(smooth, 0.0)
        return np.degrees(np.arctan2(vdr, thdr)).astype(np.float32)

    # -- second derivatives ---------------------------------------------
    def hessian(self, a: np.ndarray, scale_km: float):
        """Scale-normalised (sigma^2) Hessian components in pixel units + valid mask."""
        f, m = self._prep(a)
        s = self.sigma_px(scale_km)
        hrr = ndimage.gaussian_filter(f, s, order=(2, 0), mode="nearest") * s * s
        hcc = ndimage.gaussian_filter(f, s, order=(0, 2), mode="nearest") * s * s
        hrc = ndimage.gaussian_filter(f, s, order=(1, 1), mode="nearest") * s * s
        return hrr, hrc, hcc, m

    @staticmethod
    def _eig2(a, b, c):
        """Eigen-decomposition of [[a, b], [b, c]] ordered |l1| >= |l2|; v1 in (row, col)."""
        tr = 0.5 * (a + c)
        d = np.sqrt((0.5 * (a - c)) ** 2 + b * b)
        e_hi, e_lo = tr + d, tr - d
        swap = np.abs(e_lo) > np.abs(e_hi)
        l1 = np.where(swap, e_lo, e_hi)
        l2 = np.where(swap, e_hi, e_lo)
        vr, vc = l1 - c, b
        degenerate = np.hypot(vr, vc) < 1e-12
        vr = np.where(degenerate, b, vr)
        vc = np.where(degenerate, l1 - a, vc)
        return l1, l2, vr, vc

    def ridge(self, a: np.ndarray, scale_km: float, polarity: str = "bright") -> tuple[np.ndarray, np.ndarray]:
        """Hessian line detector. Returns (line strength, strike deg).

        ``bright`` detects maxima ridges (e.g. ridges of THDR = edges of sources),
        ``dark`` detects troughs (valleys), ``both`` either.
        """
        hrr, hrc, hcc, m = self.hessian(a, scale_km)
        l1, l2, vr, vc = self._eig2(hrr, hrc, hcc)
        aniso = 1.0 - np.abs(l2) / np.maximum(np.abs(l1), 1e-12)
        if polarity == "bright":
            strength = np.maximum(-l1, 0) * aniso
        elif polarity == "dark":
            strength = np.maximum(l1, 0) * aniso
        else:
            strength = np.abs(l1) * aniso
        # along-strike vector is perpendicular to v1: (dr, dc) = (-vc, vr) -> east=vr, north=vc
        strike = np.degrees(np.arctan2(vr * self.grid.pixel_km_x, vc * self.grid.pixel_km_y)) % 180
        return _restore(strength, m), _restore(strike, m)

    def curvature(self, a: np.ndarray, scale_km: float) -> dict[str, np.ndarray]:
        """Principal curvatures of the scale-normalised Hessian (field treated as a surface)."""
        hrr, hrc, hcc, m = self.hessian(a, scale_km)
        tr = 0.5 * (hrr + hcc)
        d = np.sqrt((0.5 * (hrr - hcc)) ** 2 + hrc * hrc)
        k1, k2 = tr + d, tr - d
        return {
            "mean": _restore(tr, m),
            "max_abs": _restore(np.maximum(np.abs(k1), np.abs(k2)), m),
            "k1": _restore(k1, m),
            "k2": _restore(k2, m),
        }

    # -- wavelets --------------------------------------------------------
    def log_wavelet(self, a: np.ndarray, scale_km: float) -> np.ndarray:
        """Mexican-hat (scale-normalised -LoG) wavelet coefficient."""
        f, m = self._prep(a)
        s = self.sigma_px(scale_km)
        return _restore(-s * s * ndimage.gaussian_laplace(f, s, mode="nearest"), m)

    def log_gabor(self, a: np.ndarray, wavelength_km: float, n_orient: int = 8,
                  sigma_on_f: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
        """Oriented log-Gabor quadrature energy. Returns (max energy over orientations, strike deg)."""
        f, m = self._prep(a)
        f = f - f[m].mean() if m.any() else f
        h, w = f.shape
        ph, pw = max(h // 8, 1), max(w // 8, 1)
        F = np.fft.fft2(np.pad(f, ((ph, ph), (pw, pw)), mode="reflect").astype(np.float32))
        fy = np.fft.fftfreq(F.shape[0], d=self.grid.pixel_km_y)
        fx = np.fft.fftfreq(F.shape[1], d=self.grid.pixel_km_x)
        FN, FX = np.meshgrid(-fy, fx, indexing="ij")          # north, east frequencies
        radius = np.hypot(FN, FX)
        radius[0, 0] = 1.0
        radial = np.exp(-np.log(radius * wavelength_km) ** 2 / (2 * np.log(sigma_on_f) ** 2))
        radial[0, 0] = 0.0
        theta = np.arctan2(FN, FX)
        sigma_theta = np.pi / n_orient / 1.2
        best = np.zeros((h, w), np.float32)
        best_strike = np.zeros((h, w), np.float32)
        for k in range(n_orient):
            ang = k * np.pi / n_orient                          # wave-vector direction, CCW from east
            dtheta = np.abs(np.arctan2(np.sin(theta - ang), np.cos(theta - ang)))
            spread = np.exp(-dtheta ** 2 / (2 * sigma_theta ** 2))
            energy = np.abs(np.fft.ifft2(F * (radial * spread).astype(np.complex64)))[ph:ph + h, pw:pw + w]
            better = energy > best
            best = np.where(better, energy, best).astype(np.float32)
            best_strike = np.where(better, (180.0 - np.degrees(ang)) % 180, best_strike).astype(np.float32)
        return _restore(best, m), _restore(best_strike, m)

    # -- orientation -----------------------------------------------------
    def structure_tensor(self, a: np.ndarray, scale_km: float, integration: float = 3.0):
        """Dominant strike (deg), coherence [0, 1] and gradient energy of the local fabric."""
        f, m = self._prep(a)
        s = self.sigma_px(scale_km)
        g_r = ndimage.gaussian_filter(f, s, order=(1, 0), mode="nearest") / self.grid.pixel_km_y
        g_c = ndimage.gaussian_filter(f, s, order=(0, 1), mode="nearest") / self.grid.pixel_km_x
        ge, gn = g_c, -g_r
        si = s * integration
        jee = ndimage.gaussian_filter(ge * ge, si, mode="nearest")
        jnn = ndimage.gaussian_filter(gn * gn, si, mode="nearest")
        jen = ndimage.gaussian_filter(ge * gn, si, mode="nearest")
        phi = 0.5 * np.arctan2(2 * jen, jee - jnn)            # gradient direction, CCW from east
        strike = (90.0 - np.degrees(phi) + 90.0) % 180
        energy = jee + jnn
        coherence = np.sqrt((jee - jnn) ** 2 + 4 * jen ** 2) / np.maximum(energy, 1e-30)
        return _restore(strike, m), _restore(coherence, m), _restore(energy, m)

    # -- convenience -----------------------------------------------------
    def transform(self, name: str, a: np.ndarray, scale_km: float, fault_strike_deg: float = 0.0) -> np.ndarray:
        """Named fault-sensitive response, oriented so that larger = more fault-like."""
        if name == "thdr":
            return self.thdr(a, scale_km)
        if name == "dir_deriv":
            return np.abs(self.directional_derivative(a, scale_km, fault_strike_deg + 90.0))
        if name == "tilt_edge":
            return 1.0 - np.abs(self.tilt(a, scale_km)) / 90.0
        if name == "thdr_ridge":
            return self.ridge(self.thdr(a, scale_km), scale_km, "bright")[0]
        if name == "curvature":
            return self.curvature(a, scale_km)["max_abs"]
        if name == "mexican_hat":
            return np.abs(self.log_wavelet(a, scale_km))
        if name == "log_gabor":
            return self.log_gabor(a, 2.0 * scale_km)[0]
        raise KeyError(name)


TRANSFORMS = ("thdr", "dir_deriv", "tilt_edge", "thdr_ridge", "curvature", "mexican_hat", "log_gabor")
TRANSFORM_LABELS = {
    "thdr": "Total horiz. derivative",
    "dir_deriv": "Directional derivative (perp. to fault trend)",
    "tilt_edge": "Tilt-angle edge (1-|TA|/90)",
    "thdr_ridge": "Hessian ridge of THDR",
    "curvature": "Max |principal curvature|",
    "mexican_hat": "|Mexican-hat wavelet|",
    "log_gabor": "Oriented log-Gabor energy",
}
