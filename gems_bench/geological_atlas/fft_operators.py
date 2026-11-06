"""Wavenumber-domain operators for potential fields on the projected 100 m grid.

Angular wavenumbers ``k = 2 pi f`` (rad/m, f from ``fftfreq(n, d=pixel_m)``), so first derivatives of a field in
nT come out in nT/m and upward continuation by h metres is ``exp(-|k| h)``. Preparation against ringing and
wrap-around: nearest-valid fill of nodata / off-footprint pixels, de-mean over the footprint, reflect padding by
``pad_frac`` on every side (fast FFT length), cosine taper over ``taper_px`` inside the padding, crop back.
Products are NaN outside the footprint and within the edge guard. Geographic axes: x = east (columns),
y = north (= minus the row direction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy import ndimage


def nearest_fill(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all():
        return a.astype(np.float64)
    if not valid.any():
        raise ValueError("no valid pixel to fill from")
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return a[tuple(idx)].astype(np.float64)


def _taper(n: int, lo: int, hi: int, t: int) -> np.ndarray:
    w = np.zeros(n + lo + hi)
    w[lo:lo + n] = 1.0
    a, b = min(t, lo), min(t, hi)
    if a:
        w[lo - a:lo] = 0.5 * (1 - np.cos(np.pi * np.arange(1, a + 1) / (a + 1)))
    if b:
        w[lo + n:lo + n + b] = 0.5 * (1 - np.cos(np.pi * np.arange(b, 0, -1) / (b + 1)))
    return w


class FieldSpectrum:
    def __init__(self, field: np.ndarray, footprint: np.ndarray, pixel_m: float, pad_frac: float, taper_px: int):
        ok = np.isfinite(field) & footprint
        z = nearest_fill(np.where(ok, field, 0.0), ok)
        self.mean = float(z[ok].mean())
        z -= self.mean
        h, w = z.shape
        H = sfft.next_fast_len(h + 2 * int(np.ceil(pad_frac * h)), real=True)
        W = sfft.next_fast_len(w + 2 * int(np.ceil(pad_frac * w)), real=True)
        lo_r, lo_c = (H - h) // 2, (W - w) // 2
        zp = np.pad(z, ((lo_r, H - h - lo_r), (lo_c, W - w - lo_c)), mode="reflect")
        self.window = np.outer(_taper(h, lo_r, H - h - lo_r, taper_px), _taper(w, lo_c, W - w - lo_c, taper_px))
        zp *= self.window
        self.shape, self.crop, self.px = (H, W), (slice(lo_r, lo_r + h), slice(lo_c, lo_c + w)), pixel_m
        self.F = sfft.rfft2(zp, workers=-1)
        k_row = 2 * np.pi * sfft.fftfreq(H, d=pixel_m)[:, None]
        self.kx = 2 * np.pi * sfft.rfftfreq(W, d=pixel_m)[None, :]
        self.ky = -k_row                      # north component (rows run south)
        self.k = np.sqrt(self.kx ** 2 + self.ky ** 2)

    def apply(self, mult) -> np.ndarray:
        return sfft.irfft2(self.F * mult, s=self.shape, workers=-1)[self.crop]

    def dx(self, m=1.0):
        return self.apply(1j * self.kx * m)

    def dy(self, m=1.0):
        return self.apply(1j * self.ky * m)

    def dz(self, m=1.0):
        return self.apply(self.k * m)

    def second(self, m=1.0):
        """(d2/dx2, d2/dy2, d2/dxdy)."""
        return (self.apply(-(self.kx ** 2) * m), self.apply(-(self.ky ** 2) * m), self.apply(-(self.kx * self.ky) * m))

    def upward(self, h: float):
        return np.exp(-self.k * float(h))

    def radial_spectrum(self, n_bins: int = 60) -> pd.DataFrame:
        """Azimuthally averaged power |F|^2 of the tapered, padded field vs angular wavenumber (rad/km)."""
        P = np.abs(self.F) ** 2
        P[:, 1:-1] *= 2.0                     # rfft: interior columns stand for +-kx
        k = self.k.ravel()
        kmax = float(np.pi / self.px)        # Nyquist
        edges = np.linspace(0, kmax, n_bins + 1)
        idx = np.digitize(k, edges) - 1
        ok = (idx >= 0) & (idx < n_bins)
        pw = np.bincount(idx[ok], weights=P.ravel()[ok], minlength=n_bins)
        cnt = np.bincount(idx[ok], minlength=n_bins)
        mid = 0.5 * (edges[1:] + edges[:-1])
        df = pd.DataFrame({"k_rad_per_km": mid * 1e3, "wavelength_km": np.where(mid > 0, 2 * np.pi / mid / 1e3, np.inf),
                           "power_mean": pw / np.maximum(cnt, 1), "n": cnt})
        df["log10_power"] = np.log10(df.power_mean.clip(lower=1e-30))
        return df[df.n > 0].reset_index(drop=True)

    def band_energy(self, bands_km=((0.2, 1), (1, 3), (3, 10), (10, 30), (30, 1e9))) -> pd.DataFrame:
        P = np.abs(self.F) ** 2
        P[:, 1:-1] *= 2.0
        lam = np.where(self.k > 0, 2 * np.pi / np.maximum(self.k, 1e-30) / 1e3, np.inf)
        tot = float(P[self.k > 0].sum())
        rows = []
        for lo, hi in bands_km:
            m = (lam >= lo) & (lam < hi) & (self.k > 0)
            rows.append({"wavelength_from_km": lo, "wavelength_to_km": hi if hi < 1e8 else np.inf,
                         "energy_share": float(P[m].sum() / tot) if tot else np.nan})
        return pd.DataFrame(rows)
