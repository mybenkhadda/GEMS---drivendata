"""Potential-field transforms in the wavenumber domain with physical scaling.

Wavenumbers are angular, ``k = 2 pi f`` with ``f`` in cycles / metre, so a derivative of a field in nT is
returned in nT/m and upward continuation by ``h`` metres is ``exp(-|k| h)``. Before the FFT the field is:
1. filled inside its grid by nearest-valid values (nodata and outside-footprint pixels);
2. de-meaned over the data footprint;
3. padded by ``pad_frac`` of each dimension on every side with reflected values;
4. multiplied by a cosine taper that falls from 1 at the data extent to 0 over ``taper_px`` pixels,
so the periodic FFT sees no wrap-around step. Results are cropped back and set to NaN outside the footprint and
within the configured edge guard. ``rfft2`` is used; axis 0 is the row (southward) axis - only rotation-
invariant magnitudes (THD, analytic signal, tilt) leave this module, so the axis sign does not matter.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy import ndimage


def nearest_fill(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all():
        return a.astype(np.float64)
    if not valid.any():
        raise ValueError("field has no valid pixel")
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return a[tuple(idx)].astype(np.float64)


def _taper_1d(n: int, pad_lo: int, pad_hi: int, taper: int) -> np.ndarray:
    """1 over the data, cosine ramp to 0 over ``taper`` px inside each pad, 0 beyond."""
    w = np.zeros(n + pad_lo + pad_hi)
    w[pad_lo:pad_lo + n] = 1.0
    t_lo, t_hi = min(taper, pad_lo), min(taper, pad_hi)
    if t_lo > 0:
        w[pad_lo - t_lo:pad_lo] = 0.5 * (1 - np.cos(np.pi * np.arange(1, t_lo + 1) / (t_lo + 1)))
    if t_hi > 0:
        w[pad_lo + n:pad_lo + n + t_hi] = 0.5 * (1 - np.cos(np.pi * np.arange(t_hi, 0, -1) / (t_hi + 1)))
    return w


class FieldSpectrum:
    def __init__(self, field: np.ndarray, footprint: np.ndarray, pixel_m: float, pad_frac: float, taper_px: int):
        ok = np.isfinite(field) & footprint
        z = nearest_fill(np.where(ok, field, 0.0), ok)
        z -= z[ok].mean()
        h, w = z.shape
        ph, pw = int(np.ceil(pad_frac * h)), int(np.ceil(pad_frac * w))
        H, W = sfft.next_fast_len(h + 2 * ph, real=True), sfft.next_fast_len(w + 2 * pw, real=True)
        ph2, pw2 = (H - h) // 2, (W - w) // 2
        zp = np.pad(z, ((ph2, H - h - ph2), (pw2, W - w - pw2)), mode="reflect")
        zp *= np.outer(_taper_1d(h, ph2, H - h - ph2, taper_px), _taper_1d(w, pw2, W - w - pw2, taper_px))
        self.shape, self.crop = (H, W), (slice(ph2, ph2 + h), slice(pw2, pw2 + w))
        self.F = sfft.rfft2(zp, workers=-1)
        ky = 2 * np.pi * sfft.fftfreq(H, d=pixel_m)
        kx = 2 * np.pi * sfft.rfftfreq(W, d=pixel_m)
        self.ky, self.kx = ky[:, None], kx[None, :]
        self.k = np.sqrt(self.ky ** 2 + self.kx ** 2)

    def apply(self, mult) -> np.ndarray:
        out = sfft.irfft2(self.F * mult, s=self.shape, workers=-1)
        return out[self.crop]

    def derivatives(self, mult=1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(d/dx, d/dy, d/dz upward-positive vertical derivative) of the (optionally filtered) field."""
        return (self.apply(1j * self.kx * mult), self.apply(1j * self.ky * mult), self.apply(self.k * mult))


def potential_field_features(field: np.ndarray, footprint: np.ndarray, prefix: str, unit: str, pixel_m: float,
                             heights, pad_frac: float, taper_px: int, edge_guard_m: float,
                             cont_guard_cap_m: float) -> tuple[dict[str, np.ndarray], list[dict]]:
    spec = FieldSpectrum(field, footprint, pixel_m, pad_frac, taper_px)
    dist_edge = ndimage.distance_transform_edt(footprint) * pixel_m      # distance to the nearest non-footprint px
    out, reg = {}, []

    def put(name, arr, u, guard, desc):
        a = np.asarray(arr, np.float32)
        a[~footprint | (dist_edge <= guard)] = np.nan
        out[name] = a
        reg.append({"channel": name, "group": "field", "unit": u, "edge_guard_m": guard, "kind": "scalar",
                    "definition": desc})

    raw = np.where(footprint, field, np.nan)
    put(f"{prefix}_field", raw, unit, 0.0, "raw field as delivered (nodata -> NaN)")
    dx, dy, dz = spec.derivatives()
    thd = np.hypot(dx, dy)
    put(f"{prefix}_vd", dz, f"{unit}/m", edge_guard_m, "first vertical derivative F^-1{|k| F}")
    put(f"{prefix}_thd", thd, f"{unit}/m", edge_guard_m, "total horizontal derivative sqrt(fx^2+fy^2)")
    put(f"{prefix}_tilt", np.arctan2(dz, thd + 1e-12 * np.nanmax(thd)), "rad", edge_guard_m,
        "tilt angle atan2(fz, THD + eps)")
    put(f"{prefix}_as", np.sqrt(dx ** 2 + dy ** 2 + dz ** 2), f"{unit}/m", edge_guard_m,
        "analytic-signal amplitude sqrt(fx^2+fy^2+fz^2)")
    del dx, dy, dz, thd
    base = spec.apply(1.0)                                                  # de-meaned field after padding round trip
    ucs = {}
    for h in heights:
        up = np.exp(-spec.k * float(h))
        ucs[h] = spec.apply(up)
        put(f"{prefix}_uc{h}", ucs[h], unit, min(float(h), cont_guard_cap_m), f"upward continuation exp(-|k| {h} m)")
    for h in heights:
        put(f"{prefix}_res{h}", base - ucs[h], unit, min(float(h), cont_guard_cap_m),
            f"residual field minus upward continuation to {h} m")
    for h in heights:
        up = np.exp(-spec.k * float(h))
        gx, gy = spec.apply(1j * spec.kx * up), spec.apply(1j * spec.ky * up)
        put(f"{prefix}_thd_uc{h}", np.hypot(gx, gy), f"{unit}/m", max(edge_guard_m, min(float(h), cont_guard_cap_m)),
            f"total horizontal derivative after continuation to {h} m")
    return out, reg
