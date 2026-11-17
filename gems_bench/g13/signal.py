"""Geophysical signal processing (sections 08-10), mask-aware.

Conventions
-----------
* Arrays are north-up rasters: row index grows *south*, column index grows *east*. ``x`` = east, ``y`` = north.
* Orientations are **axial strike angles** ``theta`` in [0, pi): 0 = east-west strike, pi/2 = north-south strike,
  counter-clockwise from east. Gradient direction is perpendicular to strike. Axial quantities are encoded as
  ``(cos 2 theta, sin 2 theta)`` so 0 and 180 degrees are the same strike.
* Nodata never leaks into a filter: fields are filled by normalised convolution before any filtering, the output is
  set to NaN where the input was nodata, and ``boundary_distance`` lets later stages audit edge dependence.
* Spectral (FFT) operators work on the full 100 m grid (about 3.7 k x 3.3 k pixels: cheap), with reflect padding and a
  cosine taper so the periodic FFT does not create artificial edges at the raster border.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi

EPS = 1e-12


# =============================================================================== nodata handling
def fill_nodata(x: np.ndarray, valid: np.ndarray | None = None, max_sigma: float = 256.0) -> np.ndarray:
    """Fill NaN / invalid cells by multiscale normalised convolution (smooth, no hard edges).

    Valid cells keep their values exactly; each gap is filled from the nearest scale whose support reaches it.
    """
    x = np.asarray(x, np.float32)
    ok = np.isfinite(x) if valid is None else (np.asarray(valid, bool) & np.isfinite(x))
    out = np.where(ok, x, 0.0).astype(np.float32)
    if ok.all():
        return out
    if not ok.any():
        return np.zeros_like(out)
    filled = ok.copy()
    sigma = 2.0
    base = np.where(ok, x, 0.0).astype(np.float32)
    w = ok.astype(np.float32)
    while not filled.all() and sigma <= max_sigma:
        num = ndi.gaussian_filter(base, sigma, mode="nearest", truncate=3.0)
        den = ndi.gaussian_filter(w, sigma, mode="nearest", truncate=3.0)
        reach = (den > 1e-3) & ~filled
        out[reach] = num[reach] / den[reach]
        filled |= reach
        sigma *= 2.0
    if not filled.all():
        out[~filled] = float(np.mean(x[ok]))
    return out


def boundary_distance(valid: np.ndarray) -> np.ndarray:
    """Distance (pixels) from each valid cell to the nearest invalid cell or raster edge."""
    v = np.pad(np.asarray(valid, bool), 1, constant_values=False)
    return ndi.distance_transform_edt(v)[1:-1, 1:-1].astype(np.float32)


def masked_smooth(x: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised-convolution Gaussian smoothing: invalid cells contribute nothing."""
    w = np.asarray(valid, np.float32)
    num = ndi.gaussian_filter(np.where(valid, x, 0.0).astype(np.float32), sigma, mode="reflect", truncate=3.0)
    den = ndi.gaussian_filter(w, sigma, mode="reflect", truncate=3.0)
    return num / np.maximum(den, 1e-5)


def remask(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    a[~valid] = np.nan
    return a


# =============================================================================== spectral field
def _fast_len(n: int) -> int:
    return sfft.next_fast_len(int(n), real=True)


class SpectralField:
    """FFT of one filled, padded, tapered potential field with cached wavenumbers.

    ``dx`` / ``dy`` / ``dz`` are first derivatives (east, north, vertical: F^-1[|k| F]); ``upward(h)`` is upward
    continuation by ``h`` metres (F^-1[e^{-|k| h} F]). All outputs are cropped back to the input shape.
    """

    def __init__(self, x: np.ndarray, valid: np.ndarray, pixel_m: float = 100.0, pad: int = 256):
        self.valid = np.asarray(valid, bool) & np.isfinite(x)
        f = fill_nodata(x, self.valid)
        self.mean = float(f[self.valid].mean()) if self.valid.any() else 0.0
        f = f - self.mean
        h, w = f.shape
        self.shape, self.pad, self.pixel_m = (h, w), pad, pixel_m
        H, W = _fast_len(h + 2 * pad), _fast_len(w + 2 * pad)
        p = np.pad(f, ((pad, H - h - pad), (pad, W - w - pad)), mode="reflect")
        # cosine taper over the padding band (to zero mean) removes the wrap-around discontinuity
        ty = _taper(H, pad, h)
        tx = _taper(W, pad, w)
        p = p * ty[:, None] * tx[None, :]
        self.F = sfft.rfft2(p.astype(np.float32), workers=-1)
        ky = 2 * np.pi * sfft.fftfreq(H, d=pixel_m)
        kx = 2 * np.pi * sfft.rfftfreq(W, d=pixel_m)
        self.KX, self.KR = np.meshgrid(kx, ky)  # KR = wavenumber along rows (south-positive)
        self.K = np.hypot(self.KX, self.KR)
        self.HW = (H, W)

    def _inv(self, G: np.ndarray) -> np.ndarray:
        a = sfft.irfft2(G, s=self.HW, workers=-1)
        h, w = self.shape
        return a[self.pad : self.pad + h, self.pad : self.pad + w].astype(np.float32)

    def field(self) -> np.ndarray:
        return self._inv(self.F) + self.mean

    def dx(self) -> np.ndarray:
        return self._inv(1j * self.KX * self.F)

    def dy(self) -> np.ndarray:  # north derivative = -d/drow
        return -self._inv(1j * self.KR * self.F)

    def dz(self) -> np.ndarray:
        return self._inv(self.K * self.F)

    def upward(self, h_m: float) -> np.ndarray:
        return self._inv(np.exp(-self.K * h_m) * self.F) + self.mean

    def upward_derivatives(self, h_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        G = np.exp(-self.K * h_m) * self.F
        return self._inv(1j * self.KX * G), -self._inv(1j * self.KR * G), self._inv(self.K * G)

    def tensor(self) -> dict[str, np.ndarray]:
        """Gradient tensor of the field treated as a potential-field component (Txx ... Tzz)."""
        kx, kr, k, F = self.KX, -self.KR, self.K, self.F  # north wavenumber = -KR
        return {
            "xx": self._inv(-kx * kx * F),
            "yy": self._inv(-kr * kr * F),
            "zz": self._inv(k * k * F),
            "xy": self._inv(-kx * kr * F),
            "xz": self._inv(1j * kx * k * F),
            "yz": self._inv(1j * kr * k * F),
        }

    def bandpass_loggabor(self, wavelength_m: float, sigma_on_f: float = 0.55) -> np.ndarray:
        """Radial log-Gabor transfer function centred on ``wavelength_m`` (spectral domain array)."""
        k0 = 2 * np.pi / wavelength_m
        with np.errstate(divide="ignore"):
            g = np.exp(-(np.log(np.maximum(self.K, EPS) / k0) ** 2) / (2 * np.log(sigma_on_f) ** 2))
        g[self.K == 0] = 0.0
        return g


def _taper(n_total: int, pad: int, n_core: int) -> np.ndarray:
    t = np.ones(n_total, np.float32)
    if pad <= 0:
        return t
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, pad, dtype=np.float32)))
    t[:pad] = ramp
    end = pad + n_core
    tail = n_total - end
    if tail > 0:
        t[end:] = ramp[::-1][:tail] if tail <= pad else np.r_[ramp[::-1], np.zeros(tail - pad, np.float32)]
    return t


# =============================================================================== potential-field products
def potential_field_products(
    x: np.ndarray, valid: np.ndarray, *, pixel_m: float = 100.0, heights_m=(500, 1000, 2000, 4000)
) -> dict[str, np.ndarray]:
    """Edge-detection products of a potential field (section 09/10), all mask-aware.

    thd   total horizontal derivative sqrt(fx^2 + fy^2)          maxima over steep contacts
    vd    first vertical derivative                              sharpens shallow sources
    tilt  arctan(vd / thd) in [-pi/2, pi/2]                      amplitude-balanced edge (zero over contacts)
    tdx   horizontal gradient of the tilt                        sharp edge locator
    asig  analytic signal sqrt(fx^2 + fy^2 + vd^2)               edges largely independent of magnetisation direction
    theta strike angle of the horizontal gradient (axial)        orientation
    ucH   upward continuation at H m, its THD and the residual   depth-band separation
    nss   normalised source strength sqrt(-l2^2 - l1 l3)         rotation-invariant source proxy (Beiki 2012 form)
    """
    sf = SpectralField(x, valid, pixel_m)
    fx, fy, fz = sf.dx(), sf.dy(), sf.dz()
    thd = np.hypot(fx, fy)
    tilt = np.arctan2(fz, thd + EPS)
    tr, tc = np.gradient(tilt, pixel_m)
    out = {
        "thd": thd,
        "vd": fz,
        "tilt": tilt,
        "tdx": np.hypot(tc, -tr),
        "asig": np.sqrt(fx**2 + fy**2 + fz**2),
        "theta": np.mod(np.arctan2(fy, fx) + np.pi / 2, np.pi),
    }
    for h in heights_m:
        ux, uy, uz = sf.upward_derivatives(h)
        out[f"uc{h}_thd"] = np.hypot(ux, uy)
        out[f"uc{h}_tilt"] = np.arctan2(uz, np.hypot(ux, uy) + EPS)
    base = sf.field()
    for h in heights_m:
        out[f"res{h}"] = base - sf.upward(h)
    out["nss"] = normalized_source_strength(sf)
    v = sf.valid
    return {k: remask(a, v) for k, a in out.items()}


def normalized_source_strength(sf: SpectralField, chunk_rows: int = 256) -> np.ndarray:
    T = sf.tensor()
    h, w = sf.shape
    out = np.empty((h, w), np.float32)
    for r0 in range(0, h, chunk_rows):
        r1 = min(h, r0 + chunk_rows)
        M = np.empty((r1 - r0, w, 3, 3), np.float64)
        M[..., 0, 0], M[..., 1, 1], M[..., 2, 2] = T["xx"][r0:r1], T["yy"][r0:r1], T["zz"][r0:r1]
        M[..., 0, 1] = M[..., 1, 0] = T["xy"][r0:r1]
        M[..., 0, 2] = M[..., 2, 0] = T["xz"][r0:r1]
        M[..., 1, 2] = M[..., 2, 1] = T["yz"][r0:r1]
        lam = np.linalg.eigvalsh(M)  # ascending: l3 <= l2 <= l1
        l3, l2, l1 = lam[..., 0], lam[..., 1], lam[..., 2]
        out[r0:r1] = np.sqrt(np.clip(-(l2**2) - l1 * l3, 0.0, None))
    return out


# =============================================================================== local differential geometry
def gaussian_gradient(x: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """(d/dx east, d/dy north) of a Gaussian-smoothed field, per pixel.

    Smoothing (normalised kernel) followed by central differences: exact on linear fields, unlike a truncated
    derivative-of-Gaussian kernel whose truncation biases the gradient magnitude by ~1 %."""
    sm = ndi.gaussian_filter(np.asarray(x, np.float32), sigma, mode="nearest", truncate=4.0)
    gr, gc = np.gradient(sm)
    return gc.astype(np.float32), (-gr).astype(np.float32)


def hessian_eigen(x: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale-normalised Hessian eigenvalues (l1 >= l2) and the strike of the dominant curvature axis.

    A ridge (linear high) has l2 << 0, a valley (linear low) l1 >> 0; ``theta`` is the strike of the line, i.e.
    perpendicular to the eigenvector of the largest-|l| eigenvalue.
    """
    s2 = sigma**2
    hrr = ndi.gaussian_filter(x, sigma, order=(2, 0), mode="reflect", truncate=3.0) * s2
    hcc = ndi.gaussian_filter(x, sigma, order=(0, 2), mode="reflect", truncate=3.0) * s2
    hrc = ndi.gaussian_filter(x, sigma, order=(1, 1), mode="reflect", truncate=3.0) * s2
    # in (x east, y north): hxx = hcc, hyy = hrr, hxy = -hrc
    hxx, hyy, hxy = hcc, hrr, -hrc
    tr, det = hxx + hyy, hxx * hyy - hxy**2
    disc = np.sqrt(np.maximum((tr / 2) ** 2 - det, 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    # eigenvector of the largest |l|: angle phi; the line strike is phi + 90 deg
    phi = 0.5 * np.arctan2(2 * hxy, hxx - hyy)
    big = np.abs(l1) >= np.abs(l2)
    phi = np.where(big, phi, phi + np.pi / 2)
    theta = np.mod(phi + np.pi / 2, np.pi)
    return l1.astype(np.float32), l2.astype(np.float32), theta.astype(np.float32)


def structure_tensor(x: np.ndarray, sigma_grad: float, sigma_int: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coherence in [0, 1], strike ``theta`` (axial, perpendicular to the dominant gradient) and energy."""
    gx, gy = gaussian_gradient(x, sigma_grad)
    jxx = ndi.gaussian_filter(gx * gx, sigma_int, mode="reflect")
    jyy = ndi.gaussian_filter(gy * gy, sigma_int, mode="reflect")
    jxy = ndi.gaussian_filter(gx * gy, sigma_int, mode="reflect")
    tr = jxx + jyy
    coh = np.hypot(jxx - jyy, 2 * jxy) / np.maximum(tr, EPS)
    phi = 0.5 * np.arctan2(2 * jxy, jxx - jyy)  # dominant gradient direction
    theta = np.mod(phi + np.pi / 2, np.pi)
    return np.clip(coh, 0, 1).astype(np.float32), theta.astype(np.float32), tr.astype(np.float32)


def steer_second_derivative(x: np.ndarray, sigma: float, theta: float) -> np.ndarray:
    """Steerable G2 response across a line of strike ``theta`` (Freeman & Adelson): second derivative along the
    line normal. Negative for ridges, positive for valleys."""
    hrr = ndi.gaussian_filter(x, sigma, order=(2, 0), mode="reflect") * sigma**2
    hcc = ndi.gaussian_filter(x, sigma, order=(0, 2), mode="reflect") * sigma**2
    hrc = ndi.gaussian_filter(x, sigma, order=(1, 1), mode="reflect") * sigma**2
    nx, ny = -np.sin(theta), np.cos(theta)  # unit normal (east, north)
    hxx, hyy, hxy = hcc, hrr, -hrc
    return (nx * nx * hxx + 2 * nx * ny * hxy + ny * ny * hyy).astype(np.float32)


# =============================================================================== directional spectral filters
def gabor_bank(
    x: np.ndarray, valid: np.ndarray, wavelengths_m, n_orient: int = 8, pixel_m: float = 100.0, angular_sigma: float = 0.45
) -> dict[str, np.ndarray]:
    """Oriented log-Gabor energy (FFT domain). Per wavelength: max energy over orientations, the strike of the
    maximum (axial), anisotropy (max - min) / sum, and directional coherence of the energy distribution."""
    sf = SpectralField(x, valid, pixel_m)
    ang = np.arctan2(-sf.KR, sf.KX)  # spectral angle of the wave vector (north-positive)
    out = {}
    for lam in wavelengths_m:
        radial = sf.bandpass_loggabor(lam)
        energies = []
        for j in range(n_orient):
            th = j * np.pi / n_orient  # wave-vector direction; strike = th + 90 deg
            d = np.angle(np.exp(1j * 2 * (ang - th))) / 2  # axial difference in [-pi/2, pi/2]
            G = radial * np.exp(-(d**2) / (2 * angular_sigma**2))
            # analytic (one-sided in angle) energy: |f * g_even + i f * g_odd| ~ amplitude of band-passed field
            even = sf._inv(G * sf.F)
            r = np.hypot(even, sf._inv(G * sf.F * 1j * np.sign(np.cos(ang - th) + EPS)))
            energies.append(r)
        E = np.stack(energies)
        jmax = np.argmax(E, 0)
        emax, emin, esum = E.max(0), E.min(0), E.sum(0) + EPS
        strike = np.mod(jmax * np.pi / n_orient + np.pi / 2, np.pi)
        th_all = np.arange(n_orient)[:, None, None] * np.pi / n_orient
        c2 = (E * np.cos(2 * th_all)).sum(0) / esum
        s2 = (E * np.sin(2 * th_all)).sum(0) / esum
        out[f"gabor{lam}_energy"] = remask(np.log1p(emax), sf.valid)
        out[f"gabor{lam}_aniso"] = remask((emax - emin) / esum * n_orient, sf.valid)
        out[f"gabor{lam}_dircoh"] = remask(np.hypot(c2, s2), sf.valid)
        out[f"gabor{lam}_theta"] = remask(strike, sf.valid)
    return out


def monogenic(x: np.ndarray, valid: np.ndarray, wavelengths_m, pixel_m: float = 100.0) -> dict[str, np.ndarray]:
    """Riesz-transform monogenic signal (Felsberg & Sommer) per wavelength, plus multiscale phase congruency.

    local amplitude A, local phase phi (0 = line-like peak, +-pi/2 = edge-like step) and local orientation.
    Phase congruency PC = |sum_s f_s| / sum_s A_s (energy over amplitude), in [0, 1]: high where all scales agree
    on a feature, which is a scale-invariant edge/line detector independent of contrast.
    """
    sf = SpectralField(x, valid, pixel_m)
    k = np.maximum(sf.K, EPS)
    r1, r2 = 1j * sf.KX / k, 1j * (-sf.KR) / k  # Riesz kernels (east, north)
    out = {}
    se, sx, sy, sa = 0.0, 0.0, 0.0, 0.0
    for lam in wavelengths_m:
        G = sf.bandpass_loggabor(lam) * sf.F
        fe, fx, fy = sf._inv(G), sf._inv(r1 * G), sf._inv(r2 * G)
        odd = np.hypot(fx, fy)
        A = np.hypot(fe, odd)
        out[f"mono{lam}_amp"] = remask(np.log1p(A / (np.nanstd(A) + EPS)), sf.valid)
        out[f"mono{lam}_phase"] = remask(np.arctan2(odd, fe), sf.valid)  # [0, pi]: 0 peak, pi/2 edge, pi trough
        se, sx, sy, sa = se + fe, sx + fx, sy + fy, sa + A
    noise = np.percentile(sa, 5) if np.size(sa) else 0.0
    pc = np.sqrt(se**2 + sx**2 + sy**2) / (sa + noise + EPS)
    out["phase_congruency"] = remask(np.clip(pc, 0, 1), sf.valid)
    return out


# =============================================================================== texture
def local_entropy(x: np.ndarray, valid: np.ndarray, window_px: int = 9, bins: int = 16) -> np.ndarray:
    """Shannon entropy (bits) of the locally binned field (bins from global valid quantiles, label-free)."""
    v = np.asarray(valid, bool) & np.isfinite(x)
    if not v.any():
        return np.full(x.shape, np.nan, np.float32)
    edges = np.quantile(x[v], np.linspace(0, 1, bins + 1)[1:-1])
    q = np.digitize(np.where(v, x, edges[0]), edges)
    w = ndi.uniform_filter(v.astype(np.float32), window_px, mode="reflect")
    H = np.zeros(x.shape, np.float32)
    for b in range(bins):
        p = ndi.uniform_filter(((q == b) & v).astype(np.float32), window_px, mode="reflect") / np.maximum(w, 1e-6)
        H -= np.where(p > 0, p * np.log2(np.maximum(p, 1e-12)), 0.0)
    return remask(H, v)


# =============================================================================== orientation helpers
def axial_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest angle between two axial orientations, in [0, pi/2]."""
    d = np.abs(np.mod(a - b + np.pi / 2, np.pi) - np.pi / 2)
    return d.astype(np.float32)


def axial_encode(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.cos(2 * theta).astype(np.float32), np.sin(2 * theta).astype(np.float32)


def orientation_persistence(thetas: list[np.ndarray], weights: list[np.ndarray]) -> np.ndarray:
    """Resultant length of weighted axial orientations across scales (1 = same strike at every scale)."""
    c = sum(w * np.cos(2 * t) for t, w in zip(thetas, weights))
    s = sum(w * np.sin(2 * t) for t, w in zip(thetas, weights))
    tot = sum(weights) + EPS
    return (np.hypot(c, s) / tot).astype(np.float32)


# =============================================================================== windowing
def windowed_apply(func, x: np.ndarray, *, block: int = 512, halo: int = 96, **kw) -> np.ndarray:
    """Apply a local (finite-support) filter tile by tile with a halo; identical to the full-array result when the
    filter support is <= ``halo`` (tested). Use for local filters on rasters too large for one pass."""
    h, w = x.shape
    out = np.empty((h, w), np.float32)
    for r in range(0, h, block):
        for c in range(0, w, block):
            r0, c0 = max(0, r - halo), max(0, c - halo)
            r1, c1 = min(h, r + block + halo), min(w, c + block + halo)
            y = func(x[r0:r1, c0:c1], **kw)
            out[r : min(h, r + block), c : min(w, c + block)] = y[
                r - r0 : r - r0 + min(block, h - r), c - c0 : c - c0 + min(block, w - c)
            ]
    return out


def multiscale_relief(x: np.ndarray, valid: np.ndarray, sigmas_px) -> dict[str, np.ndarray]:
    """Topographic position index and local standard deviation at several scales (mask-aware)."""
    out = {}
    for s in sigmas_px:
        m = masked_smooth(x, valid, s)
        m2 = masked_smooth(x * x, valid, s)
        out[f"tpi_s{s}"] = remask(x - m, valid)
        out[f"lstd_s{s}"] = remask(np.sqrt(np.maximum(m2 - m * m, 0)), valid)
    return out
