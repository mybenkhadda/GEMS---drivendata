"""Section 03: Fourier, wavelet and continuation layers of the magnetic and gravity fields (label-free).

Directional windowed FFT (``window_spectra``)
    Overlapping windows (``window_px`` = 64 px = 6.4 km, ``stride_px`` = 16) of the nearest-filled field; a window is
    used only when >= ``min_valid_frac`` of it is inside the data footprint. Each window: least-squares plane removed
    (regional gradient leakage), 2-D Hann taper (spectral leakage / edge artefacts), |FFT|^2. Per window, inside the
    "short" wavelength band: power-weighted doubled-angle resultant of the WAVEVECTOR azimuth -> anisotropy
    (resultant length, 0 isotropic ... 1 one direction) and dominant wavevector azimuth; directional entropy of the
    sector powers; log short / long band energy ratio; radial log-power slope; log band energy.

    Orientation convention: a linear structure striking theta concentrates spectral energy along the wavevector
    PERPENDICULAR to it, so ``trace strike = (wavevector azimuth + 90) mod 180`` (azimuths clockwise from north,
    axial). Strike is stored as (cos 2t, sin 2t) so the 0 / 180 deg wrap is continuous. The radial slope is a
    spectral-shape descriptor only: it is NOT interpreted as a source depth.

    Window statistics live on a coarse grid of window centres and are bilinearly resampled to the 100 m grid (edges:
    nearest window); no full (windows x W x W) tensor is ever kept.

Oriented wavelets (``wavelet_layers``)
    Steerable one-sided log-Gabor filters (radial log-Gaussian around 1 / wavelength, angular Gaussian around each of
    ``n_orient`` orientations in [0, pi)) applied in the wavenumber domain of the reflect-padded field (one FFT per
    field, one inverse FFT per filter; only running aggregates are kept). Per scale: energy (sum of quadrature
    amplitudes); across orientations: doubled-angle resultant -> dominant orientation (trace strike = wavevector + 90)
    and directional coherence; across scales: persistence (share of scales with energy rank >= q), fine / coarse
    energy ratio. These are compared label-free with the existing Gabor / ridge / DoG features (Spearman).

Continuation (``continuation_layers``): upward continuation (``m8.potential_fields.upward``, stable) at the given
heights; wavelength-separated bands = up(h1) - up(h2) and the residual f - up(h_min). No downward continuation.

Optional STFT (``spectral.stft.enabled``): the windowed-FFT descriptors at a finer window / stride, ablation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

PX_KM = 0.1
BOUNDED = ("_c2", "_s2", "coherence", "persistence", "anisotropy", "dir_entropy")   # stored as is (not standardised)


# =============================================================================== windowed FFT
def _plane_projector(W: int) -> np.ndarray:
    yy, xx = np.mgrid[0:W, 0:W].astype(np.float64)
    A = np.stack([np.ones(W * W), xx.ravel() / W, yy.ravel() / W], 1)
    return A @ np.linalg.pinv(A)                                     # (W^2, W^2) projection onto planes


def _freq(W: int, px_km: float):
    f = np.fft.fftshift(np.fft.fftfreq(W, d=px_km))                  # cycles / km
    fr, fc = np.meshgrid(f, f, indexing="ij")                        # rows (south), cols (east)
    kE, kN = fc, -fr
    kmag = np.hypot(kE, kN)
    az = (np.degrees(np.arctan2(kE, kN)) % 180.0)                     # wavevector azimuth, axial
    return kmag, az


def window_spectra(field: np.ndarray, valid: np.ndarray, fc: dict, px_km: float = PX_KM) -> dict:
    """Coarse-grid window descriptors + the window-centre coordinates (rows, cols)."""
    from ..m8.potential_fields import fill

    W, S = int(fc["window_px"]), int(fc["stride_px"])
    f, m = fill(field)
    m &= np.asarray(valid, bool)
    f = f.astype(np.float32)
    H, Wd = f.shape
    view = np.lib.stride_tricks.sliding_window_view(f, (W, W))[::S, ::S]
    mview = np.lib.stride_tricks.sliding_window_view(m, (W, W))[::S, ::S]
    nr, nc = view.shape[:2]
    P = _plane_projector(W).astype(np.float32)
    taper = np.outer(np.hanning(W), np.hanning(W)).astype(np.float32)
    kmag, az = _freq(W, px_km)
    lo_s, hi_s = 1.0 / float(fc["short_km"][1]), 1.0 / float(fc["short_km"][0])
    lo_l, hi_l = 1.0 / float(fc["long_km"][1]), 1.0 / float(fc["long_km"][0])
    band_s = (kmag >= lo_s) & (kmag <= hi_s)
    band_l = (kmag >= lo_l) & (kmag < lo_s)
    nsec = int(fc["n_sectors"])
    sec = np.minimum((az / (180.0 / nsec)).astype(int), nsec - 1)
    e2 = np.exp(2j * np.radians(az))
    rb = np.geomspace(max(lo_l, 1.0 / (W * px_km)), 0.5 / px_km, 12)
    rbin = np.digitize(kmag, rb)
    out = {k: np.full((nr, nc), np.nan, np.float32) for k in
           ("anisotropy", "k_azimuth", "dir_entropy", "short_long", "radial_slope", "log_energy")}
    sectors = np.full((nr, nc, nsec), np.nan, np.float32)
    min_valid = float(fc["min_valid_frac"])
    for i in range(nr):
        vf = mview[i].reshape(nc, -1).mean(1)
        ok = vf >= min_valid
        if not ok.any():
            continue
        blk = view[i][ok].reshape(-1, W * W).astype(np.float32)
        blk = blk - blk @ P.T                                         # plane removed (leakage of regional trend)
        blk = blk.reshape(-1, W, W) * taper
        pw = np.abs(np.fft.fftshift(np.fft.fft2(blk), axes=(1, 2))) ** 2
        es = (pw * band_s).sum((1, 2))
        el = (pw * band_l).sum((1, 2))
        res = (pw * band_s * e2).sum((1, 2))
        aniso = np.abs(res) / np.maximum(es, 1e-12)
        kaz = (np.degrees(0.5 * np.angle(res)) % 180.0)
        sp = np.stack([(pw * (band_s & (sec == j))).sum((1, 2)) for j in range(nsec)], 1)
        p = sp / np.maximum(sp.sum(1, keepdims=True), 1e-12)
        ent = -(p * np.log(np.maximum(p, 1e-12))).sum(1) / np.log(nsec)
        prof = np.stack([pw[:, rbin == j].mean(1) if (rbin == j).any() else np.full(len(pw), np.nan)
                         for j in range(1, len(rb))], 1)
        lk = np.log(np.sqrt(rb[1:] * rb[:-1]))
        lp = np.log(np.maximum(prof, 1e-20))
        good = np.isfinite(lp).all(0)
        slope = np.polyfit(lk[good], lp[:, good].T, 1)[0] if good.sum() >= 3 else np.full(len(pw), np.nan)
        cols = np.flatnonzero(ok)
        out["anisotropy"][i, cols] = aniso
        out["k_azimuth"][i, cols] = kaz
        out["dir_entropy"][i, cols] = ent
        out["short_long"][i, cols] = np.log(np.maximum(es, 1e-12) / np.maximum(el, 1e-12))
        out["radial_slope"][i, cols] = slope
        out["log_energy"][i, cols] = np.log(np.maximum(es, 1e-12))
        sectors[i, cols] = p
    out["rows"] = W / 2.0 - 0.5 + S * np.arange(nr)
    out["cols"] = W / 2.0 - 0.5 + S * np.arange(nc)
    out["sectors"] = sectors
    return out


def trace_strike_from_wavevector(k_az_deg):
    """Strike of the linear structure whose spectral energy points along wavevector azimuth ``k_az_deg``."""
    return (np.asarray(k_az_deg, np.float64) + 90.0) % 180.0


def to_native(coarse: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape) -> np.ndarray:
    """Bilinear resampling of a window-centre grid to the native grid (NaN windows filled by nearest first)."""
    from gems_eda.enhance import fill_nan

    a = fill_nan(np.asarray(coarse, np.float32)) if np.isfinite(coarse).any() else np.zeros_like(coarse)
    H, W = shape
    step_r = rows[1] - rows[0] if len(rows) > 1 else 1.0
    step_c = cols[1] - cols[0] if len(cols) > 1 else 1.0
    ri = (np.arange(H) - rows[0]) / step_r
    ci = (np.arange(W) - cols[0]) / step_c
    out = np.empty((H, W), np.float32)
    for a0 in range(0, H, 512):
        rr = np.repeat(ri[a0:a0 + 512, None], W, 1)
        cc = np.repeat(ci[None, :], rr.shape[0], 0)
        out[a0:a0 + 512] = ndimage.map_coordinates(a, [rr, cc], order=1, mode="nearest")
    return out


def fft_layers(field: np.ndarray, valid: np.ndarray, fc: dict, prefix: str) -> tuple[dict, dict]:
    """Native-grid descriptor layers ({name: array}) + the coarse products (for figures)."""
    w = window_spectra(field, valid, fc)
    shp = field.shape
    strike = trace_strike_from_wavevector(w["k_azimuth"])
    c2 = np.where(np.isfinite(strike), np.cos(np.radians(2 * strike)), np.nan) * w["anisotropy"]
    s2 = np.where(np.isfinite(strike), np.sin(np.radians(2 * strike)), np.nan) * w["anisotropy"]
    lay = {f"{prefix}_fft_anisotropy": w["anisotropy"], f"{prefix}_fft_strike_c2": c2, f"{prefix}_fft_strike_s2": s2,
           f"{prefix}_fft_dir_entropy": w["dir_entropy"], f"{prefix}_fft_short_long": w["short_long"],
           f"{prefix}_fft_radial_slope": w["radial_slope"], f"{prefix}_fft_log_energy": w["log_energy"]}
    v = np.asarray(valid, bool)
    nat = {k: np.where(v, to_native(a, w["rows"], w["cols"], shp), np.nan).astype(np.float32) for k, a in lay.items()}
    return nat, w


# =============================================================================== wavelets
def _padded_fft(field: np.ndarray, valid: np.ndarray):
    import scipy.fft as sfft

    from ..m8.potential_fields import fill

    f, m = fill(field)
    f = f - float(np.mean(f[m])) if m.any() else f
    H, W = f.shape
    ph, pw = H // 8, W // 8
    F = np.pad(f, ((ph, ph), (pw, pw)), mode="reflect").astype(np.float32)
    return sfft.fft2(F, workers=-1), (ph, pw, H, W), m & np.asarray(valid, bool)


def wavelet_layers(field: np.ndarray, valid: np.ndarray, wc: dict, prefix: str, px_km: float = PX_KM) -> dict:
    """Log-Gabor energy per scale, orientation / coherence / persistence / fine-coarse ratio (native grid)."""
    import scipy.fft as sfft

    from ..rs_features import prank

    F, (ph, pw, H, W), m = _padded_fft(field, valid)
    Hp, Wp = F.shape
    fr = np.fft.fftfreq(Hp, d=px_km)[:, None]
    fcl = np.fft.fftfreq(Wp, d=px_km)[None, :]
    kE, kN = fcl, -fr
    kmag = np.hypot(kE, kN)
    kmag[0, 0] = 1e-9
    kdir = np.arctan2(kE, kN)                                          # full-circle wavevector azimuth (rad)
    sig = np.log(float(wc["bandwidth"]))
    sa = np.radians(float(wc["angular_sigma_deg"]))
    n_o = int(wc["n_orient"])
    thetas = np.pi * np.arange(n_o) / n_o
    out, energies = {}, []
    for lam in wc["wavelengths_km"]:
        f0 = 1.0 / float(lam)
        radial = np.exp(-(np.log(kmag / f0) ** 2) / (2 * sig ** 2)).astype(np.float32)
        radial[0, 0] = 0.0
        E = np.zeros((H, W), np.float32)
        vc = np.zeros((H, W), np.complex64)
        for th in thetas:
            d = (kdir - th + np.pi) % (2 * np.pi) - np.pi              # one-sided: quadrature (analytic) response
            ang = np.exp(-(d ** 2) / (2 * sa ** 2)).astype(np.float32)
            r = sfft.ifft2(F * (radial * ang), workers=-1)[ph:ph + H, pw:pw + W]
            A = np.abs(r).astype(np.float32)
            E += A
            vc += A * np.complex64(np.exp(2j * th))
            del r
        coh = (np.abs(vc) / np.maximum(E, 1e-12)).astype(np.float32)
        strike = (np.degrees(0.5 * np.angle(vc)) + 90.0) % 180.0       # trace = wavevector + 90
        out[f"{prefix}_wav_energy@{lam:g}km"] = np.log1p(E / max(float(np.median(E[m])) if m.any() else 1.0, 1e-12))
        out[f"{prefix}_wav_coherence@{lam:g}km"] = coh
        out[f"{prefix}_wav_strike_c2@{lam:g}km"] = (np.cos(np.radians(2 * strike)) * coh).astype(np.float32)
        out[f"{prefix}_wav_strike_s2@{lam:g}km"] = (np.sin(np.radians(2 * strike)) * coh).astype(np.float32)
        energies.append(E)
    q = float(wc["persistence_q"])
    ranks = [prank(np.where(m, e, np.nan), m) for e in energies]
    out[f"{prefix}_wav_persistence"] = np.mean([np.nan_to_num(r) >= q for r in ranks], 0).astype(np.float32)
    out[f"{prefix}_wav_fine_coarse"] = np.log((energies[0] + 1e-6) / (energies[-1] + 1e-6)).astype(np.float32)
    out[f"{prefix}_wav_coherence_mean"] = np.mean([out[f"{prefix}_wav_coherence@{lam:g}km"]
                                                   for lam in wc["wavelengths_km"]], 0).astype(np.float32)
    return {k: np.where(m, v, np.nan).astype(np.float32) for k, v in out.items()}


# =============================================================================== continuation
def continuation_layers(field: np.ndarray, valid: np.ndarray, cc: dict, prefix: str) -> dict:
    from ..m8.potential_fields import fill, upward

    f, m = fill(field)
    m &= np.asarray(valid, bool)
    up = {float(h): upward(f, float(h)) for h in cc["heights_km"]}
    out = {}
    hmin = min(up)
    out[f"{prefix}_resid_{hmin:g}km"] = (f - up[hmin]).astype(np.float32)
    for h1, h2 in cc.get("bands") or []:
        a = up.get(float(h1), upward(f, float(h1)))
        b = up.get(float(h2), upward(f, float(h2)))
        out[f"{prefix}_band_{float(h1):g}_{float(h2):g}km"] = (a - b).astype(np.float32)
    return {k: np.where(m, v, np.nan).astype(np.float32) for k, v in out.items()}


# =============================================================================== orchestration
def build(m9) -> tuple[pd.DataFrame, dict]:
    """Compute (once) every spectral layer into the M9 layer cache; returns (registry, figure products)."""
    from gems_eda.enhance import standardize_signal

    from .features import file_fp

    sc = m9.cfg["spectral"]
    data = m9.ctx.data
    valid = np.asarray(data.valid, bool)
    rows, figs = [], {}

    def put(name, a, kind, field, note):
        full = f"m9:spec:{name}"
        if not m9.layers.has(full) or m9.cfg.get("overwrite_existing"):
            a = np.where(valid, a, np.nan).astype(np.float32)
            z = a if any(t in name for t in BOUNDED) else standardize_signal(a)[0]
            m9.layers.put(full, z)
        rows.append({"name": full, "family": "spectral", "kind": kind, "field": field, "note": note,
                     "fingerprint": file_fp(m9.layers.path(full))})

    for fam, band in (sc["fields"] or {}).items():
        if band not in data.band_names:
            m9.say(f"s03 {fam}: band {band} absent -> skipped")
            continue
        a = data.read_band(band)
        nat, w = fft_layers(a, valid, sc["fft"], fam)
        for k, v in nat.items():
            put(k, v, "fft", band, f"windowed FFT ({sc['fft']['window_px']} px / stride {sc['fft']['stride_px']})")
        figs[fam] = {"sectors": w["sectors"], "rows": w["rows"], "cols": w["cols"], "anisotropy": w["anisotropy"],
                     "k_azimuth": w["k_azimuth"]}
        del nat
        if sc["stft"].get("enabled"):
            st = {**sc["fft"], "window_px": sc["stft"]["window_px"], "stride_px": sc["stft"]["stride_px"]}
            nat, _ = fft_layers(a, valid, st, f"{fam}_stft")
            for k, v in nat.items():
                if k.endswith(("anisotropy", "strike_c2", "strike_s2")):
                    put(k, v, "stft", band, "short-time FFT ablation")
            del nat
        for k, v in wavelet_layers(a, valid, sc["wavelet"], fam).items():
            put(k, v, "wavelet", band, "log-Gabor steerable wavelet")
        if sc["continuation"].get("enabled") and fam in ("magnetic", "gravity"):
            for k, v in continuation_layers(a, valid, sc["continuation"], fam).items():
                put(k, v, "continuation", band, "upward continuation (stable; no downward continuation)")
        m9.layers.release()
    return pd.DataFrame(rows), figs


def compare_existing(m9, names: list[str]) -> pd.DataFrame:
    """Label-free Spearman between each new spectral layer and the existing Gabor / ridge / DoG / coherence features."""
    from .features import available, correlation_matrix

    ref = [n for n in m9.cfg["spectral"]["compare_with"] if available(m9, n)]
    if not ref or not names:
        return pd.DataFrame()
    C = correlation_matrix(m9, list(names) + ref)
    if not len(C):
        return C
    C = C[C.a.isin(names) & C.b.isin(ref)].copy()
    C["abs_spearman"] = C.spearman.abs()
    return C


def select(m9, reg: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Greedy label-free redundancy pruning of the spectral layers against each other and the selected inputs."""
    from .features import _ranks, _sample_idx, selected_registry

    if not len(reg):
        return reg
    base = selected_registry(m9)
    base = list(base[base.selected.astype(bool)].name) if len(base) else []
    idx = _sample_idx(np.asarray(m9.ctx.data.valid, bool), 80_000, int(m9.cfg["seed"]) + 3)
    names = list(reg.name)
    R = _ranks(m9, names + base, idx)
    C = np.abs(np.corrcoef(R))
    thr = float(m9.cfg["features"]["redundancy"]["max_abs_spearman"])
    nb = len(names)
    kept, sel, why = [], [], []
    for i, n in enumerate(names):
        others = kept + list(range(nb, nb + len(base)))
        mx = float(C[i, others].max()) if others else 0.0
        ok = mx <= thr and len(kept) < cap
        sel.append(ok)
        why.append("kept" if ok else ("cap reached" if mx <= thr else f"max |rho| {mx:.3f}"))
        if ok:
            kept.append(i)
    return reg.assign(selected=sel, reason=why)
