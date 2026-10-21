"""Phase-4 remote-sensing / geomorphometric / structural feature bank (native 100 m grid, label-free).

The catalogue lives in ``configs/phase4/rs_features.yaml``; this module turns it into ``RSFeatureSpec``s and
computes them. Features are cached by ``phase4_context.Phase4FeatureStore`` in
``outputs/phase4/cache/features/`` (the Phase-2 cache is never written to).

Conventions (same as gems_eda.enhance)
-------------------------------------
* Arrays are (row, col), rows increase southward; gradients are (east, north).
* Strikes are azimuths clockwise from north in [0, 180); orientation statistics are axial (2 theta).
* A Gaussian "scale" s km means sigma = s / 2 km, unless a transform documents otherwise
  (tpi: disk radius; tri: neighbour lag; range: window width; geomorphon: look-up distance).
* NaNs are nearest-filled before filtering and restored afterwards (no smearing of nodata).

Terrain caveat: the only elevation band is detrended (``det_elev``). Slope, curvature, TPI and hillshade
here describe the detrended surface; every terrain spec carries ``basis="detrended_elevation"``.

Nothing here reads labels. Percentile ranks use a fixed random sample of valid pixels (label-free).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from gems_eda.config import PRIMARY_FIELD
from gems_eda.enhance import fill_nan, nan_gaussian

from .features import FeatureSpec

GROUPS_RS = ("RS_T", "RS_P", "RS_X")
SEED = 2026


@dataclass(frozen=True)
class RSFeatureSpec(FeatureSpec):
    family: str = ""
    basis: str = ""
    output: str = ""
    params: tuple = ()                  # ((key, value), ...) — hashable copy of the YAML params
    standardize: bool = True            # False for bounded / count features (geomorphon, persistence count)

    def p(self, key, default=None):
        return dict(self.params).get(key, default)


# ---------------------------------------------------------------------------
# configuration -> specs
# ---------------------------------------------------------------------------
def default_config_path(root: Path | None = None) -> Path:
    from gems_eda.config import find_project_root

    p = find_project_root(root) / "configs" / "phase4" / "rs_features.yaml"
    if not p.exists():                             # Colab: configs live in the code checkout
        from .paths import config_path

        p = config_path("phase4", "rs_features.yaml")
    return p


def load_rs_config(path: Path | None = None) -> dict:
    import yaml

    p = Path(path) if path else default_config_path()
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _freeze(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def rs_name(modality: str, transform: str, output: str, scale_km: float, tag: str = "") -> str:
    out = f"_{output}" if output else ""
    tg = f"_{tag}" if tag else ""
    return f"rs:{modality}:{transform}{out}{tg}@{scale_km:g}km"


def _band(x: str) -> str:
    """Modality name -> primary band (PRIMARY_FIELD), raw band names pass through."""
    return PRIMARY_FIELD.get(x, x)


def _rs_support(transform: str, s: float, params: dict) -> float:
    """~radius (km) of input influence, for the buffer audit (features.leakage_audit)."""
    if transform == "tpi":
        return s
    if transform in ("tri", "range"):
        return s
    if transform == "geomorphon":
        return s * 1.5
    if transform == "gabor_bank":
        return 2 * (2 * s)
    if transform in ("persistence", "xmod_persistence"):
        return 3 * max(params.get("scales_km", [s])) / 2
    if transform == "edge_contrast":
        return 3 * params.get("regional_km", 10.0) / 2
    if transform == "normal_contrast":
        return max(3 * params.get("orient_scale_km", 2.0) * 3 / 2, params.get("offset_km", 0.3) + 3 * s / 2)
    if transform in ("orient_agree", "orient_conf", "coherence"):
        return 3 * s * 3 / 2
    if transform in ("analytic_signal", "tdx"):
        return 6 * s / 2
    if transform == "dog":
        return 3 * s * params.get("ratio", 3.0) / 2
    return 3 * s / 2


def build_rs_specs(band_names: list[str], config: dict | None = None) -> list[RSFeatureSpec]:
    cfg = config or load_rs_config()
    specs: list[RSFeatureSpec] = []
    for block_name, block in cfg["blocks"].items():
        modality = block.get("modality", block_name)
        source = block.get("source", "")
        if source and source not in band_names:
            print(f"[rs_features] block {block_name}: band {source} missing -> skipped")
            continue
        group = block.get("group", "RS_X")
        for f in block["features"]:
            if not f.get("enabled", True):
                continue
            params = dict(f.get("params", {}) or {})
            src = source
            if modality == "xmod":
                mods = params.get("sources") or [params[k] for k in ("a", "b", "src", "orient_src") if k in params]
                src = "+".join(dict.fromkeys(_band(m) for m in mods))
            for s in f.get("scales_km", [0.0]):
                for out in (f.get("outputs") or [""]):
                    specs.append(RSFeatureSpec(
                        name=rs_name(modality, f["transform"], out, float(s), f.get("tag", "")),
                        groups=(group,), modality=modality if modality != "xmod" else "fused", source=src,
                        transform=f["transform"], scale_km=float(s), kind="local", origin="native", units="robust-z",
                        support_km=_rs_support(f["transform"], float(s), params), note=f.get("note", ""),
                        family=f.get("family", f["transform"]), basis=block.get("basis", ""), output=str(out),
                        params=_freeze(params), standardize=bool(f.get("standardize", True))))
    names = [s.name for s in specs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate RS feature names: {sorted(dup)}")
    return specs


def spec_table(specs: list[RSFeatureSpec]) -> pd.DataFrame:
    rows = []
    for s in specs:
        d = asdict(s)
        d["groups"] = ",".join(s.groups)
        d["params"] = str(dict(s.params))
        rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# shared helpers (memoised per store build)
# ---------------------------------------------------------------------------
_MEMO_MAX = 6


def _memo(store, key, make):
    cache = store.__dict__.setdefault("_rs_tmp", OrderedDict())
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    val = make()
    cache[key] = val
    while len(cache) > _MEMO_MAX:
        cache.popitem(last=False)
    return val


def clear_memo(store) -> None:
    store.__dict__.pop("_rs_tmp", None)


def _src(store, band: str) -> np.ndarray:
    return store._source(_band(band))


def _restore(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    a[~m] = np.nan
    return a


def _px_km(store) -> float:
    return store.data.grid.pixel_km


def prank(a: np.ndarray, valid: np.ndarray, n: int = 400_000, seed: int = SEED) -> np.ndarray:
    """Empirical percentile rank in [0, 1] against a fixed random sample of valid pixels (label-free)."""
    m = valid & np.isfinite(a)
    v = a[m]
    if v.size > n:
        v = np.random.default_rng(seed).choice(v, n, replace=False)
    s = np.sort(v)
    out = np.full(a.shape, np.nan, np.float32)
    if s.size:
        out[m] = (np.searchsorted(s, a[m], side="right") / s.size).astype(np.float32)
    return out


def _disk_mean(a: np.ndarray, radius_px: int) -> np.ndarray:
    """NaN-aware mean over a disk (centre excluded) by normalised FFT convolution."""
    from scipy.signal import fftconvolve

    m = np.isfinite(a)
    f = np.where(m, a, 0.0).astype(np.float64)
    yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    k = (yy * yy + xx * xx <= radius_px * radius_px).astype(np.float64)
    k[radius_px, radius_px] = 0.0
    num = fftconvolve(f, k, mode="same")
    den = fftconvolve(m.astype(np.float64), k, mode="same")
    return (num / np.maximum(den, 1e-6)).astype(np.float32)


def _derivs(store, band: str, s_km: float):
    """Gaussian derivatives of a field in physical units (field / m): fx, fy, fxx, fyy, fxy (east/north axes)."""
    def make():
        g = store.data.grid
        f = fill_nan(_src(store, band))
        sg = store.enh.sigma_px(s_km)
        gx, gy = g.pixel_km_x * 1000.0, g.pixel_km_y * 1000.0
        gf = ndimage.gaussian_filter
        d_r = gf(f, sg, order=(1, 0), mode="nearest")
        d_c = gf(f, sg, order=(0, 1), mode="nearest")
        d_rr = gf(f, sg, order=(2, 0), mode="nearest")
        d_cc = gf(f, sg, order=(0, 2), mode="nearest")
        d_rc = gf(f, sg, order=(1, 1), mode="nearest")
        out = (d_c / gx, -d_r / gy, d_cc / gx ** 2, d_rr / gy ** 2, -d_rc / (gx * gy))
        return tuple(np.asarray(x, np.float32) for x in out)
    return _memo(store, ("derivs", _band(band), s_km), make)


def _structure(store, band: str, s_km: float):
    """(strike deg, coherence) of the structure tensor (gems_eda convention)."""
    def make():
        st, coh, _ = store.enh.structure_tensor(_src(store, band), s_km)
        return st, coh
    return _memo(store, ("st", _band(band), s_km), make)


def _pf(store, band: str, s_km: float):
    """(THDR, VDR) of the Gaussian-smoothed potential field (same recipe as Enhancer.tilt)."""
    def make():
        e = store.enh
        sm = nan_gaussian(_src(store, band), e.sigma_px(s_km))
        vdr = e.vertical_derivative(sm)
        thdr = e.thdr(sm, 0.0)
        return thdr.astype(np.float32), vdr.astype(np.float32)
    return _memo(store, ("pf", _band(band), s_km), make)


def _ridge_rank(store, band: str, s_km: float, mode: str) -> np.ndarray:
    def make():
        e, a = store.enh, _src(store, band)
        r = e.ridge(e.thdr(a, s_km), s_km, "bright")[0] if mode == "thdr" else e.ridge(a, s_km, "both")[0]
        return prank(r, store.data.valid)
    return _memo(store, ("rrank", _band(band), s_km, mode), make)


def _gabor_bank(store, band: str, wavelength_km: float, n_orient: int = 8, sigma_on_f: float = 0.55):
    """Oriented log-Gabor quadrature energies (n_orient, H, W) and their per-bin strikes (Enhancer.log_gabor recipe)."""
    def make():
        g = store.data.grid
        a = _src(store, band)
        m = np.isfinite(a)
        f = fill_nan(a)
        f = f - f[m].mean() if m.any() else f
        h, w = f.shape
        ph, pw = max(h // 8, 1), max(w // 8, 1)
        F = np.fft.fft2(np.pad(f, ((ph, ph), (pw, pw)), mode="reflect").astype(np.float32))
        fy = np.fft.fftfreq(F.shape[0], d=g.pixel_km_y)
        fx = np.fft.fftfreq(F.shape[1], d=g.pixel_km_x)
        FN, FX = np.meshgrid(-fy, fx, indexing="ij")
        radius = np.hypot(FN, FX)
        radius[0, 0] = 1.0
        radial = np.exp(-np.log(radius * wavelength_km) ** 2 / (2 * np.log(sigma_on_f) ** 2))
        radial[0, 0] = 0.0
        theta = np.arctan2(FN, FX)
        sigma_theta = np.pi / n_orient / 1.2
        E = np.zeros((n_orient, h, w), np.float32)
        strikes = np.zeros(n_orient)
        for k in range(n_orient):
            ang = k * np.pi / n_orient
            dtheta = np.abs(np.arctan2(np.sin(theta - ang), np.cos(theta - ang)))
            spread = np.exp(-dtheta ** 2 / (2 * sigma_theta ** 2))
            E[k] = np.abs(np.fft.ifft2(F * (radial * spread).astype(np.complex64)))[ph:ph + h, pw:pw + w]
            strikes[k] = (180.0 - np.degrees(ang)) % 180
        return E, strikes, m
    return _memo(store, ("gabor", _band(band), wavelength_km, n_orient), make)


# ---------------------------------------------------------------------------
# terrain / geomorphometry
# ---------------------------------------------------------------------------
def t_tpi(store, spec):
    a = _src(store, spec.source)
    r = max(int(round(spec.scale_km / _px_km(store))), 1)
    return _restore(a - _disk_mean(a, r), np.isfinite(a))


def t_tri(store, spec):
    a = _src(store, spec.source)
    m = np.isfinite(a)
    f = fill_nan(a)
    lag = max(int(round(spec.scale_km / _px_km(store))), 1)
    if lag > 1:
        f = ndimage.gaussian_filter(f, lag / 2.0, mode="nearest")
    H, W = f.shape
    P = np.pad(f, lag, mode="edge")
    acc = np.zeros_like(f)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            acc += np.abs(P[lag + dr * lag:lag + dr * lag + H, lag + dc * lag:lag + dc * lag + W] - f)
    return _restore(acc / 8.0, m)


def t_range(store, spec):
    a = _src(store, spec.source)
    f = fill_nan(a)
    size = max(int(round(spec.scale_km / _px_km(store))) | 1, 3)
    out = ndimage.maximum_filter(f, size, mode="nearest") - ndimage.minimum_filter(f, size, mode="nearest")
    return _restore(out, np.isfinite(a))


_FLAT_EPS = np.tan(np.radians(0.1)) ** 2


def t_curv_profile(store, spec):
    fx, fy, fxx, fyy, fxy = _derivs(store, spec.source, spec.scale_km)
    p = fx * fx + fy * fy
    num = fxx * fx * fx + 2 * fxy * fx * fy + fyy * fy * fy
    return _restore(num / ((p + _FLAT_EPS) * (1 + p) ** 1.5), np.isfinite(_src(store, spec.source)))


def t_curv_plan(store, spec):
    """Tangential curvature (plan curvature x sin slope): finite on flats unlike the plan curvature itself."""
    fx, fy, fxx, fyy, fxy = _derivs(store, spec.source, spec.scale_km)
    p = fx * fx + fy * fy
    num = fxx * fy * fy - 2 * fxy * fx * fy + fyy * fx * fx
    return _restore(num / ((p + _FLAT_EPS) * np.sqrt(1 + p)), np.isfinite(_src(store, spec.source)))


def t_laplacian(store, spec):
    return store.enh.log_wavelet(_src(store, spec.source), spec.scale_km)


def _hillshade(fx, fy, azimuth_deg: float, altitude_deg: float, z: float) -> np.ndarray:
    slope = np.arctan(z * np.hypot(fx, fy))
    aspect = np.arctan2(-fx, -fy)                       # downslope azimuth, clockwise from north
    zen = np.radians(90.0 - altitude_deg)
    return np.cos(zen) * np.cos(slope) + np.sin(zen) * np.sin(slope) * np.cos(np.radians(azimuth_deg) - aspect)


def t_hillshade(store, spec):
    fx, fy, *_ = _derivs(store, spec.source, spec.scale_km)
    alt, z = float(spec.p("altitude_deg", 30.0)), float(spec.p("z_factor", 1.0))
    if spec.output.startswith("perp_s"):
        az = [(float(spec.output[len("perp_s"):]) + 90.0) % 360]
    elif spec.output == "mdow":
        az = [225.0, 270.0, 315.0, 360.0]
    else:
        raise KeyError(spec.output)
    hs = np.mean([_hillshade(fx, fy, a, alt, z) for a in az], 0)
    return _restore(hs, np.isfinite(_src(store, spec.source)))


def t_slope_break(store, spec):
    e, a = store.enh, _src(store, spec.source)
    return e.thdr(e.thdr(a, spec.scale_km), spec.scale_km)


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]   # N, NE, E, SE, S, SW, W, NW


def geomorphon_signs(f: np.ndarray, lag_px: int, flat_deg: float, pixel_m: float) -> np.ndarray:
    """Ternary (+1 higher, -1 lower, 0 flat) elevation difference to the 8 look-up points at ``lag_px``."""
    H, W = f.shape
    P = np.pad(f, lag_px, mode="edge")
    t = np.tan(np.radians(flat_deg))
    signs = np.zeros((8, H, W), np.int8)
    for i, (dr, dc) in enumerate(_DIRS8):
        d = P[lag_px + dr * lag_px:lag_px + dr * lag_px + H, lag_px + dc * lag_px:lag_px + dc * lag_px + W] - f
        thr = t * lag_px * pixel_m * np.hypot(dr, dc)
        signs[i] = (d > thr).astype(np.int8) - (d < -thr).astype(np.int8)
    return signs


def geomorphon_outputs(signs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(convexity = (n_lower - n_higher) / 8, linear = +1 ridge / -1 valley / 0) from the ternary pattern.

    Linear form along axis (i, i+4): the 6 off-axis look-ups share one non-zero sign and the two axis
    look-ups do not (ridge: 6 lower, axis not lower; valley: 6 higher, axis not higher).
    """
    n_plus = (signs > 0).sum(0)
    n_minus = (signs < 0).sum(0)
    convexity = (n_minus - n_plus).astype(np.float32) / 8.0
    ridge = np.zeros(signs.shape[1:], bool)
    valley = np.zeros_like(ridge)
    for i in range(4):
        off = [j for j in range(8) if j not in (i, i + 4)]
        so = signs[off]
        ridge |= (so < 0).all(0) & (signs[i] >= 0) & (signs[i + 4] >= 0)
        valley |= (so > 0).all(0) & (signs[i] <= 0) & (signs[i + 4] <= 0)
    linear = ridge.astype(np.float32) - valley.astype(np.float32)
    return convexity, linear


def t_geomorphon(store, spec):
    a = _src(store, spec.source)
    lag = max(int(round(spec.scale_km / _px_km(store))), 1)

    def make():
        return geomorphon_outputs(geomorphon_signs(fill_nan(a), lag, float(spec.p("flat_deg", 1.0)),
                                                   _px_km(store) * 1000.0))
    conv, lin = _memo(store, ("geomorphon", spec.source, lag, spec.p("flat_deg", 1.0)), make)
    return _restore(conv if spec.output == "convexity" else lin, np.isfinite(a))


def t_coherence(store, spec):
    return _structure(store, spec.source, spec.scale_km)[1]


def t_persistence(store, spec):
    scales = [float(s) for s in spec.p("scales_km", (0.5, 1.0, 2.0, 4.0))]
    q, mode = float(spec.p("q", 0.9)), spec.p("mode", "thdr")
    U = np.stack([_ridge_rank(store, spec.source, s, mode) for s in scales])
    if spec.output == "frac":
        out = (U >= q).mean(0)
    elif spec.output == "min":
        out = U.min(0)
    else:
        raise KeyError(spec.output)
    return _restore(out, np.isfinite(_src(store, spec.source)))


# ---------------------------------------------------------------------------
# potential fields
# ---------------------------------------------------------------------------
def t_analytic_signal(store, spec):
    thdr, vdr = _pf(store, spec.source, spec.scale_km)
    return np.sqrt(thdr * thdr + vdr * vdr)


def t_tdx(store, spec):
    thdr, vdr = _pf(store, spec.source, spec.scale_km)
    return np.degrees(np.arctan2(thdr, np.abs(vdr))).astype(np.float32)


def t_thdr_ridge(store, spec):
    e, a = store.enh, _src(store, spec.source)
    return e.ridge(e.thdr(a, spec.scale_km), spec.scale_km, "bright")[0]


def t_hessian_ridge(store, spec):
    return store.enh.ridge(_src(store, spec.source), spec.scale_km, "both")[0]


def t_dog(store, spec):
    e, a = store.enh, _src(store, spec.source)
    r = float(spec.p("ratio", 3.0))
    return nan_gaussian(a, e.sigma_px(spec.scale_km)) - nan_gaussian(a, e.sigma_px(spec.scale_km * r))


def t_gabor_bank(store, spec):
    n = int(spec.p("n_orient", 8))
    E, strikes, m = _gabor_bank(store, spec.source, 2.0 * spec.scale_km, n)
    if spec.output == "max":
        return _restore(E.max(0), m)
    sg = store.enh.sigma_px(2.0 * spec.scale_km)                 # smooth the energies over ~ one wavelength

    def make():
        Es = np.stack([ndimage.gaussian_filter(E[k], sg, mode="nearest") for k in range(n)])
        return Es, Es.sum(0) + 1e-12
    Es, tot = _memo(store, ("gabor_s", spec.source, spec.scale_km, n), make)
    if spec.output == "entropy":
        P = Es / tot
        H = -(P * np.log(np.clip(P, 1e-12, None))).sum(0) / np.log(n)
        return _restore(H, m)
    if spec.output.startswith("fam_s"):
        st = float(spec.output[len("fam_s"):])
        k = int(np.argmin(np.abs(((strikes - st) + 90) % 180 - 90)))
        return _restore(Es[k] / tot, m)
    raise KeyError(spec.output)


# ---------------------------------------------------------------------------
# cross-modal evidence (axial orientation statistics; agreement is never forced)
# ---------------------------------------------------------------------------
def t_orient_agree(store, spec):
    sa, ca = _structure(store, spec.p("a"), spec.scale_km)
    sb, cb = _structure(store, spec.p("b"), spec.scale_km)
    return (ca * cb * np.cos(2 * np.radians(sa - sb))).astype(np.float32)


def t_edge_agree(store, spec):
    e, v = store.enh, store.data.valid
    ua = _memo(store, ("thdr_rank", _band(spec.p("a")), spec.scale_km),
               lambda: prank(e.thdr(_src(store, spec.p("a")), spec.scale_km), v))
    ub = _memo(store, ("thdr_rank", _band(spec.p("b")), spec.scale_km),
               lambda: prank(e.thdr(_src(store, spec.p("b")), spec.scale_km), v))
    if spec.output == "min":
        return np.fmin(ua, ub)
    if spec.output == "max":
        return np.fmax(ua, ub)
    raise KeyError(spec.output)


def t_orient_conf(store, spec):
    acc = 0.0
    srcs = list(spec.p("sources"))
    for s in srcs:
        st, c = _structure(store, s, spec.scale_km)
        acc = acc + np.nan_to_num(c) * np.exp(2j * np.radians(np.nan_to_num(st)))
    out = np.abs(acc) / len(srcs)
    return _restore(out, store.data.valid)


def t_edge_contrast(store, spec):
    e = store.enh
    a = _src(store, spec.p("src"))
    loc = e.thdr(a, spec.scale_km)
    reg = nan_gaussian(loc, e.sigma_px(float(spec.p("regional_km", 10.0))))
    eps = float(np.nanquantile(loc[store.data.valid], 0.01)) + 1e-12
    return np.log((loc + eps) / (reg + eps)).astype(np.float32)


def t_normal_contrast(store, spec):
    """Field step across vs along the local fabric strike (strike from a label-free structure tensor)."""
    e = store.enh
    a = _src(store, spec.p("src"))
    m = np.isfinite(a)
    st, _ = _structure(store, spec.p("orient_src"), float(spec.p("orient_scale_km", 2.0)))
    th = np.radians(np.nan_to_num(st))
    S = fill_nan(nan_gaussian(a, e.sigma_px(spec.scale_km)))
    d = float(spec.p("offset_km", 0.3)) / _px_km(store)
    rr, cc = np.indices(S.shape, dtype=np.float32)
    nr, nc = (np.sin(th) * d).astype(np.float32), (np.cos(th) * d).astype(np.float32)      # fault-normal (row, col)
    tr, tc = (-np.cos(th) * d).astype(np.float32), (np.sin(th) * d).astype(np.float32)     # along strike (row, col)

    def at(dr, dc):
        return ndimage.map_coordinates(S, [rr + dr, cc + dc], order=1, mode="nearest")
    step_n = np.abs(at(nr, nc) - at(-nr, -nc))
    if spec.output == "step":
        return _restore(step_n, m)
    step_t = np.abs(at(tr, tc) - at(-tr, -tc))
    return _restore((step_n - step_t) / (step_n + step_t + 1e-9), m)


def t_xmod_persistence(store, spec):
    scales = [float(s) for s in spec.p("scales_km")]
    q, min_frac = float(spec.p("q", 0.9)), float(spec.p("min_frac", 0.5))
    count = np.zeros(store.data.shape, np.float32)
    for s in spec.p("sources"):
        mode = "ridge" if s == "terrain" else "thdr"
        U = np.stack([_ridge_rank(store, s, sc, mode) for sc in scales])
        count += ((U >= q).mean(0) >= min_frac).astype(np.float32)
    return _restore(count, store.data.valid)


TRANSFORMS = {
    "tpi": t_tpi, "tri": t_tri, "range": t_range, "curv_profile": t_curv_profile, "curv_plan": t_curv_plan,
    "laplacian": t_laplacian, "hillshade": t_hillshade, "slope_break": t_slope_break, "geomorphon": t_geomorphon,
    "coherence": t_coherence, "persistence": t_persistence, "analytic_signal": t_analytic_signal, "tdx": t_tdx,
    "thdr_ridge": t_thdr_ridge, "hessian_ridge": t_hessian_ridge, "dog": t_dog, "gabor_bank": t_gabor_bank,
    "orient_agree": t_orient_agree, "edge_agree": t_edge_agree, "orient_conf": t_orient_conf,
    "edge_contrast": t_edge_contrast, "normal_contrast": t_normal_contrast, "xmod_persistence": t_xmod_persistence,
}


def compute(store, spec: RSFeatureSpec) -> np.ndarray:
    return np.asarray(TRANSFORMS[spec.transform](store, spec), np.float32)


# ---------------------------------------------------------------------------
# redundancy audit, groups, importance
# ---------------------------------------------------------------------------
def redundancy_audit(store, rs_names: list[str], ref_names: list[str], n: int = 150_000,
                     threshold: float = 0.95, seed: int = SEED) -> pd.DataFrame:
    """Max |Spearman| of every RS feature against the existing features (+ earlier RS features), label-free sample.

    Flags near-duplicates (e.g. analytic signal vs the provided tmi_hg / tmi_vg bands). Earlier RS features
    in ``rs_names`` order also count as references so two RS features cannot both survive as duplicates.
    """
    from scipy.stats import rankdata

    v = store.data.valid
    idx = np.flatnonzero(v.reshape(-1))
    idx = np.sort(np.random.default_rng(seed).choice(idx, min(n, idx.size), replace=False))
    refs = [r for r in ref_names if store.has(r)]
    names = [r for r in rs_names if store.has(r)]

    def ranks(name):
        x = np.asarray(store.get(name)).reshape(-1)[idx].astype(np.float64)
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        r = rankdata(x)
        return (r - r.mean()) / (r.std() + 1e-12)
    R = {r: ranks(r) for r in refs}
    rows = []
    for nm in names:
        x = ranks(nm)
        best, partner = 0.0, ""
        for r, y in R.items():
            rho = float(np.mean(x * y))
            if abs(rho) > abs(best):
                best, partner = rho, r
        red = abs(best) >= threshold
        rows.append({"name": nm, "max_abs_spearman": abs(best), "signed_rho": best, "partner": partner,
                     "redundant": red, "threshold": threshold, "n_sample": len(idx)})
        if not red:
            R[nm] = x                    # kept RS features become references for later ones
    return pd.DataFrame(rows)


def rs_group_names(specs: list[RSFeatureSpec], group: str, available: set[str] | None = None,
                   exclude: set[str] | None = None) -> list[str]:
    return [s.name for s in specs if group in s.groups and (available is None or s.name in available)
            and not (exclude and s.name in exclude)]


def family_of(store, name: str) -> str:
    s = store.specs.get(name)
    if isinstance(s, RSFeatureSpec):
        return f"{s.groups[0]}:{s.family}"
    if s is None:
        return "unknown"
    return f"base:{s.modality}"


def group_importance(importance: pd.DataFrame, store) -> pd.DataFrame:
    """Gain share per feature family and fold (descriptive only: gain is not incremental predictive value)."""
    imp = importance.copy()
    imp["family"] = [family_of(store, f) for f in imp.feature]
    g = imp.groupby(["fold", "family"]).gain.sum().reset_index()
    g["gain_share"] = g.gain / g.groupby("fold").gain.transform("sum")
    out = g.pivot_table(index="family", columns="fold", values="gain_share").add_prefix("share_fold").reset_index()
    out["share_mean"] = out[[c for c in out if c.startswith("share_fold")]].mean(1)
    return out.sort_values("share_mean", ascending=False)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def feature_atlas(store, names: list[str], windows: list[tuple[int, int, int]], labels: np.ndarray | None = None,
                  ncols: int = 6):
    """Small-multiple atlas: rows = windows (r0, c0, size_px), columns = features; labels as thin dark overlay."""
    import matplotlib.pyplot as plt

    names = [n for n in names if store.has(n)]
    nrow, ncol = len(windows), len(names)
    fig, axs = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 1.9 * nrow + 0.4), squeeze=False, constrained_layout=True)
    for i, (r0, c0, s) in enumerate(windows):
        for j, n in enumerate(names):
            ax = axs[i, j]
            a = np.asarray(store.get(n)[r0:r0 + s, c0:c0 + s], np.float32)
            lo, hi = np.nanpercentile(a, [2, 98]) if np.isfinite(a).any() else (0, 1)
            ax.imshow(a, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            if labels is not None:
                lab = labels[r0:r0 + s, c0:c0 + s]
                ax.contour(lab.astype(float), levels=[0.5], colors="#eb6834", linewidths=0.5)
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(n.replace("rs:", "").replace("@", "\n@"), fontsize=6.5)
    fig.suptitle("Feature transformation atlas (orange = mapped faults; contrast 2-98 %)", fontsize=9, x=0.01, ha="left")
    return fig


def orientation_rose(store, seg_tab: pd.DataFrame, labels: np.ndarray, scale_km: float = 2.0,
                     sources=("terrain", "magnetic", "gravity"), n_bins: int = 18, sample: int = 300_000):
    """Coherence-weighted strike histograms per modality (valid pixels vs label pixels) vs mapped segment strikes."""
    import matplotlib.pyplot as plt

    v = store.data.valid
    rng = np.random.default_rng(SEED)
    edges = np.linspace(0, 180, n_bins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    fig, axs = plt.subplots(1, len(sources) + 1, figsize=(3.2 * (len(sources) + 1), 3.0), constrained_layout=True)
    seg = seg_tab.dropna(subset=["azimuth"])
    h, _ = np.histogram(seg.azimuth % 180, edges, weights=seg.length_km.fillna(1.0))
    axs[0].bar(ctr, h / h.sum(), width=180 / n_bins * 0.9, color="#8a8985")
    axs[0].set_title("mapped segments (km-weighted)", fontsize=9, loc="left")
    for ax, s in zip(axs[1:], sources):
        st, coh = store.enh.structure_tensor(_src(store, s), scale_km)[:2]
        for mask, col, lab in ((v, "#2a78d6", "all valid"), (v & labels, "#eb6834", "on mapped faults")):
            idx = np.flatnonzero(mask.reshape(-1))
            idx = rng.choice(idx, min(sample, idx.size), replace=False)
            a, w = st.reshape(-1)[idx], coh.reshape(-1)[idx]
            ok = np.isfinite(a) & np.isfinite(w)
            hh, _ = np.histogram(a[ok], edges, weights=w[ok])
            ax.step(ctr, hh / max(hh.sum(), 1e-12), where="mid", color=col, label=lab, linewidth=1.4)
        ax.set_title(f"{s} fabric strike @{scale_km:g} km", fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False)
    for ax in axs:
        ax.set_xlabel("strike (deg)", fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.6)
    return fig
