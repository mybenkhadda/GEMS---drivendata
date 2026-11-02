"""Neighbourhood (texture) context for classical pixel models (label-free, masked, vectorised).

Every statistic is computed over the FINITE pixels of a square window only (box filters of the value, its square
and the finite mask, ``mode='constant'``): an invalid pixel is never read as a physical zero. A statistic is NaN
where the window holds fewer than ``min_support`` finite pixels or where the centre pixel is missing.

Per source band ``<b>`` and window ``w<k>`` (odd, pixels)::

    std, range, min, max, mean     masked box statistics (min / max via +/-inf substitution)
    q10 / q50 / q90                masked quantiles (sliding window; small windows only - cost grows as k^2)
    ent                            normalised Shannon entropy of the value quantised into ``levels`` bins whose
                                   edges are FOOTPRINT quantiles of the band (label-free; fixed for every tile)
    edged                          density of thin gradient-maximum edges (sigma 100 m)
    gvar                           gradient variability = std / mean of |grad| in the window
    aniso                          box-integrated structure-tensor coherence (local anisotropy)
    oh<j>                          orientation histogram: share of gradient energy in axial bin j of n
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .feature_physics import field_at_scale, gradient_axis, nms_edges


def _box(a: np.ndarray, size: int) -> np.ndarray:
    return ndimage.uniform_filter(a, int(size), mode="constant", cval=0.0)


def local_statistics(a: np.ndarray, size: int, stats=("std", "range"), min_support: float = 0.3) -> dict:
    """Masked window statistics of ``a`` (dict stat -> array)."""
    a = np.asarray(a, np.float64)
    m = np.isfinite(a)
    a0 = np.where(m, a, 0.0)
    n = _box(m.astype(np.float64), size)
    ok = (n >= min_support) & m
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        if {"mean", "std"} & set(stats):
            mean = _box(a0, size) / np.maximum(n, 1e-12)
            if "mean" in stats:
                out["mean"] = mean
            if "std" in stats:
                var = _box(a0 * a0, size) / np.maximum(n, 1e-12) - mean * mean
                out["std"] = np.sqrt(np.maximum(var, 0.0))
        if {"min", "max", "range"} & set(stats):
            mn = ndimage.minimum_filter(np.where(m, a, np.inf), int(size), mode="constant", cval=np.inf)
            mx = ndimage.maximum_filter(np.where(m, a, -np.inf), int(size), mode="constant", cval=-np.inf)
            mn, mx = np.where(np.isfinite(mn), mn, np.nan), np.where(np.isfinite(mx), mx, np.nan)
            if "min" in stats:
                out["min"] = mn
            if "max" in stats:
                out["max"] = mx
            if "range" in stats:
                out["range"] = mx - mn
    return {k: np.where(ok, v, np.nan) for k, v in out.items()}


def local_quantiles(a: np.ndarray, size: int, qs=(0.1, 0.5, 0.9), min_support: float = 0.3,
                    chunk_rows: int = 64) -> dict:
    """Masked window quantiles (``np.nanquantile`` over a sliding window, processed in row chunks)."""
    from numpy.lib.stride_tricks import sliding_window_view

    a = np.asarray(a, np.float64)
    m = np.isfinite(a)
    r = int(size) // 2
    p = np.pad(np.where(m, a, np.nan), r, mode="constant", constant_values=np.nan)
    h, w = a.shape
    out = {q: np.full((h, w), np.nan) for q in qs}
    n = _box(m.astype(np.float64), size)
    for i0 in range(0, h, chunk_rows):
        i1 = min(i0 + chunk_rows, h)
        win = sliding_window_view(p[i0:i1 + 2 * r], (int(size), int(size))).reshape(i1 - i0, w, -1)
        with np.errstate(invalid="ignore"), _quiet():
            vals = np.nanquantile(win, list(qs), axis=-1)
        for j, q in enumerate(qs):
            out[q][i0:i1] = vals[j]
    ok = (n >= min_support) & m
    return {q: np.where(ok, v, np.nan) for q, v in out.items()}


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def local_entropy(a: np.ndarray, size: int, edges, min_support: float = 0.3) -> np.ndarray:
    """Normalised Shannon entropy in [0, 1] of the quantised value (bins = ``edges``, len(edges) + 1 levels)."""
    a = np.asarray(a, np.float64)
    m = np.isfinite(a)
    q = np.digitize(np.where(m, a, 0.0), np.asarray(edges, np.float64))
    L = len(edges) + 1
    n = _box(m.astype(np.float64), size)
    H = np.zeros(a.shape)
    for lev in range(L):
        with np.errstate(invalid="ignore", divide="ignore"):
            pl = _box(((q == lev) & m).astype(np.float64), size) / np.maximum(n, 1e-12)
            H -= np.where(pl > 0, pl * np.log(pl), 0.0)
    H /= np.log(L) if L > 1 else 1.0
    return np.where((n >= min_support) & m, H, np.nan)


def orientation_histograms(sr: np.ndarray, sc: np.ndarray, size: int, n_bins: int = 4,
                           min_support: float = 0.3) -> dict:
    """Share of gradient energy |grad|^2 per axial orientation bin (bins of 180 / n_bins degrees)."""
    g2 = sr * sr + sc * sc
    m = np.isfinite(g2)
    phi = np.mod(gradient_axis(np.where(m, sr, 0.0), np.where(m, sc, 0.0)), np.pi)
    b = np.minimum((phi / (np.pi / n_bins)).astype(int), n_bins - 1)
    e = np.where(m, g2, 0.0)
    tot = _box(e, size)
    n = _box(m.astype(np.float64), size)
    out = {}
    for j in range(n_bins):
        with np.errstate(invalid="ignore", divide="ignore"):
            out[j] = np.where((n >= min_support) & m & (tot > 0), _box(np.where(b == j, e, 0.0), size) / tot, np.nan)
    return out


def edge_density(edges: np.ndarray, size: int, min_support: float = 0.3) -> np.ndarray:
    m = np.isfinite(edges)
    n = _box(m.astype(np.float64), size)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = _box(np.where(m, edges, 0.0), size) / np.maximum(n, 1e-12)
    return np.where((n >= min_support) & m, d, np.nan)


def gradient_variability(hgm: np.ndarray, size: int, min_support: float = 0.3) -> np.ndarray:
    s = local_statistics(hgm, size, ("mean", "std"), min_support)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s["std"] / (s["mean"] + 1e-12)


def local_anisotropy(sr: np.ndarray, sc: np.ndarray, size: int, min_support: float = 0.3) -> np.ndarray:
    """Coherence ((l1 - l2) / (l1 + l2))^2 of the box-integrated structure tensor."""
    m = np.isfinite(sr) & np.isfinite(sc)
    ge, gn = np.where(m, sc, 0.0), np.where(m, -sr, 0.0)
    jee, jen, jnn = _box(ge * ge, size), _box(ge * gn, size), _box(gn * gn, size)
    n = _box(m.astype(np.float64), size)
    tr = jee + jnn
    disc = np.sqrt(np.maximum((jee - jnn) ** 2 + 4 * jen * jen, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(tr > 0, (disc / tr) ** 2, 0.0)
    return np.where((n >= min_support) & m, coh, np.nan)


# =============================================================================== names + bank
def texture_names(cfg: dict) -> list[tuple[str, str, int]]:
    """(feature id, kind, window) in the canonical order of ``texture_feature_bank``."""
    out = []
    for b in cfg["sources"]:
        for w in cfg["windows"]:
            for st in cfg.get("stats", ["std", "range"]):
                out.append((f"tex_{b}_{st}_w{w}", st, int(w)))
        for w in cfg.get("mean_windows", []):
            out.append((f"tex_{b}_mean_w{w}", "mean", int(w)))
        for w in cfg.get("quantile_windows", []):
            for q in cfg.get("quantiles", [0.1, 0.5, 0.9]):
                out.append((f"tex_{b}_q{int(round(100 * q)):02d}_w{w}", f"q{q}", int(w)))
        for w in cfg.get("entropy_windows", []):
            out.append((f"tex_{b}_ent_w{w}", "ent", int(w)))
    for b in cfg.get("edge_fields", []):
        for w in cfg.get("edge_windows", []):
            out.append((f"tex_{b}_edged_w{w}", "edged", int(w)))
            out.append((f"tex_{b}_gvar_w{w}", "gvar", int(w)))
            out.append((f"tex_{b}_aniso_w{w}", "aniso", int(w)))
        for w in cfg.get("orientation_windows", []):
            for j in range(int(cfg.get("orientation_bins", 4))):
                out.append((f"tex_{b}_oh{j}_w{w}", f"oh{j}", int(w)))
    return out


def required_halo_px(cfg: dict) -> int:
    ws = [int(w) for k in ("windows", "mean_windows", "quantile_windows", "entropy_windows", "edge_windows",
                           "orientation_windows") for w in cfg.get(k, [])]
    return (max(ws) // 2 if ws else 0) + 4 + 2       # window radius + sigma-100 m gradient support


def texture_feature_bank(fields: dict, cfg: dict, entropy_edges: dict, px_km: float = 0.1) -> dict:
    """All texture features of ``texture_names(cfg)``; ``fields`` = {band id: padded tile array};
    ``entropy_edges`` = {band id: footprint quantile edges} (from the raw band statistics pass)."""
    ms = float(cfg.get("min_support", 0.3))
    out = {}
    for b in cfg["sources"]:
        a = fields[b]
        valid = np.isfinite(a)

        def put(name, v, valid=valid):
            out[name] = np.where(valid, v, np.nan)

        stats = list(cfg.get("stats", ["std", "range"]))
        for w in cfg["windows"]:
            s = local_statistics(a, w, stats, ms)
            for st in stats:
                put(f"tex_{b}_{st}_w{w}", s[st])
        for w in cfg.get("mean_windows", []):
            put(f"tex_{b}_mean_w{w}", local_statistics(a, w, ("mean",), ms)["mean"])
        for w in cfg.get("quantile_windows", []):
            qs = list(cfg.get("quantiles", [0.1, 0.5, 0.9]))
            Q = local_quantiles(a, w, qs, ms)
            for q in qs:
                put(f"tex_{b}_q{int(round(100 * q)):02d}_w{w}", Q[q])
        for w in cfg.get("entropy_windows", []):
            put(f"tex_{b}_ent_w{w}", local_entropy(a, w, entropy_edges[b], ms))
    for b in cfg.get("edge_fields", []):
        a = fields[b]
        valid = np.isfinite(a)
        d = field_at_scale(a, 100.0, px_km)
        e = nms_edges(d["hgm"], d["sr"], d["sc"])
        for w in cfg.get("edge_windows", []):
            out[f"tex_{b}_edged_w{w}"] = np.where(valid, edge_density(e, w, ms), np.nan)
            out[f"tex_{b}_gvar_w{w}"] = np.where(valid, gradient_variability(d["hgm"], w, ms), np.nan)
            out[f"tex_{b}_aniso_w{w}"] = np.where(valid, local_anisotropy(d["sr"], d["sc"], w, ms), np.nan)
        for w in cfg.get("orientation_windows", []):
            H = orientation_histograms(d["sr"], d["sc"], w, int(cfg.get("orientation_bins", 4)), ms)
            for j, v in H.items():
                out[f"tex_{b}_oh{j}_w{w}"] = np.where(valid, v, np.nan)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}
