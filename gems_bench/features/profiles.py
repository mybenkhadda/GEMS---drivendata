"""Fault-normal structural profile features (Phase-5 Experiment D).

Hypothesis
----------
A fault is a *discontinuity across* a direction and *continuity along* it. A broad regional gradient,
a basin margin or a lithological contact can produce a large first derivative, but (i) its profile across
strike is typically smooth rather than step-like at the 100 m - 1 km scale, (ii) it lacks a lateral-offset
signature, and (iii) it does not persist as a sharp feature across scales. Sampling each field on short
profiles perpendicular (and parallel) to a label-free local strike turns these properties into features.

Orientation is estimated from measurements only (structure tensor of ``orient_src`` at ``orient_scale_km``),
never from fault labels, so the features are valid at inference time and transductively computable.
They still respond to lithological contacts and dikes: they are evidence of structure, not of faulting.

Profile at pixel x, smoothed field S (Gaussian, ``scale_km``), unit normal n, unit strike t, step d:
``v_k = S(x + k d n)`` for k = -K..K. Outputs (``outputs`` in the YAML):

``step``        |mean(v_1..v_K) - mean(v_-K..v_-1)| / local regional std          (cross-strike contrast)
``asymmetry``   sum|odd part| / (sum|odd| + sum|even|), odd_k = (v_k - v_-k)/2, even_k = (v_k + v_-k)/2 - v_0
                1 = step / monocline-like (offset), 0 = ridge / valley-like (symmetric)
``sign_change`` -(v_0 - v_-1)(v_1 - v_0) / (|.||.|): +1 ridge or valley axis, -1 monotonic ramp
``sharpness``   log1p( (|v_1 - v_-1| / 2) / (|v_K - v_-K| / 2K) ): gradient concentrated at the centre
``offset``      1 - min_s c(s) / c(0), c(s) = |S(x - d n + s t) - S(x + d n)|, s in +-shifts * d:
                a strike-slip-like lateral offset makes a shifted match better than the direct one
``lineation``   1 - std_along / (std_along + std_across): continuity along strike vs change across
``persistence`` (separate transform) fraction of ``scales_km`` where ``step`` is in the top (1 - q) of the map

Compute cost: one ``map_coordinates`` per sample (~16 per (field, scale)); intermediate samples are memoised
per store build (``rs_features._memo``), so all outputs of one (field, scale) share the sampling.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from gems_eda.enhance import fill_nan, nan_gaussian

from .. import rs_features as rs
from ..geology.orientation import along_normal_offsets

PROFILE_OUTPUTS = ("step", "asymmetry", "sign_change", "sharpness", "offset", "lineation")
LABEL_WORDS = ("label", "fault", "truth", "target")


def _check_label_free(*sources: str) -> None:
    for s in sources:
        if any(w in str(s).lower() for w in LABEL_WORDS):
            raise ValueError(f"profile orientation / source must be a measured field, got {s!r}")


def _make_samples(store, src: str, orient_src: str, orient_scale_km: float, scale_km: float, unit_km: float,
                  K: int, shifts: tuple[int, ...], full: bool = True) -> dict:
    """Sampled rasters for one (field, orientation, scale); ``full=False`` keeps only the normal profile."""
    _check_label_free(src, orient_src)
    a = rs._src(store, src)
    sg = store.enh.sigma_px(scale_km)
    S = fill_nan(nan_gaussian(a, sg)).astype(np.float32)
    st, _ = rs._structure(store, orient_src, orient_scale_km)
    d = unit_km / rs._px_km(store)
    (tr, tc), (nr, nc) = along_normal_offsets(st, d)
    rr, cc = np.indices(S.shape, dtype=np.float32)

    def at(dr, dc):
        return ndimage.map_coordinates(S, [rr + dr, cc + dc], order=1, mode="nearest").astype(np.float32)

    v = {0: S}
    for k in range(1, K + 1):
        v[k], v[-k] = at(k * nr, k * nc), at(-k * nr, -k * nc)
    reg = nan_gaussian(a, 5.0 * sg)
    loc_std = np.sqrt(np.maximum(nan_gaussian((a - reg) ** 2, 5.0 * sg), 1e-12))
    out = {"v": v, "loc_std": fill_nan(loc_std).astype(np.float32)}
    if full:
        out["along"] = [at(j * tr, j * tc) for j in range(-K, K + 1) if j]
        signed = [x for s in shifts for x in (s, -s)]
        out["shifted"] = {x: at(-nr + x * tr, -nc + x * tc) for x in signed}
    return out


def _samples(store, src: str, orient_src: str, orient_scale_km: float, scale_km: float, unit_km: float,
             K: int, shifts: tuple[int, ...]) -> dict:
    """One-entry memo (~18 full-grid float32 rasters, ~0.9 GB): all outputs of one (field, scale) share it.

    Kept apart from the 6-entry ``rs_features`` memo so profile sampling never multiplies resident memory.
    """
    key = ("profile", rs._band(src), rs._band(orient_src), orient_scale_km, scale_km, unit_km, K, tuple(shifts))
    memo = store.__dict__.get("_p5_profile")
    if memo is None or memo[0] != key:
        store.__dict__["_p5_profile"] = None             # free the previous set before building the next
        memo = (key, _make_samples(store, src, orient_src, orient_scale_km, scale_km, unit_km, K, shifts))
        store.__dict__["_p5_profile"] = memo
    return memo[1]


def clear_memo(store) -> None:
    store.__dict__.pop("_p5_profile", None)


def profile_output(sm: dict, output: str, K: int) -> np.ndarray:
    v, eps = sm["v"], 1e-9
    if output == "step":
        right = np.mean([v[k] for k in range(1, K + 1)], 0)
        left = np.mean([v[-k] for k in range(1, K + 1)], 0)
        return np.abs(right - left) / (sm["loc_std"] + eps)
    if output == "asymmetry":
        odd = sum(np.abs(v[k] - v[-k]) / 2 for k in range(1, K + 1))
        even = sum(np.abs((v[k] + v[-k]) / 2 - v[0]) for k in range(1, K + 1))
        return odd / (odd + even + eps)
    if output == "sign_change":
        dl, dr = v[0] - v[-1], v[1] - v[0]
        return -(dl * dr) / (np.abs(dl) * np.abs(dr) + eps)
    if output == "sharpness":
        centre = np.abs(v[1] - v[-1]) / 2
        mean_g = np.abs(v[K] - v[-K]) / (2 * K)
        return np.log1p(centre / (mean_g + eps))
    if output == "offset":
        c0 = np.abs(v[-1] - v[1])
        cs = np.min(np.stack([np.abs(x - v[1]) for x in sm["shifted"].values()]), 0)
        return np.clip(1 - cs / (c0 + eps), 0, 1)
    if output == "lineation":
        along = np.std(np.stack([v[0]] + sm["along"]), 0)
        across = np.std(np.stack([v[k] for k in range(-K, K + 1)]), 0)
        return 1 - along / (along + across + eps)
    raise KeyError(output)


def t_profile(store, spec) -> np.ndarray:
    K = int(spec.p("K", 3))
    sm = _samples(store, spec.p("src"), spec.p("orient_src"), float(spec.p("orient_scale_km", 2.0)), spec.scale_km,
                  float(spec.p("unit_km", 0.2)), K, tuple(int(s) for s in spec.p("shifts", (1, 2))))
    out = profile_output(sm, spec.output, K)
    return rs._restore(out, np.isfinite(rs._src(store, spec.p("src"))))


def t_profile_persistence(store, spec) -> np.ndarray:
    """Fraction of scales where the profile ``step`` ranks in the top (1 - q) of valid pixels."""
    K, q = int(spec.p("K", 3)), float(spec.p("q", 0.9))
    hits = []
    for s in spec.p("scales_km"):                    # profile reach grows with the smoothing scale
        sm = _make_samples(store, spec.p("src"), spec.p("orient_src"), float(spec.p("orient_scale_km", 2.0)), float(s),
                           float(spec.p("unit_km", 0.2)) * float(s) / float(spec.p("scales_km")[0]), K, (), full=False)
        hits.append(rs.prank(profile_output(sm, "step", K), store.data.valid) >= q)
        del sm
    return rs._restore(np.mean(hits, 0), store.data.valid)


TRANSFORMS = {"profile": t_profile, "profile_persistence": t_profile_persistence}


def support_km(transform: str, scale_km: float, params: dict) -> float:
    """~radius of input influence (buffer audit): orientation window, profile reach, regional normaliser."""
    unit, K = float(params.get("unit_km", 0.2)), int(params.get("K", 3))
    orient = 3 * float(params.get("orient_scale_km", 2.0)) * 3 / 2
    scales = params.get("scales_km", [scale_km]) if transform == "profile_persistence" else [scale_km]
    reach = max(K * unit * float(s) / float(scales[0]) + 3 * float(s) / 2 for s in scales)
    return max(orient, reach, 5 * 3 * max(float(s) for s in scales) / 2 / 2)
