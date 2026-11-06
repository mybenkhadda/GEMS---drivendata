"""Robust quantile normalisation, computed once per feature cache and saved next to it.

scalar channel:  x' = clip((x - q50) / max(q_high - q_low, tiny), -clip, clip)
axial channel:   x' = x               ((cos 2t, sin 2t) components keep their sign symmetry)
NaN -> 0 after normalisation (= the median of a scalar channel, = "no orientation" of an axial channel).

Statistics use a deterministic sample of footprint pixels and only the input rasters (never labels), so they
are identical for every fold; this is a label-free transductive choice recorded in FEATURE_SCHEMA.md.
"""
from __future__ import annotations

import numpy as np

from .contracts import SCALAR, channel_kind


def robust_stats(stack: np.ndarray, names: list[str], footprint: np.ndarray, cfg: dict, seed: int) -> dict:
    n = cfg["normalization"]
    idx = np.flatnonzero(footprint.ravel())
    rng = np.random.default_rng(seed)
    if len(idx) > n["sample_px"]:
        idx = np.sort(rng.choice(idx, int(n["sample_px"]), replace=False))
    out = {}
    for i, name in enumerate(names):
        v = stack[i].ravel()[idx]
        v = v[np.isfinite(v)]
        kind = channel_kind(name)
        rec = {"kind": kind, "finite_frac_sample": float(len(v) / max(len(idx), 1))}
        if kind == SCALAR and len(v):
            ql, q50, qh = np.quantile(v, [n["q_low"], 0.5, n["q_high"]])
            scale = float(qh - ql)
            if not np.isfinite(scale) or scale <= 1e-12 * max(abs(q50), 1.0):
                scale = float(np.std(v)) or 1.0
            rec.update(center=float(q50), scale=scale, q_low=float(ql), q_high=float(qh), clip=float(n["clip"]))
        elif kind == SCALAR:
            rec.update(center=0.0, scale=1.0, q_low=np.nan, q_high=np.nan, clip=float(n["clip"]), empty=True)
        else:
            rec.update(center=0.0, scale=1.0, clip=1.0)
        out[name] = rec
    return out


def apply(arr: np.ndarray, rec: dict, dtype=np.float16) -> np.ndarray:
    x = (arr.astype(np.float32) - rec["center"]) / rec["scale"]
    x = np.clip(x, -rec["clip"], rec["clip"])
    x[~np.isfinite(x)] = 0.0
    return x.astype(dtype)
