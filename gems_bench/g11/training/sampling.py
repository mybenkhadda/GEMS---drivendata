"""Spatially balanced mixture sampling of pixels / patch centres, restricted to ONE split.

Categories (``sampling.mix``): positive (mapped fault), near (within ``labels.near_band_m``, not fault),
background (unlabeled), hard_geophysics (top-decile geophysical edge strength, unlabeled), hard_terrain (top-decile
terrain scarp evidence, unlabeled), disagreement (top-decile model disagreement, unlabeled). Within a category,
samples are spread over ``balance_block_km`` blocks (at most ceil(n / n_blocks) per block) so dense regions do not
dominate. Hard negatives use only label-free evidence and only the requested split's labels (never test labels).
"""
from __future__ import annotations

import numpy as np


def _spread(idx: np.ndarray, block: np.ndarray, n: int, rng) -> np.ndarray:
    if idx.size <= n:
        return idx
    b = block[idx]
    order = rng.permutation(idx.size)
    idx, b = idx[order], b[order]
    srt = np.argsort(b, kind="stable")                    # random order kept within each block
    bs = b[srt]
    first = np.r_[0, np.flatnonzero(np.diff(bs)) + 1]
    n_blocks = len(first)
    starts = np.repeat(first, np.diff(np.r_[first, bs.size]))
    rank = np.empty(idx.size, np.int64)
    rank[srt] = np.arange(bs.size) - starts
    cap = int(np.ceil(n / n_blocks))
    keep = idx[rank < cap]
    return keep if keep.size <= n else rng.choice(keep, n, replace=False)


def category_masks(region: np.ndarray, fault: np.ndarray, dist_m: np.ndarray, near_m: float,
                   hard_geo: np.ndarray | None = None, hard_terrain: np.ndarray | None = None,
                   disagreement: np.ndarray | None = None) -> dict:
    unl = region & ~fault & (dist_m > near_m)
    out = {"positive": region & fault, "near": region & ~fault & (dist_m <= near_m), "background": unl}

    def top(a):
        if a is None:
            return None
        v = a[unl & np.isfinite(a)]
        if not v.size:
            return None
        return unl & (np.nan_to_num(a, nan=-np.inf) >= np.quantile(v, 0.9))

    for k, a in (("hard_geophysics", hard_geo), ("hard_terrain", hard_terrain), ("disagreement", disagreement)):
        m = top(a)
        if m is not None:
            out[k] = m
    return out


def mixture_sample(cats: dict, mix: dict, n: int, block_px: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (flat indices, category codes, realised counts). Missing categories' shares go to background."""
    rng = np.random.default_rng(seed)
    shape = next(iter(cats.values())).shape
    nbc = -(-shape[1] // block_px)
    ri, ci = np.indices(shape, dtype=np.int32)
    block = ((ri // block_px) * nbc + ci // block_px).ravel()
    del ri, ci
    share = {k: v for k, v in mix.items() if k in cats and cats[k].any()}
    lost = 1.0 - sum(share.values())
    if "background" in share:
        share["background"] += lost
    names = list(share)
    picks, codes, counts = [], [], {}
    for i, k in enumerate(names):
        idx = np.flatnonzero(cats[k].ravel())
        s = _spread(idx, block, int(round(share[k] * n)), rng)
        picks.append(s)
        codes.append(np.full(s.size, i, np.int8))
        counts[k] = int(s.size)
    if not picks:
        return np.array([], np.int64), np.array([], np.int8), {}
    return np.concatenate(picks), np.concatenate(codes), {"counts": counts, "names": names}
