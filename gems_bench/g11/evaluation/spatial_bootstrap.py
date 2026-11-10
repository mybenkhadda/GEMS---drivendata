"""Spatial block bootstrap: resample geographic blocks (not pixels) with replacement.

``block_bootstrap(stat_fn, block_ids, n_boot)`` calls ``stat_fn(index_array)`` on the concatenated member indices
of each resampled block set; returns the point estimate and percentile CI. Neighbouring pixels are not treated as
independent observations.
"""
from __future__ import annotations

import numpy as np


def block_ids(rows, cols, block_px: int, width: int) -> np.ndarray:
    nbc = -(-width // block_px)
    return (np.asarray(rows) // block_px) * nbc + np.asarray(cols) // block_px


def block_bootstrap(stat_fn, blocks: np.ndarray, n_boot: int = 500, ci: float = 0.9, seed: int = 2026) -> dict:
    blocks = np.asarray(blocks)
    ub, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ub))]
    est = float(stat_fn(np.arange(blocks.size)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ub), len(ub))
        idx = np.concatenate([members[k] for k in pick]) if len(pick) else np.array([], int)
        try:
            v = float(stat_fn(idx))
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    a = (1 - ci) / 2
    lo, hi = (float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))) if vals else (np.nan, np.nan)
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "n_blocks": int(len(ub)), "n_boot_valid": len(vals)}
