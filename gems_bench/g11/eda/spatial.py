"""Spatial sampling helpers for EDA: one random valid pixel per block (independence-respecting samples) and an
empirical semivariogram / correlation-length estimate used to justify block sizes and embargo widths."""
from __future__ import annotations

import numpy as np
import pandas as pd


def block_sample(mask: np.ndarray, block_px: int, seed: int = 2026, max_n: int | None = None):
    rng = np.random.default_rng(seed)
    r = rng.random(mask.shape, dtype=np.float32)
    r[~mask] = 2.0
    nbc = -(-mask.shape[1] // block_px)
    ri, ci = np.indices(mask.shape, dtype=np.int32)
    bid = ((ri // block_px) * nbc + ci // block_px).ravel()
    order = np.lexsort((r.ravel(), bid))
    _, first = np.unique(bid[order], return_index=True)
    pick = order[first]
    pick = pick[r.ravel()[pick] < 2.0]
    if max_n and pick.size > max_n:
        pick = rng.choice(pick, max_n, replace=False)
    rows, cols = np.unravel_index(pick, mask.shape)
    return rows, cols, bid[pick]


def correlation_length(a: np.ndarray, valid: np.ndarray, px_m: float, max_lag_px: int = 60, n: int = 20000,
                       seed: int = 2026) -> pd.DataFrame:
    """Correlation of a field with itself at increasing E-W / N-S lags (random valid pairs)."""
    rng = np.random.default_rng(seed)
    rr, cc = np.nonzero(valid & np.isfinite(a))
    if rr.size < 100:
        return pd.DataFrame()
    pick = rng.choice(rr.size, min(n, rr.size), replace=False)
    rr, cc = rr[pick], cc[pick]
    rows = []
    for lag in np.unique(np.r_[1, 2, 3, 5, 8, 12, 20, 30, 45, max_lag_px]):
        for dr, dc in ((0, lag), (lag, 0)):
            r2, c2 = rr + dr, cc + dc
            ok = (r2 < a.shape[0]) & (c2 < a.shape[1])
            x, y = a[rr[ok], cc[ok]], a[r2[ok], c2[ok]]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 50:
                rows.append({"lag_m": lag * px_m, "direction": "EW" if dc else "NS",
                             "corr": float(np.corrcoef(x[m], y[m])[0, 1])})
    return pd.DataFrame(rows)


def decorrelation_distance(cl: pd.DataFrame, thr: float = 0.2) -> float:
    if cl.empty:
        return np.nan
    g = cl.groupby("lag_m")["corr"].mean()
    below = g[g < thr]
    return float(below.index.min()) if len(below) else float(g.index.max())
