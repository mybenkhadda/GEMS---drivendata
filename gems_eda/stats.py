"""Small statistical helpers: rank AUC, axial (orientation) statistics, percentiles."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auc_score(pos, neg) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg); NaNs dropped."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 10 or len(neg) < 10:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def percentile_in(values, reference) -> np.ndarray:
    """Empirical CDF of ``reference`` evaluated at ``values`` (NaN-safe)."""
    ref = np.sort(np.asarray(reference, float)[np.isfinite(reference)])
    values = np.asarray(values, float)
    out = np.full(values.shape, np.nan)
    if len(ref) == 0:
        return out
    ok = np.isfinite(values)
    out[ok] = np.searchsorted(ref, values[ok], side="right") / len(ref)
    return out


def rank_normalize(a: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Map finite values to their percentile rank in (0, 1]."""
    out = np.full(a.shape, np.nan, dtype="float32")
    m = np.isfinite(a) if valid is None else (np.isfinite(a) & valid)
    if m.any():
        out[m] = (rankdata(a[m]) / m.sum()).astype("float32")
    return out


def axial_mean(az_deg, weights=None) -> tuple[float, float]:
    """Mean of axial data (0-180 deg) via angle doubling; returns (mean_az, resultant R)."""
    az = np.radians(2 * np.asarray(az_deg, float))
    w = np.ones_like(az) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(az) & np.isfinite(w)
    if not ok.any() or w[ok].sum() <= 0:
        return float("nan"), float("nan")
    c = np.sum(w[ok] * np.cos(az[ok])) / w[ok].sum()
    s = np.sum(w[ok] * np.sin(az[ok])) / w[ok].sum()
    return float(np.degrees(np.arctan2(s, c)) / 2 % 180), float(np.hypot(c, s))


def axial_hist(az_deg, weights=None, bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """Weighted orientation histogram on [0, 180); returns (bin centres, density summing to 1)."""
    az = np.asarray(az_deg, float) % 180
    w = np.ones_like(az) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(az) & np.isfinite(w)
    edges = np.linspace(0, 180, bins + 1)
    h, _ = np.histogram(az[ok], bins=edges, weights=w[ok])
    total = h.sum()
    return 0.5 * (edges[:-1] + edges[1:]), (h / total if total > 0 else h.astype(float))


def axial_diff(a_deg, b_deg) -> np.ndarray:
    """Smallest angle between two axial orientations, in [0, 90]."""
    d = np.abs((np.asarray(a_deg, float) - np.asarray(b_deg, float)) % 180)
    return np.minimum(d, 180 - d)


def subsample(idx: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return idx if len(idx) <= k else rng.choice(idx, size=k, replace=False)
