"""Association of features with the mapped-fault TARGET (evaluation / diagnostics only).

Population: valid pixels of the evaluated area (feature finite, optional footprint). Positives = mapped-fault pixels.
Every other valid pixel is UNLABELLED - not a confirmed non-fault - so AUROC / AP here measure how well a feature ranks
mapped faults above unlabelled pixels (a PU ranking), not classification accuracy, and they are not DTI.

Direction: ``high`` (large values rank first), ``low`` (small values first) or ``frozen`` (chosen on a DEVELOPMENT
mask, then applied unchanged to the evaluation mask). Never choose a direction on held-out labels.
Pixels are spatially autocorrelated: use ``block_bootstrap`` (spatial blocks) for uncertainty and
``negative_control`` (shifted targets) to calibrate what an uninformative feature scores.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import rankdata

BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
DIST_EDGES_KM = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf)
INSUFFICIENT = "INSUFFICIENT_DATA"


# =============================================================================== core ranking metrics
def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Probability a random positive outranks a random unlabelled pixel (ties count 1/2)."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return np.nan
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    """Step-wise AP over distinct score thresholds (tied scores enter together)."""
    y = np.asarray(y, bool)
    P = int(y.sum())
    if not P or P == len(y):
        return np.nan
    o = np.argsort(-score, kind="mergesort")
    s, t = score[o], y[o]
    tp, fp = np.cumsum(t), np.cumsum(~t)
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]          # end of each tie group
    tp, fp = tp[last], fp[last]
    prec, rec = tp / (tp + fp), tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def recall_at(score: np.ndarray, y: np.ndarray, budget: float) -> float:
    """Share of positives among the top ``budget`` fraction of pixels (ceil; ties at the cutoff by order)."""
    y = np.asarray(y, bool)
    if not y.any():
        return np.nan
    k = max(1, int(math.ceil(budget * len(score))))
    top = np.argpartition(-score, k - 1)[:k]
    return float(y[top].sum() / y.sum())


def cliffs_delta_from_auc(a: float) -> float:
    return 2 * a - 1 if np.isfinite(a) else np.nan


def oriented(x: np.ndarray, direction: str) -> np.ndarray:
    if direction not in ("high", "low"):
        raise ValueError("direction must be 'high' or 'low'")
    return x if direction == "high" else -x


def choose_direction(x: np.ndarray, y: np.ndarray, dev_mask: np.ndarray) -> str:
    """Direction from the DEVELOPMENT pixels only (AUROC >= 0.5 -> high)."""
    m = dev_mask & np.isfinite(x)
    a = auroc(x[m], y[m])
    return "high" if not np.isfinite(a) or a >= 0.5 else "low"


def population(x, faults, mask=None):
    x = np.asarray(x, np.float32)
    m = np.isfinite(x)
    if mask is not None:
        m &= np.asarray(mask, bool)
    if faults is None:
        raise ValueError("no mapped-fault target loaded")
    return x[m].astype(np.float64), np.asarray(faults, bool)[m], m


def metrics(x, faults, mask=None, direction="high", budgets=BUDGETS, near_px=3, faults_ext=None, roi=None,
            pixel_m=100.0) -> dict:
    """All association metrics of one feature on the evaluated pixels (see module docstring)."""
    v, y, m = population(x, faults, mask)
    n, p = len(v), int(y.sum())
    out = {"n_valid": n, "n_fault_px": p, "prevalence": p / n if n else np.nan, "direction": direction}
    if n < 10 or p < 3 or p == n or np.nanstd(v) < 1e-12:
        out["status"] = INSUFFICIENT
        return out
    s = oriented(v, direction)
    a = auroc(s, y)
    ap = average_precision(s, y)
    fa, un = v[y], v[~y]
    mad_u = 1.4826 * np.median(np.abs(un - np.median(un)))
    rs = rankdata(v)
    out.update(status="OK", point_biserial=float(np.corrcoef(v, y.astype(float))[0, 1]),
               spearman=float(np.corrcoef(rs, y.astype(float))[0, 1]), auroc=a, auroc_high=auroc(v, y),
               average_precision=ap, ap_lift=ap / (p / n), median_fault=float(np.median(fa)),
               median_unlabelled=float(np.median(un)), median_difference=float(np.median(fa) - np.median(un)),
               cliffs_delta=cliffs_delta_from_auc(auroc(v, y)),
               robust_effect=float((np.median(fa) - np.median(un)) / mad_u) if mad_u > 0 else np.nan,
               association_sign=int(np.sign(auroc(v, y) - 0.5)) if np.isfinite(auroc(v, y)) else 0)
    for b in budgets:
        out[f"recall_at_{b * 100:g}pct"] = recall_at(s, y, b)
    out.update(enrichment_near_traces(x, faults, mask, direction, 0.01, near_px, faults_ext, roi))
    return out


def enrichment_near_traces(x, faults, mask, direction, budget=0.01, near_px=3, faults_ext=None, roi=None) -> dict:
    """Share of the top-``budget`` pixels within ``near_px`` of a mapped trace / share of all valid pixels there."""
    f = np.asarray(faults if faults_ext is None else faults_ext, bool)
    d = ndimage.distance_transform_edt(~f) if f.any() else np.full(f.shape, np.inf)
    if roi is not None:
        d = d[roi]
    near = d <= near_px
    v, _, m = population(x, faults, mask)
    nm = near[m]
    k = max(1, int(math.ceil(budget * len(v))))
    top = np.argpartition(-oriented(v, direction), k - 1)[:k]
    base = float(nm.mean()) if len(nm) else np.nan
    share = float(nm[top].mean())
    return {"top1pct_near_share": share, "near_share_all": base,
            "near_trace_enrichment": share / base if base else np.nan}


def leaderboard(layers: dict, faults, mask=None, direction_mode="high", dev_mask=None, eval_mask=None,
                budgets=BUDGETS, faults_ext=None, roi=None) -> pd.DataFrame:
    """One row per layer. ``direction_mode``: 'high' | 'low' | 'frozen' (direction chosen on ``dev_mask``, metrics
    on ``eval_mask``) | 'best_exploratory' (better direction on the same pixels: optimistic, flagged)."""
    rows = []
    for name, x in layers.items():
        x = np.asarray(x, np.float32)
        if direction_mode == "frozen":
            if dev_mask is None or eval_mask is None:
                raise ValueError("frozen direction needs dev_mask and eval_mask")
            d = choose_direction(x, np.asarray(faults, bool), dev_mask)
            r = metrics(x, faults, eval_mask if mask is None else (eval_mask & mask), d, budgets, faults_ext=faults_ext,
                        roi=roi)
            r["direction_source"] = "development mask"
        elif direction_mode == "best_exploratory":
            r_hi = metrics(x, faults, mask, "high", budgets, faults_ext=faults_ext, roi=roi)
            d = "high" if r_hi.get("auroc_high", 0.5) >= 0.5 else "low"
            r = r_hi if d == "high" else metrics(x, faults, mask, "low", budgets, faults_ext=faults_ext, roi=roi)
            r["direction_source"] = "SAME pixels (optimistic; exploratory only)"
        else:
            r = metrics(x, faults, mask, direction_mode, budgets, faults_ext=faults_ext, roi=roi)
            r["direction_source"] = "fixed by user"
        rows.append({"feature": name, **r})
    T = pd.DataFrame(rows)
    if "auroc" in T:
        T = T.sort_values("auroc", ascending=False, na_position="last", ignore_index=True)
    return T


# =============================================================================== uncertainty / controls
def block_ids(shape, block_px: int) -> np.ndarray:
    r, c = np.indices(shape)
    return (r // block_px) * (shape[1] // block_px + 1) + c // block_px


def block_bootstrap(x, faults, mask=None, direction="high", block_px=20, n_boot=200, max_px=60000, seed=2026):
    """AUROC / AP-lift 95 % CI from resampling spatial blocks (block_px pixels; GEMS 20 px = 2 km) with replacement.
    Pixels are thinned to ``max_px`` first (uniformly) to bound cost."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float32)
    m = np.isfinite(x) & (np.asarray(mask, bool) if mask is not None else True)
    b = block_ids(x.shape, int(block_px))[m]
    v, y = oriented(x[m].astype(np.float64), direction), np.asarray(faults, bool)[m]
    if len(v) > max_px:
        i = rng.choice(len(v), max_px, replace=False)
        v, y, b = v[i], y[i], b[i]
    ub, inv = np.unique(b, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(ub))]
    A, L = [], []
    for _ in range(int(n_boot)):
        sel = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if y[sel].any() and (~y[sel]).any():
            A.append(auroc(v[sel], y[sel]))
            L.append(average_precision(v[sel], y[sel]) / y[sel].mean())
    q = lambda z: (float(np.nanpercentile(z, 2.5)), float(np.nanpercentile(z, 97.5))) if z else (np.nan, np.nan)  # noqa
    return {"auroc_ci": q(A), "ap_lift_ci": q(L), "n_blocks": len(groups), "n_boot_valid": len(A),
            "block_km": block_px * 0.1}


def negative_control(x, faults, mask=None, direction="high", n=20, min_shift_px=50, seed=2026) -> dict:
    """AUROC of the feature against circularly SHIFTED copies of the target (spatial surrogates that keep the trace
    geometry and density but break the co-location; shifts >= ``min_shift_px`` in both axes). Mirrors are not used:
    a mirrored trace can coincide with itself (e.g. a horizontal trace under a left-right flip). The feature's real
    AUROC should exceed this null."""
    rng = np.random.default_rng(seed)
    f = np.asarray(faults, bool)
    H, W = f.shape
    null = []
    for _ in range(int(n)):
        dr = int(rng.integers(min_shift_px, H - min_shift_px)) if H > 2 * min_shift_px else 0
        dc = int(rng.integers(min_shift_px, W - min_shift_px)) if W > 2 * min_shift_px else 0
        if not dr and not dc:
            continue
        g = np.roll(f, (dr, dc), axis=(0, 1))
        v, y, _ = population(x, g, mask)
        if y.any() and (~y).any():
            null.append(auroc(oriented(v, direction), y))
    v, y, _ = population(x, f, mask)
    real = auroc(oriented(v, direction), y)
    null = np.asarray(null)
    return {"auroc": real, "null_mean": float(null.mean()) if null.size else np.nan,
            "null_p95": float(np.percentile(null, 95)) if null.size else np.nan,
            "null_max": float(null.max()) if null.size else np.nan, "n_surrogates": int(null.size),
            "exceeds_null_p95": bool(null.size and real > np.percentile(null, 95))}


def matched_control_effect(x, faults, covariate, mask=None, tile_px=50, n_bins=3, min_dist_px=10,
                           pixel_m=100.0) -> dict:
    """Cliff's delta of mapped-fault pixels vs unlabelled pixels >= ``min_dist_px`` from any trace in the SAME spatial
    tile (``tile_px``) and the SAME tercile of a label-free covariate (e.g. det_elev_slope), weighted by the number
    of fault pixels per stratum."""
    x = np.asarray(x, np.float32)
    f = np.asarray(faults, bool)
    cov = np.asarray(covariate, np.float32)
    m = np.isfinite(x) & np.isfinite(cov) & (np.asarray(mask, bool) if mask is not None else True)
    far = ndimage.distance_transform_edt(~f) >= min_dist_px if f.any() else np.ones(f.shape, bool)
    q = np.nanquantile(cov[m], np.linspace(0, 1, n_bins + 1)[1:-1]) if m.any() else []
    strata = block_ids(x.shape, tile_px) * 10 + np.digitize(np.nan_to_num(cov), q)
    rows, num, den = [], 0.0, 0
    for s in np.unique(strata[m & f]):
        a = x[m & f & (strata == s)]
        b = x[m & far & ~f & (strata == s)]
        if len(a) < 3 or len(b) < 10:
            continue
        d = cliffs_delta_from_auc(auroc(np.r_[a, b], np.r_[np.ones(len(a), bool), np.zeros(len(b), bool)]))
        rows.append({"stratum": int(s), "n_fault": len(a), "n_control": len(b), "delta": d})
        num += d * len(a)
        den += len(a)
    return {"matched_delta": num / den if den else np.nan, "n_strata": len(rows), "n_fault_matched": den,
            "control_min_dist_km": min_dist_px * pixel_m / 1000, "strata": pd.DataFrame(rows)}


# =============================================================================== proximity / cross sections
def proximity_profile(x, faults_ext, roi, pixel_m=100.0, edges_km=DIST_EDGES_KM, mask=None) -> pd.DataFrame:
    """Feature quantiles by distance band to mapped traces (distance measured on the padded window so traces just
    outside the AOI count). Band 0 = on a mapped-fault pixel. Retrospective diagnostic, not a predictor."""
    f = np.asarray(faults_ext, bool)
    d = (ndimage.distance_transform_edt(~f) * pixel_m / 1000.0) if f.any() else np.full(f.shape, np.inf)
    d = d[roi]
    x = np.asarray(x, np.float32)
    m = np.isfinite(x) & (np.asarray(mask, bool) if mask is not None else True)
    rows = []
    bands = [("on fault", d == 0)] + [(f"{lo:g}-{hi:g} km" if np.isfinite(hi) else f"> {lo:g} km",
                                       (d > lo) & (d <= hi)) for lo, hi in zip(edges_km[:-1], edges_km[1:])]
    for lab, sel in bands:
        v = x[m & sel]
        r = {"band": lab, "n": int(v.size)}
        if v.size:
            r.update(zip(("p05", "p25", "median", "p75", "p95"), np.percentile(v, [5, 25, 50, 75, 95]).tolist()))
            r["mean"] = float(v.mean())
        rows.append(r)
    return pd.DataFrame(rows)


def fault_normal_profiles(x, faults_ext, roi, half_px=30, max_stations=2000, sigma=2.0, seed=2026) -> pd.DataFrame:
    """Aligned profiles of ``x`` along the local normal of mapped traces (orientation from the structure tensor of
    the fault mask). RETROSPECTIVE: uses mapped geometry, so it can only describe known faults, never generate
    candidates. Returns offset (px), median, p25, p75, n."""
    from scipy.ndimage import map_coordinates

    from . import ops as O

    f = np.asarray(faults_ext, bool)
    if not f.any():
        return pd.DataFrame(columns=["offset_px", "median", "p25", "p75", "n"])
    strike = O.orientation(f.astype(np.float32), sigma)          # strike of the mask fabric
    r0, c0 = roi[0].start, roi[1].start
    rr, cc = np.nonzero(f[roi])
    rng = np.random.default_rng(seed)
    if len(rr) > max_stations:
        i = rng.choice(len(rr), max_stations, replace=False)
        rr, cc = rr[i], cc[i]
    th = np.radians(strike[rr + r0, cc + c0] + 90.0)            # normal azimuth
    dn, de = np.cos(th), np.sin(th)                              # unit (north, east)
    off = np.arange(-half_px, half_px + 1)
    R = rr[:, None] - off[None, :] * dn[:, None]                 # north = -row
    C = cc[:, None] + off[None, :] * de[:, None]
    a = np.asarray(x, np.float32)
    ok = np.isfinite(a)
    val = map_coordinates(np.where(ok, a, 0), [R.ravel(), C.ravel()], order=1, mode="constant", cval=np.nan)
    w = map_coordinates(ok.astype(np.float32), [R.ravel(), C.ravel()], order=1, mode="constant", cval=0)
    P = np.where(w > 0.99, val, np.nan).reshape(R.shape)
    with np.errstate(all="ignore"):
        return pd.DataFrame({"offset_px": off, "median": np.nanmedian(P, 0), "p25": np.nanpercentile(P, 25, 0),
                             "p75": np.nanpercentile(P, 75, 0), "n": np.isfinite(P).sum(0)})


# =============================================================================== regional / tiles
def tile_masks(shape, mode: str, n: int = 3, tile_px: int = 50, domains: np.ndarray | None = None):
    """Yield (tile_id, label, (r0, r1, c0, c1) or None, mask). mode: 'grid' (n x n), 'fixed' (tile_px), 'domain'."""
    H, W = shape
    if mode == "grid":
        for i in range(n):
            for j in range(n):
                r0, r1, c0, c1 = H * i // n, H * (i + 1) // n, W * j // n, W * (j + 1) // n
                m = np.zeros(shape, bool)
                m[r0:r1, c0:c1] = True
                yield (i, j), f"{i},{j}", (r0, r1, c0, c1), m
    elif mode == "fixed":
        for i, r0 in enumerate(range(0, H, tile_px)):
            for j, c0 in enumerate(range(0, W, tile_px)):
                r1, c1 = min(H, r0 + tile_px), min(W, c0 + tile_px)
                m = np.zeros(shape, bool)
                m[r0:r1, c0:c1] = True
                yield (i, j), f"{i},{j}", (r0, r1, c0, c1), m
    elif mode == "domain":
        if domains is None:
            raise ValueError("no label-free domain raster available")
        for d in np.unique(domains[domains >= 0]):
            yield (int(d), 0), f"domain {int(d)}", None, domains == d
    else:
        raise ValueError("mode must be grid | fixed | domain")


def tile_table(x, faults, mode="grid", n=3, tile_px=50, domains=None, mask=None, direction="high", budget=0.01,
               min_pos=20, min_neg=200, control_min_dist_px=10) -> pd.DataFrame:
    """Per tile: support, prevalence, point-biserial, AUROC, AP lift, recall@budget (within the tile) and a local
    matched-control effect (Cliff's delta vs unlabelled pixels >= control_min_dist_px from traces in the same tile).
    Tiles below ``min_pos`` positives / ``min_neg`` unlabelled pixels or without variation are INSUFFICIENT_DATA."""
    x = np.asarray(x, np.float32)
    f = np.asarray(faults, bool)
    far = ndimage.distance_transform_edt(~f) >= control_min_dist_px if f.any() else np.ones(f.shape, bool)
    rows = []
    for tid, lab, box, m in tile_masks(x.shape, mode, n, tile_px, domains):
        mm = m & np.isfinite(x) & (np.asarray(mask, bool) if mask is not None else True)
        v, y = x[mm].astype(np.float64), f[mm]
        p, q = int(y.sum()), int((~y).sum())
        r = {"tile_row": tid[0], "tile_col": tid[1], "label": lab, "n_valid": int(len(v)), "n_fault_px": p,
             "prevalence": p / len(v) if len(v) else np.nan}
        if box:
            r.update(zip(("r0", "r1", "c0", "c1"), box))
        if p < min_pos or q < min_neg or (len(v) and np.std(v) < 1e-12) or not len(v):
            r["status"] = INSUFFICIENT
        else:
            s = oriented(v, direction)
            a = auroc(s, y)
            ctl = x[m & far & ~f & np.isfinite(x)]
            r.update(status="OK", point_biserial=float(np.corrcoef(v, y.astype(float))[0, 1]), auroc=a,
                     ap_lift=average_precision(s, y) / (p / len(v)), recall_at_budget=recall_at(s, y, budget),
                     matched_delta=cliffs_delta_from_auc(auroc(np.r_[x[mm & f], ctl], np.r_[np.ones(p, bool),
                                                                                        np.zeros(len(ctl), bool)]))
                     if len(ctl) >= min_neg // 4 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def moving_association(x, faults, window_px=21, min_valid_share=0.6, min_pos=3):
    """Moving-window point-biserial correlation (Pearson with the binary target); NaN where the window holds fewer
    than ``min_pos`` fault pixels or insufficient valid pixels."""
    from .stats import local_corr

    f = np.asarray(faults, np.float32)
    r, n = local_corr(np.asarray(x, np.float32), f, window_px=window_px, min_valid_share=min_valid_share)
    pos = ndimage.uniform_filter(np.asarray(faults, np.float32), size=window_px, mode="constant") * window_px ** 2
    return np.where(pos >= min_pos, r, np.nan).astype(np.float32), n


# =============================================================================== dev / eval splits
def split_masks(shape, mode="halves_ew", block_px=50, buffer_px=5, seed=2026):
    """(development mask, evaluation mask) inside an AOI, separated by a ``buffer_px`` gap: 'halves_ew' (west dev /
    east eval), 'halves_ns' or 'checkerboard' (blocks of block_px)."""
    H, W = shape
    r, c = np.indices(shape)
    if mode == "halves_ew":
        dev, ev = c < W // 2 - buffer_px, c >= W // 2 + buffer_px
    elif mode == "halves_ns":
        dev, ev = r < H // 2 - buffer_px, r >= H // 2 + buffer_px
    elif mode == "checkerboard":
        b = ((r // block_px) + (c // block_px)) % 2 == 0
        edge = ndimage.binary_dilation(b, iterations=buffer_px) & ndimage.binary_dilation(~b, iterations=buffer_px)
        dev, ev = b & ~edge, ~b & ~edge
    else:
        raise ValueError("mode must be halves_ew | halves_ns | checkerboard")
    return dev, ev
