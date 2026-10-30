"""Full-grid association / recovery metrics (vectorised for ~5 M valid native pixels).

Population: valid pixels (submission footprint & finite feature). Positives: mapped-fault pixels. Every other valid
pixel is UNLABELLED (not a confirmed non-fault). Rankings are PU rankings, not classification accuracy, not DTI.
All budget-based quantities use the FULL valid population of the evaluated region (no sampling).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import rankdata

BUDGETS = (0.005, 0.01, 0.02, 0.05)
DIST_EDGES_KM = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf)
DIST_LABELS = ("0 m (on trace)", "0-100 m", "100-200 m", "200-300 m", "300-500 m", "0.5-1 km", "1-2 km", "2-5 km",
               "> 5 km")
INSUFFICIENT = "INSUFFICIENT_DATA"
S8 = np.ones((3, 3), bool)


def _ap(score, y):
    """Average precision over distinct thresholds (ties enter together)."""
    P = int(y.sum())
    if not P or P == len(y):
        return np.nan
    o = np.argsort(-score, kind="mergesort")
    s, t = score[o], y[o]
    tp, fp = np.cumsum(t), np.cumsum(~t)
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp, fp = tp[last], fp[last]
    return float(np.sum(np.diff(np.r_[0.0, tp / P]) * (tp / (tp + fp))))


def _topk(score, k):
    return np.argpartition(-score, k - 1)[:k]


def association_both(x, faults, valid, budgets=BUDGETS, near300=None) -> dict:
    """Metrics for BOTH directions from one ranking. ``near300``: bool map of pixels within 300 m of a mapped trace
    (for top-1 % enrichment). Returns a flat dict (``*_high`` / ``*_low`` suffixes)."""
    x = np.asarray(x, np.float32)
    m = valid & np.isfinite(x)
    v = x[m].astype(np.float64)
    y = np.asarray(faults, bool)[m]
    n, p = int(v.size), int(y.sum())
    out = {"n_valid": n, "n_fault_px": p, "prevalence": p / n if n else np.nan}
    if n < 50 or p < 5 or p == n or float(v.std()) < 1e-12:
        out["status"] = INSUFFICIENT
        return out
    r = rankdata(v)
    a_hi = float((r[y].sum() - p * (p + 1) / 2) / (p * (n - p)))
    yf = y.astype(np.float64)
    out.update(status="OK", point_biserial=float(np.corrcoef(v, yf)[0, 1]), spearman=float(np.corrcoef(r, yf)[0, 1]),
               auroc_high=a_hi, auroc_low=1.0 - a_hi, cliffs_delta=2 * a_hi - 1,
               ap_high=_ap(v, y), ap_low=_ap(-v, y), median_fault=float(np.median(v[y])),
               median_unlabelled=float(np.median(v[~y])))
    out["median_difference"] = out["median_fault"] - out["median_unlabelled"]
    out["ap_lift_high"], out["ap_lift_low"] = out["ap_high"] / out["prevalence"], out["ap_low"] / out["prevalence"]
    out["better_direction_EXPLORATORY"] = "high" if a_hi >= 0.5 else "low"
    nz = near300[m] if near300 is not None else None
    for d, s in (("high", v), ("low", -v)):
        for b in budgets:
            k = max(1, int(math.ceil(b * n)))
            top = _topk(s, k)
            out[f"recall_{b * 100:g}pct_{d}"] = float(y[top].sum() / p)
            if nz is not None and abs(b - 0.01) < 1e-12:
                out[f"top1pct_near300_share_{d}"] = float(nz[top].mean())
    if nz is not None:
        out["near300_share_all"] = float(nz.mean())
    return out


def oriented(x, direction):
    return np.asarray(x, np.float32) if direction == "high" else -np.asarray(x, np.float32)


def top_mask(score, valid, budget):
    s = np.where(valid & np.isfinite(score), np.asarray(score, np.float64), -np.inf)
    n = int((valid & np.isfinite(score)).sum())
    k = max(1, int(math.ceil(budget * n)))
    flat = np.argpartition(-s.ravel(), k - 1)[:k]
    out = np.zeros(s.size, bool)
    out[flat] = True
    return out.reshape(s.shape) & valid & np.isfinite(score)


def fault_components(faults):
    lab, n = ndimage.label(np.asarray(faults, bool), S8)
    return lab, n


def recovery(score, faults, valid, budget, comp_lab, n_comp, branch_paths=None, radius_px=3, dist_km=None,
             share=0.5) -> dict:
    """Exact / within-300 m pixel recall, component and branch coverage, off-trace selected area and localisation
    index (share of selected pixels within 300 m / within 2 km of a mapped trace)."""
    f = np.asarray(faults, bool) & valid
    sel = top_mask(score, valid, budget)
    near = ndimage.binary_dilation(sel, S8, iterations=radius_px) if sel.any() else sel
    P = int(f.sum())
    out = {"budget": budget, "selected_px": int(sel.sum()),
           "recall_exact": float((sel & f).sum() / P) if P else np.nan,
           "recall_300m": float((near & f).sum() / P) if P else np.nan}
    if n_comp:
        idx = np.arange(1, n_comp + 1)
        size = ndimage.sum(f, comp_lab, idx)
        hit = ndimage.sum(near & f, comp_lab, idx)
        ok = size > 0
        rec = (hit[ok] / size[ok]) >= share
        out.update(components=int(ok.sum()), component_recall=float(rec.mean()) if ok.any() else np.nan)
        out["_comp_recovered"] = np.flatnonzero(ok)[rec] + 1
    if branch_paths:
        got = [float(near[p[:, 0], p[:, 1]].mean()) >= share for p in branch_paths if valid[p[:, 0], p[:, 1]].mean()
               >= 0.5]
        out.update(branches=len(got), branch_recall=float(np.mean(got)) if got else np.nan)
    if dist_km is not None:
        d = dist_km[sel]
        out.update(selected_within_300m=float((d <= 0.3).mean()) if d.size else np.nan,
                   selected_within_2km=float((d <= 2.0).mean()) if d.size else np.nan,
                   selected_unlabelled_beyond_300m_px=int((d > 0.3).sum()))
        out["localization_index"] = (out["selected_within_300m"] / out["selected_within_2km"]
                                     if out.get("selected_within_2km") else np.nan)
    return out


def unique_components(score, ref, faults, valid, budget, comp_lab, n_comp, radius_px=3, share=0.5) -> dict:
    """Mapped components recovered by the feature but not by the reference (both at the same budget)."""
    a = recovery(score, faults, valid, budget, comp_lab, n_comp, radius_px=radius_px, share=share)
    b = recovery(ref, faults, valid, budget, comp_lab, n_comp, radius_px=radius_px, share=share)
    A, B = set(a.get("_comp_recovered", []).tolist()), set(b.get("_comp_recovered", []).tolist())
    n = max(a.get("components", 0), 1)
    return {"feature_components": len(A), "reference_components": len(B), "both": len(A & B),
            "feature_only": len(A - B), "reference_only": len(B - A),
            "unique_share_of_components": len(A - B) / n, "marginal_component_recall": len(A - B) / n,
            "_feature_only_ids": sorted(A - B)}


def distance_response(x, dist_km, valid, faults, n_sample=200_000, seed=2026) -> pd.DataFrame:
    """Quantiles of ``x`` per distance band (bands sampled to n_sample px; on-trace band = mapped pixels)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float32)
    m = valid & np.isfinite(x)
    rows = []
    bands = [np.asarray(faults, bool)] + [(dist_km > lo) & (dist_km <= hi) for lo, hi in
                                          zip(DIST_EDGES_KM[:-1], DIST_EDGES_KM[1:])]
    for lab, sel in zip(DIST_LABELS, bands):
        v = x[m & sel]
        r = {"band": lab, "n": int(v.size)}
        if v.size > n_sample:
            v = rng.choice(v, n_sample, replace=False)
        if v.size:
            r.update(zip(("p05", "p25", "median", "p75", "p95"), np.percentile(v, [5, 25, 50, 75, 95]).tolist()))
        rows.append(r)
    T = pd.DataFrame(rows)
    far = T.iloc[-1]
    iqr = (far.get("p75", np.nan) - far.get("p25", np.nan))
    T["normalized_response"] = (T["median"] - far.get("median", np.nan)) / iqr if iqr and np.isfinite(iqr) else np.nan
    return T


def response_class(T: pd.DataFrame, min_effect=0.1) -> dict:
    """Trace-localising vs broad-corridor vs weak response from a distance-response table."""
    r = T.set_index("band").normalized_response
    near = r.reindex(DIST_LABELS[:4]).abs()
    mid = r.reindex(DIST_LABELS[4:7]).abs()
    peak = r.abs().idxmax() if r.notna().any() else None
    tr, co = float(np.nanmax(near)) if near.notna().any() else np.nan, float(np.nanmax(mid)) if mid.notna().any() \
        else np.nan
    if not np.isfinite(tr) or max(tr, co if np.isfinite(co) else 0) < min_effect:
        cls = "WEAK"
    elif peak in DIST_LABELS[:4] and tr >= (co if np.isfinite(co) else 0):
        cls = "LOCALIZES_TRACE (peak within 300 m)"
    else:
        cls = "BROAD_CORRIDOR (peak beyond 300 m)"
    return {"peak_band": peak, "trace_response": tr, "corridor_response": co, "response_class": cls}


def matched_controls(x, faults, covariate, valid, tile_px=50, n_bins=3, min_dist_px=10, dist_px=None) -> dict:
    """Cliff's delta of mapped-fault pixels vs unlabelled pixels >= min_dist_px from traces in the SAME tile and the
    SAME tercile of a label-free covariate; weighted by fault pixels per stratum. Sort-based (no per-stratum masks)."""
    x = np.asarray(x, np.float32)
    f = np.asarray(faults, bool)
    cov = np.asarray(covariate, np.float32)
    m = valid & np.isfinite(x) & np.isfinite(cov)
    d = dist_px if dist_px is not None else ndimage.distance_transform_edt(~f)
    q = np.nanquantile(cov[m], np.linspace(0, 1, n_bins + 1)[1:-1])
    rr, cc = np.nonzero(m)
    nbc = x.shape[1] // tile_px + 1
    strata = ((rr // tile_px) * nbc + cc // tile_px) * 10 + np.digitize(cov[rr, cc], q)
    isf = f[rr, cc]
    ctl = ~isf & (d[rr, cc] >= min_dist_px)
    vals = x[rr, cc]
    num = den = 0.0
    n_str = 0
    fs, cs = strata[isf], strata[ctl]
    fv, cv = vals[isf], vals[ctl]
    of, oc = np.argsort(fs, kind="stable"), np.argsort(cs, kind="stable")
    fs, fv, cs, cv = fs[of], fv[of], cs[oc], cv[oc]
    uf, sf = np.unique(fs, return_index=True)
    ef = np.r_[sf[1:], len(fs)]
    for s, a0, a1 in zip(uf, sf, ef):
        b0, b1 = np.searchsorted(cs, s, "left"), np.searchsorted(cs, s, "right")
        if a1 - a0 < 3 or b1 - b0 < 10:
            continue
        a, b = fv[a0:a1].astype(np.float64), cv[b0:b1].astype(np.float64)
        r = rankdata(np.r_[a, b])
        auc = (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        num += (2 * auc - 1) * len(a)
        den += len(a)
        n_str += 1
    return {"matched_delta": num / den if den else np.nan, "n_strata": n_str, "n_fault_matched": int(den)}


# =============================================================================== regional (sorted blocks)
def block_index(shape, tile_rows, tile_cols):
    r, c = np.indices(shape)
    nbc = -(-shape[1] // tile_cols)
    return (r // tile_rows) * nbc + c // tile_cols, -(-shape[0] // tile_rows), nbc


def regional(x, faults, valid, block_id, budget=0.01, min_pos=20, min_neg=200, global_direction="high",
             labels: dict | None = None) -> pd.DataFrame:
    """Per block: support, prevalence, AUROC (high), AP lift (global direction), recall@budget within the block,
    point-biserial, local polarity; INSUFFICIENT_DATA below min_pos / min_neg or without variation. Only blocks with
    valid pixels appear; insufficient blocks have NaN metrics (never 0)."""
    x = np.asarray(x, np.float32)
    m = valid & np.isfinite(x) & (block_id >= 0)
    b = block_id[m]
    v = x[m].astype(np.float64)
    y = np.asarray(faults, bool)[m]
    o = np.argsort(b, kind="stable")
    b, v, y = b[o], v[o], y[o]
    ub, st = np.unique(b, return_index=True)
    en = np.r_[st[1:], len(b)]
    rows = []
    for k, s0, s1 in zip(ub, st, en):
        vv, yy = v[s0:s1], y[s0:s1]
        p, n = int(yy.sum()), int(s1 - s0)
        r = {"block": int(k), "label": (labels or {}).get(int(k), str(int(k))), "n_valid": n, "n_fault_px": p,
             "prevalence": p / n}
        if p < min_pos or n - p < min_neg or vv.std() < 1e-12:
            r["status"] = INSUFFICIENT
        else:
            rk = rankdata(vv)
            a = (rk[yy].sum() - p * (p + 1) / 2) / (p * (n - p))
            sv = vv if global_direction == "high" else -vv
            kk = max(1, int(math.ceil(budget * n)))
            r.update(status="OK", auroc_high=float(a), polarity="high" if a >= 0.5 else "low",
                     auroc_global_direction=float(a if global_direction == "high" else 1 - a),
                     ap_lift_global_direction=_ap(sv, yy) / (p / n),
                     recall_at_budget=float(yy[_topk(sv, kk)].sum() / p),
                     point_biserial=float(np.corrcoef(vv, yy.astype(float))[0, 1]))
        rows.append(r)
    return pd.DataFrame(rows)


def block_map(T: pd.DataFrame, col: str, nbr: int, nbc: int) -> np.ndarray:
    Z = np.full((nbr, nbc), np.nan)
    if len(T) and col in T:
        ok = T[T.status == "OK"]
        Z.flat[ok.block.to_numpy()] = ok[col].to_numpy(float)
    return Z


# =============================================================================== held-out
def fold_masks(roles: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(development, evaluation) masks of one GeoFolds role band: dev = train (0) or inner_val (3); eval = val (1);
    buffers (2, 4) and invalid (-1) are excluded."""
    return valid & np.isin(roles, (0, 3)), valid & (roles == 1)


def heldout_feature(x, faults, dev, ev, budgets=(0.01,), near300=None) -> dict:
    """Direction frozen on ``dev`` (AUROC >= 0.5 -> high), metrics on ``ev`` only."""
    d = association_both(x, faults, dev, budgets=budgets)
    if d.get("status") != "OK":
        return {"status": INSUFFICIENT, "direction": None}
    direction = "high" if d["auroc_high"] >= 0.5 else "low"
    e = association_both(x, faults, ev, budgets=budgets, near300=near300)
    if e.get("status") != "OK":
        return {"status": INSUFFICIENT, "direction": direction, "dev_auroc": max(d["auroc_high"], d["auroc_low"])}
    out = {"status": "OK", "direction": direction, "dev_auroc": d[f"auroc_{direction}"],
           "eval_auroc": e[f"auroc_{direction}"], "eval_ap_lift": e[f"ap_lift_{direction}"],
           "eval_n_valid": e["n_valid"], "eval_n_fault_px": e["n_fault_px"]}
    for b in budgets:
        out[f"eval_recall_{b * 100:g}pct"] = e[f"recall_{b * 100:g}pct_{direction}"]
    return out


# =============================================================================== classification
def classify(row: dict, th: dict) -> tuple[str, str]:
    """PROMOTE / EXPERIMENTAL / REDUNDANT / REJECT from held-out, recovery, stability, redundancy and reliability.
    ``th``: thresholds (see DEFAULT_THRESHOLDS). Missing evidence never promotes."""
    why = []
    if row.get("numerically_reliable") is False:
        return "REJECT", "numerically unreliable (" + str(row.get("reliability_note")) + ")"
    folds_ok = row.get("heldout_folds_ci_above_half")
    n_folds = row.get("heldout_folds")
    stab = row.get("regional_same_polarity_share")
    uniq = row.get("unique_component_share_vs_best_reference")
    red = row.get("max_abs_spearman_vs_earlier")
    if n_folds and folds_ok is not None and folds_ok == 0 and (stab is None or stab < th["min_stability"]):
        return "REJECT", f"held-out CI includes 0.5 in every fold; regional stability {stab}"
    if red is not None and red >= th["redundancy_rho"] and (uniq is None or uniq < th["min_unique_share"]):
        return "REDUNDANT", f"|rho| {red:.2f} with {row.get('most_similar')} and unique recovery {uniq}"
    if (n_folds and folds_ok is not None and folds_ok >= math.ceil(th["fold_share"] * n_folds)
            and row.get("heldout_direction_consistent") and stab is not None and stab >= th["min_stability"]
            and uniq is not None and uniq >= th["min_unique_share"]):
        return "PROMOTE", (f"CI above 0.5 in {folds_ok}/{n_folds} folds, consistent polarity, stability {stab:.2f}, "
                           f"unique components {uniq:.3f}")
    if not n_folds:
        why.append("no held-out evaluation")
    if stab is None:
        why.append("no regional evaluation")
    return "EXPERIMENTAL", "; ".join(why) or "association present but generalisation / added recovery not established"


DEFAULT_THRESHOLDS = {"fold_share": 2 / 3, "min_stability": 0.6, "min_unique_share": 0.02, "redundancy_rho": 0.9}
