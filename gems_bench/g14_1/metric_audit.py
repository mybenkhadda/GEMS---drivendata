"""DTI failure decomposition, failure classification and inner-validation methodology (Phases E3, F4, F5, H).

Metric vocabulary (kept distinct everywhere, see docs/g14_1/CORRECTED_EVALUATION.md):

``training surrogate``      ``g14.losses.dti_surrogate`` on patches (differentiable, loss only)
``patch diagnostic``        the same quantity reported per validation batch (G14's "inner surrogate"); diagnostic only
``region DTI``              official components (``gems_bench.metric`` / ``gems3.official``) pooled over a whole
                            evaluation region: TP, FP, FN are SUMMED first, the ratio is taken once
``leaderboard DTI``         the competition's number; never computed here

Failure classes F01-F10 are assigned only when the evidence fields they need are present; every assignment carries
its evidence values and the rule that fired. Rules and thresholds are explicit constants, not learned.

UNTESTED: written without execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

ALPHA, BETA = 0.2, 0.8

RULES = {
    "F01": "alpha*FP >= beta*FN and predicted area >= 10 x truth area (FP mass dominates the denominator)",
    "F02": ">= 30 % of predicted skeleton within 1 km of truth but < 50 % of that within 300 m (near but displaced)",
    "F03": ">= 50 % of the top-a area (before known masking) on or within 3 px of known faults",
    "F04": ">= 50 % of truth components >= 2 km have component recall < 0.05 (whole systems missed)",
    "F05": "fragmentation >= 20 components / 100 km and median component < 1 km",
    "F06": "largest predicted component >= 3 x the longest truth component and > 50 % of it farther than 1 km from truth",
    "F07": "FP pixels over-represented (>= 2 x base rate) on label-free edge / acquisition-boundary rasters",
    "F08": "a regime with >= 200 truth px scores < 0.5 x the pooled DTI",
    "F09": "submission parity failure (grid, shift, flip, binarisation, footprint) reported by geometry.submission_parity",
    "F10": "raw-score AUC >= 0.75 near truth but the chosen operating point reaches <= 0.5 x the best cross-fitted "
           "representation DTI (ranking present, representation / threshold loses it)",
}


# =============================================================================== decomposition
def _truth_components(truth: np.ndarray, km: float):
    from ..geology.skeleton import skeletonize

    lab, n = ndimage.label(truth, structure=np.ones((3, 3), bool))
    if n == 0:
        return lab, np.zeros(0)
    sk = skeletonize(truth)
    length = ndimage.sum(sk.astype(np.float32), lab, np.arange(1, n + 1)) * km
    return lab, np.asarray(length)


def decompose_unit(u, q_crop: np.ndarray, km: float = 0.1, long_km: float = 5.0) -> dict:
    """Components and geometry of one representation on one unit (q_crop: the prediction raster on the unit crop)."""
    from ..geology.skeleton import skeletonize

    c = u.components(q_crop)
    sc = u.scorer
    p = np.asarray(q_crop, np.float64).copy()
    p[sc.zero] = 0
    sup = sc.support(p, cleaned=True)
    truth = u.eval_truth()
    pos = (p > 0) & u.ev
    sk = skeletonize(pos)
    lab, n = ndimage.label(pos, structure=np.ones((3, 3), bool))
    comp_km = ndimage.sum(sk.astype(np.float32), lab, np.arange(1, n + 1)) * km if n else np.zeros(0)
    out = {"task": u.task, "zone": u.zone, "dti": c.dti, "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn,
           "n_truth": c.n_truth, "recall_w": c.tp / max(c.n_truth, 1), "precision_w": c.tp / max(c.tp + c.fp, 1e-12),
           "alpha_fp": ALPHA * c.fp, "beta_fn": BETA * c.fn, "pred_mass": c.pred_mass, "pred_px": int(pos.sum()),
           "pred_frac": float(pos.sum() / max(u.scored_mask().sum(), 1)), "n_components": int(n),
           "skeleton_km": float(sk.sum() * km), "median_component_km": float(np.median(comp_km)) if n else np.nan,
           "largest_component_km": float(comp_km.max()) if n else 0.0,
           "fragmentation_per_100km": float(n / max(sk.sum() * km, 1e-9) * 100) if n else np.nan}
    if truth.any():
        dt = ndimage.distance_transform_edt(~truth) * km
        dp = ndimage.distance_transform_edt(~pos) * km if pos.any() else np.full(truth.shape, np.inf)
        near = dt[sk] <= 1.0
        out.update(skeleton_within_1km=float(near.mean()) if sk.any() else np.nan,
                   skeleton_within_300m=float((dt[sk] <= 0.3).mean()) if sk.any() else np.nan,
                   loc_err_pred_mean_km=float(dt[sk][near].mean()) if near.any() else np.nan,
                   loc_err_pred_median_km=float(np.median(dt[sk][near])) if near.any() else np.nan,
                   loc_err_truth_mean_km=float(np.minimum(dp[truth], 5).mean()),
                   loc_err_truth_median_km=float(np.median(np.minimum(dp[truth], 5))))
        tl, tlen = _truth_components(truth, km)
        gr, gc = sc.g_rc
        ids = tl[gr, gc]
        if len(tlen):
            rec = np.array([sup[ids == i + 1].mean() if (ids == i + 1).any() else np.nan for i in range(len(tlen))])
            longm = tlen >= long_km
            out.update(n_truth_components=int(len(tlen)), longest_truth_km=float(tlen.max()),
                       long_component_recall=float(np.nanmean(rec[longm])) if longm.any() else np.nan,
                       share_components_ge2km_missed=float(np.nanmean(rec[tlen >= 2] < 0.05)) if (tlen >= 2).any() else np.nan,
                       largest_comp_far_share=_far_share(lab, comp_km, dt) if n else np.nan)
    u.release()
    return out


def _far_share(lab, comp_km, dt) -> float:
    i = int(np.argmax(comp_km)) + 1
    m = lab == i
    return float((dt[m] > 1.0).mean())


def decompose(maps: dict, units: list, choice: dict, km: float = 0.1, order: str = "pixel_exact") -> pd.DataFrame:
    """Decomposition at the chosen operating point of each unit. ``choice``: {(task, zone): (rep, frac)}."""
    from .representations import apply_rep

    rows = []
    for u in units:
        k = (u.task, u.zone)
        if k not in maps or k not in choice:
            continue
        rep, fr = choice[k]
        s = np.asarray(u.crop(maps[k]), np.float32)
        q, acc = apply_rep(rep, s, u.ev, u.exclude, u.scored_mask(), None if fr is None or fr < 0 else fr, order)
        rows.append({**decompose_unit(u, q, km), "rep": rep, "frac": fr, **{f"acc_{a}": b for a, b in acc.items()
                                                                          if a not in ("rep", "order")}})
    return pd.DataFrame(rows)


# =============================================================================== failure classification
def classify(row: dict, map_level: dict | None = None, parity: dict | None = None, ranking: dict | None = None,
             best_rep_dti: float | None = None, regime_rows: pd.DataFrame | None = None,
             edge_overlap: dict | None = None) -> list[dict]:
    """Failure classes with evidence for one (model, unit) decomposition row. Missing evidence -> class not assessed."""
    out = []

    def add(code, fired, **ev):
        out.append({"class": code, "status": "FIRED" if fired else "not_fired", "rule": RULES[code], **ev})

    def skip(code, why):
        out.append({"class": code, "status": "not_assessed", "rule": RULES[code], "why": why})

    if {"alpha_fp", "beta_fn", "pred_px", "n_truth"} <= set(row):
        add("F01", row["alpha_fp"] >= row["beta_fn"] and row["pred_px"] >= 10 * max(row["n_truth"], 1),
            alpha_fp=row["alpha_fp"], beta_fn=row["beta_fn"], area_ratio=row["pred_px"] / max(row["n_truth"], 1))
    if row.get("skeleton_within_1km") is not None and not np.isnan(row.get("skeleton_within_1km", np.nan)):
        w1, w3 = row["skeleton_within_1km"], row["skeleton_within_300m"]
        add("F02", w1 >= 0.3 and w3 < 0.5 * w1, within_1km=w1, within_300m=w3)
    else:
        skip("F02", "no truth or empty prediction")
    if map_level and "top_share_near_known" in map_level:
        add("F03", map_level["top_share_near_known"] >= 0.5, top_share_near_known=map_level["top_share_near_known"])
    else:
        skip("F03", "map-level known-fault proximity not supplied")
    if row.get("share_components_ge2km_missed") is not None and not np.isnan(row.get("share_components_ge2km_missed", np.nan)):
        add("F04", row["share_components_ge2km_missed"] >= 0.5, share_missed=row["share_components_ge2km_missed"])
    else:
        skip("F04", "no truth component >= 2 km")
    if not np.isnan(row.get("fragmentation_per_100km", np.nan)):
        add("F05", row["fragmentation_per_100km"] >= 20 and (row.get("median_component_km") or 0) < 1,
            fragmentation=row["fragmentation_per_100km"], median_component_km=row.get("median_component_km"))
    if row.get("longest_truth_km") and not np.isnan(row.get("largest_comp_far_share", np.nan)):
        add("F06", row["largest_component_km"] >= 3 * row["longest_truth_km"] and row["largest_comp_far_share"] > 0.5,
            largest_component_km=row["largest_component_km"], longest_truth_km=row["longest_truth_km"],
            far_share=row["largest_comp_far_share"])
    else:
        skip("F06", "no truth components")
    if edge_overlap:
        fired = any(v.get("enrichment", 0) >= 2 for v in edge_overlap.values())
        add("F07", fired, **{f"{k}_enrichment": v.get("enrichment") for k, v in edge_overlap.items()})
    else:
        skip("F07", "no label-free edge / boundary rasters supplied")
    if regime_rows is not None and len(regime_rows):
        pooled = row.get("dti", np.nan)
        bad = regime_rows[(regime_rows.n_truth >= 200) & (regime_rows.dti < 0.5 * pooled)]
        add("F08", len(bad) > 0, regimes=",".join(str(int(r)) for r in bad.regime), pooled_dti=pooled)
    else:
        skip("F08", "no regime table")
    if parity:
        fails = [k for k in ("binary",) if parity.get(k)] + (["grid"] if parity.get("same_grid") != "SAME_GRID" else []) \
            + (["shift"] if abs(parity.get("shift_dy_px", 0)) > 0.25 or abs(parity.get("shift_dx_px", 0)) > 0.25 else []) \
            + (["d4"] if parity.get("best_d4_transform", "identity") != "identity" else []) \
            + (["footprint"] if parity.get("nan_inside_template", 0) or parity.get("finite_outside_template", 0) else [])
        add("F09", bool(set(fails) - {"binary"}), parity_issues=",".join(fails))
    else:
        skip("F09", "no parity report")
    if ranking and best_rep_dti is not None and not np.isnan(ranking.get("auc", np.nan)):
        add("F10", ranking["auc"] >= 0.75 and row.get("dti", 0) <= 0.5 * best_rep_dti, auc=ranking["auc"],
            chosen_dti=row.get("dti"), best_rep_dti=best_rep_dti)
    else:
        skip("F10", "ranking or representation sweep not supplied")
    return out


def edge_overlap(fp_mask: np.ndarray, rasters: dict[str, np.ndarray], valid: np.ndarray, q: float = 0.9) -> dict:
    """Enrichment of false-positive pixels on the top decile of label-free rasters (terrain edges, coverage
    boundaries...): P(raster high | FP) / P(raster high)."""
    out = {}
    for k, r in rasters.items():
        a = np.asarray(r, np.float32)
        m = valid & np.isfinite(a)
        if not m.any() or not (fp_mask & m).any():
            continue
        thr = np.quantile(a[m], q)
        hi = m & (a >= thr)
        base = hi[m].mean()
        out[k] = {"enrichment": float(hi[fp_mask & m].mean() / max(base, 1e-9)), "fp_px": int((fp_mask & m).sum())}
    return out


def coverage_boundary(valid: np.ndarray, band_valid: dict[str, np.ndarray], width_px: int = 2) -> np.ndarray:
    """Label-free raster: 1 within ``width_px`` of any per-band coverage edge (survey / acquisition boundaries)."""
    edge = np.zeros(valid.shape, bool)
    for m in band_valid.values():
        e = m ^ ndimage.binary_erosion(m)
        edge |= ndimage.binary_dilation(e, iterations=width_px)
    return (edge & valid).astype(np.float32)


# =============================================================================== inner-validation methodology
def ratio_of_sums_vs_mean_of_ratios(batches: pd.DataFrame) -> dict:
    """``batches``: rows with tp_w, fp_w, fn_w per batch / patch. The G14 inner number is the mean of per-batch
    ratios; the region DTI is the ratio of summed components."""
    r = batches.tp_w / (batches.tp_w + ALPHA * batches.fp_w + BETA * batches.fn_w + 1e-9)
    pooled = batches.tp_w.sum() / (batches.tp_w.sum() + ALPHA * batches.fp_w.sum() + BETA * batches.fn_w.sum() + 1e-9)
    return {"mean_of_ratios": float(r.mean()), "ratio_of_sums": float(pooled), "n": int(len(batches)),
            "share_batches_without_truth": float((batches.get("n_truth", batches.tp_w + batches.fn_w) == 0).mean())}


def region_components(pred_full: np.ndarray, truth: np.ndarray, region: np.ndarray, exclude: np.ndarray | None,
                      pixel_m: tuple, exclude_supports_tp: bool = False):
    """Official components over a whole region (one ratio). ``exclude``: pixel-exact known / context faults."""
    from ..metric import MaskedScorer

    sc = MaskedScorer(truth, region, exclude, pixel_m=pixel_m, exclude_supports_tp=exclude_supports_tp)
    return sc.components(np.nan_to_num(np.asarray(pred_full, np.float64)))


def tiles(region: np.ndarray, tile_px: int, min_frac: float = 0.5) -> list[tuple[int, int]]:
    """Deterministic non-overlapping tiles with >= min_frac of their pixels in ``region``."""
    H, W = region.shape
    out = []
    for t in range(0, H, tile_px):
        for l in range(0, W, tile_px):
            if region[t:t + tile_px, l:l + tile_px].mean() >= min_frac:
                out.append((t, l))
    return out


def tiled_components(pred_full: np.ndarray, truth: np.ndarray, region: np.ndarray, exclude: np.ndarray | None,
                     pixel_m: tuple, tile_px: int = 128) -> pd.DataFrame:
    """Per-tile official components (tile = evaluation mask), for ratio-of-sums vs mean-of-ratios comparisons and
    uniform (non-enriched) patch statistics."""
    from ..metric import dti_components

    rows = []
    for t, l in tiles(region, tile_px):
        ev = np.zeros(region.shape, bool)
        ev[t:t + tile_px, l:l + tile_px] = True
        ev &= region
        c = dti_components(np.nan_to_num(pred_full), truth, eval_mask=ev, exclude_mask=exclude, pixel_m=pixel_m,
                           exclude_supports_tp=False)
        rows.append({"top": t, "left": l, "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn, "n_truth": c.n_truth,
                     "dti": c.dti})
    return pd.DataFrame(rows)


def recall_by_known_distance(u, q_crop: np.ndarray, context_crop: np.ndarray, km: float = 0.1,
                             bins_km=(0, 1, 2, 5, 10, 20, np.inf)) -> pd.DataFrame:
    """Weighted recall of held-out truth binned by its distance to the nearest permitted known fault (D4)."""
    sc = u.scorer
    sup = sc.support(np.asarray(q_crop, np.float64))
    gr, gc = sc.g_rc
    d = (ndimage.distance_transform_edt(~context_crop) * km if context_crop.any()
         else np.full(context_crop.shape, np.inf))[gr, gc]
    rows = []
    for lo, hi in zip(bins_km[:-1], bins_km[1:]):
        m = (d >= lo) & (d < hi)
        rows.append({"task": u.task, "zone": u.zone, "dist_bin_km": lo, "n_truth": int(m.sum()),
                     "tp_w": float(sup[m].sum()), "recall_w": float(sup[m].mean()) if m.any() else np.nan})
    u.release()
    return pd.DataFrame(rows)


def reliability(u, s_crop: np.ndarray, n_bins: int = 15, tol_px: int = 3) -> pd.DataFrame:
    """Mean predicted p per quantile bin vs share of pixels within ``tol_px`` of held-out truth (calibration curve)."""
    truth = u.eval_truth()
    near = ndimage.distance_transform_edt(~truth) <= tol_px if truth.any() else np.zeros(truth.shape, bool)
    m = u.scored_mask() & np.isfinite(s_crop)
    s, y = np.asarray(s_crop, np.float64)[m], near[m]
    if not s.size:
        return pd.DataFrame()
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    df = pd.DataFrame({"b": b, "s": s, "y": y}).groupby("b").agg(mean_p=("s", "mean"), observed=("y", "mean"),
                                                                  n=("s", "size")).reset_index()
    df.insert(0, "zone", u.zone)
    df.insert(0, "task", u.task)
    return df


def ranking_consistency(df: pd.DataFrame, inner_cols: list[str], external_cols: list[str]) -> pd.DataFrame:
    """Spearman and Kendall rank correlations between inner metrics and external (held-out) metrics across
    experiments / checkpoints (Phase F5). Needs >= 3 rows to be meaningful; reported regardless with n."""
    from scipy.stats import kendalltau, spearmanr

    rows = []
    for a in inner_cols:
        for b in external_cols:
            t = df[[a, b]].dropna()
            if len(t) < 3:
                rows.append({"inner": a, "external": b, "n": len(t), "spearman": np.nan, "kendall": np.nan})
                continue
            rows.append({"inner": a, "external": b, "n": len(t), "spearman": float(spearmanr(t[a], t[b])[0]),
                         "kendall": float(kendalltau(t[a], t[b])[0])})
    return pd.DataFrame(rows)
