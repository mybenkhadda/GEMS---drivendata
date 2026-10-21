"""Pseudo-discovery re-scoring from the saved Phase-2 discovery predictions (no retraining).

For split d: truth = withheld systems, evaluation zone = within 10 km of them. Three policies for
the *retained* (training-visible) faults, which mirror the unknown hidden-test convention:

``excl``         retained faults dilated 300 m are excluded (neither TP nor FP)          [Phase-2 'excl']
``retained_fp``  retained faults are not truth; predictions on them are FP               [NEW: hidden-test analogue
                                                                                          if known faults are not excluded]
``all_truth``    retained faults are truth too                                            [Phase-2 'no_exclusion']

Prediction variants: ``model`` (as predicted) and ``suppress_known`` (prediction set to 0 within
``suppress_m`` of retained faults) — the latter is the candidate submission policy if the
leaderboard counts known faults as FP.

All comparisons use binary top-a masks of the *evaluation zone* (matched area), so G1's
"recall 0.96" (a >= 0.5 map over most of the zone) can be compared fairly with G3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .context import P3Context
from .curves import DEFAULT_AREA_FRACS, BinaryCurve, crossfit_select, pooled

POLICIES = ("excl", "retained_fp", "all_truth")


def _dilate(mask: np.ndarray, metres: float, pixel_m=(100.0, 100.0)) -> np.ndarray:
    return ndimage.distance_transform_edt(~mask, sampling=pixel_m) <= metres


def split_masks(ctx: P3Context, d: int) -> dict:
    s, lab, v = ctx.splits, ctx.data.labels, ctx.data.valid
    G = s.withheld(d)
    retained = lab & ~G
    ez = s.eval_zone(d) & v
    return {"G": G, "retained": retained, "ez": ez, "excl": _dilate(retained, 300.0) & ~G,
            "far": ctx.dist_to_labels_km() > 1.0}


def policy_truth_exclude(m: dict, policy: str):
    if policy == "excl":
        return m["G"], m["excl"]
    if policy == "retained_fp":
        return m["G"], None
    if policy == "all_truth":
        return m["G"] | (m["retained"] & m["ez"]), None
    raise KeyError(policy)


def suppress_known(pred: np.ndarray, retained: np.ndarray, suppress_m: float = 300.0) -> np.ndarray:
    q = pred.copy()
    q[_dilate(retained, suppress_m) & np.isfinite(q)] = 0.0
    return q


def discovery_curves(ctx: P3Context, models, fracs=DEFAULT_AREA_FRACS, policies=POLICIES,
                     variants=("model", "suppress_known"), extra_preds: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Long table (split, model, policy, variant, target_frac, components) + the BinaryCurve objects."""
    rows, objs = [], {}
    for d in range(ctx.splits.n_splits):
        m = split_masks(ctx, d)
        for model in models:
            p = (extra_preds or {}).get((model, d))
            p = ctx.discovery_pred(model, d) if p is None else p
            if p is None:
                continue
            for var in variants:
                q = p if var == "model" else suppress_known(p, m["retained"])
                for pol in policies:
                    truth, ex = policy_truth_exclude(m, pol)
                    c = BinaryCurve(q, truth, m["ez"], ex, far_mask=m["far"])
                    df = c.at_area(fracs)
                    df.insert(0, "variant", var)
                    df.insert(0, "policy", pol)
                    df.insert(0, "model", model)
                    df.insert(0, "zone", d)
                    rows.append(df)
                    objs[(d, model, pol, var)] = c
    return pd.concat(rows, ignore_index=True), objs


def discovery_summary(curves: pd.DataFrame, matched=(0.01, 0.02, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Pooled-over-splits matched-budget rows + cross-fitted area (chosen on 2 splits, scored on the 3rd)."""
    pool = pooled(curves, ["model", "policy", "variant", "target_frac"])
    out = pool[pool.target_frac.isin(matched)].copy()
    out["row"] = "matched_budget"
    cf = []
    for (m, pol, var), g in curves.groupby(["model", "policy", "variant"]):
        per, s = crossfit_select(g, ["target_frac"])
        cf.append({"model": m, "policy": pol, "variant": var, "row": "crossfit_area", "dti": s["dti_crossfit"],
                   "dti_insample_best": s["dti_insample"], "fold_min": s["fold_min"], "fold_max": s["fold_max"],
                   "target_frac": s["params_all"]["target_frac"]})
    return pd.concat([out, pd.DataFrame(cf)], ignore_index=True)


def systems_at_budget(ctx: P3Context, objs: dict, frac: float, policy: str = "excl", variant: str = "model") -> pd.DataFrame:
    """Per withheld system: exact pixel count, weighted recall and recall@300 at a matched area budget."""
    rows = []
    for (d, model, pol, var), c in objs.items():
        if pol != policy or var != variant or c.n == 0:
            continue
        thr = float(c.threshold_for_area([frac])[0])
        rc = c.truth_rc
        sid = ctx.splits.system_map[rc[:, 0], rc[:, 1]]
        df = pd.DataFrame({"system_id": sid, "tp": c.per_truth_tp(thr), "hit300": c.per_truth_hit(thr, 300.0),
                           "hit100": c.per_truth_hit(thr, 100.0)})
        g = df.groupby("system_id").agg(n_px=("tp", "size"), weighted_recall=("tp", "mean"),
                                         recall_300m=("hit300", "mean"), recall_100m=("hit100", "mean")).reset_index()
        g["model"], g["split"], g["frac"], g["policy"], g["variant"] = model, d, frac, pol, var
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    sysattr = ctx.splits.systems.rename(columns={"n_px": "system_n_px"})
    out = out.merge(sysattr, on="system_id", how="left")
    out["length_bin"] = pd.cut(out.trace_km, [0, 10, 20, 50, np.inf], labels=["5-10", "10-20", "20-50", ">50"])
    return out


def sensitivity_exact(per_sys: pd.DataFrame) -> pd.DataFrame:
    """Recall by length bin and region, weighted by the exact number of scored withheld pixels."""
    rows = []
    for col in ("length_bin", "region_name"):
        for (m, val), g in per_sys.groupby(["model", col], observed=True):
            w = g.n_px.astype(float)
            rows.append({"model": m, "stratum": col, "value": str(val), "n_systems": len(g), "n_px": int(w.sum()),
                         "trace_km": float(g.trace_km.sum()), "weighted_recall": float(np.average(g.weighted_recall, weights=w)),
                         "recall@300m": float(np.average(g.recall_300m, weights=w)), "weight_basis": "n_px_exact"})
    return pd.DataFrame(rows)


def unique_recoveries(per_sys: pd.DataFrame, a: str, b: str, margin: float = 0.3, floor: float = 0.5) -> pd.DataFrame:
    """Systems recovered by ``a`` (recall >= floor) and missed by ``b`` by at least ``margin`` at the same budget."""
    w = per_sys.pivot_table(index=["split", "system_id"], columns="model", values="weighted_recall").reset_index()
    if a not in w or b not in w:
        return pd.DataFrame()
    w["delta"] = w[a] - w[b]
    sel = w[(w[a] >= floor) & (w.delta >= margin)]
    return sel.merge(per_sys[["split", "system_id", "trace_km", "region_name", "n_px"]].drop_duplicates(),
                     on=["split", "system_id"], how="left").sort_values("delta", ascending=False)


def phase2_sensitivity_exact(ctx: P3Context, models=("G1", "G2", "G3")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run the Phase-2 scoring (CV transform, threshold 0.5) with the fixed exact-n_px weighting."""
    from .. import discovery as disc
    from ..ensemble import apply_transform_file
    from ..submission import load_transform

    summ, per = [], []
    for d in range(ctx.splits.n_splits):
        for m in models:
            p = ctx.discovery_pred(m, d)
            if p is None:
                continue
            if m != "B0_lineament":
                p = apply_transform_file(p, load_transform(ctx.phase2_run(m).dir))
            s, ps = disc.evaluate_split(p, ctx.data, ctx.splits, d, m, threshold=ctx.cfg.report_threshold)
            summ.append(s)
            per.append(ps)
    per = pd.concat(per, ignore_index=True)
    sens = disc.sensitivity_tables(per, ctx.seg, ctx.seg_tab, ctx.splits)
    sens["weight_basis"] = "n_px_exact"
    return pd.DataFrame(summ), sens
