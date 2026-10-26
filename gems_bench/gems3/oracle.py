"""Candidate headroom ("oracle recall"): if we kept the ideal subset of candidates, how much could we recover?

The oracle READS THE UNIT'S TRUTH, so every number here is an upper bound for a ranker restricted to these
candidates - never a model score. Per unit:

* coverage of the full candidate set: weighted recall (TP / |G|), recall within 300 m, system recall (a fault system
  is recovered when >= ``system_recovered_frac`` of its scored pixels reach TP support >= ``system_recovered_support``),
  short-system recall (< ``short_system_km``), candidate density (km of candidate per 100 km^2);
* the unranked set rasterised as 1-px lines (all candidates = 1): its DTI;
* oracle DTI: candidates whose mean kernel weight k(dist to truth) >= t, for every t of ``kd_thresholds``;
* the same coverage numbers per source (provenance), so weak modalities are visible.

Go / no-go (``decision``): continue to the ranker only if pooled system recall and the oracle DTI gain over the
best cross-fitted representation (metric stage) both clear ``continue_if``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..metric import DTI_RADIUS_M
from .candidates import CandidateSet
from .official import Unit


def candidate_truth_stats(cs: CandidateSet, u: Unit) -> pd.DataFrame:
    """Per candidate: mean / max kernel weight to the unit's scored truth and share within 300 m."""
    G = u.eval_truth()
    if not G.any() or not len(cs):
        return pd.DataFrame({"kd_mean": np.zeros(len(cs)), "kd_max": np.zeros(len(cs)), "near300": np.zeros(len(cs)),
                             "dist_median_km": np.full(len(cs), np.inf)})
    d = ndimage.distance_transform_edt(~G, sampling=(100.0, 100.0))
    kd = np.clip(1.0 - d / DTI_RADIUS_M, 0, None)
    rows = []
    for p in cs.paths:
        k, dd = kd[p[:, 0], p[:, 1]], d[p[:, 0], p[:, 1]]
        rows.append({"kd_mean": float(k.mean()), "kd_max": float(k.max()), "near300": float((dd < DTI_RADIUS_M).mean()),
                     "dist_median_km": float(np.median(dd)) / 1000.0})
    return pd.DataFrame(rows)


def _systems(ctx, u: Unit):
    """System id of every scored truth pixel (FastScorer order) and the short-system flags."""
    gr, gc = u.scorer.g_rc
    sm = np.asarray(ctx.splits.system_map)[u.sl][gr, gc]
    tab = ctx.splits.systems.set_index("system_id")
    return sm, tab


def coverage(ctx, u: Unit, cs: CandidateSet, cfg: dict, values=None) -> dict:
    oc = cfg["oracle"]
    q = cs.raster(u.ev.shape, values)
    c = u.components(q)
    out = {"n_candidates": len(cs), **{k: v for k, v in c.as_dict().items() if k in ("dti", "tp_w", "fp_w", "fn_w",
                                                                                      "n_truth", "recall_w")}}
    G = u.eval_truth()
    if G.any() and (q > 0).any():
        dq = ndimage.distance_transform_edt(~(q > 0), sampling=(100.0, 100.0))
        out["recall@300m"] = float((dq[G] < DTI_RADIUS_M).mean())
    else:
        out["recall@300m"] = 0.0 if G.any() else np.nan
    sup = u.scorer.support(q)
    sm, tab = _systems(ctx, u)
    out.update(n_systems=0, system_recall=np.nan, n_short_systems=0, short_system_recall=np.nan)
    if len(sup):
        df = pd.DataFrame({"sys": sm, "hit": sup >= oc["system_recovered_support"]})
        df = df[df.sys >= 0]
        rec = df.groupby("sys").hit.mean() >= oc["system_recovered_frac"]
        km = tab.trace_km.reindex(rec.index)
        short = km < oc["short_system_km"]
        out.update(n_systems=int(len(rec)), system_recall=float(rec.mean()) if len(rec) else np.nan,
                   n_short_systems=int(short.sum()),
                   short_system_recall=float(rec[short].mean()) if short.any() else np.nan)
    ev_km2 = float(u.ev.sum()) * 0.01
    out["candidate_km_per_100km2"] = float(sum(len(p) for p in cs.paths) * 0.112 / max(ev_km2, 1e-9) * 100.0)
    return out


def unit_oracle(ctx, u: Unit, cs: CandidateSet, cfg: dict) -> dict[str, pd.DataFrame]:
    st = candidate_truth_stats(cs, u)
    base = {"task": u.task, "zone": u.zone}
    summary = [{**base, "subset": "all_unranked", **coverage(ctx, u, cs, cfg)}]
    curve = []
    for t in cfg["oracle"]["kd_thresholds"]:
        idx = np.flatnonzero(st.kd_mean.to_numpy() >= t)
        sub = cs.subset(idx)
        curve.append({**base, "kd_threshold": t, **coverage(ctx, u, sub, cfg)})
    by_src = []
    srcs = sorted({s for x in cs.table.get("sources", pd.Series(dtype=str)) for s in str(x).split(";") if s})
    for s in srcs:
        idx = [i for i, x in enumerate(cs.table.sources) if s in str(x).split(";")]
        sub = cs.subset(idx)
        good = np.flatnonzero(st.kd_mean.to_numpy()[idx] >= 0.3)
        by_src.append({**base, "source": s, "subset": "all", **coverage(ctx, u, sub, cfg)})
        by_src.append({**base, "source": s, "subset": "oracle_kd>=0.3", **coverage(ctx, u, sub.subset(good), cfg)})
    return {"summary": pd.DataFrame(summary), "curve": pd.DataFrame(curve), "by_source": pd.DataFrame(by_src),
            "stats": pd.concat([cs.table[["cand_id"]].reset_index(drop=True), st], axis=1).assign(**base)}


def pooled_oracle(curve: pd.DataFrame) -> pd.DataFrame:
    """Pooled DTI per (task, kd_threshold) over the task's units + the best threshold per task."""
    g = curve.groupby(["task", "kd_threshold"])[["tp_w", "fp_w", "fn_w", "n_truth"]].sum().reset_index()
    g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
    g["recall_w"] = g.tp_w / g.n_truth.clip(lower=1)
    rec = curve.groupby(["task", "kd_threshold"]).apply(
        lambda d: pd.Series({"system_recall": np.nansum(d.system_recall * d.n_systems) / max(d.n_systems.sum(), 1),
                             "short_system_recall": np.nansum(d.short_system_recall * d.n_short_systems)
                             / max(d.n_short_systems.sum(), 1)})).reset_index()
    return g.merge(rec, on=["task", "kd_threshold"])


def decision(pooled: pd.DataFrame, summary: pd.DataFrame, best_crossfit: dict, cfg: dict) -> pd.DataFrame:
    ci = cfg["oracle"]["continue_if"]
    rows = []
    for task, g in pooled.groupby("task"):
        top = g.loc[g.dti.idxmax()]
        allr = summary[summary.task == task]
        sys_all = float(np.nansum(allr.system_recall * allr.n_systems) / max(allr.n_systems.sum(), 1)) \
            if "system_recall" in allr else np.nan
        ref = best_crossfit.get(task, np.nan)
        gain = float(top.dti - ref) if np.isfinite(ref) else np.nan
        ok = (sys_all >= ci["system_recall"]) and (np.isfinite(gain) and gain >= ci["oracle_dti_gain"])
        rows.append({"task": task, "oracle_dti": float(top.dti), "oracle_kd_threshold": float(top.kd_threshold),
                     "system_recall_all_candidates": sys_all, "best_crossfit_representation_dti": ref,
                     "oracle_gain": gain, "decision": "CONTINUE" if ok else "STOP",
                     "rule": f"system_recall >= {ci['system_recall']} and oracle gain >= {ci['oracle_dti_gain']}",
                     "note": "oracle reads the unit's truth: an upper bound, not a score"})
    return pd.DataFrame(rows)
