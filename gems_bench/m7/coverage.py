"""Section 04: candidate oracle and discovery-coverage audit (THE gate before any ranker).

Everything here READS THE UNIT'S TRUTH, for diagnosis only; nothing computed here is deployable, and no generator
parameter is tuned from it (the generator configuration is fixed before this section runs).

Definitions (per unit, then pooled over the units of a task):

* coverage of a candidate subset = the subset rasterised as 1-px lines (value 1) and scored with the official
  scorer: weighted recall (TP / |G|), recall within 300 m, system recall (a withheld system is covered when >=
  ``system_recovered_frac`` of its scored pixels get TP support >= ``system_recovered_support``), short-system recall;
* recall-density curve: candidates ordered by the LABEL-FREE structural score, cumulated until the candidate length
  density (km per 100 km^2 of evaluation zone) reaches each cap -> coverage at each cap;
* oracle DTI (upper bound conditional on this pool): the subset of candidates whose mean kernel weight
  k(d) = max(1 - d / 300 m, 0) to the scored truth is >= t, rasterised as 1-px lines; the best t per unit is reported
  (in-sample by construction: the oracle reads the truth);
* incremental coverage: among systems missed by BOTH B0 and B1 (section 02), the share the pool covers.

Gate (``oracle.go_if``): pooled system recall at ``max_density_km_per_100km2`` >= ``system_recall`` AND (incremental
coverage of B0+B1-missed systems >= ``incremental_recall_missed_by_both`` OR oracle DTI gain over the best section-01
baseline >= ``oracle_dti_gain``). The recorded earlier generator covered ~12.5 % of withheld systems.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..gems3.oracle import candidate_truth_stats, coverage
from .protocol import system_hits


def _ocfg(cfg: dict) -> dict:
    ev = cfg["evaluation"]
    return {"oracle": {"system_recovered_support": ev["system_recovered_support"],
                       "system_recovered_frac": ev["system_recovered_frac"], "short_system_km": ev["short_system_km"]}}


def density_order(cs) -> np.ndarray:
    t = cs.table
    s = t["structural_score"].to_numpy() if "structural_score" in t else np.zeros(len(t))
    return np.argsort(-np.nan_to_num(s, nan=-1.0), kind="stable")


def density_subset(cs, u, cap_km_per_100km2: float):
    order = density_order(cs)
    km = np.array([len(cs.paths[i]) * 0.112 for i in order])
    ev_km2 = float(u.ev.sum()) * 0.01
    keep = order[np.cumsum(km) <= cap_km_per_100km2 * ev_km2 / 100.0]
    return cs.subset(sorted(keep.tolist()))


def unit_audit(m7, u, cs) -> dict:
    ctx, cfg = m7.ctx, m7.cfg
    oc = cfg["oracle"]
    oo = _ocfg(cfg)
    base = {"task": u.task, "zone": u.zone}
    summary = [{**base, "subset": "all_candidates", **coverage(ctx, u, cs, oo)}]
    curve = []
    for cap in oc["density_caps_km_per_100km2"]:
        sub = density_subset(cs, u, float(cap))
        curve.append({**base, "density_cap": float(cap), **coverage(ctx, u, sub, oo)})
    st = candidate_truth_stats(cs, u)
    orc = []
    for t in oc["kd_thresholds"]:
        sub = cs.subset(np.flatnonzero(st.kd_mean.to_numpy() >= t).tolist())
        orc.append({**base, "kd_threshold": float(t), **coverage(ctx, u, sub, oo)})
    by = []
    for col in ("source", "modality", "kind", "evidence_class"):
        if col not in cs.table:
            continue
        for val in sorted(cs.table[col].dropna().astype(str).unique()):
            idx = np.flatnonzero(cs.table[col].astype(str).to_numpy() == val)
            by.append({**base, "group": col, "value": val, **coverage(ctx, u, cs.subset(idx.tolist()), oo)})
    # provenance-wide: a candidate counts for every source that produced a duplicate of it
    if "sources" in cs.table:
        fams = sorted({s.split("@")[0] for x in cs.table.sources for s in str(x).split(";") if s})
        for f in fams:
            idx = [i for i, x in enumerate(cs.table.sources) if f in {s.split("@")[0] for s in str(x).split(";")}]
            by.append({**base, "group": "provenance", "value": f, **coverage(ctx, u, cs.subset(idx), oo)})
    cap_sub = density_subset(cs, u, float(oc["max_density_km_per_100km2"]))
    sys_all = system_hits(ctx, u, cs.raster(u.ev.shape), cfg).assign(subset="all_candidates")
    sys_cap = system_hits(ctx, u, cap_sub.raster(u.ev.shape), cfg).assign(subset="at_max_density")
    systems = pd.concat([sys_all, sys_cap], ignore_index=True).assign(**base)
    return {"summary": pd.DataFrame(summary), "curve": pd.DataFrame(curve), "oracle": pd.DataFrame(orc),
            "by_group": pd.DataFrame(by), "systems": systems,
            "truth_stats": pd.concat([cs.table[["cand_id"]].reset_index(drop=True), st], axis=1).assign(**base)}


def pooled(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Pooled components + system-weighted recalls over the units of each task."""
    if not len(df):
        return df
    g = df.groupby(keys)[["tp_w", "fp_w", "fn_w", "n_truth"]].sum().reset_index()
    g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
    g["recall_w"] = g.tp_w / g.n_truth.clip(lower=1)
    w = df.assign(_s=df.system_recall * df.n_systems, _ss=df.short_system_recall.fillna(0) * df.n_short_systems,
                  _d=df.candidate_km_per_100km2 * df.n_truth.clip(lower=1))
    agg = w.groupby(keys).agg(n_systems=("n_systems", "sum"), _s=("_s", "sum"), n_short=("n_short_systems", "sum"),
                              _ss=("_ss", "sum"), _d=("_d", "sum"), _n=("n_truth", "sum"),
                              recall_300m=("recall@300m", "mean")).reset_index()
    agg["system_recall"] = agg._s / agg.n_systems.clip(lower=1)
    agg["short_system_recall"] = agg._ss / agg.n_short.clip(lower=1)
    agg["density_km_per_100km2"] = agg._d / agg._n.clip(lower=1)
    return g.merge(agg.drop(columns=["_s", "_ss", "_d", "_n"]), on=keys)


def incremental(systems: pd.DataFrame, s02_systems: pd.DataFrame) -> pd.DataFrame:
    """Per task / subset: coverage among systems in each B0 / B1 class (section 02), esp. 'neither'."""
    if not len(systems) or not len(s02_systems):
        return pd.DataFrame()
    m = systems.merge(s02_systems[["task", "zone", "system_id", "class"]], on=["task", "zone", "system_id"], how="left")
    m["class"] = m["class"].fillna("unclassified")
    g = m.groupby(["task", "subset", "class"]).agg(n_systems=("recovered", "size"),
                                                  covered=("recovered", "sum")).reset_index()
    g["coverage"] = g.covered / g.n_systems.clip(lower=1)
    return g


def by_stratum(systems: pd.DataFrame, attrs: pd.DataFrame, strata=("length_bin", "strike_sector", "topology",
               "magnetic_gradient_tercile", "gravity_gradient_tercile", "terrain_slope_tercile", "region_name",
               "low_geophysical_contrast", "near_parallel")) -> pd.DataFrame:
    if not len(systems) or not len(attrs):
        return pd.DataFrame()
    m = systems.merge(attrs, on="system_id", how="left")
    rows = []
    for (task, sub), g in m.groupby(["task", "subset"]):
        for st in strata:
            if st not in g:
                continue
            for val, gg in g.groupby(st):
                rec = gg.recovered.astype(bool)
                rows.append({"task": task, "subset": sub, "stratum": st, "value": str(val), "n_systems": len(gg),
                             "system_recall": float(rec.mean()),
                             "length_recall": float(gg.trace_km[rec].sum() / max(gg.trace_km.sum(), 1e-9))})
    return pd.DataFrame(rows)


def decision(m7, pooled_curve: pd.DataFrame, pooled_oracle: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    cfg = m7.cfg
    oc, gi = cfg["oracle"], cfg["oracle"]["go_if"]
    prim = cfg["evaluation"]["primary_task"]
    base = m7.best_baseline_dti(prim)
    ref = float((cfg.get("recorded_reference") or {}).get("candidate_system_recall", np.nan))
    rows = []
    for task in sorted(set(pooled_curve.task)) if len(pooled_curve) else []:
        pc = pooled_curve[pooled_curve.task == task]
        at = pc[np.isclose(pc.density_cap, float(oc["max_density_km_per_100km2"]))]
        sr = float(at.system_recall.iloc[0]) if len(at) else np.nan
        po = pooled_oracle[pooled_oracle.task == task] if len(pooled_oracle) else pooled_oracle
        odti = float(po.dti.max()) if len(po) else np.nan
        bdti = base if task == prim else m7.best_baseline_dti(task)
        gain = odti - bdti if np.isfinite(odti) and np.isfinite(bdti) else np.nan
        ic = inc[(inc.task == task) & (inc.subset == "at_max_density") & (inc["class"] == "neither")] if len(inc) else inc
        incr = float(ic.coverage.iloc[0]) if len(ic) else np.nan
        ok_sr = np.isfinite(sr) and sr >= float(gi["system_recall"])
        ok_inc = np.isfinite(incr) and incr >= float(gi["incremental_recall_missed_by_both"])
        ok_gain = np.isfinite(gain) and gain >= float(gi["oracle_dti_gain"])
        go = ok_sr and (ok_inc or ok_gain)
        rows.append({"task": task, "system_recall_at_max_density": sr, "reference_system_recall": ref,
                     "coverage_of_missed_by_both": incr, "oracle_dti": odti, "best_baseline_dti_crossfit": bdti,
                     "oracle_gain": gain, "pass_system_recall": ok_sr, "pass_incremental": ok_inc,
                     "pass_oracle_gain": ok_gain, "decision": "GO" if go else "NO_GO",
                     "rule": (f"system_recall >= {gi['system_recall']} AND (missed-by-both coverage >= "
                              f"{gi['incremental_recall_missed_by_both']} OR oracle gain >= {gi['oracle_dti_gain']})"),
                     "note": "oracle numbers read the truth: diagnostic upper bounds, never a score"})
    return pd.DataFrame(rows)


def miss_diagnostics(systems: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """For systems the pool misses: which 100 m modality carries signal (percentile >= 0.8) - a hint of which
    modality might recover them - or none (candidates for DEM / other data)."""
    if not len(systems) or not len(attrs):
        return pd.DataFrame()
    m = systems[(systems.subset == "all_candidates") & ~systems.recovered.astype(bool)].merge(attrs, on="system_id",
                                                                                              how="left")
    pct = [c for c in ("magnetic_gradient_pct", "gravity_gradient_pct", "terrain_gradient_pct", "terrain_slope_pct",
                       "crossmodal_agreement_pct") if c in m]
    rows = []
    for r in m.itertuples():
        strong = [c.replace("_pct", "") for c in pct if np.isfinite(getattr(r, c)) and getattr(r, c) >= 0.8]
        weak = all((not np.isfinite(getattr(r, c))) or getattr(r, c) < 0.5 for c in pct)
        hint = ("strong 100 m signal not extracted: " + ", ".join(strong)) if strong else \
            ("no 100 m signature (DEM / other data)" if weak else "moderate signal only")
        rows.append({"task": r.task, "zone": r.zone, "system_id": r.system_id, "trace_km": getattr(r, "trace_km", np.nan),
                     "length_bin": getattr(r, "length_bin", ""), "strike_sector": getattr(r, "strike_sector", ""),
                     "hint": hint, **{c: getattr(r, c) for c in pct}})
    d = pd.DataFrame(rows)
    return d


def run(m7, sets: dict) -> dict:
    parts = {k: [] for k in ("summary", "curve", "oracle", "by_group", "systems", "truth_stats")}
    for task in ("geo", m7.cfg["evaluation"]["primary_task"]):
        for u in m7.units_of(task):
            cs = sets.get(u.key)
            if cs is None or not len(cs):
                continue
            r = unit_audit(m7, u, cs)
            for k in parts:
                parts[k].append(r[k])
            print(f"[m7] s04 {u.key}: {len(cs)} candidates audited")
    res = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in parts.items()}
    res["pooled_curve"] = pooled(res["curve"], ["task", "density_cap"])
    res["pooled_oracle"] = pooled(res["oracle"], ["task", "kd_threshold"])
    res["pooled_groups"] = pooled(res["by_group"], ["task", "group", "value"])
    s02 = m7.ws.load_table("system_table", "failure_analysis/tables")
    attrs = m7.ws.load_table("system_attributes", "failure_analysis/tables")
    res["incremental"] = incremental(res["systems"], s02)
    res["by_stratum"] = by_stratum(res["systems"], attrs)
    res["decision"] = decision(m7, res["pooled_curve"], res["pooled_oracle"], res["incremental"])
    res["miss_diagnostics"] = miss_diagnostics(res["systems"], attrs)
    if len(res["miss_diagnostics"]):
        md = res["miss_diagnostics"]
        res["miss_summary"] = md.groupby(["task", "hint"]).agg(n_systems=("system_id", "size"),
                                                                trace_km=("trace_km", "sum")).reset_index()
    else:
        res["miss_summary"] = pd.DataFrame()
    return res


def figures(m7, res: dict) -> None:
    from .figures import bar_panels16, line_panels16

    ws, sub = m7.ws, "candidate_generation/oracle"
    pc = res["pooled_curve"]
    ref = (m7.cfg.get("recorded_reference") or {}).get("candidate_system_recall")
    if len(pc):
        d = pc.melt(id_vars=["task", "density_cap"], value_vars=["system_recall", "recall_w", "short_system_recall"],
                    var_name="series", value_name="recall")
        ws.save_fig(line_panels16(d, "density_cap", "recall", "series", "task", logx=True,
                                  xlabel="candidate density cap (km per 100 km^2, label-free order)", ylabel="recall",
                                  title="Recall-density curves of the candidate pool (dashed: recorded 12.5 % reference)",
                                  hline={t: ref for t in d.task.unique()} if ref is not None else None),
                    "recall_density_curves", d, sub)
    po = res["pooled_oracle"]
    if len(po):
        ws.save_fig(line_panels16(po.assign(series="oracle DTI (reads truth)"), "kd_threshold", "dti", "series", "task",
                                  xlabel="kept if mean k(dist to truth) >= t", ylabel="DTI",
                                  title="Oracle DTI of the pool (upper bound, diagnostic)"), "oracle_dti", po, sub)
    pg = res["pooled_groups"]
    if len(pg):
        d = pg[pg.group.isin(["modality", "provenance"])].assign(cat=lambda x: x.group + ": " + x.value)
        ws.save_fig(bar_panels16(d, "cat", "system_recall", "task", xlabel="system recall (all candidates of the group)",
                                 title="Coverage by modality / provenance"), "coverage_by_modality", d, sub)
    inc = res["incremental"]
    if len(inc):
        d = inc[inc.subset == "at_max_density"].assign(cat=lambda x: x["class"])
        ws.save_fig(bar_panels16(d, "cat", "coverage", "task", xlabel="share of systems covered by the pool",
                                 title="Coverage by B0 / B1 recovery class (neither = missed by both)"),
                    "coverage_by_b0b1_class", d, sub)
    bs = res["by_stratum"]
    if len(bs):
        d = bs[(bs.subset == "at_max_density")].assign(cat=lambda x: x.stratum + ": " + x.value)
        ws.save_fig(bar_panels16(d, "cat", "system_recall", "task", xlabel="system recall at max density",
                                 title="Coverage by geological stratum"), "coverage_by_stratum", d, sub)
