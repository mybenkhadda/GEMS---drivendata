"""Section 07: incremental discovery value of the 1 m DEM evidence (pilot coverage only).

Three candidate sets, scored on IDENTICAL masks (each unit restricted to DEM-covered pixels: a new protocol, never
pooled with full-zone numbers) and at a MATCHED candidate density (the sparser set's density; each set keeps its
highest-confidence candidates up to it):

    A  geophysics   section-03 label-free candidates of the 100 m geophysical / terrain sources (kind = feature)
    B  DEM          crests of the aggregated DEM evidence layers (``dem.candidate_layers``) at 100 m
    C  combined     A + B fused (duplicates merged, provenance kept)

Outputs: system / length / short-fault recall, coverage of low-geophysical-contrast systems, candidate density far
(> 1 km) from any mapped fault (unlabelled, NOT proven false: an upper bound on false-candidate density), recall by
terrain context, the systems recovered only by DEM, and the systems missed by B0 and B1 that DEM recovers.
Surface morphology cannot reveal every fault (blind, buried or eroded traces): the decision is about measured
incremental value on this pilot, not about completeness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..gems3.candidates import CandidateSet
from ..gems3.oracle import coverage
from .coverage import _ocfg
from .protocol import restrict_unit, system_hits


def dem_candidates(m7, mos, covered: np.ndarray) -> tuple[CandidateSet, pd.DataFrame]:
    """Label-free crests of the DEM evidence layers on the full grid (inside the covered footprint)."""
    from .generator import source_candidates

    cc = m7.cfg["candidates"]
    paths, rows, logs = [], [], []
    for b in m7.cfg["dem"]["candidate_layers"]:
        if not mos.has(b):
            logs.append({"source": b, "status": "band missing in mosaic"})
            continue
        a = mos.band(b)
        ps, rs, lg = source_candidates(np.where(covered, a, np.nan), covered & np.isfinite(a),
                                       {"name": f"dem_{b}", "mode": "ridge", "modality": "dem", "kind": "dem",
                                        "scales_m": [200, 400]}, cc)
        paths += ps
        rows += rs
        logs += [{**x, "status": "ok"} for x in lg]
    return CandidateSet(pd.DataFrame(rows), paths), pd.DataFrame(logs)


def _density(cs, u) -> float:
    return float(sum(len(p) for p in cs.paths) * 0.112 / max(u.ev.sum() * 0.01, 1e-9) * 100.0)


def _cap(cs, u, cap: float):
    from .generator import structural_score

    if not len(cs):
        return cs
    t = cs.table
    s = structural_score(t.assign(length_km=[len(p) * 0.112 for p in cs.paths]))
    order = np.argsort(-s, kind="stable")
    km = np.array([len(cs.paths[i]) * 0.112 for i in order])
    keep = order[np.cumsum(km) <= cap * u.ev.sum() * 0.01 / 100.0]
    return cs.subset(sorted(keep.tolist()))


def _clip(cs, u, covered_crop: np.ndarray):
    keep, paths = [], []
    for i, p in enumerate(cs.paths):
        q = p[covered_crop[p[:, 0], p[:, 1]] & u.ev[p[:, 0], p[:, 1]]]
        if len(q) >= 3:
            keep.append(i)
            paths.append(q)
    return CandidateSet(cs.table.iloc[keep].reset_index(drop=True), paths)


def run(m7, unit_sets: dict, mos) -> dict:
    from .generator import clip_to_unit, fuse

    cfg = m7.cfg
    d = cfg["dem"]
    covered = mos.coverage(d["min_coverage"]) & m7.ctx.data.valid
    lab = np.asarray(m7.ctx.data.labels, bool)
    from .protocol import distance_km

    far_km = distance_km(lab) > 1.0
    B_full, blog = dem_candidates(m7, mos, covered)
    attrs = m7.ws.load_table("system_attributes", "failure_analysis/tables")
    s02 = m7.ws.load_table("system_table", "failure_analysis/tables")
    rows, sysrows, base_rows = [], [], []
    oo = _ocfg(cfg)
    for task in ("geo", cfg["evaluation"]["primary_task"]):
        for u in m7.units_of(task):
            cv = covered[u.sl]
            if not (cv & u.ev).any():
                continue
            uc = restrict_unit(u, cv, "@dem")
            if not uc.eval_truth().any():
                continue
            full = unit_sets.get(u.key)
            if full is None:
                continue
            A = _clip(full.subset(np.flatnonzero(full.table.kind.astype(str).to_numpy() == "feature").tolist()), u, cv)
            Bu = _clip(clip_to_unit(B_full, u, 3), u, cv)
            raw = CandidateSet(pd.concat([A.table, Bu.table], ignore_index=True), A.paths + Bu.paths)
            C = fuse(raw, u.ev.shape, cfg["candidates"]) if len(raw) else raw
            dens = {k: _density(s, uc) for k, s in (("A_geophysics", A), ("B_dem", Bu), ("C_combined", C))}
            cap = min(v for v in (dens["A_geophysics"], dens["B_dem"]) if v > 0) if any(
                v > 0 for v in (dens["A_geophysics"], dens["B_dem"])) else 0.0
            for name, cs in (("A_geophysics", A), ("B_dem", Bu), ("C_combined", C)):
                capped = _cap(cs, uc, cap) if cap > 0 else cs
                cov = coverage(m7.ctx, uc, capped, oo)
                fk = far_km[u.sl]
                far_len = sum(len(p) * 0.112 for p in capped.paths if fk[p[:, 0], p[:, 1]].mean() >= 0.5)
                far = float(far_len / max(uc.ev.sum() * 0.01, 1e-9) * 100.0)
                rows.append({"task": task, "zone": u.zone, "set": name, "native_density": dens[name],
                             "matched_density_cap": cap, **cov, "far_candidate_km_per_100km2": far,
                             "covered_km2": float(uc.ev.sum() * 0.01), "protocol": "DEM-covered restriction"})
                h = system_hits(m7.ctx, uc, capped.raster(uc.ev.shape), cfg).assign(task=task, zone=u.zone, set=name)
                sysrows.append(h)
            # baseline on the same restricted mask (context, same protocol)
            from .baseline import heldout_map

            q = heldout_map(m7, task, "fusion_thin", u.zone)
            if q is not None:
                c = uc.components(q)
                base_rows.append({"task": task, "zone": u.zone, "method": "fusion_thin (section 01 map)",
                                  **c.as_dict()})
    res = pd.DataFrame(rows)
    sysd = pd.concat(sysrows, ignore_index=True) if sysrows else pd.DataFrame()
    venn = venn_table(sysd, attrs, s02)
    pooled = pooled_sets(res)
    dec = decision(m7, pooled, venn, sysd)
    return {"results": res, "pooled": pooled, "systems": sysd, "venn": venn, "baseline_restricted": pd.DataFrame(base_rows),
            "dem_candidate_log": blog, "decision": dec, "dem_candidates_full": B_full}


def pooled_sets(res: pd.DataFrame) -> pd.DataFrame:
    if not len(res):
        return res
    g = res.groupby(["task", "set"])[["tp_w", "fp_w", "fn_w", "n_truth"]].sum().reset_index()
    g["dti_unranked"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
    w = res.assign(_s=res.system_recall.fillna(0) * res.n_systems, _ss=res.short_system_recall.fillna(0) * res.n_short_systems)
    a = w.groupby(["task", "set"]).agg(n_systems=("n_systems", "sum"), _s=("_s", "sum"), n_short=("n_short_systems", "sum"),
                                       _ss=("_ss", "sum"), recall_300m=("recall@300m", "mean"),
                                       far_km_per_100km2=("far_candidate_km_per_100km2", "mean"),
                                       covered_km2=("covered_km2", "sum")).reset_index()
    a["system_recall"] = a._s / a.n_systems.clip(lower=1)
    a["short_system_recall"] = a._ss / a.n_short.clip(lower=1)
    g["recall_w"] = g.tp_w / g.n_truth.clip(lower=1)
    return g.merge(a.drop(columns=["_s", "_ss"]), on=["task", "set"])


def venn_table(sysd: pd.DataFrame, attrs: pd.DataFrame, s02: pd.DataFrame) -> pd.DataFrame:
    """Per system: covered by geophysics (A) / DEM (B) -> geophysics_only / dem_only / both / neither (+ km, and the
    B0 / B1 class from section 02, and the terrain / contrast context)."""
    if not len(sysd):
        return pd.DataFrame()
    p = sysd.pivot_table(index=["task", "zone", "system_id"], columns="set", values="recovered", aggfunc="first")
    p = p.fillna(False).astype(bool).reset_index()
    flag = (lambda c: p[c].astype(bool) if c in p else pd.Series(False, index=p.index))
    a, b = flag("A_geophysics"), flag("B_dem")
    p["venn"] = np.select([a & b, a & ~b, ~a & b], ["both", "geophysics_only", "dem_only"], "neither")
    if len(attrs):
        keep = [c for c in ("system_id", "trace_km", "length_bin", "terrain_slope_tercile", "low_geophysical_contrast",
                            "strike_sector") if c in attrs]
        p = p.merge(attrs[keep], on="system_id", how="left")
    if len(s02):
        p = p.merge(s02[["task", "zone", "system_id", "class"]], on=["task", "zone", "system_id"], how="left")
    return p


def decision(m7, pooled: pd.DataFrame, venn: pd.DataFrame, sysd: pd.DataFrame) -> pd.DataFrame:
    gi = m7.cfg["dem"]["go_if"]
    prim = m7.cfg["evaluation"]["primary_task"]
    est = m7.read_json("dem/pilot/dem_cost_estimate.json") or {}
    v = venn[venn.task == prim] if len(venn) else venn
    n_cov = int(len(v))
    dem_only = int((v.venn == "dem_only").sum()) if len(v) else 0
    pc = pooled[pooled.task == prim] if len(pooled) else pooled
    rec = {r.set: r.system_recall for r in pc.itertuples()} if len(pc) else {}
    inc = rec.get("C_combined", np.nan) - rec.get("A_geophysics", np.nan)
    hrs, gib = est.get("projected_hours_all", np.nan), est.get("projected_raw_gib_all", np.nan)
    cost_ok = (not np.isfinite(hrs) or hrs <= gi["max_full_hours"]) and (not np.isfinite(gib) or gib <= gi["max_full_gib"])
    if n_cov < int(gi["min_systems_covered"]):
        verdict = "INCONCLUSIVE"
        why = f"only {n_cov} withheld systems inside DEM coverage (< {gi['min_systems_covered']}): extend the pilot"
    elif dem_only >= int(gi["additional_systems"]) and np.isfinite(inc) and inc >= float(gi["incremental_system_recall"]) \
            and cost_ok:
        verdict, why = "EXPAND", "DEM recovers additional systems at an acceptable projected cost"
    elif dem_only >= int(gi["additional_systems"]) and not cost_ok:
        verdict, why = "EXPAND_TARGETED", "value shown, but the full-coverage cost exceeds the limits: expand selectively"
    else:
        verdict, why = "STOP", "no measured incremental discovery value on the pilot"
    missed = v[(v.get("class") == "neither") & v.venn.isin(["dem_only", "both"])] if len(v) and "class" in v else v.iloc[:0]
    return pd.DataFrame([{"task": prim, "withheld_systems_in_coverage": n_cov, "dem_only_systems": dem_only,
                          "system_recall_A": rec.get("A_geophysics", np.nan), "system_recall_B": rec.get("B_dem", np.nan),
                          "system_recall_C": rec.get("C_combined", np.nan), "incremental_C_minus_A": inc,
                          "missed_by_b0b1_recovered_by_dem": int(len(missed)), "projected_hours_all": hrs,
                          "projected_raw_gib_all": gib, "decision": verdict, "reason": why,
                          "rule": (f"withheld systems in coverage >= {gi['min_systems_covered']}; dem_only >= "
                                   f"{gi['additional_systems']}; C - A system recall >= {gi['incremental_system_recall']};"
                                   f" hours <= {gi['max_full_hours']}, GiB <= {gi['max_full_gib']}")}])
