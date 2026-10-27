"""P6.1 X1: deployable W2 policies, training / inference parity and the one export contract (P6_RECOVERY_HANDOFF.md).

A *policy* is a candidate space (ranker rows x reps x fracs x blends of the boost benchmark grid) plus ONE selection
objective. For every outer zone z the cell (candidate, blend, rep, frac) is chosen on rows that do not overlap zone z
(both tasks), then applied unchanged to zone z of BOTH tasks: the discovery and geographic numbers of a policy come
from the same per-zone decisions, and the export uses the same objective on all zones (``deploy_cell``). This is what
the historical W2 file lacked (rep / blend from the discovery choice, frac from the geographic one).

    T = policy_tables(G, cfg, prim, gates, tasks, exclusion)   # pure: comparison, selection, per zone, frontier, choice
    rec = build_recipe(...)                                     # the only way to describe an export
    info = write_validated(p6, rec, q, ledger)                  # GeoTIFF (never overwritten) + audit_export

Nothing here reads held-out labels except through the saved official components of the grid. Rows tagged
``role = oracle*`` (``candidate_recall``) are refused by every selection function (``OracleLeak``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import boost as B
from . import stack as ST

COMP = B.COMP
CELL = ["candidate", "blend", "rep", "frac"]
OBJECTIVES = ("disc", "disc_geo_floor", "joint", "separate")
RECIPE_POLICY_KEYS = ("selection_id", "space", "objective", "rep", "frac", "blend")


class PolicyMismatch(ValueError):
    """An export recipe whose representation, area or blend come from different selections."""


class OracleLeak(RuntimeError):
    """Diagnostic oracle rows (held-out labels used to build the map) reached a selection function."""


# =============================================================================== pure: policies
def guard_oracle(G: pd.DataFrame) -> None:
    if len(G) and "role" in G and G.role.astype(str).str.startswith("oracle").any():
        raise OracleLeak("oracle rows (label-built diagnostic maps) must never enter a selection or a reported score")


def policy_rows(G: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Rows of the policy's space; ``blend`` NaN (CNN rows) becomes -1 so cells are hashable and comparable."""
    guard_oracle(G)
    g = G[G.candidate.isin(list(spec["candidates"]))].copy()
    g["blend"] = pd.to_numeric(g.blend, errors="coerce").fillna(-1.0).astype(float)
    g["frac"] = g.frac.astype(float)
    if spec.get("reps"):
        g = g[g.rep.isin(list(spec["reps"]))]
    if spec.get("fracs"):
        g = g[np.isclose(g.frac.to_numpy()[:, None], np.asarray(spec["fracs"], float)[None]).any(1)]
    if spec.get("blends"):
        g = g[np.isclose(g.blend.to_numpy()[:, None], np.asarray(spec["blends"], float)[None]).any(1)]
    fx = spec.get("fixed")
    if fx:
        g = g[(g.rep == fx["rep"]) & np.isclose(g.frac, float(fx["frac"])) &
              np.isclose(g.blend, float(fx.get("blend", 1.0)))]
    return g


def _cell_scores(S: pd.DataFrame, task: str, zones) -> pd.DataFrame:
    """Pooled components + DTI of every cell over ``zones`` of ``task``; only cells present in every zone."""
    d = S[(S.task == task) & S.zone.isin(list(zones))]
    if not len(d):
        return pd.DataFrame(columns=CELL + COMP + ["dti", "n_zones"])
    g = d.groupby(CELL, dropna=False).agg(**{c: (c, "sum") for c in COMP}, n_zones=("zone", "nunique")).reset_index()
    g = g[g.n_zones == len(set(zones))]
    g["dti"] = B.dti(g.tp_w, g.fp_w, g.fn_w)
    return g


def _key(r) -> str:
    return f"{r['candidate']}|{float(r['blend']):g}|{r['rep']}|{float(r['frac']):g}"


def _as_cell(r) -> dict:
    return {"candidate": str(r["candidate"]), "blend": float(r["blend"]), "rep": str(r["rep"]), "frac": float(r["frac"])}


def select_cell(S: pd.DataFrame, disc_zones, geo_zones, objective: str, prim: str,
                geo_floor: float) -> tuple[dict | None, dict]:
    """One cell chosen on the selection zones only. ``disc``: max pooled discovery DTI. ``disc_geo_floor``: max
    discovery DTI among cells whose pooled geographic DTI on the geo selection zones is >= ``geo_floor`` (max
    discovery when none qualifies, flagged). ``joint``: max mean of both. ``geo``: max geographic DTI.
    Deterministic tie-break: the cell key string."""
    guard_oracle(S)
    if objective not in ("disc", "disc_geo_floor", "joint", "geo"):
        raise ValueError(f"objective {objective!r}")
    D = _cell_scores(S, prim, disc_zones) if objective != "geo" else None
    Gs = _cell_scores(S, "geo", geo_zones) if objective != "disc" else None
    if objective == "geo":
        T = Gs.rename(columns={"dti": "geo_dev"})
        T["disc_dev"] = np.nan
        T["score"] = T.geo_dev
    elif objective == "disc":
        T = D.rename(columns={"dti": "disc_dev"})
        T["geo_dev"] = np.nan
        T["score"] = T.disc_dev
    else:
        T = D[CELL + ["dti"]].rename(columns={"dti": "disc_dev"}).merge(
            Gs[CELL + ["dti"]].rename(columns={"dti": "geo_dev"}), on=CELL, how="inner") if len(Gs) else \
            D[CELL + ["dti"]].rename(columns={"dti": "disc_dev"}).assign(geo_dev=np.nan)
        if objective == "joint":
            T["score"] = 0.5 * (T.disc_dev + T.geo_dev)
        else:
            ok = T.geo_dev >= float(geo_floor)
            T["score"] = np.where(ok, T.disc_dev, -np.inf) if ok.any() else T.disc_dev
    if not len(T) or not np.isfinite(T.score.to_numpy(float)).any():
        return None, {"floor_met": None, "n_cells": int(len(T))}
    T = T.assign(_k=[_key(r) for r in T.to_dict("records")]).sort_values(["score", "_k"], ascending=[False, True])
    best = T.iloc[0]
    floor_met = bool((T.geo_dev >= float(geo_floor)).any()) if objective == "disc_geo_floor" else None
    return _as_cell(best), {"disc_dev": float(best.disc_dev) if pd.notna(best.disc_dev) else np.nan,
                            "geo_dev": float(best.geo_dev) if pd.notna(best.geo_dev) else np.nan,
                            "floor_met": floor_met, "n_cells": int(len(T))}


def held_row(S: pd.DataFrame, cell: dict, zone, task: str) -> dict | None:
    m = (S.task == task) & (S.zone == zone) & (S.candidate == cell["candidate"]) & np.isclose(S.blend, cell["blend"]) \
        & (S.rep == cell["rep"]) & np.isclose(S.frac, cell["frac"])
    r = S[m]
    if len(r) > 1:
        raise ValueError(f"duplicate grid rows for {task}:{zone} {cell}")
    return r.iloc[0].to_dict() if len(r) else None


def same_index_exclusion(zones) -> dict:
    """Outer zone z -> zones removed from selection in each task (the default: index z in both tasks)."""
    return {z: {"disc": {z}, "geo": {z}} for z in zones}


def overlap_exclusion(overlap: pd.DataFrame, zones, prim: str, min_px: int = 1) -> tuple[dict, bool]:
    """From a unit-overlap table (task_a, zone_a, task_b, zone_b, overlap_px): for outer z, every unit whose
    evaluation mask overlaps disc z or geo z is removed from selection. Returns (exclusion, feasible); infeasible
    (some outer zone keeps no disc or no geo selection zone) falls back to ``same_index_exclusion``."""
    ex = {}
    for z in zones:
        hit = overlap[(overlap.overlap_px >= min_px) & (overlap.zone_a == z) & overlap.task_a.isin([prim, "geo"])]
        ex[z] = {"disc": {z} | set(hit[hit.task_b == prim].zone_b), "geo": {z} | set(hit[hit.task_b == "geo"].zone_b)}
    ok = all(set(zones) - ex[z]["disc"] and set(zones) - ex[z]["geo"] for z in zones)
    return (ex, True) if ok else (same_index_exclusion(zones), False)


def evaluate_policy(G: pd.DataFrame, name: str, spec: dict, objective: str, prim: str, geo_floor: float,
                    tasks: list[str], zones=None, exclusion: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Nested evaluation of one policy: per outer zone, select on the non-excluded zones, apply to the outer zone
    of every task. ``separate`` reproduces the boost numbers (each task selects on itself: NOT one deployable
    policy unless both tasks pick the same cells). Returns (summary, per-zone rows)."""
    S = policy_rows(G, spec)
    zones = sorted(S.zone.unique()) if zones is None else sorted(zones)
    ex = exclusion or same_index_exclusion(zones)
    fixed = spec.get("fixed")
    rows = []
    none = {"floor_met": None, "n_cells": 1, "disc_dev": np.nan, "geo_dev": np.nan}
    for z in zones:
        dz = [x for x in zones if x not in ex.get(z, {}).get("disc", {z})]
        gz = [x for x in zones if x not in ex.get(z, {}).get("geo", {z})]
        if fixed:
            shared = ((_as_cell(S.iloc[0]), none) if len(S) else (None, none))
        elif objective != "separate":                          # ONE cell per outer zone, applied to every task
            shared = select_cell(S, dz, gz, objective, prim, geo_floor)
        for task in tasks:
            if objective == "separate" and not fixed:
                cell, info = select_cell(S, dz, gz, "disc" if task == prim else "geo", prim, geo_floor)
            else:
                cell, info = shared
            r = held_row(S, cell, z, task) if cell else None
            rows.append({"policy": name, "objective": objective, "task": task, "zone": z,
                         "selection_disc_zones": ";".join(map(str, dz)), "selection_geo_zones": ";".join(map(str, gz)),
                         **(cell or {k: None for k in CELL}), **{k: info.get(k) for k in ("disc_dev", "geo_dev",
                                                                                          "floor_met", "n_cells")},
                         **{c: (float(r[c]) if r else np.nan) for c in COMP},
                         "dti": float(B.dti(r["tp_w"], r["fp_w"], r["fn_w"])) if r else np.nan,
                         "area_px": r.get("area_px") if r else np.nan,
                         "area_frac_eligible": r.get("area_frac_eligible") if r else np.nan,
                         "length_km": r.get("length_km") if r else np.nan,
                         "n_segments": r.get("n_segments") if r else np.nan,
                         "short_share": r.get("short_share") if r else np.nan,
                         "systems": (r.get("systems") if r else None), "held_row_found": r is not None})
    P = pd.DataFrame(rows)
    out = {"policy": name, "objective": objective, "fixed": bool(fixed), "n_zones": len(zones)}
    for task in tasks:
        pre = "disc" if task == prim else "geo"
        p = P[(P.task == task) & P.held_row_found] if len(P) else P
        zs = {int(r.zone): float(r.dti) for r in p.itertuples()}
        complete = len(zs) == len(zones)
        out.update({f"{pre}_dti": float(B.dti(p.tp_w.sum(), p.fp_w.sum(), p.fn_w.sum())) if complete else np.nan,
                    f"{pre}_worst": min(zs.values()) if complete else np.nan,
                    f"{pre}_worst_zone": min(zs, key=zs.get) if complete else None, f"{pre}_per_zone": zs,
                    f"{pre}_fp_w": float(p.fp_w.sum()) if complete else np.nan,
                    f"{pre}_precision_w": float(p.tp_w.sum() / max(p.tp_w.sum() + p.fp_w.sum(), 1e-9)) if complete
                    else np.nan,
                    f"{pre}_recall_w": float(p.tp_w.sum() / max(p.tp_w.sum() + p.fn_w.sum(), 1e-9)) if complete
                    else np.nan})
    if fixed:
        dep, dinfo = (_as_cell(S.iloc[0]) if len(S) else None), {}
    elif objective == "separate":
        dep, dinfo = select_cell(S, zones, zones, "disc", prim, geo_floor)
        geo_dep, _ = select_cell(S, zones, zones, "geo", prim, geo_floor)
        out["geo_deploy_cell"] = geo_dep
    else:
        dep, dinfo = select_cell(S, zones, zones, objective, prim, geo_floor)
    Q = P.dropna(subset=["rep"]) if len(P) else P
    Q = Q.assign(_cell=[_key(r) for r in Q.to_dict("records")]) if len(Q) else Q.assign(_cell=pd.Series(dtype=str))
    out.update(deploy_cell=dep, deploy_dev=dinfo, n_distinct_cells=int(Q._cell.nunique()) if len(Q) else 0)
    if objective == "separate":                                # one file can carry only one cell
        per_z = Q.groupby("zone")._cell.nunique() if len(Q) else pd.Series(dtype=int)   # distinct keys per zone
        out["deployable"] = bool(len(per_z) and (per_z == 1).all() and dep == out.get("geo_deploy_cell"))
    else:
        out["deployable"] = dep is not None
    return out, P


def systems_vs_reference(P: pd.DataFrame, R: pd.DataFrame, prim: str) -> dict:
    """Systems of a policy's held-out choices against a reference policy's held-out choices (same zones)."""
    a = {int(r.zone): B.system_set(r.systems) for r in P[P.task == prim].itertuples()} if len(P) else {}
    b = {int(r.zone): B.system_set(r.systems) for r in R[R.task == prim].itertuples()} if len(R) else {}
    g = [f"{z}:{i}" for z in sorted(a) for i in sorted(a[z] - b.get(z, set()))]
    lo = [f"{z}:{i}" for z in sorted(b) for i in sorted(b[z] - a.get(z, set()))]
    return {"systems_recovered": sum(len(v) for v in a.values()), "systems_reference": sum(len(v) for v in b.values()),
            "systems_gained": len(g), "systems_lost": len(lo), "gained_ids": ";".join(g), "lost_ids": ";".join(lo)}


def deploy_choice(summaries: dict, gates: dict, roles: dict | None = None) -> tuple[str | None, list[str]]:
    """Predeclared deployment rule: among deployable, selection-based (non-fixed) deploy candidates, the highest
    nested discovery DTI with geographic DTI >= floor and worst zone >= floor; ties -> name. Without any qualifying
    policy: the highest discovery DTI, flagged (never silently)."""
    roles = roles or {}
    ok, why = [], []
    for n, s in summaries.items():
        if roles.get(n, "deploy_candidate") != "deploy_candidate" or s.get("fixed") or not s.get("deployable"):
            continue
        d, g, w = s.get("disc_dti"), s.get("geo_dti"), s.get("disc_worst")
        if d is None or not np.isfinite(d):
            continue
        meets = (g is not None and np.isfinite(g) and g >= float(gates["geo_floor"])) and \
            (w is not None and np.isfinite(w) and w >= float(gates["worst_floor"]) - 1e-12)
        ok.append((bool(meets), float(d), n))
    if not ok:
        return None, ["no deployable policy with a discovery DTI"]
    best = sorted(ok, key=lambda t: (-int(t[0]), -t[1], t[2]))[0]
    if not best[0]:
        why.append("no policy meets both the geographic and the worst-zone floor: highest discovery DTI kept, "
                   "NOT promotable")
    return best[2], why


def meta_nested(G: pd.DataFrame, specs: dict, objective: str, prim: str, gates: dict, tasks: list[str],
                roles: dict | None = None, exclusion: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Honest estimate of 'choose the policy, then deploy it': for each outer zone the policy is chosen with
    ``deploy_choice`` on the inner zones only (each inner policy itself nested), its cell selected on the inner
    zones and applied to the outer zone. With three zones every inner selection sees one zone: noisy, but never
    the held-out zone."""
    zones = sorted(set(G.zone.unique()))
    ex = exclusion or same_index_exclusion(zones)
    rows = []
    for z in zones:
        inner = [x for x in zones if x not in ex.get(z, {}).get("disc", {z}) | ex.get(z, {}).get("geo", {z})]
        if len(inner) < 2:                                      # nothing left to nest the policy choice on
            continue
        summ = {n: evaluate_policy(G, n, sp, objective, prim, float(gates["geo_floor"]), tasks, inner)[0]
                for n, sp in specs.items()}
        name, _ = deploy_choice(summ, gates, roles)
        if name is None:
            continue
        S = policy_rows(G, specs[name])
        dz = [x for x in zones if x not in ex.get(z, {}).get("disc", {z})]
        gz = [x for x in zones if x not in ex.get(z, {}).get("geo", {z})]
        cell, _ = (( _as_cell(S.iloc[0]), {}) if specs[name].get("fixed") else
                   select_cell(S, dz, gz, "disc" if objective == "separate" else objective, prim,
                               float(gates["geo_floor"])))
        for task in tasks:
            r = held_row(S, cell, z, task) if cell else None
            rows.append({"policy": "META(choose policy on inner zones)", "chosen_policy": name, "objective": objective,
                         "task": task, "zone": z, "selection_disc_zones": ";".join(map(str, dz)),
                         "selection_geo_zones": ";".join(map(str, gz)), **(cell or {}),
                         **{c: (float(r[c]) if r else np.nan) for c in COMP},
                         "dti": float(B.dti(r["tp_w"], r["fp_w"], r["fn_w"])) if r else np.nan})
    P = pd.DataFrame(rows)
    out = {"policy": "META", "objective": objective}
    for task in tasks:
        pre = "disc" if task == prim else "geo"
        p = P[P.task == task].dropna(subset=["dti"]) if len(P) else P
        ok = len(p) == len(zones)
        out[f"{pre}_dti"] = float(B.dti(p.tp_w.sum(), p.fp_w.sum(), p.fn_w.sum())) if ok else np.nan
        out[f"{pre}_worst"] = float(p.dti.min()) if ok else np.nan
    out["chosen_per_zone"] = {int(r.zone): r.chosen_policy for r in P[P.task == prim].itertuples()} if len(P) else {}
    return out, P


def policy_tables(G: pd.DataFrame, cfg: dict, prim: str, gates: dict, tasks: list[str],
                  exclusion: dict | None = None, benchmark: pd.DataFrame | None = None) -> dict:
    """Every X1 table from the benchmark grid (pure). ``cfg``: ``export_parity`` section; ``benchmark``: the boost
    ``nested_selection.csv`` for the reproduction check."""
    guard_oracle(G)
    specs = {**{n: {**s, "_role": "deploy_candidate"} for n, s in cfg["policies"].items()},
             **{n: {**s, "_role": "reference"} for n, s in (cfg.get("references") or {}).items()}}
    roles = {n: s["_role"] for n, s in specs.items()}
    obj0 = cfg["objective"]
    objs = [obj0] + [o for o in cfg.get("sensitivity_objectives", []) if o != obj0] + ["separate"]
    summ, pers = [], []
    ref_name = next((n for n, r in roles.items() if r == "reference"), None)
    for obj in objs:
        res = {n: evaluate_policy(G, n, sp, obj, prim, float(gates["geo_floor"]), tasks, exclusion=exclusion)
               for n, sp in specs.items()}
        R = res[ref_name][1] if ref_name else pd.DataFrame()
        for n, (s, P) in res.items():
            s = {**s, "role": roles[n] if not specs[n].get("fixed") else ("historical" if roles[n] ==
                                                                          "deploy_candidate" else roles[n]),
                 "benchmark_space": specs[n].get("benchmark_space"), **systems_vs_reference(P, R, prim)}
            if benchmark is not None and len(benchmark) and specs[n].get("benchmark_space") and obj == "separate":
                b = benchmark[benchmark.space == specs[n]["benchmark_space"]]
                if len(b):
                    s["benchmark_disc_dti"] = float(b.disc_dti.iloc[0])
                    s["benchmark_geo_dti"] = float(b.geo_dti.iloc[0])
                    s["reproduces_benchmark"] = bool(abs(s["benchmark_disc_dti"] - s.get("disc_dti", np.nan)) < 1e-6
                                                     and abs(s["benchmark_geo_dti"] - s.get("geo_dti", np.nan)) < 1e-6)
            s["meets_geo_floor"] = bool(np.isfinite(s.get("geo_dti", np.nan)) and s["geo_dti"] >= gates["geo_floor"])
            s["meets_worst_floor"] = bool(np.isfinite(s.get("disc_worst", np.nan)) and
                                          s["disc_worst"] >= gates["worst_floor"] - 1e-12)
            s["beats_w2_recorded"] = bool(np.isfinite(s.get("disc_dti", np.nan)) and
                                          s["disc_dti"] > float(gates["w2_recorded_disc"]))
            summ.append(s)
            pers.append(P)
    S = pd.DataFrame(summ)
    PZ = pd.concat(pers, ignore_index=True) if pers else pd.DataFrame()
    main = {r["policy"]: r for r in summ if r["objective"] == obj0}
    chosen, why = deploy_choice(main, gates, roles)
    mspecs = {n: sp for n, sp in specs.items() if roles[n] == "deploy_candidate"}
    meta, MP = meta_nested(G, mspecs, obj0, prim, gates, tasks, roles, exclusion)
    sel_cols = ["policy", "objective", "zone", "task", "selection_disc_zones", "selection_geo_zones", *CELL,
                "disc_dev", "geo_dev", "floor_met", "n_cells"]
    NS = pd.concat([PZ[[c for c in sel_cols if c in PZ]], MP.reindex(columns=sel_cols + ["chosen_policy"])],
                   ignore_index=True) if len(PZ) else PZ
    fr = [{"kind": "nested_policy", "policy": r["policy"], "objective": r["objective"], "role": r["role"],
           "deployable": r["deployable"], "disc_dti": r.get("disc_dti"), "geo_dti": r.get("geo_dti"),
           "disc_worst": r.get("disc_worst")} for r in summ]
    M = B.matched_comparison(G[G.task.isin(tasks)]) if len(G) else pd.DataFrame()
    if len(M):
        d = M[M.task == prim][["candidate", "blend", "rep", "frac", "dti", "worst_zone_dti"]]
        g = M[M.task == "geo"][["candidate", "blend", "rep", "frac", "dti"]].rename(columns={"dti": "geo_dti"})
        X = d.merge(g, on=["candidate", "blend", "rep", "frac"], how="inner")
        fr += [{"kind": "fixed_cell (unselected, in-sample; not a nested estimate)", "policy": _key(r),
                "objective": "none", "role": "diagnostic", "deployable": True, "disc_dti": r["dti"],
                "geo_dti": r["geo_dti"], "disc_worst": r["worst_zone_dti"], "candidate": r["candidate"]}
               for r in X.to_dict("records")]
    return {"comparison": S, "per_zone": PZ, "selection": NS, "frontier": pd.DataFrame(fr), "chosen": chosen,
            "chosen_reasons": why, "chosen_summary": main.get(chosen), "meta": meta, "meta_per_zone": MP,
            "specs": specs, "objective": obj0}


# =============================================================================== pure: parity statistics
def agreement(a: np.ndarray, b: np.ndarray, mask: np.ndarray, top_fracs=(0.01, 0.05), n: int = 200_000,
              seed: int = 0) -> dict:
    """Spearman / Pearson on a pixel sample, Jaccard of the top ``top_fracs`` pixels, quantile shifts and the KS
    statistic between ``a`` (training construction) and ``b`` (inference construction) inside ``mask``."""
    from scipy.stats import ks_2samp, pearsonr, spearmanr

    m = np.asarray(mask, bool) & np.isfinite(a) & np.isfinite(b)
    out = {"n_px": int(m.sum()), "nan_share_train": float(1 - np.isfinite(np.asarray(a)[mask]).mean()) if mask.any()
           else np.nan, "nan_share_infer": float(1 - np.isfinite(np.asarray(b)[mask]).mean()) if mask.any() else np.nan}
    va, vb = np.asarray(a, np.float64)[m], np.asarray(b, np.float64)[m]
    if va.size < 10:
        return {**out, "spearman": np.nan, "pearson": np.nan, "ks": np.nan,
                **{f"top{int(round(f * 100))}_jaccard": np.nan for f in top_fracs}}
    idx = np.random.default_rng(seed).choice(va.size, min(n, va.size), replace=False)
    sa, sb = va[idx], vb[idx]
    const = sa.std() == 0 or sb.std() == 0
    out["spearman"] = np.nan if const else float(spearmanr(sa, sb)[0])
    out["pearson"] = np.nan if const else float(pearsonr(sa, sb)[0])
    out["ks"] = float(ks_2samp(sa, sb).statistic)
    for f in top_fracs:
        k = max(int(round(f * va.size)), 1)
        ta, tb = np.argpartition(-va, k - 1)[:k], np.argpartition(-vb, k - 1)[:k]
        out[f"top{int(round(f * 100))}_jaccard"] = float(len(np.intersect1d(ta, tb)) / max(len(np.union1d(ta, tb)), 1))
    for q in (0.05, 0.5, 0.95):
        out[f"q{int(q * 100):02d}_train"], out[f"q{int(q * 100):02d}_infer"] = float(np.quantile(va, q)), \
            float(np.quantile(vb, q))
    out["median_shift"] = out["q50_infer"] - out["q50_train"]
    return out


def map_agreement(qa: np.ndarray, qb: np.ndarray) -> dict:
    a, b = np.asarray(qa) > 0, np.asarray(qb) > 0
    inter, uni = int((a & b).sum()), int((a | b).sum())
    return {"px_train": int(a.sum()), "px_infer": int(b.sum()), "jaccard": inter / uni if uni else np.nan,
            "dice": 2 * inter / (a.sum() + b.sum()) if (a.sum() + b.sum()) else np.nan}


def classify_parity(r: dict, pc: dict, expected: bool = False) -> str:
    """PASS / WARNING / BLOCKED; ``expected`` (by-design differences such as the known-fault map) never BLOCKs.
    Diagnostic only: a label-free numerical difference is not by itself proof of leakage or of harm."""
    rho = r.get("spearman")
    if rho is None or not np.isfinite(rho):
        return "WARNING"
    if rho < float(pc["block_rank_corr"]) and not expected:
        return "BLOCKED"
    j = r.get("top1_jaccard")
    if expected or rho < float(pc["warn_rank_corr"]) or (j is not None and np.isfinite(j) and j < 0.2):
        return "WARNING"
    return "PASS"


# =============================================================================== export contract (pure)
def build_recipe(*, candidate: str, kind: str, policy: dict, model: dict, sources: dict, feature_schema: list,
                 preprocessing: dict) -> dict:
    """The complete, single-selection description of one export. ``policy`` must name one ``selection_id``;
    any ``rep_source`` / ``frac_source`` / ``blend_source`` that differs from it raises ``PolicyMismatch`` (the
    historical W2 defect: rep from the discovery choice, frac from the geographic one)."""
    from ..pipeline.m6 import fingerprint

    miss = [k for k in RECIPE_POLICY_KEYS if policy.get(k) is None]
    if miss:
        raise ValueError(f"export recipe policy incomplete: {miss}")
    sid = policy["selection_id"]
    bad = {k: policy[k] for k in ("rep_source", "frac_source", "blend_source") if policy.get(k) not in (None, sid)}
    if bad:
        raise PolicyMismatch(f"rep / frac / blend must come from one selection ({sid}); got {bad}")
    if list(model.get("features") or feature_schema) != list(feature_schema):
        raise ValueError("model features differ from the feature schema")
    pol = {**policy, "rep": str(policy["rep"]), "frac": float(policy["frac"]), "blend": float(policy["blend"]),
           "rep_source": sid, "frac_source": sid, "blend_source": sid}
    rec = {"candidate": candidate, "kind": kind, "policy": pol, "model": model, "sources": sources,
           "feature_schema": list(feature_schema), "preprocessing": preprocessing, "contract_version": "p6r-export.1"}
    rec["fingerprint"] = fingerprint("p6r-export.1", rec)
    return rec


def plan_audit(rec: dict) -> dict:
    """Checks on a recipe before any file exists (writes nothing)."""
    checks = [{"check": "single selection supplies rep, frac and blend",
               "ok": len({rec["policy"].get(k) for k in ("rep_source", "frac_source", "blend_source")}) == 1},
              {"check": "area fraction in (0, 1)", "ok": 0.0 < float(rec["policy"]["frac"]) < 1.0},
              {"check": "blend in [0, 1]", "ok": 0.0 <= float(rec["policy"]["blend"]) <= 1.0},
              {"check": "feature schema non-empty and unique", "ok": bool(rec["feature_schema"]) and
               len(set(rec["feature_schema"])) == len(rec["feature_schema"])},
              {"check": "every source hash recorded", "ok": all(v is not None for v in _flat_hashes(rec["sources"]))},
              {"check": "eligibility / known treatment declared", "ok": bool(rec["preprocessing"].get("eligibility"))
               and bool(rec["preprocessing"].get("known"))}]
    return {"stage": "plan", "recipe_fp": rec["fingerprint"], "checks": checks,
            "status": "PASS" if all(c["ok"] for c in checks) else "BLOCKED"}


def _flat_hashes(d) -> list:
    if isinstance(d, dict):
        return [x for v in d.values() for x in _flat_hashes(v)]
    return [d]


def audit_export(path: Path | None, rec: dict, data, eligible: np.ndarray, tol: float = 0.02,
                 expected_epsg: int = 32611) -> dict:
    """After writing: CRS, shape, transform, nodata pattern, value range, eligibility, known-fault pixels, area
    budget, sidecar parameters == recipe, file hash. Hard checks decide PASS / FAILED; facts are recorded."""
    from .boost_audit import file_hash

    checks = []

    def chk(name, ok, detail="", hard=True):
        checks.append({"check": name, "ok": bool(ok), "hard": hard, "detail": detail})

    if path is None or not Path(path).exists():
        chk("file exists", False, str(path))
        return {"status": "FAILED", "checks": checks, "file": None, "sha256": None, "recipe_fp": rec["fingerprint"]}
    import rasterio

    t = data.template_profile
    with rasterio.open(path) as src:
        a = src.read(1)
        meta = {"count": src.count, "dtype": src.dtypes[0], "epsg": src.crs.to_epsg() if src.crs else None,
                "shape": [int(src.height), int(src.width)], "nodata": src.nodata}
        chk("single float32 band", src.count == 1 and src.dtypes[0] == "float32", f"{src.count} x {src.dtypes[0]}")
        chk("CRS equals template", src.crs == t["crs"] and meta["epsg"] == int(expected_epsg), str(meta["epsg"]))
        chk("transform equals template", src.transform == t["transform"])
        chk("shape equals template", (src.height, src.width) == (t["height"], t["width"]), str(meta["shape"]))
    tv, valid = np.asarray(data.template_valid, bool), np.asarray(data.valid, bool)
    fin = np.isfinite(a)
    chk("nodata (NaN) exactly where the template is NaN", bool((~fin[~tv]).all()) and bool(fin[tv & valid].all()))
    v = a[fin]
    chk("values in [0, 1]", (not v.size) or (v.min() >= 0 and v.max() <= 1), f"{v.min() if v.size else ''}"
        f"..{v.max() if v.size else ''}")
    chk("binary values {0, 1}", (not v.size) or set(np.unique(v[:2_000_000]).tolist()) <= {0.0, 1.0}, hard=False)
    pos = np.nan_to_num(a) > 0
    el = np.asarray(eligible, bool)
    chk("no positive pixel outside the eligible budget domain", int((pos & ~el).sum()) == 0,
        f"{int((pos & ~el).sum())} px")
    chk("positive pixels on mapped faults (recorded; the scorer masks them)", True,
        f"{int((pos & np.asarray(data.labels, bool)).sum())} px", hard=False)
    fe = float((pos & el).sum() / max(int(el.sum()), 1))
    f = float(rec["policy"]["frac"])
    chk(f"area <= frac x (1 + {tol:g})", fe <= f * (1 + tol) + 1e-9, f"{fe:.5f} of eligible vs {f:g}")
    side = Path(path).with_suffix(".json")
    sc = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
    p = sc.get("parameters") or {}
    chk("sidecar parameters == recipe policy", bool(sc) and p.get("rep") == rec["policy"]["rep"] and
        p.get("frac") is not None and abs(float(p["frac"]) - f) < 1e-12 and p.get("blend") is not None and
        abs(float(p["blend"]) - float(rec["policy"]["blend"])) < 1e-12, json.dumps(p, default=str))
    chk("sidecar recipe fingerprint == recipe", sc.get("recipe_fp") == rec["fingerprint"])
    chk("sidecar features == recipe schema", list(sc.get("features") or []) == list(rec["feature_schema"]))
    sha = file_hash(Path(path), 1e9)
    hard_fail = [c["check"] for c in checks if c["hard"] and not c["ok"]]
    return {"status": "PASS" if not hard_fail else "FAILED", "failed": hard_fail, "checks": checks,
            "file": Path(path).name, "sha256": sha, "recipe_fp": rec["fingerprint"], "raster": meta,
            "area_frac_eligible": fe, "written": time.strftime("%Y-%m-%dT%H:%M:%S")}


# =============================================================================== export (Colab side)
def save_plan(p6, rec: dict, rel_dir: str) -> dict:
    """``<rel_dir>/<fp8>/export_plan.json`` (recipe + plan audit) BEFORE any file is generated."""
    pa = plan_audit(rec)
    p6.ws.save_json(json.loads(json.dumps({"recipe": rec, "plan_audit": pa}, default=B._js)),
                    f"{rel_dir}/{rec['fingerprint'][:8]}/export_plan.json")
    return pa


def write_validated(p6, rec: dict, q: np.ndarray, ledger: list, tol: float = 0.02) -> dict:
    """The single writer: ``m7.export._write`` (template finalise + validate; never overwrites) + ``audit_export``.
    A file already written for the same recipe fingerprint is reused (no new file, no new log row)."""
    from ..gems3 import submission as SUB
    from ..m7 import export as E

    name = f"p6r_{rec['candidate']}_{rec['fingerprint'][:8]}"
    sub = p6.ws.root / "submissions"
    data = p6.ctx.data
    el = SUB.eligible_mask(data, 0, "eligible")
    prev = sorted(sub.glob(f"{name}*.tif")) if sub.exists() else []
    if prev and prev[-1].with_suffix(".json").exists():
        info = json.loads(prev[-1].with_suffix(".json").read_text(encoding="utf-8"))
        return {**info, "reused": True, "audit": audit_export(prev[-1], rec, data, el, tol)}
    pol = rec["policy"]
    man = {"model_id": name, "method": "p6_recovery", "candidate": rec["candidate"], "kind": rec["kind"],
           "parameters": {"blend": pol["blend"], "rep": pol["rep"], "frac": pol["frac"], "ring": 0},
           "features": rec["feature_schema"], "frac_source": pol["selection_id"], "recipe_fp": rec["fingerprint"],
           "recipe": rec}
    try:
        info = E._write(p6.view, name, np.asarray(q, np.float32), ledger, man, 0)
    except AssertionError as e:                                # validate() raises listing every failed check
        files = sorted(sub.glob(f"{name}*.tif"))
        info = {**man, "submission": files[-1].name if files else None, "uploaded": False,
                "validation": {"checks_failed": [str(e)]}}
    path = sub / info["submission"] if info.get("submission") else None
    return {**info, "reused": False, "audit": audit_export(path, rec, data, el, tol)}


def w2_policy_map(p6, rec: dict) -> np.ndarray:
    """Full-region map of a W2 policy recipe from the frozen W2 full-region score (no refit): exactly the
    ``stack.full_region`` construction, with rep, frac and blend from ONE selection."""
    from ..gems3 import representation as R
    from ..gems3 import submission as SUB
    from .boost_audit import file_hash

    data = p6.ctx.data
    valid = np.asarray(data.valid, bool)
    fs = p6.ws.root / rec["model"]["score_source"]
    if file_hash(fs, 1e9) != rec["sources"]["w2_full_score"]:
        raise RuntimeError(f"{fs} changed since the plan was written (sha256 mismatch): export refused")
    score = read_band(fs)
    maps, _ = p6.view.full_maps()
    pr, pol = rec["preprocessing"], rec["policy"]
    F = R.fuse(R.rank01(maps[0], valid), R.rank01(maps[1], valid), float(pr["fusion_w"]))
    lam = float(pol["blend"])
    s = ((1.0 - lam) * F + lam * R.rank01(np.nan_to_num(score, nan=0.0), valid)).astype(np.float32)
    reps = p6.m9.sh["baseline"]["representations"]
    return R.apply(reps[pol["rep"]], s, valid, float(pol["frac"]), known=data.labels,
                   sigma_px=float(pr["sigma_px"]), eligible=SUB.eligible_mask(data, 0, "eligible"))


def read_band(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as s:
        return s.read(1).astype(np.float32)


def w2_recipe(rc, name: str, summary: dict) -> dict:
    """Recipe of the chosen W2 policy (frozen W2 model, its full-region score, the policy's all-zone cell)."""
    from .boost_audit import file_hash

    bst, p6 = rc.bst, rc.p6
    cell = summary["deploy_cell"]
    if cell is None:
        raise ValueError(f"{name}: no deployment cell")
    root = p6.ws.root
    fs = root / "stack" / "maps" / "full_score.tif"
    w = float(bst.sel[bst.prim][p6.cfg["probes"]["champion"]]["params_all"]["w"])
    sh = p6.m9.sh
    sid = f"{name}|{summary['objective']}|all_zones"
    return build_recipe(
        candidate=f"w2_{name}", kind="w2_policy",
        policy={"selection_id": sid, "space": json.dumps({k: v for k, v in rc.cfg["export_parity"]["policies"]
                                                          .get(name, {}).items()}, default=str),
                "objective": summary["objective"], "rep": cell["rep"], "frac": cell["frac"], "blend": cell["blend"],
                "grid_candidate": cell["candidate"]},
        model={"kind": "W2 LightGBM stacker (frozen; full-region score reused, no refit)",
               "score_source": "stack/maps/full_score.tif", "model_file": "stack/models/stack_full.joblib",
               "features": bst.w2_features(), "refit": False},
        sources={"w2_full_score": file_hash(fs, 1e9), "w2_candidate_json": file_hash(root / "stack" / "candidate.json"),
                 "w2_rows": bst.w2_inputs()["rows_sha256"], "benchmark_grid": file_hash(bst.root / "benchmark" /
                                                                                          "grid.csv")},
        feature_schema=bst.w2_features(),
        preprocessing={"ranking": "gems3.representation.rank01 inside data.valid", "fusion_w": w,
                       "fusion": "R.fuse(rank01(B0), rank01(B1), w) with the champion's all-zone w",
                       "blend": "s = (1 - blend) fusion + blend rank01(W2 score)", "sigma_px":
                       float(sh["evaluation"]["ridge_sigma_px"]), "representation": sh["baseline"]
                       ["representations"][cell["rep"]], "eligibility": "gems3.submission.eligible_mask(data, 0, "
                       "'eligible') (valid & template finite & not a known fault)", "known": "data.labels (every "
                       "mapped fault) passed to R.apply", "ring_px": 0})


# =============================================================================== Colab side: parity tables
def unit_overlap(bst) -> pd.DataFrame:
    """Pixel overlap of every pair of unit evaluation masks on the full grid (selection exclusion)."""
    H, W = bst.p6.ctx.data.shape
    us = bst.units()
    rows = []
    for a in us:
        ma = np.zeros((H, W), bool)
        ma[a.sl] = a.ev
        for b in us:
            mb = np.zeros((H, W), bool)
            mb[b.sl] = b.ev
            rows.append({"task_a": a.task, "zone_a": int(a.zone), "task_b": b.task, "zone_b": int(b.zone),
                         "overlap_px": int((ma & mb).sum())})
    return pd.DataFrame(rows)


def parity_tables(rc, cell: dict | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training vs inference construction on every unit (Colab; loads the full-region features once):
    W2 score (held-out map vs full-region score crop), each W2 feature (unit construction vs full-region
    construction, as consumed by the model), the sampled training rows vs the inference value at the same pixels,
    and the binary map at the deployment cell built from each side."""
    from ..gems3 import representation as R

    bst, p6 = rc.bst, rc.p6
    pc = rc.cfg["export_parity"]["parity"]
    seed = int(p6.cfg["seed"])
    tf = tuple(pc["top_fracs"])
    data = p6.ctx.data
    valid = np.asarray(data.valid, bool)
    names = bst.w2_features()
    fs = p6.ws.root / "stack" / "maps" / "full_score.tif"
    full_score = read_band(fs) if fs.exists() else None
    raw_full = bst.full_raw(names, None)
    maps = raw_full.pop("_maps")
    inf_c = {f: (ST.rank_in(raw_full[f], valid) if f in ST.RANKED else np.asarray(raw_full[f], np.float32))
             for f in names}
    w_all = float(bst.sel[bst.prim][p6.cfg["probes"]["champion"]]["params_all"]["w"])
    T0 = bst.rows0()
    rows, shift = [], []
    known = set(ST.KNOWN)
    for u in bst.units():
        ev = u.ev
        base = {"unit": u.key, "task": u.task, "zone": int(u.zone)}
        if full_score is not None:
            tr_s, inf_s = ST.rank_in(bst.w2_map(u), ev), ST.rank_in(full_score[u.sl], ev)
            r = agreement(tr_s, inf_s, ev, tf, int(pc["sample_px"]), seed)
            rows.append({**base, "check": "w2_score", "feature": "w2_score", **r,
                         "status": classify_parity(r, pc), "note": "held-out cross-fitted W2 map vs the full-region "
                         "W2 score crop, both ranked inside the unit"})
        else:
            rows.append({**base, "check": "w2_score", "feature": "w2_score", "status": "BLOCKED",
                         "note": "stack/maps/full_score.tif absent: the exported score cannot be compared"})
        raw_u = bst.raw(u)
        m = ((T0.task == u.task) & (T0.zone == int(u.zone))).to_numpy()
        rr, cc = T0.r_full.to_numpy()[m].astype(int), T0.c_full.to_numpy()[m].astype(int)
        for f in names:
            if f not in raw_u:
                rows.append({**base, "check": "feature", "feature": f, "status": "BLOCKED",
                             "note": "absent from the unit features"})
                continue
            tr = ST.rank_in(raw_u[f], ev) if f in ST.RANKED else np.asarray(raw_u[f], np.float32)
            inf = inf_c[f][u.sl]
            exp = f in known
            r = agreement(tr, inf, ev, tf, int(pc["sample_px"]), seed)
            rows.append({**base, "check": "feature", "feature": f, **r, "status": classify_parity(r, pc, exp),
                         "expected_by_design": exp, "note": ("permitted known map (training) vs every mapped fault "
                                                            "(inference): by design" if exp else
                                                            ("ranked inside the unit (training) vs inside the valid "
                                                             "footprint (inference)" if f in ST.RANKED else
                                                             "raw value"))})
            if m.any():
                vt, vi = T0.loc[m, f].to_numpy(np.float64), inf_c[f][rr, cc].astype(np.float64)
                ok = np.isfinite(vt) & np.isfinite(vi)
                from scipy.stats import spearmanr

                rho = float(spearmanr(vt[ok], vi[ok])[0]) if ok.sum() >= 10 and vt[ok].std() > 0 and \
                    vi[ok].std() > 0 else np.nan
                shift.append({**base, "feature": f, "comparison": "training rows vs inference value at the same pixels",
                              "n_rows": int(m.sum()), "spearman_same_px": rho,
                              "mean_abs_diff": float(np.abs(vt[ok] - vi[ok]).mean()) if ok.any() else np.nan,
                              **B.quantiles(vt, prefix="train_q"), **B.quantiles(vi, prefix="infer_q"),
                              "nan_rows_train": int((~np.isfinite(vt)).sum()),
                              "nan_rows_infer": int((~np.isfinite(vi)).sum()),
                              "expected_by_design": exp})
        if cell is not None and full_score is not None:
            lam = float(cell["blend"])
            F_tr = bst.fusion(u)
            F_inf = R.fuse(R.rank01(maps[0][u.sl], ev), R.rank01(maps[1][u.sl], ev), w_all)
            s_tr = ((1 - lam) * F_tr + lam * bst.stack_rank(u, bst.w2_map(u))).astype(np.float32)
            s_inf = ((1 - lam) * F_inf + lam * R.rank01(np.nan_to_num(full_score[u.sl], nan=0.0), ev)).astype(np.float32)
            ma = map_agreement(bst._apply(u, s_tr, cell["rep"], cell["frac"]),
                               bst._apply(u, s_inf, cell["rep"], cell["frac"]))
            rows.append({**base, "check": "map_agreement", "feature": "deploy_cell", **ma, "cell": json.dumps(cell),
                         "status": "PASS" if (ma["jaccard"] or 0) >= 0.5 else "WARNING",
                         "note": "binary map at the deployment cell: training construction vs inference construction"})
            rows.append({**base, "check": "fusion_weight", "feature": "fusion_w", "w_train": bst.champion_w(u),
                         "w_infer": w_all, "status": "PASS" if abs(bst.champion_w(u) - w_all) < 1e-9 or lam == 1.0
                         else "WARNING", "note": "irrelevant at blend 1.0" if lam == 1.0 else "per-zone vs all-zone w"})
        bst._raw.pop(u.key, None)
        u.release()
    return pd.DataFrame(rows), pd.DataFrame(shift)


# =============================================================================== Colab side: stage
def run(rc) -> dict:
    """X1 stage (``rc.export_parity()``): tables from the boost benchmark grid, parity, plan; no file is written."""
    bst, p6 = rc.bst, rc.p6
    cfg, gates = rc.cfg["export_parity"], rc.gates()
    G = rc.benchmark_grid()
    stage = "export_parity"
    fp = rc.fp(stage, cfg, gates, rc.cfg["tasks"], rc.input_hashes())
    if rc._done(stage, fp):
        return rc.load(stage)
    t0 = rc._begin(stage, fp)
    zones = sorted(G.zone.unique())
    ov = unit_overlap(bst)
    ex, feasible = overlap_exclusion(ov, zones, bst.prim)
    c2 = dict(cfg)
    if not feasible and cfg["objective"] in ("disc_geo_floor", "joint"):
        c2["objective"] = "disc"                                 # geography of every zone overlaps: geo in the
        c2["sensitivity_objectives"] = [cfg["objective"]] + list(cfg.get("sensitivity_objectives", []))  # selection
    T = policy_tables(G, c2, bst.prim, gates, rc.cfg["tasks"], ex, bst.load_table("nested_selection", "benchmark"))
    rc.save_table(T["comparison"].pipe(lambda d: pd.DataFrame([B._flat(r) for r in d.to_dict("records")])),
                  "policy_comparison", stage)
    rc.save_table(T["selection"], "nested_policy_selection", stage)
    rc.save_table(T["per_zone"].drop(columns=["systems"], errors="ignore"), "per_zone_metrics", stage)
    rc.save_table(ov, "unit_overlap", stage)
    rc.fig_frontier(T["frontier"], gates, stage)
    chosen, cs = T["chosen"], T["chosen_summary"] or {}
    par, sh = (parity_tables(rc, cs.get("deploy_cell")) if cfg.get("parity_numeric") else
               (pd.DataFrame([{"check": "numeric", "feature": "*", "status": "NOT_RUN",
                               "note": "export_parity.parity_numeric = false"}]), pd.DataFrame()))
    rc.save_table(par, "train_inference_parity", stage)
    rc.save_table(sh, "feature_distribution_shift", stage)
    plan, rec = {"status": "NOT_PLANNED"}, None
    if chosen:
        try:
            rec = w2_recipe(rc, chosen, cs)
            plan = save_plan(p6, rec, rc.sub("export_parity") + "/plans")
        except Exception as e:                                   # a plan failure is recorded, never hidden
            plan = {"status": "BLOCKED", "reason": f"{type(e).__name__}: {e}"}
    rc.save_json({"chosen_policy": chosen, "objective": T["objective"], "deploy_cell": cs.get("deploy_cell"),
                  "recipe": rec, "plan_audit": plan, "rule": "rep, frac and blend of ONE all-zone selection under the "
                  "predeclared objective; nothing is copied from a held-out zone"}, stage, "export_policy")
    rc.save_json({"status": "NOT_EXPORTED", "plan_audit": plan, "note": "the file is written only by the export "
                  "stage after selection PASS (RECOVERY_RUN['export'])"}, stage, "export_audit")
    blocked_par = par[par.status == "BLOCKED"] if len(par) else par
    ref = float(gates["w2_recorded_disc"])
    d = cs.get("disc_dti", np.nan)
    verdict = ("PASS" if chosen and np.isfinite(d) and d > ref and cs.get("meets_geo_floor") and
               cs.get("meets_worst_floor") and not len(blocked_par) else "FAIL" if chosen else "BLOCKED")
    dec = {"stage": stage, "verdict": verdict, "chosen_policy": chosen, "objective": T["objective"],
           "objective_predeclared": cfg["objective"], "overlap_exclusion_feasible": feasible,
           "chosen_summary": cs, "chosen_reasons": T["chosen_reasons"], "meta_nested": T["meta"],
           "w2_recorded_disc": ref, "gain_vs_w2_recorded": (d - ref) if np.isfinite(d) else None,
           "parity_blocked": blocked_par[["unit", "feature"]].to_dict("records") if len(blocked_par) else [],
           "plan_status": plan.get("status"),
           "uploaded_cell": T["comparison"][T["comparison"].role == "historical"][
               ["policy", "objective", "disc_dti", "geo_dti"]].to_dict("records"),
           "note": "nested offline numbers, not leaderboard scores; META = honest estimate of choosing the policy"}
    rc.save_json(dec, stage, "decision")
    rc._finish(stage, fp, t0, {"inputs": rc.input_hashes(), "exclusion": {str(k): {t: sorted(v) for t, v in e.items()}
                                                                          for k, e in ex.items()}},
               {"chosen": chosen, "verdict": verdict}, schema=bst.w2_features(),
               selection={"objective": T["objective"], "exclusion_feasible": feasible})
    return rc.load(stage)
