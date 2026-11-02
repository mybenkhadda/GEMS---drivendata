"""Stage C: equal-budget candidate recall of experts and unions.

Every construction is scored on the SAME candidate inventory (GeoMoE stage 04 segments of the unit), rasterised with
the SAME representation (config ``recall.rep``) at the SAME share of the unit's eligible domain, by the official
scorer, with the canonical system-recovery rule (``geomoe.scenarios.system_recovery``: support >= 0.5 on >= 50 % of a
system's scored pixels). Constructions:
  * expert X: segment score = mean of its member layers' segment means (ranked inside the unit, label-free);
  * union (``max_rank``): max over member experts of their unit ranks, ONE budget for the union;
  * union (``quota``): each of the n members gets budget / n, selections OR-ed (area <= budget);
  * anchors: the pixel W2 map at the same rep / budget and (frozen cell only) the submitted W2 policy.
Final budgets (0.5 / 1 / 2 %) are deployable areas; 5 / 10 % rows are recall CEILINGS (role ``diagnostic``) and are
never compared as deployable results. The oracle ceiling (truth-ordered segments) is read from GeoMoE stage 04
(``candidates/oracle_curve_parts.parquet``) and stays a diagnostic upper bound.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANCHOR_PIXEL, ANCHOR_FROZEN = "W2_pixel_map", "W2_submitted_policy"


# =============================================================================== pure
def unit_rank(v) -> np.ndarray:
    """Average-rank ECDF in (0, 1] of the finite values of one unit; NaN stays NaN (label-free)."""
    x = np.asarray(v, float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if ok.any():
        out[ok] = pd.Series(x[ok]).rank(method="average").to_numpy() / ok.sum()
    return out


def expert_scores(T: pd.DataFrame, experts: dict) -> pd.DataFrame:
    """{expert: unit rank of the mean of its available member segment means} for the rows of ONE unit."""
    out = {}
    for name, members in experts.items():
        cols = [f"mean_{m}" for m in members if f"mean_{m}" in T]
        if not cols:
            out[name] = np.full(len(T), np.nan)
            continue
        with np.errstate(invalid="ignore"):
            raw = np.nanmean(T[cols].to_numpy(float), axis=1) if len(T) else np.zeros(0)
        out[name] = unit_rank(raw)
    return pd.DataFrame(out, index=T.index)


def union_max_rank(E: pd.DataFrame, members: list[str]) -> np.ndarray:
    have = [m for m in members if m in E]
    if not have:
        return np.full(len(E), np.nan)
    with np.errstate(invalid="ignore"):
        V = E[have].to_numpy(float)
        return np.where(np.isfinite(V).any(1), np.nanmax(np.where(np.isfinite(V), V, -np.inf), 1), np.nan)


def constructions(rc: dict) -> dict:
    """{name: (kind, members)} of every construction of Stage C (experts, unions, unions minus one member)."""
    out = {k: ("expert", [k]) for k in rc["experts"]}
    for u, mem in rc["unions"].items():
        out[u] = ("union_max_rank", list(mem))
        if rc.get("quota_rule"):
            out[f"{u}|quota"] = ("union_quota", list(mem))
    full = rc["unions"].get("Full")
    if full:
        for m in full:
            if m != "W2":
                out[f"Full-minus-{m}"] = ("union_max_rank", [x for x in full if x != m])
    return out


def recovered_sets(R: pd.DataFrame) -> dict:
    """{(unit, construction, frac): set of recovered system ids}."""
    out = {}
    for r in R.itertuples():
        out[(r.unit, r.construction, float(r.frac))] = {int(float(x)) for x in str(r.systems).split(";")
                                                       if x not in ("", "nan")}
    return out


def new_lost(R: pd.DataFrame, ref: str = "W2") -> pd.DataFrame:
    """Per (unit, construction, frac): systems recovered that the reference (same construction family, same budget)
    misses (new) and the reverse (lost)."""
    S = recovered_sets(R)
    rows = []
    for (unit, c, f), s in S.items():
        base = S.get((unit, ref, f))
        if base is None or c == ref:
            continue
        rows.append({"unit": unit, "construction": c, "frac": f, "n_new": len(s - base), "n_lost": len(base - s),
                     "new_ids": ";".join(map(str, sorted(s - base))), "lost_ids": ";".join(map(str, sorted(base - s))),
                     "n_recovered": len(s), "n_recovered_ref": len(base)})
    return pd.DataFrame(rows)


def pooled(R: pd.DataFrame) -> pd.DataFrame:
    """Pooled components per (scenario, construction, frac, role): DTI from summed components, recovered counts."""
    if not len(R):
        return R
    g = R.groupby(["scenario", "construction", "kind", "frac", "role"]).agg(
        tp_w=("tp_w", "sum"), fp_w=("fp_w", "sum"), fn_w=("fn_w", "sum"), n_truth=("n_truth", "sum"),
        n_systems=("n_systems", "sum"), n_recovered=("n_recovered", "sum"),
        area_frac_eligible=("area_frac_eligible", "mean"), n_units=("unit", "nunique")).reset_index()
    g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
    g["system_recall"] = g.n_recovered / g.n_systems.clip(lower=1)
    return g


def marginal(P: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-expert-out marginal value: Full minus (Full without X), per scenario / budget."""
    rows = []
    for (sc, f), g in P[P.construction.astype(str).str.startswith("Full")].groupby(["scenario", "frac"]):
        full = g[g.construction == "Full"]
        if not len(full):
            continue
        fr = full.iloc[0]
        for r in g[g.construction.astype(str).str.startswith("Full-minus-")].itertuples():
            rows.append({"scenario": sc, "frac": f, "expert": r.construction.replace("Full-minus-", ""),
                         "dti_full": fr.dti, "dti_without": r.dti, "marginal_dti": fr.dti - r.dti,
                         "recovered_full": fr.n_recovered, "recovered_without": r.n_recovered,
                         "marginal_systems": int(fr.n_recovered - r.n_recovered)})
    return pd.DataFrame(rows)


def recovery_matrix(R: pd.DataFrame, units_systems: dict, targets: dict | None = None) -> pd.DataFrame:
    """Long matrix: one row per (unit, system, construction, frac) with ``recovered``; ``target`` flags the Stage-B
    investigation systems. ``units_systems`` = {unit: [system ids scored in that unit]}."""
    S = recovered_sets(R)
    tgt = {(f"V1:{int(z)}", int(s)) for z, ids in (targets or {}).items() for s in ids}
    rows = []
    for (unit, c, f), rec in S.items():
        for s in units_systems.get(unit, []):
            rows.append({"unit": unit, "system_id": int(s), "construction": c, "frac": f, "recovered": int(s) in rec,
                         "target": (unit, int(s)) in tgt})
    return pd.DataFrame(rows)


# =============================================================================== Colab side
def unit_rows(G, su, lab: np.ndarray, T: pd.DataFrame, rc: dict, rep: dict, fracs: list, roles: dict,
              w2_pixel: np.ndarray | None, frozen: tuple | None, frozen_map: np.ndarray | None) -> list[dict]:
    """Score every construction x budget on one unit. ``rep`` = representation dict, ``roles`` = {frac: role}."""
    from ..gems3 import representation as R
    from ..geomoe import evaluate as EV
    from ..geomoe.scenarios import system_recovery

    u = su.unit
    el = G.elig(u)
    sig = float(G.sh["evaluation"]["ridge_sigma_px"])
    E = expert_scores(T, rc["experts"])
    smaps = {}

    def smap_of(score):
        return EV.score_map(lab, T.label.to_numpy(), np.asarray(score, float), T.n_px.to_numpy())

    for name in rc["experts"]:
        smaps[name] = smap_of(E[name].to_numpy())

    def q_of(sm, f):
        return R.apply(rep, sm, u.ev, float(f), known=u.known, sigma_px=sig, eligible=el)

    def row(q, name, kind, f, extra=None):
        r = G.score(u, q)
        hs = system_recovery(u, q, su.sysmap, G.sys_support, G.sys_frac)
        rec = sorted(int(x) for x in hs.system_id[hs.recovered]) if len(hs) else []
        return {**r, "scenario": su.scenario, "zone": su.zone, "unit": su.key, "construction": name, "kind": kind,
                "frac": float(f), "role": roles.get(float(f), "final"), "n_systems": int(len(hs)),
                "n_recovered": len(rec), "systems": ";".join(map(str, rec)), **(extra or {})}

    rows = []
    for name, (kind, mem) in constructions(rc).items():
        if kind == "expert":
            sm = smaps[name]
        elif kind == "union_max_rank":
            sm = smap_of(union_max_rank(E, mem))
        else:
            sm = None
        avail = [m for m in mem if m in E and np.isfinite(E[m].to_numpy()).any()]
        for f in fracs:
            if kind == "union_quota":
                parts = [q_of(smaps[m], float(f) / max(len(avail), 1)) for m in avail]
                q = np.max(np.stack(parts), 0) if parts else np.zeros(u.ev.shape, np.float32)
            else:
                if not np.isfinite(sm).any():
                    rows.append({"scenario": su.scenario, "zone": su.zone, "unit": su.key, "construction": name,
                                 "kind": kind, "frac": float(f), "role": roles.get(float(f), "final"),
                                 "status": "UNAVAILABLE (no member layer on this unit)"})
                    continue
                q = q_of(sm, f)
            rows.append(row(q, name, kind, f, {"members_available": ";".join(avail), "status": "MEASURED"}))
    if w2_pixel is not None:
        for f in fracs:
            rows.append(row(q_of(w2_pixel, f), ANCHOR_PIXEL, "anchor", f, {"status": "MEASURED",
                                                                           "members_available": "w2 pixel map"}))
    if frozen is not None and frozen_map is not None:
        q = EV.policy_q(G, u, frozen_map, frozen[0], frozen[1])
        rows.append(row(q, ANCHOR_FROZEN, "anchor_frozen", frozen[1], {"status": "MEASURED", "rep_used": frozen[0],
                                                                       "members_available": "submitted policy"}))
    return rows
