"""Stage I: DTI-aware selection at matched area, fixed-policy model choice, promotion gates.

* Score map of a model on a unit: segment score on the segment's pixels (``geomoe.evaluate.score_map``), plus (when
  ``selection.partial_segments``) a 1e-9 x pixel-W2-rank tie-break INSIDE each segment, so a budget can end inside a
  segment (partial segment) without changing the order between segments.
* Every model is rasterised with the SAME predeclared representations x budgets (config ``selection``) on the unit's
  eligible domain, scored by the official scorer, systems by the canonical rule. ``area_frac_eligible`` (hard pixels)
  and ``mass_frac_eligible`` (sum of soft values) are both recorded; a soft representation is compared at matched
  selection area (same top-k before the soft operation).
* Model choice: ``geomoe.evaluate.fixed_policy`` (one rule, development units only). The reference is the frozen
  submitted W2 policy (``geomoe`` stage 05 rows, no selection) and G0 (segment W2, same construction as the
  challengers).
* Geographic guardrail: V3 pooled DTI of the challenger under the EXACT frozen W2 cell (same representation dict as
  the submission factory, same budget) vs the frozen W2 V3 DTI.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

PROMOTE, PROMISING, RETAIN, NOT_EVAL = ("PROMOTE_CANDIDATE", "PROMISING_SCREENING_ONLY", "RETAIN_W2",
                                        "NOT_EVALUATED")


# =============================================================================== pure
def partial_map(seg_map: np.ndarray, w2_pixel: np.ndarray | None, eps: float = 1e-9) -> np.ndarray:
    """Segment map + eps x pixel W2 rank inside each segment (NaN stays NaN)."""
    s = np.asarray(seg_map, np.float64)
    if w2_pixel is None:
        return s
    return np.where(np.isfinite(s), s + eps * np.nan_to_num(np.asarray(w2_pixel, np.float64), nan=0.0), np.nan)


def dti(tp, fp, fn) -> float:
    return float(tp / (tp + 0.2 * fp + 0.8 * fn + 1e-6))


def attach_systems(P: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Per-unit fixed-policy rows + the recovered systems of the chosen cell (``fixed_policy`` keeps components
    only)."""
    if not len(P) or "systems" in P:
        return P
    k = ["unit", "model", "rep", "frac"]
    g = grid[k + ["systems", "n_recovered", "n_systems"]].drop_duplicates(k)
    return P.merge(g, on=k, how="left")


def systems_of(rows: pd.DataFrame) -> dict:
    return {r.unit: {int(float(x)) for x in str(r.systems).split(";") if x not in ("", "nan")}
            for r in rows.itertuples()}


def promotion(model: str, strictness: str, S: pd.DataFrame, P: pd.DataFrame, ref_S: pd.DataFrame,
              ref_P: pd.DataFrame, geo_m: dict | None, geo_ref: dict | None, pc: dict, checks: dict,
              deferred=("export_parity_verified",)) -> dict:
    """Promotion decision of one challenger.

    ``S`` / ``P``: fixed-policy summary / per-unit rows of the challenger; ``ref_*``: frozen-W2 rows; ``geo_*``:
    {'dti': ...} V3 pooled under the exact frozen cell; ``checks``: {gate: bool} (provenance / upstream honesty /
    export parity; a missing gate counts as not verified). ``deferred`` gates can only be verified on the written file
    (Stage K): they are reported DEFERRED and re-checked there; a PROMOTE_CANDIDATE never becomes a validated file
    without them."""
    out = {"model": model, "strictness": strictness, "gates": {}, "reasons": []}
    s1, r1 = S[S.scenario == "V1"], ref_S[ref_S.scenario == "V1"]
    if not len(s1) or not len(r1):
        return {**out, "status": NOT_EVAL, "reasons": ["V1 (discovery) rows missing for the challenger or W2"]}
    g = float(s1.dti.iloc[0]) - float(r1.dti.iloc[0])
    out.update(disc_dti=float(s1.dti.iloc[0]), disc_dti_w2=float(r1.dti.iloc[0]), disc_gain=g)
    gates = {"disc_gain >= min_disc_gain": g >= float(pc["min_disc_gain"])}
    pm = P[P.scenario == "V1"].set_index("zone").dti_heldout
    pr = ref_P[ref_P.scenario == "V1"].set_index("zone").dti_heldout
    common = sorted(set(pm.index) & set(pr.index))
    drops = {int(z): float(pm[z] - pr[z]) for z in common}
    out["per_zone_delta"] = drops
    gates["no zone drops more than max_worst_zone_drop"] = bool(common) and min(drops.values()) >= -float(
        pc["max_worst_zone_drop"])
    if geo_m is not None and geo_ref is not None:
        gd = float(geo_m["dti"]) - float(geo_ref["dti"])
        out.update(geo_dti=float(geo_m["dti"]), geo_dti_w2=float(geo_ref["dti"]), geo_delta=gd)
        gates["geo (exact frozen cell) >= -max_geo_drop"] = gd >= -float(pc["max_geo_drop"])
    else:
        gates["geo (exact frozen cell) >= -max_geo_drop"] = False
        out["reasons"].append("geographic guardrail not measured")
    sm, sr = systems_of(P[P.scenario == "V1"]), systems_of(ref_P[ref_P.scenario == "V1"])
    new = sum(len(sm.get(u, set()) - sr.get(u, set())) for u in sm)
    lost = sum(len(sr.get(u, set()) - sm.get(u, set())) for u in sr)
    out.update(new_systems=new, lost_systems=lost, net_systems=new - lost)
    gates["net extra systems >= min_extra_systems"] = (new - lost) >= int(pc["min_extra_systems"])
    req = [k for k in pc["require"] if k not in deferred]
    for k in req:
        gates[k] = bool(checks.get(k, False))
    out["gates"] = gates
    out["deferred_gates"] = {k: "DEFERRED_TO_STAGE_K" for k in pc["require"] if k in deferred}
    out["recommended_gain_met"] = g >= float(pc["recommend_disc_gain"])
    metric = [k for k in gates if k not in pc["require"]]
    metric_ok = all(gates[k] for k in metric)
    verified = all(gates[k] for k in req)
    if metric_ok and strictness != "STRICT_LABEL_FREE":
        out["status"] = PROMISING
        out["reasons"].append(f"metric gates pass but the evaluation is {strictness}: no validated candidate "
                              "(strict nesting required)")
    elif metric_ok and verified:
        out["status"] = PROMOTE
    elif metric_ok:
        out["status"] = RETAIN
        out["reasons"].append("verification gates not all passed: " + ", ".join(k for k in req if not gates[k]))
    else:
        out["status"] = RETAIN
        out["reasons"].append("failed: " + ", ".join(k for k in metric if not gates[k]))
    return out


def decision_table(decisions: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**{k: v for k, v in d.items() if k not in ("gates", "per_zone_delta", "deferred_gates")},
                          "gates": json.dumps(d.get("gates", {})), "deferred_gates": json.dumps(d.get(
                              "deferred_gates", {})), "per_zone_delta": json.dumps(d.get("per_zone_delta", {}))}
                         for d in decisions])


# =============================================================================== Colab side
def apply_rep(G, u, smap: np.ndarray, rep: dict, frac: float, crest=None) -> np.ndarray:
    from ..gems3 import representation as R

    return R.apply(rep, smap, u.ev, float(frac), known=u.known, crest=crest,
                   sigma_px=float(G.sh["evaluation"]["ridge_sigma_px"]), eligible=G.elig(u))


def grid_rows(G, su, smap: np.ndarray, model: str, reps: dict, budgets, extra: dict | None = None) -> list[dict]:
    from ..gems3 import representation as R
    from ..geomoe.scenarios import system_recovery

    u = su.unit
    el = G.elig(u)
    crest = R.crest_map(np.nan_to_num(smap, nan=0.0), u.ev, float(G.sh["evaluation"]["ridge_sigma_px"])) \
        if any(r.get("select") == "crest" for r in reps.values()) else None
    rows = []
    for name, rep in reps.items():
        for f in budgets:
            q = apply_rep(G, u, smap, rep, f, crest if rep.get("select") == "crest" else None)
            r = G.score(u, q)
            hs = system_recovery(u, q, su.sysmap, G.sys_support, G.sys_frac)
            rec = sorted(int(x) for x in hs.system_id[hs.recovered]) if len(hs) else []
            rows.append({**r, "scenario": su.scenario, "unit": su.key, "model": model, "rep": name, "frac": float(f),
                         "mass_frac_eligible": float(np.asarray(q, float)[el].sum() / max(int(el.sum()), 1)),
                         "n_systems": int(len(hs)), "n_recovered": len(rec), "systems": ";".join(map(str, rec)),
                         "role": "heldout", **(extra or {})})
    return rows


def exact_policy_row(G, su, smap: np.ndarray, model: str, cell: tuple) -> dict:
    """The challenger under the EXACT frozen W2 cell (submission-factory representation dict and budget)."""
    from ..geomoe import evaluate as EV
    from ..geomoe.scenarios import system_recovery

    q = EV.policy_q(G, su.unit, smap, str(cell[0]), float(cell[1]))
    r = G.score(su.unit, q)
    hs = system_recovery(su.unit, q, su.sysmap, G.sys_support, G.sys_frac)
    rec = sorted(int(x) for x in hs.system_id[hs.recovered]) if len(hs) else []
    return {**r, "scenario": su.scenario, "unit": su.key, "model": model, "rep": str(cell[0]), "frac": float(cell[1]),
            "n_systems": int(len(hs)), "n_recovered": len(rec), "systems": ";".join(map(str, rec)),
            "role": "exact_frozen_cell"}
