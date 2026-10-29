"""Predeclared decisions (pure): generation-vs-ranking gate, model selection, the single next action.

Every rule reads persisted tables and the config only; nothing here looks at a public score or an oracle row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENERATION, RANKING, SATURATED, INCONCLUSIVE = "GENERATION", "RANKING", "NEITHER_DOMINANT", "INCONCLUSIVE"
RETAIN_W2 = "RETAIN_W2"
INDEPENDENT = ("E2_potential", "E3_structural", "E4_terrain_network")


def _f(x, default=np.nan) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


# =============================================================================== coverage gate (stage 06)
def coverage_summary(C: pd.DataFrame, U: pd.DataFrame, w2_sys: pd.DataFrame) -> pd.DataFrame:
    """Per scenario: systems, coverage share of each pool, W2 deployed recovery, unique systems per family.

    ``C``: system_coverage rows (scenario, unit, system_id, pool, covered); ``U``: unique_coverage rows;
    ``w2_sys``: W2 (A_pixel) nested held-out recovery (scenario, unit, system_id, recovered)."""
    rows = []
    for sc, g in C.groupby("scenario"):
        n = g[["unit", "system_id"]].drop_duplicates().shape[0]
        r = {"scenario": sc, "n_systems": int(n)}
        for pool, h in g.groupby("pool"):
            r[f"cov_{pool}"] = float(h.covered.mean()) if len(h) else np.nan
        w = w2_sys[w2_sys.scenario == sc] if len(w2_sys) else w2_sys
        r["w2_recovered_share"] = float(w.recovered.sum() / max(n, 1)) if len(w) else np.nan
        u = U[U.scenario == sc] if len(U) else U
        for f in ("E1_learned", *INDEPENDENT):
            r[f"unique_{f}"] = int(u[f"unique_{f}"].sum()) if len(u) and f"unique_{f}" in u else 0
        if len(u):
            ind = np.zeros(len(u), bool)
            for f in INDEPENDENT:
                if f"covered_{f}" in u:
                    ind |= u[f"covered_{f}"].to_numpy(bool)
            r["independent_not_E1"] = int((ind & ~u["covered_E1_learned"].to_numpy(bool)).sum())
        else:
            r["independent_not_E1"] = 0
        rows.append(r)
    return pd.DataFrame(rows)


def coverage_decision(S: pd.DataFrame, gc: dict, primary: str) -> dict:
    """GENERATION when the unified pool covers too few systems; RANKING when it covers enough but W2's deployed map
    recovers far fewer; NEITHER_DOMINANT otherwise. Overall = majority over scenarios (primary breaks ties).
    ``geology_unique``: independent families cover systems no learned source covers, enough to matter."""
    per = {}
    for r in S.to_dict("records"):
        pool, w2 = _f(r.get("cov_inventory")), _f(r.get("w2_recovered_share"))
        if not np.isfinite(pool) or not r.get("n_systems"):
            v = INCONCLUSIVE
        elif pool < float(gc["min_pool_coverage"]):
            v = GENERATION
        elif np.isfinite(w2) and pool - w2 >= float(gc["min_ranking_gap"]):
            v = RANKING
        else:
            v = SATURATED
        uq = int(r.get("independent_not_E1", 0))
        per[r["scenario"]] = {"verdict": v, "pool_coverage": pool, "w2_recovered": w2, "gap": pool - w2,
                              "n_systems": int(r.get("n_systems", 0)), "independent_not_E1": uq,
                              "independent_share": uq / max(int(r.get("n_systems", 0)), 1)}
    votes = [p["verdict"] for p in per.values() if p["verdict"] != INCONCLUSIVE]
    if not votes:
        overall = INCONCLUSIVE
    else:
        cnt = pd.Series(votes).value_counts()
        top = cnt[cnt == cnt.max()].index.tolist()
        overall = top[0] if len(top) == 1 else (per.get(primary, {}).get("verdict") if per.get(primary, {}).get(
            "verdict") in top else sorted(top)[0])
    uniq = any(p["independent_not_E1"] >= int(gc["min_unique_systems"]) and
               p["independent_share"] >= float(gc["min_unique_share"]) for p in per.values())
    return {"overall": overall, "per_scenario": per, "geology_unique": bool(uniq),
            "rule": {"GENERATION": f"inventory coverage < {gc['min_pool_coverage']}",
                     "RANKING": f"coverage >= {gc['min_pool_coverage']} and coverage - W2 recovered >= "
                                f"{gc['min_ranking_gap']}", "NEITHER_DOMINANT": "otherwise",
                     "geology_unique": f">= {gc['min_unique_systems']} systems and >= {gc['min_unique_share']:.0%} "
                                       "covered by E2/E3/E4 and by no E1 source, in at least one scenario"},
            "learned_gate_allowed": overall in (RANKING, SATURATED)}


# =============================================================================== model selection
def model_checks(S: pd.DataFrame, P: pd.DataFrame, model: str, ref: str, sc_list, sel: dict) -> dict:
    """Gains of ``model`` over ``ref`` per scenario (summary DTI + paired bootstrap) and the W2-rule verdict."""
    out = {"model": model, "reference": ref, "per_scenario": {}}
    imp = drop = 0
    for sc in sc_list:
        a = S[(S.scenario == sc) & (S.model == model)]
        b = S[(S.scenario == sc) & (S.model == ref)]
        if not len(a) or not len(b):
            out["per_scenario"][sc] = {"status": "MISSING"}
            continue
        d = _f(a.dti.iloc[0]) - _f(b.dti.iloc[0])
        wd = _f(a.worst_zone_dti.iloc[0]) - _f(b.worst_zone_dti.iloc[0])
        p = P[(P.scenario == sc) & (P.method == model) & (P.reference == ref)] if len(P) else P
        pp = _f(p.p_delta_positive.iloc[0]) if len(p) else np.nan
        out["per_scenario"][sc] = {"delta_dti": d, "delta_worst": wd, "p_delta_positive": pp,
                                   "ci_lo": _f(p.ci_lo.iloc[0]) if len(p) else np.nan,
                                   "ci_hi": _f(p.ci_hi.iloc[0]) if len(p) else np.nan}
        imp += int(d >= float(sel["min_gain_vs_w2"]))
        drop += int(d < -float(sel["max_scenario_drop"]) or wd < -float(sel["max_worst_zone_drop"]))
    out.update(n_improved=imp, n_dropped=drop,
               beats_w2=bool(imp >= int(sel["min_scenarios_improved"]) and drop == 0))
    return out


def upgrade(S: pd.DataFrame, P: pd.DataFrame, model: str, base: str, sc_list, gain: float, sel: dict) -> dict:
    """Does ``model`` beat ``base`` by >= gain with P(delta > 0) >= min_p in >= moe_min_scenarios scenarios, without
    losing more than max_scenario_drop anywhere?"""
    c = model_checks(S, P, model, base, sc_list, {**sel, "min_gain_vs_w2": gain})
    ok = [sc for sc, v in c["per_scenario"].items() if _f(v.get("delta_dti")) >= gain and
          _f(v.get("p_delta_positive"), 0.0) >= float(sel["min_p_positive"])]
    lost = [sc for sc, v in c["per_scenario"].items() if _f(v.get("delta_dti"), 0.0) < -float(sel["max_scenario_drop"])]
    return {**c, "scenarios_ok": ok, "scenarios_lost": lost,
            "upgrade": bool(len(ok) >= int(sel["moe_min_scenarios"]) and not lost)}


def select_model(S: pd.DataFrame, P: pd.DataFrame, hard: dict, collapsed: bool, sel: dict,
                 sc_list=("V1", "V2", "V3")) -> dict:
    """Predeclared rule. Hard requirements first (``hard``: {check: bool}); then the simplest model that beats W2
    (``A_pixel``) on >= min_scenarios_improved scenarios with no scenario / worst-zone collapse; C_all replaces B only
    with a clear paired gain over B, D only with a larger one and no routing collapse. Otherwise RETAIN_W2."""
    reasons, trail = [], []
    failed = [k for k, v in hard.items() if not v]
    if failed:
        return {"selected": RETAIN_W2, "status": "BLOCKED", "reasons": [f"hard requirement failed: {failed}"],
                "trail": trail}
    have = set(S.model) if len(S) else set()
    vs = {m: model_checks(S, P, m, "A_pixel", sc_list, sel) for m in ("B", "C_all", "D") if m in have}
    trail += [{"step": f"{m} vs W2", **v} for m, v in vs.items()]
    pick = "B" if vs.get("B", {}).get("beats_w2") else None
    if "C_all" in vs and vs["C_all"]["beats_w2"]:
        if pick is None:                                   # static fusion fails the W2 rule, the rule gate passes it
            pick = "C_all"
        else:
            u = upgrade(S, P, "C_all", "B", sc_list, float(sel["rule_min_gain_vs_static"]), sel)
            trail.append({"step": "C_all vs B", **u})
            if u["upgrade"]:
                pick = "C_all"
            else:
                reasons.append("rule gate does not beat static fusion by the predeclared margin: B kept")
    if "D" in vs and vs["D"]["beats_w2"]:
        if collapsed:
            reasons.append("learned gate collapsed onto one expert: not promotable (static fusion fallback)")
        elif "B" in have:
            u = upgrade(S, P, "D", "B", sc_list, float(sel["min_moe_gain_vs_static"]), sel)
            trail.append({"step": "D vs B", **u})
            if u["upgrade"]:
                pick = "D"
            else:
                reasons.append("learned gate does not beat static fusion by the predeclared margin: simpler model "
                               "kept")
    if pick is None:
        reasons.append("no fused model beats W2 on enough scenarios without a scenario / worst-zone drop")
        return {"selected": RETAIN_W2, "status": "RETAIN_W2", "reasons": reasons, "trail": trail}
    reasons.append(f"{pick}: beats W2 on {vs[pick]['n_improved']} scenario(s), no collapse")
    return {"selected": pick, "status": "SELECTED", "reasons": reasons, "trail": trail}


# =============================================================================== next action
def next_action(cov: dict, selection: dict, export: dict | None, hard: dict, missed: pd.DataFrame) -> dict:
    """Exactly one recommended next modeling action, from the evidence (decision tree, first match)."""
    failed = [k for k, v in (hard or {}).items() if not v]
    if failed:
        return {"action": f"Repair the failed validity check(s) {failed} and re-run this notebook; no model "
                          "conclusion is valid until they pass.", "because": "hard requirement failed"}
    if export and export.get("status") == "READY_FOR_UPLOAD" and selection.get("selected") not in (None, RETAIN_W2):
        return {"action": f"Upload {export.get('file')} manually (one slot) and record its public score with "
                          "record_result; start no new model before that score is known.",
                "because": f"{selection['selected']} passed the predeclared selection rule and every hard export check"}
    ov = cov.get("overall")
    if ov == GENERATION:
        top = "unknown"
        if missed is not None and len(missed) and "subtype" in missed:
            k = missed.groupby(["subtype", "length_bin"]).length_km.sum().sort_values(ascending=False)
            top = " / ".join(map(str, k.index[0])) if len(k) else top
        return {"action": f"Build new candidate proposals targeted at the most-missed structures ({top}): no source "
                          "covers them, so no ranker or gate can recover them.",
                "because": "candidate coverage below the predeclared gate"}
    if ov == RANKING and cov.get("geology_unique"):
        return {"action": "Train a candidate-level ranker on the unified pool restricted to the systems the "
                          "independent families uniquely cover (the gate's routing did not convert that coverage).",
                "because": "coverage is sufficient, deployed recovery lags it, and independent geology adds unique "
                           "systems"}
    if ov == RANKING:
        return {"action": "Improve ranking inside the learned-source (E1) pool only; drop the independent geology "
                          "families from the next experiment (they add no unique coverage).",
                "because": "coverage is sufficient but the independent families cover nothing the CNN / W2 pool "
                           "does not"}
    return {"action": "Keep W2 as the reference; test only the representation / area policy of the submission "
                      "(the pool and the ranking are near their measured ceiling).",
            "because": f"coverage verdict {ov}"}
