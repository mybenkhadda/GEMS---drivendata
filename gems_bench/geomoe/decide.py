"""Predeclared decisions (pure): stage dependencies, the generation-vs-ranking gate, model selection, next action.

Every rule reads persisted tables and the config only; nothing here looks at a public score or an oracle row. A
negative scientific conclusion is only possible from measured inputs: a stage that did not run yields NOT_EVALUATED,
never "geology adds nothing" or "W2 wins".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------- stage statuses
COMPLETE, FAILED, BLOCKED_UPSTREAM = "COMPLETE", "FAILED", "BLOCKED_UPSTREAM"
NOT_EVALUATED, SKIPPED_OK, REUSED = "NOT_EVALUATED", "SKIPPED_BY_VALID_DECISION", "REUSED_VALID_CACHE"
TESTS_FAILED, BLOCKED = "TESTS_FAILED", "BLOCKED"
VALID = (COMPLETE, REUSED, SKIPPED_OK)                 # an upstream stage in one of these may feed a later stage
PRODUCED = (COMPLETE, REUSED)                          # its outputs exist and were measured
# compute stages become BLOCKED_UPSTREAM, decision stages NOT_EVALUATED when an upstream stage is invalid
DECISION_STAGES = ("06_decision", "10_learned", "11_deployment", "12_export")
DEPENDS = {"00_tests": (), "01_attach": (), "02_audit": ("01_attach",), "03_scenarios": ("01_attach",),
           "04_candidates": ("00_tests", "03_scenarios"), "05_oracle": ("04_candidates",),
           "06_decision": ("05_oracle",), "07_experts": ("04_candidates",), "08_static": ("05_oracle", "07_experts"),
           "09_rules": ("08_static",), "10_learned": ("06_decision", "08_static"),
           "10b_compare": ("05_oracle", "08_static"), "11_deployment": ("10b_compare",),
           "12_export": ("10b_compare",), "13_reports": (), "14_integrity": ()}

# ------------------------------------------------------------------------------- scientific verdicts
GENERATION, RANKING, MIXED, NOT_DOMINANT = "GENERATION_LIMITED", "RANKING_LIMITED", "MIXED", "NOT_DOMINANT"
RETAIN_W2, NO_MEASUREMENT = "RETAIN_W2", "NO_MEASURED_COMPARISON"
INDEPENDENT = ("E2_potential", "E3_structural", "E4_terrain_network")
REFERENCE = "A_submitted_W2"


def _f(x, default=np.nan) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def dependency_status(stage: str, statuses: dict, strict: bool = True, depends: dict | None = None,
                      decision_stages=None) -> tuple[str | None, list[str]]:
    """(status to record instead of running, invalid upstream stages). ``None`` = the stage may run.
    A stage whose upstream is invalid is BLOCKED_UPSTREAM (compute) or NOT_EVALUATED (decision). ``depends`` /
    ``decision_stages`` default to this experiment's tables (another runner passes its own)."""
    deps = DEPENDS if depends is None else depends
    dec = DECISION_STAGES if decision_stages is None else decision_stages
    bad = [s for s in deps.get(stage, ()) if statuses.get(s) not in VALID]
    if not bad or not strict:
        return None, bad
    return (NOT_EVALUATED if stage in dec else BLOCKED_UPSTREAM), bad


# =============================================================================== coverage gate (stage 06)
def coverage_summary(C: pd.DataFrame, U: pd.DataFrame, w2_sys: pd.DataFrame, avail: pd.DataFrame | None = None
                     ) -> pd.DataFrame:
    """Per scenario: systems, system / length coverage of each pool, W2 deployed recovery, unique systems per family,
    and whether each independent family was actually EVALUATED (available on at least one unit of the scenario).

    ``C``: system_coverage rows (scenario, unit, system_id, pool, near_share, covered, n_px); ``U``: unique_coverage;
    ``w2_sys``: frozen submitted W2 held-out recovery (scenario, unit, system_id, recovered)."""
    rows = []
    for sc, g in C.groupby("scenario"):
        n = g[["unit", "system_id"]].drop_duplicates().shape[0]
        r = {"scenario": sc, "n_systems": int(n)}
        for pool, h in g.groupby("pool"):
            r[f"cov_{pool}"] = float(h.covered.mean()) if len(h) else np.nan
            if "n_px" in h:
                r[f"lencov_{pool}"] = float((h.near_share * h.n_px).sum() / max(float(h.n_px.sum()), 1e-9))
        w = w2_sys[w2_sys.scenario == sc] if len(w2_sys) else w2_sys
        r["w2_recovered_share"] = float(w.recovered.sum() / max(n, 1)) if len(w) else np.nan
        u = U[U.scenario == sc] if len(U) else U
        for f in ("E1_learned", *INDEPENDENT):
            r[f"unique_{f}"] = int(u[f"unique_{f}"].sum()) if len(u) and f"unique_{f}" in u else 0
        ind = np.zeros(len(u), bool)
        for f in INDEPENDENT:
            if len(u) and f"covered_{f}" in u:
                ind |= u[f"covered_{f}"].to_numpy(bool)
        r["independent_not_E1"] = int((ind & ~u["covered_E1_learned"].to_numpy(bool)).sum()) if len(u) else 0
        if len(u) and len(w):
            wr = w.set_index(["unit", "system_id"]).recovered
            notw2 = ~np.array([bool(wr.get((a, int(b)), False)) for a, b in zip(u.unit, u.system_id)])
            r["independent_not_W2"] = int((ind & notw2).sum())
        for f in INDEPENDENT:
            r[f"evaluated_{f}"] = bool(avail is not None and len(avail) and
                                       ((avail.scenario == sc) & (avail.expert == f) & (avail.member == "*") &
                                        (avail.available_share > 0)).any())
        rows.append(r)
    return pd.DataFrame(rows)


def scenario_verdict(pool: float, w2: float, n: int, gc: dict) -> str:
    if not n or not np.isfinite(pool) or not np.isfinite(w2):
        return NOT_EVALUATED
    gen = pool < float(gc["min_pool_coverage"])
    rank = pool - w2 >= float(gc["min_ranking_gap"])
    return MIXED if gen and rank else GENERATION if gen else RANKING if rank else NOT_DOMINANT


def coverage_decision(S: pd.DataFrame, gc: dict) -> dict:
    """Per-scenario verdicts are kept (disagreement is NOT resolved by a majority vote). ``overall`` is the common
    verdict when every evaluated scenario agrees, else MIXED; NOT_EVALUATED when nothing was measured.
    ``geology_unique``: True / False only when an independent family was evaluated somewhere; None otherwise."""
    per = {}
    for r in (S.to_dict("records") if S is not None and len(S) else []):
        n = int(r.get("n_systems") or 0)
        pool, w2 = _f(r.get("cov_inventory")), _f(r.get("w2_recovered_share"))
        uq = int(r.get("independent_not_E1") or 0)
        ev = {f: bool(r.get(f"evaluated_{f}")) for f in INDEPENDENT}
        per[r["scenario"]] = {"verdict": scenario_verdict(pool, w2, n, gc), "pool_coverage": pool,
                              "w2_recovered": w2, "gap": pool - w2 if np.isfinite(pool) and np.isfinite(w2) else None,
                              "n_systems": n, "independent_not_E1": uq, "independent_share": uq / max(n, 1),
                              "families_evaluated": ev}
    measured = {k: v["verdict"] for k, v in per.items() if v["verdict"] != NOT_EVALUATED}
    overall = NOT_EVALUATED if not measured else (next(iter(measured.values())) if len(set(measured.values())) == 1
                                                  else MIXED)
    any_eval = any(any(v["families_evaluated"].values()) for v in per.values())
    uniq = (any(p["independent_not_E1"] >= int(gc["min_unique_systems"]) and
                p["independent_share"] >= float(gc["min_unique_share"]) for p in per.values()) if any_eval else None)
    ranking_somewhere = any(v in (RANKING, MIXED, NOT_DOMINANT) for v in measured.values())
    return {"overall": overall, "per_scenario": per, "geology_unique": uniq,
            "geology_status": "EVALUATED" if any_eval else NOT_EVALUATED,
            "learned_gate_allowed": bool(measured) and ranking_somewhere,
            "disagreement": sorted(set(measured.values())) if len(set(measured.values())) > 1 else [],
            "rule": {GENERATION: f"inventory system coverage < {gc['min_pool_coverage']}",
                     RANKING: f"coverage - frozen W2 recovered >= {gc['min_ranking_gap']}",
                     MIXED: "both limits hold in a scenario, or scenarios disagree (overall)",
                     NOT_DOMINANT: "neither limit holds", NOT_EVALUATED: "no measured coverage",
                     "geology_unique": f">= {gc['min_unique_systems']} systems and >= {gc['min_unique_share']:.0%} "
                                       "covered by E2/E3/E4 and by no E1 source; None if no family was evaluated"}}


# =============================================================================== model selection
def model_checks(S: pd.DataFrame, P: pd.DataFrame, model: str, ref: str, sc_list, sel: dict) -> dict:
    """Gains of ``model`` over ``ref`` per scenario (fixed-policy summary + paired bootstrap)."""
    out = {"model": model, "reference": ref, "per_scenario": {}}
    imp = drop = measured = 0
    for sc in sc_list:
        a = S[(S.scenario == sc) & (S.model == model)]
        b = S[(S.scenario == sc) & (S.model == ref)]
        if not len(a) or not len(b):
            out["per_scenario"][sc] = {"status": "MISSING"}
            continue
        measured += 1
        d = _f(a.dti.iloc[0]) - _f(b.dti.iloc[0])
        wd = _f(a.worst_zone_dti.iloc[0]) - _f(b.worst_zone_dti.iloc[0])
        p = P[(P.scenario == sc) & (P.method == model) & (P.reference == ref)] if len(P) else P
        out["per_scenario"][sc] = {"delta_dti": d, "delta_worst": wd,
                                   "p_delta_positive": _f(p.p_delta_positive.iloc[0]) if len(p) else np.nan,
                                   "ci_lo": _f(p.ci_lo.iloc[0]) if len(p) else np.nan,
                                   "ci_hi": _f(p.ci_hi.iloc[0]) if len(p) else np.nan}
        imp += int(d >= float(sel["min_gain_vs_w2"]))
        drop += int(d < -float(sel["max_scenario_drop"]) or wd < -float(sel["max_worst_zone_drop"]))
    out.update(n_measured=measured, n_improved=imp, n_dropped=drop,
               beats_ref=bool(measured == len(sc_list) and imp >= int(sel["min_scenarios_improved"]) and drop == 0))
    return out


def upgrade(S, P, model: str, base: str, sc_list, gain: float, sel: dict) -> dict:
    c = model_checks(S, P, model, base, sc_list, {**sel, "min_gain_vs_w2": gain})
    ok = [sc for sc, v in c["per_scenario"].items() if _f(v.get("delta_dti")) >= gain and
          _f(v.get("p_delta_positive"), 0.0) >= float(sel["min_p_positive"])]
    lost = [sc for sc, v in c["per_scenario"].items() if _f(v.get("delta_dti"), 0.0) < -float(sel["max_scenario_drop"])]
    return {**c, "scenarios_ok": ok, "scenarios_lost": lost,
            "upgrade": bool(len(ok) >= int(sel["moe_min_scenarios"]) and not lost)}


def select_model(S: pd.DataFrame, P: pd.DataFrame, hard: dict, collapsed: bool, sel: dict,
                 sc_list=("V1", "V2", "V3"), ref: str = REFERENCE) -> dict:
    """Predeclared rule on FIXED-POLICY numbers (one deployable cell rule per model, no per-scenario retuning).
    Hard requirements first; then the simplest fused model that beats the frozen submitted W2 on
    >= min_scenarios_improved scenarios (every scenario measured) with no scenario / worst-zone collapse; C_all over B
    only with a paired gain; D over B only with a larger one and no routing collapse. RETAIN_W2 is a MEASURED outcome
    only when the reference and at least one fused model were evaluated on every scenario; otherwise
    NO_MEASURED_COMPARISON."""
    trail, reasons = [], []
    failed = [k for k, v in hard.items() if not v]
    if failed:
        return {"selected": RETAIN_W2, "status": BLOCKED, "measured": False,
                "reasons": [f"hard requirement failed: {failed}"], "trail": trail}
    have = set(S.model) if len(S) else set()
    if ref not in have:
        return {"selected": None, "status": NO_MEASUREMENT, "measured": False,
                "reasons": [f"reference {ref} was not evaluated"], "trail": trail}
    vs = {m: model_checks(S, P, m, ref, sc_list, sel) for m in ("B", "C_all", "D") if m in have}
    trail += [{"step": f"{m} vs {ref}", **v} for m, v in vs.items()]
    if not any(v["n_measured"] == len(sc_list) for v in vs.values()):
        return {"selected": None, "status": NO_MEASUREMENT, "measured": False,
                "reasons": ["no fused model was evaluated on every scenario"], "trail": trail}
    pick = "B" if vs.get("B", {}).get("beats_ref") else None
    if vs.get("C_all", {}).get("beats_ref"):
        if pick is None:
            pick = "C_all"
        else:
            u = upgrade(S, P, "C_all", "B", sc_list, float(sel["rule_min_gain_vs_static"]), sel)
            trail.append({"step": "C_all vs B", **u})
            pick = "C_all" if u["upgrade"] else pick
            if not u["upgrade"]:
                reasons.append("rule gate does not beat static fusion by the predeclared margin: B kept")
    if vs.get("D", {}).get("beats_ref"):
        if collapsed:
            reasons.append("learned gate collapsed onto one expert: not promotable")
        elif "B" in have:
            u = upgrade(S, P, "D", "B", sc_list, float(sel["min_moe_gain_vs_static"]), sel)
            trail.append({"step": "D vs B", **u})
            if u["upgrade"]:
                pick = "D"
            else:
                reasons.append("learned gate does not beat static fusion by the predeclared margin")
    if pick is None:
        reasons.append(f"no fused model beats {ref} on enough scenarios without a scenario / worst-zone drop")
        return {"selected": RETAIN_W2, "status": RETAIN_W2, "measured": True, "reasons": reasons, "trail": trail}
    reasons.append(f"{pick}: beats {ref} on {vs[pick]['n_improved']} scenario(s), no collapse")
    return {"selected": pick, "status": "SELECTED", "measured": True, "reasons": reasons, "trail": trail}


def effect_decomposition(S: pd.DataFrame, sc_list=("V1", "V2", "V3")) -> pd.DataFrame:
    """Where a difference comes from (fixed-policy DTI deltas): candidate pool (A_seg - A_submitted_W2), fusion
    (B - A_seg), routing (C_all - B), learning (D - B); V2 = network completion, V3 = geographic generalisation."""
    steps = [("candidate_pool", "A_seg", REFERENCE), ("expert_fusion", "B", "A_seg"), ("geological_routing", "C_all", "B"),
             ("learned_gate", "D", "B"), ("total_vs_reference", "B", REFERENCE)]
    rows = []
    for name, a, b in steps:
        for sc in sc_list:
            x, y = S[(S.scenario == sc) & (S.model == a)], S[(S.scenario == sc) & (S.model == b)]
            rows.append({"effect": name, "model": a, "reference": b, "scenario": sc,
                         "role": {"V2": "network completion", "V3": "geographic generalisation"}.get(sc, "discovery"),
                         "delta_dti": _f(x.dti.iloc[0]) - _f(y.dti.iloc[0]) if len(x) and len(y) else np.nan,
                         "status": "MEASURED" if len(x) and len(y) else NOT_EVALUATED})
    return pd.DataFrame(rows)


# =============================================================================== next action
def next_action(cov: dict, selection: dict, export: dict | None, hard: dict, missed: pd.DataFrame,
                stages: dict | None = None) -> dict:
    """Exactly one recommended next action, from the evidence (decision tree, first match)."""
    stages = stages or {}
    broken = [s for s, v in stages.items() if v in (FAILED, TESTS_FAILED)]
    if broken:
        return {"action": f"Fix the failing stage(s) {broken} (see reports/EXECUTION_STATUS.md for the exception and "
                          "affected file) and re-run; no scientific conclusion exists yet.",
                "because": "the experiment did not run to the measurement"}
    failed = [k for k, v in (hard or {}).items() if not v]
    if failed:
        return {"action": f"Repair the failed validity check(s) {failed} and re-run; no model conclusion is valid "
                          "until they pass.", "because": "hard requirement failed"}
    ov = (cov or {}).get("overall")
    if ov in (None, NOT_EVALUATED):
        return {"action": "Run the FAST_DIAGNOSTIC stages [00]-[06] to completion: candidate coverage has not been "
                          "measured.", "because": "coverage NOT_EVALUATED"}
    if export and export.get("status") == "READY_FOR_UPLOAD" and selection.get("selected") not in (None, RETAIN_W2):
        return {"action": f"Upload {export.get('file')} manually (one slot) and record its public score; start no new "
                          "model before that score is known.",
                "because": f"{selection['selected']} passed the predeclared rule and every hard export check"}
    if ov == GENERATION:
        top = "unknown"
        if missed is not None and len(missed) and {"subtype", "length_bin", "length_km"} <= set(missed.columns):
            k = missed.groupby(["subtype", "length_bin"]).length_km.sum().sort_values(ascending=False)
            top = " / ".join(map(str, k.index[0])) if len(k) else top
        return {"action": f"Build new candidate proposals targeted at the most-missed structures ({top}).",
                "because": "candidate coverage below the predeclared gate in every evaluated scenario"}
    if ov in (RANKING, MIXED) and cov.get("geology_unique"):
        return {"action": "Train a candidate-level ranker restricted to the systems the independent families uniquely "
                          "cover (the gate did not convert that coverage).",
                "because": f"coverage verdict {ov}, independent geology adds unique systems"}
    if ov in (RANKING, MIXED):
        return {"action": "Improve ranking inside the learned-source (E1) pool only.",
                "because": f"coverage verdict {ov}; independent families add no unique coverage "
                           f"(geology status {cov.get('geology_status')})"}
    return {"action": "Keep W2 as the reference; test only the representation / area policy of the submission.",
            "because": f"coverage verdict {ov}"}
