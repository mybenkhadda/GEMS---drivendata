"""Stage 13 reports, written from persisted tables only (a fresh runtime reproduces them from Drive):

    reports/EXECUTION_STATUS.md           every stage: status, dependencies, failure, affected file, conclusion allowed
    reports/SCIENTIFIC_FINDINGS.md        the ten questions; CONFIRMED (measured here) kept apart from INFERRED
    reports/MODEL_COMPARISON.md           frozen submitted W2 vs every model under the fixed-policy rule
    reports/LEADERBOARD_RECOMMENDATION.md which file (if any) is worth one manual upload, and why
    reports/NEXT_ACTION.md                exactly one next action (decide.next_action)

A question whose evidence comes from a stage that did not produce valid outputs is answered NOT EVALUATED with the
blocking stage; it is never answered from defaults, recorded history or a skipped computation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import decide as DE
from .decide import NOT_EVALUATED, PRODUCED, REFERENCE

PUBLIC_W2 = 0.0942


def md(df: pd.DataFrame | None, cols=None, digits: int = 4) -> str:
    from ..p6.boost_audit import md_table

    return md_table(df, cols, digits) if df is not None and len(df) else "_ABSENT_\n"


def _f(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{v:.4f}" if np.isfinite(v) else "n/a"


def load(G) -> dict:
    ws = G.ws
    t, j = ws.load_table, (lambda rel: ws.read_json(rel) or {})
    stages = {}
    for s, v in G.status.items():
        stages[s] = v if v != "NOT_RUN" else (G.manifest(s).get("status") or "NOT_RUN")
    return {"S": t("gating/model_summary.csv"), "Sd": t("gating/model_summary_scenario_selected.csv"),
            "P": t("gating/paired_comparison.csv"), "Z": t("validation/per_zone_metrics.csv"),
            "N": t("validation/near_known_metrics.csv"), "C": t("candidates/oracle_coverage.csv"),
            "R": t("audit/candidate_bottleneck_reconciliation.csv"), "H": t("audit/protocol_comparison.csv"),
            "M": t("candidates/missed_systems.csv"), "E": t("audit/metric_equivalence.csv"),
            "L": t("audit/label_leakage_audit.csv"), "A": t("experts/source_availability.csv"),
            "D": t("gating/deployment_comparison.csv"), "FB": t("validation/frozen_w2_baseline.csv"),
            "PP": t("audit/baseline_policy_parity.csv"), "V2": t("validation/v2_by_subtype.csv"),
            "V2c": t("validation/v2_coverage_by_subtype.csv"), "X": t("gating/effect_decomposition.csv"),
            "K": t("candidates/kernel_weighted_oracle.csv"), "I": t("audit/stage_integrity.csv"),
            "cov": j("candidates/coverage_decision.json"), "sel": j("gating/selection.json"),
            "dep": j("gating/deployment_decision.json"), "sub": j("submissions/submission_manifest.json"),
            "bf": j("audit/g02_backfill.json"), "stages": stages,
            "learned_reason": G.manifest("10_learned").get("reason")}


def produced(d: dict, *stages) -> bool:
    return all(d["stages"].get(s) in PRODUCED for s in stages)


def blocked(d: dict, *stages) -> str:
    bad = [f"{s}={d['stages'].get(s)}" for s in stages if d["stages"].get(s) not in PRODUCED]
    return f"**NOT EVALUATED** (stage(s) {', '.join(bad)}; see EXECUTION_STATUS.md)"


def val(S: pd.DataFrame, sc: str, m: str, col: str = "dti"):
    r = S[(S.scenario == sc) & (S.model == m)] if len(S) else S
    return float(r[col].iloc[0]) if len(r) and col in r else np.nan


def delta_line(d: dict, m: str, ref: str) -> str:
    P = d["P"]
    p = P[(P.method == m) & (P.reference == ref)] if len(P) else P
    if not len(p):
        return f"{m} vs {ref}: no paired comparison"
    return "; ".join(f"{r.scenario} {r.delta_dti:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] P(>0)={r.p_delta_positive:.2f}"
                     for r in p.itertuples())


# =============================================================================== the ten questions
def findings(d: dict) -> list[str]:
    S, C, cov, sel = d["S"], d["C"], d["cov"], d["sel"]
    out = []

    def q(n, title, body):
        out.extend([f"### {n}. {title}", "", *body, ""])

    # 1
    if produced(d, "05_oracle") and len(C):
        body = [f"* CONFIRMED {r.scenario}: unified pool covers {_f(getattr(r, 'cov_inventory', np.nan))} of "
                f"{int(r.n_systems)} systems; frozen submitted W2 recovers {_f(r.w2_recovered_share)}; covered and "
                f"not recovered by W2 = see candidates/unique_systems.csv" for r in C.itertuples()]
    else:
        body = [blocked(d, "04_candidates", "05_oracle")]
    q(1, "Does the new candidate pool recover fault systems W2 misses?", body)
    # 2
    if produced(d, "05_oracle") and len(C):
        cols = ["scenario", "n_systems"] + [c for c in C if c.startswith(("unique_", "evaluated_"))] + \
               [c for c in ("independent_not_E1", "independent_not_W2") if c in C]
        body = [md(C, cols), f"Geology status: **{cov.get('geology_status', NOT_EVALUATED)}**, unique coverage under "
                f"the predeclared rule: **{cov.get('geology_unique')}** (None = no independent family evaluated).",
                "INFERRED caveat: potential-field gradients and G02 layers are also W2 inputs; only systems no E1 "
                "source covers count as unique gains.", f"G02 backfill: `{json.dumps(d['bf'])[:400]}`"]
    else:
        body = [blocked(d, "05_oracle")]
    q(2, "Which expert families provide genuinely unique recovery?", body)
    # 3
    if produced(d, "05_oracle"):
        v1, v2, v3 = (val(S, sc, REFERENCE) for sc in ("V1", "V2", "V3")) if len(S) else (np.nan,) * 3
        if not np.isfinite(v2) and len(d["FB"]):
            fb = d["FB"]
            v1, v2, v3 = (DE._f(fb[fb.scenario == sc].dti.mean()) for sc in ("V1", "V2", "V3"))
        body = [f"* CONFIRMED frozen submitted W2 DTI: V1 {_f(v1)}, V2 {_f(v2)}, V3 {_f(v3)} (one frozen policy for all "
                f"three; public {PUBLIC_W2}).", "", "V2 by subtype (official components under each subtype's own mask "
                "where 10b ran; geometric coverage otherwise):", "", md(d["V2"] if len(d["V2"]) else d["V2c"]),
                "Context fairness: `B_ctxW2` scores E4 with W2's own known context and drops tips-only candidates; "
                "`B - B_ctxW2` is the share of any V2 gain due to the additional retained-network context, not to "
                "fusion (" + delta_line(d, "B_ctxW2", "B") + ").",
                "INFERRED: a low V2 relative to V1 / V3 indicates a network-completion weakness of W2 only if B_ctxW2 "
                "(no extra context) also beats it on V2."]
    else:
        body = [blocked(d, "05_oracle")]
    q(3, "Is V2 network completion a significant weakness of W2 under a fair comparison?", body)
    # 4
    if produced(d, "06_decision"):
        body = [f"**{cov.get('overall')}** (scenarios disagree: {cov.get('disagreement')})", "",
                md(pd.DataFrame([{"scenario": k, **{a: b for a, b in v.items() if a != 'families_evaluated'}}
                                 for k, v in (cov.get("per_scenario") or {}).items()])),
                "M7 vs X2 under one definition:", "", md(d["R"], ["scenario", "pool", "system_coverage",
                                                              "length_coverage", "ranking_limited_length_share",
                                                              "generation_limited_length_share"])]
    else:
        body = [blocked(d, "05_oracle", "06_decision")]
    q(4, "Is candidate generation or candidate ranking the dominant limitation?", body)
    # 5-8
    if produced(d, "10b_compare") and sel.get("measured"):
        q(5, "Does static fusion improve official DTI?", [f"B vs frozen W2: {delta_line(d, 'B', REFERENCE)}",
                                                          f"B vs A_seg (fusion alone): see gating/"
                                                          "effect_decomposition.csv"])
        q(6, "Does geological routing add incremental value?", [f"C_all vs B: {delta_line(d, 'C_all', 'B')}",
                                                                "Per-rule ablations: gating/model_summary.csv "
                                                                "(C_minus_*)."])
        q(7, "Does the learned GeoMoE outperform simpler alternatives?",
          [f"D vs B: {delta_line(d, 'D', 'B')}" if d["stages"].get("10_learned") in PRODUCED else
           f"D not trained: stage 10 {d['stages'].get('10_learned')} ({d.get('learned_reason')})"])
        m = sel.get("selected") if sel.get("selected") not in (None, DE.RETAIN_W2) else "B"
        q(8, "Does the improvement survive geographic validation?",
          [f"V3 (fixed policy): {m} {_f(val(S, 'V3', m))} vs frozen W2 {_f(val(S, 'V3', REFERENCE))}; worst zone "
           f"{_f(val(S, 'V3', m, 'worst_zone_dti'))} vs {_f(val(S, 'V3', REFERENCE, 'worst_zone_dti'))}"])
    else:
        for n, t in ((5, "Does static fusion improve official DTI?"), (6, "Does geological routing add incremental "
                                                                         "value?"),
                     (7, "Does the learned GeoMoE outperform simpler alternatives?"),
                     (8, "Does the improvement survive geographic validation?")):
            q(n, t, [blocked(d, "08_static", "10b_compare") + f" (selection status {sel.get('status')})"])
    # 9
    if produced(d, "11_deployment"):
        q(9, "Can the best model be deployed consistently across the whole region?",
          [md(d["D"]), f"Decision: `{json.dumps(d['dep'].get('constructions'), default=str)[:800]}`"])
    else:
        q(9, "Can the best model be deployed consistently across the whole region?", [blocked(d, "11_deployment")])
    # 10
    files = d["sub"].get("files") or []
    q(10, "Is there a technically valid GeoTIFF worth testing publicly?",
      [f"* `{f.get('file')}` **{f.get('status')}** ({f.get('role')}), model {f.get('model')}, sha256 `{f.get('sha256')}`"
       for f in files] or [blocked(d, "12_export")])
    return out


# =============================================================================== files
def write_all(G) -> None:
    from . import GEOMOE_CODE_REV

    d = load(G)
    ws = G.ws
    I = d["I"]
    ex = ["# Execution status", "", f"Code rev `{GEOMOE_CODE_REV}`, commit `{G.head}`, mode `{G.nb['mode']}`.", "",
          "Statuses: COMPLETE / REUSED_VALID_CACHE (measured), SKIPPED_BY_VALID_DECISION (a recorded decision), "
          "FAILED (exception), TESTS_FAILED, BLOCKED (technical gate), BLOCKED_UPSTREAM / NOT_EVALUATED (an upstream "
          "stage was invalid: nothing was concluded).", "",
          md(I, ["stage", "actual_status", "required_dependencies", "missing_outputs", "failure_reason",
                 "affected_file", "scientific_conclusion_allowed"]) if len(I) else
          md(pd.DataFrame([{"stage": k, "status": v} for k, v in d["stages"].items()])), ""]
    errs = ws.read_json("manifests/runtime_errors.json") or []
    if errs:
        ex += ["## Exceptions", ""] + [f"* `{e.get('stage')}`: {str(e.get('error'))[:400]} (file: "
                                       f"{e.get('affected_file')})" for e in errs] + [""]
    ws.save_text("\n".join(ex), "reports/EXECUTION_STATUS.md")
    na = DE.next_action(d["cov"], d["sel"], (d["sub"].get("files") or [{}])[0], d["sel"].get("hard_checks") or {},
                        d["M"], d["stages"])
    sci = ["# Scientific findings", "", "CONFIRMED = measured by this run's valid stages. INFERRED = interpretation. "
           "Offline numbers are cross-fitted estimates, not leaderboard scores; the public W2 score 0.0942 is an "
           "external benchmark only.", "", *findings(d), "## Next action", "", f"**{na['action']}** ({na['because']})",
           ""]
    ws.save_text("\n".join(sci), "reports/SCIENTIFIC_FINDINGS.md")
    mc = ["# Model comparison", "", f"Reference: **{REFERENCE}** = the frozen submitted W2 policy, one cell for every "
          "scenario, no selection. Every other model: one fixed-policy rule (cell chosen on development units that do "
          "not overlap the outer unit), applied unchanged to V1, V2, V3. Scenario-selected numbers are diagnostics.",
          "", "## Frozen submitted W2", "", md(d["FB"], ["scenario", "zone", "rep", "frac", "area_frac_eligible",
                                                         "tp_w", "fp_w", "fn_w", "dti", "policy_source"]),
          "## Policy parity (vs factory evaluation construction and frozen boost-A grid)", "", md(d["PP"]),
          "## Fixed-policy summary", "", md(d["S"], ["scenario", "model", "protocol", "dti", "worst_zone_dti",
                                                     "worst_zone", "fold_std", "params_all"]),
          "## Paired comparisons (spatial block bootstrap, descriptive)", "", md(d["P"]),
          "## Effect decomposition", "", md(d["X"]), "## Near-known bands", "", md(d["N"]),
          "## Scenario-selected diagnostics (NOT used for selection)", "", md(d["Sd"], ["scenario", "model", "dti",
                                                                                         "worst_zone_dti"]),
          "## Selection", "", "`" + json.dumps({k: d["sel"].get(k) for k in ("selected", "status", "measured",
                                                                               "reasons", "hard_checks")},
                                               default=str) + "`", ""]
    ws.save_text("\n".join(mc), "reports/MODEL_COMPARISON.md")
    files = d["sub"].get("files") or []
    lb = ["# Leaderboard recommendation", "", f"Public reference: W2 champion {PUBLIC_W2} (one scored file).", ""]
    ready = [f for f in files if f.get("status") == "READY_FOR_UPLOAD"]
    if ready and d["sel"].get("status") == "SELECTED":
        f = ready[0]
        lb += [f"Upload **one** file manually: `{f.get('path_drive')}` (sha256 `{f.get('sha256')}`, model "
               f"{f.get('model')}, recipe {f.get('recipe_id')}). It passed the predeclared rule on measured "
               "fixed-policy numbers and every hard check on the read-back file.", "",
               "Do not use further slots to tune: a public score never changes a selection here."]
    elif ready:
        f = ready[0]
        lb += [f"Optional EXPLORATORY file: `{f.get('path_drive')}` (sha256 `{f.get('sha256')}`). It did NOT pass the "
               "selection rule; upload only to test a stated hypothesis, never as a new champion."]
    else:
        lb += [f"No file to upload. Selection status: {d['sel'].get('status', NOT_EVALUATED)}"
               f"{' (no measured comparison: not evidence for W2)' if not d['sel'].get('measured') else ''}. "
               f"{d['sub'].get('note', '')}"]
    ws.save_text("\n".join(lb + [""]), "reports/LEADERBOARD_RECOMMENDATION.md")
    ws.save_text("\n".join(["# Next action", "", na["action"], "", f"Evidence: {na['because']}.", "",
                            "Derived by `gems_bench.geomoe.decide.next_action` (first matching rule of a fixed tree).",
                            ""]), "reports/NEXT_ACTION.md")
    ws.push()
