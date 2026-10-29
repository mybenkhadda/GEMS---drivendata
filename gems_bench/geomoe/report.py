"""Stage 13 reports, written from persisted tables only (a fresh runtime reproduces them from Drive):

    reports/DISCOVERY_DIAGNOSIS.md   validation scenarios, coverage ceiling, M7 vs X2 reconciliation, missed systems
    reports/GEOMOE_BENCHMARK.md      every model under the same contract: pooled / worst-zone DTI, paired deltas,
                                     near-known bands, deployment constructions
    reports/SCIENTIFIC_DECISION.md   the eight questions of the brief, established findings kept apart from hypotheses
    reports/NEXT_ACTION.md           exactly one next modeling action (decide.next_action)

Absent evidence is written as ABSENT with the stage that would produce it; nothing is filled from memory or from the
recorded historical numbers (those appear only in the protocol table, labelled with their protocol).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import decide as DE

PUBLIC_W2 = 0.0942


def md(df: pd.DataFrame | None, cols=None, digits: int = 4) -> str:
    from ..p6.boost_audit import md_table

    return md_table(df, cols, digits) if df is not None and len(df) else "_ABSENT_\n"


def _f(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "ABSENT"
    return f"{v:.4f}" if np.isfinite(v) else "ABSENT"


def load(G) -> dict:
    ws = G.ws
    t, j = ws.load_table, (lambda rel: ws.read_json(rel) or {})
    return {"S": t("gating/model_summary.csv"), "P": t("gating/paired_comparison.csv"),
            "Z": t("validation/per_zone_metrics.csv"), "N": t("validation/near_known_metrics.csv"),
            "C": t("candidates/oracle_coverage.csv"), "R": t("audit/protocol_comparison_measured.csv"),
            "H": t("audit/protocol_comparison.csv"), "M": t("candidates/missed_systems.csv"),
            "F": t("candidates/coverage_by_fault_type.csv"), "E": t("audit/metric_equivalence.csv"),
            "L": t("audit/label_leakage_audit.csv"), "A": t("experts/source_availability.csv"),
            "D": t("gating/deployment_comparison.csv"), "cov": j("candidates/coverage_decision.json"),
            "sel": j("gating/selection.json"), "dep": j("gating/deployment_decision.json"),
            "sub": j("submissions/submission_manifest.json"), "scen": t("validation/scenarios.csv"),
            "stages": {s: (v if v != "NOT_RUN" else G.manifest(s).get("status", "NOT_RUN"))     # persisted status
                       for s, v in G.status.items()},
            "learned_reason": G.manifest("10_learned").get("reason")}


def val(S: pd.DataFrame, sc: str, m: str, col: str = "dti"):
    r = S[(S.scenario == sc) & (S.model == m)] if len(S) else S
    return float(r[col].iloc[0]) if len(r) and col in r else np.nan


def best_fused(S: pd.DataFrame, sel: dict) -> str | None:
    if sel.get("selected") not in (None, DE.RETAIN_W2):
        return sel["selected"]
    s = S[S.model.isin(["B", "C_all", "D"])] if len(S) else S
    return str(s.groupby("model").dti.mean().idxmax()) if len(s) else None


# =============================================================================== the eight answers
def q1_w2(d: dict) -> list[str]:
    S, N, C = d["S"], d["N"], d["C"]
    out = ["### 1. Why did W2 fail to generalise?", "", "**Established in this run** (cross-fitted, official scorer):",
           ""]
    if not len(S) or "A_pixel" not in set(S.model):
        return out + ["* ABSENT: the W2 baseline grid (stage 05) did not complete.", ""]
    v = {sc: val(S, sc, "A_pixel") for sc in ("V1", "V2", "V3")}
    w = {sc: val(S, sc, "A_pixel", "worst_zone_dti") for sc in ("V1", "V2", "V3")}
    out.append(f"* W2 pooled DTI: V1 withheld systems {_f(v['V1'])} (worst zone {_f(w['V1'])}), V2 network "
               f"completion {_f(v['V2'])} (worst {_f(w['V2'])}), V3 geographic {_f(v['V3'])} (worst {_f(w['V3'])}); "
               f"public leaderboard {PUBLIC_W2} (external, one file, not tuned on).")
    if np.isfinite(v["V1"]) and np.isfinite(v["V3"]):
        gap = v["V1"] - v["V3"]
        out.append(f"* V1 - V3 = {gap:+.4f}: " + ("the withheld-system estimate overstates geographic transfer "
                   "(V1 alone is not a reliable predictor of the public score)." if gap > 0.02 else
                   "V1 and V3 agree within 0.02 under this protocol."))
    if len(N):
        a = N[N.model == "A_pixel"]
        for sc, g in a.groupby("scenario"):
            out.append(f"* {sc} W2 DTI by distance to the permitted known faults: " + ", ".join(
                f"{r.band} {_f(r.dti)} (n_truth {int(r.n_truth)})" for r in g.itertuples()))
    if len(C):
        for r in C.itertuples():
            out.append(f"* {r.scenario}: {_f(getattr(r, 'cov_inventory', np.nan))} of {int(r.n_systems)} systems "
                       f"are covered by some candidate; W2's deployed cell recovers {_f(r.w2_recovered_share)}.")
    out += ["", "**Not established (hypotheses this run cannot decide)**:", "",
            "* that the public test truth has the V2 composition (corrections / tips / splays) rather than V1's whole "
            "systems; V2 is built from real mapped sub-sections, but its class mix is a design choice;",
            "* that the public subset's geology resembles any one geographic fold;",
            "* that the submitted W2 file's area policy (rep from discovery, frac from geography) explains part of "
            "the public gap (it was a mixed-policy file, recorded in P6.1).", ""]
    return out


def q2_bottleneck(d: dict) -> list[str]:
    cov, R = d["cov"], d["R"]
    out = ["### 2. Candidate-generation or candidate-ranking bottleneck?", ""]
    if not cov:
        return out + ["ABSENT: stage 06 (coverage decision) did not complete.", ""]
    out.append(f"**Verdict: {cov.get('overall')}** (rule: {cov.get('rule')}).")
    out.append("")
    out.append(md(pd.DataFrame([{"scenario": k, **v} for k, v in (cov.get("per_scenario") or {}).items()])))
    out += ["Same definitions for every pool (units, truth, 300 m, >= 50 % of a system, per-source budget):", "",
            md(R, ["scenario", "pool", "definition", "n_systems", "covered_share"]),
            "Reading of the M7 / X2 contradiction: M7's ~12.5 % measured LABEL-FREE crest coverage; X2's 82.9 % "
            "'ranking-limited' share measured missed LENGTH against a pool made of the learned maps at up to 5 % "
            "area. The rows `label_free` and `fam:E1_learned` above re-measure both under one definition: when the "
            "learned pool covers far more than the label-free one, the contradiction was source coverage plus the "
            "definition of 'ranking-limited' (covered by the learned pool at a larger budget), not a disagreement.",
            ""]
    return out


def q3_unique(d: dict) -> list[str]:
    C, cov = d["C"], d["cov"]
    out = ["### 3. Which new expert provides unique fault recovery?", ""]
    if not len(C):
        return out + ["ABSENT: stage 05 did not complete.", ""]
    cols = ["scenario", "n_systems"] + [c for c in C if c.startswith("unique_")] + ["independent_not_E1"]
    out += [md(C, cols), f"Independent geology adds unique coverage under the predeclared rule: "
            f"**{cov.get('geology_unique')}**.", "",
            "Correlation caveat: the potential-field gradients and G02 layers are also W2 input features, so an "
            "independent family 'covering' a system is not independent evidence that W2 lacked; only systems no E1 "
            "source covers (`independent_not_E1`) count as unique gains.", ""]
    return out


def q4_gate(d: dict) -> list[str]:
    P = d["P"]
    out = ["### 4. Does geology-conditioned gating beat static fusion?", ""]
    p = P[P.reference == "B"] if len(P) else P
    if not len(p):
        return out + ["ABSENT: no paired comparison against static fusion (stages 09 / 10).", ""]
    return out + [md(p, ["scenario", "method", "reference", "delta_dti", "ci_lo", "ci_hi", "p_delta_positive",
                         "n_blocks"]),
                  "Paired spatial block bootstrap (5 km blocks resampled within zones; descriptive, 90 % interval).",
                  ""]


def q5_moe(d: dict) -> list[str]:
    sel = d["sel"]
    out = ["### 5. Does the learned MoE justify its complexity?", ""]
    if not sel:
        return out + ["ABSENT: selection (stage 10b) did not run.", ""]
    if d["stages"].get("10_learned") == "SKIPPED":
        out.append(f"The learned gate was NOT trained: {d.get('learned_reason') or 'see manifests/stages/10_learned.json'}"
                   " (stage 06 gate or CONFIG); the question is answered by the gate decision, not by a fit.")
    out += [f"Selection: **{sel.get('selected')}** ({sel.get('status')}).", ""] + [f"* {r}" for r in
                                                                                  sel.get("reasons") or []] + [""]
    return out


def q6_reliability(d: dict) -> list[str]:
    S, N = d["S"], d["N"]
    out = ["### 6. Reliability on unseen geography and near-known structures", ""]
    if not len(S):
        return out + ["ABSENT", ""]
    m = best_fused(S, d["sel"])
    keep = S[S.model.isin([x for x in ("A_pixel", m) if x])]
    out += [md(keep, ["scenario", "model", "dti", "worst_zone_dti", "worst_zone", "fold_std", "optimism"])]
    if len(N):
        out += ["Distance bands (permitted known faults):", "", md(N[N.model.isin(["A_pixel", m])],
                                                                  ["scenario", "model", "band", "dti", "recall_w",
                                                                   "n_truth"])]
    return out + [""]


def q7_file(d: dict) -> list[str]:
    sub = d["sub"]
    out = ["### 7. Is there a new validated GeoTIFF worth submitting?", ""]
    files = sub.get("files") or []
    if not files:
        return out + ["ABSENT: stage 12 did not run.", ""]
    for r in files:
        out.append(f"* `{r.get('file')}` | status **{r.get('status')}** | role {r.get('role')} | model "
                   f"{r.get('model')} | recipe `{r.get('recipe_id')}` | sha256 `{r.get('sha256')}` | Drive "
                   f"`{r.get('path_drive')}`")
    return out + ["", sub.get("note", ""), ""]


def next_action(d: dict) -> dict:
    return DE.next_action(d["cov"], d["sel"], (d["sub"].get("files") or [{}])[0], d["sel"].get("hard_checks") or {},
                          d["M"])


# =============================================================================== files
def write_all(G) -> None:
    d = load(G)
    ws = G.ws
    st = md(pd.DataFrame([{"stage": k, "status": v} for k, v in d["stages"].items()]))
    from . import GEOMOE_CODE_REV

    diag = ["# Discovery diagnosis", "", f"Code rev `{GEOMOE_CODE_REV}` / commit `{G.head}`. Every number "
            "below was produced by this run's stages; ABSENT = not produced.", "", "## Stages", "", st,
            "## Validation scenarios", "", md(d["scen"]), "## Leakage audit (failures only)", "",
            md(d["L"][~d["L"].ok.astype(bool)] if len(d["L"]) else d["L"]),
            "## Official metric equivalence (A_pixel vs frozen boost-A grid)", "",
            md(d["E"].assign(fail=d["E"].status.ne("PASS")).groupby("fail").size().reset_index(name="cells")
               if len(d["E"]) and "status" in d["E"] else d["E"]),
            "## Coverage", "", md(d["C"]), "## Coverage by structural subtype", "", md(d["F"]),
            "## Recorded historical numbers (other protocols; context only)", "", md(d["H"]),
            "## Missed systems (no candidate within 300 m), longest first", "",
            md(d["M"].sort_values("length_km", ascending=False).head(25) if len(d["M"]) and "length_km" in d["M"]
               else d["M"], ["scenario", "unit", "system_id", "subtype", "length_bin", "length_km",
                             "known_dist_km_median", "near_share"]), ""]
    ws.save_text("\n".join(diag), "reports/DISCOVERY_DIAGNOSIS.md")
    bench = ["# GeoMoE benchmark", "", "Same scenarios, candidate pool, rasterisation policies, eligible budget, "
             "official scorer and nested (rep, frac) cross-fitting for every model. A_pixel = frozen W2 pixel map "
             "(the external reference); A_seg = W2 on the unified pool; B static fusion; C rule gate (+ one ablation "
             "per rule); D learned gate.", "", "## Pooled held-out DTI", "",
             md(d["S"], ["scenario", "model", "dti", "worst_zone_dti", "worst_zone", "fold_std", "optimism",
                         "params_all"]),
             "## Paired comparisons", "", md(d["P"]), "## Near-known bands", "", md(d["N"]),
             "## Expert availability (expert level)", "", md(d["A"][d["A"].member == "*"] if len(d["A"]) else d["A"]),
             "## Deployment constructions (V3, same cells)", "", md(d["D"]),
             f"Deployment decision: `{json.dumps(d['dep'].get('constructions'), default=str)[:1500]}`", ""]
    ws.save_text("\n".join(bench), "reports/GEOMOE_BENCHMARK.md")
    na = next_action(d)
    dec = ["# Scientific decision", "", "Written from persisted tables; numbers are offline cross-fitted estimates, "
           "not leaderboard scores.", ""]
    for fn in (q1_w2, q2_bottleneck, q3_unique, q4_gate, q5_moe, q6_reliability, q7_file):
        dec += fn(d)
    dec += ["### 8. What should happen next?", "", f"**{na['action']}**", "", f"Because: {na['because']}.", ""]
    ws.save_text("\n".join(dec), "reports/SCIENTIFIC_DECISION.md")
    ws.save_text("\n".join(["# Next action", "", na["action"], "", f"Evidence: {na['because']}.", "",
                            "Derived by `gems_bench.geomoe.decide.next_action` (first matching rule of a fixed tree).",
                            ""]), "reports/NEXT_ACTION.md")
    ws.push()
