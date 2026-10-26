"""M8 research report (rebuilt from the saved tables only: no recomputation, no invented number).

Every answer cites the table it comes from; a section that did not run is reported as NOT EVALUATED. Offline values
are nested-cross-fitted pseudo-discovery proxies, never leaderboard scores; oracle values read the truth.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _md(df: pd.DataFrame, n: int = 40) -> str:
    if df is None or not len(df):
        return "_(empty)_"
    d = df.head(n)
    try:
        return d.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        return "```\n" + d.to_string(index=False) + "\n```"


def answers(m8) -> list[dict]:
    ws, prim = m8.ws, m8.prim
    out = []

    def add(q, a, ev):
        out.append({"question": q, "answer": a, "evidence": ev})

    r = ws.load_table("reasons", "signal_recovery")
    if len(r):
        g = r[r.task == prim].failure_category.value_counts()
        add("1. Which fault signatures exist in the raw fields?",
            f"raw response present (any modality, any definition) for "
            f"{int((r[r.task == prim].failure_category != 'information_limit_under_tested_operators').sum())} of "
            f"{int((r.task == prim).sum())} withheld systems", "signal_recovery/reasons.csv")
        add("2. Why does the existing candidate generator discard them?",
            "; ".join(f"{k}: {int(v)}" for k, v in g.items()), "signal_recovery/reasons.csv (failure_reason column)")
    else:
        add("1. Which fault signatures exist in the raw fields?", "NOT EVALUATED (section 02 not run)", "-")
        add("2. Why does the existing candidate generator discard them?", "NOT EVALUATED", "-")
    g1 = ws.load_table("gate1", "signal_recovery")
    g2 = ws.load_table("gate2", "candidate_generation/oracle")
    if len(g1):
        x = g1[g1.task == prim].iloc[0] if (g1.task == prim).any() else g1.iloc[0]
        add("3. Which physical transformations recover genuinely new structures?",
            f"G1 {x.decision}: {int(x.additional_systems)} systems covered by M8 physics and not by M7 linked candidates "
            f"({x.diagnosis or 'see by-family coverage'})",
            "signal_recovery/gate1.csv, candidate_generation/oracle/pooled_groups.csv, "
            "geophysical_features/structural_worms/coverage_vs_fixed_scale.csv")
    else:
        add("3. Which physical transformations recover genuinely new structures?", "NOT EVALUATED", "-")
    g3 = ws.load_table("gate3", "monte_carlo/stability")
    add("4. Does Monte Carlo uncertainty reveal robust or additional candidates?",
        (f"G3 {g3.decision.iloc[0]}: {int(g3.additional_systems_at_gate_density.iloc[0])} additional systems at matched "
         f"density; ranking gain {g3.ranking_gain_physics_mc.iloc[0]}") if len(g3) else "NOT EVALUATED",
        "monte_carlo/stability/gate3.csv, stability_vs_truth.csv")
    g4 = ws.load_table("gate4", "dem/candidates")
    add("5. Does DEM add independent information?",
        (f"G4 {g4.decision.iloc[0]}: {g4.get('reason', pd.Series(['']))[0]}") if len(g4) else "NOT EVALUATED",
        "dem/candidates/gate4.csv")
    cs = ws.load_table("benchmark_selection", "fault_networks/benchmark")
    sc = ws.load_table("system_c_summary", "fault_networks")
    if len(cs):
        b = cs.sort_values("dti_crossfit", ascending=False).groupby("class").head(1)
        add("6. Can fault-network completion recover missing mapped geometry?",
            "; ".join(f"{r['class']}: best {r.completer} DTI {r.dti_crossfit:.3f} (delta vs best M7 "
                      f"{r.get('delta_vs_best_m7', np.nan):+.3f})" for _, r in b.iterrows()) +
            (f" | System C disc DTI {sc[sc.method == 'completion_all'].dti_crossfit.iloc[0]:.4f}"
             if len(sc) and (sc.method == "completion_all").any() else ""),
            "fault_networks/benchmark/benchmark_selection.csv, fault_networks/system_c_summary.csv")
    else:
        add("6. Can fault-network completion recover missing mapped geometry?", "NOT EVALUATED", "-")
    g5 = ws.load_table("gate5", "ranking")
    add("7. Can candidate-level ranking distinguish faults from unrelated structural boundaries?",
        (f"G5 {g5.decision.iloc[0]}: best learned {g5.get('best_learned', pd.Series(['-']))[0]} vs best control "
         f"{g5.get('best_control', pd.Series(['-']))[0]} (gain {g5.get('gain', pd.Series([np.nan]))[0]})")
        if len(g5) else "NOT EVALUATED", "ranking/gate5.csv, ranking/ranking_summary.csv")
    g6 = ws.load_table("gate6", "ensemble")
    P = ws.load_table("method_pooled", "validation/metrics")
    if len(g6):
        add("8. Do these methods improve the deployable official-style DTI?",
            f"G6 {g6.decision.iloc[0]} ({g6.method.iloc[0]}): {g6.reason.iloc[0]}",
            "ensemble/gate6.csv, validation/metrics/method_pooled.csv, reports/decision_verdicts.csv")
    else:
        add("8. Do these methods improve the deployable official-style DTI?", "NOT EVALUATED", "-")
    pz = ws.load_table("paired_zone_deltas", "validation/metrics")
    if len(g6) and len(pz):
        m = g6.method.iloc[0]
        z = pz[pz.method == m]
        add("9. Does the selected model generalise across geographic regions?",
            ", ".join(f"{t}:{int(zz)} {d:+.4f}" for t, zz, d in zip(z.task, z.zone, z.delta_vs_ref)) or "no paired rows",
            "validation/metrics/paired_zone_deltas.csv, regional_dti.csv")
    else:
        add("9. Does the selected model generalise across geographic regions?", "NOT EVALUATED", "-")
    sub = ws.load_table("submission_index", "submissions")
    add("10. Is the final submission scientifically and technically defensible?",
        (f"{len(sub)} validated file(s); checks failed: "
         f"{sorted(set(';'.join(sub.checks_failed.fillna('').astype(str)).split(';')) - {''}) or 'none'}")
        if len(sub) else "NO FILE WRITTEN (section 15 off or not run)", "submissions/*.json, leaderboard_log.csv")
    return out


def report(m8, gates: pd.DataFrame) -> str:
    ws = m8.ws
    L = ["# GEMS M8 - physics-guided probabilistic fault discovery", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} from the saved tables of `{ws.root}` (profile "
         f"**{m8.profile['name']}**{', provisional' if m8.profile['provisional'] else ''}). Every number is an offline, "
         "nested-cross-fitted pseudo-discovery proxy; **none is a leaderboard score**. Oracle values read the truth "
         "(diagnostic upper bounds). The zones were inspected in earlier sessions: not an untouched test set.", ""]
    L += ["## Gates", "", _md(gates.drop(columns=["details"], errors="ignore")), ""]
    sections = [("Baselines (section 01)", "baseline_table", "baseline"),
                ("Recorded M7 vs this run", "recorded_vs_this_run", "baseline"),
                ("Compatibility checks", "compatibility", "baseline"),
                ("Signal recovery: failure categories (section 02)", "summary", "signal_recovery"),
                ("Signal recovery: recall by stage", "stage_recall", "signal_recovery"),
                ("Band provenance (section 03)", "band_provenance", "geophysical_features"),
                ("Effective resolution", "effective_resolution", "geophysical_features"),
                ("Worms vs fixed scale (section 03b)", "coverage_vs_fixed_scale", "geophysical_features/structural_worms"),
                ("Magnetic-gravity agreement (section 04)", "mag_grav_agreement", "geophysical_features/multimodal"),
                ("Coverage gate (section 06)", "gate2", "candidate_generation/oracle"),
                ("Pooled recall-density (M8 vs M7)", "pooled_curve", "candidate_generation/oracle"),
                ("Monte Carlo stability vs truth (section 07)", "stability_vs_truth", "monte_carlo/stability"),
                ("Completion benchmark (section 08)", "benchmark_selection", "fault_networks/benchmark"),
                ("System C", "system_c_summary", "fault_networks"),
                ("DEM increment (section 09)", "results", "dem/candidates"),
                ("Ranking (section 11)", "ranking_summary", "ranking"),
                ("Graph (section 12)", "graph_summary", "graphs"),
                ("Rasterisation (section 13)", "rasterization_summary", "rasterization"),
                ("Representation ablation", "rasterization_ablation", "rasterization"),
                ("Fusion (section 14)", "fusion_summary", "ensemble"),
                ("Model comparison", "method_pooled", "validation/metrics"),
                ("Verdicts", "decision_verdicts", "reports"),
                ("Candidate false-length density", "false_length", "validation/metrics")]
    for title, name, sub in sections:
        t = ws.load_immutable_table(name, sub) if name == "baseline_table" else ws.load_table(name, sub)
        if len(t):
            L += [f"## {title}", "", _md(t), ""]
    L += ["## Answers", ""] + [f"**{a['question']}** {a['answer']}  \n_evidence: {a['evidence']}_" for a in answers(m8)]
    L += ["", "## Caveats", "",
          "* Pseudo-discovery withholds whole systems; new faults next to known traces are under-represented, so the "
          "known-fault ring may look better here than on the hidden test (both ring variants are exported).",
          "* Monte Carlo frequency is processing stability under ASSUMED perturbation distributions (heuristic unless "
          "measured noise was supplied), never a calibrated fault probability.",
          "* 'Raw response absent' means absent under the tested definitions and operators at 100 m, not invisible.",
          "* Provided derivative bands enter physical formulas only when verified (band provenance table).",
          "* DEM numbers use DEM-covered sub-zones only (a separate protocol, never pooled with full-zone numbers).",
          "* The hidden new-fault density is unknown: the submission area fraction is the largest unverified choice.", ""]
    return "\n".join(L)
