"""GEMS3_RESEARCH_REPORT.md and metric_optimization_report.md, written only from saved tables.

Every number is a cross-fitted held-out estimate on the development folds / splits unless labelled otherwise;
pseudo-discovery numbers are proxies, never leaderboard estimates; oracle numbers read the truth (upper bounds).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _md(df: pd.DataFrame, cols=None, n: int = 30, digits: int = 4) -> str:
    if df is None or not len(df):
        return "_not produced_\n"
    d = df[[c for c in (cols or df.columns) if c in df]].head(n).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(digits)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def summary_table(g3) -> pd.DataFrame:
    ws = g3.ws
    rows = []
    sel = g3.selection() or {"choice": {}}
    for task, c in sel["choice"].items():
        rows.append({"task": task, "method": "selected representation (rep, w, frac[, ring])",
                     "dti_crossfit": c.get("dti_crossfit"), "dti_insample": c.get("dti_insample"),
                     "params": json.dumps(c.get("params_all"))})
    ab = ws.load_table("ensemble_ablation", "metric_optimization")
    for r in ab.to_dict("records") if len(ab) else []:
        name = g3.models[0] if r["w"] == 1.0 else g3.models[1] if r["w"] == 0.0 else f"rank fusion w={r['w']:g}"
        rows.append({"task": r["task"], "method": f"{name} binary top-a", "dti_crossfit": r["dti_crossfit"],
                     "dti_insample": r["dti_insample"], "params": f"frac={r['frac_all_zones']}"})
    sp = ws.load_table("selected_parameters", "metric_optimization")
    if len(sp) and "selection" in sp:
        for r in sp[sp.selection == "absolute_threshold"].to_dict("records"):
            rows.append({"task": r["task"], "method": f"{r['model']} absolute threshold", "dti_crossfit": r["dti_crossfit"],
                         "dti_insample": r["dti_insample"], "params": r.get("params_all")})
    raw = ws.load_table("raw_scores", "metric_optimization")
    if len(raw):
        g = raw.groupby(["task", "model"])[["tp_w", "fp_w", "fn_w"]].sum().reset_index()
        for r in g.itertuples():
            rows.append({"task": r.task, "method": f"{r.model} raw sigmoid (soft, no selection)",
                         "dti_crossfit": r.tp_w / (r.tp_w + 0.2 * r.fp_w + 0.8 * r.fn_w + 1e-6), "params": "-"})
    rk = ws.load_table("ranker_summary", "ranker")
    for r in rk.to_dict("records") if len(rk) else []:
        rows.append({"task": r["task"], "method": f"candidates: {r['method']}", "dti_crossfit": r["dti_crossfit"],
                     "dti_insample": r["dti_insample"], "params": f"frac={r['frac_all_zones']}"})
    out = pd.DataFrame(rows)
    return out.sort_values(["task", "dti_crossfit"], ascending=[True, False]).reset_index(drop=True) if len(out) else out


def metric_report(g3) -> str:
    ws, m = g3.ws, g3.models
    sel = g3.selection() or {"choice": {}}
    L = ["# GEMS3 metric optimisation (Milestone A)", "",
         "No retraining. Every DTI below is pooled over held-out zones, each zone's parameters chosen on the other "
         "zones of the same task and only among candidates derived from its own dev choice. `disc_official` = "
         "pseudo-discovery under the organizers' pixel-exact known-fault mask (a proxy, not a leaderboard estimate).", ""]
    L += ["## Summary (cross-fitted)", "", _md(summary_table(g3), ["task", "method", "dti_crossfit", "dti_insample", "params"], 60)]
    L += ["## Representation ablation", "", _md(ws.load_table("representation_ablation", "metric_optimization"),
                                                  ["task", "rep", "family", "dti_crossfit", "optimism", "fold_min",
                                                   "area_frac_mean", "params_all"], 60)]
    L += ["## Fusion weight (binary top-a, area cross-fitted)", "",
          f"w = weight of {m[0]}; w = 1 is {m[0]} alone, w = 0 is {m[1]} alone.", "",
          _md(ws.load_table("ensemble_ablation", "metric_optimization"),
              ["task", "w", "dti_crossfit", "dti_insample", "optimism", "fold_min", "frac_all_zones"], 40)]
    kr = ws.load_table("known_ring_ablation", "metric_optimization")
    if len(kr):
        L += ["## Known-fault ring (disc tasks, pooled, in-sample grid)", "",
              _md(kr, ["task", "rep", "w", "frac", "ring", "dti", "recall_w", "area_frac"], 40)]
    L += ["## Selected for submission (all-zone choice; in-sample)", "",
          "```json", json.dumps(sel.get("choice", {}).get(sel.get("primary_task", ""), {}).get("params_all", {}), indent=1),
          "```", "", "Area for the full region: `submission.area_frac_source` (default: the geo all-zone fraction of "
          "the same representation and weight, `geo_frac` in manifests/selected_representation.json).", ""]
    L += ["## Decision rule", "",
          "Output representation is a significant bottleneck if the selected representation beats the better member's "
          "binary top-a map by a consistent margin on held-out zones (see `ensembles/paired_fold_deltas.csv`, worst "
          "zone included) without inflating area. Otherwise keep binary top-a.", ""]
    return "\n".join(L)


def research_report(g3) -> str:
    ws = g3.ws
    L = ["# GEMS 3.0 research report", "",
         f"Members: {', '.join(g3.models)} (FULL experiments, reused; no CNN retraining). Primary task: "
         f"`{g3.cfg['primary_task']}`. Generated from saved tables only.", "",
         "Reading rules: geographic DTI = rediscovery of mapped faults in held-out folds; `disc_official` = withheld "
         "systems under the official known-fault mask (proxy); oracle = reads the truth (upper bound). No number here "
         "is a leaderboard score.", ""]
    log = []
    p = ws.root / "manifests" / "stage_log.json"
    if p.exists():
        log = json.loads(p.read_text(encoding="utf-8"))
    L += ["## 1. Stages", "", _md(pd.DataFrame(log), ["stage", "outcome", "seconds", "error"])]
    L += ["## 2. Official metric", "", "Contract (docs/GEMS_OFFICIAL_METRIC_AUDIT.md):", "",
          _md(ws.load_table("official_contract", "audit"), ["item", "status", "source"], 40),
          "Masking conventions on synthetic cases:", "",
          _md(ws.load_table("masking_audit", "audit"), ["case", "convention", "dti", "tp_w", "fp_w"], 40)]
    comp = ws.load_table("official_metric_comparison", "audit")
    if len(comp):
        L += ["B0 / B1 discovery DTI under each convention (pooled splits):", "",
              _md(comp[comp.unit == "pooled_splits"], ["task", "model", "target_frac", "dti", "recall_w", "fp_w"], 40)]
    L += ["## 3. Baselines, thresholds, widths, fusion (cross-fitted)", "",
          _md(summary_table(g3), ["task", "method", "dti_crossfit", "dti_insample", "params"], 60),
          "Paired held-out deltas per zone: `ensembles/paired_fold_deltas.csv`.", ""]
    sub = ws.load_table("submission_candidates", "submissions")
    L += ["## 4. Submission readiness", "", _md(sub), "Files in `submissions/` (validated against the template; "
          "never uploaded). Raw member maps and the fused score are in `predictions/`.", ""]
    L += ["## 5. Candidates and oracle headroom", "",
          _md(ws.load_table("label_free_sources", "proposals")),
          _md(ws.load_table("oracle_decision", "proposals"))]
    orc = ws.load_table("oracle_recall", "proposals")
    if len(orc) and "subset" in orc:
        L += [_md(orc[orc.subset == "all_unranked"], ["task", "zone", "n_candidates", "recall_w", "recall@300m",
                                                      "system_recall", "short_system_recall", "dti",
                                                      "candidate_km_per_100km2"])]
    L += ["## 6. Candidate ranker", "", _md(ws.load_table("ranker_vs_pixel_models", "ranker")),
          _md(ws.load_table("training_audit", "ranker"))]
    L += ["## 7. DEM gate", "", _md(ws.load_table("decision", "dem")), _md(ws.load_table("coverage", "dem"))]
    L += ["## 8. Map completion (deterministic baseline)", "", _md(ws.load_table("selection", "map_completion"))]
    L += ["## 9. Recommended next experiment", "", recommendation(g3), ""]
    return "\n".join(L)


def recommendation(g3) -> str:
    """Rule-based: what the saved evidence supports (no projection of improvements)."""
    ws = g3.ws
    out = []
    summ = summary_table(g3)
    task = g3.cfg["primary_task"]
    if len(summ):
        s = summ[summ.task == task].dropna(subset=["dti_crossfit"])
        if len(s):
            top = s.iloc[0]
            out.append(f"* Best held-out method on `{task}`: **{top.method}** (cross-fitted DTI {top.dti_crossfit:.4f}).")
    dec = ws.load_table("oracle_decision", "proposals")
    if len(dec):
        d = dec[dec.task == task]
        if len(d):
            r = d.iloc[0]
            out.append(f"* Candidate oracle on `{task}`: {r.decision} (oracle DTI {r.oracle_dti:.4f}, gain "
                       f"{r.oracle_gain:+.4f} over the selected representation, system recall "
                       f"{r.system_recall_all_candidates:.3f}).")
            out.append("  - CONTINUE -> enable `run_candidate_ranker`." if r.decision == "CONTINUE" else
                       "  - STOP -> improve candidate generation (sources, scales, tip lengths) before any ranker.")
    dem = ws.load_table("decision", "dem")
    if len(dem):
        out.append(f"* DEM gate: {dem.decision.iloc[0]} (mean held-out AP gain {dem.mean_ap_gain.iloc[0]:+.4f}).")
    mc = ws.load_table("selection", "map_completion")
    if len(mc):
        best = mc.sort_values("dti_crossfit", ascending=False).iloc[0]
        out.append(f"* Map completion: endpoint extension recovers `{best['class']}` best "
                   f"(cross-fitted DTI {best.dti_crossfit:.4f}); a learned completer is justified only for classes where "
                   "this baseline is clearly above zero.")
    if not out:
        out.append("* No stage has produced evidence yet: run the GEMS3 cells.")
    return "\n".join(out)
