"""Figures (each saved with ``<name>_data.csv``) and reports, from persisted tables only. A figure / section whose
inputs were not produced is skipped and named; nothing is filled from defaults or history."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .atlas import latest_reviews
from .benchmark import REF


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def md(df, cols=None):
    from ..p6.boost_audit import md_table

    return md_table(df, cols, 4) if df is not None and len(df) else "_ABSENT (stage not produced)_\n"


# =============================================================================== figures
def fig_atlas(R):
    C = R.ws.load_table("atlas/atlas_index.csv")
    if not len(C):
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 8))
    for k, g in C.groupby("category"):
        ax.scatter(g.x_utm / 1e3, g.y_utm / 1e3, s=25, label=k)
    ax.set(xlabel="UTM 11N E (km)", ylabel="N (km)", title="Atlas candidates by category (regional diversity)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    return fig, C[["candidate_id", "category", "region_id", "review_split", "x_utm", "y_utm"]]


def fig_modality(R):
    F = R.load_parquet("physics/structural_evidence.parquet")
    if not len(F):
        return None
    cols = ["GEOLOGICAL_CONTRAST", "GEOLOGICAL_DISRUPTION", "STRUCTURAL_CONTINUITY", "MULTI_SUPPORT",
            "alternative_explanation_flag"]
    D = F.groupby("category")[[c for c in cols if c in F]].mean().reset_index()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(D))
    for i, c in enumerate([c for c in cols if c in D]):
        ax.bar(x + i * 0.16, D[c], 0.16, label=c)
    ax.set_xticks(x + 0.32)
    ax.set_xticklabels(D.category, rotation=30, ha="right", fontsize=7)
    ax.set(ylabel="share of candidates", title="Observed evidence classes by category (observations, not labels)")
    ax.legend(fontsize=7)
    return fig, D


def fig_scale(R):
    L = R.ws.load_table("physics/localization_error.csv")
    if not len(L) or "truth_within_0.3km" not in L:
        return None
    D = L.groupby(["source", "scale_km"], dropna=False)[[c for c in L if c.startswith("truth_within_")]].mean() \
        .reset_index()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    for s, g in D.groupby("source"):
        ax.plot(g.scale_km.fillna(0), g["truth_within_0.3km"], "o-", label=f"{s} <= 300 m")
        ax.plot(g.scale_km.fillna(0), g["truth_within_3km"], "x--", label=f"{s} <= 3 km")
    ax.set(xlabel="smoothing scale (km)", ylabel="share of mapped fault pixels near a crest",
           title="Trace vs corridor: localisation of crests vs scale (V3 held-out folds, equal 2 % budget)")
    ax.legend(fontsize=6)
    return fig, D


def fig_localization(R):
    L = R.ws.load_table("physics/localization_error.csv")
    if not len(L) or "pred_to_truth_median_km" not in L:
        return None
    D = L[["source", "scale_km", "unit", "pred_to_truth_median_km", "truth_to_pred_median_km", "corridor_index"]]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    lab = D.source + "@" + D.scale_km.fillna(0).astype(str)
    ax.scatter(D.pred_to_truth_median_km, D.truth_to_pred_median_km)
    for x, y, t in zip(D.pred_to_truth_median_km, D.truth_to_pred_median_km, lab):
        ax.annotate(t, (x, y), fontsize=6)
    ax.axvline(0.3, color="r", lw=0.8)
    ax.axhline(0.3, color="r", lw=0.8)
    ax.set(xlabel="median predicted -> mapped (km)", ylabel="median mapped -> predicted (km)",
           title="Localisation error (red = official 300 m tolerance)")
    return fig, D


def fig_budget(R, y="dti"):
    C = R.ws.load_table("benchmark/tp_fp_fn_curves.csv")
    if not len(C):
        return None
    D = C[C.model.isin(["W2", REF, "A1_label_free", "A2_B0", "A3_B1"])]
    scs = list(dict.fromkeys(D.scenario))
    plt = _plt()
    fig, axs = plt.subplots(1, len(scs), figsize=(5 * len(scs), 4), squeeze=False)
    for ax, sc in zip(axs[0], scs):
        for (m, rp), g in D[D.scenario == sc].groupby(["model", "rep"]):
            if y == "dti":
                ax.plot(g.mass_frac, g.dti, "*" if m == REF else "o-", label=f"{m}|{rp}", ms=8 if m == REF else 3)
            else:
                for comp, ls in (("tp_w", "-"), ("fp_w", "--"), ("fn_w", ":")):
                    if m in ("W2", REF):
                        ax.plot(g.mass_frac, g[comp], ls, label=f"{comp} {rp}" if m == "W2" else None)
        ax.set(title=sc, xlabel="realised mass / eligible", ylabel="official DTI" if y == "dti" else "weighted count")
    axs[0][-1].legend(fontsize=5)
    return fig, D


def fig_binary_vs_continuous(R):
    A = R.ws.load_table("benchmark/area_response.csv")
    if not len(A):
        return None
    D = A[(A.model == "W2") & (A.budget_axis == "mass_frac")]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    for (sc, rp), g in D.groupby(["scenario", "rep"]):
        ax.plot(g.budget, g.dti_interp, "o-", label=f"{sc} {rp}")
    ax.set(xlabel="common realised probability mass / eligible", ylabel="interpolated DTI",
           title="Binary vs soft vs continuous W2 outputs at EQUAL realised mass")
    ax.legend(fontsize=6)
    return fig, D


def fig_profiles(R):
    P = R.ws.load_table("physics/scale_profiles.csv")
    if not len(P):
        return None
    ids = list(dict.fromkeys(P.candidate_id))[:6]
    D = P[P.candidate_id.isin(ids)]
    plt = _plt()
    fig, axs = plt.subplots(1, len(ids), figsize=(3.5 * len(ids), 3), squeeze=False)
    for ax, cid in zip(axs[0], ids):
        for (m, s), g in D[D.candidate_id == cid].groupby(["modality", "smoothing_px"]):
            v = g.value - np.nanmean(g.value)
            ax.plot(g.distance_km, v / (np.nanstd(v) or 1), label=f"{m} s{s}")
        ax.axvspan(-0.3, 0.3, color="r", alpha=0.15)
        ax.set(title=cid, xlabel="km across trace")
    axs[0][0].legend(fontsize=5)
    fig.suptitle("Perpendicular profiles (z-scored; red = +-300 m)")
    return fig, D


def fig_models(R):
    S = R.ws.load_table("benchmark/model_comparison.csv")
    if not len(S):
        return None
    D = S[~S.get("diagnostic", pd.Series(False, index=S.index)).fillna(False).astype(bool)][
        ["scenario", "model", "dti", "worst_zone_dti", "protocol"]]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    ms = list(dict.fromkeys(D.model))
    for i, m in enumerate(ms):
        g = D[D.model == m]
        ax.bar(np.arange(g.scenario.nunique()) + i * 0.8 / len(ms), g.dti, 0.8 / len(ms), label=m)
    ax.set_xticks(np.arange(D.scenario.nunique()) + 0.4)
    ax.set_xticklabels(list(dict.fromkeys(D.scenario)))
    ax.set(ylabel="held-out DTI", title="A1-A5 vs frozen submitted W2 (A4 frozen cell; others fixed policy)")
    ax.legend(fontsize=6)
    return fig, D


def fig_cases(R):
    C = R.ws.load_table("atlas/atlas_index.csv")
    if not len(C):
        return None
    pick = pd.concat([C[C.category == k].head(1) for k in ("B_missed_mapped", "C_w2_unmapped", "D_lineament_low_w2",
                                                           "F_suspected_nonfault")])
    plt = _plt()
    from matplotlib.image import imread

    fig, axs = plt.subplots(1, len(pick), figsize=(6 * len(pick), 4), squeeze=False)
    for ax, r in zip(axs[0], pick.itertuples()):
        p = R.ws.path(f"atlas/panels/{r.candidate_id}_diagnostic.png")
        if p.exists():
            ax.imshow(imread(str(p)))
        ax.set(title=f"{r.candidate_id} {r.category}")
        ax.set_axis_off()
    return fig, pick[["candidate_id", "category", "x_utm", "y_utm"]]


FIGURES = {"geological_atlas_overview": fig_atlas, "modality_evidence": fig_modality, "scale_analysis": fig_scale,
           "localization_analysis": fig_localization, "dti_vs_budget": fig_budget,
           "tp_fp_fn_vs_budget": lambda R: fig_budget(R, "comp"), "binary_vs_continuous": fig_binary_vs_continuous,
           "localization_profiles": fig_profiles, "model_comparison": fig_models, "success_failure_cases": fig_cases}


def figures(R) -> list[str]:
    miss = []
    for name, fn in FIGURES.items():
        try:
            out = fn(R)
        except Exception as e:
            miss.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if out is None:
            miss.append(f"{name}: inputs absent")
            continue
        R.ws.save_fig(out[0], f"figures/{name}.png", out[1])
    R.ws.save_text("\n".join(miss) + "\n" if miss else "none\n", "figures/MISSING.txt")
    return miss


# =============================================================================== reports
def reports(R) -> dict:
    ws = R.ws
    t, j = ws.load_table, (lambda r: ws.read_json(r) or {})
    A = t("audit/prior_results_audit.csv")
    I = t("audit/source_inventory.csv")
    st = {s: (v if v != "NOT_RUN" else R.manifest(s).get("status", "NOT_RUN")) for s, v in R.status.items()}
    sp, pd_, ch = j("benchmark/selected_policy.json"), j("terrain/pilot_decision.json"), j(
        "challenger/selected_challenger.json")
    sub = j("submissions/submission_manifest.json")
    Q = [("1. Are we learning fault evidence or patterns associated with mapped faults?",
          "benchmark/source_complementarity.csv + A1 (label-free) vs A4 per scenario; V1 >> public is the warning"),
         ("2. Are broad potential-field anomalies too ambiguous for 300 m?", "physics/localization_error.csv "
          "(truth within 300 m vs 3 km, corridor index)"),
         ("3. Does the skeleton representation discard useful predictions?", "benchmark/area_response.csv, "
          "selected_policy.json"),
         ("4. Are geophysical edges confused with contacts / intrusions / artifacts?",
          "physics/structural_evidence.parquet (contrast without disruption) + atlas reviews"),
         ("5. Do independent geological observations exist?", "audit/source_inventory.csv (independent_labels), "
          "atlas/reviewed_annotations.csv"),
         ("6. Can high-resolution terrain localise structures?", "terrain/pilot_decision.json"),
         ("7. Is the existing validation optimistic about hidden faults?", "audit/validation_contract.csv"),
         ("8. What would convince a geologist?", "MULTI_SUPPORT + DISRUPTION observations and an EVIDENCE_ONLY "
          "review naming no alternative explanation")]
    aud = ["# Scientific audit", "", "## Prior results (classification + Colab check)", "",
           md(A, ["finding_id", "classification", "check_result", "claim", "reason", "check_detail"]),
           "## Central questions -> where the evidence is", "", *[f"* **{q}** {w}" for q, w in Q], "",
           "## Stage status", "", md(pd.DataFrame([{"stage": k, "status": v} for k, v in st.items()])), ""]
    ws.save_text("\n".join(aud), "reports/SCIENTIFIC_AUDIT.md")
    miss = I[~I.present.astype(bool)] if len(I) and "present" in I else I
    ws.save_text("\n".join(["# What we are missing", "", "Sources that are absent or unavailable (a missing source is "
                            "never evidence against a fault):", "", md(miss), ws.path("audit/known_limitations.md")
                            .read_text(encoding="utf-8") if ws.path("audit/known_limitations.md").exists() else ""]),
                 "reports/WHAT_WE_ARE_MISSING.md")
    C, F = t("atlas/atlas_index.csv"), R.load_parquet("physics/structural_evidence.parquet")
    rv = R.load_reviews()
    ws.save_text("\n".join(["# Atlas findings", "", f"{len(C)} candidates; reviews: {len(rv)} events.", "",
                            md(C.groupby(["category", "review_split"]).size().reset_index(name="n") if len(C) else C),
                            "## Observed evidence classes by category (observations, not labels)", "",
                            md(F.groupby("category")[[c for c in F if c.isupper()]].mean().reset_index() if len(F)
                               else F), "## Latest reviews by class", "",
                            md(latest_reviews(rv).groupby(["label_class", "interpretation"]).size()
                               .reset_index(name="n") if len(rv) else rv), "Case studies: atlas/case_studies.md", ""]),
                 "reports/ATLAS_FINDINGS.md")
    S = t("benchmark/model_comparison.csv")
    ws.save_text("\n".join(["# Benchmark findings", "", f"Reference: **{REF}** (frozen submitted cell "
                            f"{(sp.get('frozen_policy') or {}).get('cell')}). Other rows: fixed-policy rule.", "",
                            md(S, ["scenario", "model", "protocol", "dti", "worst_zone_dti", "params_all", "caveat"]),
                            "## Paired comparisons", "", md(t("benchmark/paired_comparison.csv")),
                            "## Representation decision", "", f"**{sp.get('status', 'NOT_EVALUATED')}**: "
                            f"{sp.get('rule', '')}", f"deployable cell {sp.get('deployable_cell')}, exportable "
                            f"{sp.get('exportable')}", "", "## Source complementarity", "",
                            md(t("benchmark/source_complementarity.csv")), "## Reviewed agreement (A6)", "",
                            md(t("benchmark/reviewed_agreement.csv")), ""]), "reports/BENCHMARK_FINDINGS.md")
    ws.save_text("\n".join(["# DEM pilot findings", "", f"Decision: **{pd_.get('decision', 'NOT_EVALUATED')}** "
                            f"({pd_.get('source', 'not run')})", f"Reasons: {pd_.get('reasons')}", "",
                            md(t("terrain/pilot_comparison.csv")), md(t("terrain/tile_coverage.csv")), ""]),
                 "reports/DEM_PILOT_FINDINGS.md")
    ws.save_text("\n".join(["# Challenger decision", "", f"Selected: **{ch.get('selected_case', 'NOT_EVALUATED')}** - "
                            f"{ch.get('note', '')}", "", "Conditions (NOT_EVALUATED = not measured, never negative):",
                            "", f"`{json.dumps(ch.get('conditions'), default=str)}`", "", f"Export: {ch.get('export')}",
                            "", md(t("challenger/comparison.csv")), md(t("challenger/ablation.csv")), ""]),
                 "reports/CHALLENGER_DECISION.md")
    files = sub.get("files") or []
    ready = [f for f in files if f.get("status") == "READY_FOR_UPLOAD"]
    lb = ["# Leaderboard recommendation", "", "Public reference: W2 0.0942 (external only).", ""]
    lb += ([f"Upload ONE file manually: `{f.get('path_drive')}` (sha256 `{f.get('sha256')}`, role {f.get('role')})."
            for f in ready[:1]] or [f"No file to upload: {sub.get('note', 'export stage not run')}"])
    ws.save_text("\n".join(lb + [""]), "reports/LEADERBOARD_RECOMMENDATION.md")
    nxt = next_action(st, sp, ch, rv, ready)
    ws.save_text(f"# Next action\n\n{nxt}\n", "reports/NEXT_ACTION.md")
    ws.push()
    return {"next_action": nxt}


def next_action(st: dict, sp: dict, ch: dict, rv: pd.DataFrame, ready: list) -> str:
    bad = [s for s, v in st.items() if v in ("FAILED", "TESTS_FAILED", "BLOCKED")]
    if bad:
        return f"Fix the failing stage(s) {bad} (manifests/stages/*.json: exception + affected file) and re-run."
    if st.get("04_atlas") not in ("COMPLETE", "REUSED_VALID_CACHE"):
        return "Run mode ATLAS to completion."
    if not len(rv):
        return ("Review the atlas: open atlas/atlas_overview.html, review EVIDENCE_ONLY panels first (widget cell or "
                "annotation_template.csv), then run mode BENCHMARK.")
    if not sp:
        return "Run mode BENCHMARK (representation + baselines vs the frozen submitted W2)."
    if ready:
        return f"Upload {ready[0].get('file')} manually and record its public score; no further tuning before."
    c = (ch or {}).get("selected_case")
    if not c:
        return "Run mode CHALLENGER (decision from the measured evidence)."
    return {"CASE_2_VERIFIER": "Extend reviews in TRAIN regions and rasterise verified candidates for an official-"
                               "metric test.",
            "CASE_3_TERRAIN": "Expand the DEM pilot to the windows with unique terrain structures (opt-in download).",
            "CASE_4_REPRESENTATION": "Request an authorised GPU session for one self-supervised geophysical "
                                     "representation experiment.",
            "NO_CHALLENGER": "Keep W2; review more EVIDENCE_ONLY candidates (independent evidence is the gap).",
            "CASE_1_ALLOCATION": "Export blocked or not ready: see submissions/export_gates.csv."}.get(c, c)
