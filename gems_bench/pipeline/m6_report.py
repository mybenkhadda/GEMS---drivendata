"""``reports/m6/M6_RESEARCH_REPORT.md``: built only from the tables the M6 units saved (no number is typed in).

Sections follow the milestone: audit (M6.1), paired comparison (M6.2), calibration (M6.3), ensembles (M6.4),
complementarity and geological errors (M6.5), postprocessing (M6.6), selection gate (M6.7), limitations.
A section whose unit has not run says so instead of showing an empty table.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .screening import _md_table


def _num(x, fmt="{:+.4f}") -> str:
    try:
        x = float(x)
        return fmt.format(x) if np.isfinite(x) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def _section(title: str, df: pd.DataFrame, cols: list[str], intro: str = "", n: int = 40) -> list[str]:
    lines = [f"## {title}", ""]
    if intro:
        lines += [intro, ""]
    lines += [_md_table(df, cols, n) if df is not None and len(df) else "_not run (enable the phase in M6_CONFIG)._\n"]
    return lines


def central_answer(m6) -> list[str]:
    """Numbers for the central question, read from the candidate / gate tables; no interpretation beyond them."""
    ws = m6.ws
    et = ws.load_table("ensemble_candidates")
    rec = json.loads((ws.root / "manifests" / "recommendation.json").read_text(encoding="utf-8")) \
        if (ws.root / "manifests" / "recommendation.json").exists() else {}
    prim = json.loads((ws.root / "manifests" / "ensemble_primary.json").read_text(encoding="utf-8")).get("primary") \
        if (ws.root / "manifests" / "ensemble_primary.json").exists() else None
    b0, b1 = m6.pair
    pp = f"disc_dti_{m6.cfg['primary_policy']}"
    lines = ["## Central question", "",
             f"Can an ensemble combine {b0}'s short-fault recall and retained-FP robustness with {b1}'s geographic "
             "generalisation and discovery DTI?", ""]
    if not len(et) or prim is None or prim not in set(et.candidate):
        return lines + ["_The ensemble phase has not produced results yet._", ""]
    r = et.set_index("candidate").loc[prim]
    lines += [f"Primary (prespecified) candidate: `{prim}` (members {r.members}, nested weights per fold "
              f"{r.geo_weights_per_fold}).", ""]
    rows = []
    for metric in ("geo_dti", "geo_fold_min", pp, "disc_dti_retained_fp", "short_recall_w"):
        rows.append({"metric": metric, "ensemble": r.get(metric, np.nan),
                     f"delta_vs_{b0}": r.get(f"delta_{metric}_vs_{b0}", np.nan),
                     f"delta_vs_{b1}": r.get(f"delta_{metric}_vs_{b1}", np.nan)})
    lines += [_md_table(pd.DataFrame(rows), list(rows[0])), ""]
    d = ws.load_table("promotion_decisions")
    if len(d):
        v = d[(d.candidate == prim) & (d.control == "ALL")]
        if len(v):
            lines += [f"Gate verdict for `{prim}`: **{v.verdict.iloc[0]}** ({v.reason.iloc[0]}).", ""]
    if rec:
        lines += [f"Recommendation: **{rec.get('recommended') or 'none'}** ({rec.get('verdict')}; rule: {rec.get('rule')}).", ""]
    return lines


def build_report(m6) -> str:
    ws, cfg = m6.ws, m6.cfg
    prov = json.loads((ws.root / "manifests" / "provenance.json").read_text(encoding="utf-8")) \
        if (ws.root / "manifests" / "provenance.json").exists() else {}
    pp = cfg["primary_policy"]
    L = ["# GEMS 2.1 - M6 research report: model selection, ensembles and geological error analysis", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `gems_bench.pipeline.m6` (version {prov.get('m6_version', '')}) "
         f"at code commit `{(prov.get('code') or {}).get('git_commit', 'unknown')}`; evaluation protocol "
         f"`{(prov.get('protocol') or {}).get('protocol_hash', '')}`. Every number is read from the tables under "
         "`reports/m6/`, computed by this run from saved FULL predictions. No model was trained. Local scores are not "
         "leaderboard scores; geographic and pseudo-discovery DTI are different quantities and are never averaged._", ""]
    L += central_answer(m6)
    inv = ws.load_table("artifact_inventory", sub=".")
    L += _section("M6.1 Artifact audit", inv, ["model", "kind", "status", "experiment_id", "profile", "n_disc_maps",
                                               "feature_version", "git_commit", "reasons"],
                  "Status: COMPLETE / PARTIAL / INCOMPATIBLE / MISSING / INVALID. Only COMPLETE FULL runs are analysed "
                  "and eligible for promotion. Provenance: `provenance_audit.csv`; pairwise: `model_compatibility.csv`.")
    rc = ws.load_table("reference_check")
    if len(rc):
        diff = rc[rc.status == "DIFFERENT"]
        L += ["### Reference observations vs persisted / recomputed scores", "",
              f"{len(diff)} of {len(rc)} reference numbers differ beyond tolerance (reported, not reconciled).", "",
              _md_table(rc, ["model", "metric", "reference_observation", "persisted", "recomputed", "status"]), ""]
    mc = ws.load_table("model_comparison")
    L += _section("M6.2 Consolidated comparison (one protocol)", mc,
                  ["model", "geo_dti_area", "geo_fold_min", "geo_fold_std", f"disc_dti_{pp}", "disc_dti_retained_fp",
                   "short_recall_w", "geo_far_share_at_match"])
    ps = ws.load_table("paired_summary")
    L += _section("Paired differences (per fold / split)", ps,
                  ["task", "model", "reference", "policy", "n_units", "mean_delta", "min_delta", "max_delta",
                   "units_improved", "sign_pattern", "effect_size_dz"],
                  "Three folds / three splits: directional consistency and effect sizes are descriptive, not "
                  "population-level tests.")
    bt = ws.load_table("spatial_bootstrap")
    L += _section("Spatial-block bootstrap of held-out differences", bt,
                  ["task", "model", "reference", "observed_delta", "boot_q05", "boot_q95", "p_delta_positive",
                   "n_blocks", "block_km"],
                  "Blocks (the folds' own sampling lattice) are resampled inside each held-out zone, the same draw for "
                  "both models, at the cross-fitted operating point.")
    cr = ws.load_table("calibration_results")
    L += _section("M6.3 Calibration and area budget", cr,
                  ["model", "family", "label_free", "dti_crossfit", "dti_insample", "optimism", "fold_min",
                   "choice_on_grid_edge"],
                  "Every label-dependent parameter is chosen on the other folds (`calibration_selection_audit.csv`).")
    iv = ws.load_table("calibration_invariance")
    if len(iv):
        L += ["Invariance check (binary per-fold masks of raw vs rank-normalised scores must coincide):", "",
              _md_table(iv, ["model", "max_abs_diff", "status"]), ""]
    tr = ws.load_table("threshold_transfer")
    if len(tr):
        t = tr.groupby(["model", "policy"]).agg(heldout_dti=("heldout_dti", "mean"),
                                                 area_ratio_min=("area_ratio_heldout_vs_dev", "min"),
                                                 area_ratio_max=("area_ratio_heldout_vs_dev", "max")).reset_index()
        L += ["Threshold transfer (absolute score cut vs per-fold area fraction, both chosen on the other folds):", "",
              _md_table(t, list(t.columns)), ""]
    et = ws.load_table("ensemble_candidates")
    L += _section("M6.4 Ensemble candidates (nested selection)", et,
                  ["candidate", "members", "method", "normalisation", "discovery_weights", "geo_dti", "geo_fold_min",
                   f"disc_dti_{pp}", "disc_dti_retained_fp", "short_recall_w", f"delta_geo_dti_vs_{m6.pair[0]}",
                   f"delta_geo_dti_vs_{m6.pair[1]}", f"delta_disc_dti_{pp}_vs_{m6.pair[0]}",
                   f"delta_disc_dti_{pp}_vs_{m6.pair[1]}", "weight_dispersion", "relative_inference_cost",
                   "extension_justified"],
                  "Geographic weights and area budget per held-out fold are chosen on the other folds; discovery weights "
                  "never see the scored split's withheld systems (`ensemble_index_audit.csv`, "
                  "`ensemble_selection_discovery.csv`).")
    pc_ = ws.load_table("model_complementarity")
    if len(pc_):
        pc_ = pc_[pc_.zone.astype(str) == "all"]
    L += _section("M6.5 Complementarity at identical area budgets", pc_,
                  ["budget", "pearson", "spearman", "iou", "overlap_a_in_b", "a_only_km2", "b_only_km2", "both_km2",
                   "a_only_far_km2", "b_only_far_km2"])
    fs = ws.load_table("fault_system_summary")
    if len(fs):
        fs = fs[fs.stratum.isin(["length_bin", "fold"])]
    L += _section("Mapped faults recovered by both / one / neither (geographic OOF)", fs,
                  ["stratum", "value", "class", "n_segments", "trace_km", "share_of_stratum_km"], n=60)
    dc = ws.load_table("discovery_system_classes")
    if len(dc):
        dcs = dc.groupby(["split", "class"]).agg(n_systems=("system_id", "size"), trace_km=("trace_km", "sum")).reset_index()
        L += _section("Withheld systems by recovery class (pseudo-discovery)", dcs, list(dcs.columns))
    ea = ws.load_table("error_attribution")
    L += _section("Attributes of faults missed by both models (descriptive)", ea,
                  ["attribute", "hypothesis", "median_missed_by_both", "median_recovered_by_both",
                   "ratio_missed_over_recovered", "n_missed", "n_recovered"])
    pa = ws.load_table("postprocessing_ablation")
    L += _section("M6.6 Postprocessing ablation (nested)", pa,
                  ["model", "operation", "dti_crossfit", "delta_vs_identity", "delta_short_recall_w", "n_components",
                   "delta_fragmentation_per_100km", "delta_n_junctions", "delta_far_share", "n_folds_applied"],
                  "Each operation's parameter and area budget are chosen on the other folds; the identity is always a "
                  "candidate. An operation is combined only if it helped alone without costing short-fault recall.")
    pdx = ws.load_table("promotion_decisions")
    L += _section("M6.7 Selection gate", pdx[pdx.control == "ALL"] if len(pdx) else pdx,
                  ["candidate", "verdict", "reason"])
    if len(pdx):
        crit = [c for c in pdx.columns if c[:2].rstrip("_").isdigit()]
        L += ["Per control:", "", _md_table(pdx[pdx.control != "ALL"], ["candidate", "control", "verdict"] + crit, 80), ""]
    L += ["## Limitations", "",
          "* Three geographic folds and three pseudo-discovery splits: every difference is reported per unit; no "
          "population-level significance is claimed.",
          "* Mapped faults are an incomplete truth: predictions away from mapped faults are unmapped predictions, not "
          "confirmed faults, and pseudo-discovery only approximates undiscovered faults.",
          "* The block bootstrap is conditional on the protocol (folds, splits, cross-fitted budgets).",
          "* Postprocessing and ensemble choices are nested on the folds; the final full-region export would apply "
          "the rule chosen on all folds, whose in-sample optimism is reported (`dti_insample_coarse`, `optimism`).",
          "* Registry promotion and leaderboard submission stay manual decisions.", ""]
    log = ws.load_table("unit_log", sub="manifests")
    if len(log):
        L += ["## Analysis units", "", _md_table(log, ["unit", "outcome", "seconds", "fingerprint"]), ""]
    return "\n".join(L)
