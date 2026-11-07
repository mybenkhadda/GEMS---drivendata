"""Evidence-grounded interpretations (Observation -> Interpretation -> Limitation -> Modeling implication)
generated ONLY from computed statistics, plus the final report files.

Nothing here is pre-written about results: every number is read from a stage result; when a stage has not
run, the text says so.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .cache import atomic_write_csv, atomic_write_json, atomic_write_parquet, atomic_write_text
from .contracts import StageResult

GROUP_STAGE = {
    "prediction_distribution": "baseline_diagnostics", "known_fault_proximity": "known_fault_proximity_diagnostics",
    "split_integrity": "audit_original_splits", "original_dti": "recompute_original_metrics",
    "full_area_coverage": "ensure_full_area_predictions", "full_area_dti": "score_full_area_holdouts",
    "error_atlas": "spatial_error_atlas", "expert_complementarity": "expert_diagnostics",
    "geological_disagreement": "geological_disagreement_diagnostics", "structural_strata": "structural_stratification",
    "candidate_lineaments": "candidate_lineament_review", "transform_tradeoffs": "evaluate_fixed_transform_shortlist",
    "crossfit_uncertainty": "crossfit_recovery_decision",
}


def _f(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}g}"


def _pct(x):
    try:
        return f"{100 * float(x):.1f}%" if np.isfinite(float(x)) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def _block(o, i, l, m) -> dict:
    return {"observation": o, "interpretation": i, "limitation": l, "implication": m}


def interpret(group: str, run) -> dict:
    stage = GROUP_STAGE[group]
    R: StageResult | None = run.results.get(stage)
    if R is None:
        return _block(f"Stage `{stage}` has not run in this session.", "-", "-", "-")
    s = R.stats
    ref = run.cfg["reference_values"]
    if group == "prediction_distribution":
        th = R.thresholds
        low = 1 - float(th[th.threshold == 0.05].mass_fraction_above.iloc[0]) if (th.threshold == 0.05).any() else np.nan
        conc = s["share_mass_top_1pct_pixels"]
        return _block(
            f"{s['eligible_pixels_outside_mapped']:,} eligible pixels outside mapped faults carry a total mass of "
            f"{_f(s['total_mass_outside_mapped'], 6)} (user-reported V1 diagnostic: {ref['mass_outside_mapped']:,} over "
            f"{ref['eligible_outside_mapped']:,}). Median {_f(s['median'])}, q99 {_f(s['q99'])}; the top 1% of pixels carry "
            f"{_pct(conc)} of the mass; pixels with p < 0.05 carry {_pct(low)}; Gini {_f(s['gini'], 3)}. Nodata checks: "
            f"{s['nan_inside_eligible']} NaN inside, {s['finite_outside_valid']} finite outside, {s['out_of_range_values']} out of range.",
            ("Confidence is strongly concentrated in a small set of pixels." if conc >= 0.3 else
             "Much of the mass is diffuse background spread over many pixels rather than a few sharp structures."),
            "A distribution says nothing about whether the mass is on real faults.",
            f"FP_w is linear in p, so every unit of mass away from new truth costs alpha = {run.cfg['metric']['alpha']}; "
            f"the maximum FP penalty of this raster is about {_f(run.cfg['metric']['alpha'] * s['total_mass_outside_mapped'], 5)}.")
    if group == "known_fault_proximity":
        t = R.table
        return _block(
            f"{_pct(s['mass_within_1km'])} of the mass lies within 1 km of mapped faults on {_pct(s['area_within_1km'])} of the "
            f"area (user-reported: {_pct(ref['mass_within_1km'])} / {_pct(ref['area_within_1km'])}); {_pct(s['mass_beyond_3km'])} "
            f"beyond 3 km. Enrichment by band: " + ", ".join(f"{r.band} {_f(r.enrichment, 3)}x" for r in t.itertuples())
            + f". Mean confidence near trace endpoints / mid-trace (same distance) = {_f(s['endpoint_ratio'], 3)}. "
            f"{s['n_far_high_conf_components']} high-confidence components lie > 3 km from any mapped trace. Median block "
            f"concentration ratio {_f(s['median_block_concentration'], 3)}.",
            ("The model behaves mostly like a mapped-fault neighbourhood detector: mass is enriched next to known traces."
             if s["enrichment_0_300"] >= 2 else "Mass is only moderately enriched near mapped traces."),
            "Proximity is a diagnostic; near-trace pixels can host genuine new branches, extensions or corrections, so "
            "enrichment alone does not show the mass is wrong.",
            "If hidden new faults are mostly away from mapped traces, near-trace mass is penalised as FP by the official "
            "convention; the full-area holdout (section 08) measures this directly. No near-fault suppression is applied.")
    if group == "split_integrity":
        return _block(
            f"All checkpoint train_mask_sha values {'match' if s['all_match'] else 'DO NOT match'} the reconstruction. "
            f"{s['n_held_components']} held components; {_pct(s['share_held_within_1km_of_visible'])} lie within 1 km and "
            f"{_pct(s['share_held_within_3km_of_visible'])} within 3 km of a training-visible mapped trace; median held length "
            f"{_f(s['median_held_length_km'], 3)} km.",
            ("Many withheld 'systems' are fragments adjacent to traces the checkpoint saw in training, so the discovery "
             "splits partly test continuation of familiar structures." if (s["share_held_within_1km_of_visible"] or 0) >= 0.3
             else "Most held components are separated from training-visible traces by more than 1 km."),
            "Connected-component grouping is a raster property, not a verified geological-system partition.",
            "A geologically grouped split would require retraining and is out of scope here.")
    if group == "original_dti":
        c = R.components
        return _block(
            f"Recomputed pooled geographic {_f(s.get('pooled_GEOGRAPHIC'), 6)} (user-reported {ref['geographic_proxy_dti']}), "
            f"withheld-system {_f(s.get('pooled_WITHHELD_SYSTEM'), 6)} (user-reported {ref['narrow_discovery_proxy_dti']}). "
            f"Max |recomputed - V1 table| fold DTI {_f(s.get('max_abs_diff_vs_v1_table'))}. ROI status: "
            + ", ".join(f"{r.unit} {r.roi_status}" for r in c.itertuples() if hasattr(r, "roi_status")) + ".",
            "The original proxies are reproduced from the saved rasters." if (s.get("max_abs_diff_vs_v1_table") or 1) < 1e-4
            else "Recomputed values differ from the V1 table; see the discrepancy table before trusting the original numbers.",
            "Both original protocols score inside label-centred zones; neither measures full-area false positives.",
            "These numbers are kept separate from the new full-area pseudo-discovery score.")
    if group == "full_area_coverage":
        t = R.artifact_table
        return _block(
            "Full-area status: " + ", ".join(f"{r.unit} {r.status}" for r in t.itertuples()) + ". Reproduction vs V1 inside V1 "
            "ROIs: " + ", ".join(f"{r.unit} {r.reproduction_status} (max diff {_f(r.reproduction_max_abs_diff, 3)})"
                                 for r in t.itertuples()) + ".",
            "Where REPRODUCED, the checkpoint, normalizer and feature bank reproduce V1 exactly up to GPU noise, so the "
            "new full-area rasters extend the same model rather than a different one.",
            "Reproduction is only checkable where a V1 raster exists.",
            "A DIFFERS status blocks promotion (provenance gate).")
    if group == "full_area_dti":
        return _block(
            f"Pooled discovery DTI: narrow V1 {_f(s.get('pooled_WITHHELD_SYSTEM_NARROW_V1'), 5)}, narrow on re-inferred rasters "
            f"{_f(s.get('pooled_DISCOVERY_NARROW_REINFERRED'), 5)}, full-area (training-visible exclusion) "
            f"{_f(s.get('pooled_DISCOVERY_FULL_AREA'), 5)}, full-area sensitivity (all non-held excluded) "
            f"{_f(s.get('pooled_DISCOVERY_FULL_AREA_sensitivity'), 5)}. FP share of the full-area denominator "
            f"{_pct(s.get('full_area_fp_share'))}. User-reported leaderboard: {ref['leaderboard_dti']}.",
            ("Moving from label-centred to full-area accounting lowers the discovery score substantially, mainly through "
             "false-positive mass outside the narrow zones." if (s.get("full_to_narrow_ratio") or 1) < 0.7 else
             "Full-area accounting changes the discovery score only modestly."),
            "The full-area score counts real but unmapped faults as false positives; it is a pseudo-discovery proxy, not "
            "an unbiased estimate of the private leaderboard.",
            "Recovery decisions use the full-area protocol, never the narrow one.")
    if group == "error_atlas":
        return _block(
            f"Regions: {s.get('n_false_positive', 0)} high-confidence FP regions (top-ranked), {s.get('n_missed', 0)} missed and "
            f"{s.get('n_correct', 0)} correctly supported held components. FP mass within 1 km of training-visible traces: "
            f"{_pct(s.get('fp_share_within_1km_visible'))}; beyond 3 km: {_pct(s.get('fp_share_beyond_3km_visible'))}.",
            "FP mass close to visible traces indicates rediscovery-like behaviour; FP mass far from all traces indicates "
            "either spurious structure or unmapped faults.",
            "Unmapped faults cannot be distinguished from spurious detections without new truth.",
            "Section 11 checks which label-free evidence the FP and missed regions carry.")
    if group == "expert_complementarity":
        t = R.summary
        return _block(
            "; ".join(f"{r.unit}: {r.verdict}, Spearman(e0,e1) {_f(r.spearman_e0_e1, 3)}, top-1% Jaccard "
                      f"{_f(r.jaccard_top_e0_e1, 3)}, median strength {_f(r.strength_median, 3)}, router-vs-prior Spearman "
                      f"{_f(r.router_vs_label_free_prior_spearman, 3)}" for r in t.itertuples()) + ".",
            "REDUNDANT_OR_SCALE_ONLY or WEAK_MIXING means the mixture adds little beyond the global head; DIVERGENT_HEADS "
            "is necessary but not sufficient for complementary specialisation.",
            "Labels are descriptive thresholds fixed in config; a textured router map is not used as evidence.",
            "If the specialists are redundant, inference-only recovery cannot exploit them; separating them would need a new "
            "training milestone.")
    if group == "geological_disagreement":
        ff = R.summary
        top = ff.sort_values("share", ascending=False).groupby("kind").head(2) if len(ff) else ff
        return _block(
            "Most frequent label-free evidence patterns: " + "; ".join(f"{r.kind}: {r.flag} ({_pct(r.share)})"
                                                                       for r in top.itertuples()) + ".",
            "Patterns are 'consistent with' statements: a strong geophysical edge is not a confirmed fault.",
            "Missing Atlas layers and absent DEM coverage are reported as missing evidence, not weak evidence.",
            "Only label-free evidence may enter a postprocessing policy (policy 4).")
    if group == "structural_strata":
        lc = R.table
        pl = lc[lc.unit == "pooled"] if len(lc) else lc
        return _block(
            "Pooled weighted recall by length class: " + ", ".join(f"{r.length_class} {_f(r.recall_w, 3)} (n={r.n_components})"
                                                                for r in pl.itertuples())
            + f". Median strike difference at supported truth {_f(s.get('median_strike_diff_deg'), 3)} deg "
              f"({_pct(s.get('share_within_20deg'))} within 20 deg).",
            "Recall concentrated on long traces suggests the model detects continuous regional structures better than short "
            "fragments.",
            "Length classes come from 8-connected raster fragments; block denominators are coverage-aware.",
            "Short or poorly aligned faults are not fixable by a monotone score transform.")
    if group == "candidate_lineaments":
        return _block(
            f"{s.get('n_candidates', 0)} Atlas proposals (brief context: {ref['n_lineaments']}); {s.get('n_v1_filter_pass', 0)} pass "
            f"the V1 review filter; {s.get('n_geology_and_model_high', 0)} have high label-free support and model rank >= 0.99; "
            f"{s.get('n_geology_high_model_low', 0)} high support but low model rank. Spearman(support, model) "
            f"{_f(s.get('spearman_support_vs_model'), 3)}.",
            "High-support candidates the model ignores are where label-free geology and the model disagree most.",
            "All proposals are unverified; nothing here is a label.",
            "Candidates are a review layer only; they are not added to any submission.")
    if group == "transform_tradeoffs":
        return _block("Pooled discovery DTI per policy (all folds, descriptive only): " + ", ".join(
            f"{k} {_f(v, 5)}" for k, v in s.get("pooled_by_policy", {}).items()) + ".",
            "Differences between monotone transforms only redistribute confidence; they cannot create detections.",
            "Selection is cross-fitted in section 15; these all-fold numbers are never used to choose.",
            "-")
    if group == "crossfit_uncertainty":
        g = R.gates
        return _block(
            f"Cross-fitted pooled DTI {_f(s['crossfit_pooled_dti'], 5)} vs identity {_f(s['identity_pooled_dti'], 5)} "
            f"(relative {_f(s['relative_gain'], 3)}); bootstrap CI [{_f(s['ci_low'], 3)}, {_f(s['ci_high'], 3)}]; "
            f"failed gates: {', '.join(g[~g.passed].gate) or 'none'}.",
            f"Decision: {s['decision']}.",
            "Three folds give limited power; the block bootstrap ignores between-fold model variance.",
            "Without promotion, the V1 submission stays as it is; no fallback is offered.")
    return _block("-", "-", "-", "-")


def to_markdown(group: str, b: dict) -> str:
    return (f"**{group.replace('_', ' ').title()}**\n\n- **Observation.** {b['observation']}\n"
            f"- **Interpretation.** {b['interpretation']}\n- **Limitation.** {b['limitation']}\n"
            f"- **Modeling implication.** {b['implication']}\n")


def build(run) -> dict:
    P = run.paths
    d = P.d("09_reports")
    for g in GROUP_STAGE:
        if g not in run.interpretations and GROUP_STAGE[g] in run.results:
            run.interpretations[g] = interpret(g, run)
    res = run.results
    rows = []

    def put(ctx, name, val, note=""):
        rows.append({"context": ctx, "quantity": name, "value": val, "note": note})

    ref = run.cfg["reference_values"]
    put("leaderboard", "V1 leaderboard DTI (user-reported)", ref["leaderboard_dti"], "official, not recomputable")
    o = res.get("recompute_original_metrics")
    if o:
        put("A geographic proxy", "pooled DTI (recomputed)", o.stats.get("pooled_GEOGRAPHIC"), "offline proxy")
        put("B narrow withheld-system proxy", "pooled DTI (recomputed)", o.stats.get("pooled_WITHHELD_SYSTEM"), "offline proxy")
    f = res.get("score_full_area_holdouts")
    if f:
        put("C full-area pseudo-discovery", "pooled DTI (training-visible exclusion)", f.stats.get("pooled_DISCOVERY_FULL_AREA"),
            "offline proxy; unmapped faults count as FP")
        put("C full-area pseudo-discovery", "pooled DTI (all non-held excluded)",
            f.stats.get("pooled_DISCOVERY_FULL_AREA_sensitivity"), "sensitivity")
    x = res.get("crossfit_recovery_decision")
    put("decision", "recovery decision", x.stats["decision"] if x else "not run", "")
    sub = res.get("export_guarded_submission")
    put("submission", "export status", sub.stats.get("status") if sub else "export disabled / not run", "")
    summary = pd.DataFrame(rows)
    lines = [f"# GeoRegime V2 recovery - executive summary\n", f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}; "
             f"code commit {run.cfg.get('code_commit')}; config hash {run.cfg_hash}; code hash {run.code_hash}.\n",
             "All DTI values are **offline proxies**; only the leaderboard value is official.\n",
             "| context | quantity | value | note |", "|---|---|---|---|"]
    lines += [f"| {r.context} | {r.quantity} | {_f(r.value, 6) if isinstance(r.value, (int, float)) else r.value} | {r.note} |"
              for r in summary.itertuples()]
    lines.append("\nNo model was trained or fine-tuned. Original V1 and Atlas files were read-only.\n")
    atomic_write_text(d / "executive_summary.md", "\n".join(lines))
    sf = ["# Scientific findings (generated from computed statistics)\n"]
    sf += [to_markdown(g, b) for g, b in run.interpretations.items()]
    atomic_write_text(d / "scientific_findings.md", "\n".join(sf))
    lim = ["# Limitations\n",
           "- The full-area pseudo-discovery score treats real but unmapped faults as false positives; it is not an unbiased "
           "estimate of the private leaderboard score.",
           "- Discovery splits are 8-connected raster components balanced by pixel count (V1), not verified geological systems.",
           "- `exclude_supports_tp = False` follows an organizer statement; it is not verified against a public scorer.",
           "- Three discovery folds give limited statistical power; the block bootstrap ignores model-refit variance.",
           "- Mapped-fault distances are diagnostics only and never enter predictions or postprocessing."]
    for k, r in res.items():
        for l_ in r.limitations:
            lim.append(f"- [{k}] {l_}")
    atomic_write_text(d / "limitations.md", "\n".join(lim) + "\n")
    inv = run.artifact_inventory()
    atomic_write_json(d / "artifact_manifest.json", {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config_hash": run.cfg_hash,
        "code_hash": run.code_hash, "code_commit": run.cfg.get("code_commit"),
        "source_fingerprint": getattr(run, "source_fp", None), "stages": {k: v.summary_dict() for k, v in res.items()},
        "figures": {k: [str(p) for p in v] for k, v in run.figures.items()},
        "files": inv.to_dict("records")})
    long = []
    for k, r in res.items():
        for tn, t in r.tables.items():
            if not isinstance(t, pd.DataFrame) or t.empty or len(t) > 200_000:
                continue
            for i, rec in enumerate(t.to_dict("records")):
                for col, v in rec.items():
                    num = float(v) if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) else np.nan
                    long.append((k, tn, i, str(col), str(v), num))
    allt = pd.DataFrame(long, columns=["stage", "table", "row", "column", "value_str", "value_num"])
    if run.cfg["save_parquet"]:
        atomic_write_parquet(P.d("08_tables") / "all_diagnostics.parquet", allt)
    regs = []
    for k, tn in (("spatial_error_atlas", "region_table"), ("structural_stratification", "regional"),
                  ("crossfit_recovery_decision", "regional")):
        r = res.get(k)
        if r is not None and tn in r.tables and len(r.tables[tn]):
            regs.append(r.tables[tn].assign(source=f"{k}.{tn}"))
    if regs:
        atomic_write_csv(P.d("08_tables") / "regional_summaries.csv", pd.concat(regs, ignore_index=True))
    return {"summary": summary, "inventory": inv}
