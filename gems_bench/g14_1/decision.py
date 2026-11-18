"""Scientific decision framework (Phase U): hypotheses -> evidence -> outcomes A-G.

Inputs are the tables / JSON written by the stages (``pipeline.G141Pipeline.collect``). Each hypothesis gets one
status from explicit rules: SUPPORTED, CONTRADICTED, INCONCLUSIVE or NOT_EVALUATED (evidence missing). No status is
inferred from missing evidence and no numeric confidence is invented: the "uncertainty" field states what is missing
or what would change the verdict. Several outcomes can apply at once; Outcome F (more context / transformers) is
only proposed when a context architecture beat the corrected local CNN under identical conditions.

UNTESTED: written without execution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = {
    "A": ("Export problem", "fix geometry / rendering before any retraining", "low", "No"),
    "B": ("Calibration problem", "replace the operating point (representation, area, soft calibration)", "low", "No"),
    "C": ("Known-context shortcut", "conditional discovery targets or geophysics-only inputs", "medium", "Yes"),
    "D": ("Validation mismatch", "full-region discovery validation for selection and calibration", "low-medium", "Yes (selection)"),
    "E": ("Geophysical generalisation failure", "features, supervision, augmentation, stratification", "high", "Yes"),
    "F": ("Insufficient spatial context", "regional CNN, then transformer experiments", "high", "Yes"),
    "G": ("Complementary weak model", "keep G14 as residual / ensemble member", "low", "No"),
}


def _get(R: dict, *path, default=None):
    cur = R
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _row(h, outcome, status, support, contra, metrics, regimes, uncertainty):
    name, action, cost, retrain = OUTCOMES[outcome]
    return {"hypothesis": h, "outcome": f"{outcome}: {name}", "status": status, "supporting_evidence": support,
            "contradictory_evidence": contra, "relevant_metrics": metrics, "affected_regimes": regimes,
            "uncertainty": uncertainty, "recommended_action": action, "implementation_cost": cost,
            "retraining_required": retrain}


def evaluate(R: dict) -> pd.DataFrame:
    rows = []

    # A: export / geometry
    par = _get(R, "parity")
    if par:
        issues = []
        if par.get("same_grid") != "SAME_GRID":
            issues.append("grid differs from template")
        if par.get("identical") is False:
            issues.append(f"exported values differ from the rendered map (max diff {par.get('max_abs_diff')})")
        if abs(par.get("shift_dy_px", 0)) > 0.25 or abs(par.get("shift_dx_px", 0)) > 0.25:
            issues.append(f"shift ({par.get('shift_dy_px'):.2f}, {par.get('shift_dx_px'):.2f}) px")
        if par.get("best_d4_transform", "identity") != "identity":
            issues.append(f"best D4 match {par['best_d4_transform']}")
        rows.append(_row("H_A: the official file is misaligned or distorted", "A",
                         "SUPPORTED" if issues else "CONTRADICTED", "; ".join(issues),
                         "" if issues else "exported file identical to the rendered map on the template grid",
                         "parity: shift, D4 match, max diff, footprint", "all", "covers export only; the label "
                         "raster registration is checked separately (ridge offset)"))
    else:
        rows.append(_row("H_A: the official file is misaligned or distorted", "A", "NOT_EVALUATED", "", "", "", "",
                         "run R01 with the submitted GeoTIFF and the full map"))

    # B: calibration / representation
    best = _get(R, "sweep", "best_crossfit_dti")
    chosen = _get(R, "sweep", "g14_original_choice_dti")
    auc = _get(R, "ranking", "auc_mean")
    if best is not None and chosen is not None:
        gain = best - chosen
        rows.append(_row("H_B: ranking is useful but the representation / area destroys it", "B",
                         "SUPPORTED" if gain > 0.01 else ("CONTRADICTED" if gain <= 0.002 else "INCONCLUSIVE"),
                         f"best cross-fitted {best:.4f} vs crest_w1@8% {chosen:.4f}" if gain > 0 else "",
                         "" if gain > 0.002 else "no representation beats the original choice out of zone",
                         f"cross-fitted DTI by representation; raw AUC {auc}", _get(R, "sweep", "task", default="geo"),
                         "geo-unit maps only unless discovery maps exist: the discovery optimum may differ"))
    else:
        rows.append(_row("H_B: ranking is useful but the representation / area destroys it", "B", "NOT_EVALUATED",
                         "", "", "", "", "run R03 on held-out maps"))

    # C: shortcut
    ov = _get(R, "leakage", "train_pos_in_context_mean")
    copy_inner = _get(R, "leakage", "copy_kernel_inner_dti_mean")
    cf_drop = _get(R, "counterfactual", "inner_drop_target_removed")
    near = _get(R, "full_map", "top0.08_eligible_share_within_3px_known")
    sup, con = [], []
    if ov is not None:
        (sup if ov > 0.9 else con).append(f"{ov:.2f} of training positives are context pixels")
    if copy_inner is not None:
        (sup if copy_inner > 0.5 else con).append(f"copying the context reaches inner DTI {copy_inner:.3f}")
    if cf_drop is not None:
        (sup if cf_drop > 0.2 else con).append(f"inner DTI falls by {cf_drop:.3f} when inner faults leave the context")
    if near is not None:
        (sup if near > 0.3 else con).append(f"{near:.2f} of the eligible 8 % area is within 300 m of known faults")
    status = "NOT_EVALUATED" if not (sup or con) else ("SUPPORTED" if len(sup) > len(con) else
                                                       "CONTRADICTED" if not sup else "INCONCLUSIVE")
    rows.append(_row("H_C: the network copies known-fault context instead of learning geophysics", "C", status,
                     "; ".join(sup), "; ".join(con), "target/context overlap, copy baseline, counterfactuals, "
                     "mass near known faults", "all", "a counterfactual drop alone does not prove leakage "
                     "(distribution shift); the copy baseline and overlap are direct"))

    # D: validation mismatch
    es, ir = _get(R, "inner", "enriched_surrogate"), _get(R, "inner", "full_region_dti")
    area_geo, area_disc = _get(R, "area_optimum", "geo"), _get(R, "area_optimum", "disc_official")
    sup, con = [], []
    if es is not None and ir is not None:
        (sup if es - ir > 0.2 else con).append(f"enriched surrogate {es:.3f} vs region inner DTI {ir:.3f}")
    if area_geo is not None and area_disc is not None:
        (sup if area_geo >= 2 * area_disc else con).append(f"area optimum geo {area_geo} vs discovery {area_disc}")
    rho = _get(R, "ranking_consistency", "spearman_inner_vs_disc")
    if rho is not None:
        (sup if rho < 0.5 else con).append(f"Spearman(inner, discovery) = {rho:.2f}")
    status = "NOT_EVALUATED" if not (sup or con) else ("SUPPORTED" if len(sup) >= len(con) and sup else
                                                       "CONTRADICTED" if not sup else "INCONCLUSIVE")
    rows.append(_row("H_D: inner / geographic validation does not predict discovery performance", "D", status,
                     "; ".join(sup), "; ".join(con), "patch vs region DTI, area optimum by task, rank correlation",
                     "all", "rank correlation needs >= 3 trained models"))

    # E: geophysical generalisation
    m1, champ = _get(R, "models", "M1_geophysics_only", "disc_dti"), _get(R, "champion", "disc_dti")
    if m1 is not None and champ is not None:
        rows.append(_row("H_E: geophysical inputs alone do not generalise to new fault systems", "E",
                         "SUPPORTED" if m1 < 0.5 * champ else ("CONTRADICTED" if m1 >= champ else "INCONCLUSIVE"),
                         f"M1 discovery {m1:.4f} vs champion {champ:.4f}", "", "discovery DTI, feature ablations, "
                         "regime breakdown", str(_get(R, "regimes", "weak", default="")),
                         "depends on the feature store the model consumed (provenance table)"))
    else:
        rows.append(_row("H_E: geophysical inputs alone do not generalise to new fault systems", "E", "NOT_EVALUATED",
                         "", "", "", "", "needs R09 (M1) and a discovery-scored champion"))

    # F: spatial context
    ctx = _get(R, "architecture", "best_context_minus_local")
    ci = _get(R, "architecture", "ci_low")
    if ctx is not None:
        rows.append(_row("H_F: a larger spatial context is what is missing", "F",
                         "SUPPORTED" if ctx > 0 and ci is not None and ci > 0 else
                         ("CONTRADICTED" if ctx <= 0 else "INCONCLUSIVE"),
                         f"best context architecture - corrected local CNN = {ctx:+.4f} (CI low {ci})", "",
                         "paired bootstrap on identical units", "", "only valid under identical features, "
                         "supervision, folds and masks"))
    else:
        rows.append(_row("H_F: a larger spatial context is what is missing", "F", "NOT_EVALUATED", "", "", "", "",
                         "needs R13-R15; never assumed"))

    # G: complementarity
    d = _get(R, "blend", "delta_vs_champion")
    if d is not None:
        rows.append(_row("H_G: G14 adds information the champion lacks", "G",
                         "SUPPORTED" if d > 0.002 else ("CONTRADICTED" if d <= 0 else "INCONCLUSIVE"),
                         f"cross-fitted blend - champion = {d:+.4f}", "", "cross-fitted blend sweep, unique recovery",
                         "", "geo units unless G14 discovery maps exist"))
    else:
        rows.append(_row("H_G: G14 adds information the champion lacks", "G", "NOT_EVALUATED", "", "", "", "",
                         "needs R05 with champion maps on the same units"))
    return pd.DataFrame(rows)


def outcomes(table: pd.DataFrame) -> list[str]:
    return [o.split(":")[0] for o, s in zip(table.outcome, table.status) if s == "SUPPORTED"]


def write_report(table: pd.DataFrame, R: dict, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# G14.1 decision report", "",
             "Generated from executed stage outputs only. Offline numbers are proxies on held-out units; no "
             "leaderboard number is produced or implied.", "",
             f"Applicable outcomes: {', '.join(outcomes(table)) or 'none established yet'}", ""]
    for _, r in table.iterrows():
        lines += [f"## {r.hypothesis}", "", f"* outcome: {r.outcome}", f"* status: **{r.status}**",
                  f"* supporting evidence: {r.supporting_evidence or '-'}",
                  f"* contradictory evidence: {r.contradictory_evidence or '-'}",
                  f"* metrics: {r.relevant_metrics or '-'}", f"* regimes: {r.affected_regimes or '-'}",
                  f"* uncertainty: {r.uncertainty or '-'}", f"* action: {r.recommended_action} "
                  f"(cost {r.implementation_cost}; retraining: {r.retraining_required})", ""]
    out.write_text("\n".join(lines))
    table.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".evidence.json").write_text(json.dumps(R, indent=1, default=str))
    return out
