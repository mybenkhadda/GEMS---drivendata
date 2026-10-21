"""Calibration, ensemble, scoring and mask integrity audit (Phase 3A).

Two kinds of rows end up in ``calibration_audit.csv``:

* ``CODE_AUDIT`` — findings established by reading the Phase-2 code (file:line evidence). They
  are facts about the implementation, not experimental results.
* data checks — computed from saved artifacts (OOF maps, Phase-2 tables, folds, full maps).

Status vocabulary: PASS (no problem), INFO (documented property), WARN (can bias comparisons),
FAIL (invalidates a reported comparison or number), FIXED (corrected in the code base).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import ndimage

from ..ensemble import read_map
from .context import PHASE2_RUNS, P3Context

CODE_AUDIT = [
    ("C01", "calibration", "PASS",
     "Output transforms are fitted on all valid pixels of the *other* folds and scored on the held-out fold; "
     "a fold's labels never select its own transform.", "metric.py:291-334 (transform_component_table, cross_fitted_transform)"),
    ("C02", "calibration", "FAIL",
     "Transform *family* differs by model type: rank_mean and B0 baselines were given RANK_TRANSFORM_GRID "
     "(floor 0.8-0.99, scale up to 2 -> top-q pixels saturate to exactly 1, a binary mask) while GBMs only got the "
     "probability grid (floor <= 0.4). The 0.1125 vs 0.0815 comparison confounds ensembling with transform family.",
     "ensemble.py:119-122, gbm.py:258,261"),
    ("C03", "calibration", "WARN",
     "Probability-grid optimum for G3 sits on the grid edge (scale=3 max, floor=0.4 max): single-model DTI is "
     "truncated by the search grid.", "outputs/benchmark/experiments/lgbm_G3_s2026/transform_crossfit.csv"),
    ("C04", "ensemble", "FAIL",
     "Weighted ensemble map is built with weights selected on ALL folds (weights_all_folds); only its output "
     "transform is cross-fitted, so held-out labels choose the weights of the reported 0.0815.",
     "ensemble.py:112-115"),
    ("C05", "ensemble", "WARN",
     "Ensemble weights are optimised on raw member maps but the ensemble is scored after a transform: "
     "inconsistent objectives (weights=1.0 on G3 reproduces G3 exactly).", "ensemble.py:71-98"),
    ("C06", "ensemble", "WARN",
     "Member set (top-3 by pooled DTI) and ensemble kind (mean / rank / weighted) are selected on the same folds "
     "that report the score: max-of-several selection optimism, not nested.", "02_fault_benchmark.ipynb section 8"),
    ("C07", "ranking", "PASS",
     "Rank normalisation uses only valid pixels (NaN -> 0 inside valid), invalid pixels never enter the "
     "distribution.", "ensemble.py:rank_normalise"),
    ("C08", "ranking", "WARN",
     "Ordinal ranks break ties by raster order (north to south): a threshold falling inside a tied plateau "
     "keeps its northern part. Material only if large tie blocks exist (see data check D03).", "ensemble.py:rank_normalise"),
    ("C09", "ranking", "WARN",
     "Ranks are global over the stitched OOF map, a mixture of three fold models with different score scales: "
     "label-free (no leakage) but the area each fold receives under a global cut depends on its model's scale.",
     "ensemble.py:110, gbm.py:250"),
    ("C10", "masks", "INFO",
     "Geographic CV scores all mapped faults as truth (no exclusion); consistent with 'reproduce known faults'.",
     "metric.py:dti_by_zone, gbm.py:_finish_cv"),
    ("C11", "masks", "WARN",
     "Pseudo-discovery reported 'excl' (retained faults dilated 300 m excluded) and 'no_exclusion' (retained "
     "faults counted as TRUTH). The hidden-test analogue when known faults are not excluded — withheld faults are "
     "truth and predictions on retained faults count as FP — was missing. Added in discovery3.", "discovery.py:30-38"),
    ("C12", "submission", "WARN",
     "Submission maps are fold-model AVERAGES predicting everywhere, including each model's own training area, "
     "while OOF maps are stitched single-model predictions. The output transform tuned on OOF scale is applied to "
     "a different score distribution, and mapped faults get in-sample scores (see data check D06).",
     "submission.py:31-42, 116-126"),
    ("C13", "discovery", "FIXED",
     "per_sys merged with systems that also has n_px -> n_px_x / n_px_y; sensitivity weights silently fell back "
     "to trace_km.", "discovery.py:evaluate_split (fixed: systems.n_px renamed system_n_px)"),
    ("C14", "discovery", "FAIL",
     "Pseudo-discovery recall@tol and extra_confident_km2 use threshold 0.5 on transformed maps whose scales "
     "differ per model (G1 is >= 0.5 over most of the evaluation zone): high G1 recall is an area effect, not "
     "localisation. Replaced by matched-area budgets (budget.py, discovery3.py).", "discovery.py:43-64"),
    ("C15", "discovery", "INFO",
     "Discovery models train on the whole map; evaluation-zone background pixels were training negatives "
     "(FP optimistic). Rounds = median CV best iteration (G1 7, G2 7, G3 17).", "gbm.py:fit_full_gbm, notebook 02 cell 34"),
    ("C16", "cnn", "WARN",
     "CNN CV loss weight = inner-train zone; no ignore band around thin labels (GBM sampling has one). "
     "torch.cuda.amp.GradScaler is deprecated in torch>=2.4. Both handled in cnn3.", "cnn.py:305-352, 383-385"),
    ("C17", "lgbm", "INFO",
     "Early stopping watches [binary_logloss, auc] without first_metric_only: it stops when either stalls "
     "for 100 rounds; inner-val logloss rises after ~10-25 rounds while inner-val AUC is flat from round 1.",
     "gbm.py:37-56, lgbm_*/history.csv"),
    ("C18", "lgbm", "FIXED",
     "LightGBM rejected feature names with ':' '@' '.' (notebook monkey-patch). Sanitised inside train_lgbm.",
     "gbm.py:lgbm_safe_names"),
]


def code_audit_table() -> pd.DataFrame:
    return pd.DataFrame(CODE_AUDIT, columns=["check_id", "area", "status", "finding", "evidence"]).assign(kind="code", value="")


# ---------------------------------------------------------------------------
# data checks
# ---------------------------------------------------------------------------
def _row(cid, area, status, finding, value="", evidence=""):
    return {"check_id": cid, "area": area, "status": status, "finding": finding, "value": value,
            "evidence": evidence, "kind": "data"}


def check_invalid(ctx: P3Context, names) -> list[dict]:
    rows = []
    v = ctx.data.valid
    for n in names:
        p = ctx.oof(n)
        nan_in, fin_out = int((~np.isfinite(p) & v).sum()), int((np.isfinite(p) & ~v).sum())
        lo, hi = float(np.nanmin(p)), float(np.nanmax(p))
        st = "PASS" if nan_in == 0 and fin_out == 0 and lo >= 0 and hi <= 1 else "WARN"
        rows.append(_row("D01", "masks", st, f"{n}: NaN inside valid / finite outside valid / range",
                         json.dumps({"nan_inside_valid": nan_in, "finite_outside_valid": fin_out, "min": lo, "max": hi})))
    return rows


def check_ties(ctx: P3Context, names, top_fracs=(0.05, 0.10, 0.20)) -> list[dict]:
    """Share of tied values and how a global ordinal cut splits tied plateaus by position."""
    rows = []
    v = ctx.data.valid
    rr = np.nonzero(v)[0]
    for n in names:
        x = np.nan_to_num(ctx.oof(n)[v], nan=0.0)
        vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        tied_share = float((cnt[inv] > 1).mean())
        detail = {"distinct_values": int(len(vals)), "tied_pixel_share": tied_share, "largest_tie_block": int(cnt.max())}
        order = np.argsort(x, kind="stable")
        worst = 0.0
        for q in top_fracs:
            k = int(np.ceil(q * len(x)))
            thr = x[order[len(x) - k]]
            block = x == thr
            if block.sum() > 1:
                kept = np.zeros(len(x), bool)
                kept[order[len(x) - k:]] = True
                kb, db = kept & block, ~kept & block
                if kb.any() and db.any():
                    worst = max(worst, float(block.sum()) / k)
                    detail[f"top{q:g}_plateau_px"] = int(block.sum())
                    detail[f"top{q:g}_mean_row_kept_vs_dropped"] = [float(rr[kb].mean()), float(rr[db].mean())]
        detail["max_plateau_share_of_cut"] = worst
        st = "FAIL" if worst > 0.05 else ("WARN" if worst > 0.005 else "PASS")
        rows.append(_row("D03", "ranking", st, f"{n}: ties and positional tie-breaking at top-q cuts", json.dumps(detail)))
    return rows


def check_p2_grid_edges(ctx: P3Context) -> list[dict]:
    from ..metric import DEFAULT_TRANSFORM_GRID
    from ..gbm import RANK_TRANSFORM_GRID

    rows = []
    for name, exp in PHASE2_RUNS.items():
        f = ctx.bench.experiments / exp / "transform_crossfit.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        grid = RANK_TRANSFORM_GRID if name.startswith("B0") else DEFAULT_TRANSFORM_GRID
        edges = {k: bool(t[k].isin([min(v), max(v)]).all()) for k, v in grid.items() if len(v) > 2}
        st = "WARN" if any(edges.values()) else "PASS"
        rows.append(_row("D02", "calibration", st, f"{name}: Phase-2 cross-fitted choice on grid edge (per parameter, all folds)",
                         json.dumps({"edges": edges, "choices": t[["fold", "gamma", "scale", "floor"]].to_dict("records")}),
                         f"{exp}/transform_crossfit.csv"))
    return rows


def check_weighted_leak(ctx: P3Context) -> list[dict]:
    wf, cfg = ctx.bench.ensembles / "weights_crossfit.csv", ctx.bench.ensembles / "ens_weighted.json"
    if not (wf.exists() and cfg.exists()):
        return []
    per = pd.read_csv(wf)
    allw = json.loads(cfg.read_text())["weights"]
    diff = [json.loads(w) != allw for w in per.w]
    return [_row("D04", "ensemble", "FAIL" if any(diff) else "WARN",
                 "Weighted ensemble: per-fold cross-fitted weights vs the all-fold weights used to build the scored map",
                 json.dumps({"per_fold": per[["fold", "w", "dti_heldout"]].to_dict("records"), "all_folds": allw}),
                 "ensembles/weights_crossfit.csv, ens_weighted.json")]


def check_fold_masks(ctx: P3Context) -> list[dict]:
    rows = []
    g, f, lab = ctx.data.grid, ctx.folds, ctx.data.labels
    for k in ctx.fold_ids:
        val, tr = f.val(k), f.train(k)
        d = ndimage.distance_transform_edt(~val, sampling=(g.pixel_km_y, g.pixel_km_x))
        min_gap = float(d[tr].min()) if tr.any() else np.nan
        iv = f.inner_val(k)
        st = "PASS" if (not (val & tr).any()) and min_gap >= ctx.cfg.buffer_km - g.pixel_km else "FAIL"
        rows.append(_row("D05", "masks", st, f"fold {k}: val/train disjoint, min train-to-val distance (km), labels in buffer",
                         json.dumps({"overlap_px": int((val & tr).sum()), "min_gap_km": min_gap,
                                     "buffer_label_px": int((f.buffer(k) & lab).sum()),
                                     "inner_val_in_train": bool((iv & ~tr).sum() == 0)})))
    return rows


def check_discovery_masks(ctx: P3Context) -> list[dict]:
    rows = []
    s, lab, v = ctx.splits, ctx.data.labels, ctx.data.valid
    for d in range(s.n_splits):
        wh, ign, ez = s.withheld(d), s.ignore(d), s.eval_zone(d)
        info = {"withheld_px": int(wh.sum()), "withheld_not_label": int((wh & ~lab).sum()),
                "withheld_outside_ignore": int((wh & ~ign).sum()),
                "ignore_outside_eval": int((ign & v & ~ez).sum()),
                "retained_labels_in_eval_zone": int((lab & ~wh & ez & v).sum()),
                "retained_labels_inside_ignore": int((lab & ~wh & ign).sum())}
        st = "PASS" if info["withheld_not_label"] == 0 and info["withheld_outside_ignore"] == 0 else "FAIL"
        if info["retained_labels_inside_ignore"]:
            st = "WARN" if st == "PASS" else st
        rows.append(_row("D07", "discovery", st, f"split {d}: withheld systems are labels, inside the ignore ring; "
                         "retained labels are never ignored (they stay positives)", json.dumps(info)))
    return rows


def check_oof_vs_full(ctx: P3Context, pairs: dict[str, str], area_fracs=(0.05, 0.10, 0.20)) -> list[dict]:
    """OOF (stitched single models) vs full-region submission map (fold-model average)."""
    rows = []
    v, lab, fm = ctx.data.valid, ctx.data.labels, ctx.folds.fold_map
    for name, full_path in pairs.items():
        from pathlib import Path

        if not Path(full_path).exists():
            continue
        oof, full = ctx.oof(name), read_map(full_path)
        qs = [0.5, 0.8, 0.9, 0.95, 0.99]
        info = {"oof_quantiles": np.nanquantile(oof[v], qs).round(4).tolist(),
                "full_quantiles": np.nanquantile(full[v], qs).round(4).tolist(),
                "mean_on_labels_oof": float(np.nanmean(oof[lab & v])), "mean_on_labels_full": float(np.nanmean(full[lab & v])),
                "mean_far_oof": float(np.nanmean(oof[v & ~lab])), "mean_far_full": float(np.nanmean(full[v & ~lab]))}
        for a in area_fracs:          # where does a global top-a cut of the full map put its area, by fold?
            thr = np.nanquantile(full[v], 1 - a)
            info[f"full_top{a:g}_share_by_fold"] = [float(((full >= thr) & (fm == k)).sum() / max((fm == k).sum(), 1))
                                                   for k in ctx.fold_ids]
        ratio = info["mean_on_labels_full"] / max(info["mean_on_labels_oof"], 1e-9)
        st = "WARN" if ratio > 1.2 or abs(info["full_quantiles"][2] - info["oof_quantiles"][2]) > 0.05 else "PASS"
        rows.append(_row("D06", "submission", st, f"{name}: OOF vs full-region submission distribution "
                         "(label-pixel score inflation = in-sample predictions on mapped faults)", json.dumps(info),
                         str(full_path)))
    return rows


def run_data_checks(ctx: P3Context, names, full_pairs: dict[str, str]) -> pd.DataFrame:
    rows = []
    rows += check_invalid(ctx, names)
    rows += check_p2_grid_edges(ctx)
    rows += check_ties(ctx, names)
    rows += check_weighted_leak(ctx)
    rows += check_fold_masks(ctx)
    rows += check_oof_vs_full(ctx, full_pairs)
    rows += check_discovery_masks(ctx)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# which local metric is closest to the competition?
# ---------------------------------------------------------------------------
METRIC_CORRESPONDENCE = pd.DataFrame([
    {"local_metric": "geographic pooled DTI, binary top-a mask, cross-fitted a", "truth": "all mapped faults in held-out folds",
     "known_faults": "truth", "closest_to": "reproducing the USGS map in unseen regions",
     "use": "model ranking / localisation quality; NOT the new-fault leaderboard"},
    {"local_metric": "pseudo-discovery DTI, retained faults counted as FP (discovery3 'retained_fp')",
     "truth": "withheld systems only", "known_faults": "predictions on them are FP",
     "closest_to": "leaderboard IF known faults are not excluded from hidden scoring",
     "use": "primary proxy for new-fault scoring under policy 'model' / 'suppress_known'"},
    {"local_metric": "pseudo-discovery DTI, retained faults excluded (discovery3 'excl')",
     "truth": "withheld systems only", "known_faults": "neither TP nor FP",
     "closest_to": "leaderboard IF the organisers exclude known faults", "use": "secondary proxy"},
    {"local_metric": "pseudo-discovery DTI, all faults truth (Phase-2 'no_exclusion')",
     "truth": "withheld + retained", "known_faults": "truth",
     "closest_to": "nothing in the competition", "use": "sanity only"},
    {"local_metric": "density-adjusted DTI (budget.density_sensitivity)", "truth": "mapped faults thinned to rho x density",
     "known_faults": "-", "closest_to": "operating-point choice when hidden truth is sparser than USGS labels",
     "use": "choose the area budget of leaderboard probes"},
])


def write_calibration_findings(path, audit: pd.DataFrame, comp: pd.DataFrame | None) -> None:
    """calibration_findings.md: audit verdicts + transform comparison + conditional reading rules."""
    lines = ["# Phase 3A — calibration integrity findings", "",
             "_Generated by notebooks/03_calibration_and_fault_discovery.ipynb. Code findings come from reading the "
             "Phase-2 implementation; data checks and DTI values come from the user's run._", "",
             "## Audit verdicts", ""]
    for st in ("FAIL", "WARN", "FIXED", "PASS", "INFO"):
        sub = audit[audit.status == st]
        if len(sub):
            lines.append(f"**{st}** ({len(sub)})")
            lines += [f"- `{r.check_id}` [{r.area}] {r.finding}" for r in sub.itertuples()]
            lines.append("")
    if comp is not None and len(comp):
        lines += ["## Transform families (cross-fitted, pooled held-out DTI)", ""]
        piv = comp.pivot_table(index="family", columns="model", values="dti_crossfit")
        lines += [piv.round(4).to_markdown(), ""]
        best_single = comp[~comp.model.str.startswith("ens")].groupby("model").dti_crossfit.max()
        lines += ["## Reading rules", "",
                  "- If `hard_threshold` >= every soft family for single models, the DTI-optimal output is binary (as the "
                  "vertex argument in curves.py predicts) and probability calibration should not be tuned further.",
                  "- If `rank_power_p2` on a single model (e.g. G3) is within 0.01 of the Phase-2 rank ensemble 0.1125, the "
                  "Phase-2 ensemble gain was a transform-family artifact (C02), not ensembling.",
                  "- `isotonic` (calibrated probabilities) is expected to score far below threshold families: calibration "
                  "and DTI optimality are different goals.",
                  "- `optimism` = in-sample minus cross-fitted DTI; > 0.01 means the family over-fits its own grid.", "",
                  f"Best cross-fitted single-model DTI per model: {best_single.round(4).to_dict()}"]
    from pathlib import Path

    Path(path).write_text("\n".join(lines), encoding="utf-8")
