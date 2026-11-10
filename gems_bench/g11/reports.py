"""Final execution report (11_reports/): G11_FINAL_EXECUTION_REPORT.{md,html} + G11_EXPERIMENT_LEADERBOARD.csv.

Everything is read from stage manifests and saved tables; a quantity that was not produced is reported as
"NOT RUN" (never estimated). DTI values are the offline proxy unless an installed official evaluator was used.
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import atomic_write_text, read_json


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else 0


def stage_table(P) -> pd.DataFrame:
    rows = []
    for f in sorted((P.ctx.root / "manifests").glob("*.json")):
        r = read_json(f) or {}
        rows.append({"stage": r.get("stage", f.stem), "status": r.get("status"), "started": r.get("started_utc"),
                     "completed": r.get("completed_utc"), "outputs": len(r.get("outputs", [])),
                     "error": r.get("error", "")})
    return pd.DataFrame(rows, columns=["stage", "status", "started", "completed", "outputs", "error"])


def summaries(P, plan=None, runtime_info=None) -> dict:
    tr = pd.DataFrame(P.tracker.records)
    st = stage_table(P)
    ev = P.ctx.d("evaluate")
    lb = pd.read_csv(ev / "leaderboard.csv") if (ev / "leaderboard.csv").is_file() else pd.DataFrame()
    lbf = pd.read_csv(ev / "leaderboard_final.csv") if (ev / "leaderboard_final.csv").is_file() else pd.DataFrame()
    sel = read_json(ev / "selection.json") or {}
    cov = read_json(P.ctx.d("ingest") / "coverage_report.json") or {}
    feats = read_json(P.ctx.root / "manifests" / "features.json") or {}
    sub = read_json(P.ctx.d("submit") / "submission_record.json") or {}
    inf = read_json(P.ctx.d("infer") / "inference_manifest.json") or {}
    gpu_s = float(tr[tr.peak_gpu_gb.notna()].seconds.sum()) if len(tr) and "peak_gpu_gb" in tr else 0.0
    exec_s = {
        "completed_stages": st[st.status == "COMPLETE"].stage.tolist(),
        "failed_stages": st[st.status == "FAILED"].stage.tolist(),
        "incomplete_stages": st[st.status == "RUNNING"].stage.tolist(),
        "skipped_steps": [s for s in (plan.table().query("~enabled").step.tolist() if plan else [])],
        "total_runtime_h": round(float(tr.seconds.sum()) / 3600, 3) if len(tr) else None,
        "gpu_hours": round(gpu_s / 3600, 3),
        "peak_ram_gb": float(tr.peak_ram_gb.max()) if len(tr) and tr.peak_ram_gb.notna().any() else None,
        "peak_gpu_gb": float(tr.peak_gpu_gb.max()) if len(tr) and "peak_gpu_gb" in tr and tr.peak_gpu_gb.notna().any()
        else None,
        "total_output_gb": round(_dir_size(P.ctx.root) / 1024 ** 3, 3),
        "runtime": runtime_info or {}}
    data_s = {"geophysics_valid_km2": cov.get("geophysics_valid_km2"), "dem_covered_km2": cov.get("dem_covered_km2"),
              "dem_tiles": cov.get("dem_tiles"), "usable_dem_tiles": cov.get("dem_tiles_usable"),
              "fault_fraction_with_usable_dem": cov.get("fault_fraction_with_usable_dem"),
              "fault_coverage_lost_to_unit_exclusion": cov.get("fault_coverage_lost_to_unit_exclusion"),
              "feature_count": (feats.get("stats") or {}).get("n_predictors"),
              "training_samples": {r.experiment_id: r.n_samples for r in lb.itertuples()} if len(lb) else {}}
    best = sel.get("selected_experiment")
    model_s = {"trained_models": lb.experiment_id.unique().tolist() if len(lb) else [], "best_validation_model": best,
               "best_validation_pr_auc": sel.get("validation_pr_auc")}
    if len(lb):
        v = lb[lb.split == "validation"].set_index("experiment_id")
        if "E5" in v.index:
            singles = v.drop(index="E5")
            model_s["fusion_improvement_pr_auc"] = float(v.loc["E5", "PR_AUC"] - singles.PR_AUC.max()) if len(singles) else None
        if best in v.index:
            for k in ("candidate_recall", "predicted_length_km", "offline_DTI_proxy"):
                model_s[f"validation_{k}"] = v.loc[best, k]
    if len(lbf) and best:
        t = lbf[(lbf.split == "test") & (lbf.experiment_id == best)]
        if len(t):
            model_s.update(heldout_test_pr_auc=float(t.PR_AUC.iloc[0]), heldout_test_offline_DTI_proxy=float(
                t.offline_DTI_proxy.iloc[0]), heldout_test_candidate_recall=t.candidate_recall.iloc[0])
    model_s["candidate_budget_km"] = inf.get("budget_km")
    sub_s = {"output_path": sub.get("path"), "geotiff_validation": (sub.get("validation") or {}).get("passed"),
             "official_schema_compatible": (sub.get("validation") or {}).get("passed"),
             "official_DTI": (sub.get("official_evaluation") or {}).get("official_DTI", "NOT RUN" if not sub else None),
             "official_evaluator_status": (sub.get("official_evaluation") or {}).get("status"),
             "uploaded": sub.get("uploaded", False), "smoke_test": sub.get("smoke_test")}
    return {"execution": exec_s, "data": data_s, "models": model_s, "submission": sub_s, "stages": st, "tracker": tr,
            "leaderboard": pd.concat([lb, lbf], ignore_index=True) if len(lbf) else lb}


def _md_kv(d: dict) -> str:
    def f(v):
        if isinstance(v, float):
            return f"{v:.4g}" if np.isfinite(v) else "n/a"
        return "NOT RUN" if v is None else str(v)

    return "\n".join(f"- **{k}**: {f(v)}" for k, v in d.items() if k != "runtime")


def build_final_report(P, plan=None, runtime_info=None) -> dict:
    s = summaries(P, plan, runtime_info)
    d = P.ctx.d("reports")
    lb_path = atomic_write_text(d / "G11_EXPERIMENT_LEADERBOARD.csv", s["leaderboard"].to_csv(index=False))
    md = ["# G11 final execution report", "",
          f"Config hash `{P.ctx.cfg_hash}` · code commit `{P.ctx.commit}` · output `{P.ctx.root}`", "",
          "All DTI values are the repository OFFLINE DTI proxy (`gems_bench.metric`), not the official leaderboard DTI.",
          "", "## Execution summary", "", _md_kv(s["execution"]), "", "## Data summary", "", _md_kv(s["data"]), "",
          "## Model summary", "", _md_kv(s["models"]), "", "## Submission summary", "", _md_kv(s["submission"]), "",
          "## Stages", ""]
    from .pipeline import md_table

    md.append(md_table(s["stages"]) if len(s["stages"]) else "no stage manifest")
    md += ["", "## Leaderboard (validation; test rows only after the validation-based selection)", ""]
    lb = s["leaderboard"]
    cols = [c for c in ("experiment_id", "split", "model", "PR_AUC", "PR_AUC_ci_low", "PR_AUC_ci_high", "ROC_AUC",
                        "calibration_error", "candidate_recall", "predicted_length_km", "fragmentation",
                        "offline_DTI_proxy", "training_time_s", "peak_GPU_memory_gb", "peak_RAM_gb") if c in lb]
    md.append(md_table(lb[cols]) if len(lb) else "no evaluated experiment")
    text = "\n".join(md) + "\n"
    mdp = atomic_write_text(d / "G11_FINAL_EXECUTION_REPORT.md", text)
    body = "".join(f"<p>{html.escape(line)}</p>" if not line.startswith(("|", "#", "-")) else
                   (f"<h2>{html.escape(line.lstrip('# '))}</h2>" if line.startswith("#") else
                    f"<div>{html.escape(line)}</div>") for line in md if line and not line.startswith("|"))
    tables = "".join(f"<h3>{n}</h3>" + t.to_html(index=False, float_format=lambda x: f"{x:.4g}")
                     for n, t in (("Stages", s["stages"]), ("Leaderboard", lb[cols] if len(lb) else lb)) if len(t))
    htmlp = atomic_write_text(d / "G11_FINAL_EXECUTION_REPORT.html", "<html><head><meta charset='utf-8'>"
                              "<title>G11 final execution report</title><style>body{font-family:system-ui,sans-serif;"
                              "max-width:1200px;margin:auto;padding:16px}table{border-collapse:collapse;font-size:12px}"
                              "td,th{border:1px solid #ccc;padding:3px 6px}@media(prefers-color-scheme:dark){body{"
                              "background:#151515;color:#e6e6e6}td,th{border-color:#444}}</style></head><body>"
                              + body + tables + "</body></html>")
    return {**{k: v for k, v in s.items() if k not in ("stages", "tracker", "leaderboard")},
            "paths": {"md": str(mdp), "html": str(htmlp), "leaderboard": str(lb_path)}}
