"""Submission-factory tables, figures and reports, built ONLY from the persisted candidate manifests
(``manifests/candidates/<id>.json``): a report never depends on notebook variables.

Numbers are labelled by protocol: ``single_policy`` (this run, nested, one cell per outer zone chosen on discovery
labels and applied to both tasks: comparable across candidates) vs ``recorded`` historical values (protocol stated;
a ``separate``-protocol geographic value is not comparable). No offline number is a leaderboard score.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import READY

# Categorical slots of the validated default palette (dataviz references/palette.md), fixed order per candidate.
COLORS = {"w2_champion": "#2a78d6", "w2_crest": "#eb6834", "w2_skeleton": "#1baf7a", "b0_b1_fallback": "#eda100"}
TASK_COLORS = {"discovery": "#2a78d6", "geographic": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MANIFEST_COLUMNS = ["candidate", "status", "READY_FOR_UPLOAD", "upload_priority", "recommendation_rank", "file",
                    "path_drive", "recipe_id", "sha256", "size_mb", "shape", "crs", "dtype", "TECHNICAL_VALIDATION",
                    "EXPORT_PARITY", "OFFICIAL_FORMAT", "GEOGRAPHIC_TARGET_MET", "geographic_warning", "disc_dti",
                    "disc_worst", "geo_dti", "geo_worst", "p6g_eligible", "evaluation", "duplicate_of", "source_candidate",
                    "label", "rationale", "first_reason"]


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _md(df):
    from .reporting import _md as md

    return md(df)


def _ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# =============================================================================== tables (pure)
def manifest_rows(recs: list[dict], ranking: list[dict], target: float) -> pd.DataFrame:
    rk = {r["candidate"]: r for r in ranking}
    rows = []
    for r in recs:
        k = rk.get(r["candidate"], {})
        geo = _f(r.get("geo_dti"))
        warn = "" if r.get("GEOGRAPHIC_TARGET_MET") is not False else (
            f"geographic risk: single-policy geo DTI {geo:.6f} < internal target {target} (exportable; not "
            "geographically robust)")
        rows.append({**{c: r.get(c) for c in MANIFEST_COLUMNS}, "upload_priority": k.get("upload_priority"),
                     "recommendation_rank": k.get("recommendation_rank"), "rationale": k.get("rationale"),
                     "geographic_warning": warn,
                     "first_reason": (r.get("reasons") or r.get("warnings") or [""])[0],
                     "shape": json.dumps(r.get("shape")) if r.get("shape") is not None else None})
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def score_rows(recs: list[dict]) -> pd.DataFrame:
    keys = ("disc_dti", "disc_worst", "disc_worst_zone", "geo_dti", "geo_worst", "geo_worst_zone", "disc_tp_w",
            "disc_fp_w", "disc_fn_w", "disc_n_truth", "geo_tp_w", "geo_fp_w", "geo_fn_w", "geo_n_truth",
            "nested_disc_dti", "nested_geo_dti", "p6g_eligible", "evaluation", "evaluation_detail", "positive_px",
            "GEOGRAPHIC_TARGET_MET", "EXPORT_PARITY", "metric_source")
    return pd.DataFrame([{"candidate": r["candidate"], "recipe_id": r.get("recipe_id"), "protocol": "single_policy",
                          **{k: r.get(k) for k in keys}} for r in recs if r.get("disc_dti") is not None])


def per_zone_rows(recs: list[dict]) -> pd.DataFrame:
    out = [pd.DataFrame(r["per_zone"]).assign(candidate=r["candidate"]) for r in recs if r.get("per_zone")]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def comparative_rows(recs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in recs:
        for m in ("disc_dti", "disc_worst", "geo_dti"):
            if r.get(m) is not None:
                rows.append({"candidate": r["candidate"], "kind": "this run (frozen grid, nested)", "protocol":
                             "single_policy", "metric": m, "value": r.get(m), "comparable": True, "note": ""})
        h = r.get("historical") or {}
        for m, v in h.items():
            if isinstance(v, (int, float)):
                sep = h.get("protocol") == "separate" and m.startswith("geo")
                rows.append({"candidate": r["candidate"], "kind": f"recorded ({h.get('source')})",
                             "protocol": h.get("protocol"), "metric": m, "value": v, "comparable": not sep,
                             "note": h.get("note", "") if sep else ""})
    return pd.DataFrame(rows)


# =============================================================================== figures
def figures(ws, recs: list[dict], target: float) -> None:
    from .reporting import _plt

    plt = _plt()
    S = score_rows(recs)
    if len(S):
        D = S[["candidate", "disc_dti", "geo_dti"]].copy()
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        x = np.arange(len(D))
        for i, (col, lab) in enumerate((("disc_dti", "discovery"), ("geo_dti", "geographic"))):
            ax.bar(x + (i - 0.5) * 0.38, D[col].astype(float), 0.36, color=TASK_COLORS[lab], label=lab,
                   edgecolor="white", linewidth=2)
        ax.axhline(target, color=INK2, linewidth=1, linestyle="--")
        ax.text(len(D) - 0.5, target, f" internal geo target {target}", color=INK2, fontsize=8, va="bottom", ha="right")
        ax.set_xticks(x, D.candidate, fontsize=8)
        ax.set_ylabel("official DTI (offline, nested single policy)", fontsize=8, color=INK2)
        ax.set_title("Candidate comparison", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8)
        _ax(ax)
        ws.save_fig(fig, "figures/candidate_comparison.png", D)

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for r in D.itertuples():
            ax.scatter(r.disc_dti, r.geo_dti, s=70, color=COLORS.get(r.candidate, INK2), edgecolor="white",
                       linewidth=2, zorder=3, label=r.candidate)
            ax.annotate(r.candidate, (r.disc_dti, r.geo_dti), textcoords="offset points", xytext=(6, 4), fontsize=8,
                        color=INK)
        ax.axhline(target, color=INK2, linewidth=1, linestyle="--")
        ax.set_xlabel("discovery DTI", fontsize=8, color=INK2)
        ax.set_ylabel("geographic DTI", fontsize=8, color=INK2)
        ax.set_title("Discovery vs geographic trade-off (offline)", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        _ax(ax)
        ws.save_fig(fig, "figures/discovery_geographic_tradeoff.png", D)
    Z = per_zone_rows(recs)
    if len(Z):
        D = Z[["candidate", "task", "zone", "dti"]].copy()
        tasks = list(dict.fromkeys(D.task))
        fig, axs = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.4), sharey=True, squeeze=False)
        cands = list(dict.fromkeys(D.candidate))
        for ax, t in zip(axs[0], tasks):
            d = D[D.task == t]
            zones = sorted(d.zone.unique())
            w = 0.8 / max(len(cands), 1)
            for j, c in enumerate(cands):
                v = [float(d[(d.candidate == c) & (d.zone == z)].dti.iloc[0]) if ((d.candidate == c) & (d.zone == z)
                                                                                  ).any() else np.nan for z in zones]
                ax.bar(np.arange(len(zones)) + (j - (len(cands) - 1) / 2) * w, v, w * 0.92,
                       color=COLORS.get(c, INK2), label=c, edgecolor="white", linewidth=1)
            ax.set_xticks(np.arange(len(zones)), [f"zone {int(z)}" for z in zones], fontsize=8)
            ax.set_title(t, fontsize=9, color=INK, loc="left")
            _ax(ax)
        axs[0][0].set_ylabel("held-out DTI", fontsize=8, color=INK2)
        axs[0][-1].legend(frameon=False, fontsize=7)
        fig.suptitle("Per-zone official DTI", fontsize=10, color=INK, x=0.02, ha="left")
        ws.save_fig(fig, "figures/per_zone_dti.png", D)
    C = pd.concat([pd.DataFrame(r["zones"]).assign(candidate=r["candidate"]) for r in recs if r.get("zones")],
                  ignore_index=True) if any(r.get("zones") for r in recs) else pd.DataFrame()
    if len(C):
        D = C[[c for c in ("candidate", "zone", "eligible_px", "positive_px", "frac_of_eligible",
                           "positive_on_known_px", "fusion_w") if c in C]].copy()
        fig, ax = plt.subplots(figsize=(7, 3.4))
        cands = list(dict.fromkeys(D.candidate))
        zones = sorted(D.zone.unique())
        w = 0.8 / max(len(cands), 1)
        for j, c in enumerate(cands):
            d = D[D.candidate == c].set_index("zone").reindex(zones)
            ax.bar(np.arange(len(zones)) + (j - (len(cands) - 1) / 2) * w, 100 * d.frac_of_eligible.astype(float),
                   w * 0.92, color=COLORS.get(c, INK2), label=c, edgecolor="white", linewidth=1)
        ax.set_xticks(np.arange(len(zones)), [f"geo zone {int(z)}" for z in zones], fontsize=8)
        ax.set_ylabel("predicted area, % of eligible", fontsize=8, color=INK2)
        ax.set_title("Prediction coverage of the exported map", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=7)
        _ax(ax)
        ws.save_fig(fig, "figures/prediction_coverage.png", D)


# =============================================================================== reports
def readme_upload(M: pd.DataFrame, drive_dir: str) -> str:
    ready = M[M.status == READY].sort_values("recommendation_rank")
    lines = ["# Upload instructions (manual only)", "",
             f"Folder: `{drive_dir}submissions/`", ""]
    if not len(ready):
        lines += ["**No file is READY_FOR_UPLOAD.** Do not upload anything; read `reports/FAILURE_DIAGNOSTICS.md`.", ""]
    else:
        p = ready[ready.upload_priority == "PRIMARY"]
        a = ready[ready.upload_priority == "ALTERNATIVE"]
        if len(p):
            r = p.iloc[0]
            lines += ["## Upload this file first", "", f"**`{r.file}`** (candidate `{r.candidate}`, recipe "
                      f"`{r.recipe_id}`)", "", f"* sha256 `{r.sha256}`",
                      f"* offline nested DTI: discovery {_f(r.disc_dti):.6f}, geographic {_f(r.geo_dti):.6f} "
                      "(NOT leaderboard scores)", f"* {r.geographic_warning or 'geographic target met'}", ""]
        if len(a):
            r = a.iloc[0]
            lines += ["## One meaningful alternative (optional, a later slot)", "",
                      f"`{r.file}` (candidate `{r.candidate}`): {r.rationale or ''}", ""]
        lines += ["## Steps", "",
                  "1. Download the file from the Drive folder above.",
                  "2. Check it: `sha256sum <file>` must equal its line in `submission_sha256.txt`.",
                  "3. DrivenData -> GEMS competition -> Submissions -> upload the .tif "
                  "(3 scored submissions per rolling 7 days: never spend two on DUPLICATE files).",
                  "4. Record the public score in the notebook's result-entry cell (`F.record_result`). Never re-select "
                  "or tune on the public score.", ""]
    lines += ["## Every candidate", "", _md(M[["candidate", "status", "upload_priority", "file", "disc_dti", "geo_dti",
                                               "GEOGRAPHIC_TARGET_MET", "first_reason"]]), ""]
    return "\n".join(lines)


def executive_summary(M: pd.DataFrame, stage: dict, meta: dict) -> str:
    ready = M[M.status == READY]
    p = ready[ready.upload_priority == "PRIMARY"]
    lines = ["# Submission factory - executive summary", "",
             f"Run mode `{meta.get('run_mode')}` | commit `{meta.get('commit')}` | code `{meta.get('code_rev')}` | "
             f"written {meta.get('written')}", "",
             f"* READY_FOR_UPLOAD files: **{len(ready)}** of {len(M)} candidates",
             f"* Primary upload: **{p.iloc[0].file if len(p) else 'none'}**",
             "* Offline numbers are nested held-out estimates of the official metric, not leaderboard scores. "
             "READY_FOR_UPLOAD is technical validity; it is not a promise of leaderboard performance.", "",
             "## Stages", "", _md(pd.DataFrame([{"stage": k, "status": v} for k, v in stage.items()])), "",
             "## Candidates", "", _md(M[["candidate", "status", "TECHNICAL_VALIDATION", "EXPORT_PARITY",
                                         "OFFICIAL_FORMAT", "GEOGRAPHIC_TARGET_MET", "disc_dti", "geo_dti",
                                         "upload_priority"]]), ""]
    return "\n".join(lines)


def failure_report(recs: list[dict], errors: pd.DataFrame) -> str:
    lines = ["# Failure diagnostics", ""]
    bad = [r for r in recs if r.get("status") not in (READY, "VALIDATED_NOT_WRITTEN", "DUPLICATE")]
    if not bad and not len(errors):
        lines += ["No candidate failed and no runtime error was recorded.", ""]
    for r in bad:
        lines += [f"## {r['candidate']}: {r.get('status')} (stage {r.get('stage_reached')})", "",
                  *[f"* {x}" for x in (r.get("reasons") or ["(no reason recorded)"])], ""]
    for r in recs:
        if r.get("warnings"):
            lines += [f"## Warnings - {r['candidate']}", "", *[f"* {w}" for w in r["warnings"]], ""]
    if len(errors):
        lines += ["## Runtime errors (diagnostics/runtime_errors.csv)", "",
                  _md(errors[["candidate", "stage", "failure_type", "artifact", "explanation"]]), ""]
    return "\n".join(lines)


def next_action(M: pd.DataFrame) -> str:
    ready = M[M.status == READY]
    p = ready[ready.upload_priority == "PRIMARY"]
    if len(p):
        r = p.iloc[0]
        act = (f"Upload `{r.file}` by hand (see `submissions/README_UPLOAD.md`), then record its public score with "
               "`F.record_result`. Do not start new research before the public baseline exists.")
    elif (M.status == "VALIDATED_NOT_WRITTEN").any():
        act = "Candidates validated without files (DIAGNOSTIC mode or a file cap): re-run with run_mode FULL."
    else:
        act = ("No valid file. Fix the first blocker in `reports/FAILURE_DIAGNOSTICS.md` and re-run (cached stages "
               "are reused).")
    return "\n".join(["# Next action", "", act, ""])


def write_all(ws, recs: list[dict], ranking: list[dict], c: dict, stage: dict, meta: dict) -> None:
    target = float(c["geographic_target"])
    M = manifest_rows(recs, ranking, target)
    ws.save_table(M, "submissions/submission_manifest.csv", parquet=bool(c["save_parquet"]))
    ws.save_json({"run": meta, "files": M.to_dict("records"), "uploaded": False,
                  "note": "READY_FOR_UPLOAD = technical validity on the read-back file; RECOMMENDATION_RANK = the "
                          "predeclared offline ordering; GEOGRAPHIC_TARGET_MET = internal research target only"},
                 "submissions/submission_manifest.json")
    ready = M[M.status == READY]
    ws.save_text("".join(f"{r.sha256}  {r.file}\n" for r in ready.itertuples()), "submissions/submission_sha256.txt")
    ws.save_table(M[["candidate", "status", "file", "disc_dti", "disc_worst", "geo_dti", "geo_worst",
                     "GEOGRAPHIC_TARGET_MET", "p6g_eligible", "duplicate_of"]], "submissions/submission_comparison.csv")
    ws.save_table(pd.DataFrame(ranking), "submissions/submission_ranking.csv")
    ws.save_text(readme_upload(M, meta.get("drive_dir", "")), "submissions/README_UPLOAD.md")
    S = score_rows(recs)
    ws.save_table(S, "metrics/official_scores.csv", parquet=bool(c["save_parquet"]))
    ws.save_table(per_zone_rows(recs), "metrics/per_zone_scores.csv", parquet=bool(c["save_parquet"]))
    ws.save_table(comparative_rows(recs), "metrics/comparative_metrics.csv")
    try:
        figures(ws, recs, target)
    except Exception as e:                               # a figure never decides validity; recorded in the status
        meta = {**meta, "figure_error": f"{type(e).__name__}: {e}"}
    E = ws.load_table("diagnostics/runtime_errors.csv")
    ws.save_text(executive_summary(M, stage, meta), "reports/EXECUTIVE_SUMMARY.md")
    ws.save_text("\n".join(["# Execution status", "", _md(pd.DataFrame([{"stage": k, "status": v}
                                                                        for k, v in stage.items()])), "",
                            "```json", json.dumps(meta, indent=1, default=str), "```", ""]),
                 "reports/EXECUTION_STATUS.md")
    ws.save_text(failure_report(recs, E), "reports/FAILURE_DIAGNOSTICS.md")
    ws.save_text(next_action(M), "reports/NEXT_ACTION.md")
    ws.save_json({**meta, "stages": stage, "candidates": {r["candidate"]: r.get("status") for r in recs},
                  "config": c}, "manifests/run.json")
