"""Reports, figures (PNG + plotting data CSV) and the manual leaderboard log of Milestone 1.

Every report quotes only persisted tables; recorded historical numbers are labelled ``recorded``; public leaderboard
scores are recorded after a MANUAL upload and never feed back into selection (no tuning on the public score).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import BLOCKED, READY

LB_COLUMNS = ["submission_id", "submission_file", "file_sha256", "recipe_id", "candidate", "validation_disc_dti",
              "validation_geo_dti", "public_dti", "submitted_at", "recorded_at", "discrepancy_public_minus_offline",
              "notes"]


def _md(df: pd.DataFrame) -> str:
    if df is None or not len(df):
        return "_(empty)_"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


# =============================================================================== reports
def recovery_report(C: pd.DataFrame, sel: dict) -> str:
    cols = [c for c in ("candidate_id", "role", "status", "eligible", "disc_dti", "disc_worst", "geo_dti",
                        "geo_floor_met", "recorded_disc_dti", "recorded_geo_dti", "decision", "eligibility_reasons")
            if c in C]
    return "\n".join(["# M1 champion recovery", "",
                      f"Status: **{sel['status']}** | champion `{sel.get('champion')}` | decision {sel.get('decision')} "
                      f"| objective `{sel.get('objective')}` | recipe `{sel.get('recipe_id')}`", "",
                      "Numbers: nested single-policy estimates from the frozen boost-A grid (one cell per outer zone, "
                      "chosen on the other zones). `recorded_*` columns are historical records from the configs.", "",
                      _md(C[cols]), "", "## Reasons", "", *(f"* {r}" for r in sel.get("reasons") or ["none"]), "",
                      f"META (honest estimate of choosing among eligible candidates): {sel.get('meta_nested')}", ""])


def validation_report(st: dict, V: pd.DataFrame, RR: pd.DataFrame, Z: pd.DataFrame, par: pd.DataFrame) -> str:
    return "\n".join(["# M1 official validation", "", f"Status: **{st['status']}**", "",
                      *(f"* {r}" for r in st["reasons"] or ["every check passed"]), "",
                      "## Recomputed per-zone official components vs saved grid", "",
                      _md(V[[c for c in ("task", "zone", "rep", "frac", "saved_dti", "recomputed_dti", "status")
                             if c in V]]), "", "## Recorded values vs reproduced", "", _md(RR), "",
                      "## Zone-stitched area budget", "", _md(Z), "", "## Zone-stitched parity (dry build)", "",
                      _md(par[[c for c in ("unit", "check", "status", "spearman", "jaccard") if c in par]]), "",
                      f"Validated array sha256: `{st['validated_array_sha256']}`", ""])


# =============================================================================== figures
def validation_figures(sess, V: pd.DataFrame, RR: pd.DataFrame, Z: pd.DataFrame, par: pd.DataFrame,
                       crow: dict) -> list[str]:
    plt = _plt()
    made = []
    if len(V):
        fig, ax = plt.subplots(figsize=(7, 4))
        d = V.pivot_table(index="zone", columns="task", values="recomputed_dti")
        d.plot.bar(ax=ax)
        ax.set(title=f"Official DTI by held-out zone - {crow['candidate_id']}", ylabel="DTI (nested cell)")
        sess.ws.save_fig(fig, "figures/dti_by_region.png", V[["task", "zone", "rep", "frac", "recomputed_dti"]])
        made.append("dti_by_region")
    if len(RR):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(RR.recorded, RR.reproduced.astype(float))
        lim = [0, float(np.nanmax([RR.recorded.max(), RR.reproduced.astype(float).max()])) * 1.05]
        ax.plot(lim, lim, color="grey", lw=0.8)
        for r in RR.itertuples():
            ax.annotate(f"{r.recorded_id}.{r.metric}", (r.recorded, float(r.reproduced or np.nan)), fontsize=6)
        ax.set(title="Recorded vs reproduced (frozen grids)", xlabel="recorded", ylabel="reproduced")
        sess.ws.save_fig(fig, "figures/recorded_vs_reproduced.png", RR)
        made.append("recorded_vs_reproduced")
    if len(Z):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(Z.zone.astype(str), Z.frac_of_eligible, label="exported share of eligible")
        ax.axhline(float(Z.target_frac.iloc[0]), color="k", ls="--", label="policy frac")
        ax.set(title="Area-budget consistency per geographic zone", xlabel="zone", ylabel="share of eligible px")
        ax.legend()
        sess.ws.save_fig(fig, "figures/area_budget_consistency.png", Z)
        made.append("area_budget_consistency")
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(Z))
        ax.bar(x - 0.2, Z.eligible_px, 0.4, label="eligible px")
        ax.bar(x + 0.2, Z.positive_px, 0.4, label="positive px")
        ax.set(xticks=x, xticklabels=Z.zone.astype(str), title="Prediction coverage per zone", yscale="log")
        ax.legend()
        sess.ws.save_fig(fig, "figures/prediction_coverage.png", Z)
        made.append("prediction_coverage")
    return made


def parity_figure(sess, R: pd.DataFrame) -> None:
    if not len(R) or "spearman" not in R:
        return
    d = R[pd.to_numeric(R.spearman, errors="coerce").notna()].copy()
    if not len(d):
        return
    plt = _plt()
    d["spearman"] = d.spearman.astype(float)
    fig, ax = plt.subplots(figsize=(8, 4))
    lab = d.scope + " | " + d.unit.astype(str)
    ax.barh(lab, d.spearman, color=np.where(d.scope == "champion", "tab:green", "tab:red"))
    ax.axvline(float(sess.cfg["validation"]["parity_min_spearman"]), color="k", ls="--")
    ax.set(title="Score-map parity (Spearman): champion construction vs historical export", xlabel="Spearman")
    sess.ws.save_fig(fig, "figures/score_map_parity.png", d[["scope", "unit", "check", "spearman", "status"]])


# =============================================================================== leaderboard
def validate_entry(e: dict) -> list[str]:
    """Problems of a manual leaderboard entry (empty list = acceptable)."""
    why = []
    for k in ("submission_id", "submission_file", "public_dti", "submitted_at"):
        if e.get(k) in (None, ""):
            why.append(f"{k} is missing")
    p = e.get("public_dti")
    try:
        if p is not None and not 0.0 <= float(p) <= 1.0:
            why.append(f"public_dti {p} outside [0, 1]")
    except (TypeError, ValueError):
        why.append(f"public_dti {p!r} is not a number")
    s = str(e.get("submitted_at") or "")
    if s and not (len(s) >= 10 and s[4] == "-" and s[7] == "-"):
        why.append(f"submitted_at {s!r} is not YYYY-MM-DD[...]")
    return why


def leaderboard_row(e: dict, man: dict, file_sha: str | None) -> tuple[dict | None, list[str]]:
    """Row for one manual submission; refused when the file is not the READY M1 file or its sha changed."""
    why = validate_entry(e)
    if man.get("status") != READY:
        why.append(f"submission_manifest status is {man.get('status')}, not {READY}")
    if e.get("submission_file") and e["submission_file"] != man.get("file"):
        why.append(f"file {e['submission_file']} is not the M1 file {man.get('file')}")
    if file_sha is not None and file_sha != man.get("sha256"):
        why.append("file sha256 differs from the manifest (the file changed after validation)")
    if why:
        return None, why
    off = (man.get("offline") or {}).get("disc_dti")
    return {"submission_id": str(e["submission_id"]), "submission_file": e["submission_file"],
            "file_sha256": man.get("sha256"), "recipe_id": man.get("recipe_id"), "candidate": man.get("candidate"),
            "validation_disc_dti": off, "validation_geo_dti": (man.get("offline") or {}).get("geo_dti"),
            "public_dti": float(e["public_dti"]), "submitted_at": e["submitted_at"],
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "discrepancy_public_minus_offline": float(e["public_dti"]) - float(off) if off is not None else None,
            "notes": e.get("notes", "")}, []


def append_history(H: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Append; a duplicate submission id or an already-scored file is refused (history is append-only)."""
    if len(H) and (H.submission_id.astype(str) == str(row["submission_id"])).any():
        raise ValueError(f"submission id {row['submission_id']} already recorded")
    if len(H) and (H.file_sha256 == row["file_sha256"]).any():
        raise ValueError("this file already has a recorded public score")
    return pd.concat([H, pd.DataFrame([row], columns=LB_COLUMNS)], ignore_index=True) if len(H) else \
        pd.DataFrame([row], columns=LB_COLUMNS)


def record_leaderboard(sess, entry: dict | None) -> dict:
    """Notebook 04: validate + append the manual entry (if any), persist history, review markdown, plot."""
    from ..artifacts.atomic import sha256_file

    H = sess.ws.load_table("leaderboard/leaderboard_history.csv")
    H = H if len(H) else pd.DataFrame(columns=LB_COLUMNS)
    msg = []
    if entry and any(v not in (None, "") for k, v in entry.items() if k != "notes"):
        man = sess.ws.read_json("submissions/submission_manifest.json") or {}
        p = sess.ws.path(f"submissions/{entry.get('submission_file')}")
        sha = sha256_file(p) if entry.get("submission_file") and p.exists() else None
        row, why = leaderboard_row(entry, man, sha)
        if why:
            msg = [f"REJECTED: {w}" for w in why]
        else:
            H = append_history(H, row)
            sess.ws.save_table(H, "leaderboard/leaderboard_history.csv", parquet=True)
            msg = [f"recorded {row['submission_id']}: public {row['public_dti']}"]
    lines = ["# M1 leaderboard review", "",
             "Public scores are a spatial subset of the test area: they are recorded for the history and never used "
             "to select or tune anything.", "", _md(H), "", *(f"* {m}" for m in msg)]
    sess.ws.save_text("\n".join(lines), "leaderboard/leaderboard_review.md")
    if len(H):
        plt = _plt()
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(H))
        ax.bar(x - 0.2, pd.to_numeric(H.validation_disc_dti, errors="coerce"), 0.4, label="offline nested disc DTI")
        ax.bar(x + 0.2, H.public_dti.astype(float), 0.4, label="public DTI")
        ax.set(xticks=x, xticklabels=H.submission_id.astype(str), title="Recorded submissions")
        ax.legend()
        sess.ws.save_fig(fig, "leaderboard/leaderboard_history.png", H)
    sess.ws.push()
    return {"history": H, "messages": msg, "status": BLOCKED if msg and msg[0].startswith("REJECTED") else "OK"}
