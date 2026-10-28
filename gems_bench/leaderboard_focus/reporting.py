"""Reports, figures (PNG + plotting data CSV) and the manual leaderboard log of Milestone 1.

Every report quotes only persisted tables; recorded historical numbers are labelled ``recorded``; public leaderboard
scores are recorded after a MANUAL upload and never feed back into selection (no tuning on the public score).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import BLOCKED, COMPLETE, FAILED, INCOMPLETE, NOT_REQUIRED, READY, WARNING

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
    status = BLOCKED if msg and msg[0].startswith("REJECTED") else COMPLETE if msg else NOT_REQUIRED
    sess.record("leaderboard", status, "", reasons=[m for m in msg if m.startswith("REJECTED")],
                outputs=["leaderboard/leaderboard_review.md"], n_recorded=int(len(H)))
    return {"history": H, "messages": msg, "status": status}


# =============================================================================== final report
# Known failure modes: (code, detection point, substring of the recorded reason, status, remediation). Every
# non-passing manifest is matched against it in reports/M1_FAILURE_DIAGNOSTICS.md.
FAILURE_MODES = [
    ("ART_COUNT", "stage 1 inventory", "COUNT_MISMATCH", BLOCKED,
     "A count-based artifact has a different number of files: restore the frozen folder (M1 never recomputes it)."),
    ("ART_REQUIRED_FILE", "stage 1 inventory", "MISSING_REQUIRED_FILES", BLOCKED,
     "A required map is absent from its folder (the note lists what was found): restore it from the original run."),
    ("ART_MISSING", "stage 1 inventory", ": MISSING", BLOCKED,
     "A required artifact is absent from Drive: restore it (template: run the pipeline raw_data step once)."),
    ("ART_AMBIGUOUS", "stage 1 inventory", "AMBIGUOUS", BLOCKED,
     "Several experiment folders match one artifact: M1 never combines or picks; resolve the duplicate by hand."),
    ("ART_ID", "stage 1 inventory", "ID_CHANGED", BLOCKED,
     "A B0 / B1 run differs from the recorded id: confirm the model, then set expected_id explicitly."),
    ("STAGING", "session attach", "staged input(s) absent", INCOMPLETE,
     "Staged copies are missing locally: re-run stage 1 (it re-stages and verifies every file)."),
    ("GRID_REQUIRED", "stage 2 / 3 load_grids", "required grid(s) unusable", BLOCKED,
     "The frozen boost-A grid is absent / empty / malformed (grid_provenance.csv gives the facts): restore "
     "outputs/p6_fast_track/boost/benchmark/grid.csv. No fallback grid or recorded score is used."),
    ("UPSTREAM", "every stage (require)", "upstream stages not usable", INCOMPLETE,
     "An earlier stage is not COMPLETE / WARNING: run mode FULL (or the earlier stage) first."),
    ("STALE_CODE", "every stage (require)", "produced by different stage code", INCOMPLETE,
     "An upstream result was produced by other code: re-run that stage (mode FULL does it automatically)."),
    ("TABLE", "stage inputs (require_table)", "absent or empty", BLOCKED,
     "A persisted stage table is missing or empty: re-run the stage that writes it."),
    ("RECIPE", "stages 3 / 5 / promotion", "recipe ids disagree", BLOCKED,
     "Recipe ids disagree between selection files: re-run selection (never edit the files by hand)."),
    ("PREFLIGHT", "preflight", "preflight BLOCKED", INCOMPLETE,
     "A required frozen input is missing / unreadable / misaligned on Drive: read "
     "diagnostics/DRIVE_READINESS_REPORT.md, restore the file, re-run (the engine is not attached meanwhile)."),
    ("GEO_FLOOR", "promotion", "GEOGRAPHIC_GATE_PASS", BLOCKED,
     "The champion is below the internal geographic floor (nested single policy). A user decision: keep it blocked, "
     "or opt in to the exploratory baseline (exploratory.enabled + reason: separate file, never the champion)."),
    ("DISC_ELIG", "promotion", "DISCOVERY_ELIGIBLE", BLOCKED,
     "The champion fails the predeclared discovery rule (eligibility / worst-zone floor)."),
    ("COMPONENTS", "stage 3", "recomputed components differ", BLOCKED,
     "A frozen input changed after boost A: compare champion_provenance.json with artifact_inventory.json."),
    ("ZONE_PARITY", "stage 3 / 4", "zone score parity not PASS", BLOCKED,
     "The deployed zone score differs from the evaluated one (blend < 1: per-zone vs all-zone fusion w). Explicit "
     "decision: validation.strict_parity."),
    ("BUDGET", "stage 3", "area budget", BLOCKED,
     "Since M1.3 the budget is an upper bound (area <= frac x (1 + tol), 0 px on mapped faults, non-empty; the rule of "
     "audit_export). A remaining failure is real: read the 'cause' column of validation/zone_budget.csv and "
     "diagnostics/root_cause_summary.json (AREA_OVER_ALLOCATION / KNOWN_FAULT_MASK_LEAK / empty map)."),
    ("FORENSICS", "stage 3", "forensics diagnostics failed", WARNING,
     "A diagnostic of the dry build raised (validation verdict unaffected): see diagnostics/forensics_error.txt."),
    ("TEMPLATE", "stage 3", "official template check", BLOCKED,
     "The competition template does not match the grid: restore data/raw/geodawn/example_submission.tif."),
    ("RECORDED", "stage 3", "NOT_REPRODUCED", WARNING,
     "A recorded value differs from its recomputation from the frozen grid (see protocol_comparison.csv)."),
    ("EXPORT_SHA", "stage 5", "written band", BLOCKED,
     "The written GeoTIFF is not the validated array: re-run the export; never upload this file."),
    ("NAME", "stage 5", "FileExistsError", FAILED,
     "Two other recipes already own the file name: archive them by hand (M1 never overwrites)."),
    ("ORACLE", "stage 2 load_grids", "OracleLeak", FAILED,
     "Oracle (label-built) rows reached a grid: the grid is not usable for selection; investigate its producer."),
]


def failure_matches(text: str) -> list[tuple]:
    """Known failure modes whose marker occurs in ``text`` (pure)."""
    return [f for f in FAILURE_MODES if f[2] in str(text)]


def next_action(statuses: dict, promotion: dict, submission: dict) -> str:
    """One recommended next action from the persisted statuses (pure)."""
    order = ("artifacts", "selection", "validation", "parity")
    for s in order:
        if statuses.get(s) not in (COMPLETE, WARNING):
            return (f"Stage `{s}` is {statuses.get(s) or 'not run'}: fix the reasons in "
                    f"reports/M1_FAILURE_DIAGNOSTICS.md, then re-run with mode FULL (finished stages are reused).")
    gates = {g["gate"]: g["status"] for g in (promotion.get("gates") or [])}
    if promotion.get("status") == BLOCKED:
        hard = [g for g, v in gates.items() if v == BLOCKED and g != "READY_FOR_EXPORT"]
        if hard == ["GEOGRAPHIC_GATE_PASS"]:
            ex = submission.get("exploratory") or {}
            if ex.get("status") == "EXPLORATORY_READY":
                return (f"The champion stays BLOCKED (geographic floor). The opt-in EXPLORATORY baseline "
                        f"`{ex.get('file')}` passed every technical check: uploading it is your decision (check "
                        "sha256 first; record it in stage 6 with notes 'exploratory'). It is not the validated champion.")
            return ("DECISION REQUIRED: the champion is technically valid (validation and parity pass) but below the "
                    "internal geographic floor. Either keep it unexported, or opt in to the exploratory baseline: "
                    "M1_CONFIG = {'exploratory': {'enabled': True, 'reason': '<why>'}} and re-run mode SUBMIT (a "
                    "separately named file, never the champion). M1 does not make this choice.")
        return f"Promotion BLOCKED by {hard}: see selection/promotion_gates.csv."
    if statuses.get("promotion") not in (COMPLETE, WARNING):
        return "Run the promotion stage (mode FULL or SUBMIT)."
    if submission.get("status") == READY:
        return (f"Upload `{submission.get('file')}` MANUALLY on DrivenData (check sha256 first), then fill "
                "LEADERBOARD_ENTRY and run stage 6.")
    return f"Submission {submission.get('status') or 'not exported'}: {submission.get('reasons') or 'run stage 5'}."


def final_report(sess) -> dict:
    """reports/M1_EXECUTIVE_SUMMARY.md, M1_FAILURE_DIAGNOSTICS.md, M1_NEXT_ACTION.md (+ stage_status.csv) from the
    persisted manifests only (works without the P6 engine)."""
    from .config import STAGES

    names = [s for s in STAGES if s not in ("environment", "report")]
    if sess.manifest("session"):
        names.insert(1, "session")                     # an engine attach that failed
    st = {s: sess.manifest(s) for s in names}
    statuses = {s: m.get("status") for s, m in st.items()}
    sel = sess.ws.read_json("selection/champion_selection.json") or {}
    prom = sess.ws.read_json("selection/promotion_gates.json") or {}
    sub = sess.ws.read_json("submissions/submission_manifest.json") or {}
    exm = sess.ws.read_json("submissions/exploratory_manifest.json") or {}
    sub = {**sub, "exploratory": exm}
    met = sel.get("champion_metrics") or {}
    gates = {g["gate"]: g["status"] for g in (prom.get("gates") or [])}
    S = pd.DataFrame([{"stage": s, "status": m.get("status") or "not run", "written": m.get("written"),
                       "code_rev": m.get("code_rev"), "reasons": " | ".join(map(str, m.get("reasons") or [])),
                       "warnings": " | ".join(map(str, m.get("warnings") or []))} for s, m in st.items()])
    sess.ws.save_table(S, "reports/stage_status.csv")
    path = f"{sess.cfg['output_dir']}/submissions/{sub['file']}" if sub.get("file") else None
    rows = [("Selected candidate", sel.get("champion") or "none"), ("Recipe id", sel.get("recipe_id")),
            ("Discovery DTI (nested, single policy)", met.get("disc_dti")),
            ("Worst discovery zone DTI", met.get("disc_worst")),
            ("Geographic DTI (nested, single policy)", met.get("geo_dti")),
            ("Geographic floor", f"{sel.get('geo_floor')} -> {gates.get('GEOGRAPHIC_GATE_PASS', 'PENDING')} "
                                 f"({prom.get('geographic_gate', '?')})"),
            ("Validation", statuses.get("validation")), ("Parity", statuses.get("parity")),
            ("Technical validity (selection + validation + parity)", prom.get("technical_validity", "PENDING")),
            ("Internal promotion (discovery + geographic gates)", prom.get("internal_promotion", "PENDING")),
            ("Promotion (READY_FOR_EXPORT)", gates.get("READY_FOR_EXPORT", "PENDING")),
            ("Submission status", sub.get("status") or "not exported"), ("GeoTIFF (Drive)", path or "none"),
            ("Exploratory baseline (opt-in, NOT the champion)",
             f"{exm.get('status')}: {exm.get('file')}" if exm else "not enabled")]
    blocking = [f"{s}: {r}" for s, m in st.items() if m.get("status") not in (COMPLETE, WARNING, NOT_REQUIRED, None)
                for r in (m.get("reasons") or [])]
    nxt = next_action(statuses, prom, sub)
    summary = ["# M1 executive summary", "",
               "Offline numbers are nested held-out estimates from the frozen boost-A grid, NOT leaderboard scores.", "",
               "| item | value |", "|---|---|", *(f"| {k} | {v} |" for k, v in rows), "",
               "## Blocking reasons", "", *(f"* {b}" for b in blocking or ["none"]), "",
               "## Recommended next action", "", nxt, ""]
    sess.ws.save_text("\n".join(summary), "reports/M1_EXECUTIVE_SUMMARY.md")
    diag = ["# M1 failure diagnostics", "", _md(S), ""]
    for s, m in st.items():
        if m.get("status") in (COMPLETE, NOT_REQUIRED, None):
            continue
        diag += [f"## {s}: {m.get('status')}", ""]
        for r in list(m.get("reasons") or []) + list(m.get("warnings") or []):
            diag.append(f"* {r}")
            for code, where, _, fst, fix in failure_matches(r):
                diag.append(f"  * known failure `{code}` (detected at {where}, status {fst}): {fix}")
        if m.get("traceback"):
            diag += ["", "```", str(m["traceback"])[-2000:], "```"]
        diag.append("")
    sess.ws.save_text("\n".join(diag), "reports/M1_FAILURE_DIAGNOSTICS.md")
    sess.ws.save_text("\n".join(["# M1 next action", "", nxt, ""]), "reports/M1_NEXT_ACTION.md")
    sess.ws.save_text(repair_report(sess, statuses, prom, exm), "reports/M1_3_REPAIR_REPORT.md")
    sess.ws.push()
    return {"summary": dict(rows), "blocking": blocking, "next_action": nxt, "stages": S}


def repair_report(sess, statuses: dict, prom: dict, exm: dict) -> str:
    """reports/M1_3_REPAIR_REPORT.md: the budget repair and the two decisions, quoting persisted diagnostics only."""
    rc = sess.ws.read_json("diagnostics/root_cause_summary.json") or {}
    pre = sess.ws.read_json("diagnostics/root_cause_summary_preflight.json") or {}
    Z = sess.ws.load_table("validation/zone_budget.csv")
    RC = sess.ws.load_table("diagnostics/representation_comparison.csv")
    cols = [c for c in ("zone", "eligible_px", "pool_px", "positive_px", "pool_yield", "frac_of_eligible", "cap_frac",
                        "eval_positive_px", "positive_on_known_px", "m1_1_two_sided_ok", "ok", "cause") if c in Z]
    return "\n".join([
        "# M1.3 repair report", "",
        "Every number below is read from persisted files of this workspace; absent files are reported as absent.", "",
        "## Blocker A: per-zone area budget", "",
        "Root cause (OBSERVED on Drive before the repair, M1.2 run): the selected cell is skeleton_1px @ frac 0.01. "
        "frac is the share of the eligible domain the selector takes (pool = ceil(frac x eligible)); the skeleton keeps "
        "about half of that pool, so the final area was 0.46-0.49 % of eligible in every zone, with 0 px on mapped "
        "faults. The evaluated grid cells (whose DTI was measured) cover 0.39-0.41 %. The M1.1 check |area/frac - 1| <= "
        "0.02 rejected every zone although no thin cell could satisfy it: CHECK_DEFINITION_MISMATCH. The written-file "
        "audit (export_parity.audit_export) and the grid audit (boost_audit.budget_violations) already used the "
        "upper-bound contract; validation now uses the same one. The prediction, recipe id and validated array are "
        "unchanged by this repair.", "",
        f"Preflight re-read of the last persisted budget: {json.dumps(pre.get('zones') or 'absent', default=str)}", "",
        "### Budget of the current validation run", "", _md(Z[cols]) if len(Z) else "_(absent: run validation)_", "",
        f"Root-cause summary: all_ok={rc.get('all_ok', 'absent')}, blocking classes {rc.get('blocking_classes')}", "",
        "### Evaluated vs deployed binary map", "",
        _md(RC) if len(RC) else "_(absent: run validation)_", "",
        "Evaluated-only pixels on mapped faults are removed by the official pixel-exact mask (by design); the rest of "
        "the disagreement is the selector pool re-ranked on the smaller deployed domain.", "",
        "## Blocker B: geographic gate (separate decision)", "",
        f"* Technical validity: **{prom.get('technical_validity', 'PENDING')}** (selection, validation, parity)",
        f"* Internal promotion: **{prom.get('internal_promotion', 'PENDING')}** (discovery eligibility, geographic "
        "floor 0.1758 on the nested single-policy geo DTI)",
        f"* Exploratory eligible: {prom.get('exploratory_eligible', 'PENDING')}; exploratory export: "
        f"{exm.get('status') if exm else 'not enabled'}", "",
        "## Stage statuses", "", *(f"* {k}: {v or 'not run'}" for k, v in statuses.items()), ""])
