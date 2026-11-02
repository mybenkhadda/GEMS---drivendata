"""Z_report: reports from PERSISTED tables only (no number is computed here that a table does not hold)."""
from __future__ import annotations

import json
import time

import pandas as pd

DELIVERABLES = (
    ("1 preflight / dependency inventory", "A_inventory", ("preflight/dependency_inventory.csv",
                                                           "preflight/g05_run_selection.json")),
    ("2 failure atlas + case browser", "B_atlas", ("atlas/failure_atlas.csv", "atlas/case_browser.html",
                                                   "atlas/ATLAS_SUMMARY.md")),
    ("3 equal-budget recall", "C_recall", ("recall/recall_summary.csv", "recall/new_lost_vs_w2.csv",
                                           "recall/leave_one_expert_out.csv")),
    ("4 recovery matrix", "C_recall", ("recall/recovery_matrix.parquet", "recall/target_system_recovery.csv")),
    ("5 hypothesis registry", "D_registry", ("registry/fault_hypothesis_registry.parquet",
                                             "registry/hypothesis_geometry.gpkg")),
    ("6 feature dictionary + provenance", "D_registry", ("registry/feature_dictionary.csv",
                                                         "registry/feature_provenance.csv")),
    ("7 displacement evidence + controls", "E_displacement", ("displacement/displacement_evidence.parquet",
                                                              "displacement/control_summary.csv")),
    ("8 candidate graph", "F_graph", ("graph/graph_features.parquet", "graph/graph_summary.csv")),
    ("9 residual rankers (cross-fitted)", "G_rankers", ("rankers/model_registry.json", "rankers/label_summary.csv")),
    ("10 strict-nesting audit", "H_nesting", ("nesting/nesting_audit.csv", "nesting/retraining_plan.csv")),
    ("11 selection + promotion decision", "I_selection", ("selection/model_summary.csv",
                                                          "selection/promotion_decisions.csv",
                                                          "selection/decision.json")),
    ("12 reconstruction / submission manifest", "K_reconstruct", ("reconstruction/reconstruction_manifest.json",
                                                                  "submissions/submission_manifest.json")),
)


def deliverables(R) -> pd.DataFrame:
    rows = []
    for name, stage, files in DELIVERABLES:
        for f in files:
            rows.append({"deliverable": name, "stage": stage, "stage_status": R.status.get(stage), "file": f,
                         "exists": R.ws.path(f).exists()})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols=None, n: int = 30) -> str:
    if df is None or not len(df):
        return "_(no rows)_\n"
    d = df[[c for c in (cols or df.columns) if c in df]].head(n)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "".join("| " + " | ".join(fmt(v) for v in r) + " |\n" for r in d.itertuples(index=False))


def write(R) -> None:
    ws = R.ws
    Dv = deliverables(R)
    ws.save_table(Dv, "reports/DELIVERABLES.csv")
    st = pd.DataFrame([{"stage": s, "status": R.status.get(s)} for s in R.STAGES])
    L = ["# GEMS G05.1 - fault-system recovery & intelligent ranking", "",
         f"run `{R.run_id}`, code `{R.code_rev}`, commit `{R.head}`, written {time.strftime('%Y-%m-%d %H:%M')}", "",
         "Every number below is read from a table of this run. Historical references were measured under OTHER "
         "protocols and are never compared as if they were this run; the public leaderboard value is an external "
         "reference, never a target.", "", "## Stage status", "", md_table(st), "",
         "## Historical references (other protocols)", "", md_table(pd.DataFrame(R.cfg51["historical"])), ""]
    A = ws.load_table("atlas/target_systems.csv")
    L += ["## B. Failure atlas (targets)", "", md_table(A, ["unit", "system_id", "category", "w2_rank_median",
                                                            "inventory_near_share", "hit_share"]), ""]
    P = ws.load_table("recall/recall_summary.csv")
    if len(P):
        P = P[(P.scenario == "V1") & (P.role == "final")].sort_values(["frac", "dti"], ascending=[True, False])
    L += ["## C. Equal-budget recall (V1, final budgets)", "",
          md_table(P, ["construction", "kind", "frac", "dti", "n_recovered", "n_systems", "area_frac_eligible"], 60),
          ""]
    E = ws.load_table("displacement/control_summary.csv")
    L += ["## E. Displacement-style evidence vs negative control", "", md_table(E), ""]
    N = ws.read_json("nesting/strictness.json")
    L += ["## H. Strictness", "", f"```\n{json.dumps(N, indent=1) if N else 'Stage H not run'}\n```", ""]
    S = ws.load_table("selection/model_summary.csv")
    L += ["## I. Fixed-policy results (vs frozen submitted W2)", "",
          md_table(S, ["scenario", "model", "dti", "worst_zone_dti", "worst_zone", "protocol"], 60), "",
          "Learned models (G1-G5) are SCREENING_ONLY: their numbers are screening results, NOT nested DTI.", ""]
    D = ws.load_table("selection/promotion_decisions.csv")
    L += ["### Promotion decisions", "", md_table(D, ["model", "strictness", "status", "disc_gain", "geo_delta",
                                                     "net_systems", "reasons"]), ""]
    dec = ws.read_json("selection/decision.json") or {}
    L += [f"**Decision:** {dec.get('status', 'NOT_EVALUATED')} -> {dec.get('selected')}; final "
          f"{dec.get('final_status', '-')}. Nothing was uploaded.", "",
          "## Deliverables", "", md_table(Dv, n=60)]
    ws.save_text("\n".join(L), "reports/G051_REPORT.md")
    F = R.failure_summary()
    ws.save_text("# Execution status\n\n" + md_table(st) + "\n## Not completed\n\n" + md_table(F),
                 "reports/EXECUTION_STATUS.md")
