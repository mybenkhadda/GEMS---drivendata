"""Stage e15: dashboards D1-D10, findings table, feature-engineering hypothesis register, modeling decision, reports
and the executive summary. Everything is read back from the figure interpretation records written by the figure
stages (figures/interpretations/*.json): no number or verdict is written here that a figure did not measure."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import EDA_CODE_REV
from . import viz as V
from .core import DESCRIPTIVE, INCONCLUSIVE, NOT_AVAILABLE, NOT_SUPPORTED, SUPPORTED, Result, Spec

TESTED = (SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE)
PART_NAMES = {"A": "Geological landscape", "B": "Fault geometry", "C": "Fault signatures", "D": "Scale and "
              "localisation", "E": "Cross-modal evidence", "F": "Label bias", "G": "Regional generalisation",
              "H": "Model forensics", "I": "Discovery candidates", "DEM": "DEM pilot"}
DASHBOARDS = [("D1", "landscape", "A"), ("D2", "geometry", "B"), ("D3", "signatures", "C"), ("D4", "scales", "D"),
              ("D5", "crossmodal", "E"), ("D6", "label_bias", "F"), ("D7", "regional", "G"),
              ("D8", "forensics", "H"), ("D9", "candidates", "I"), ("D10", "executive", None)]
REPORTS = {"DATA_UNDERSTANDING.md": ("A",), "GEOLOGICAL_LANDSCAPE.md": ("A",), "FAULT_SIGNATURES.md": ("C", "E"),
           "SPATIAL_SCALE_ANALYSIS.md": ("D",), "LABEL_BIAS.md": ("F", "G"), "MODEL_FORENSICS.md": ("H",),
           "DISCOVERY_CANDIDATES.md": ("I", "DEM")}

# feature-engineering hypotheses: (category, hypothesis, source figures). Status comes from those figures.
HYPOTHESES = [
    ("magnetic_gradients", "Fault-normal magnetic gradient / step features separate faults from matched controls",
     ("F023", "F029")),
    ("gravity_gradients", "Gravity gradient / step features add information beyond magnetics", ("F024", "F052")),
    ("terrain", "Detrended-terrain steps or slope breaks mark faults", ("F025",)),
    ("structural_continuity", "Disruption (low side correlation, offsets, weak along-strike persistence) marks faults",
     ("F030", "F034", "F055")),
    ("fabric_orientation", "Magnetic / gravity fabric alignment is a fault indicator", ("F050",)),
    ("multi_family_agreement", "Coincident crests of independent families mark faults", ("F048", "F049")),
    ("crossmodal_offset", "Coincident magnetic and gravity gradient maxima localise faults", ("F051", "F032")),
    ("scale_selection", "One detection scale and one localisation scale suffice", ("F038", "F039", "F040", "F074")),
    ("network_topology", "Intersections / network context carry evidence beyond amplitude", ("F056", "F021")),
    ("label_incompleteness", "Unmapped lineaments resemble mapped ones (PU-aware training needed)",
     ("F058", "F061", "F063", "F067")),
    ("regional_normalisation", "Signatures transfer to regions not used for selection", ("F077", "F078", "F073")),
    ("model_underuse", "W2 misses carry physical evidence the model under-uses", ("F084", "F085", "F088")),
]

# the modeling decision: (question id, question, source figures)
QUESTIONS = [
    ("Q1", "Which physical signals separate mapped faults from matched controls?", ("F029", "F023", "F024", "F025")),
    ("Q2", "Are fault signatures contrasts (steps) or disruptions (texture / continuity)?",
     ("F030", "F031", "F034", "F055")),
    ("Q3", "Are frozen-W2 losses mainly detection or localisation failures?", ("F080", "F085", "F087")),
    ("Q4", "Do missed faults carry independent physical evidence?", ("F084", "F088")),
    ("Q5", "Which scales detect and which localise faults?", ("F038", "F039", "F040", "F047")),
    ("Q6", "Do independent families add information (complementarity)?", ("F048", "F049", "F052")),
    ("Q7", "Which bands are redundant?", ("F053",)),
    ("Q8", "How strong is label-selection bias / incompleteness?", ("F058", "F060", "F061", "F067")),
    ("Q9", "Do signatures transfer across regions?", ("F077", "F078")),
    ("Q10", "How large is the distribution shift between regions?", ("F073", "F076")),
    ("Q11", "Are W2 false positives errors or plausible unmapped structures?", ("F086",)),
    ("Q12", "Do short / particular-strike faults fail more?", ("F081", "F082")),
    ("Q13", "Which candidates deserve expert review first?", ("F091", "F093", "F098")),
    ("Q14", "Does high-resolution terrain add evidence?", ("F099",)),
    ("Q15", "What is the next modelling step, and what would invalidate it?", ()),
]


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:                                  # tabulate absent: plain text, same content
        return "```\n" + df.to_string(index=False) + "\n```"


def interp(E, fid: str) -> dict:
    return E.ws.read_json(f"figures/interpretations/{fid}.json") or {}


def status(E, fid: str) -> str:
    return (interp(E, fid).get("measured") or {}).get("status", "NOT_RUN")


def catalog(E) -> pd.DataFrame:
    C = E.ws.load_table("figures/FIGURE_CATALOG.csv")
    return C if len(C) else pd.DataFrame(columns=["figure_id", "part", "status", "priority", "png", "title"])


# =============================================================================== findings
def findings(E) -> pd.DataFrame:
    rows = []
    for r in catalog(E).itertuples():
        if str(r.figure_id).startswith("D"):
            continue
        j = interp(E, r.figure_id)
        m = j.get("measured") or {}
        rows.append({"finding_id": f"EDA-{r.figure_id}", "figure_id": r.figure_id, "part": r.part,
                     "question": j.get("question"), "evidence_status": m.get("status"),
                     "effect": m.get("effect"), "ci_lo": m.get("ci_lo"), "ci_hi": m.get("ci_hi"),
                     "n": m.get("n"), "measured": j.get("interpretation"),
                     "modeling_implication": j.get("modeling_implication"), "limitations": j.get("limitations"),
                     "artifact": (j.get("artifacts") or {}).get("png")})
    return pd.DataFrame(rows)


def validated_md(F: pd.DataFrame) -> str:
    out = ["# Validated EDA findings", "", f"Code rev {EDA_CODE_REV}. Status rule: SUPPORTED when the 95 % cluster-"
           "bootstrap CI (over fault systems or regions, as stated) excludes the +-min_effect equivalence band; "
           "NOT_SUPPORTED when the CI lies inside it; otherwise INCONCLUSIVE. DESCRIPTIVE figures make no test "
           "claim. Unmapped locations are not confirmed non-faults.", ""]
    for st in (*TESTED, DESCRIPTIVE, NOT_AVAILABLE):
        g = F[F.evidence_status == st]
        out += [f"## {st} ({len(g)})", ""]
        out += [f"* **{r.figure_id}** {r.question} - {r.measured}" for r in g.itertuples()] or ["(none)"]
        out.append("")
    return "\n".join(out)


# =============================================================================== hypotheses
def hypothesis_status(sts: list[str]) -> str:
    t = [s for s in sts if s in TESTED]
    if not t:
        return "NOT_TESTED"
    if SUPPORTED in t and NOT_SUPPORTED not in t:
        return "PROMOTE_TO_EXPERIMENT"
    if NOT_SUPPORTED in t and SUPPORTED not in t:
        return "DO_NOT_PRIORITISE"
    return "MIXED_EVIDENCE"


def hypotheses(E) -> pd.DataFrame:
    rows = []
    for k, (cat, h, figs) in enumerate(HYPOTHESES, start=1):
        sts = [status(E, f) for f in figs]
        rows.append({"hypothesis_id": f"H{k:02d}", "category": cat, "hypothesis": h, "source_figures": ";".join(figs),
                     "source_statuses": ";".join(sts), "register_status": hypothesis_status(sts),
                     "evidence": " | ".join(f"{f}: {interp(E, f).get('interpretation', 'not run')[:160]}"
                                            for f in figs),
                     "test_before_use": "official-metric held-out comparison at equal realized budget "
                                        "(never the public leaderboard)"})
    return pd.DataFrame(rows)


# =============================================================================== decision
def num(E, fid, key):
    v = (interp(E, fid).get("measured") or {}).get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def decision(E) -> dict:
    """Recommendation derived from the forensics and evidence figures (rules recorded in the report)."""
    sh = (interp(E, "F080").get("measured") or {})
    loc, mrh, det = (float(sh[k]) if sh.get(k) is not None else np.nan
                     for k in ("localisation_failure", "missed_ranked_high", "detection_failure"))
    high = num(E, "F085", "share")
    rules = []
    if all(np.isfinite([loc, mrh, det])) and loc + mrh > det:
        rec = "REPRESENTATION_AND_BUDGET"
        rules.append(f"localisation failures ({loc:.2f}) + high-rank misses ({mrh:.2f}) of mapped length exceed "
                     f"pure detection failures ({det:.2f}) (F080): W2 ranks most missed faults; improve thinning / "
                     "snapping / budget allocation before adding features")
    elif status(E, "F084") == NOT_SUPPORTED or (status(E, "F084") == SUPPORTED and num(E, "F084", "effect") < 0):
        rec = "NEW_EVIDENCE_FEATURES"
        rules.append("missed branches carry independent evidence comparable to detected ones (F084): add the "
                     "promoted hypothesis features to a held-out challenger")
    elif status(E, "F058") == NOT_SUPPORTED or status(E, "F061") == NOT_SUPPORTED:
        rec = "LABEL_REVIEW_AND_PU"
        rules.append("unmapped lineaments resemble mapped ones (F058 / F061): expand labels through blind review and "
                     "use PU-aware objectives")
    elif all(status(E, f) in ("NOT_RUN", NOT_AVAILABLE) for f in ("F080", "F084", "F085")):
        rec = "INSUFFICIENT_EVIDENCE"
        rules.append("model forensics not available (engine not attached): no recommendation")
    else:
        rec = "KEEP_W2_AND_REVIEW_CANDIDATES"
        rules.append(f"pure detection failures dominate (F080) and F084 is {status(E, 'F084')} (effect "
                     f"{num(E, 'F084', 'effect'):+.3f}): no evidence that the model under-uses its inputs; spend "
                     "effort on blind candidate review before new models")
    inval = {"REPRESENTATION_AND_BUDGET": "a representation change that does not raise held-out pooled DTI at equal "
                                          "realized budget (paired block bootstrap CI including 0) invalidates it",
             "NEW_EVIDENCE_FEATURES": "a challenger with the new features that does not beat frozen W2 on the "
                                      "held-out V3 zones at equal budget invalidates it",
             "LABEL_REVIEW_AND_PU": "if < 1/3 of evidence-only reviewed unmapped candidates are LIKELY / POSSIBLE "
                                    "faults, label incompleteness is not the bottleneck",
             "KEEP_W2_AND_REVIEW_CANDIDATES": "independent review confirming many missed-fault evidence windows "
                                              "invalidates it",
             "INSUFFICIENT_EVIDENCE": "n/a - run e11 with the frozen engine attached"}[rec]
    return {"recommendation": rec, "rules": rules, "invalidation": inval,
            "inputs": {"F080": sh, "F085_high_rank_share": high, "F084": status(E, "F084"),
                       "F058": status(E, "F058"), "F061": status(E, "F061")}}


def decision_md(E, dec: dict) -> str:
    out = ["# Modeling decision", "", f"Code rev {EDA_CODE_REV}; generated {time.strftime('%Y-%m-%d %H:%M')}. "
           "Answers are the measured interpretations of the cited figures; no model was trained in this EDA. "
           "The public leaderboard is never a tuning target.", ""]
    for qid, q, figs in QUESTIONS:
        out += [f"## {qid}. {q}", ""]
        if qid == "Q15":
            out += [f"**Recommendation: {dec['recommendation']}**", "", *[f"* {r}" for r in dec["rules"]], "",
                    f"Invalidation criterion: {dec['invalidation']}", ""]
            continue
        for f in figs:
            j = interp(E, f)
            out.append(f"* {f} [{(j.get('measured') or {}).get('status', 'NOT_RUN')}]: "
                       f"{j.get('interpretation', 'not run')} Implication: {j.get('modeling_implication', '')}")
        out.append("")
    return "\n".join(out)


# =============================================================================== dashboards / reports
def pick_executive(C: pd.DataFrame) -> pd.DataFrame:
    C = C[~C.figure_id.astype(str).str.startswith("D") & C.png.notna() & (C.status != NOT_AVAILABLE)].copy()
    C["tested"] = C.status.isin(TESTED)
    C = C.sort_values(["priority", "tested", "figure_id"], ascending=[True, False, True])
    p1 = C[C.priority == 1]
    out = p1.head(15)
    if len(out) < 10:
        out = pd.concat([out, C[C.priority != 1].head(10 - len(out))])
    return out


def mosaic(E, rows: pd.DataFrame, title: str):
    import matplotlib.image as mpimg

    rows = rows[rows.png.notna()].head(12)
    n = max(len(rows), 1)
    fig, ax = V.grid(n, 3 if n > 4 else 2, 6, 4.6)
    for a_, r in zip(ax, rows.itertuples()):
        p = E.ws.path(r.png)
        if p.exists():
            a_.imshow(mpimg.imread(str(p)))
        a_.set_axis_off()
        a_.set_title(f"{r.figure_id} [{r.status}] {str(r.title)[:60]}", fontsize=7)
    fig.suptitle(title, fontsize=11)
    return fig


def dashboards(E, C: pd.DataFrame) -> list[str]:
    done = []
    for did, name, part in DASHBOARDS:
        rows = pick_executive(C) if part is None else C[(C.part == part) & ~C.figure_id.astype(str).str.startswith("D")]
        title = "Executive selection" if part is None else f"Part {part}: {PART_NAMES[part]}"
        spec = Spec(did, name, title, part or "X", f"Overview of {title.lower()}", "Mosaic of the part's figures",
                    "One view per part", "None (mosaic)", {DESCRIPTIVE: "Dashboard."}, {DESCRIPTIVE: "See figures."},
                    "Reduced-resolution copies of the figures.", 1)
        if not len(rows):
            E.reg.save(spec, None, None, Result(NOT_AVAILABLE, "no figure in this part"))
            continue
        E.reg.save(spec, mosaic(E, rows, title), rows[["figure_id", "status", "title"]],
                   Result(DESCRIPTIVE, f"{len(rows)} figures; statuses {rows.status.value_counts().to_dict()}", {},
                          len(rows)))
        done.append(did)
    return done


def part_report(E, name: str, parts: tuple, C: pd.DataFrame) -> str:
    out = [f"# {name[:-3].replace('_', ' ').title()}", "", f"Code rev {EDA_CODE_REV}. Figures of part(s) "
           f"{', '.join(parts)}; every paragraph is the figure's own measured interpretation.", ""]
    if name == "DATA_UNDERSTANDING.md":
        Fi = E.ws.load_table("audit/raw_data_inventory.csv")
        A = E.ws.load_table("audit/previous_eda_audit.csv")
        out += ["## Inputs", "", md_table(Fi) if len(Fi) else "(inventory not run)", "",
                "## Audit of the previous EDA", "", md_table(A) if len(A) else "(not run)", ""]
    for r in C[C.part.isin(parts)].sort_values("figure_id").itertuples():
        j = interp(E, r.figure_id)
        out += [f"## {r.figure_id} {r.title} [{r.status}]", "", f"*Question:* {j.get('question', '')}", "",
                f"{j.get('interpretation', '')}", "", f"*Modeling implication:* {j.get('modeling_implication', '')}",
                f"*Limitations:* {j.get('limitations', '')}", ""]
        if isinstance(r.png, str):
            out += [f"![{r.figure_id}](../{r.png})", ""]
    return "\n".join(out)


def executive_md(E, C: pd.DataFrame, F: pd.DataFrame, H: pd.DataFrame, dec: dict) -> str:
    X = pick_executive(C)
    cnt = F.evidence_status.value_counts().to_dict() if len(F) else {}
    out = ["# Executive research summary", "", f"Code rev {EDA_CODE_REV}; mode {E.c['mode']}; commit {E.head}. "
           f"{len(F)} figures recorded; statuses {cnt}. Selected figures are chosen automatically (priority 1 first, "
           "tested results before descriptive ones, 10-15 figures).", "",
           f"## Recommendation: {dec['recommendation']}", "", *[f"* {r}" for r in dec["rules"]], "",
           f"Invalidation criterion: {dec['invalidation']}", "", "## Key figures", ""]
    for r in X.itertuples():
        j = interp(E, r.figure_id)
        out += [f"### {r.figure_id} {r.title} [{r.status}]", "", j.get("interpretation", ""), "",
                f"![{r.figure_id}](../{r.png})", ""]
    out += ["## Hypotheses promoted to experiments", ""]
    P = H[H.register_status == "PROMOTE_TO_EXPERIMENT"] if len(H) else H
    out += [f"* {r.hypothesis_id} ({r.category}): {r.hypothesis}" for r in P.itertuples()] or ["(none)"]
    out += ["", "Unmapped locations are not confirmed non-faults; candidates are not discoveries until independently "
            "reviewed; no result here was tuned on the public leaderboard.", ""]
    return "\n".join(out)


def run(E) -> dict:
    C = catalog(E)
    F = findings(E)
    E.ws.save_table(F, "tables/evidence_synthesis.csv")
    E.ws.save_text(validated_md(F), "reports/VALIDATED_EDA_FINDINGS.md")
    H = hypotheses(E)
    E.ws.save_table(H, "tables/feature_engineering_hypotheses.csv")
    E.ws.save_text("# Feature-engineering hypotheses\n\n" + "Register status from the source figures: PROMOTE_TO_"
                   "EXPERIMENT (supported, no contrary test), DO_NOT_PRIORITISE, MIXED_EVIDENCE, NOT_TESTED. A promoted "
                   "hypothesis is a candidate experiment, not a result.\n\n" + (md_table(H) if len(H) else "(none)") + "\n",
                   "reports/FEATURE_ENGINEERING_HYPOTHESES.md")
    dec = decision(E)
    E.ws.save_json(dec, "reports/modeling_decision.json")
    E.ws.save_text(decision_md(E, dec), "reports/MODELING_DECISION.md")
    D = dashboards(E, C)
    C = catalog(E)
    for name, parts in REPORTS.items():
        E.ws.save_text(part_report(E, name, parts, C), f"reports/{name}")
    E.ws.save_text(executive_md(E, C, F, H, dec), "reports/EXECUTIVE_RESEARCH_SUMMARY.md")
    E.ws.save_json({"code_rev": EDA_CODE_REV, "commit": E.head, "mode": E.c["mode"], "config": E.c,
                    "stage_status": dict(E.status), "n_figures": int(len(F)),
                    "status_counts": F.evidence_status.value_counts().to_dict() if len(F) else {},
                    "dashboards": D, "recommendation": dec["recommendation"],
                    "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, "manifests/run_manifest.json")
    return {"note": f"{len(F)} findings; {len(D)} dashboards; recommendation {dec['recommendation']}",
            "recommendation": dec["recommendation"]}
