"""Section 19: scientific report (rebuilt from saved tables only) and the model registry.

The report never states a number that is not in a saved table, labels PILOT decisions PROVISIONAL, separates oracle
(diagnostic) from cross-fitted (deployable) values, and repeats the honesty notes: the zones were inspected in
earlier sessions (not an untouched test set), pseudo-discovery under-represents new faults next to known traces, and
three folds / three splits give descriptive paired evidence, not significance.
"""
from __future__ import annotations

import time

import pandas as pd


def _md(df: pd.DataFrame, n: int = 25) -> str:
    if df is None or not len(df):
        return "_not produced_\n"
    d = df.head(n).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    cols = list(d.columns)
    out = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    for _, r in d.iterrows():
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return out


def registry(m9) -> pd.DataFrame:
    """One row per trained (variant, unit) + every submission file (reports/model_registry/model_registry.csv)."""
    rec = m9.ws.load_table("training_units", "training")
    rows = []
    keys = m9.cube_keys()
    for r in rec.to_dict("records") if len(rec) else []:
        rows.append({"record": "model", **r, "config_fp": m9.fp("config"), "cube": keys.get(r.get("variant", ""), "")})
    idx = m9.ws.load_table("submission_index", "submissions")
    for r in idx.to_dict("records") if len(idx) else []:
        rows.append({"record": "submission", "variant": r.get("method"), "unit": r.get("role"), "checkpoint": r.get("file"),
                     "status": r.get("status"), "config_fp": m9.fp("config")})
    return pd.DataFrame(rows)


def report(m9, gates: pd.DataFrame) -> str:
    ws = m9.ws
    prof = m9.profile
    pooled = ws.load_table("method_pooled", "validation/metrics")
    lines = [f"# M9 GeoFault production report\n",
             f"Written {time.strftime('%Y-%m-%d %H:%M')} - profile **{prof['name']}**"
             + (" (PROVISIONAL decisions)" if prof["provisional"] else "") + "\n",
             "All numbers come from saved tables of this run (official-style metric, eligible budget, nested choices). "
             "They are offline proxies, never leaderboard results. The zones were inspected in earlier sessions: they "
             "are not an untouched test set. Pseudo-discovery withholds whole systems, so new faults next to known traces "
             "are under-represented. Oracle values are diagnostic upper bounds.\n",
             "## Gates\n", _md(gates[["gate", "component", "decision", "meaning", "next_action"]] if len(gates) else gates, 40),
             "## Baseline (section 01)\n", _md(ws.load_immutable_table("baseline_table", "baseline")),
             "## Feature screen (sections 02-03)\n", _md(ws.load_table("screen_pooled", "spectral")),
             "## Dense validation (section 09)\n",
             _md(pooled[pooled.method.str.startswith(("m9_dense", "s01_"))] if len(pooled) else pooled, 40),
             "### Reconstruction benchmark (CV-C)\n", _md(ws.load_table("reconstruction_pooled", "validation/metrics")),
             "### Variant deltas\n", _md(ws.load_table("variant_deltas", "validation/metrics")),
             "### Recall by system length (CV-D)\n", _md(ws.load_table("length_strata", "validation/metrics"), 40),
             "### DTI by geological domain (CV-E)\n", _md(ws.load_table("domain_dti", "validation/metrics"), 40),
             "### Known-fault proximity (CV-F)\n", _md(ws.load_table("known_proximity", "validation/metrics"), 40),
             "## Tracing oracle (section 11)\n", _md(ws.load_table("pooled_oracle", "tracing/oracle")),
             "## Candidate rankers (section 15)\n", _md(ws.load_table("summary", "gbm")), _md(ws.load_table("choice", "gbm")),
             "## Final comparison and verdicts (section 16)\n", _md(pooled, 60),
             _md(ws.load_table("decision_verdicts", "reports"), 40),
             "## Submission files (section 18)\n", _md(ws.load_table("submission_index", "submissions")),
             "## Limitations\n",
             "* Nothing above was measured before the notebook ran; empty sections were not run or were gated.\n"
             "* The hidden new-fault density is unknown: the submission area fraction (geo all-zone choice) is the "
             "largest unverified assumption.\n"
             "* Monte Carlo occupancy is processing stability, not a calibrated fault probability.\n"
             "* PILOT trains one fold and one split: nested selection is impossible there (borrowed choices are labelled).\n"]
    return "\n".join(lines)
