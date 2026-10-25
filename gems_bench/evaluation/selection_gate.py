"""M6.7: final model-selection gate. Extends the section-17 promotion protocol; never forces a winner.

Each candidate (single FULL model or an ensemble of FULL models) is assessed against **both** controls (B0 and
B1). Criteria 1-6 are the section-17 criteria (``scoreboard.promotion_decision`` with the protocol's rules);
M6 adds 7-10 and evidence checks:

 1 primary discovery        disc DTI (primary policy) gain >= min_disc_gain
 2 geographic               geo DTI drop <= max_geo_drop                  (section 17: 3)
 3 worst-fold robustness    geo fold-min and discovery split-min drop <= max_fold_min_drop, splits beating
 4 retained-FP behaviour    retained_fp discovery drop <= max_alt_drop     (section 17: 2)
 5 short-fault recall       < 2 km weighted recall drop <= max_short_recall_drop
 6 area calibration         cross-fitted area within max_area_ratio of the control's; far-field share bounded
 7 spatial stability        block-bootstrap P(geo delta > 0) not below min_p_positive when geo is the claimed gain
 8 no leakage               every selection audit row reads only non-held-out zones
 9 provenance               every member is a COMPLETE FULL run on the shared protocol
10 computational cost       relative inference cost <= max_inference_cost_ratio

Verdict per control: ``INSUFFICIENT_EVIDENCE`` (a required number is missing, or provenance is incomplete),
``REJECT`` (leakage, a non-FULL member, or no gain on the primary objective together with a regression beyond
tolerance on geography or retained-FP), ``PROMOTE`` (every criterion passes), otherwise ``REVIEW``.
Overall verdict: the most cautious of the two (INSUFFICIENT_EVIDENCE > REJECT > REVIEW > PROMOTE).
The recommended candidate is the PROMOTE candidate with the highest primary discovery DTI (the protocol's
designated objective); when none passes, ``recommendation`` says so. Nothing is promoted in the registry here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VERDICTS = ("PROMOTE", "REVIEW", "REJECT", "INSUFFICIENT_EVIDENCE")
_CAUTION = {"INSUFFICIENT_EVIDENCE": 3, "REJECT": 2, "REVIEW": 1, "PROMOTE": 0}

GATE_DEFAULTS = {
    "max_short_recall_drop": 0.01,
    "max_area_ratio": 1.5,
    "max_inference_cost_ratio": 4.0,
    "min_p_positive": 0.5,
    "required_profile": "FULL",
}


def _v(row, col):
    try:
        x = float(row.get(col, np.nan))
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def gate_one(sc: dict, cand: str, ctrl: str, *, rules: dict, gate: dict, evidence: dict) -> dict:
    """Assess ``cand`` against ``ctrl``; ``evidence`` = {provenance_ok, members_full, leakage_ok, cost, boot_p}."""
    from .scoreboard import DEFAULT_RULES, promotion_decision

    R = {**DEFAULT_RULES, **(rules or {})}
    G = {**GATE_DEFAULTS, **(gate or {})}
    s17 = promotion_decision(sc, cand, ctrl, R, artifacts_ok=evidence.get("provenance_ok"))
    s = sc["score"].set_index("model")
    if cand not in s.index or ctrl not in s.index:
        return {"candidate": cand, "control": ctrl, "verdict": "INSUFFICIENT_EVIDENCE",
                "reason": "missing scorecard rows", "criteria": {}, "evidence": {}}
    a, b = s.loc[cand], s.loc[ctrl]
    pp, ap = R["primary_policy"], R["alt_policy"]
    c17 = s17.get("criteria", {})
    crit, ev = {}, {}

    def put(key, ok, value):
        crit[key] = "UNKNOWN" if ok is None else ("PASS" if ok else "FAIL")
        ev[key] = value

    d1 = _v(a, f"disc_dti_{pp}") - _v(b, f"disc_dti_{pp}")
    put("1_primary_discovery", None if np.isnan(d1) else c17.get("1_primary_discovery"), f"{d1:+.4f}")
    d2 = _v(a, "geo_dti") - _v(b, "geo_dti")
    put("2_geographic", None if np.isnan(d2) else c17.get("3_geographic_competitive"), f"{d2:+.4f}")
    dmin = _v(a, "geo_fold_min") - _v(b, "geo_fold_min")
    put("3_worst_fold", None if np.isnan(dmin) else c17.get("6_stability"), s17.get("notes", {}).get("stability", ""))
    d4 = _v(a, f"disc_dti_{ap}") - _v(b, f"disc_dti_{ap}")
    put("4_retained_fp", None if np.isnan(d4) else c17.get("2_alt_policy_no_regression"), f"{d4:+.4f}")
    d5 = _v(a, "short_recall_w") - _v(b, "short_recall_w")
    put("5_short_fault_recall", None if np.isnan(d5) else bool(d5 >= -G["max_short_recall_drop"]), f"{d5:+.4f}")
    ra = _v(a, f"disc_frac_{pp}") / max(_v(b, f"disc_frac_{pp}"), 1e-9)
    rg = _v(a, "geo_frac") / max(_v(b, "geo_frac"), 1e-9)
    area_ok = None if np.isnan(ra) and np.isnan(rg) else bool(
        all(np.isnan(x) or x <= G["max_area_ratio"] for x in (ra, rg)) and c17.get("5_fp_footprint", True))
    put("6_area_calibration", area_ok, f"area ratio geo x{rg:.2f}, disc x{ra:.2f}; far-share {s17.get('notes', {}).get('fp', '')}")
    bp = evidence.get("boot_p", {}).get(ctrl)
    put("7_spatial_stability", None if bp is None else bool(d2 <= 0 or bp >= G["min_p_positive"]),
        "n/a" if bp is None else f"P(geo delta > 0) = {bp:.2f}")
    put("8_no_leakage", evidence.get("leakage_ok"), "selection audit")
    put("9_provenance", evidence.get("provenance_ok"), evidence.get("provenance_note", ""))
    cost = evidence.get("cost")
    put("10_cost", None if cost is None else bool(cost <= G["max_inference_cost_ratio"]),
        "n/a" if cost is None else f"x{cost:.1f} single-model inference")
    if evidence.get("members_full") is False:
        verdict, why = "REJECT", "a member is not a FULL-protocol run"
    elif crit["8_no_leakage"] == "FAIL":
        verdict, why = "REJECT", "selection used held-out labels"
    elif any(crit[k] == "UNKNOWN" for k in ("1_primary_discovery", "2_geographic", "4_retained_fp",
                                             "5_short_fault_recall", "8_no_leakage", "9_provenance")):
        verdict, why = "INSUFFICIENT_EVIDENCE", "required evidence missing: " + ", ".join(
            k for k, v in crit.items() if v == "UNKNOWN")
    elif crit["9_provenance"] == "FAIL":
        verdict, why = "INSUFFICIENT_EVIDENCE", "provenance incomplete"
    elif all(v == "PASS" for v in crit.values() if v != "UNKNOWN"):
        verdict, why = "PROMOTE", "every criterion passes"
    elif crit["1_primary_discovery"] == "FAIL" and (crit["2_geographic"] == "FAIL" or crit["4_retained_fp"] == "FAIL"):
        verdict, why = "REJECT", "no primary gain and a regression beyond tolerance"
    else:
        verdict, why = "REVIEW", "failed: " + ", ".join(k for k, v in crit.items() if v == "FAIL")
    return {"candidate": cand, "control": ctrl, "verdict": verdict, "reason": why, "criteria": crit,
            "evidence": ev, "section17": s17}


def gate(sc: dict, candidates: list[str], controls: list[str], *, rules: dict, gate_cfg: dict,
         evidence: dict[str, dict], primary_policy: str = "excl") -> tuple[pd.DataFrame, dict]:
    """(decision table: one row per candidate x control + one overall row, recommendation)."""
    rows, overall = [], {}
    for c in candidates:
        res = [gate_one(sc, c, k, rules=rules, gate=gate_cfg, evidence=evidence.get(c, {})) for k in controls if k != c]
        for r in res:
            rows.append({"candidate": c, "control": r["control"], "verdict": r["verdict"], "reason": r["reason"],
                         **{k: v for k, v in r["criteria"].items()},
                         **{f"evidence_{k}": v for k, v in r["evidence"].items()}})
        if res:
            v = max((r["verdict"] for r in res), key=_CAUTION.get)
            overall[c] = v
            rows.append({"candidate": c, "control": "ALL", "verdict": v,
                         "reason": "most cautious verdict over controls " + ", ".join(r["control"] for r in res)})
    table = pd.DataFrame(rows)
    s = sc["score"].set_index("model")
    ok = [c for c, v in overall.items() if v == "PROMOTE" and c in s.index]
    col = f"disc_dti_{primary_policy}"
    if ok:
        best = max(ok, key=lambda c: _v(s.loc[c], col))
        rec = {"recommended": best, "verdict": "PROMOTE",
               "rule": f"highest {col} among PROMOTE candidates (the protocol's designated primary objective)",
               "note": "registry promotion stays a manual, explicit step (gems_bench.experiment.promote)"}
    else:
        rec = {"recommended": None, "verdict": "NO_CANDIDATE_PASSES",
               "rule": "no candidate passed every criterion against both controls",
               "review": [c for c, v in overall.items() if v == "REVIEW"]}
    return table, rec
