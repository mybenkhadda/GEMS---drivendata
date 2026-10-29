"""Synthetic runtime self-tests of the reset's pure functions (stage [00]; also called by the pytest file)."""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from . import atlas as AT
from . import benchmark as BM
from . import challenger as CH
from . import physics as PH
from . import terrain as TE


def t_ramp_mass():
    rng = np.random.default_rng(0)
    s = rng.random((60, 60))
    el = np.ones(s.shape, bool)
    el[:5] = False
    q = BM.ramp_map(s, el, 0.02)
    assert abs(q.sum() - 0.02 * el.sum()) / (0.02 * el.sum()) < 0.01 and (q[~el] == 0).all() and q.max() <= 1
    return f"mass {q.sum():.1f} vs target {0.02 * el.sum():.1f}"


def t_evidence_weighted():
    q = np.zeros((4, 4))
    q[1, :] = 1
    e = np.full((4, 4), 0.5)
    w = BM.evidence_weighted(q, e)
    assert np.allclose(w[1], 0.75) and (w[0] == 0).all()
    return "values 0.5 + 0.5 x evidence on selected pixels only"


def t_no_extrapolation():
    g = pd.DataFrame({"mass_frac": [0.01, 0.02], "dti": [0.1, 0.2]})
    r = BM.interpolate_budget(g, [0.015, 0.05])
    assert abs(r[0]["dti_interp"] - 0.15) < 1e-12 and np.isnan(r[1]["dti_interp"])
    return "interpolation inside the measured range only"


def t_selection_separation():
    pools = {"A": [(np.array([10]), np.array([10])), (np.array([11]), np.array([11])),
                   (np.array([90]), np.array([90]))],
             "B": [(np.array([12]), np.array([12])), (np.array([50]), np.array([50]))]}
    k = AT.select(pools, {"A": 2, "B": 2}, lambda r, c: int(r > 50), 20, np.random.default_rng(1))
    pts = [(x["row"], x["col"]) for x in k]
    d = min(np.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(pts) for b in pts[i + 1:])
    assert d >= 20 and len(k) == 3, (pts, d)
    assert AT.quotas({"A": 1, "B": 1, "C": 1}, 10) == {"A": 4, "B": 3, "C": 3}
    return "min separation enforced; categories never filled from another category"


def t_review_provenance():
    cat = pd.DataFrame({"candidate_id": ["c1", "c2"], "review_split": ["TRAIN", "EVAL"], "geometry_wkt": ["", ""]})
    R0 = pd.DataFrame(columns=list(AT.REVIEW_FIELDS))
    ok = {"candidate_id": "c1", "interpretation": "LIKELY_FAULT", "confidence": "HIGH", "review_mode": "EVIDENCE_ONLY",
          "reviewer": "me", "evidence_sources": "magnetic;gravity"}
    R1, e = AT.append_reviews(R0, [ok], cat)
    assert not e and len(R1) == 1 and R1.iloc[0].label_class == "INDEPENDENT_REVIEWED"
    R2, _ = AT.append_reviews(R1, [ok], cat)
    assert len(R2) == 1, "identical repeat appended"
    diag = {**ok, "review_mode": "DIAGNOSTIC"}
    R3, _ = AT.append_reviews(R2, [diag], cat)
    assert len(R3) == 2 and R3.iloc[-1].label_class == "DIAGNOSTIC_REVIEWED" and R3.iloc[0].label_class == \
        "INDEPENDENT_REVIEWED", "earlier row edited"
    dep = {**ok, "evidence_sources": "magnetic;w2"}
    R4, _ = AT.append_reviews(R3, [dep], cat)
    assert R4.iloc[-1].label_class == "DIAGNOSTIC_REVIEWED"           # cites a model: not independent
    _, e = AT.append_reviews(R4, [{**ok, "interpretation": "FAULT"}, {**ok, "candidate_id": "zz"},
                                  {**ok, "reviewer": ""}], cat)
    assert len(e) == 3
    neg, _ = AT.append_reviews(R0, [{**ok, "interpretation": "LIKELY_CONTACT"}], cat, ("LIKELY_CONTACT",))
    assert neg.iloc[0].label_class == "HARD_NEGATIVE" and AT.latest_reviews(R4).shape[0] == 1
    return "append-only; evidence-only + no model source = independent; invalid rows refused"


def t_side_correlation_offset():
    x = np.arange(200)
    pattern = np.sin(x / 6.0)
    a = np.zeros((60, 200))
    a[:30] = pattern
    a[30:] = np.roll(pattern, 4)                       # offset of 4 px across row 30
    rr, cc = np.full(150, 30.0), np.arange(25, 175, dtype=float)
    r = PH.side_correlation(a, rr, cc, 0.0, 5, 8)
    assert abs(abs(r["side_lag_px"]) - 4) <= 1 and r["side_corr"] > 0.9, r
    b = np.tile(pattern, (60, 1))
    r2 = PH.side_correlation(b, rr, cc, 0.0, 5, 8)
    assert r2["side_lag_px"] == 0 and r2["side_corr"] > 0.99
    return "apparent offset detected at the right lag; a continuous pattern correlates at lag 0"


def t_flags_contact_vs_disruption():
    th = {"present_rank": 0.8, "coherence": 0.5, "step_z": 1.0, "disruption_corr": 0.3, "offset_corr": 0.5,
          "offset_lag_px": 2}
    contact = {"magnetic": {"status": "PRESENT", "step_z": 2.0, "side_corr": 0.95, "side_lag_px": 0, "coherence": 0.6},
               "gravity": {"status": "ABSENT", "step_z": 0.1, "side_corr": 0.9, "side_lag_px": 0},
               "terrain": {"status": "UNAVAILABLE"}}
    f = PH.evidence_flags(contact, th)
    assert f["GEOLOGICAL_CONTRAST"] and not f["GEOLOGICAL_DISRUPTION"] and f["alternative_explanation_flag"]
    assert f["unavailable_modalities"] == "terrain"
    fault = {**contact, "gravity": {"status": "PRESENT", "step_z": 0.2, "side_corr": 0.1, "side_lag_px": 0}}
    g = PH.evidence_flags(fault, th)
    assert g["GEOLOGICAL_DISRUPTION"] and not g["alternative_explanation_flag"]
    return "contrast-only -> alternative explanation flagged; disruption recorded; UNAVAILABLE kept apart"


def t_modality_status():
    pc = {"sigmas_px": [1, 3, 8], "thresholds": {"present_rank": 0.8}, "parallel_offset_km": 0.5, "max_lag_px": 3}
    rr, cc = np.full(20, 10.0), np.arange(5, 25, dtype=float)
    assert PH.modality_evidence({"field": None}, rr, cc, 0.0, pc)["status"] == PH.UNAVAILABLE
    w = {"field": np.full((30, 30), np.nan)}
    assert PH.modality_evidence(w, rr, cc, 0.0, pc)["status"] == PH.NOT_OBSERVED
    return "UNAVAILABLE / NOT_OBSERVED never become ABSENT"


def t_localization():
    t = np.zeros((50, 50), bool)
    t[25, 5:45] = True
    p = np.roll(t, 10, 0)                                # 1 km off
    ev = np.ones(t.shape, bool)
    r = PH.localization(p, t, ev, [0.3, 1.0, 3.0])
    assert r["truth_within_0.3km"] == 0.0 and r["truth_within_3km"] == 1.0 and r["corridor_index"] > 1e6
    return "a 1 km offset is a corridor, not a trace"


def t_challenger_cases():
    cc = {"min_reviewed_positive": 2, "min_reviewed_negative": 2, "localization_floor": 0.15}
    rv0 = {"train_pos": 0, "train_neg": 0}
    d = CH.decide(None, rv0, None, None, cc)
    assert d["selected_case"] == "NO_CHALLENGER" and d["conditions"]["CASE_1_ALLOCATION"] == "NOT_EVALUATED"
    d1 = CH.decide({"credible": True, "exportable": True}, rv0, None, None, cc)
    assert d1["selected_case"] == "CASE_1_ALLOCATION"
    d2 = CH.decide({"credible": False}, {"train_pos": 3, "train_neg": 3}, {"decision": "CONTINUE"}, None, cc)
    assert d2["selected_case"] == "CASE_3_TERRAIN"
    loc = pd.DataFrame({"source": ["a", "b"], "truth_within_0.3km": [0.05, 0.1]})
    d4 = CH.decide({"credible": False}, rv0, {"decision": "STOP"}, loc, cc)
    assert d4["selected_case"] == "CASE_4_REPRESENTATION" and not d4["implemented"]
    return "first measured case wins; unmeasured conditions stay NOT_EVALUATED"


def t_verifier_targets():
    cat = pd.DataFrame({"candidate_id": ["a", "b", "c"], "category": ["X"] * 3, "region_id": [0, 1, 2],
                        "review_split": ["TRAIN", "TRAIN", "EVAL"], "distance_to_known_fault_km": [1, 2, 3],
                        "length_km": [1, 1, 1]})
    flags = pd.DataFrame({"candidate_id": ["a", "b", "c"], "category": ["X"] * 3, "MULTI_SUPPORT": [1, 0, 1]})
    lat = pd.DataFrame({"candidate_id": ["a", "b"], "interpretation": ["LIKELY_FAULT", "INSUFFICIENT_EVIDENCE"],
                        "label_class": ["INDEPENDENT_REVIEWED", "UNVERIFIED"]})
    T = CH.verifier_table(flags, pd.DataFrame(), lat, cat, ("LIKELY_CONTACT",))
    assert T.target.iloc[0] == 1.0 and np.isnan(T.target.iloc[1]) and np.isnan(T.target.iloc[2])
    return "unreviewed and insufficient-evidence candidates are excluded, never negatives"


def t_terrain_stop():
    tc = {"min_covered_share": 0.01, "min_unique_structures": 3}
    assert TE.stop_rule(0.001, 10, tc)["decision"] == "STOP"
    assert TE.stop_rule(0.05, 1, tc)["decision"] == "STOP"
    assert TE.stop_rule(0.05, 5, tc)["decision"] == "CONTINUE"
    return "coverage and unique-structure stop rules"


def t_hillshade_and_orientation():
    z = np.tile(np.arange(30, dtype=float), (30, 1))
    h = PH.multi_hillshade(z)
    assert np.nanmin(h) >= 0 and np.nanmax(h) <= 1
    rr, cc = np.arange(20, dtype=float), np.arange(20, dtype=float)
    assert abs(PH.orientation(rr, cc) - np.pi / 4) < 1e-6
    return "hillshade in [0, 1]; 45 deg trace orientation"


TESTS = [("ramp_mass", True, t_ramp_mass), ("evidence_weighted", False, t_evidence_weighted),
         ("no_extrapolation", True, t_no_extrapolation), ("selection_separation", True, t_selection_separation),
         ("review_provenance", True, t_review_provenance), ("side_correlation_offset", True, t_side_correlation_offset),
         ("flags_contact_vs_disruption", True, t_flags_contact_vs_disruption),
         ("modality_status", True, t_modality_status), ("localization", True, t_localization),
         ("challenger_cases", True, t_challenger_cases), ("verifier_targets", True, t_verifier_targets),
         ("terrain_stop", False, t_terrain_stop), ("hillshade_and_orientation", False, t_hillshade_and_orientation)]


def run_selftests() -> pd.DataFrame:
    rows = []
    for name, critical, fn in TESTS:
        try:
            rows.append({"test": name, "critical": critical, "status": "PASS", "detail": fn()})
        except Exception as e:
            rows.append({"test": name, "critical": critical, "status": "FAIL",
                         "detail": f"{type(e).__name__}: {e} | {traceback.format_exc()[-600:]}"})
    return pd.DataFrame(rows)
