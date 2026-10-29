"""Synthetic runtime self-tests of the GeoMoE contracts (no Drive, no data, no GPU).

Run in Colab before any GeoTIFF is produced (stage 12 refuses to export on a critical failure) and by
``tests/test_discovery_geomoe.py``. Every test returns a detail string or raises.
"""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from . import candidates as CA
from . import decide as DE
from . import evaluate as EV
from . import experts as EX
from . import gating as GA
from . import scenarios as SC


def _lines(shape=(80, 80)):
    lab = np.zeros(shape, bool)
    lab[10, 5:70] = True                       # long trace
    lab[10:40, 40] = True                      # branch from a junction
    lab[60, 10:25] = True                      # short isolated fault
    for i in range(30):                        # diagonal trace
        lab[45 + i // 2, 30 + i] = True
    return lab


def _unit(truth, ev, exclude=None, supports=False, known=None):
    from ..gems3.official import Unit

    sl = (slice(0, truth.shape[0]), slice(0, truth.shape[1]))
    return Unit("v2_net", 0, sl, ev, truth, exclude, supports, known)


# =============================================================================== tests
def t_withhold_real_pixels_only():
    lab = _lines()
    zone = np.zeros(lab.shape, bool)
    zone[:, :60] = True
    cc = {"classes": list(SC.V2_CLASSES), "sample_frac": 1.0, "max_events_per_class": 50, "tip_km": [0.3],
          "subsection_km": [0.2], "min_trace_km": 2.0, "short_km": 2.0, "branch_max_km": 4.0}
    keep, wh, eid, E = SC.withhold_network(lab, zone, np.random.default_rng(0), cc)
    assert wh.any(), "nothing withheld"
    assert (wh <= (lab & zone)).all(), "withheld pixel that is not a mapped pixel of the zone"
    assert not (keep & wh).any(), "a pixel is both retained and withheld"
    assert ((keep | wh) == (lab & zone)).all(), "retained + withheld != zone faults"
    assert ((eid >= 0) == wh).all() and set(np.unique(eid[wh])) == set(E.event_id), "event ids inconsistent"
    return f"{len(E)} events, {int(wh.sum())} px withheld; classes {sorted(set(E['class']))}"


def t_leakage_checks_detect_overlap():
    t = np.zeros((10, 10), bool)
    t[5, 2:8] = True
    k = np.zeros_like(t)
    k[1, :] = True
    ok = SC.leakage_checks("x", t, k, k, t | k)
    assert all(c["ok"] for c in ok), ok
    k[5, 3] = True
    bad = SC.leakage_checks("x", t, k, None)
    assert not bad[0]["ok"], "known/truth overlap not detected"
    return "overlap detected, clean case passes"


def t_official_metric_equivalence():
    from ..metric import OFFICIAL_EXCLUDE_SUPPORTS_TP, dti_components

    rng = np.random.default_rng(1)
    lab = _lines()
    ev = np.ones(lab.shape, bool)
    ex = np.zeros(lab.shape, bool)
    ex[10, 5:40] = True
    truth = lab & ~ex
    u = _unit(truth, ev, ex, bool(OFFICIAL_EXCLUDE_SUPPORTS_TP))
    q = (rng.random(lab.shape) > 0.97).astype(np.float32)
    q[np.roll(truth, 1, 0)] = 1.0
    a = u.components(q)
    b = dti_components(q, truth, eval_mask=ev, exclude_mask=ex, exclude_supports_tp=bool(OFFICIAL_EXCLUDE_SUPPORTS_TP))
    for k in ("tp", "fp", "fn"):
        assert abs(getattr(a, k) - getattr(b, k)) < 1e-6, (k, getattr(a, k), getattr(b, k))
    d = a.as_dict()
    from ..p6.boost import dti as bdti

    assert abs(EV.dti(d["tp_w"], d["fp_w"], d["fn_w"]) - bdti(d["tp_w"], d["fp_w"], d["fn_w"])) < 1e-12
    return f"FastScorer == metric.dti_components (tp {a.tp:.3f}, fp {a.fp:.3f}, fn {a.fn:.3f})"


def t_banded_components_sum():
    lab = _lines()
    ev = np.ones(lab.shape, bool)
    u = _unit(lab, ev, None, True)
    q = np.roll(lab, 2, 1).astype(np.float32)
    dist = np.tile(np.linspace(0, 8, lab.shape[1]), (lab.shape[0], 1))
    B = SC.banded_components(u, q, dist, [1.0, 5.0])
    c = u.components(q).as_dict()
    for k in ("tp_w", "fp_w", "fn_w"):
        assert abs(B[k].sum() - c[k]) < 1e-6, (k, B[k].sum(), c[k])
    return "near + mid + far == unit components"


def t_known_fault_masking():
    from ..metric import OFFICIAL_EXCLUDE_SUPPORTS_TP

    lab = _lines()
    ex = np.zeros(lab.shape, bool)
    ex[10, :] = lab[10, :]
    truth = lab & ~ex
    u = _unit(truth, np.ones(lab.shape, bool), ex, bool(OFFICIAL_EXCLUDE_SUPPORTS_TP))
    on_known = ex.astype(np.float32)
    c = u.components(on_known).as_dict()
    assert c["fp_w"] == 0.0, "a prediction on a masked mapped fault was penalised"
    return f"predictions on masked known faults: fp {c['fp_w']}, tp {c['tp_w']:.3f} (supports_tp=" \
           f"{bool(OFFICIAL_EXCLUDE_SUPPORTS_TP)})"


def t_split_and_dedup():
    m = np.zeros((40, 120), bool)
    m[20, 5:110] = True
    lab = CA.split_long(m.astype(np.int32), 30)
    sizes = np.bincount(lab.ravel())[1:]
    assert sizes.max() <= 31 and int(lab.max()) >= 4, sizes
    pools = {"w2": m, "g02c": np.roll(m, 1, 0), "mag_hgrad": np.zeros_like(m)}
    pools["mag_hgrad"][5:35, 60] = True                           # a crossing line is never a duplicate
    L, D = CA.inventory(pools, {"dedup": {"tol_px": 2, "min_share": 0.6, "max_angle_deg": 20},
                                "max_segment_km": 3.0, "min_segment_px": 2})
    assert D[D.source == "g02c"].duplicate.all(), "parallel copy not deduplicated"
    assert not D[D.source == "mag_hgrad"].duplicate.any(), "crossing line dropped"
    return f"{int(L.max())} segments after dedup + split"


def t_candidate_provenance():
    m = np.zeros((30, 30), bool)
    m[15, 2:28] = True
    lab = m.astype(np.int32)
    T = CA.segment_table(lab, {"w2": np.ones(m.shape)}, {"w2": m, "g02c": m.copy()}, 3, np.ones(m.shape, bool), m)
    assert T.n_proposers.iloc[0] == 2 and set(T.proposed_by.iloc[0].split(";")) == {"w2", "g02c"}
    assert T.y_near_truth.iloc[0] == 1.0
    return "proposed_by keeps every source with a line on the segment"


def t_oracle_accounting():
    lab = np.zeros((20, 20), np.int32)
    lab[5, 2:10] = 1
    lab[15, 2:10] = 2
    truth = np.zeros((20, 20), bool)
    truth[6, 2:10] = True
    o = CA.oracle_order(lab, truth, 3)
    assert list(o) == [1], o
    px = pd.DataFrame({"r": [6] * 8, "c": list(range(2, 10)), "system_id": [7] * 8, "known_dist_km": 0.0})
    C = CA.system_coverage(px, {"a": lab == 1, "b": lab == 2}, 3, 0.5)
    assert C.set_index("pool").covered.to_dict() == {"a": True, "b": False}
    try:
        EV.guard_oracle(pd.DataFrame({"role": ["heldout", "oracle"]}))
    except ValueError:
        return "oracle selects only truth-supported segments; oracle rows refused by selection"
    raise AssertionError("oracle row not refused")


def t_missing_expert_never_zero():
    S = np.array([[0.9, np.nan, 0.1, np.nan], [np.nan, np.nan, np.nan, np.nan], [0.5, 0.5, 0.5, 0.5]])
    sc, eff = GA.combine(S, np.array([0.25, 0.25, 0.25, 0.25]))
    assert abs(sc[0] - 0.5) < 1e-12, "missing experts were averaged as zeros"
    assert np.isnan(sc[1]) and (eff[1] == 0).all(), "a candidate without experts got a score"
    assert np.allclose(eff[[0, 2]].sum(1), 1.0) and eff[0, 1] == 0 and eff[0, 3] == 0
    T = pd.DataFrame({"mean_dem": [np.nan, 0.8], "fin_dem": [0.0, 1.0], "mean_network": [0.2, 0.2],
                      "fin_network": [1.0, 1.0]})
    R = EX.expert_raw(T, {"E4_terrain_network": ["network", "dem"]})
    assert abs(R.raw_E4_terrain_network.iloc[0] - 0.2) < 1e-12, "absent DEM entered the terrain expert"
    try:
        GA.combine(S, np.array([1.0, -0.1, 0.0, 0.1]))
    except ValueError:
        return "missing experts renormalised away; DEM absence is not evidence"
    raise AssertionError("negative weight accepted")


def t_crossfit_normalisation():
    rng = np.random.default_rng(3)
    tr, held = rng.normal(0, 1, 500), rng.normal(5, 1, 200)
    k = EX.fit_ecdf(tr)
    k2 = EX.fit_ecdf(tr.copy())
    a = EX.apply_ecdf(k, held)
    assert np.array_equal(k, k2) and (a >= 0).all() and (a <= 1).all() and a.mean() > 0.95, "held-out leak?"
    v = EX.apply_ecdf(k, np.array([np.nan, -10, 10]))
    assert np.isnan(v[0]) and v[1] == 0.0 and v[2] == 1.0
    assert np.isnan(EX.apply_ecdf(None, np.array([1.0]))).all()
    return "knots depend on training values only; NaN preserved; shifted held-out zone saturates (not refitted)"


def t_gate_weights_and_gradient():
    rng = np.random.default_rng(4)
    n, d, E = 200, 3, 4
    Z = rng.normal(size=(n, d))
    S = rng.random((n, E))
    S[::5, 3] = np.nan
    y = (rng.random(n) > 0.7).astype(float)
    w = np.ones(n)
    prior = np.full(E, 0.25)
    th = rng.normal(0, 0.3, E * d + E + 2)
    _, _, g, _ = GA.gate_forward(th, Z, S, prior)
    assert np.allclose(g.sum(1), 1.0) and (g >= 0).all() and (g[::5, 3] == 0).all(), "gate weights invalid"
    J, grad = GA.gate_objective(th, Z, S, y, w, prior, 0.01, 0.05)
    num = np.zeros_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th)
        e[i] = 1e-6
        num[i] = (GA.gate_objective(th + e, Z, S, y, w, prior, 0.01, 0.05)[0] -
                  GA.gate_objective(th - e, Z, S, y, w, prior, 0.01, 0.05)[0]) / 2e-6
    err = float(np.max(np.abs(num - grad)) / max(np.max(np.abs(num)), 1e-9))
    assert err < 1e-4, f"analytic gradient off by {err}"
    return f"weights >= 0, sum 1, 0 on missing experts; gradient rel. error {err:.1e}"


def t_collapse_detection():
    S = np.ones((100, 4))
    g = np.zeros((100, 4))
    g[:, 0] = 0.97
    g[:, 1:] = 0.01
    assert GA.collapse(g, S, {"mean_max_weight": 0.9, "dominant_share": 0.9})["collapsed"]
    g2 = np.full((100, 4), 0.25)
    assert not GA.collapse(g2, S, {"mean_max_weight": 0.9, "dominant_share": 0.9})["collapsed"]
    return "collapse flagged on a one-expert gate, not on a uniform gate"


def t_pu_targets_and_training_rows():
    T = pd.DataFrame({"y_near_truth": [0.9, 0.1, 0.0, 0.6], "n_px": [10, 10, 20, 5], "zone": [0, 0, 1, 2],
                      "row_full": [1, 2, 8, 9], "col_full": [1, 2, 8, 9]})
    y, w = GA.pu_targets(T, 0.5, 0.3)
    assert list(y) == [1, 0, 0, 1] and np.allclose(w, [10, 3, 6, 5]), (y, w)
    zm = np.zeros((10, 10), bool)
    zm[8:, 8:] = True                                             # held zone 1 + buffer
    m = GA.training_rows(T, 1, zm)
    assert list(m) == [True, True, False, False], m
    try:
        GA.assert_disjoint([0, 1], 1)
    except GA.LeakageError:
        return "unlabelled = down-weighted (never confirmed negative); held zone + buffer excluded"
    raise AssertionError("held-out zone in training not refused")


def t_raster_and_budget():
    lab = np.zeros((30, 30), np.int32)
    lab[5, 2:20], lab[10, 2:8], lab[20, 2:28] = 1, 2, 3
    sm = EV.score_map(lab, np.array([1, 2, 3]), np.array([0.5, 0.5, 0.9]), np.array([18, 6, 26]))
    assert np.isnan(sm[lab == 0]).all(), "non-candidate pixels selectable"
    assert sm[20, 5] > sm[5, 5] > sm[10, 5], "tie-break / ordering wrong"
    tied = EV.score_map(lab, np.array([1, 2, 3]), np.ones(3), np.array([6, 6, 6]))
    assert len({float(tied[5, 5]), float(tied[10, 5]), float(tied[20, 5])}) == 3, "exact ties not broken"
    from ..gems3.representation import top_mask

    ev = np.ones(lab.shape, bool)
    q = top_mask(sm, ev, 30 / ev.size)
    assert (q <= (lab > 0)).all(), "non-candidate pixel selected"
    assert q.sum() <= 30 + 18 and q[lab == 3].all() and (q[lab == 1].all() or not q[lab == 1].any()), \
        "more than one segment of overshoot, or a partial tied segment"
    R = EV.budget_rows(pd.DataFrame({"area_frac_eligible": [0.0101, 0.0103], "frac": [0.01, 0.01]}), 0.02)
    assert list(R.budget_ok) == [True, False]
    return f"{int(q.sum())} px for a 30 px budget: candidates only, whole segments, <= one segment overshoot"


def t_rule_gate():
    S = np.array([[0.2, 0.9, 0.3, 0.1], [0.9, 0.85, 0.1, np.nan]])
    D = pd.DataFrame({"q_mean_mag_hgrad": [0.95, 0.1], "q_mean_grav_hgrad": [0.1, 0.1], "q_mean_g02c": [0.1, 0.1],
                      "q_mean_g02a": [0.1, 0.1], "length_km": [5.0, 1.0], "min_tip_dist_km": [9.0, 9.0],
                      "mean_tip_along_cos": [0.0, 0.0], "mean_dem_observed": [0.0, 0.0]})
    rc = {"factor": 2.0, "R1_terrain_scarp": {"dem_q": 0.8}, "R2_structural_displacement": {"pf_q": 0.7,
          "g02a_q": 0.7}, "R3_tip_continuation": {"tip_km": 1.0, "tip_cos": 0.7},
          "R4_magnetic_edge_only": {"mag_q": 0.8, "grav_q_max": 0.5, "g02c_q_max": 0.5},
          "R5_short_multisource": {"short_km": 2.0, "min_agree": 2, "agree_q": 0.8}}
    M = GA.rule_masks(D, rc)
    w = np.full(4, 0.25)
    s0, _ = GA.rule_gate(S, w, M, (), rc)
    s1, e1 = GA.rule_gate(S, w, M, ("R4_magnetic_edge_only",), rc)
    assert e1[0, 1] < 0.25 and s1[0] < s0[0], "R4 did not down-weight the magnetic-only edge"
    s5, _ = GA.rule_gate(S, w, M, ("R5_short_multisource",), rc)
    assert s5[1] > s0[1] and abs(s5[1] - 0.875) < 1e-9, "R5 did not preserve the short multi-source candidate"
    assert len(GA.rule_variants()) == 1 + len(GA.RULES)
    return "R4 down-weights, R5 preserves; one ablation per rule"


def t_selection_falsifiable():
    rows = []
    for sc in ("V1", "V2", "V3"):
        rows += [{"scenario": sc, "model": "A_pixel", "dti": 0.20, "worst_zone_dti": 0.15},
                 {"scenario": sc, "model": "B", "dti": 0.22, "worst_zone_dti": 0.16},
                 {"scenario": sc, "model": "D", "dti": 0.222, "worst_zone_dti": 0.16}]
    S = pd.DataFrame(rows)
    P = pd.DataFrame([{"scenario": sc, "method": m, "reference": r, "p_delta_positive": 0.9, "ci_lo": 0, "ci_hi": 0}
                      for sc in ("V1", "V2", "V3") for m, r in (("B", "A_pixel"), ("D", "A_pixel"), ("D", "B"))])
    sel = {"min_gain_vs_w2": 0.005, "min_scenarios_improved": 2, "max_scenario_drop": 0.01,
           "max_worst_zone_drop": 0.01, "min_moe_gain_vs_static": 0.01, "min_p_positive": 0.8, "moe_min_scenarios": 2,
           "rule_min_gain_vs_static": 0.005}
    hard = {"no_leakage": True}
    a = DE.select_model(S, P, hard, False, sel)
    assert a["selected"] == "B", a                                 # tiny MoE gain -> simpler model
    S2 = S.assign(dti=np.where(S.model == "A_pixel", 0.30, S.dti))
    assert DE.select_model(S2, P, hard, False, sel)["selected"] == DE.RETAIN_W2
    assert DE.select_model(S, P, {"no_leakage": False}, False, sel)["status"] == "BLOCKED"
    S3 = S.assign(dti=np.where(S.model == "D", 0.25, S.dti))
    assert DE.select_model(S3, P, hard, False, sel)["selected"] == "D"
    assert DE.select_model(S3, P, hard, True, sel)["selected"] == "B"   # collapsed gate never promoted
    return "tiny MoE gain -> B; W2 better -> RETAIN_W2; leakage -> BLOCKED; collapsed D never promoted"


def t_coverage_gate():
    gc = {"min_pool_coverage": 0.6, "min_ranking_gap": 0.2, "min_unique_systems": 3, "min_unique_share": 0.05}
    S = pd.DataFrame([{"scenario": "V1", "n_systems": 40, "cov_inventory": 0.4, "w2_recovered_share": 0.3,
                       "independent_not_E1": 1},
                      {"scenario": "V2", "n_systems": 40, "cov_inventory": 0.5, "w2_recovered_share": 0.3,
                       "independent_not_E1": 0},
                      {"scenario": "V3", "n_systems": 40, "cov_inventory": 0.9, "w2_recovered_share": 0.4,
                       "independent_not_E1": 6}])
    d = DE.coverage_decision(S, gc, "V1")
    assert d["overall"] == DE.GENERATION and d["per_scenario"]["V3"]["verdict"] == DE.RANKING
    assert d["geology_unique"] and not d["learned_gate_allowed"]
    return "GENERATION majority blocks the learned gate; unique independent coverage detected"


def t_readback_contract():
    from ..leaderboard_focus.factory import readback_checks, verdict

    a = np.zeros((4, 5), np.float32)
    a[0, 0] = np.nan
    a[2, 2] = 1.0
    tpl = {"height": 4, "width": 5, "transform": [100.0, 0.0, 0.0, 0.0, -100.0, 0.0]}
    facts = {"array": a, "count": 1, "dtype": "float32", "epsg": 32611, "res": (100.0, 100.0), "shape": (4, 5),
             "transform": [100.0, 0.0, 0.0, 0.0, -100.0, 0.0], "nodata": float("nan")}
    exp = {"count": 1, "dtype": "float32", "epsg": 32611, "res_m": 100}
    ok = readback_checks(facts, tpl, exp, 0.1, 10.0, "s", "s", "f", "f", "h")
    assert verdict(ok) == "PASS", [c for c in ok if not c["ok"]]
    bad = readback_checks({**facts, "epsg": 4326}, tpl, exp, 0.1, 10.0, "s", "t", "f", "f", "h")
    assert verdict(bad) == "FAIL"
    return "factory read-back contract: PASS on a valid band, FAIL on CRS / sha mismatch"


TESTS = [("withhold_real_pixels_only", True, t_withhold_real_pixels_only),
         ("leakage_checks_detect_overlap", True, t_leakage_checks_detect_overlap),
         ("official_metric_equivalence", True, t_official_metric_equivalence),
         ("banded_components_sum", True, t_banded_components_sum),
         ("known_fault_masking", True, t_known_fault_masking),
         ("split_and_dedup", False, t_split_and_dedup),
         ("candidate_provenance", False, t_candidate_provenance),
         ("oracle_accounting", True, t_oracle_accounting),
         ("missing_expert_never_zero", True, t_missing_expert_never_zero),
         ("crossfit_normalisation", True, t_crossfit_normalisation),
         ("gate_weights_and_gradient", True, t_gate_weights_and_gradient),
         ("collapse_detection", False, t_collapse_detection),
         ("pu_targets_and_training_rows", True, t_pu_targets_and_training_rows),
         ("raster_and_budget", True, t_raster_and_budget),
         ("rule_gate", False, t_rule_gate),
         ("selection_falsifiable", True, t_selection_falsifiable),
         ("coverage_gate", True, t_coverage_gate),
         ("readback_contract", True, t_readback_contract)]


def run_selftests() -> pd.DataFrame:
    rows = []
    for name, critical, fn in TESTS:
        try:
            rows.append({"test": name, "critical": critical, "status": "PASS", "detail": fn()})
        except Exception as e:
            rows.append({"test": name, "critical": critical, "status": "FAIL",
                         "detail": f"{type(e).__name__}: {e} | {traceback.format_exc()[-600:]}"})
    return pd.DataFrame(rows)
