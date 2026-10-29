"""Synthetic tests of the discovery-first GeoMoE (gems_bench/geomoe). No data, no Drive, no GPU.

Run in Colab by notebooks/colab/GEMS_DISCOVERY_GEOMOE.ipynb (cell 3) before any export. Written, not run locally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from gems_bench.geomoe import candidates as CA  # noqa: E402
from gems_bench.geomoe import decide as DE  # noqa: E402
from gems_bench.geomoe import evaluate as EV  # noqa: E402
from gems_bench.geomoe import experts as EX  # noqa: E402
from gems_bench.geomoe import gating as GA  # noqa: E402
from gems_bench.geomoe import run as RN  # noqa: E402
from gems_bench.geomoe import scenarios as SC  # noqa: E402
from gems_bench.geomoe import selftest as STT  # noqa: E402


def test_runtime_selftests_all_pass():
    T = STT.run_selftests()
    bad = T[T.status != "PASS"]
    assert not len(bad), bad.to_dict("records")


# ------------------------------------------------------------------------------- geographic / discovery leakage
def test_training_rows_never_contain_the_held_out_zone():
    T = pd.DataFrame({"zone": [0, 1, 2, 1], "row_full": [0, 1, 2, 3], "col_full": [0, 1, 2, 3]})
    m = GA.training_rows(T, 1, None)
    assert not m[T.zone.to_numpy() == 1].any()
    with pytest.raises(GA.LeakageError):
        GA.assert_disjoint(T.zone[np.ones(4, bool)].unique(), 1)


def test_buffer_drops_training_candidates_near_the_held_out_zone():
    T = pd.DataFrame({"zone": [0, 0], "row_full": [1, 9], "col_full": [1, 9]})
    zm = np.zeros((10, 10), bool)
    zm[8:, 8:] = True
    assert list(GA.training_rows(T, 1, zm)) == [True, False]


def test_v2_withheld_pixels_never_in_known_context_or_exclusion():
    lab = np.zeros((60, 60), bool)
    lab[20, 5:55] = True
    lab[20:45, 30] = True
    zone = np.ones(lab.shape, bool)
    cc = {"classes": list(SC.V2_CLASSES), "sample_frac": 1.0, "max_events_per_class": 20, "tip_km": [0.3],
          "subsection_km": [0.2], "min_trace_km": 2.0, "short_km": 1.0, "branch_max_km": 3.0}
    keep, wh, _, E = SC.withhold_network(lab, zone, np.random.default_rng(5), cc)
    assert wh.any() and not (keep & wh).any()
    checks = SC.leakage_checks("V2:0", wh, keep, keep, lab)
    assert all(c["ok"] for c in checks)
    assert set(E["class"]) <= set(SC.V2_CLASSES) and "lateral_shift" not in set(E["class"])


def test_v2_withholding_is_deterministic_under_the_seed():
    lab = np.zeros((50, 50), bool)
    lab[10, 2:48] = True
    lab[30, 2:30] = True
    cc = {"classes": list(SC.V2_CLASSES), "sample_frac": 0.5, "max_events_per_class": 10, "tip_km": [0.3, 0.5],
          "subsection_km": [0.2], "min_trace_km": 2.0, "short_km": 1.0, "branch_max_km": 3.0}
    a = SC.withhold_network(lab, np.ones(lab.shape, bool), np.random.default_rng(7), cc)[1]
    b = SC.withhold_network(lab, np.ones(lab.shape, bool), np.random.default_rng(7), cc)[1]
    assert np.array_equal(a, b)


def test_faults_outside_the_zone_are_untouched():
    lab = np.zeros((40, 40), bool)
    lab[10, :] = True
    zone = np.zeros(lab.shape, bool)
    zone[:, :20] = True
    cc = {"classes": ["tip_extension", "subsection"], "sample_frac": 1.0, "max_events_per_class": 10,
          "tip_km": [0.5], "subsection_km": [0.3], "min_trace_km": 1.0, "short_km": 0.5, "branch_max_km": 1.0}
    keep, wh, _, _ = SC.withhold_network(lab, zone, np.random.default_rng(0), cc)
    assert not wh[:, 20:].any() and not keep[:, 20:].any()


# ------------------------------------------------------------------------------- masking / metric
def test_banded_components_partition_the_official_components():
    assert "PASS" == STT.run_selftests().set_index("test").status["banded_components_sum"]


def test_official_metric_equivalence_and_known_masking():
    T = STT.run_selftests().set_index("test").status
    assert T["official_metric_equivalence"] == "PASS" and T["known_fault_masking"] == "PASS"


def test_dti_expression_matches_boost():
    from gems_bench.p6.boost import dti

    for tp, fp, fn in ((3.0, 1.0, 2.0), (0.0, 5.0, 1.0), (10.0, 0.0, 0.0)):
        assert abs(EV.dti(tp, fp, fn) - dti(tp, fp, fn)) < 1e-12


# ------------------------------------------------------------------------------- candidates
def test_candidate_dedup_keeps_crossings_and_short_segments():
    m = np.zeros((40, 40), bool)
    m[20, 5:35] = True
    short = np.zeros_like(m)
    short[5, 5:8] = True
    cross = np.zeros_like(m)
    cross[5:35, 20] = True
    lab, D = CA.inventory({"w2": m, "fusion": np.roll(m, 1, 0), "g02c": cross, "mag_hgrad": short},
                          {"dedup": {"tol_px": 2, "min_share": 0.6, "max_angle_deg": 20}, "max_segment_km": 10.0,
                           "min_segment_px": 2})
    assert D[D.source == "fusion"].duplicate.all()
    assert not D[D.source.isin(["g02c", "mag_hgrad"])].duplicate.any()
    assert (lab[5, 5:8] > 0).all()


def test_split_long_bounds_segment_length():
    m = np.zeros((10, 200), np.int32)
    m[5, 2:190] = 1
    lab = CA.split_long(m, 25)
    assert np.bincount(lab.ravel())[1:].max() <= 26


def test_connectivity_counts_touching_segments():
    lab = np.zeros((10, 10), np.int32)
    lab[5, 1:5], lab[5, 5:9], lab[1:5, 5] = 1, 2, 3
    deg = CA.connectivity(lab)
    assert deg[0] >= 1 and deg[1] >= 1 and deg[2] >= 1


def test_oracle_coverage_accounting_and_unique_families():
    px = pd.DataFrame({"r": [1, 1, 1, 8, 8], "c": [1, 2, 3, 8, 9], "system_id": [1, 1, 1, 2, 2],
                       "known_dist_km": 0.0})
    a = np.zeros((10, 10), bool)
    a[1, 1:4] = True
    b = np.zeros((10, 10), bool)
    b[8, 8:10] = True
    C = CA.system_coverage(px, {"fam:E1_learned": a, "fam:E2_potential": b, "fam:E3_structural": a,
                                "fam:E4_terrain_network": np.zeros_like(a)}, 0, 0.5)
    U = CA.unique_coverage(C, list(EX.FAMILIES)).set_index("system_id")
    assert U.loc[2, "unique_E2_potential"] and not U.loc[1, "unique_E1_learned"]


# ------------------------------------------------------------------------------- experts / normalisation
def test_missing_expert_availability_is_explicit():
    T = pd.DataFrame({"mean_g02c": [np.nan, 0.4], "fin_g02c": [0.0, 1.0]})
    R = EX.expert_raw(T, {"E3_structural": ["g02c", "g02a"]})
    assert not R.avail_E3_structural.iloc[0] and R.avail_E3_structural.iloc[1]
    A = EX.availability_table(pd.concat([T.assign(scenario="V1", zone=0), R], axis=1), {"E3_structural": ["g02c",
                                                                                                         "g02a"]},
                              ["scenario", "zone"])
    assert (A[(A.member == "g02a")].status == "ABSENT").all()


def test_crossfitted_normalisation_uses_training_rows_only():
    T = pd.DataFrame({"raw_E1_learned": [0.1, 0.2, 0.3, 50.0], "zone": [0, 0, 0, 1]})
    n_tr = EX.fit_normalizer(T[T.zone == 0], ["raw_E1_learned"], 11)
    n_all = EX.fit_normalizer(T, ["raw_E1_learned"], 11)
    assert not np.array_equal(n_tr["raw_E1_learned"], n_all["raw_E1_learned"])
    v = EX.normalize(T, n_tr)
    assert v.raw_E1_learned.iloc[3] == 1.0                          # held value saturates, never refits


def test_network_evidence_bounded_and_directional():
    v = EX.network_evidence(np.array([0.0, 0.0, 10.0]), np.array([1.0, -1.0, 1.0]), 2.5)
    assert v[0] == 1.0 and v[1] == 0.0 and 0 < v[2] < 0.05


# ------------------------------------------------------------------------------- gating
def test_gate_weights_normalised_and_masked():
    S = np.array([[0.2, np.nan, 0.5, 0.9]])
    sc, eff = GA.combine(S, np.array([1.0, 1.0, 1.0, 1.0]))
    assert abs(eff.sum() - 1.0) < 1e-12 and eff[0, 1] == 0.0


def test_learned_gate_starts_at_static_fusion_and_does_not_collapse_without_signal():
    rng = np.random.default_rng(0)
    n = 600
    S = rng.random((n, 4))
    Z = rng.normal(size=(n, 3))
    y = (rng.random(n) > 0.8).astype(np.int8)                        # labels unrelated to Z and S
    w = np.ones(n)
    prior = np.array([0.4, 0.2, 0.2, 0.2])
    m = GA.fit_gate(Z, S, y, w, prior, {"l2": 0.01, "kl_to_static": 0.5, "max_iter": 200})
    _, g = GA.gate_predict(m, Z, S)
    assert np.allclose(g.sum(1), 1.0)
    assert not GA.collapse(g, S, {"mean_max_weight": 0.9, "dominant_share": 0.9})["collapsed"]
    assert abs(g.mean(0)[0] - 0.4) < 0.15                            # stays near the static prior


def test_pu_targets_never_create_confirmed_negatives():
    T = pd.DataFrame({"y_near_truth": [0.0, 1.0], "n_px": [10, 10]})
    y, w = GA.pu_targets(T, 0.5, 0.3)
    assert w[0] < w[1] and w[0] > 0


def test_static_fit_prefers_the_informative_expert():
    rng = np.random.default_rng(2)
    n = 400
    y = (rng.random(n) > 0.7).astype(int)
    S = np.column_stack([y + rng.normal(0, 0.1, n), rng.random(n), rng.random(n), rng.random(n)])
    f = GA.fit_static(S, y, np.ones(n), 0.25)
    assert f["weights"][0] >= 0.5


def test_rule_variants_ablate_each_rule_once():
    v = GA.rule_variants()
    assert v["C_all"] == GA.RULES
    for r in GA.RULES:
        assert r not in v[f"C_minus_{r.split('_')[0]}"]


# ------------------------------------------------------------------------------- fair budgets / raster
def test_every_model_gets_the_same_budget_rule():
    R = EV.budget_rows(pd.DataFrame({"model": ["A_submitted_W2", "B"], "area_frac_eligible": [0.0100, 0.0100],
                                     "frac": [0.01, 0.01]}), 0.02)
    assert R.budget_ok.all()


def test_score_map_has_no_duplicate_mass():
    lab = np.zeros((5, 5), np.int32)
    lab[1, :] = 1
    lab[3, :] = 2
    sm = EV.score_map(lab, np.array([2, 1]), np.array([0.9, 0.1]), np.array([5, 5]))
    assert np.isnan(sm[0]).all() and sm[3, 0] > sm[1, 0]


def test_oracle_rows_refused_by_selection_tables():
    with pytest.raises(ValueError):
        EV.guard_oracle(pd.DataFrame({"role": ["oracle"]}))


# ------------------------------------------------------------------------------- decisions
def test_selection_can_reject_the_moe_and_retain_w2():
    T = STT.run_selftests().set_index("test").status
    assert T["selection_falsifiable"] == "PASS" and T["coverage_gate"] == "PASS"


def test_next_action_is_single_and_follows_the_tree():
    cov = {"overall": DE.GENERATION, "geology_unique": False}
    M = pd.DataFrame({"subtype": ["tip_extension", "branch"], "length_bin": ["<1km", "1-2km"], "length_km": [5, 1]})
    a = DE.next_action(cov, {"selected": DE.RETAIN_W2}, None, {"no_leakage": True}, M)
    assert "tip_extension" in a["action"] and isinstance(a["action"], str)
    b = DE.next_action(cov, {"selected": "B"}, {"status": "READY_FOR_UPLOAD", "file": "x.tif"}, {}, M)
    assert "x.tif" in b["action"]
    c = DE.next_action(cov, {}, None, {"no_leakage": False}, M)
    assert "Repair" in c["action"]


# ------------------------------------------------------------------------------- configuration
def test_notebook_config_guards():
    with pytest.raises(RN.ConfigError):
        RN.notebook_config({"device": "cuda"})
    with pytest.raises(RN.ConfigError):
        RN.notebook_config({"export": {"manual_upload": False}})
    with pytest.raises(RN.ConfigError):
        RN.notebook_config({"experts": {"dem": "download"}})
    c = RN.notebook_config({"mode": "resume"})
    assert c["mode"] == "RESUME" and c["reuse_cached"]


def test_yaml_config_loads_and_is_bounded():
    pytest.importorskip("yaml")
    c = RN.load_config()
    assert c["primary"] in c["scenarios"] and set(c["experts"]) == set(EX.FAMILIES)
    assert "lateral_shift" not in c["v2"]["classes"]
    assert len(GA.simplex_grid(len(EX.FAMILIES), c["static"]["grid_step"])) == 35


# ------------------------------------------------------------------------------- full-region parity / readback
def test_deployment_parity_spearman():
    from gems_bench.geomoe.deployment import spearman

    a = np.arange(10.0)
    assert spearman(a, a) == pytest.approx(1.0) and spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(spearman(np.array([1.0, np.nan]), np.array([1.0, 2.0])))


def test_submission_readback_contract():
    pytest.importorskip("rasterio")
    assert STT.run_selftests().set_index("test").status["readback_contract"] == "PASS"


# ------------------------------------------------------------------------------- dgm.2 regressions / enforcement
def test_regression_unique_coverage_colab_typeerror():
    """dgm.1-r1 Colab: TypeError DataFrame.any() takes 1 positional argument (F[others].any(1))."""
    px = pd.DataFrame({"r": [1, 1, 1, 8, 8], "c": [1, 2, 3, 8, 9], "system_id": [1, 1, 1, 2, 2],
                       "known_dist_km": 0.0})
    a = np.zeros((10, 10), bool)
    a[1, 1:4] = True
    b = np.zeros((10, 10), bool)
    b[8, 8:10] = True
    C = CA.system_coverage(px, {"fam:E1_learned": a, "fam:E2_potential": b, "fam:E3_structural": a,
                                "fam:E4_terrain_network": np.zeros_like(a)}, 0, 0.5)
    U = CA.unique_coverage(C, list(EX.FAMILIES))
    assert set(U.system_id) == {1, 2} and U.set_index("system_id").loc[2, "unique_E2_potential"]
    assert {"n_px", "near_share"} <= set(C.columns)


def test_band_coverage_sums_to_truth_pixels():
    px = pd.DataFrame({"r": [1, 2, 3], "c": [1, 1, 1], "system_id": [1, 1, 1], "known_dist_km": [0.5, 3.0, 9.0]})
    m = np.zeros((5, 5), bool)
    m[1, 1] = True
    B = CA.band_coverage(px, {"inventory": m}, 0, [1.0, 5.0])
    assert B.truth_px.sum() == 3 and B[B.band == "near"].covered_px.iloc[0] == 1


def test_inventory_view_exposes_named_columns_and_absent_truncation():
    T = pd.DataFrame({"unit": ["V1:0"], "label": [1], "scenario": ["V1"], "zone": [0], "theta": [0.0],
                      "row_full": [1.0], "col_full": [2.0], "n_px": [5], "length_km": [0.56], "straightness": [1.0],
                      "connectivity": [0], "proposed_by": ["w2"], "mean_w2": [0.9], "on_w2": [1.0]})
    V = CA.inventory_view(T)
    assert V.candidate_id.iloc[0] == "V1:0#1" and V.w2_score.iloc[0] == 0.9
    assert np.isnan(V.g02_truncation.iloc[0]) and "y_near_truth" not in V


def test_stage_dependencies_block_and_never_conclude():
    st = {"00_tests": DE.COMPLETE, "03_scenarios": DE.COMPLETE, "04_candidates": DE.FAILED}
    for s, want in (("05_oracle", DE.BLOCKED_UPSTREAM), ("07_experts", DE.BLOCKED_UPSTREAM)):
        assert DE.dependency_status(s, st)[0] == want
    assert DE.dependency_status("06_decision", {"05_oracle": DE.BLOCKED_UPSTREAM})[0] == DE.NOT_EVALUATED
    assert DE.dependency_status("11_deployment", {"10b_compare": DE.FAILED})[0] == DE.NOT_EVALUATED
    assert DE.dependency_status("05_oracle", st, strict=False)[0] is None


def test_unmeasured_coverage_is_not_a_negative_geology_result():
    d = DE.coverage_decision(pd.DataFrame(), {"min_pool_coverage": 0.6, "min_ranking_gap": 0.2,
                                              "min_unique_systems": 3, "min_unique_share": 0.05})
    assert d["overall"] == DE.NOT_EVALUATED and d["geology_unique"] is None and not d["learned_gate_allowed"]


def test_pytest_failure_is_enforced():
    T = pd.DataFrame([{"test": "a", "critical": True, "status": "PASS"}])
    st, why = RN.tests_verdict(T, 1, "FAILED tests/test_discovery_geomoe.py::test_oracle - TypeError")
    assert st == DE.TESTS_FAILED and any("test_oracle" in w for w in why)


def test_new_config_keys_map_onto_old_ones_and_neural_training_refused():
    c = RN.notebook_config({"mode": "FAST_DIAGNOSTIC", "dem": "off", "export_validated_candidate": False,
                            "allow_exploratory_export": True})
    assert c["experts"]["dem"] == "off" and not c["export"]["enabled"] and c["export"]["exploratory"]
    old = RN.notebook_config({"mode": "FULL", "export": {"exploratory": True}})
    assert old["export"]["exploratory"] and old["mode"] == "FULL"
    with pytest.raises(RN.ConfigError):
        RN.notebook_config({"train_neural_models": True})


def test_v2_subtype_components_partition_truth():
    from gems_bench.gems3.official import Unit

    truth = np.zeros((20, 20), bool)
    truth[5, 2:8] = True
    truth[12, 2:8] = True
    sysmap = np.full(truth.shape, -1, np.int32)
    sysmap[5, 2:8], sysmap[12, 2:8] = 0, 1
    ev = np.ones(truth.shape, bool)
    u = Unit("v2_net", 0, (slice(0, 20), slice(0, 20)), ev, truth, np.zeros_like(truth), False, None)
    E = pd.DataFrame({"event_id": [0, 1], "class": ["tip_extension", "branch"]})
    q = truth.astype(np.float32)
    K = EV.subtype_components(u, q, sysmap, E).set_index("subtype")
    assert K.loc["branch", "n_truth"] == 6 and K.loc["tip_extension", "n_truth"] == 6
    assert K.loc["branch", "fp_w"] == 0.0                           # the other class is masked, not a false positive


def test_fixed_policy_and_frozen_reference_tables():
    T = STT.run_selftests().set_index("test").status
    for t in ("fixed_policy_never_uses_outer_truth", "no_comparison_is_not_a_w2_win", "dependency_enforcement",
              "tests_enforcement", "cache_validation", "unique_coverage_regression"):
        assert T[t] == "PASS", t
