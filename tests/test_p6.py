"""P6: synthetic checks of the probe algebra and log, the selection rules, the stacker's leakage guards and features,
seed allocation / rank averaging / TTA, the throughput estimate, the DEM gate helpers and the completion geometry
(no Drive, no real data, no trained model)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _line(shape, r, c0, c1):
    m = np.zeros(shape, bool)
    m[r, c0:c1] = True
    return m


# ============================================================================ W0 probes
def test_kernel_mass_matches_offsets():
    from gems_bench.metric import kernel_offsets
    from gems_bench.p6.probes import kernel_mass

    c = kernel_mass()
    assert c == pytest.approx(sum(k for _, _, k in kernel_offsets(300.0, (100.0, 100.0))))
    assert 9.3 < c < 9.5


@pytest.mark.parametrize("truth_kind", ["isolated", "line", "cluster"])
def test_density_bracket_contains_true_density(truth_kind):
    from gems_bench.gems3.official import FastScorer
    from gems_bench.p6.probes import density_bounds

    H, W = 80, 90
    truth = np.zeros((H, W), bool)
    if truth_kind == "isolated":
        truth[10, 10] = truth[50, 60] = truth[30, 75] = True
    elif truth_kind == "line":
        truth |= _line((H, W), 40, 10, 70)
    else:
        truth[20:26, 20:26] = True
    known = _line((H, W), 70, 5, 80)
    ev = np.ones((H, W), bool)
    sc = FastScorer(truth, ev, known, exclude_supports_tp=False)
    p = np.where(known, 0.0, 1.0)
    d = sc.components(p).dti
    rho = truth.sum() / (ev & ~known).sum()
    b = density_bounds(d)
    assert b["rho_lo"] - 1e-9 <= rho <= b["rho_hi"] + 1e-9


def test_density_bounds_invalid_score():
    from gems_bench.p6.probes import density_bounds

    assert np.isnan(density_bounds(np.nan)["rho_lo"]) and np.isnan(density_bounds(0.0)["rho_hi"])


def test_all_eligible_map_and_facts():
    from gems_bench.p6.probes import all_eligible_map, raster_facts

    tv = np.ones((4, 5), bool)
    tv[0, 0] = False
    kn = np.zeros((4, 5), bool)
    kn[2, 2] = True
    a = all_eligible_map(tv, kn)
    assert np.isnan(a[0, 0]) and a[2, 2] == 0.0 and a[1, 1] == 1.0
    f = raster_facts(a)
    assert f["valid_px"] == 19 and f["positive_px"] == 18 and f["positive_frac_valid"] == pytest.approx(18 / 19)


def test_probe_plan_skips_duplicate_c():
    from gems_bench.p6.probes import probe_plan

    sel = {"disc_official": {"fusion_thin": {"params_all": {"frac": 0.05}}}}
    p = {r["probe"]: r for r in probe_plan(sel, "disc_official", "fusion_thin", "ABCD", 0.05)}
    assert "skip" in p["C"] and p["B"]["match"] == "A" and p["D"]["frac"] == 1.0
    p = {r["probe"]: r for r in probe_plan(sel, "disc_official", "fusion_thin", "ABC", 0.02)}
    assert p["C"]["frac"] == 0.05 and "D" not in p
    p = {r["probe"]: r for r in probe_plan({}, "disc_official", "fusion_thin", "C", 0.02)}
    assert "skip" in p["C"]


def _log(rows):
    from gems_bench.p6.probes import LOG_COLUMNS

    return pd.DataFrame(rows).reindex(columns=LOG_COLUMNS)


def test_merge_log_keeps_manual_scores():
    from gems_bench.p6.probes import merge_log

    old = _log([{"probe": "A", "sha256": "x", "public_score": 0.21, "notes": "typed"}])
    new = _log([{"probe": "A", "sha256": "x", "public_score": np.nan}, {"probe": "B", "sha256": "y"}])
    m = merge_log(old, new)
    assert len(m) == 2 and m.set_index("sha256").loc["x", "public_score"] == 0.21
    assert len(merge_log(pd.DataFrame(), new)) == 2


@pytest.mark.parametrize("a,b,verdict", [(0.20, 0.23, "broad_discovery"), (0.20, 0.17, "near_known_completion"),
                                         (0.20, 0.202, "inconclusive")])
def test_interpret_ring(a, b, verdict):
    from gems_bench.p6.probes import interpret

    log = _log([{"probe": "A", "sha256": "1", "public_score": a, "date_utc": "1", "area_frac_eligible": 0.05},
                {"probe": "B", "sha256": "2", "public_score": b, "date_utc": "2", "area_frac_eligible": 0.05}])
    out = interpret(log, 0.005)
    assert out["verdict"] == verdict and out["w5_enabled"] == (verdict == "near_known_completion")


def test_interpret_pending_budget_and_density():
    from gems_bench.p6.probes import interpret

    assert interpret(_log([]))["verdict"] == "PENDING"
    log = _log([{"probe": "A", "sha256": "1", "public_score": 0.20, "date_utc": "1", "area_frac_eligible": 0.05},
                {"probe": "C", "sha256": "3", "public_score": 0.24, "date_utc": "2", "area_frac_eligible": 0.02},
                {"probe": "D", "sha256": "4", "public_score": 0.01, "date_utc": "3", "area_frac_eligible": 1.0}])
    out = interpret(log)
    assert out["verdict"] == "different_budget" and "smaller" in " ".join(out["findings"])
    assert out["density"]["rho_lo"] < out["density"]["rho_hi"]


def test_read_log_union(tmp_path, monkeypatch):
    from gems_bench.p6 import probes

    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _log([{"probe": "A", "sha256": "x", "date_utc": "1"}]).to_csv(a, index=False)
    _log([{"probe": "A", "sha256": "x", "date_utc": "1", "public_score": 0.3}]).to_csv(b, index=False)
    monkeypatch.setattr(probes, "log_paths", lambda p6: [a, b])
    t = probes.read_log(None)
    assert len(t) == 1 and t.public_score.iloc[0] == 0.3


# ============================================================================ orchestration
def test_config_and_budget(tmp_path):
    from gems_bench.p6.run import Budget, BudgetExceeded, load_p6_config

    cfg = load_p6_config()
    assert cfg["budget"]["gpu_hours"] == 12.0
    with pytest.raises(ValueError):
        load_p6_config({"budget": {"workstreams": {"w1": {"gpu": 11.0}}}})
    b = Budget(tmp_path / "ledger.csv", cfg["budget"])
    b.add("w3", "gpu", 3600.0 * 1.5, "profile")
    assert b.spent_h("w3", "gpu") == pytest.approx(1.5) and b.left_h("w3", "gpu") == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded):
        b.check("w3", "gpu", 1.0)
    b.add("w3", "gpu", 3600.0, "more")
    with pytest.raises(BudgetExceeded):
        b.check("w3", "gpu")
    assert Budget(tmp_path / "ledger.csv", cfg["budget"]).spent_h("w3", "gpu") == pytest.approx(2.5)
    t = b.table()
    assert set(t.workstream) == {"w0", "w1", "w2", "w3", "w4", "w5", "w6"}


def test_w5_blocked_without_probe_evidence(tmp_path, monkeypatch):
    from gems_bench.p6 import probes
    from gems_bench.p6.run import P6FastTrack

    monkeypatch.setattr(probes, "log_paths", lambda p6: [tmp_path / "none.csv"])
    m9 = SimpleNamespace(local=tmp_path, pc=None)
    p6 = P6FastTrack(m9, {"push_to_drive": False})
    out = p6.w5_completion()
    assert out["status"] == "BLOCKED" and (tmp_path / "outputs/p6_fast_track/probes/interpretation.json").exists()


# ============================================================================ W6 selection
def test_gate_rules():
    from gems_bench.p6.select import gate

    ref = {"disc_dti": 0.162, "disc_worst": 0.126, "geo_dti": 0.183}
    rules = {"disc_min": 0.167, "worst_min": 0.126, "max_geo_drop": 0.005, "min_gain_over_ref": 0.005}
    assert gate({"disc_dti": 0.17, "disc_worst": 0.13, "geo_dti": 0.179}, ref, rules)[0]
    ok, why = gate({"disc_dti": 0.166, "disc_worst": 0.13, "geo_dti": 0.18}, ref, rules)
    assert not ok and "disc" in why[0]
    assert not gate({"disc_dti": 0.17, "disc_worst": 0.12, "geo_dti": 0.18}, ref, rules)[0]
    assert not gate({"disc_dti": 0.17, "disc_worst": 0.13, "geo_dti": 0.17}, ref, rules)[0]
    assert not gate({"disc_dti": None}, ref, rules)[0]


def test_choose_order_and_fallback_preference():
    from gems_bench.p6.select import choose

    t = pd.DataFrame([
        {"name": "fallback", "role": "fallback", "eligible": True, "disc_dti": 0.162, "disc_worst": 0.126,
         "geo_dti": 0.183, "unique_systems": 0},
        {"name": "a", "role": "candidate", "eligible": True, "disc_dti": 0.171, "disc_worst": 0.120, "geo_dti": 0.18,
         "unique_systems": 2},
        {"name": "b", "role": "candidate", "eligible": True, "disc_dti": 0.170, "disc_worst": 0.131, "geo_dti": 0.18,
         "unique_systems": 1},
        {"name": "c", "role": "candidate", "eligible": False, "disc_dti": 0.30, "disc_worst": 0.3, "geo_dti": 0.3,
         "unique_systems": 9}])
    row, trace = choose(t, tolerance=0.003)
    assert row["name"] == "b" and trace.criterion.iloc[0] == "disc_dti"
    tie = t.iloc[:1].copy()
    tie2 = pd.concat([tie, tie.assign(name="twin", role="candidate")], ignore_index=True)
    assert choose(tie2)[0]["role"] == "fallback"


def test_reference_from_selection():
    from gems_bench.p6.select import reference_from_selection

    sel = {"disc_official": {"fusion_thin": {"dti_crossfit": 0.16, "fold_min": 0.12}},
           "geo": {"fusion_thin": {"dti_crossfit": 0.18, "fold_min": 0.17}}}
    r = reference_from_selection(sel, "disc_official", "fusion_thin")
    assert r == {"disc_dti": 0.16, "disc_worst": 0.12, "geo_dti": 0.18, "geo_worst": 0.17}


# ============================================================================ W2 stacker
def test_potential_features_step_and_agreement():
    from gems_bench.p6.stack import potential_features

    a = np.zeros((40, 40), np.float32)
    a[:, 20:] = 10.0
    out = potential_features(a, a.copy())
    g = out["mag_hgrad"]
    assert np.nanargmax(g[20]) in (19, 20)
    assert np.nanmin(out["mg_orient_agree"]) > 0.99
    assert set(out) == {"mag_hgrad", "grav_hgrad", "mag_tilt", "mg_orient_agree"}
    assert potential_features(None, None) == {}


def test_endpoints_and_known_features():
    from gems_bench.p6.stack import endpoints, known_features

    k = _line((40, 60), 20, 10, 30)
    ep = endpoints(k)
    assert ep.sum() == 2 and ep[20, 10] and ep[20, 29]
    f = known_features(k, np.ones(k.shape, bool), 0.1, 20.0, 1.0, 1.0, 3)
    assert f["known_dist_km"][20, 20] == 0 and f["known_dist_km"][25, 20] == pytest.approx(0.5)
    assert f["tip_along_cos"][20, 35] > 0.9 and f["tip_along_cos"][20, 4] > 0.9   # straight ahead of both tips
    assert abs(f["tip_along_cos"][28, 29]) < 0.3                                   # beside a tip
    assert f["tip_along_cos"][20, 55] == 0.0                                        # beyond tip_max_km
    e = known_features(np.zeros((5, 5), bool), np.ones((5, 5), bool))
    assert (e["known_dist_km"] == 20.0).all() and (e["known_density"] == 0).all()


def test_rank_assemble_and_positions():
    from gems_bench.p6.stack import assemble, ev_position, rank_in

    ev = np.zeros((3, 3), bool)
    ev[1] = True
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    r = rank_in(a, ev)
    assert np.isnan(r[0, 0]) and r[1, 0] == 0.0 and r[1, 2] == 1.0
    X = assemble({"cnn_b0": a, "known_dist_km": a}, ev, ["cnn_b0", "known_dist_km", "g02c"])
    assert X.shape == (3, 3) and X[2, 0] == 1.0 and X[2, 1] == 5.0 and np.isnan(X[:, 2]).all()
    assert list(ev_position(ev, np.array([3, 5]))) == [0, 2]
    with pytest.raises(ValueError):
        ev_position(ev, np.array([0]))


def test_training_masks_never_use_faults_as_negatives():
    from gems_bench.p6.stack import training_masks

    shape = (30, 30)
    withheld = _line(shape, 10, 5, 25)
    retained = _line(shape, 20, 5, 25)
    ign = np.zeros(shape, bool)
    ign[0:3] = True
    pos, neg = training_masks(withheld, np.ones(shape, bool) & ~retained, withheld | retained, ign, 1)
    assert pos.sum() == withheld.sum() and not (neg & (withheld | retained)).any()
    assert not neg[9:12, 5:25].any() and not neg[0:3].any() and neg[15, 15] and neg[10, 0]


def test_sample_weights_ambiguous():
    from gems_bench.p6.stack import sample_weights

    pos = np.zeros((10, 10), bool)
    pos[0, :5] = True
    neg = ~pos
    amb = np.zeros((10, 10), bool)
    amb[5:] = True
    idx, y, w = sample_weights(pos, neg, amb, np.random.default_rng(0), 1000, 3, 1.0, 0.3)
    assert (y == 1).sum() == 3 and (y == 0).sum() == 95
    assert np.allclose(sorted(set(np.round(w[y == 0].astype(float), 3))), [0.3, 1.0])


def test_select_features_order_and_cap():
    from gems_bench.p6.stack import select_features

    avail = {"cnn_b0", "cnn_b1", "cnn_disagree", "g02c", "known_dist_km", "mag_hgrad"}
    assert select_features({"max_features": 25}, avail) == ["cnn_b0", "cnn_b1", "cnn_disagree", "g02c", "mag_hgrad",
                                                             "known_dist_km"]
    assert len(select_features({"max_features": 2}, avail)) == 2


def test_assert_provenance():
    from gems_bench.p6.stack import LeakageError, assert_provenance

    assert_provenance({"cnn_b0": "oof", "g02c": "g02_unit:geo_1"}, "geo", 1)
    assert_provenance({"cnn_b0": "discovery_pred:2", "g02c": "absent"}, "disc_official", 2)
    with pytest.raises(LeakageError):
        assert_provenance({"cnn_b0": "oof"}, "disc_official", 0)
    with pytest.raises(LeakageError):
        assert_provenance({"cnn_b0": "discovery_pred:1"}, "disc_official", 0)
    with pytest.raises(LeakageError):
        assert_provenance({"g02c": "g02_unit:geo_0"}, "geo", 1)


def test_permitted_known_excludes_own_truth():
    from gems_bench.p6.stack import permitted_known

    lab = np.zeros((10, 10), bool)
    lab[2, :] = lab[7, :] = True
    fm = np.zeros((10, 10), int)
    fm[5:] = 1
    wh = np.zeros((10, 10), bool)
    wh[7, :] = True
    ctx = SimpleNamespace(data=SimpleNamespace(labels=lab), folds=SimpleNamespace(fold_map=fm),
                          splits=SimpleNamespace(withheld=lambda d: wh))
    g = permitted_known(ctx, SimpleNamespace(task="geo", zone=1, is_disc=False))
    assert g[2].all() and not g[7].any()
    d = permitted_known(ctx, SimpleNamespace(task="disc_official", zone=0, is_disc=True))
    assert d[2].all() and not d[7].any()


def test_train_index_drops_heldout_zone_and_buffer():
    from gems_bench.p6.stack import train_index

    ev = np.zeros((20, 20), bool)
    ev[:, :5] = True
    u = SimpleNamespace(task="geo", zone=0, sl=(slice(0, 20), slice(0, 20)), ev=ev)
    T = pd.DataFrame({"task": ["geo"] * 4 + ["disc_official"], "zone": [0, 1, 1, 1, 1],
                      "r_full": [1, 1, 1, 1, 1], "c_full": [1, 6, 15, 19, 15]})
    m = train_index(T, u, (20, 20), 0.2)
    assert list(m) == [False, False, True, True, False]


def test_predict_unit_fills_ev_only():
    from gems_bench.p6.stack import predict_unit

    class M:
        def predict_proba(self, X):
            return np.stack([1 - X[:, 0] * 0, np.full(len(X), 0.7)], 1)
    ev = np.zeros((4, 4), bool)
    ev[1:3, 1:3] = True
    s = predict_unit(M(), {"cnn_b0": np.ones((4, 4))}, ev, ["cnn_b0"], chunk=3)
    assert np.allclose(s[ev], 0.7) and np.isnan(s[~ev]).all()


# ============================================================================ W1 seeds
def test_allocate_round_robin_within_budget():
    from gems_bench.p6.seeds import allocate

    fam = {"b0": {"extra_seeds": [1, 2]}, "b1": {"extra_seeds": [1, 2]}}
    have = {"b0": ["x"], "b1": ["y"]}
    assert allocate(fam, have, {"b0": 1.0, "b1": 1.0}, 3.0, ["b1", "b0"], 3) == [("b1", 1), ("b0", 1), ("b1", 2)]
    assert allocate(fam, have, {"b0": 2.5, "b1": 1.0}, 2.0, ["b0", "b1"], 3) == [("b1", 1), ("b1", 2)]
    assert allocate(fam, have, {"b0": 2.5, "b1": 1.0}, 1.5, ["b0", "b1"], 3) == [("b1", 1)]
    assert allocate(fam, {"b0": ["x", "y", "z"], "b1": ["y"]}, {"b0": 1, "b1": 1}, 9, ["b0", "b1"], 3) == \
        [("b1", 1), ("b1", 2)]


def test_rank_average_and_names():
    from gems_bench.p6.seeds import member_name, rank_average

    ev = np.ones((2, 3), bool)
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert np.allclose(rank_average([a, a * 10], ev), rank_average([a], ev))
    assert np.allclose(rank_average([a, -a], ev), 0.5)
    assert member_name("b0_reference", 2027) == "b0_reference_s2027"
    with pytest.raises(ValueError):
        rank_average([], ev)


def test_rot180_average():
    from gems_bench.p6.seeds import rot180_average

    x = np.random.default_rng(0).random((2, 5, 6)).astype(np.float32)
    assert np.allclose(rot180_average(lambda z: z * 2, x), x * 2)              # equivariant predictor: unchanged
    ramp = np.arange(6, dtype=np.float32)[None, None, :]
    out = rot180_average(lambda z: z + ramp, x)
    assert np.allclose(out, x + 2.5)                                           # bias averaged out


# ============================================================================ W3 throughput
def test_epoch_estimate_and_decision():
    from gems_bench.p6.throughput import decide, epoch_estimate

    t = {"batch": 0.5, "h2d": 0.05, "fwd_bwd": 0.3, "inner_val": 40.0}
    assert epoch_estimate(t, 80, 1, 1) == pytest.approx(80 * 0.5 + 40)
    assert epoch_estimate(t, 80, 2, 4) == pytest.approx(160 * 0.5 + 10)
    rows = pd.DataFrame([{"config": "a", **t, "epoch_s_est": 120.0}, {"config": "b", **t, "epoch_s_est": 80.0}])
    d = decide(rows, 90.0)
    assert d["decision"] == "PERMIT_ONE_CORE" and d["config"] == "b" and d["dominant"] == "inner_val"
    assert decide(rows.iloc[:1], 90.0)["decision"] == "STOP_M9" and decide(pd.DataFrame(), 90)["decision"] == "STOP_M9"


# ============================================================================ W4 DEM
def test_dem_helpers():
    from gems_bench.p6.dem_pilot import best_shift, evidence_map, gate, line_budget_px, select_lines, trace_recall

    shape = (40, 40)
    known = _line(shape, 20, 5, 35)
    cov = np.ones(shape, bool)
    cov[:, 38:] = False
    ev = evidence_map({"a": np.roll(known, 1, 0).astype(np.float32) + 0.01 * np.random.default_rng(1).random(shape)},
                      cov)
    assert np.isnan(ev[0, 39])
    sh = best_shift(ev, known, cov, 2)
    assert (sh["dr"], sh["dc"]) == (-1, 0)                 # roll that moves the evidence onto the traces
    n = line_budget_px(cov, 40.0)
    assert n == round(40 * (cov.sum() * 0.01) / 100 / 0.1)
    sel = select_lines(ev, cov, 30)
    assert 0 < sel.sum() <= 30 and not sel[:, 38:].any()
    assert trace_recall(sel, known, cov, 300.0) > 0.8
    assert trace_recall(sel, np.zeros(shape, bool), cov, 300.0) != trace_recall(sel, known, cov, 300.0)
    assert gate(0.4, 3, {"dr": 1, "dc": 0}, {"min_trace_recall": 0.3, "min_new_systems": 3, "max_abs_shift_px": 1})[0] \
        == "PASS"
    dec, why = gate(0.2, 1, {"dr": 2, "dc": 0}, {"min_trace_recall": 0.3, "min_new_systems": 3, "max_abs_shift_px": 1})
    assert dec == "STOP" and len(why) == 3


# ============================================================================ W5 completion
def test_gap_bridges_and_snap():
    from gems_bench.p6.completion import benchmark, gap_bridges, snap_ok, tip_directions

    shape = (30, 60)
    k = _line(shape, 15, 5, 25) | _line(shape, 15, 30, 50)
    er, ec, d = tip_directions(k, 3)
    assert len(er) == 4 and np.allclose(np.abs(d[:, 0]), 0, atol=1e-6)
    br = gap_bridges(k, 8, 15.0, 3)
    assert br[15, 25:30].all() and not (br & k).any()
    perp = _line(shape, 15, 5, 25)
    perp[20:28, 30] = True
    assert not gap_bridges(perp, 12, 15.0, 3).any()
    snap = {"max_shift_m": 300.0, "max_orientation_deg": 15.0, "min_overlap": 0.6}
    assert snap_ok(200, 10, 0.7, snap) and not snap_ok(400, 10, 0.7, snap) and not snap_ok(200, 10, 0.5, snap)
    rem = _line(shape, 15, 26, 30)
    b = benchmark(rem, br, 1.0)
    assert b["recall"] == 1.0 and b["near_share"] == 1.0
    assert np.isnan(benchmark(np.zeros(shape, bool), br)["recall"])


def test_repo_log_has_required_columns():
    from gems_bench.p6.probes import LOG_COLUMNS

    p = Path(__file__).resolve().parents[1] / "reports" / "p6" / "leaderboard_log.csv"
    assert p.exists() and list(pd.read_csv(p).columns) == LOG_COLUMNS


def test_stitch_disjoint_units():
    from gems_bench.p6.seeds import stitch

    ev0 = np.ones((2, 3), bool)
    ev1 = np.array([[True, False, True]])
    out = stitch((3, 3), [((slice(0, 2), slice(0, 3)), ev0, np.full((2, 3), 1.0)),
                          ((slice(2, 3), slice(0, 3)), ev1, np.full((1, 3), 2.0))])
    assert (out[:2] == 1).all() and out[2, 0] == 2 and np.isnan(out[2, 1])


def test_segments_and_table():
    from gems_bench.p6.select import segment_table, segments

    m = _line((20, 30), 5, 2, 12) | _line((20, 30), 15, 10, 25)
    paths = segments(m, 3)
    assert sorted(len(p) for p in paths) == [10, 15]
    conf = np.zeros((20, 30), np.float32)
    conf[5] = 0.8
    T = segment_table(paths, {"confidence": conf, "cnn_b0_rank": np.ones((20, 30))}, "fallback")
    t = T.sort_values("n_px")
    assert t.segment_confidence.iloc[0] == pytest.approx(0.8) and t.segment_confidence.iloc[1] == 0.0
    assert t.length_km.iloc[0] == pytest.approx(0.9) and (T.source_model == "fallback").all()
    assert "NOT a calibrated" in T.confidence_kind.iloc[0]
    assert segments(np.zeros((5, 5), bool)) == []
