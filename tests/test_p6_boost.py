"""P6 boost: synthetic checks of the G02 run resolution, the matched / nested benchmark tables, the gates, the
weak-zone diagnostics, the lineament hybrids (area never above the matched budget) and the matched-row ablation
(no Drive, no real data, no trained model)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _line(shape, r, c0, c1):
    m = np.zeros(shape, bool)
    m[r, c0:c1] = True
    return m


# ============================================================================ configuration / budget
def test_boost_config_bounded_and_budget_fits():
    from gems_bench.p6.boost import load_boost_config
    from gems_bench.p6.run import WORKSTREAMS, load_p6_config

    c = load_boost_config()
    assert list(c["geology"]["variants"]) == ["S0", "S1", "S2", "S3"]
    assert c["geology"]["variants"]["S0"] == [] and c["geology"]["keys"]["g01_persistence"] == "comp:P"
    assert c["grid"]["area_fracs"] == [0.005, 0.01, 0.015, 0.03, 0.05]
    assert set(c["grid"]["reps"]) == {"crest_1px", "skeleton_1px"}
    assert c["selection"]["geo_floor"] == pytest.approx(0.1758)
    assert c["champion"]["recorded"]["disc_dti"] == pytest.approx(0.224640)
    with pytest.raises(ValueError):
        load_boost_config({"robustness": {"variants": {"C1": {}, "C2": {}, "C3": {}}}})
    p6 = load_p6_config()                                    # caps (with wb) still within the global budget
    assert "wb" in WORKSTREAMS and p6["budget"]["workstreams"]["wb"]["cpu"] > 0


# ============================================================================ B: G02 run resolution
def _run(parent, rid, updated, complete=True):
    d = parent / rid
    (d / "config").mkdir(parents=True)
    (d / "manifests").mkdir()
    (d / "manifests" / "manifest.json").write_text(json.dumps({"fingerprint": rid, "updated": updated}))
    if complete:
        (d / "config" / "g02_config.json").write_text("{}")
        (d / "config" / "inputs.json").write_text("{}")


def test_resolve_run_without_latest_json(tmp_path):
    """The P6 failure: run folder staged, LATEST.json never pushed -> attach must still find the run."""
    from gems_bench.m9.research.g02_io import resolve_run

    _run(tmp_path, "g02_23686a63cb", "2026-10-27T13:57:09")
    rid, how = resolve_run(tmp_path)
    assert rid == "g02_23686a63cb" and "LATEST.json absent" in how


def test_resolve_run_order_and_incomplete(tmp_path):
    from gems_bench.m9.research.g02_io import resolve_run, set_latest

    _run(tmp_path, "g02_aaaaaaaaaa", "2026-10-01T00:00:00")
    _run(tmp_path, "g02_bbbbbbbbbb", "2026-10-20T00:00:00")
    _run(tmp_path, "g02_cccccccccc", "2026-10-30T00:00:00", complete=False)   # newest but incomplete: skipped
    assert resolve_run(tmp_path)[0] == "g02_bbbbbbbbbb"
    set_latest(tmp_path, "g02_aaaaaaaaaa", "fp", "FAST_RESEARCH")
    assert resolve_run(tmp_path) == ("g02_aaaaaaaaaa", "LATEST.json")
    assert resolve_run(tmp_path, "g02_bbbbbbbbbb") == ("g02_bbbbbbbbbb", "explicit run_id")
    with pytest.raises(FileNotFoundError):
        resolve_run(tmp_path, "g02_cccccccccc")
    with pytest.raises(FileNotFoundError):
        resolve_run(tmp_path / "nowhere")


def test_variant_status_blocks_missing_unit():
    from gems_bench.p6.boost import variant_status

    present = {"geo:0": {"g02c", "g02a"}, "geo:2": set(), "disc_official:2": {"g02a"}}
    assert variant_status(present, []) == ("RUN", [])
    st, miss = variant_status(present, ["g02c"])
    assert st == "BLOCKED" and miss == ["disc_official:2:g02c", "geo:2:g02c"]


# ============================================================================ A: tables from real units
def _units(seed=0):
    from gems_bench.gems3.official import Unit

    rng = np.random.default_rng(seed)
    out = []
    for z in range(3):
        H, W = 40, 60
        truth = _line((H, W), 10 + 5 * z, 5, 55) | _line((H, W), 30, 10, 30 + 5 * z)
        ev = np.ones((H, W), bool)
        out.append((Unit("disc_official", z, (slice(0, H), slice(0, W)), ev, truth, None, False, None), rng))
    return out


def _rows(units):
    from gems_bench.m7.protocol import score_row
    from gems_bench.p6.boost import dti, line_stats

    rows = []
    for u, rng in units:
        good = u.truth.astype(np.float32)
        for cand, q in (("perfect", good), ("half", good * (rng.random(good.shape) < 0.5)),
                        ("noise", (rng.random(good.shape) < 0.02).astype(np.float32))):
            for frac in (0.01, 0.03):
                r = score_row(u, q, u.ev, "k", candidate=cand, blend=1.0, rep="crest_1px", frac=frac)
                r["dti"] = float(dti(r["tp_w"], r["fp_w"], r["fn_w"]))
                r.update(line_stats(q))
                rows.append(r)
    return pd.DataFrame(rows)


def test_matched_and_nested_on_real_units():
    from gems_bench.p6.boost import matched_comparison, nested

    G = _rows(_units())
    M = matched_comparison(G)
    best = M.sort_values("dti", ascending=False).iloc[0]
    assert best.candidate == "perfect" and best.dti > 0.99 and best.worst_zone_dti > 0.99
    assert (M.worst_zone_dti <= M.dti + 1e-9).all() and (M.n_zones == 3).all()
    assert {"precision_w", "recall_w", "fp_share", "fn_share", "length_km", "n_segments"} <= set(M.columns)
    per, s = nested(G, ["candidate", "frac"])
    assert len(per) == 3 and (per.candidate == "perfect").all() and s["dti_crossfit"] > 0.99


def test_nested_never_selects_on_the_heldout_zone():
    """A cell that is excellent ONLY on zone z must not be chosen for zone z (selection uses the other zones)."""
    from gems_bench.p6.boost import nested

    rows = []
    for z in range(3):
        for cell, tp in (("spike", 100.0 if z == 0 else 1.0), ("steady", 10.0)):
            rows.append({"task": "disc_official", "zone": z, "cell": cell, "tp_w": tp, "fp_w": 10.0, "fn_w": 10.0,
                         "n_truth": 20.0, "systems": f"{z}{cell}"})
    per, _ = nested(pd.DataFrame(rows), ["cell"])
    assert per.set_index("zone").loc[0, "cell"] == "steady"
    assert per.set_index("zone").loc[0, "systems"] == "0steady"           # systems of the chosen held-out row


def test_ranking_gain_ladder_and_verdict():
    from gems_bench.p6.boost import load_boost_config, ranking_gain

    cfg = load_boost_config()["benchmark"]
    sp = {"fallback_published": 0.16, "fallback_fusion|crest_1px": 0.17, "fallback_fusion|skeleton_1px": 0.19,
          "fallback_fusion|rep+frac": 0.20, "w2|crest_1px": 0.18, "w2|skeleton_1px": 0.215,
          "w2|rep+frac": 0.215, "w2_space|blend+rep+frac": 0.225}
    S = pd.DataFrame([{"task": "disc_official", "space": k, "dti_crossfit": v, "fold_min": v - 0.05,
                       "per_zone": {0: v + 0.01, 1: v, 2: v - 0.05}} for k, v in sp.items()])
    M = pd.DataFrame([{"task": "disc_official", "candidate": c, "rep": r, "frac": f, "dti": d}
                      for r in ("crest_1px", "skeleton_1px") for f in (0.01, 0.05)
                      for c, d in (("fallback_fusion", 0.18), ("w2", 0.19 if f == 0.01 else 0.17))])
    R, v = ranking_gain(S, M, cfg, "disc_official")
    g = R.set_index("component").gain
    assert g["representation"] == pytest.approx(0.03) and g["ranking"] == pytest.approx(0.015)
    assert g["hybridization"] == pytest.approx(0.01) and g["total"] == pytest.approx(0.065)
    assert g["matched_cells_w2_minus_fusion"] == pytest.approx(0.0)
    assert v["w2_gain_is_ranking"] is False                  # wins only 2 / 4 matched cells
    assert v["largest_component"] == "representation"
    assert g["cell:w2_disc_choice"] == pytest.approx(0.01)


def test_systems_vs_fallback():
    from gems_bench.p6.boost import system_set, systems_vs

    per = pd.DataFrame({"zone": [0, 1, 2], "systems": ["1;2;3", "", "7"]})
    s = systems_vs(per, {0: {2, 3, 4}, 1: {5}, 2: {7}})
    assert (s["recovered"], s["reference"], s["gained"], s["lost"]) == (4, 5, 1, 2)
    assert s["gained_ids"] == "0:1" and s["lost_ids"] == "0:4;1:5"
    assert system_set(np.nan) == set() and system_set("3;4") == {3, 4}


def test_line_stats_fragmentation():
    from gems_bench.p6.boost import line_stats

    m = _line((20, 40), 5, 2, 32) | _line((20, 40), 15, 10, 12)
    s = line_stats(m, 0.1, 5)
    assert s["n_segments"] == 2 and s["length_km"] == pytest.approx(3.2) and s["short_share"] == 0.5
    assert line_stats(np.zeros((5, 5)))["n_segments"] == 0


# ============================================================================ gates
def test_accept_rejects_worst_zone_harm_and_blocked():
    from gems_bench.p6.boost import accept

    base = {"disc_dti": 0.2246, "disc_worst": 0.1534, "geo_dti": 0.18}
    ok, _ = accept({"disc_dti": 0.232, "disc_worst": 0.150, "geo_dti": 0.18}, base,
                   {"min_gain": 0.005, "max_worst_drop": 0.005}, 0.1758)
    assert ok
    ok, why = accept({"disc_dti": 0.240, "disc_worst": 0.140, "geo_dti": 0.18}, base,
                     {"min_gain": 0.005, "max_worst_drop": 0.005}, 0.1758)
    assert not ok and "worst zone" in why[0]
    ok, why = accept({"disc_dti": 0.2246, "disc_worst": 0.16, "geo_dti": 0.18}, base, {"min_gain": 0.0}, 0.1758)
    assert not ok                                            # min_gain 0 means strictly higher
    ok, why = accept({"disc_dti": 0.25, "disc_worst": 0.16, "geo_dti": 0.17}, base, {}, 0.1758)
    assert not ok and "geo" in why[0]
    ok, why = accept({"status": "BLOCKED", "reason": "geo:2 missing"}, base, {}, 0.1758)
    assert not ok and why[0].startswith("BLOCKED")
    ok, why = accept({"disc_dti": 0.25, "disc_worst": 0.16, "geo_dti": 0.18, "fp_increase": 0.3}, base,
                     {"max_fp_increase": 0.1}, 0.1758)
    assert not ok and "false-positive" in why[0]


def test_boost_gate_and_slot():
    from gems_bench.p6.boost import boost_gate

    champ = {"disc_dti": 0.224640, "disc_worst": 0.153414, "disc_dti_matched": 0.2230}
    rules = {"geo_floor": 0.1758, "slot_min_gain": 0.010}
    c = {"disc_dti": 0.2290, "disc_worst": 0.16, "geo_dti": 0.18, "provenance_ok": True}
    assert boost_gate(c, champ, rules) == (True, [], False)                  # passes, not worth a slot
    assert boost_gate({**c, "disc_dti": 0.2350}, champ, rules)[2] is True
    assert boost_gate({**c, "corrects_validation_flaw": True}, champ, rules)[2] is True
    ok, why, _ = boost_gate({**c, "disc_worst": 0.15}, champ, rules)
    assert not ok and "worst" in why[0]
    ok, why, _ = boost_gate({**c, "provenance_ok": False, "validation_failed": True}, champ, rules)
    assert not ok and len(why) == 2
    ok, why, _ = boost_gate({**c, "disc_dti": 0.2240}, {**champ, "disc_dti": 0.2, "disc_dti_matched": 0.2250}, rules)
    assert not ok                                             # must also beat the matched W2 number


def test_group_dropout_only_group_rows():
    from gems_bench.p6.boost import group_dropout

    X = np.arange(4000, dtype=np.float32).reshape(1000, 4)
    Y = group_dropout(X, [2, 3], 0.5, np.random.default_rng(0))
    assert np.isfinite(Y[:, :2]).all() and np.array_equal(np.isnan(Y[:, 2]), np.isnan(Y[:, 3]))
    assert 0.4 < np.isnan(Y[:, 2]).mean() < 0.6 and np.isfinite(X).all()
    assert np.array_equal(Y, group_dropout(X, [2, 3], 0.5, np.random.default_rng(0)), equal_nan=True)


# ============================================================================ C: diagnostics
def test_orientation_and_distance():
    from gems_bench.p6.boost import axial_orientation, distance_km, hist_distance, mean_truth_rank, orientation_hist

    h = _line((40, 40), 20, 5, 35)
    v = h.T.copy()
    th, tv = axial_orientation(h), axial_orientation(v)
    assert np.nanmax(np.minimum(th, np.pi - th)[h]) < 0.05
    assert np.allclose(tv[v][5:-5], np.pi / 2, atol=0.05)
    assert hist_distance(orientation_hist(th), orientation_hist(th)) == 0.0
    assert hist_distance(orientation_hist(th), orientation_hist(tv)) == pytest.approx(1.0)
    d = distance_km(_line((40, 40), 25, 5, 35), h, 0.1)
    assert np.allclose(d, 0.5)
    ev = np.ones((40, 40), bool)
    assert mean_truth_rank(h.astype(np.float32), h, ev) > 0.95 and np.isnan(mean_truth_rank(h, np.zeros_like(h), ev))


# ============================================================================ D: lineament hybrids
def test_seg_table_linearity_and_endpoint_support():
    from gems_bench.p6.boost import branches, seg_table

    m = _line((30, 40), 10, 2, 20)
    m[15:25, 30] = True
    m[25, 31:38] = True                                       # an L shape (diagonal joint: no junction pixel)
    lab = branches(m)
    score = np.where(m, 0.5, 0.0).astype(np.float32)
    score[10, 21:24] = 0.9                                    # continuation evidence beyond the straight line's tip
    T = seg_table(lab, score, {"geo": None}, m, ep_radius=3)
    straight = T.loc[T.label == lab[10, 5]].iloc[0]
    bent = T.loc[T.label == lab[20, 30]].iloc[0]
    assert straight.linearity > 0.99 > bent.linearity
    assert straight.endpoint_support == pytest.approx(0.45) and "mean_geo" not in T


def test_fill_budget_never_exceeds():
    from gems_bench.p6.boost import fill_budget

    lab = np.zeros((10, 30), int)
    lab[1, :10], lab[3, :5], lab[5, :20], lab[7, :3] = 1, 2, 3, 4
    out = fill_budget(lab, [3, 1, 2, 4], 27)                 # 20, skip 10 (30 > 27), 5, skip 3 (28 > 27)
    assert out.sum() == 25 and out[5].sum() == 20 and out[3].sum() == 5 and out[1].sum() == out[7].sum() == 0
    base = np.zeros_like(out)
    base[9, :5] = True
    assert fill_budget(lab, [1, 2], 12, base).sum() == 10


def test_controlled_union_matched_area():
    from gems_bench.p6.boost import controlled_union

    shape = (40, 60)
    base_full = _line(shape, 5, 0, 40) | _line(shape, 15, 0, 20)
    base_part = _line(shape, 5, 0, 40)
    fb = _line(shape, 6, 0, 40) | _line(shape, 30, 10, 30)        # one near-duplicate, one fallback-only line
    fbs = np.where(fb, 0.8, 0.0)
    k = int(base_full.sum())
    q = controlled_union(base_part, base_full, fb, fbs, np.ones(shape), k, tol_px=3)
    assert q.sum() <= k and q[base_part].all()
    assert q[30, 10:30].all() and not q[6].any()               # new system in, near-duplicate never counted as new
    assert not q[15].any()                                     # the weaker base line made room within the budget


def test_connectivity_select_prefers_coherent_lines():
    from gems_bench.p6.boost import connectivity_select

    shape = (60, 60)
    pool = _line(shape, 10, 5, 45)
    rng = np.random.default_rng(1)
    for r, c in rng.integers(20, 58, size=(40, 2)):
        pool[r, c] = True                                      # isolated specks with the same score
    s = np.where(pool, 0.7, 0.0).astype(np.float32)
    w = {"score": 2.0, "length": 1.0, "linearity": 0.5, "endpoint": 0.5, "geology": 1.0}
    q, T = connectivity_select(pool, s, 45, None, weights=w, short_px=5, short_penalty=0.5)
    assert q.sum() <= 45 and q[10, 5:45].all() and T.kept.sum() >= 1
    assert "mean_geology" not in T                              # degrades gracefully without G02-C
    geo = np.zeros(shape, np.float32)
    q2, T2 = connectivity_select(pool, s, 45, geo, weights=w)
    assert "mean_geology" in T2 and q2.sum() <= 45


def test_context_table_precision():
    from gems_bench.p6.boost import context_table

    T = pd.DataFrame({"label": [1, 2, 3, 4], "n_px": [10, 10, 20, 20], "mean_score": [0.1, 0.2, 0.8, 0.9],
                      "mean_ctx": [0.9, 0.1, 0.9, 0.1]})
    C = context_table(T, np.array([1.0, 0.0, 0.5, 0.5]), "mean_ctx").set_index(["confidence", "context"])
    assert C.loc[("low", "high"), "precision_px"] == 1.0 and C.loc[("low", "low"), "precision_px"] == 0.0
    assert C.n_segments.sum() == 4


def test_hybrid_output_is_a_valid_submission(tmp_path):
    """A D1 map on a synthetic grid finalises into a float32 {0, 1} GeoTIFF that passes the validator."""
    from affine import Affine
    from rasterio.crs import CRS

    from gems_bench.gems3.submission import finalise, validate
    from gems_bench.p6.boost import controlled_union
    from gems_bench.submission import write_submission

    H, W = 40, 50
    d = SimpleNamespace(valid=np.ones((H, W), bool), template_valid=np.ones((H, W), bool), grid=None,
                        labels=_line((H, W), 20, 0, 10),
                        template_profile={"driver": "GTiff", "height": H, "width": W, "count": 1, "dtype": "float32",
                                          "crs": CRS.from_epsg(32611),
                                          "transform": Affine(100, 0, 300000, 0, -100, 4500000)},
                        template_matches=lambda: True)
    d.valid[:2] = False
    q = controlled_union(_line((H, W), 5, 0, 30), _line((H, W), 5, 0, 40), _line((H, W), 30, 5, 25),
                         np.ones((H, W)), np.ones((H, W)), 40).astype(np.float32)
    p = write_submission(finalise(q, d), d, tmp_path / "s.tif")
    res = validate(p, d)
    assert res["dtype"] == "float32" and res["epsg"] == 32611 and not res["checks_failed"]


# ============================================================================ B/C: matched rows (no leakage path)
def _fake_boost(units, raws, T0, base):
    from gems_bench.p6.boost import Boost, load_boost_config

    b = object.__new__(Boost)
    b.cfg = load_boost_config()
    b._raw, b._T0 = raws, T0
    b.units = lambda: units
    b.raw = lambda u: raws[u.key]
    b.w2_features = lambda: base
    return b


def _row_fixture():
    from gems_bench.gems3.official import Unit
    from gems_bench.p6 import stack as ST

    rng = np.random.default_rng(3)
    units, raws, parts = [], {}, []
    for z in range(2):
        H, W = 20, 30
        sl = (slice(10 * z, 10 * z + H), slice(0, W))
        ev = np.ones((H, W), bool)
        ev[:2] = False
        u = Unit("disc_official", z, sl, ev, np.zeros((H, W), bool), None, False, None)
        raw = {"cnn_b0": rng.random((H, W)).astype(np.float32), "known_dist_km": rng.random((H, W)) * 9,
               "known_density": rng.random((H, W)), "tip_dist_km": rng.random((H, W)) * 4,
               "g02c": rng.random((H, W)).astype(np.float32)}
        raws[u.key], base = raw, ["cnn_b0", "known_dist_km"]
        idx = np.sort(rng.choice(np.flatnonzero(ev), 50, replace=False))
        X = ST.assemble(raw, ev, base)[ST.ev_position(ev, idx)]
        r, c = np.unravel_index(idx, ev.shape)
        parts.append(pd.DataFrame(X, columns=base).assign(task=u.task, zone=z, r_full=r + sl[0].start,
                                                          c_full=c + sl[1].start, y=0, w=1.0))
        units.append(u)
    return units, raws, pd.concat(parts, ignore_index=True), base


def test_rows_reuse_w2_rows_and_add_columns_exactly():
    from gems_bench.p6 import stack as ST

    units, raws, T0, base = _row_fixture()
    b = _fake_boost(units, raws, T0, base)
    T, A = b.rows(base + ["g02c"], check=True)
    assert A.max_abs_diff.max() == 0.0 and A.nan_pattern_equal.all()
    assert np.array_equal(T[base].to_numpy(), T0[base].to_numpy())               # identical W2 rows and weights
    u = units[1]
    m = (T.zone == 1).to_numpy()
    want = ST.rank_in(raws[u.key]["g02c"], u.ev)[T.r_full[m] - u.sl[0].start, T.c_full[m] - u.sl[1].start]
    assert np.allclose(T.loc[m, "g02c"], want)                                  # ranked inside ev, as at prediction
    Tk, _ = b.rows(base, view="rank_known")
    assert Tk.known_dist_km.max() <= 1.0 and not np.allclose(Tk.known_dist_km, T0.known_dist_km)
    with pytest.raises(KeyError):
        b.rows(base + ["g02a"])                                                  # absent feature: never zero-filled


def test_rows_check_detects_drift():
    units, raws, T0, base = _row_fixture()
    T0 = T0.copy()
    T0.loc[3, "cnn_b0"] += 0.5
    with pytest.raises(RuntimeError):
        _fake_boost(units, raws, T0, base).rows(base, check=True)
