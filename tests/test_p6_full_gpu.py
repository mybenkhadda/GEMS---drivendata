"""Pure-function tests of the P6 full GPU run (gpu_runtime, full_benchmark, final_submission, leaderboard, config).
Synthetic inputs only; no data, GPU or Drive access."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gems_bench.p6 import final_submission as FS
from gems_bench.p6 import full_benchmark as FB
from gems_bench.p6 import gpu_runtime as GR
from gems_bench.p6 import leaderboard as LB
from gems_bench.p6.full_gpu import classify_artifact, load_full_config


# ----------------------------------------------------------------------------------------------- config / GPU
def test_modes_merge_and_explicit_override_wins():
    full = load_full_config({"mode": "full"})
    rec = load_full_config({"mode": "recovery"})
    assert full["gpu"]["required"] and full["gpu"]["train_seeds"]
    assert not rec["gpu"]["required"] and not rec["gpu"]["train_seeds"]
    o = load_full_config({"mode": "recovery", "gpu": {"tta": True}})
    assert o["gpu"]["tta"] is True and o["gpu"]["train_seeds"] is False


def test_config_rejects_unknown_mode_and_construction():
    with pytest.raises(ValueError):
        load_full_config({"mode": "fast"})
    with pytest.raises(ValueError):
        load_full_config({"submission": {"construction": "full_refit"}})


def test_require_gpu_never_falls_back_silently():
    none = {"cuda_available": False}
    with pytest.raises(GR.GPURequired):
        GR.require_gpu(none, {"required": True})
    assert GR.require_gpu(none, {"required": False}) is None
    ok = {"cuda_available": True, "device": "cuda:0", "free_gb": 30.0, "total_gb": 40.0, "name": "A100"}
    assert GR.require_gpu(ok, {"required": True, "min_free_gb": 8}) == "cuda:0"
    low = {**ok, "free_gb": 2.0}
    assert not GR.gpu_decision(low, 8)[0]
    assert GR.gpu_stages({"train_seeds": True, "tta": True}, None) == {"train_seeds": False, "tta": False,
                                                                       "fold_average_maps": False}


def test_classify_artifact():
    assert classify_artifact(True, True, "x", False) == "REUSE"
    assert classify_artifact(True, False, "x", True) == "INCOMPATIBLE"
    assert classify_artifact(False, None, "x", True) == "RECOMPUTE"
    assert classify_artifact(False, None, "x", False) == "UNKNOWN"


# ----------------------------------------------------------------------------------------------- grids
def _rows(cand, blend, base, zones=(0, 1, 2), tasks=("geo", "disc_official"), reps=("crest_1px", "skeleton_1px"),
          fracs=(0.01, 0.03)):
    out = []
    for t in tasks:
        for z in zones:
            for r in reps:
                for f in fracs:
                    tp = base + 10 * z + (5 if r == "crest_1px" else 0) + (3 if f == 0.01 else 0)
                    out.append({"task": t, "zone": z, "candidate": cand, "blend": blend, "rep": r, "frac": f,
                                "tp_w": float(tp), "fp_w": 100.0, "fn_w": 100.0 - tp, "n_truth": 100.0 + z,
                                "protocol_key": f"{t}:{z}", "area_px": 1000, "area_frac_eligible": f})
    return out


def _grid():
    raw = pd.DataFrame(_rows("fallback_fusion", np.nan, 10) + _rows("w2", 1.0, 30) + _rows("cnn_b0", np.nan, 5) +
                       _rows("cnn_b1", np.nan, 6))
    return FB.normalize_grid(raw, "boost_a", "boost")


def test_normalize_prefixes_and_seeds_cells():
    G = _grid()
    assert set(G.candidate) == {"boost_a:fallback_fusion", "boost_a:w2", "boost_a:cnn_b0", "boost_a:cnn_b1"}
    S = pd.DataFrame(_rows("x", np.nan, 20)).drop(columns=["candidate", "blend"]).assign(method="seeds", w=0.5)
    s = FB.normalize_grid(S, "w1_seeds", "seeds")
    assert set(s.orig_candidate) == {"seeds|w0.5"} and s.blend.isna().all()
    assert FB.match_cells(["seeds|w0.25", "seeds_tta|w0.5", "seeds|w0.5"], ["seeds|*"]) == ["seeds|w0.25",
                                                                                          "seeds|w0.5"]


def test_normalize_refuses_oracle_rows():
    raw = pd.DataFrame(_rows("o", 1.0, 50)).assign(role="oracle_constrained")
    with pytest.raises(Exception, match="oracle"):
        FB.normalize_grid(raw, "x", "boost")


def test_protocol_check_flags_a_different_truth():
    G = _grid()
    bad = FB.normalize_grid(pd.DataFrame(_rows("S1", 1.0, 30)).assign(n_truth=999.0), "boost_b", "boost")
    T, ok = FB.protocol_check(pd.concat([G, bad], ignore_index=True))
    assert ok["boost_a"] is True and ok["boost_b"] is False


SEL = {"objective": "disc", "sensitivity_objectives": [], "worst_floor": 0.0,
       "geo_rule": {"mode": "relative", "max_drop": 0.01, "floor": 0.1758}, "tie_tolerance": 0.003,
       "min_gain_over_reference": 0.005, "min_zones_not_worse": 2, "slot_min_gain": 0.010, "ensemble_min_gain": 0.005,
       "references": ["ref_fb"], "never_winner_roles": ["diagnostic"]}
SPECS = {"ref_fb": {"grid": "boost_a", "cells": ["fallback_fusion"], "deploy": "fusion", "role": "reference",
                    "members": ["b0", "b1"]},
         "w2": {"grid": "boost_a", "cells": ["w2"], "deploy": "stack", "role": "candidate"},
         "b0": {"grid": "boost_a", "cells": ["cnn_b0"], "deploy": "cnn", "role": "candidate"},
         "b1": {"grid": "boost_a", "cells": ["cnn_b1"], "deploy": "cnn", "role": "candidate"},
         "absent": {"grid": "boost_b", "cells": ["S1"], "deploy": "stack", "status": "BLOCKED", "reason": "zone 2"},
         "nogrid": {"grid": "boost_a", "cells": ["H9"], "deploy": "stack"}}


def test_evaluate_candidates_single_policy_and_statuses():
    G = _grid()
    _, ok = FB.protocol_check(G)
    M, PZ = FB.evaluate_candidates(G, SPECS, "disc", "disc_official", ["geo", "disc_official"], None, 0.0, ok)
    st = dict(zip(M.name, M.status))
    assert st["w2"] == "EVALUATED" and st["absent"] == "BLOCKED" and st["nogrid"] == "NOT_FOUND"
    w = M[M.name == "w2"].iloc[0]
    assert w.deployable and w.disc_dti > M[M.name == "ref_fb"].iloc[0].disc_dti
    assert set(PZ[PZ.name == "w2"].zone) == {0, 1, 2}


def test_missing_zone_is_incomplete_never_pooled_on_fewer_zones():
    raw = pd.DataFrame(_rows("fallback_fusion", np.nan, 10) + _rows("w2", 1.0, 30, zones=(0, 1)))
    G = FB.normalize_grid(raw, "boost_a", "boost")
    M, _ = FB.evaluate_candidates(G, {k: SPECS[k] for k in ("ref_fb", "w2")}, "disc", "disc_official",
                                  ["geo", "disc_official"])
    assert dict(zip(M.name, M.status))["w2"] == "INCOMPLETE"


def test_choose_replaces_reference_only_with_gain_and_ensemble_rule():
    G = _grid()
    _, ok = FB.protocol_check(G)
    M, _ = FB.evaluate_candidates(G, SPECS, "disc", "disc_official", ["geo", "disc_official"], None, 0.0, ok)
    ch = FB.choose(M, SEL)
    assert ch["baseline"] == "ref_fb" and ch["winner"] == "w2" and ch["decision"] == "REPLACE"
    rows = {r["name"]: r for r in ch["rows"]}
    assert rows["ref_fb"]["eligible"]                       # fusion beats both members here
    assert not rows["absent"]["eligible"] and "BLOCKED" in rows["absent"]["eligibility_reasons"]
    strict = {**SEL, "min_gain_over_reference": 10.0}
    assert FB.choose(M, strict)["winner"] == "ref_fb"


def test_ensemble_not_justified_without_member_gain():
    by = {"a": {"name": "a", "status": "EVALUATED", "disc_dti": 0.20, "disc_worst": 0.15},
          "b": {"name": "b", "status": "EVALUATED", "disc_dti": 0.10, "disc_worst": 0.05}}
    ok, why = FB.ensemble_check({"members": ["a", "b"], "disc_dti": 0.202, "disc_worst": 0.16}, by, 0.005)
    assert not ok and "gain" in why
    ok, _ = FB.ensemble_check({"members": ["a", "b"], "disc_dti": 0.21, "disc_worst": 0.15}, by, 0.005)
    assert ok
    ok, why = FB.ensemble_check({"members": ["a", "zz"], "disc_dti": 0.5, "disc_worst": 0.5}, by, 0.005)
    assert not ok and "zz" in why


def test_geo_rule_modes():
    r = {"geo_dti": 0.10}
    assert FB.geo_check(r, {"mode": "report"}, 0.2)[0]
    assert not FB.geo_check(r, {"mode": "gate", "floor": 0.1758}, None)[0]
    assert FB.geo_check(r, {"mode": "relative", "max_drop": 0.01}, 0.105)[0]
    assert not FB.geo_check(r, {"mode": "relative", "max_drop": 0.01}, 0.12)[0]


# ----------------------------------------------------------------------------------------------- submission
def test_zone_partition_is_disjoint_and_reports_coverage():
    fm = np.array([[0, 0, 1], [1, 2, -1]])
    v = np.array([[True, True, True], [True, True, True]])
    z, cov = FS.zone_partition(fm, v, [0, 1, 2])
    assert cov["uncovered_px"] == 1 and cov["zone_px"] == {0: 2, 1: 2, 2: 1}
    assert (z[v & (fm < 0)] == -1).all()


def test_stitch_and_cells():
    q = FS.stitch((2, 4), [((slice(0, 2), slice(0, 2)), np.array([[1, 0], [1, 1]], bool), np.full((2, 2), 1.0)),
                           ((slice(0, 2), slice(2, 4)), np.ones((2, 2), bool), np.full((2, 2), 2.0))])
    assert q.tolist() == [[1, 0, 2, 2], [1, 1, 2, 2]]
    assert FS.split_cell({"candidate": "boost_a:w2"}) == ("boost_a", "w2")
    assert FS.seeds_weight("seeds_tta|w0.25") == ("seeds_tta", 0.25)
    with pytest.raises(ValueError):
        FS.split_cell({"candidate": "w2"})


def test_zone_score_expressions():
    src = {"fusion": np.full((2, 2), 0.2, np.float32), "stack": np.full((2, 2), 0.8, np.float32),
           "cnn_b0": np.zeros((2, 2), np.float32), "seeds_b0": np.ones((2, 2), np.float32),
           "seeds_b1": np.zeros((2, 2), np.float32)}
    s = FS.zone_score("stack", {"candidate": "boost_a:hybrid", "blend": 0.5}, src)
    assert np.allclose(s, 0.5)
    assert np.allclose(FS.zone_score("fusion", {"candidate": "boost_a:fallback_fusion", "blend": -1}, src), 0.2)
    assert np.allclose(FS.zone_score("seeds", {"candidate": "w1_seeds:seeds|w0.75", "blend": -1}, src), 0.75)
    with pytest.raises(ValueError):
        FS.zone_score("none", {"candidate": "rec_x3:H1", "blend": 1.0}, src)


def test_submission_path_never_overwrites(tmp_path):
    p, reuse = FS.submission_path(tmp_path, "submission_p6_boost", "abcdef1234")
    assert p.name == "submission_p6_boost.tif" and not reuse
    p.write_bytes(b"x")
    p.with_suffix(".json").write_text(json.dumps({"recipe_fp": "abcdef1234"}))
    assert FS.submission_path(tmp_path, "submission_p6_boost", "abcdef1234") == (p, True)
    q, reuse = FS.submission_path(tmp_path, "submission_p6_boost", "99999999zz")
    assert q.name == "submission_p6_boost__99999999.tif" and not reuse


def test_failed_validation_blocks_readiness():
    v = FS.validation_status({"official": [{"check": "a", "ok": True}],
                              "audit": [{"check": "b", "ok": False, "hard": True}, {"check": "c", "ok": False,
                                                                                    "hard": False}]})
    assert v["status"] == "FAIL" and v["failed"] == ["audit: b"]
    assert FS.readiness(v, {"decision": "REPLACE"}, "winner")["status"] == FS.NOT_READY
    ok = FS.validation_status({"official": [{"check": "a", "ok": True}]})
    assert FS.readiness(ok, {"decision": "REPLACE"}, "winner")["status"] == FS.READY
    assert FS.readiness(ok, {}, "winner")["status"] == FS.NOT_READY


# ----------------------------------------------------------------------------------------------- leaderboard
MAN = {"submission": "submission_p6_boost.tif", "sha256": "f" * 64, "candidate": "w2", "role": "winner",
       "recipe_fp": "abc", "recipe": {"kind": "zone_stitched_stack", "policy": {"rep": "crest_1px", "frac": 0.01,
                                                                               "blend": 1.0}},
       "offline": {"disc_dti": 0.24, "disc_worst": 0.19, "geo_dti": 0.11, "objective": "disc"},
       "zones": [{"eligible_px": 100, "positive_px": 1}], "readiness": {"status": "READY_FOR_UPLOAD"},
       "validation_status": "PASS", "code": {"git_commit": "deadbeef"}}


def test_leaderboard_row_validation_and_one_score_per_file():
    r = LB.make_row(MAN, 0.21, "2026-11-02")
    assert r["was_ready"] and abs(r["area_frac_eligible"] - 0.01) < 1e-12
    with pytest.raises(ValueError):
        LB.make_row(MAN, 1.5, "2026-11-02")
    with pytest.raises(ValueError):
        LB.make_row(MAN, 0.2, "02/11/2026")
    H = LB.append_row(pd.DataFrame(columns=LB.COLUMNS), r)
    with pytest.raises(ValueError, match="one file, one score"):
        LB.append_row(H, r)


def test_leaderboard_window_and_comparison():
    rows = [LB.make_row({**MAN, "sha256": c * 64, "offline": {"disc_dti": d}}, p, day, name=n)
            for c, d, p, day, n in (("a", 0.20, 0.15, "2026-11-01", "s1"), ("b", 0.24, 0.17, "2026-11-03", "s2"),
                                    ("c", 0.22, 0.20, "2026-11-10", "s3"))]
    H = pd.DataFrame(rows)
    assert LB.uploads_in_window(H, "2026-11-04") == 2
    T, s = LB.comparison(H, 0.005)
    assert s["n"] == 3 and s["pairs"] == 3 and "spearman_local_public" in s
    assert any("not a held-out validation" in c for c in LB.gap_causes(T.iloc[0].to_dict(), 0.005))
