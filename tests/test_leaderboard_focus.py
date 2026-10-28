"""Pure-function tests of Milestone 1 (leaderboard focus). Synthetic inputs only: no Drive, data or GPU.
Written in the authoring session; NOT executed there (the brief forbade any local execution)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gems_bench.leaderboard_focus import artifacts as AR
from gems_bench.leaderboard_focus import champion as CH
from gems_bench.leaderboard_focus import config as CF
from gems_bench.leaderboard_focus import parity as PA
from gems_bench.leaderboard_focus import reporting as RP
from gems_bench.leaderboard_focus import submission as SU
from gems_bench.leaderboard_focus import validation as VA


# ----------------------------------------------------------------------------------------------- config
def test_config_loads_and_rejects_unsafe_settings():
    cfg = CF.load_m1_config()
    assert cfg["output_dir"].startswith("outputs/leaderboard_focus/")
    with pytest.raises(CF.M1ConfigError):
        CF.load_m1_config({"output_dir": cfg["p6_output_dir"]})
    bad = {"champion": {"extra_candidates": {"x": {"grid": "boost_a", "cells": ["w2"], "fixed": {"rep": "a"},
                                                   "role": "candidate"}}}}
    with pytest.raises(CF.M1ConfigError):
        CF.load_m1_config(bad)


# ----------------------------------------------------------------------------------------------- artifacts
def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "drive"
    for rel in ("outputs/p6_fast_track/stack/candidate.json", "experiments/b0_reference__abc/manifest.json"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    for z in range(3):
        for t in ("geo", "disc_official"):
            p = root / f"outputs/p6_fast_track/stack/maps/{t}_{z}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, np.zeros((2, 2)))
    return root


def _cfg(entries):
    return {"artifacts": entries, "p6_output_dir": "outputs/p6_fast_track", "m9_output_dir": "m9", "hash_max_mb": 1}


def test_artifact_inventory_statuses(tmp_path):
    root = _tree(tmp_path)
    cfg = _cfg([{"id": "cand", "path": "{p6}/stack/candidate.json", "required": True, "staged": True},
                {"id": "maps", "path": "{p6}/stack/maps/*_[0-9].npy", "required": True, "staged": True,
                 "expect_count": 6},
                {"id": "few", "path": "{p6}/stack/maps/geo_*.npy", "required": False, "expect_count": 6},
                {"id": "gone", "path": "{p6}/nothing.json", "required": True},
                {"id": "b0", "path": "experiments/b0_reference__*/manifest.json", "required": True, "expect_count": 1,
                 "expected_id": "b0_reference__d54e938a"}])
    inv = AR.inventory(root, cfg, strict=True).set_index("id")
    assert inv.loc["cand", "status"] == AR.OK and inv.loc["maps", "status"] == AR.OK
    assert inv.loc["few", "status"] == AR.COUNT and inv.loc["gone", "status"] == AR.MISSING
    assert inv.loc["b0", "status"] == AR.IDCH                      # a different B0 run never silently replaces it
    s = AR.summary(inv.reset_index())
    assert s["status"] == CF.BLOCKED and any(b.startswith("gone") for b in s["blocking"])
    assert "few" in s["optional_missing"]
    plan = AR.staging_plan(inv.reset_index())
    assert len(plan) == 7 and all(not p.startswith("experiments/") for p in plan)


def test_inventory_fingerprint_changes_with_content(tmp_path):
    root = _tree(tmp_path)
    cfg = _cfg([{"id": "cand", "path": "{p6}/stack/candidate.json", "required": True}])
    a = AR.fingerprint(AR.inventory(root, cfg))
    (root / "outputs/p6_fast_track/stack/candidate.json").write_text('{"gate": "PROMOTE"}', encoding="utf-8")
    assert AR.fingerprint(AR.inventory(root, cfg)) != a


def test_classify_ambiguous_experiment():
    files = [Path("x/a"), Path("x/b")]
    st, _ = AR.classify({"expect_count": 1, "expected_id": "e1"}, files, ["e1", "e2"], True)
    assert st == AR.AMBIG


# ----------------------------------------------------------------------------------------------- stage records
def _session(tmp_path):
    from gems_bench.leaderboard_focus.session import M1Session, Workspace

    return M1Session(SimpleNamespace(bst=None, p6=None), CF.load_m1_config(), Workspace(tmp_path / "m1"))


def test_fingerprint_mismatch_recomputes_and_upstream_blocks(tmp_path):
    from gems_bench.leaderboard_focus.session import StageBlocked

    s = _session(tmp_path)
    calls = []

    def fn():
        calls.append(1)
        s.ws.save_text("x", "out.txt")
        return CF.COMPLETE, [], {}

    s.run_stage("selection", "fp1", ["out.txt"], fn)
    s.run_stage("selection", "fp1", ["out.txt"], fn)                 # cached
    s.run_stage("selection", "fp2", ["out.txt"], fn)                 # fingerprint changed
    assert len(calls) == 2
    s.record("validation", CF.BLOCKED, "v", reasons=["parity"])
    with pytest.raises(StageBlocked):
        s.require("validation")

    def needs():
        s.require("validation")
        return CF.COMPLETE, [], {}

    m = s.run_stage("parity", "p", [], needs)
    assert m["status"] == CF.INCOMPLETE


# ----------------------------------------------------------------------------------------------- champion
def test_artifact_gates_block_candidates_before_selection():
    from gems_bench.p6 import full_benchmark as FB

    specs = {"w2": {"deploy": "stack", "grid": "boost_a", "cells": ["w2"]},
             "fus": {"deploy": "fusion", "grid": "boost_a", "cells": ["fallback_fusion"]}}
    out = CH.apply_artifact_gates(specs, {"stack": ["geo_1.npy absent"]})
    assert out["w2"]["status"] == "BLOCKED" and "status" not in out["fus"]
    M = pd.DataFrame([{"name": "w2", "status": "BLOCKED", "role": "candidate", "deploy": "stack"},
                      {"name": "fus", "status": "EVALUATED", "role": "reference", "deploy": "fusion", "deployable": True,
                       "disc_dti": 0.16, "disc_worst": 0.16, "geo_dti": 0.18, "disc_per_zone": "{}"}])
    sel = {"references": ["fus"], "worst_floor": 0.15, "geo_rule": {"mode": "relative", "max_drop": 0.01},
           "ensemble_min_gain": 0.005, "tie_tolerance": 0.003, "min_gain_over_reference": 0.005,
           "min_zones_not_worse": 2, "slot_min_gain": 0.01, "never_winner_roles": ["diagnostic"]}
    ch = FB.choose(M, sel)
    assert ch["winner"] == "fus"
    assert not [r for r in ch["rows"] if r["name"] == "w2"][0]["eligible"]


def test_selection_rule_geo_floor_switch_and_candidate_names():
    full = {"selection": {"references": ["ref_p6_w2", "ref_m9_fallback", "other"], "geo_rule": {"mode": "relative"}},
            "candidates": {"ref_p6_w2": {}, "ref_m9_fallback": {}}}
    m1 = {"selection": {"overrides": {}, "geo_floor": 0.1758, "geo_floor_blocking": True},
          "champion": {"candidates": ["ref_p6_w2", "ref_m9_fallback"], "extra_candidates": {}}}
    s = CH.selection_rule(full, m1)
    assert s["geo_rule"] == {"mode": "gate", "floor": 0.1758} and s["references"] == ["ref_p6_w2", "ref_m9_fallback"]
    with pytest.raises(KeyError):
        CH.candidate_specs(full, {"champion": {"candidates": ["nope"]}})


def test_annotate_keeps_recorded_separate_and_flags_geo_floor():
    cfg = CF.load_m1_config()
    rows = [{"name": "a_w2_skeleton", "status": "EVALUATED", "deploy": "stack", "disc_dti": 0.24, "disc_worst": 0.2,
             "geo_dti": 0.11, "eligible": True, "is_winner": True, "deploy_cell": {"rep": "skeleton_1px"}}]
    C = CH.annotate(rows, cfg).iloc[0]
    assert C.decision == "CHAMPION" and C.family == "w2"
    assert C.recorded_disc_dti == pytest.approx(0.242156) and C.disc_dti == 0.24
    assert not C.geo_floor_met


# ----------------------------------------------------------------------------------------------- validation
def test_score_expressions_match_boost_a():
    F = np.random.default_rng(0).random((4, 4)).astype(np.float32)
    W = np.random.default_rng(1).random((4, 4)).astype(np.float32)
    assert VA.score_expression("w2", 1.0, F, W) is W
    assert VA.score_expression("fallback_fusion", 0.0, F, W) is F
    h = VA.score_expression("hybrid", 0.5, F, W)
    assert np.array_equal(h, (1.0 - 0.5) * F + 0.5 * W)
    with pytest.raises(ValueError):
        VA.score_expression("S1", 1.0, F, W)


def test_component_comparison_tolerance_and_pooling():
    saved = {"tp_w": 10.0, "fp_w": 100.0, "fn_w": 30.0, "n_truth": 40.0}
    ok = VA.compare_rows(saved, {**saved, "tp_w": 10.0 + 1e-9}, 1e-6, 1e-6)
    bad = VA.compare_rows(saved, {**saved, "fp_w": 101.0}, 1e-6, 1e-6)
    assert ok["status"] == "PASS" and bad["status"] == "BLOCKED"
    rows = pd.DataFrame([{"task": "disc_official", **{f"recomputed_{k}": v for k, v in saved.items()},
                          "recomputed_dti": VA.dti(10, 100, 30)}] * 2)
    p = VA.pooled(rows, "disc_official")
    assert p["disc_dti_recomputed"] == pytest.approx(VA.dti(20, 200, 60))


def test_recipe_consistency_detects_mixed_policy():
    recipe = {"deploy_kind": "stack", "policy": {"rep": "skeleton_1px", "frac": 0.015, "blend": 1.0,
                                                 "grid_candidate": "boost_a:w2", "fusion_w_all_zones": 0.5}}
    built = {"kind": "stack", "stack_dir": "stack/maps", "w_all": 0.5,
             "cell": {"rep": "skeleton_1px", "frac": 0.015, "blend": 1.0, "candidate": "boost_a:w2"}}
    assert VA.recipe_consistency(recipe, built) == []
    mixed = {**built, "cell": {**built["cell"], "frac": 0.05}}
    assert any(d.startswith("frac") for d in VA.recipe_consistency(recipe, mixed))
    other = {**built, "stack_dir": "boost/geology/maps/S1_x"}
    assert VA.recipe_consistency(recipe, other)


def test_array_sha_canonical_nan_and_budget_rows():
    a = np.array([[np.nan, 0.0], [1.0, 0.0]], np.float32)
    b = a.copy()
    b[0, 0] = np.float32("nan")
    assert VA.array_sha(a) == VA.array_sha(b) and VA.array_sha(a) != VA.array_sha(a.T)
    Z = pd.DataFrame({"zone": [0, 1], "frac_of_eligible": [0.0151, 0.02], "positive_on_known_px": [0, 0]})
    out = VA.budget_rows(Z, 0.015, 0.02)
    assert list(out.ok) == [True, False]


# ----------------------------------------------------------------------------------------------- parity
def test_parity_failure_blocks_and_causes_are_classified():
    ch = pd.DataFrame([{"unit": "geo:0", "check": "zone score vs evaluation construction", "status": "PASS"},
                       {"unit": "geo:1", "check": "zone score vs evaluation construction", "status": "WARNING"}])
    fresh = pd.DataFrame([{"scope": "historical", "check": "x", "unit": "*", "status": "BLOCKED", "detail": ""}])
    assert PA.final_status(ch, fresh, strict=True)[0] == CF.BLOCKED
    assert PA.final_status(ch, fresh, strict=False)[0] == CF.COMPLETE      # historical rows never block M1
    fresh2 = pd.DataFrame([{"scope": "champion", "check": "fold", "unit": "geo:0", "status": "BLOCKED", "detail": "3"}])
    assert PA.final_status(ch.iloc[:1], fresh2, strict=True)[0] == CF.BLOCKED
    hist = pd.DataFrame([{"unit": "geo:0", "check": "w2_score", "feature": "w2_score", "status": "BLOCKED"},
                         {"unit": "geo:0", "check": "feature", "feature": "cnn_b0", "status": "BLOCKED"},
                         {"unit": "geo:0", "check": "feature", "feature": "known_dist_km", "status": "WARNING",
                          "expected_by_design": True}])
    S = PA.summarize_historical(hist, {"cnn_b0"})
    assert set(S.cause) == {"score_map", "rank_domain", "by_design"}


def test_fold_assignment_and_zone_overlaps():
    ev = np.array([[1, 0], [1, 1]], bool)
    valid = np.ones((2, 2), bool)
    fold = np.array([[0, 1], [0, 0]])
    assert PA.fold_assignment(ev, valid, fold, 0)["status"] == "PASS"
    assert PA.fold_assignment(ev, valid, np.zeros((2, 2)), 0)["status"] == "BLOCKED"
    o = PA.zone_overlaps({0: ev, 1: ev}, valid)
    assert o["duplicated_px"] == 3 and o["missing_valid_px"] == 1


# ----------------------------------------------------------------------------------------------- GeoTIFF gate
def _facts(**kw):
    a = np.zeros((4, 4), np.float32)
    a[0, 0] = 1.0
    a[3, 3] = np.nan
    return {"count": 1, "dtype": "float32", "epsg": 32611, "res": [100.0, 100.0], "array": a, **kw}


def test_geotiff_checks_and_ready_gate():
    cfg = CF.load_m1_config()
    good = SU.m1_checks(_facts(), 1.0, cfg, "s", "s", [], "r", "r")
    assert SU.gate(good, CF.READY) == (CF.READY, [])
    assert SU.gate(good, "NOT_READY")[0] == CF.BLOCKED                  # p6 readiness propagates
    for kw, args in (({"epsg": 4326}, ("s", "s")), ({"count": 2}, ("s", "s")), ({}, ("s", "t"))):
        ch = SU.m1_checks(_facts(**kw), 1.0, cfg, *args, [], "r", "r")
        assert SU.gate(ch, CF.READY)[0] == CF.BLOCKED
    big = SU.m1_checks(_facts(), 999.0, cfg, "s", "s", [], "r", "r")
    assert SU.gate(big, CF.READY)[0] == CF.BLOCKED
    neg = _facts()
    neg["array"][1, 1] = -0.5
    assert SU.gate(SU.m1_checks(neg, 1.0, cfg, "s", "s", [], "r", "r"), CF.READY)[0] == CF.BLOCKED


def test_submission_files_are_never_overwritten(tmp_path):
    from gems_bench.p6.final_submission import submission_path

    name = SU.submission_name("submission_m1", "w2", "abc123")
    assert name == "submission_m1_w2_abc123"
    p, reuse = submission_path(tmp_path, name, "fp_one_xxxx")
    assert not reuse and p.name == f"{name}.tif"
    p.write_bytes(b"x")
    p.with_suffix(".json").write_text(json.dumps({"recipe_fp": "fp_one_xxxx"}), encoding="utf-8")
    assert submission_path(tmp_path, name, "fp_one_xxxx") == (p, True)
    q, reuse = submission_path(tmp_path, name, "fp_two_yyyy")
    assert not reuse and q.name == f"{name}__fp_two_y.tif"
    q.write_bytes(b"y")
    q.with_suffix(".json").write_text(json.dumps({"recipe_fp": "other"}), encoding="utf-8")
    with pytest.raises(FileExistsError):
        submission_path(tmp_path, name, "fp_two_yyyy")


# ----------------------------------------------------------------------------------------------- leaderboard
def test_leaderboard_log_integrity():
    man = {"status": CF.READY, "file": "f.tif", "sha256": "abc", "recipe_id": "r", "candidate": "c",
           "offline": {"disc_dti": 0.24, "geo_dti": 0.11}}
    e = {"submission_id": "s1", "submission_file": "f.tif", "public_dti": 0.21, "submitted_at": "2026-11-02", "notes": ""}
    assert RP.validate_entry({**e, "public_dti": None})
    assert RP.validate_entry({**e, "public_dti": 1.5})
    assert RP.validate_entry({**e, "submitted_at": "02/11/2026"})
    row, why = RP.leaderboard_row(e, man, "abc")
    assert not why and row["discrepancy_public_minus_offline"] == pytest.approx(-0.03)
    assert RP.leaderboard_row(e, man, "changed")[1]
    assert RP.leaderboard_row(e, {**man, "status": CF.BLOCKED}, "abc")[1]
    assert RP.leaderboard_row({**e, "submission_file": "other.tif"}, man, "abc")[1]
    H = RP.append_history(pd.DataFrame(columns=RP.LB_COLUMNS), row)
    with pytest.raises(ValueError):
        RP.append_history(H, row)
    with pytest.raises(ValueError):
        RP.append_history(H, {**row, "submission_id": "s2"})              # same file scored twice
