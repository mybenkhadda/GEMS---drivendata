"""Submission factory: pure preflight / verdict / ranking logic + the runtime self-tests (synthetic inputs only: no
Drive, competition data or GPU). Written in the authoring session; NOT executed there (the brief forbade any local
execution)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gems_bench.leaderboard_focus import factory as FA

pytest.importorskip("rasterio")


def test_selftests_all_pass(tmp_path):
    from gems_bench.leaderboard_focus.factory_selftest import run_selftests

    T = run_selftests(tmp_path)
    assert (T.status == "PASS").all(), T[T.status != "PASS"].to_dict("records")


def test_missing_w2_blocks_only_w2_candidates():
    inv = pd.DataFrame([{"id": a, "status": "OK", "note": "", "pattern": a} for a in FA.SHARED_ARTIFACTS] +
                       [{"id": "w2_maps", "status": "MISSING_REQUIRED_FILES", "note": "geo_1.npy absent",
                         "pattern": "x"}] +
                       [{"id": a, "status": "OK", "note": "", "pattern": a} for a in FA.W2_ARTIFACTS if a != "w2_maps"])
    cands = list(FA.CANDIDATES)
    b = FA.blockers(FA.dependency_rows(inv, cands), cands)
    assert not b["b0_b1_fallback"]
    assert all(b[c] for c in ("w2_champion", "w2_crest", "w2_skeleton"))


def test_missing_template_blocks_everything():
    inv = pd.DataFrame([{"id": a, "status": "MISSING" if a == "template" else "OK", "note": "", "pattern": a}
                        for a in FA.SHARED_ARTIFACTS + FA.W2_ARTIFACTS])
    cands = list(FA.CANDIDATES)
    rows = FA.dependency_rows(inv, cands)
    assert any(r["candidate"] == "*" and r["artifact"] == "template" and r["status"] == FA.BLOCKED for r in rows)
    assert all(FA.blockers(rows, cands)[c] for c in cands)


def test_grid_rows_check_candidate_cells():
    G = pd.DataFrame([{"task": t, "zone": z, "candidate": c, "blend": 1.0, "rep": r, "frac": 0.01, "tp_w": 1.0,
                       "fp_w": 1.0, "fn_w": 1.0} for t in ("disc_official", "geo") for z in (0, 1, 2)
                      for c, r in (("w2", "skeleton_1px"), ("fallback_fusion", "skeleton_1px"))])
    rows = FA.grid_rows(G, {}, list(FA.CANDIDATES))
    st = {r["candidate"]: r["status"] for r in rows if r["scope"] == "candidate"}
    assert st == {"w2_champion": FA.PASS, "w2_crest": FA.BLOCKED, "w2_skeleton": FA.PASS, "b0_b1_fallback": FA.PASS}
    assert FA.grid_rows(None, {"status": "MISSING", "detail": "absent"}, ["w2_champion"])[0]["status"] == FA.BLOCKED


def test_geo_target_reports_but_never_blocks_in_report_only():
    assert FA.decide(FA.PASS, FA.PASS, FA.PASS, False, "report_only", True) == FA.READY
    assert FA.decide(FA.PASS, FA.PASS, FA.PASS, False, "strict", True) == FA.HELD
    assert FA.decide(FA.PASS, FA.FAIL, FA.PASS, True, "report_only", True) == FA.INVALID


def test_saved_metrics_pool_components():
    P = pd.DataFrame({"task": ["disc_official"] * 2 + ["geo"] * 2, "zone": [0, 1, 0, 1], "tp_w": [2.0, 1.0, 1.0, 0.5],
                      "fp_w": [1.0, 1.0, 2.0, 2.0], "fn_w": [1.0, 2.0, 1.0, 1.0], "n_truth": [3, 3, 2, 2]})
    m = FA.saved_metrics(P, "disc_official")
    assert m["disc_dti"] == pytest.approx(3.0 / (3.0 + 0.2 * 2.0 + 0.8 * 3.0 + 1e-6))
    assert m["geo_worst_zone"] == 1 and m["disc_n_truth"] == 6


def test_verdict_soft_and_hard():
    assert FA.verdict([]) == FA.NOT_RUN
    assert FA.verdict([FA.chk("g", "a", True), FA.chk("g", "b", False, hard=False)]) == FA.WARNING
    assert FA.verdict([FA.chk("g", "a", False)]) == FA.FAIL


def test_rank_never_gives_a_slot_to_non_ready():
    rows = [{"candidate": "w2_champion", "status": "INVALID", "disc_dti": 0.3, "geo_dti": 0.3, "p6g_eligible": True},
            {"candidate": "b0_b1_fallback", "status": FA.READY, "disc_dti": 0.16, "geo_dti": 0.158,
             "p6g_eligible": False}]
    R = {r["candidate"]: r for r in FA.rank(rows)}
    assert R["w2_champion"]["upload_priority"] == "DO_NOT_UPLOAD"
    assert R["b0_b1_fallback"]["upload_priority"] == "PRIMARY" and R["b0_b1_fallback"]["recommendation_rank"] == 1


def test_config_plan_modes():
    assert FA.plan(FA.factory_config({"run_mode": "DIAGNOSTIC"})) == {"evaluate": True, "write": False}
    assert FA.plan(FA.factory_config({"run_mode": "EXPORT_ONLY"})) == {"evaluate": False, "write": True}
    assert FA.submission_name("w2_champion", "0a8692b2ab") == "submission_w2_champion_0a8692b2ab"
    assert np.isclose(FA.DEFAULTS["geographic_target"], 0.1758)
