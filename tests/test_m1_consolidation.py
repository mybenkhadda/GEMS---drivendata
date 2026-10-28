"""Milestone 1 consolidation: shared grid loader, stage machinery, promotion gates, protocol comparison, reports.
Synthetic fixtures only (no Drive, data or GPU). Written in the authoring session; NOT executed there."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gems_bench.leaderboard_focus import config as CF
from gems_bench.leaderboard_focus import gates as GT
from gems_bench.leaderboard_focus import reporting as RP
from gems_bench.leaderboard_focus import submission as SU
from gems_bench.leaderboard_focus import validation as VA
from gems_bench.leaderboard_focus import workflow as WF
from gems_bench.p6 import export_parity as EP
from gems_bench.p6 import full_benchmark as FB

GRIDS = {"boost_a": {"path": "boost/benchmark/grid.csv", "stage": "boost/benchmark", "kind": "boost"},
         "boost_b": {"path": "boost/geology/grid.csv", "stage": "boost/geology", "kind": "boost"},
         "boost_c": {"path": "boost/robustness/grid.csv", "stage": "boost/robustness", "kind": "boost"},
         "boost_d": {"path": "boost/hybrid/grid.csv", "stage": "boost/hybrid", "kind": "boost"},
         "rec_x3": {"path": "recovery/segments/grid.csv", "stage": "recovery/segments", "kind": "boost"},
         "w1_seeds": {"path": "seeds/grid.csv", "stage": "seeds", "kind": "seeds"}}


def _grid_df(n_zones=3, **extra) -> pd.DataFrame:
    rows = [{"task": t, "zone": z, "candidate": "w2", "blend": 1.0, "rep": "skeleton_1px", "frac": 0.015,
             "tp_w": 10.0 + z, "fp_w": 100.0, "fn_w": 30.0, "n_truth": 40.0, "protocol_key": f"{t}{z}", **extra}
            for t in ("disc_official", "geo") for z in range(n_zones)]
    return pd.DataFrame(rows)


def _fg(root: Path) -> SimpleNamespace:
    return SimpleNamespace(cfg={"grids": GRIDS, "not_run_sources": {}}, p6=SimpleNamespace(ws=SimpleNamespace(
        root=root)))


def _write(root: Path, rel: str, content) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, pd.DataFrame):
        content.to_csv(p, index=False)
    else:
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
    return p


def _observed_layout(tmp_path: Path) -> Path:
    """The Colab state that crashed champion.recover / validation: boost_a + rec_x3 valid, w1_seeds 1 byte, B-D absent."""
    root = tmp_path / "p6"
    _write(root, GRIDS["boost_a"]["path"], _grid_df())
    _write(root, GRIDS["rec_x3"]["path"], _grid_df().assign(candidate="H0"))
    _write(root, GRIDS["w1_seeds"]["path"], b"\n")
    return root


# =============================================================================== grid loading
def test_read_csv_checked_statuses(tmp_path):
    cases = {"zero.csv": b"", "blank.csv": b"\n", "header.csv": b"task,zone,tp_w\n", "bad.csv": b'a,b\n"x,1\n'}
    for name, content in cases.items():
        _write(tmp_path, name, content)
    st = {n: FB.read_csv_checked(tmp_path / n)[1]["status"] for n in cases}
    assert st == {"zero.csv": FB.GRID_EMPTY, "blank.csv": FB.GRID_EMPTY, "header.csv": FB.GRID_ZERO_ROWS,
                  "bad.csv": FB.GRID_MALFORMED}
    assert FB.read_csv_checked(tmp_path / "absent.csv")[1]["status"] == FB.GRID_MISSING
    _write(tmp_path, "ok.csv", _grid_df())
    T, f = FB.read_csv_checked(tmp_path / "ok.csv", ["task", "nope"])
    assert T is None and f["status"] == FB.GRID_SCHEMA and "nope" in f["detail"]


def test_observed_empty_w1_grid_no_longer_breaks_boost_a(tmp_path):
    fg = _fg(_observed_layout(tmp_path))
    G, prov = FB.load_grids(fg, sources=["boost_a"], required=["boost_a"])
    assert set(G.source) == {"boost_a"} and set(prov) == {"boost_a"}          # nothing else is even read
    assert prov["boost_a"]["usable"] and prov["boost_a"]["rows"] == 6 and prov["boost_a"]["required"]
    assert prov["boost_a"]["sha256"]
    G, prov = FB.load_grids(fg)                                                 # P6 callers: every grid
    assert set(G.source) == {"boost_a", "rec_x3"}
    assert prov["w1_seeds"]["grid_status"] == FB.GRID_EMPTY and not prov["w1_seeds"]["usable"]
    assert prov["w1_seeds"]["exists"] and prov["w1_seeds"]["bytes"] == 1
    for s in ("boost_b", "boost_c", "boost_d"):
        assert prov[s]["grid_status"] == FB.GRID_MISSING and not prov[s]["exists"]


def test_required_grid_unusable_raises_actionable_error(tmp_path):
    root = tmp_path / "p6"
    for content, status in ((b"", FB.GRID_EMPTY), (b"task,zone\n", FB.GRID_ZERO_ROWS),
                            (b'task,zone\n"0,1\n', FB.GRID_MALFORMED)):
        _write(root, GRIDS["boost_a"]["path"], content)
        with pytest.raises(FB.GridUnavailable) as e:
            FB.load_grids(_fg(root), sources=["boost_a"], required=["boost_a"])
        assert status in str(e.value) and "boost_a" in str(e.value) and "never falls back" in str(e.value)
        assert e.value.prov["boost_a"]["grid_status"] == status
    (root / GRIDS["boost_a"]["path"]).unlink()
    with pytest.raises(FB.GridUnavailable) as e:
        FB.load_grids(_fg(root), sources=["boost_a"], required=["boost_a"])
    assert e.value.prov["boost_a"]["grid_status"] == FB.GRID_MISSING


def test_unknown_or_inconsistent_grid_sources_are_rejected(tmp_path):
    fg = _fg(_observed_layout(tmp_path))
    with pytest.raises(KeyError):
        FB.load_grids(fg, sources=["boost_z"])
    with pytest.raises(ValueError):
        FB.load_grids(fg, sources=["rec_x3"], required=["boost_a"])


def test_grid_schema_and_oracle_rows(tmp_path):
    root = tmp_path / "p6"
    _write(root, GRIDS["boost_a"]["path"], _grid_df().assign(tp_w="ten"))
    g, f = FB.read_grid_csv(root / GRIDS["boost_a"]["path"], "boost_a", "boost")
    assert g is None and f["status"] == FB.GRID_SCHEMA
    _write(root, GRIDS["boost_a"]["path"], _grid_df().drop(columns=["candidate"]))
    assert FB.read_grid_csv(root / GRIDS["boost_a"]["path"], "boost_a", "boost")[1]["status"] == FB.GRID_SCHEMA
    _write(root, GRIDS["boost_a"]["path"], _grid_df(role="oracle_truth"))          # leakage is never swallowed
    with pytest.raises(EP.OracleLeak):
        FB.load_grids(_fg(root), sources=["boost_a"], required=["boost_a"])


def test_resolve_specs_marks_candidates_on_unusable_grids(tmp_path):
    fg = _fg(_observed_layout(tmp_path))
    _, prov = FB.load_grids(fg)
    specs = FB.resolve_specs(fg, prov, {"seed": {"grid": "w1_seeds", "cells": ["seeds|*"]},
                                        "a": {"grid": "boost_a", "cells": ["w2"]}})
    assert specs["seed"]["status"] == "NOT_RUN" and FB.GRID_EMPTY in specs["seed"]["reason"]
    assert "status" not in specs["a"]


# =============================================================================== configuration
def test_notebook_config_and_overrides():
    c = CF.notebook_config({"mode": "full", "drive_root": "d", "local_root": "l"})
    assert c["mode"] == "FULL" and c["required_grids"] == ["boost_a"]
    for bad in ({"mode": "TRAIN"}, {"train_cnn": True}, {"run_extra_seeds": True}, {"save_csv": False}):
        with pytest.raises(CF.M1ConfigError):
            CF.notebook_config(bad)
    ov = CF.overrides_from_notebook({**c, "save_figures": False})
    cfg = CF.load_m1_config(ov)
    assert cfg["champion"]["required_grids"] == ["boost_a"] and cfg["outputs"]["save_figures"] is False


def test_config_never_lowers_the_geo_floor_silently():
    cfg = CF.load_m1_config()
    assert cfg["promotion"]["geographic_gate"] == "mandatory" and cfg["selection"]["geo_floor"] == pytest.approx(0.1758)
    with pytest.raises(CF.M1ConfigError):
        CF.load_m1_config({"promotion": {"geographic_gate": "advisory", "override_note": "  "}})
    ok = CF.load_m1_config({"promotion": {"geographic_gate": "advisory", "override_note": "user accepted geo gap"}})
    assert ok["promotion"]["geographic_gate"] == "advisory"
    with pytest.raises(CF.M1ConfigError):
        CF.load_m1_config({"champion": {"required_grids": ["boost_b"]}})
    a = {x["id"]: x for x in cfg["artifacts"]}
    assert "geo_2.npy" in a["w2_maps"]["required_files"] and "expect_count" not in a["w2_maps"]
    assert not [x for x in cfg["artifacts"] if x["staged"] and str(x["path"]).startswith(("experiments/", "{p6}/seeds"))]


# =============================================================================== stage machinery
def _sess(tmp_path):
    from gems_bench.leaderboard_focus.session import M1Session, Workspace

    return M1Session(SimpleNamespace(bst=None, p6=None, cfg=None), CF.load_m1_config(), Workspace(tmp_path / "m1"))


def test_plan_per_mode():
    assert WF.plan("full")[:2] == ("preflight", "artifacts") and WF.plan("FULL")[-1] == "report"   # M1.3: preflight
    assert "submissions" not in WF.plan("RECOVER") and "selection" not in WF.plan("VALIDATE")
    assert WF.plan("SUBMIT") == ("preflight", "artifacts", "promotion", "submissions", "exploratory", "leaderboard",
                                 "report")
    assert all(WF.engine_needed(m) for m in CF.MODES if m != "PREFLIGHT") and not WF.engine_needed("PREFLIGHT")


def test_blocked_results_are_re_evaluated_and_complete_reused(tmp_path):
    s = _sess(tmp_path)
    calls = []

    def blocked():
        calls.append("b")
        return CF.BLOCKED, ["boost_a EMPTY_FILE"], {}

    s.run_stage("selection", "fp", [], blocked)
    s.run_stage("selection", "fp", [], blocked)
    assert calls == ["b", "b"]                                     # a cached BLOCKED is never trusted

    def ok():
        calls.append("c")
        s.ws.save_text("x", "out.txt")
        return CF.COMPLETE, [], {}

    s.run_stage("validation", "v1", ["out.txt"], ok)
    m = s.run_stage("validation", "v1", ["out.txt"], ok)
    assert calls.count("c") == 1 and m["cache"] == CF.SKIP_CACHED
    (s.ws.path("out.txt")).unlink()
    s.run_stage("validation", "v1", ["out.txt"], ok)               # output missing -> recomputed
    assert calls.count("c") == 2
    assert list((s.ws.root / "manifests" / "history").glob("validation__*.json"))


def test_warnings_invalid_inputs_and_failures(tmp_path):
    from gems_bench.leaderboard_focus.session import StageInvalid

    s = _sess(tmp_path)
    m = s.run_stage("validation", "w", [], lambda: (CF.COMPLETE, [], {}, ["recorded value not reproduced"]))
    assert m["status"] == CF.WARNING and m["warnings"]

    def invalid():
        raise StageInvalid("selection/champion_candidates.csv is absent or empty")

    assert s.run_stage("parity", "p", [], invalid)["status"] == CF.BLOCKED

    def boom():
        raise ZeroDivisionError("bug")

    with pytest.raises(ZeroDivisionError):
        s.run_stage("submissions", "s", [], boom)
    assert s.status("submissions") == CF.FAILED and "traceback" in s.manifest("submissions")


def test_require_rejects_stale_code_and_unusable_upstream(tmp_path):
    from gems_bench.leaderboard_focus.session import StageBlocked, impl_fp

    s = _sess(tmp_path)
    s.record("selection", CF.COMPLETE, "f")
    s.require("selection")                                          # same code: accepted
    m = s.manifest("selection")
    s.ws.save_json({**m, "impl": "0000"}, "manifests/selection.json")
    with pytest.raises(StageBlocked, match="different stage code"):
        s.require("selection")
    s.record("validation", CF.WARNING, "v")
    s.require("validation")                                         # WARNING is usable downstream
    s.record("parity", CF.BLOCKED, "p")
    with pytest.raises(StageBlocked):
        s.require("parity")
    assert impl_fp("selection") == impl_fp("selection") and impl_fp("promotion") is None


def test_fingerprints_follow_policy_and_inputs_but_not_the_recipe_id(tmp_path):
    from gems_bench.pipeline.m6 import fingerprint

    s = _sess(tmp_path)
    a = s.fp("selection", "inv1", {"objective": "disc"})
    assert a != s.fp("selection", "inv1", {"objective": "joint"})        # policy change
    assert a != s.fp("selection", "inv2", {"objective": "disc"})         # input change
    assert s.fp("recipe", {"rep": "x"}) == fingerprint(CF.M1_VERSION, "recipe", {"rep": "x"})


def test_workspace_tables_tolerate_empty_files(tmp_path):
    from gems_bench.leaderboard_focus.session import StageInvalid, Workspace

    ws = Workspace(tmp_path)
    _write(tmp_path, "a.csv", b"\n")
    assert ws.load_table("a.csv").empty
    with pytest.raises(StageInvalid):
        ws.require_table("a.csv")
    _write(tmp_path, "b.csv", pd.DataFrame({"x": [1]}))
    with pytest.raises(StageInvalid):
        ws.require_table("b.csv", ("y",))


# =============================================================================== promotion gates
def _gate_inputs(geo=0.105687, val=CF.COMPLETE, par=CF.COMPLETE, rid="0a8692b2ab"):
    sel_m = {"status": CF.COMPLETE, "fingerprint": "S", "recipe_id": rid}
    sel = {"champion": "a_w2_rep_frac", "recipe_id": rid, "n_evaluated": 7, "n_evaluated_meeting_geo_floor": 0}
    crow = {"candidate_id": "a_w2_rep_frac", "eligible": True, "disc_dti": 0.242156, "disc_worst": 0.198579,
            "geo_dti": geo}
    val_m = {"status": val, "fingerprint": "V", "inputs": {"selection_fp": "S"}, "warnings": ["w"]}
    par_m = {"status": par, "fingerprint": "P", "inputs": {"validation_fp": "V"}}
    return sel_m, sel, crow, rid, val_m, par_m


def test_geo_floor_failure_blocks_promotion_but_not_selection():
    cfg = CF.load_m1_config()
    T, st, why, _ = GT.evaluate(*_gate_inputs(), cfg, 0.153414)
    g = dict(zip(T.gate, T.status))
    assert g["SELECTION_COMPLETE"] == "PASS" and g["DISCOVERY_ELIGIBLE"] == "PASS"
    assert g["GEOGRAPHIC_GATE_PASS"] == CF.BLOCKED and g["READY_FOR_EXPORT"] == CF.BLOCKED and st == CF.BLOCKED
    assert any("0 of 7" in w for w in why)
    adv = CF.load_m1_config({"promotion": {"geographic_gate": "advisory", "override_note": "accepted"}})
    T, st, _, warns = GT.evaluate(*_gate_inputs(), adv, 0.153414)
    assert st == CF.WARNING and any("ADVISORY override: accepted" in w for w in warns)
    T, st, _, _ = GT.evaluate(*_gate_inputs(geo=0.19), cfg, 0.153414)
    assert st == CF.COMPLETE and dict(zip(T.gate, T.status))["READY_FOR_EXPORT"] == "PASS"


def test_parity_failure_stale_validation_and_recipe_mismatch():
    cfg = CF.load_m1_config()
    _, st, why, _ = GT.evaluate(*_gate_inputs(geo=0.19, par=CF.BLOCKED), cfg, 0.153414)
    assert st == CF.BLOCKED and any(w.startswith("PARITY_PASS") for w in why)
    sel_m, sel, crow, rid, val_m, par_m = _gate_inputs(geo=0.19)
    val_m["inputs"]["selection_fp"] = "OLD"
    T, st, _, _ = GT.evaluate(sel_m, sel, crow, rid, val_m, par_m, cfg, 0.153414)
    assert dict(zip(T.gate, T.status))["VALIDATION_PASS"] == GT.PENDING and st == CF.INCOMPLETE
    _, st, why, _ = GT.evaluate(*_gate_inputs(geo=0.19)[:3], "other_id", *_gate_inputs()[4:], cfg, 0.153414)
    assert st == CF.BLOCKED and any("recipe ids disagree" in w for w in why)


def test_export_blocked_when_promotion_or_validation_fails():
    ok = {s: CF.COMPLETE for s in ("artifacts", "selection", "validation", "parity", "promotion")}
    assert SU.precheck(ok, True) == (None, [])
    assert SU.precheck({**ok, "validation": CF.WARNING, "promotion": CF.WARNING}, True) == (None, [])
    assert SU.precheck({**ok, "promotion": CF.BLOCKED}, True)[0] == CF.BLOCKED
    assert SU.precheck({**ok, "validation": CF.BLOCKED, "promotion": CF.BLOCKED}, True)[0] == CF.BLOCKED
    assert SU.precheck({**ok, "parity": None}, True)[0] == CF.INCOMPLETE
    assert SU.precheck(ok, False)[0] == CF.NOT_REQUIRED


# =============================================================================== protocols
def test_protocol_comparability_explains_geo_gap():
    assert VA.comparability("separate", "disc_dti")[0]
    comp, why = VA.comparability("separate", "geo_dti")
    assert not comp and "not a defect" in why
    assert not VA.comparability("fixed", "disc_dti")[0]
    rec = {"source": "boost_a", "objective": "separate", "protocol": "separate", "geo_dti": 0.168841}
    assert VA.recorded_row("a", rec, "geo_dti", 0.168841 + 1e-7, 6e-6)["status"] == "REPRODUCED"
    assert VA.recorded_row("a", rec, "geo_dti", 0.17, 6e-6)["status"] == "NOT_REPRODUCED"
    assert VA.recorded_row("a", rec, "geo_dti", None, 6e-6)["status"] == "NOT_AVAILABLE"
    RR = pd.DataFrame([VA.recorded_row("a", rec, "geo_dti", 0.17, 6e-6)])
    assert VA.recorded_effect(RR, "warning")[0] == [] and VA.recorded_effect(RR, "warning")[1]
    assert VA.recorded_effect(RR, "blocking")[0]
    PC = VA.protocol_comparison({"candidate_id": "c", "disc_dti": 0.24, "disc_worst": 0.2, "geo_dti": 0.105687}, RR,
                                {"p61": {"geo_dti": 0.105687}})
    assert set(PC.protocol) == {"single_policy", "separate"}
    assert not PC[PC.value_id == "recorded:a"].comparable_to_champion.iloc[0]


# =============================================================================== scientific safeguards
def test_official_metric_and_zone_stitching():
    from gems_bench.p6.final_submission import stitch, zone_partition

    tp, fp, fn = 12.0, 40.0, 25.0
    assert VA.dti(tp, fp, fn) == pytest.approx(tp / (tp + 0.2 * fp + 0.8 * fn + 1e-6))
    fold = np.array([[0, 0, 1], [1, 2, 2], [2, 2, 9]])
    valid = np.ones((3, 3), bool)
    zid, cov = zone_partition(fold, valid, [0, 1, 2])
    assert cov["uncovered_px"] == 1 and cov["zone_px"] == {0: 2, 1: 2, 2: 4}
    parts = [(np.s_[:, :], zid == k, np.full((3, 3), k + 1.0)) for k in (0, 1, 2)]
    Q = stitch((3, 3), parts)
    assert Q[0, 0] == 1 and Q[0, 2] == 2 and Q[2, 1] == 3 and Q[2, 2] == 0      # each zone from its own map only


# =============================================================================== reports
def test_next_action_and_failure_catalogue():
    done = {s: CF.COMPLETE for s in ("artifacts", "selection", "validation", "parity")}
    prom = {"status": CF.BLOCKED, "gates": [{"gate": "GEOGRAPHIC_GATE_PASS", "status": CF.BLOCKED},
                                            {"gate": "READY_FOR_EXPORT", "status": CF.BLOCKED}]}
    assert RP.next_action(done, prom, {}).startswith("DECISION REQUIRED")
    assert "selection" in RP.next_action({**done, "selection": CF.BLOCKED}, prom, {})
    ready = RP.next_action({**done, "promotion": CF.COMPLETE}, {"status": CF.COMPLETE},
                           {"status": CF.READY, "file": "f.tif"})
    assert "MANUALLY" in ready and "f.tif" in ready
    assert [f[0] for f in RP.failure_matches("required grid(s) unusable: boost_a EMPTY_FILE")] == ["GRID_REQUIRED"]
    assert "ART_COUNT" in [f[0] for f in RP.failure_matches("b0_maps: COUNT_MISMATCH (7 files, expected 4)")]


def test_final_report_from_manifests_only(tmp_path):
    s = _sess(tmp_path)
    s.record("artifacts", CF.COMPLETE, "a")
    s.record("selection", CF.BLOCKED, "s", reasons=["required grid(s) unusable: boost_a EMPTY_FILE"])
    out = RP.final_report(s)
    for n in ("M1_EXECUTIVE_SUMMARY.md", "M1_FAILURE_DIAGNOSTICS.md", "M1_NEXT_ACTION.md", "stage_status.csv"):
        assert (s.ws.root / "reports" / n).exists()
    diag = (s.ws.root / "reports" / "M1_FAILURE_DIAGNOSTICS.md").read_text(encoding="utf-8")
    assert "GRID_REQUIRED" in diag and "selection" in out["next_action"]
    assert json.loads(json.dumps(out["summary"], default=str))["Selected candidate"] == "none"
