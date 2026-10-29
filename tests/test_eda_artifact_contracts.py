"""Artifact registry, producer / consumer graph, typed-empty semantics, NPZ grid checks, stage-state recovery and
diagnostics (gems_eda.discovery.artifacts / run / diagnostics). Synthetic only; NOT executed at authoring time."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from test_geological_eda import _runner, _write_e02, _write_e03  # noqa: F401  (shared synthetic fixtures)


def _ancestors(stage, depends):
    out, todo = set(), list(depends.get(stage, ()))
    while todo:
        s = todo.pop()
        if s not in out:
            out.add(s)
            todo += list(depends.get(s, ()))
    return out


def test_registry_covers_contract_and_consumption():
    from gems_eda.discovery import artifacts as AR
    from gems_eda.discovery.run import ExhaustiveEDA

    reg = AR.registry(ExhaustiveEDA)
    assert {a.rel for a in reg} == {r for specs in ExhaustiveEDA.OUTPUTS.values() for r, _ in specs}
    assert AR.undeclared_consumption(ExhaustiveEDA) == []
    T = AR.contract_table(ExhaustiveEDA)
    assert {"artifact", "producer_stage", "requirement", "consumers", "rules", "reload"} <= set(T.columns)
    assert set(T.requirement) <= {"required", "conditional", "optional"}


def test_every_consumer_depends_on_the_producer():
    """A stage that reads an artifact must (transitively) depend on the stage that produces it - otherwise it could
    run on a missing / stale file (e03 read e02's domains and e11 read e07's lineaments without a DEPENDS edge)."""
    from gems_eda.discovery import artifacts as AR
    from gems_eda.discovery.run import ExhaustiveEDA as X

    bad = []
    for a in AR.registry(X):
        for c in a.consumers:
            if c in AR.NON_STAGE_CONSUMERS or c == a.producer or (a.rel, c) in AR.CONDITIONAL_CONSUMERS:
                continue
            if c not in X.STAGES or a.producer not in _ancestors(c, X.DEPENDS):
                bad.append((a.rel, a.producer, c))
    assert not bad, bad


def test_stage_order_respects_dependencies():
    from gems_eda.discovery.run import ExhaustiveEDA as X

    order = {s: i for i, s in enumerate(X.STAGES)}
    assert all(order[d] < order[s] for s, ds in X.DEPENDS.items() for d in ds)


def test_typed_empty_regional_table_is_valid(tmp_path):
    from gems_eda.discovery.parts_efg import LRO_COLUMNS

    E = _runner(tmp_path)
    E.ws.save_table(pd.DataFrame(columns=list(LRO_COLUMNS)), "tables/regional_robustness.csv")
    ok, bad = E.check_outputs(E.OUTPUTS["e09_regional"])
    assert ok and not bad                                    # zero regions = legitimate, typed result


def test_headerless_empty_table_is_invalid(tmp_path):
    E = _runner(tmp_path)
    p = E.ws.path("tables/regional_robustness.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    _, bad = E.check_outputs(E.OUTPUTS["e09_regional"])
    assert bad


def test_npz_grid_shape_and_honest_maps(tmp_path):
    E = _runner(tmp_path)
    E.save_npz("rasters/spatial_domains/domains.npz", domain=np.zeros((3, 3), np.int16))
    _, bad = E.check_outputs([("rasters/spatial_domains/domains.npz", ("domain",))])
    assert bad and "!= grid" in bad[0]
    E.save_npz("atlas/cache/honest_maps.npz", w2=np.zeros((2, 3), np.float16), b0=np.zeros((2, 2)),
               b1=np.zeros((2, 2)), zone_id=np.zeros((2, 2)), w2_frozen_bits=np.packbits(np.zeros(4, bool)),
               shape=np.asarray((2, 2)))
    _, bad = E.check_outputs([("atlas/cache/honest_maps.npz", ("w2", "b0", "b1", "zone_id"))])
    assert any("w2" in b for b in bad)                     # a frozen map off the raster grid is never reusable


def test_domain_rules_reject_duplicate_sample_ids(tmp_path):
    E = _runner(tmp_path)
    S = pd.DataFrame({"sample_id": [0, 0], "kind": ["fault", "control"], "system": [1, 1], "r": [1, 1], "c": [1, 1]})
    E.save_parquet(S, "datasets/matched_controls.parquet")
    _, bad = E.check_outputs([("datasets/matched_controls.parquet", ("sample_id", "kind", "system"))])
    assert any("unique_sample_id" in b for b in bad)


def test_e03_failed_blocks_e04_then_recovers(tmp_path):
    """e03 FAILED -> e04 BLOCKED -> e03 repaired -> e04 eligible (runs)."""
    E = _runner(tmp_path)
    E.status["e01_inventory"] = "COMPLETE"
    assert E.stage("e02_landscape", [], lambda: _write_e02(E) or {"note": ""})["status"] == "COMPLETE"
    m = E.stage("e03_geometry", [], lambda: {"note": "nothing written"})
    assert m["status"] == "FAILED"
    ran = []
    assert E.stage("e04_profiles", [], lambda: ran.append(1) or {"note": ""})["status"] == "BLOCKED_UPSTREAM"
    assert not ran
    assert E.stage("e03_geometry", [], lambda: _write_e03(E) or {"note": ""})["status"] == "COMPLETE"
    E.stage("e04_profiles", [], lambda: ran.append(1) or {"note": ""})
    assert ran                                               # eligible: it ran (and failed on its own outputs)


def test_stale_blocked_manifest_does_not_block_preflight(tmp_path):
    """An old BLOCKED_UPSTREAM manifest of e04 must not outrank a newer valid e03: preflight says NEEDS_RECOMPUTE."""
    from gems_eda.discovery.preflight import BLOCKED, FROM_CACHE, RECOMPUTE, Preflight

    E = _runner(tmp_path)
    E.status["e01_inventory"] = "COMPLETE"
    assert E.stage("e02_landscape", [], lambda: _write_e02(E) or {"note": ""})["status"] == "COMPLETE"
    assert E.stage("e03_geometry", [], lambda: _write_e03(E) or {"note": ""})["status"] == "COMPLETE"
    E.ws.save_json({"stage": "e04_profiles", "status": "BLOCKED_UPSTREAM", "fingerprint": None,
                    "written": "2026-10-29T10:00:00"}, "manifests/stages/e04_profiles.json")
    P = Preflight(E)
    S = P.stages(True, True, P.cache()).set_index("stage")
    assert S.readiness["e03_geometry"] == FROM_CACHE and S.readiness["e04_profiles"] == RECOMPUTE
    assert S.readiness["e04_profiles"] != BLOCKED


def test_preflight_optional_and_conditional_stages(tmp_path):
    from gems_eda.discovery.preflight import OPT_UNAVAILABLE, Preflight

    E = _runner(tmp_path)                                    # no factory, dem disabled
    P = Preflight(E)
    S = P.stages(True, True, P.cache()).set_index("stage")
    for s in ("01_attach", "03_scenarios", "e11_forensics", "e13_dem"):
        assert S.readiness[s] == OPT_UNAVAILABLE, (s, S.reason[s])
    assert S.readiness["e12_candidates"] != OPT_UNAVAILABLE   # core atlas does not need the engine


def test_preflight_blocked_propagates_from_invalid_inputs(tmp_path):
    from gems_eda.discovery.preflight import BLOCKED, INVALID, Preflight

    E = _runner(tmp_path)
    P = Preflight(E)
    S = P.stages(False, True, P.cache()).set_index("stage")
    assert S.readiness["e01_inventory"] == INVALID and S.readiness["e04_profiles"] == BLOCKED


def test_diagnostics_classify_and_downstream():
    from gems_eda.discovery import diagnostics as DG
    from gems_eda.discovery.run import ExhaustiveEDA as X

    assert DG.classify("ProductInvalid: tables/fault_geometry.csv: schema rules violated") == "SCHEMA_VIOLATION"
    assert DG.classify("GeometryPrecisionError: linearity") == "NUMERICAL_ERROR"
    assert DG.classify("MemoryError") == "RESOURCE_FAILURE"
    assert DG.classify("InsufficientData: no fault station") == "INSUFFICIENT_DATA"
    assert DG.classify("KeyError: 'x'") == "SOURCE_CODE_ERROR"
    d = DG.downstream("e03_geometry", X.DEPENDS)
    assert "e04_profiles" in d and "e12_candidates" in d and "e01_inventory" not in d


def test_history_is_append_only_and_errors_resolve(tmp_path):
    E = _runner(tmp_path)
    E.status["e01_inventory"] = "COMPLETE"

    def boom():
        raise KeyError("synthetic")
    assert E.stage("e02_landscape", [], boom)["status"] == "FAILED"
    assert E.stage("e02_landscape", [], lambda: _write_e02(E) or {"note": ""})["status"] == "COMPLETE"
    lines = E.ws.path("manifests/history/stage_events.jsonl").read_text().splitlines()
    ev = [json.loads(x) for x in lines]
    assert [e["status"] for e in ev if e["stage"] == "e02_landscape"] == ["FAILED", "COMPLETE"]
    assert ev[0]["category"] == "SOURCE_CODE_ERROR"
    errs = E.ws.read_json("manifests/runtime_errors.json")
    assert errs and all(e.get("resolved") for e in errs if e["stage"] == "e02_landscape")
    from gems_eda.discovery import diagnostics as DG

    assert not len(DG.current_errors(E))                     # resolved history is not a current failure
    L = pd.read_csv(E.ws.path("manifests/latest_stage_state.csv")).set_index("stage")
    assert L.status["e02_landscape"] == "COMPLETE"
