"""Master-pipeline engine on synthetic stages: dependencies, cache decisions, resumption, configuration.

No raster data, GPU or network: every stage writes a small text file and validates it, so the executor's
decision logic (skip / build / resume / incompatible / blocked / interrupted) is tested in isolation.
"""
from pathlib import Path

import pytest

from gems_bench.artifacts.validation import INVALID, MISSING, OK, Check, Report
from gems_bench.pipeline.config import ConfigError, PipelineConfig, flat_to_nested
from gems_bench.pipeline.context import PipelineContext, PipelineStop, discovery_dir_name, fold_dir_name
from gems_bench.pipeline.dependencies import (MODE_TARGETS, CycleError, ancestors, closure, descendants,
                                              topological_order)
from gems_bench.pipeline.executor import (BLOCKED, BUILD, DISABLED, DONE, FAILED, INCOMPATIBLE, INTERRUPTED,
                                          MISSING_PREREQ, RESUME, SKIP, Executor)
from gems_bench.pipeline.registry import Stage, StageRegistry

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------- fixtures
@pytest.fixture
def roots(tmp_path, monkeypatch):
    local, drive = tmp_path / "local", tmp_path / "drive"
    drive.mkdir()
    for k in ("GEMS_ROOT", "GEMS_DRIVE_ROOT", "GEMS_CODE_ROOT"):
        monkeypatch.setenv(k, "")
    return local, drive


def make_pc(local, drive, **over):
    raw = {"paths": {"local_root": str(local), "drive_root": str(drive), "code_root": str(REPO)}}
    raw.update(over)
    return PipelineContext(PipelineConfig.load(None, overrides=raw))


def file_stage(name, deps=(), *, calls, behaviour=None, fp=None, enabled=None, files=1, **kw):
    """Stage writing <name>.<i>.out files containing 'x'; validation reads them back."""
    behaviour = behaviour or {}

    def run(pc):
        calls.append(name)
        if name in behaviour:
            behaviour[name](pc)
        for i in range(files):
            p = pc.local / f"{name}.{i}.out"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_text("x")
        return {"wrote": files}

    def validate(pc, root):
        rep = Report(name)
        for i in range(files):
            p = Path(root) / f"{name}.{i}.out"
            if not p.exists():
                rep.add(Check(str(p), MISSING))
            else:
                rep.add(Check(str(p), OK if p.read_text() == "x" else INVALID, "corrupt"))
        return rep

    return Stage(name, name, deps=tuple(deps), run=run, validate=validate,
                 outputs=lambda pc: [f"{name}.{i}.out" for i in range(files)],
                 fingerprint=fp or (lambda pc: {"v": 1}), enabled=enabled or (lambda pc: (True, "")), **kw)


def chain(calls, **kw):
    R = StageRegistry()
    R.add(file_stage("a", calls=calls, **kw.get("a", {})))
    R.add(file_stage("b", ("a",), calls=calls, **kw.get("b", {})))
    R.add(file_stage("c", ("b",), calls=calls, **kw.get("c", {})))
    R.check()
    return R


# ---------------------------------------------------------------------------- dependency graph
def test_topological_order_closure_and_relatives():
    calls = []
    R = StageRegistry()
    R.add(file_stage("raw", calls=calls))
    R.add(file_stage("grid", ("raw",), calls=calls))
    R.add(file_stage("folds", ("grid",), calls=calls))
    R.add(file_stage("eda", ("grid",), calls=calls))
    R.add(file_stage("features", ("folds", "eda"), calls=calls))
    R.add(file_stage("report", ("raw",), calls=calls, soft_deps=("features",)))
    order = topological_order(R)
    assert order.index("raw") < order.index("grid") < order.index("folds") < order.index("features")
    assert order.index("features") < order.index("report")          # soft deps order but never require
    assert closure(R, ["folds"]) == ["raw", "grid", "folds"]
    assert ancestors(R, "features") == {"raw", "grid", "folds", "eda"}
    assert descendants(R, "eda") == {"features"}


def test_cycles_and_unknown_dependencies_are_rejected():
    calls = []
    R = StageRegistry()
    R.add(file_stage("x", ("y",), calls=calls))
    R.add(file_stage("y", ("x",), calls=calls))
    with pytest.raises(CycleError):
        topological_order(R)
    R2 = StageRegistry()
    R2.add(file_stage("x", ("nope",), calls=calls))
    with pytest.raises(ValueError):
        R2.check()


def test_real_registry_graph_is_complete():
    from gems_bench.pipeline.stages import build_registry

    R = build_registry()                                          # check() runs inside
    for mode, targets in MODE_TARGETS.items():
        for t in targets or []:
            assert t in R, (mode, t)
    order = topological_order(R)
    assert order[0] == "bootstrap"
    assert order.index("raw_data") < order.index("native_grid") < order.index("geological_eda")
    assert order.index("segment_attributes") < order.index("feature_store") < order.index("baselines")
    assert order.index("spatial_folds") < order.index("feature_store")
    assert order.index("discovery_validation") < order.index("ensemble")
    assert order.index("full_inference") < order.index("submission")
    assert "feature_store" in ancestors(R, "dem_pilot")
    assert order.index("native_grid") < order.index("dem_inventory") < order.index("geological_eda")
    # G1 needs folds + splits (hard); the EDA / feature store are ordered first but never block it
    assert {"spatial_folds", "discovery_splits"} <= ancestors(R, "baselines")
    assert "geological_eda" not in ancestors(R, "baselines")
    assert {"geological_eda", "feature_store"} <= ancestors(R, "baselines", soft=True)
    assert "feature_store" in ancestors(R, "advanced_models")
    assert R["advanced_models"].expensive and R["dem_pilot"].expensive
    assert not R["feature_store"].expensive


# ---------------------------------------------------------------------------- cache decisions
def test_missing_builds_then_valid_cache_skips(roots):
    local, drive = roots
    calls = []
    pc = make_pc(local, drive)
    ex = Executor(pc, chain(calls))
    t = ex.run(mode="RESUME")
    assert calls == ["a", "b", "c"]
    assert set(t.outcome) == {DONE} and set(t.decision) == {BUILD}
    calls.clear()
    t2 = Executor(make_pc(local, drive), chain(calls)).run(mode="RESUME")   # new session object, same disk
    assert calls == [] and set(t2.outcome) == {SKIP}


def test_preexisting_outputs_are_adopted_not_rebuilt(roots):
    local, drive = roots
    local.mkdir(parents=True)
    (local / "a.0.out").write_text("x")
    calls = []
    pc = make_pc(local, drive)
    Executor(pc, chain(calls)).run(mode="RESUME")
    assert "a" not in calls
    assert pc.manifests.load("a")["origin"] == "adopted_existing"
    assert pc.manifests.load("b")["origin"] == "built"


def test_failure_blocks_dependents_with_exact_reason(roots):
    local, drive = roots
    calls = []

    def boom(pc):
        raise PipelineStop("labels file unreadable", "re-acquire existing_faults.tif")

    ex = Executor(make_pc(local, drive), chain(calls, b={"behaviour": {"b": boom}}))
    t = ex.run(mode="RESUME").set_index("stage")
    assert t.loc["a", "outcome"] == DONE
    assert t.loc["b", "outcome"] == FAILED and "re-acquire" in t.loc["b", "recovery"]
    assert t.loc["c", "outcome"] == BLOCKED and "b (FAILED)" in t.loc["c", "reason"]
    assert "c" not in calls
    rec = ex.pc.manifests.load("b")
    assert rec["status"] == "failed" and rec["recovery"] == "re-acquire existing_faults.tif"


def test_configuration_change_marks_incompatible_and_never_overwrites(roots):
    local, drive = roots
    calls = []
    Executor(make_pc(local, drive), chain(calls)).run(mode="RESUME")
    calls.clear()
    before = (local / "a.0.out").stat().st_mtime_ns
    R = chain(calls, a={"fp": lambda pc: {"v": 2}})
    t = Executor(make_pc(local, drive), R).run(mode="RESUME").set_index("stage")
    assert t.loc["a", "outcome"] == INCOMPATIBLE
    assert t.loc["b", "outcome"] == BLOCKED
    assert calls == [] and (local / "a.0.out").stat().st_mtime_ns == before


def test_partial_outputs_resume(roots):
    local, drive = roots
    local.mkdir(parents=True)
    (local / "a.0.out").write_text("x")                           # 1 of 2 files present
    calls = []
    R = StageRegistry()
    R.add(file_stage("a", calls=calls, files=2))
    t = Executor(make_pc(local, drive), R).run(mode="RESUME")
    assert t.decision.iloc[0] == RESUME and t.outcome.iloc[0] == DONE


def test_invalid_local_file_is_quarantined_then_rebuilt(roots):
    local, drive = roots
    local.mkdir(parents=True)
    (local / "a.0.out").write_text("truncated")
    calls = []
    R = StageRegistry()
    R.add(file_stage("a", calls=calls))
    t = Executor(make_pc(local, drive), R).run(mode="RESUME")
    assert t.outcome.iloc[0] == DONE and (local / "a.0.out").read_text() == "x"
    assert list(local.glob("a.0.out.invalid-*"))                  # kept for inspection, never reused


def test_disabled_stage_blocks_only_when_outputs_missing(roots):
    local, drive = roots
    calls = []
    off = {"enabled": lambda pc: (False, "switched off")}
    t = Executor(make_pc(local, drive), chain(calls, b=off)).run(mode="RESUME").set_index("stage")
    assert t.loc["b", "outcome"] == DISABLED and t.loc["c", "outcome"] == BLOCKED
    (local / "b.0.out").write_text("x")
    calls.clear()
    t = Executor(make_pc(local, drive), chain(calls, b=off)).run(mode="RESUME").set_index("stage")
    assert t.loc["b", "outcome"] == SKIP and t.loc["c", "outcome"] in (DONE, SKIP)


def test_validate_only_ancestors_are_not_built(roots):
    local, drive = roots
    calls = []
    t = Executor(make_pc(local, drive), chain(calls)).run(mode="SUBMISSION_ONLY", targets=["c"]).set_index("stage")
    assert t.loc["a", "outcome"] == MISSING_PREREQ
    assert t.loc["c", "outcome"] == BLOCKED and calls == []


# ---------------------------------------------------------------------------- resumption
def test_interrupt_is_recorded_and_next_session_resumes(roots):
    local, drive = roots
    calls = []

    def ctrl_c(pc):
        raise KeyboardInterrupt()

    ex = Executor(make_pc(local, drive), chain(calls, b={"behaviour": {"b": ctrl_c}}))
    t = ex.run(mode="RESUME").set_index("stage")
    assert t.loc["b", "outcome"] == INTERRUPTED
    assert "c" not in t.index                                      # the run stops cleanly after an interrupt
    assert ex.pc.manifests.load("b")["status"] == "interrupted"
    calls.clear()
    t2 = Executor(make_pc(local, drive), chain(calls)).run(mode="RESUME").set_index("stage")
    assert t2.loc["a", "outcome"] == SKIP and t2.loc["b", "outcome"] == DONE and t2.loc["c", "outcome"] == DONE


def test_records_survive_a_new_vm_via_drive(roots, tmp_path):
    local, drive = roots
    calls = []
    Executor(make_pc(local, drive), chain(calls)).run(mode="RESUME")
    assert (drive / "environment/pipeline/stages/a.json").exists()
    assert (drive / "environment/pipeline/events.jsonl").exists()
    fresh = make_pc(tmp_path / "new_vm", drive)                    # empty local disk, same Drive
    assert fresh.manifests.load("c")["status"] == "done"
    assert fresh.manifests.events().stage.tolist()[:1] == ["a"]


def test_resume_without_rebuild_only_continues_started_stages(roots):
    local, drive = roots
    calls = []
    t = Executor(make_pc(local, drive, rebuild_missing=False), chain(calls)).run(mode="RESUME").set_index("stage")
    assert t.loc["a", "outcome"] == DISABLED and calls == []


# ---------------------------------------------------------------------------- configuration
def test_config_rejects_typos_and_unsafe_values():
    with pytest.raises(ConfigError, match="did you mean 'run_eda'"):
        flat_to_nested({"run_edaa": True})
    with pytest.raises(ConfigError):
        PipelineConfig.load(None, flat={"run_mode": "TRAIN_EVERYTHING"})
    with pytest.raises(ConfigError, match="12.8"):
        PipelineConfig.load(None, overrides={"validation": {"buffer_km": 5.0}})
    with pytest.raises(ConfigError):
        PipelineConfig.load(None, flat={"train_baselines": ["NotAModel"]})
    with pytest.raises(ConfigError):
        PipelineConfig.load(None, dem={"mode": "everything"})


def test_config_defaults_are_safe_and_urls_never_persisted():
    c = PipelineConfig.load(REPO / "configs" / "pipeline" / "master.yaml",
                            flat={"urls": {"features": "https://example.org/f.tif?X-Amz-Signature=secret"}})
    assert c.mode == "RESUME"
    assert c.get("stages.run_advanced_models") is False and c.get("stages.run_full_inference") is False
    assert c.get("dem.download") is False and c.get("stages.train_baselines") == []
    pub = c.public()
    assert "secret" not in str(pub) and pub["data"]["urls"]["features"] == "example.org"
    assert "secret" not in c.table().to_string()


def test_validation_protocol_versioned_names():
    from gems_bench.pipeline.config import DEFAULTS

    v = dict(DEFAULTS["validation"])
    assert fold_dir_name(v, 2026) == "geo_k3_b20km_buf13km_s2026"
    assert discovery_dir_name(v, 2026) == "discovery_d3_link1km_s2026"
    assert fold_dir_name({**v, "clusters_per_fold": 4}, 2026) == "geo_k3_b20km_buf13km_s2026_c4"
    assert fold_dir_name({**v, "buffer_km": 15.0}, 2026) == "geo_k3_b20km_buf15km_s2026"
    assert discovery_dir_name({**v, "eval_km": 5.0}, 2026).endswith("_e5")
