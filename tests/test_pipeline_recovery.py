"""Regression tests for the first master-pipeline run on Colab (commit 8ac074d): the EDA ``NameError``, the EDA
rebuild order / resume / install rules, native segment consistency, notebook stage order, failure policies,
GPU gating on CPU runtimes, Drive output lists and reuse of cached raw files and folds.

Synthetic data only (tiny rasters / tables in ``tmp_path``): no competition data, network, GPU or training.
Authored without being executed in the authoring session - run with ``pytest tests/test_pipeline_recovery.py``.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gems_bench.artifacts.validation import OK, Check
from gems_bench.pipeline.context import PipelineStop, StageFailed
from gems_bench.pipeline.executor import BLOCKED, DONE, FAILED, Executor, error_location
from gems_bench.pipeline.registry import Stage, StageRegistry

from test_pipeline_engine import REPO, file_stage, make_pc, roots  # noqa: F401  (fixture re-export)

NOTEBOOK = REPO / "notebooks" / "colab" / "GEMS_MASTER_PIPELINE.ipynb"


# ---------------------------------------------------------------------------- 1. the Colab NameError
def test_class_body_default_cannot_read_enclosing_argument():
    """The exact pattern that failed on Colab: documents why the field is named ``rebuild_version``."""
    from dataclasses import dataclass

    def broken(version: str):
        @dataclass(frozen=True)
        class P:
            version: str = version          # noqa: F821  (class body: enclosing local is invisible)

        return P

    with pytest.raises(NameError):
        broken("v1")


def test_rebuild_paths_are_versioned_and_created(tmp_path):
    pytest.importorskip("rasterio")
    from gems_eda.config import EDAPaths

    from gems_bench.pipeline.eda_rebuild import _rebuild_paths

    p = _rebuild_paths(tmp_path, "v20261022_120000")
    assert isinstance(p, EDAPaths)
    assert p.rebuild_version == "v20261022_120000"
    assert p.out == tmp_path / "outputs" / "eda_rebuild" / "v20261022_120000"
    for d in (p.datasets, p.features, p.tables, p.figures):
        assert d.is_dir() and p.out in d.parents
    assert p.manifest_file == p.out / "manifest.json"
    # raw inputs still resolve against the project root, not the rebuild folder
    assert p.raw("features").name == "gems-geodawn-numerical-features.tif"
    assert p.out not in p.raw("features").parents
    assert not (tmp_path / "outputs" / "eda" / "datasets").exists()          # historical folder untouched
    with pytest.raises(ValueError):
        _rebuild_paths(tmp_path, "../escape")
    with pytest.raises(ValueError):
        _rebuild_paths(tmp_path, "")


# ---------------------------------------------------------------------------- 2. rebuild order / reuse / resume
@pytest.fixture
def fake_eda(monkeypatch):
    """Patch the expensive gems_eda calls; record the section order."""
    pytest.importorskip("rasterio")
    pytest.importorskip("matplotlib")
    from gems_eda import context as C
    from gems_eda import sections as S

    from gems_bench.pipeline import eda_rebuild as R

    calls = []
    ctx = SimpleNamespace(grid=SimpleNamespace(pixel_km=0.2))
    monkeypatch.setattr(R, "_preflight", lambda: calls.append("preflight"))
    monkeypatch.setattr(R, "_load_context", lambda root, paths, max_dim, seed: calls.append("load") or ctx)
    monkeypatch.setattr(C, "add_derived_signals", lambda c, km: calls.append(f"derived@{km:g}") or ["d"])
    monkeypatch.setattr(S, "s2_morphology", lambda c: calls.append("s2"))
    monkeypatch.setattr(S, "s3_profiles", lambda c, d: calls.append("s3") or {"prof": 1})
    monkeypatch.setattr(S, "s4_fused_lineaments", lambda c: calls.append("s4"))
    monkeypatch.setattr(S, "s5_crossmodal", lambda c, p: calls.append("s5"))
    monkeypatch.setattr(S, "s6_archetypes", lambda c, p: calls.append("s6"))
    monkeypatch.setattr(R, "check_eda_file", lambda path, spec: Check(str(path), OK))
    installed = []
    monkeypatch.setattr(R, "install", lambda root, src, groups, v, resume=(): installed.append((groups, v)) or [])
    return R, calls, installed


def test_rebuild_runs_sections_in_notebook_order(tmp_path, fake_eda):
    R, calls, installed = fake_eda
    rep = R.rebuild_minimal(tmp_path, version="vtest")
    assert calls == ["preflight", "load", "derived@2", "s2", "s3", "s4", "s5", "s6"]
    assert rep["rebuilt"] == ["segments", "lineaments", "crossmodal"] and rep["version"] == "vtest"
    done = json.loads((tmp_path / "outputs/eda_rebuild/vtest/REBUILD_COMPLETE.json").read_text())
    assert done["params"] == {"max_dim": 2048, "seed": 7} and done["groups"] == rep["rebuilt"]
    assert installed == [(rep["rebuilt"], "vtest")]


def test_rebuild_only_needed_sections(tmp_path, fake_eda):
    R, calls, _ = fake_eda
    R.rebuild_minimal(tmp_path, groups=["lineaments"], version="v1")
    assert [c for c in calls if c.startswith("s")] == ["s4"]                  # no profiles for lineaments


def test_finished_rebuild_is_reused_without_recomputation(tmp_path, monkeypatch):
    rasterio = pytest.importorskip("rasterio")  # noqa: F841
    from gems_bench.pipeline import eda_rebuild as R
    from test_pipeline_artifacts import _lineament_tif

    src = tmp_path / "outputs/eda_rebuild/v1"
    _lineament_tif(src / "features/lineament_maps.tif")
    (src / R.COMPLETE).write_text(json.dumps({"version": "v1", "groups": ["lineaments"],
                                              "params": {"max_dim": 2048, "seed": 7}}))
    monkeypatch.setattr(R, "_preflight", lambda: (_ for _ in ()).throw(AssertionError("must not recompute")))
    rep = R.rebuild_minimal(tmp_path, groups=["lineaments"])
    assert rep["reused_rebuild"] == "v1"
    assert rep["installed"] == ["outputs/eda/features/lineament_maps.tif"]
    assert R.install_state(tmp_path)["status"] == "complete"
    # other parameters: never reused
    assert R.completed_rebuilds(tmp_path, max_dim=1024, seed=7) == []
    assert R.rebuild_sources(tmp_path) == ["outputs/eda_rebuild/v1"]


def _segment_group(d: Path, n: int = 3, gpkg: bool = True):
    """A consistent synthetic Phase-1 segment group (ids 0..n-1) under ``d`` (= outputs/eda or a rebuild)."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString

    from test_pipeline_artifacts import grid

    g = grid()
    (d / "datasets").mkdir(parents=True, exist_ok=True)
    x0, y0 = g.transform.c, g.transform.f
    seg = pd.DataFrame({"seg_id": range(n), "length_km": [1.5, 3.0, 6.0][:n], "azimuth": [10.0, 50.0, 120.0][:n],
                        "sinuosity": [1.01, 1.1, 1.4][:n], "linearity": 0.9, "curv_mean_deg_km": [1.0, 2.0, 3.0][:n],
                        "n_junction_ends": [0, 1, 0][:n], "region": "R1", "x_mid": 0.0, "y_mid": 0.0,
                        "component": 0, "touches_edge": False})
    seg.to_parquet(d / "datasets/known_fault_segments.parquet")
    pd.DataFrame({"node_id": [0], "kind": ["endpoint"]}).to_parquet(d / "datasets/known_fault_nodes.parquet")
    pd.DataFrame({"seg_id": range(n), "primary_archetype": "magnetic", "ev_magnetic": 0.9}).to_parquet(
        d / "datasets/segment_archetypes.parquet")
    if gpkg:
        rows = [LineString([(x0 + 500 + 1500 * i, y0 - 1000), (x0 + 500 + 1500 * i, y0 - 5000)]) for i in range(n)]
        gpd.GeoDataFrame(seg[["seg_id"]], geometry=rows, crs=g.crs).to_file(d / "datasets/known_fault_segments.gpkg",
                                                                            layer="segments", driver="GPKG")
    return g


def test_interrupted_install_is_completed_from_its_rebuild(tmp_path):
    from gems_bench.artifacts import validation as V
    from gems_bench.pipeline import eda_rebuild as R

    src = tmp_path / "outputs/eda_rebuild/v1"
    _segment_group(src)
    eda = tmp_path / "outputs/eda"
    (eda / "datasets").mkdir(parents=True)
    # the session died after the first copy of the group
    (eda / "datasets/known_fault_segments.parquet").write_bytes((src / "datasets/known_fault_segments.parquet").read_bytes())
    (eda / R.INSTALL_STATE).write_text(json.dumps({"status": "in_progress", "version": "v1", "groups": ["segments"]}))
    assert R.group_states(eda)["segments"] == V.PARTIAL
    rep = R.rebuild_minimal(tmp_path, groups=["segments"])
    assert rep["resumed_install"] == ["segments"] and rep["blocked"] == {}
    assert R.group_states(eda)["segments"] == V.OK
    assert R.install_state(tmp_path)["status"] == "complete"


def test_partial_historical_group_stays_blocked(tmp_path):
    from gems_bench.pipeline import eda_rebuild as R

    eda = tmp_path / "outputs/eda"
    _segment_group(eda, gpkg=False)                    # historical upload missing its GeoPackage, no install state
    rep = R.rebuild_minimal(tmp_path, groups=["segments"])
    assert rep["blocked"] == {"segments": "partial"} and rep["installed"] == []
    assert not (eda / "datasets/known_fault_segments.gpkg").exists()


def test_preflight_names_missing_geo_stack(monkeypatch):
    import builtins

    from gems_bench.pipeline import eda_rebuild as R

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "geopandas":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(PipelineStop, match="geopandas"):
        R._preflight()


# ---------------------------------------------------------------------------- 3. native segment attributes
def _native(tmp_path, g):
    labels = np.zeros(g.shape, bool)
    labels[10:50, 5] = True                            # under segment 0
    labels[10:50, 20] = True                           # under segment 1
    labels[60, 60] = True                              # far from every segment
    return SimpleNamespace(labels=labels, grid=g, shape=g.shape, paths=SimpleNamespace(eda=tmp_path / "outputs/eda"))


def test_native_segments_schema_grid_and_provenance(tmp_path):
    from gems_bench.artifacts import validation as V
    from gems_bench.pipeline.eda_rebuild import SEG_PROVENANCE, build_native_segments
    from gems_bench.segments import load_segment_raster
    from test_pipeline_artifacts import native

    g = _segment_group(tmp_path / "outputs/eda")
    data = _native(tmp_path, g)
    made = build_native_segments(data, tmp_path)
    assert made == ["outputs/benchmark/datasets/native_segment_ids.tif",
                    "outputs/benchmark/datasets/segment_attributes.parquet"]
    bench = tmp_path / "outputs/benchmark"
    assert V.validate_native_segments(bench, grid=native(g), eda_dir=tmp_path / "outputs/eda").ok
    seg = load_segment_raster(bench / "datasets/native_segment_ids.tif")
    assert seg.shape == g.shape and seg.dtype == np.int32
    assert set(np.unique(seg[seg >= 0])) <= {0, 1, 2}
    assert (seg[~data.labels] == -1).all() and seg[60, 60] == -1           # only label pixels, within 300 m
    tab = pd.read_parquet(bench / "datasets/segment_attributes.parquet")
    assert {"seg_id", "length_km", "primary_archetype", "length_bin", "strike_sector"} <= set(tab.columns)
    assert len(tab) == 3 and tab.primary_archetype.notna().all()           # real records only, no placeholders
    prov = json.loads((bench / "datasets" / SEG_PROVENANCE).read_text())
    assert 0 < prov["label_px_assigned_frac"] < 1 and prov["n_segments"] == 3
    assert build_native_segments(data, tmp_path) == []                       # idempotent


def test_native_segments_detect_a_changed_eda_and_missing_eda(tmp_path):
    from gems_bench.artifacts import validation as V
    from gems_bench.pipeline.eda_rebuild import build_native_segments
    from test_pipeline_artifacts import native

    with pytest.raises(PipelineStop, match="Phase-1 segment artifacts are missing") as e:
        build_native_segments(SimpleNamespace(), tmp_path)                   # no EDA: useful error, nothing written
    assert "pipe.step('geological_eda')" in e.value.recovery
    assert not (tmp_path / "outputs/benchmark/datasets/native_segment_ids.tif").exists()
    g = _segment_group(tmp_path / "outputs/eda")
    build_native_segments(_native(tmp_path, g), tmp_path)
    _segment_group(tmp_path / "outputs/eda", n=2)                           # a new EDA version: different seg ids
    rep = V.validate_native_segments(tmp_path / "outputs/benchmark", grid=native(g), eda_dir=tmp_path / "outputs/eda")
    assert rep.status == V.INCOMPATIBLE
    with pytest.raises(PipelineStop, match="another Phase-1 segment set"):
        build_native_segments(_native(tmp_path, g), tmp_path)


# ---------------------------------------------------------------------------- 4. notebook order
def _notebook_steps() -> list[str]:
    import re

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                    for c in nb["cells"] if c["cell_type"] == "code")
    return re.findall(r"pipe\.step\('([a-z_]+)'\)", src)


def test_notebook_calls_every_stage_after_its_prerequisites():
    from gems_bench.pipeline.dependencies import topological_order
    from gems_bench.pipeline.stages import build_registry

    R = build_registry()
    steps = _notebook_steps()
    assert sorted(steps) == sorted(R.names()) and len(steps) == len(set(steps))
    for i, name in enumerate(steps):
        for d in (*R[name].deps, *R[name].soft_deps):
            assert steps.index(d) < i, f"{name} is called before its prerequisite {d}"
    assert steps == topological_order(R)                                     # registry = notebook order
    assert steps.index("native_grid") < steps.index("dem_inventory")         # the reported BLOCKED


def test_notebook_has_no_token_url_or_signed_link():
    for nb in (REPO / "notebooks" / "colab").glob("*.ipynb"):
        txt = nb.read_text(encoding="utf-8")
        assert "{token}@github.com" not in txt, nb.name
        assert "X-Amz-Signature" not in txt and "X-Amz-Credential" not in txt, nb.name


# ---------------------------------------------------------------------------- 5. failure policies
def _crit_chain(calls, boom, critical=True):
    R = StageRegistry()
    R.add(file_stage("a", calls=calls))
    R.add(file_stage("b", ("a",), calls=calls, behaviour={"b": boom}, critical=critical))
    R.add(file_stage("c", ("b",), calls=calls))
    R.add(file_stage("d", ("c",), calls=calls))
    R.add(file_stage("e", ("a",), calls=calls))                              # independent of b
    R.check()
    return R


def _boom(pc):
    class Broken:
        x = undefined_name  # noqa: F821  (NameError in a class body, as on Colab)

    return Broken


def test_interactive_policy_stops_on_critical_failure_with_location(roots):
    local, drive = roots
    calls = []
    ex = Executor(make_pc(local, drive, failure_policy="interactive"), _crit_chain(calls, _boom))
    with pytest.raises(StageFailed) as e:
        ex.run(mode="RESUME")
    assert e.value.stage == "b" and "NameError" in str(e.value)
    assert calls == ["a", "b"]                                               # nothing after the root failure
    res = ex.results["b"]
    assert res.outcome == FAILED and "test_pipeline_recovery.py:" in res.location and "Broken" in res.location
    assert "NameError" in res.traceback
    rec = ex.pc.manifests.load("b")
    assert rec["status"] == "failed" and rec["location"] == res.location
    assert (local / "a.0.out").exists()                                      # completed artifacts preserved


def test_interactive_policy_continues_after_non_critical_failure(roots):
    local, drive = roots
    calls = []
    ex = Executor(make_pc(local, drive, failure_policy="interactive"), _crit_chain(calls, _boom, critical=False))
    t = ex.run(mode="RESUME").set_index("stage")
    assert t.loc["b", "outcome"] == FAILED and t.loc["c", "outcome"] == BLOCKED and t.loc["e", "outcome"] == DONE


def test_batch_policy_groups_blocked_stages_under_the_root(roots, capsys):
    local, drive = roots
    calls = []
    ex = Executor(make_pc(local, drive), _crit_chain(calls, _boom))         # default: batch
    t = ex.run(mode="RESUME").set_index("stage")
    assert t.loc["e", "outcome"] == DONE                                     # independent work continues
    assert t.loc["d", "outcome"] == BLOCKED and ex.results["d"].root_causes == ["b"]
    assert "root failure: b (FAILED)" in t.loc["d", "reason"]
    rep = ex.failure_report()
    assert list(rep.root) == ["b"] and rep.iloc[0].blocked == "c, d" and rep.iloc[0].n_blocked == 2
    out = capsys.readouterr().out
    assert "[pipeline] c " not in out and "[pipeline] d " not in out          # no per-stage BLOCKED spam
    assert "ROOT b FAILED" in out and "blocked (2): c, d" in out


def test_error_location_prefers_project_frames():
    try:
        _boom(None)
    except NameError as e:
        loc = error_location(e)
    assert loc.endswith("in Broken") and "test_pipeline_recovery.py:" in loc


def test_failure_policy_is_validated():
    from gems_bench.pipeline.config import ConfigError, PipelineConfig

    assert PipelineConfig.load(None).get("failure_policy") == "batch"       # backward compatible default
    assert PipelineConfig.load(None, flat={"failure_policy": "Interactive"}).get("failure_policy") == "interactive"
    with pytest.raises(ConfigError):
        PipelineConfig.load(None, flat={"failure_policy": "stop"})


# ---------------------------------------------------------------------------- 6. GPU gating on CPU runtimes
def test_gpu_stages_are_disabled_on_cpu_and_cpu_baselines_continue(roots):
    from gems_bench.pipeline import stages as S

    local, drive = roots
    pc = make_pc(local, drive, stages={"run_advanced_models": True, "run_full_inference": True})
    pc._cache["gpu"] = {"cuda": False, "name": None, "torch": "2.11.0+cpu", "torch_cuda": None}
    R = S.build_registry()
    ok, why = R["advanced_models"].enabled(pc)
    assert not ok and "CPU-only build" in why and "Change runtime type" in why
    assert not R["full_inference"].enabled(pc)[0]
    assert S.gpu_blocked_baselines(pc, ["G1", "CNN_A_dti", "P4_B_gated"]) == ["CNN_A_dti", "P4_B_gated"]
    pc._cache["gpu"] = {"cuda": True, "name": "T4", "torch": "2.11.0+cu126", "torch_cuda": "12.6"}
    assert R["advanced_models"].enabled(pc)[0]
    assert S.gpu_blocked_baselines(pc, ["G1", "CNN_A_dti"]) == []


def test_gpu_attached_but_no_device_message(roots):
    local, drive = roots
    pc = make_pc(local, drive)
    pc._cache["gpu"] = {"cuda": False, "torch": "2.11.0+cu126", "torch_cuda": "12.6"}
    assert "no CUDA GPU is attached" in pc.cuda_ready()[1]


# ---------------------------------------------------------------------------- 7. Drive output lists
def test_eda_and_segment_outputs_are_pushed_to_drive(roots):
    from gems_bench.pipeline import stages as S

    local, drive = roots
    pc = make_pc(local, drive)
    R = S.build_registry()
    eda_out = R["geological_eda"].outputs(pc)
    for rel in ("outputs/eda/manifest.json", "outputs/eda/datasets", "outputs/eda/features/lineament_maps.tif",
                "outputs/eda/features/crossmodal_orientation.tif", "outputs/eda/INSTALL_STATE.json",
                "outputs/eda/REBUILD_PROVENANCE.json", "outputs/eda_rebuild"):
        assert rel in eda_out, rel
    assert "outputs/eda/INSTALL_STATE.json" in R["geological_eda"].stage_in(pc)
    seg = R["segment_attributes"].outputs(pc)
    assert seg == R["segment_attributes"].stage_in(pc)
    assert "outputs/benchmark/datasets/native_segments_provenance.json" in seg
    assert R["spatial_folds"].outputs(pc) == ["outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026"]
    assert R["discovery_splits"].outputs(pc) == ["outputs/benchmark/folds/discovery_d3_link1km_s2026"]
    crit = {s.name for s in R if s.critical}
    assert crit == {"bootstrap", "raw_data", "native_grid", "geological_eda", "segment_attributes", "spatial_folds",
                    "discovery_splits", "feature_store"}


# ---------------------------------------------------------------------------- 8. cached raw files and folds
def _raw_on_drive(drive: Path):
    from test_pipeline_artifacts import rnd, tif

    tif(drive / "data/raw/geodawn/gems-geodawn-numerical-features.tif", {f"b{i}": rnd(i) for i in range(19)})
    tif(drive / "data/raw/faults/existing_faults.tif", {"faults": rnd(1)})   # incompressible: > 1 KiB floor
    tif(drive / "data/raw/example_submission.tif", {"sub": rnd(2)})          # accepted alternate location


def test_valid_raw_files_on_drive_are_never_downloaded(roots, monkeypatch):
    pytest.importorskip("rasterio")
    from gems_bench.data import acquisition as A
    from gems_bench.pipeline import stages as S

    local, drive = roots
    _raw_on_drive(drive)
    monkeypatch.setattr(A, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-download")))
    pc = make_pc(local, drive, data={"use_colab_secrets": False})
    out = S.raw_run(pc)
    for k in ("features", "faults", "submission"):
        assert out["files"][k] in ("staged_from_drive", "cached"), (k, out["files"][k])
    assert S.raw_validate(pc, drive).ok


def test_valid_cached_folds_are_reused_not_regenerated(roots):
    pytest.importorskip("rasterio")
    from gems_bench.pipeline import stages as S
    from test_pipeline_artifacts import _folds, grid

    local, drive = roots
    _raw_on_drive(local)
    pc = make_pc(local, drive)
    _folds(grid()).save(local / pc.fold_rel)
    real = S.build_registry()["spatial_folds"]

    def never(pc):
        raise AssertionError("valid folds must not be regenerated")

    R = StageRegistry()
    R.add(Stage("spatial_folds", "folds", run=never, validate=real.validate, fingerprint=real.fingerprint,
                outputs=real.outputs, stage_in=real.stage_in, critical=True))
    R.check()
    t = Executor(pc, R).run(mode="RESUME").set_index("stage")
    assert t.loc["spatial_folds", "outcome"] == "SKIP_CACHED"
    assert pc.manifests.load("spatial_folds")["origin"] == "adopted_existing"


# ---------------------------------------------------------------------------- 9. feature-family prerequisites
def test_eda_feature_family_declares_its_prerequisite(tmp_path):
    from gems_bench.pipeline.stages import FAMILY_EDA_GROUPS, eda_groups_ready

    assert set(FAMILY_EDA_GROUPS) == {"phase2_eda"}                          # raw / phase2 / rs need no EDA
    assert eda_groups_ready(tmp_path, FAMILY_EDA_GROUPS["phase2_eda"]) == ["lineaments", "crossmodal"]
    assert eda_groups_ready(tmp_path, ()) == []
