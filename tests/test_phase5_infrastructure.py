"""Phase-5 infrastructure on synthetic inputs: configs, manifests, atomic writes, Drive sync, registry."""
import json

import pandas as pd
import pytest

from gems_bench.artifacts.atomic import atomic_copy, atomic_write_json, sha256_file
from gems_bench.artifacts.registry import ModelRegistry, RegistryError
from gems_bench.artifacts.store import REQUIRED_TABLES, RunDir
from gems_bench.artifacts.sync import DriveSync
from gems_bench.experiment.config import ExperimentConfig, deep_merge, load_yaml_tree
from gems_bench.experiment.manifest import ExperimentManifest, IncompatibleResume


def test_deep_merge_child_wins_lists_replaced():
    a = {"x": {"y": 1, "z": [1, 2]}, "k": 1}
    b = {"x": {"z": [3]}, "n": 2}
    assert deep_merge(a, b) == {"x": {"y": 1, "z": [3]}, "k": 1, "n": 2}


def test_extends_and_semantic_hash(tmp_path):
    (tmp_path / "base.yaml").write_text("kind: cnn5\nseed: 1\nmodel: {base: X, overrides: {a: 1}}\nruntime: {gpu: {}}\n")
    (tmp_path / "child.yaml").write_text("extends: [base.yaml]\nname: c\nmodel: {overrides: {b: 2}}\n")
    raw = load_yaml_tree(tmp_path / "child.yaml")
    assert raw["model"] == {"base": "X", "overrides": {"a": 1, "b": 2}}
    c1 = ExperimentConfig.from_yaml(tmp_path / "child.yaml")
    c2 = ExperimentConfig.from_dict({**c1.raw, "runtime": {"max_stage_minutes": 60}, "notes": "x"})
    assert c1.hash() == c2.hash()                                   # runtime / notes are non-semantic
    c3 = ExperimentConfig.from_dict({**c1.raw, "seed": 2})
    assert c3.hash() != c1.hash()
    assert c1.experiment_id().startswith("c__")


def test_manifest_resume_guard(tmp_path):
    m = ExperimentManifest.load_or_create(tmp_path / "manifest.json", experiment_id="e", config_hash="h1", git_commit="abc")
    m.set_stage("train", "running")
    m2 = ExperimentManifest.load_or_create(tmp_path / "manifest.json")
    assert m2.stage("train")["status"] == "interrupted"             # a session died mid-stage
    with pytest.raises(IncompatibleResume):
        m2.check_compatible("h2", "abc")
    with pytest.raises(IncompatibleResume):
        m2.check_compatible("h1", "def", policy="strict")
    assert m2.check_compatible("h1", "def", policy="config")        # warning only
    m2.mark_unit("cv_fold0", {"best_epoch": 7})
    assert ExperimentManifest.load_or_create(tmp_path / "manifest.json").unit_done("cv_fold0")


def test_atomic_copy_verifies(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"x" * 10_000)
    info = atomic_copy(src, tmp_path / "out" / "a.bin", verify="sha256")
    assert info["sha256"] == sha256_file(tmp_path / "out" / "a.bin")
    assert not list((tmp_path / "out").glob("*.partial"))


def test_drive_sync_skips_unchanged_and_never_deletes(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    (local / "experiments" / "e1").mkdir(parents=True)
    (local / "experiments" / "e1" / "t.csv").write_text("a\n1\n")
    s = DriveSync(local, drive)
    r1 = s.push(["experiments/e1"])
    assert (r1.action == "copy").sum() == 1
    r2 = s.push(["experiments/e1"])
    assert (r2.action == "skip").sum() == 1
    assert s.is_synced(local / "experiments" / "e1" / "t.csv")
    (drive / "experiments" / "e1" / "keep.csv").write_text("x")
    s.push(["experiments/e1"])
    assert (drive / "experiments" / "e1" / "keep.csv").exists()
    back = DriveSync(tmp_path / "other", drive).stage_in(["experiments/e1"])
    assert (back.action == "copy").sum() == 2


def test_rundir_contract(tmp_path):
    rd = RunDir(tmp_path / "exp")
    rd.save_table(pd.DataFrame({"a": [1]}), "fold_metrics")
    rd.append_table(pd.DataFrame({"unit": ["u"], "v": [1]}), "x", key=["unit"])
    rd.append_table(pd.DataFrame({"unit": ["u"], "v": [2]}), "x", key=["unit"])
    assert rd.load_table("x").v.tolist() == [2]
    stubs = rd.finalize_contract("parity")
    assert "fold_metrics" not in stubs and len(stubs) == len(REQUIRED_TABLES) - 1


def test_registry_transitions(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    reg.register("m", "candidate", kind="cnn5")
    with pytest.raises(RegistryError):
        reg.transition("m", "validated", reason="incomplete")        # no scorecard yet
    reg.transition("m", "validated", reason="ok", metrics={"geo_dti": 0.15, "disc_dti_excl": 0.18, "disc_dti_retained_fp": 0.09})
    with pytest.raises(RegistryError):
        reg.transition("m", "promoted", reason="no evidence")
    reg.transition("m", "promoted", reason="gate", evidence={"promotion_decision": {"promote": True}})
    assert reg.champion()["name"] == "m"
    assert len(reg.history()) == 3
    atomic_write_json(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text())["a"] == 1
