"""G06 persistence: fingerprints, atomic writes, cache validation (incl. corruption), archiving, inventory."""
import json

import numpy as np
import pandas as pd

from gems_bench.g06 import persistence as PS


def test_stable_hash_and_code_identity():
    assert PS.stable_hash({"a": 1, "b": [1, 2]}) == PS.stable_hash({"b": [1, 2], "a": 1})
    assert PS.stable_hash({"a": 1}) != PS.stable_hash({"a": 2})
    ci = PS.code_identity(["config", "contracts"])
    assert set(ci) == {"config", "contracts"} and all(len(v) == 16 for v in ci.values())


def test_artifact_fingerprint(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    a = PS.artifact_fingerprint(p, "sha256")
    assert a["bytes"] == 3 and len(a["sha256"]) == 64
    p.write_bytes(b"abd")
    assert PS.artifact_fingerprint(p, "sha256")["sha256"] != a["sha256"]
    assert PS.artifact_fingerprint(tmp_path / "missing") is None
    d = PS.artifact_fingerprint(tmp_path, "sha256")
    assert d["n_files"] == 1


def test_cached_stage_validation_and_corruption(tmp_path):
    PS.write_table(tmp_path / "out/t.csv", pd.DataFrame({"a": [1], "b": [2]}))
    PS.write_table(tmp_path / "out/t.parquet", pd.DataFrame({"a": [1]}))
    outs = [("out/t.csv", ("a", "b")), ("out/t.parquet", ("a",))]
    PS.save_stage_manifest(tmp_path, "s", PS.COMPLETE, "fp1", outs, {"feature_schema_hash": "h"})
    ok, probs = PS.validate_cached_stage(tmp_path, "s", "fp1", outs, schema_hash="h")
    assert ok and not probs
    assert not PS.validate_cached_stage(tmp_path, "s", "fp2", outs)[0]                  # fingerprint
    assert not PS.validate_cached_stage(tmp_path, "s", "fp1", outs, schema_hash="x")[0]  # feature schema
    assert not PS.validate_cached_stage(tmp_path, "s", "fp1", [("out/t.csv", ("a", "zz"))])[0]   # columns
    (tmp_path / "out/t.parquet").write_bytes(b"not a parquet file")                     # corruption
    ok, probs = PS.validate_cached_stage(tmp_path, "s", "fp1", outs)
    assert not ok and any("unreadable" in p for p in probs)
    PS.save_stage_manifest(tmp_path, "f", PS.FAILED, "fp1", [], {})
    assert not PS.validate_cached_stage(tmp_path, "f", "fp1", [])[0]                     # FAILED never reused
    PS.manifest_path(tmp_path, "g").parent.mkdir(parents=True, exist_ok=True)
    PS.manifest_path(tmp_path, "g").write_text("{broken", encoding="utf-8")
    assert not PS.validate_cached_stage(tmp_path, "g", "fp1", [])[0]                     # unreadable manifest


def test_atomic_write_leaves_no_partial(tmp_path):
    p = tmp_path / "x.json"
    PS.write_json(p, {"a": np.int64(3), "b": np.float32(1.5), "c": np.array([1, 2])})
    assert json.loads(p.read_text())["c"] == [1, 2]
    try:
        PS.atomic_write(tmp_path / "y.txt", lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
    except RuntimeError:
        pass
    assert not (tmp_path / "y.txt").exists()
    assert not [f for f in tmp_path.iterdir() if "y" in f.name]


def test_archive_never_silently_overwrites(tmp_path):
    PS.write_table(tmp_path / "evaluation/metrics.csv", pd.DataFrame({"v": [1]}))
    moved = PS.archive_outputs(tmp_path, "evaluate", ["evaluation/metrics.csv", "evaluation/none.csv"], "oldfp12345xx")
    assert moved == ["evaluation/metrics.csv"]
    assert (tmp_path / "history/evaluate/oldfp12345/evaluation/metrics.csv").exists()
    assert not (tmp_path / "evaluation/metrics.csv").exists()
    idx = pd.read_csv(tmp_path / "history/index.csv")
    assert idx.stage.tolist() == ["evaluate"]


def test_artifact_inventory(tmp_path):
    PS.write_table(tmp_path / "a/b.csv", pd.DataFrame({"x": [1]}))
    (tmp_path / "tiles").mkdir()
    (tmp_path / "tiles/t.npz").write_bytes(b"0")
    T = PS.write_artifact_inventory(tmp_path)
    assert "a/b.csv" in set(T.path) and not any(p.startswith("tiles/") for p in T.path)
    assert (tmp_path / "manifests/artifact_inventory.csv").exists()
