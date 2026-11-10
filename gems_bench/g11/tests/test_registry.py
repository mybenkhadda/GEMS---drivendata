import json

import numpy as np
import pandas as pd
import pytest

from gems_bench.g11 import audit as A
from gems_bench.g11.config import MODE_STAGES, ConfigError, Mode, config_hash, load_config, validate_config
from gems_bench.g11.registry import HashCache, RunContext, StageStore, atomic_write_json, atomic_write_parquet


def test_config_loads_and_hash_is_stable(cfg):
    assert config_hash(cfg) == config_hash(json.loads(json.dumps(cfg)))
    c2 = dict(cfg)
    c2["data_root"] = "/elsewhere"
    assert config_hash(c2) == config_hash(cfg)                 # roots are not part of the identity


def test_config_validation_rejects_bad_fractions(cfg):
    cfg["spatial_splits"]["fractions"]["test"] = 0.5
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_modes_cover_full_pipeline():
    full = MODE_STAGES[Mode.FULL]
    assert full[0] == "audit" and full[-2:] == ["submit", "report"] and full.index("fusion") > full.index("train")
    assert all(s in MODE_STAGES[Mode.FULL] for m in Mode for s in MODE_STAGES[m])


def test_atomic_writes_leave_no_partial(tmp_path):
    p = atomic_write_json(tmp_path / "a.json", {"x": np.float32(1.5), "n": np.int64(3)})
    assert json.loads(p.read_text())["x"] == 1.5
    atomic_write_parquet(tmp_path / "t.parquet", pd.DataFrame({"a": [1, 2], "o": [{"k": 1}, None]}))
    assert not list(tmp_path.glob("*.partial*"))


def test_stage_store_detects_changed_outputs(cfg, tmp_path):
    cfg["output_root"] = str(tmp_path / "o")
    ctx = RunContext(cfg)
    st = StageStore(ctx)
    out = atomic_write_json(tmp_path / "o" / "x.json", {"v": 1})
    fp = st.fingerprint("demo", 1)
    assert st.status("demo", fp) == (False, "MISSING")
    st.record("demo", fp, [out])
    assert st.status("demo", fp)[0]
    assert not st.status("demo", st.fingerprint("demo", 2))[0]
    atomic_write_json(out, {"v": 2})
    assert "STALE" in st.status("demo", fp)[1]
    rec = st.load("demo")
    assert rec["config_hash"] == ctx.cfg_hash and "timestamp_utc" in rec


def test_hash_cache_hashes_once(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    hc = HashCache(tmp_path / "h.jsonl")
    h = hc.sha(f)
    assert HashCache(tmp_path / "h.jsonl").mem[(str(f), 3, f.stat().st_mtime_ns)] == h


def test_discover_and_registry(synth, cfg):
    df = A.discover_artifacts([synth["root"] / "data"], ["*.tif", "*.gpkg"], max_depth=4)
    assert {"template", "geophysics_source", "fault_raster", "dem_tile", "fault_vector"} <= set(df.candidate_role)
    reg, summ = A.build_artifact_registry(cfg)
    assert summ["found"]["geodawn"] and summ["found"]["template"] and not summ["problems"]
    assert any("ABSENT" in s for s in summ["skipped"])           # optional evidence recorded as skipped
    md = A.audit_report_md(reg, summ)
    assert "Fail-fast problems" in md


def test_unlocated_numpy_prediction_is_flagged(tmp_path):
    a = np.random.default_rng(0).random((10, 12)).astype(np.float32)
    np.save(tmp_path / "T1_segformer_test_foldmasked_v2.npy", a)
    m = A.inspect_prediction(tmp_path / "T1_segformer_test_foldmasked_v2.npy")
    assert m.alignment_confidence == "UNLOCATED" and m.split == "test"
    (tmp_path / "T1_segformer_test_foldmasked_v2.json").write_text(json.dumps(
        {"crs": "EPSG:32611", "transform": [200, 0, 0, 0, -200, 0], "split": "test"}))
    assert A.inspect_prediction(tmp_path / "T1_segformer_test_foldmasked_v2.npy").alignment_confidence == "DOCUMENTED"


def test_missing_template_fails_fast(cfg):
    cfg["source_artifacts"]["template"] = ["nope.tif"]
    _, summ = A.build_artifact_registry(cfg, inspect=False)
    with pytest.raises(A.AuditError):
        A.fail_fast(summ)


def test_load_config_default_file():
    c = load_config()
    assert c["official_submission"]["evaluator_command"] is None
