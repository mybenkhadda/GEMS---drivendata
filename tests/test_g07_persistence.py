"""G07 configuration identity, cache validation and corruption handling (synthetic)."""
import pandas as pd
import pytest

from gems_bench.g07 import config as C
from gems_bench.g07 import persistence as PS


def test_default_config_and_first_run_flags():
    cfg = C.load_config()
    r = cfg["run"]
    assert r["PREFLIGHT"] and r["DEM_FEASIBILITY"] and not r["DEM_FEATURES"] and not r["EXPORT_SUBMISSION"]
    assert cfg["dem"]["competition_eligibility_verified"] is False


def test_auto_submit_and_unknown_flags_refused():
    with pytest.raises(C.ConfigError):
        C.load_config(overrides={"notebook": {"auto_submit": True}})
    with pytest.raises(C.ConfigError):
        C.load_config(overrides={"run": {"UPLOAD": True}})
    with pytest.raises(C.ConfigError):
        C.load_config(overrides={"representation": {"tie_break": "random"}})


def test_run_identity_scopes():
    cfg = C.load_config()
    src = {"features": {"sha256": "a"}}
    a = C.compute_run_id(cfg, src, "c1")
    assert a == C.compute_run_id(cfg, src, "c2")
    assert C.compute_run_id(C.load_config(overrides={"dem": {"scales_m": [4, 12]}}), src, "c1") != a     # DEM: new run
    assert C.compute_run_id(C.load_config(overrides={"models": {"lightgbm": {"num_leaves": 63}}}), src, "c1") == a
    assert C.compute_run_id(cfg, {"features": {"sha256": "b"}}, "c1") != a                              # source identity
    assert C.stage_config(cfg, "train_subsurface") != C.stage_config(
        C.load_config(overrides={"subsurface": {"nested_experiments": []}}), "train_subsurface")


def test_cache_validation_and_corruption(tmp_path):
    PS.write_table(tmp_path / "x/t.parquet", pd.DataFrame({"a": [1]}))
    outs = [("x/t.parquet", ("a",))]
    PS.save_manifest(tmp_path, "s", PS.COMPLETE, "fp", outs, {})
    assert PS.validate_cache(tmp_path, "s", "fp", outs)[0]
    assert not PS.validate_cache(tmp_path, "s", "other", outs)[0]
    (tmp_path / "x/t.parquet").write_bytes(b"garbage")
    ok, probs = PS.validate_cache(tmp_path, "s", "fp", outs)
    assert not ok and probs
    PS.save_manifest(tmp_path, "n", PS.NO_GO, "fp", [], {})
    assert not PS.validate_cache(tmp_path, "n", "fp", [])[0]                 # a NO_GO decision is re-assessed


def test_source_fingerprint_cached(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    cache = tmp_path / "cache.json"
    a = PS.source_fingerprint(p, cache)
    b = PS.source_fingerprint(p, cache)
    assert a["sha256"] == b["sha256"] and len(a["sha256"]) == 64 and cache.exists()
