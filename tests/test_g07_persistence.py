"""G07 configuration identity, cache validation and corruption handling (synthetic)."""
import pandas as pd
import pytest

from gems_bench.g07 import config as C
from gems_bench.g07 import persistence as PS


def test_default_config_and_first_run_flags():
    cfg = C.load_config()
    r = cfg["run"]
    assert r["PREFLIGHT"] and r["DEM_FEASIBILITY"] and r["DEM_FEATURES"] and r["TRAIN_SURFACE_UNET"] and r["EVALUATE"]
    assert not r["GENERATE_CANDIDATES"] and not r["TRAIN_CANDIDATE_RANKER"]
    assert not r["FULL_INFERENCE"] and not r["EXPORT_SUBMISSION"]
    assert cfg["notebook"]["use_gpu"] is True and cfg["notebook"]["dem_tile_px"] == 512
    assert cfg["dem"]["competition_eligibility_verified"] is False
    assert cfg["dem"]["vertical_unit_override"] is None                 # never hard-coded
    assert cfg["dem"]["download"]["enabled"] is False


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


def test_gating_and_budget_settings_never_invalidate_data():
    cfg = C.load_config()
    src = {"features": {"sha256": "a"}}
    a = C.compute_run_id(cfg, src, "c")
    for ov in ({"dem": {"require_go": False}}, {"dem": {"fine_budget_hours": 1.0}}, {"dem": {"coarse_tile_px": 1024}},
               {"dem": {"vertical_unit_override": "metre"}}, {"surface": {"unet": {"epochs": 2}}}):
        assert C.compute_run_id(C.load_config(overrides=ov), src, "c") == a
    assert C.compute_run_id(C.load_config(overrides={"notebook": {"dem_tile_px": 1024}}), src, "c") == a
    strict, explore = C.load_config(), C.load_config(overrides={"dem": {"require_go": False}})
    assert C.stage_config(strict, "dem_features") == C.stage_config(explore, "dem_features")
    assert C.stage_config(strict, "dem_feasibility") == C.stage_config(explore, "dem_feasibility")


def test_unet_settings_only_touch_the_unet_stage():
    a, b = C.load_config(), C.load_config(overrides={"surface": {"unet": {"epochs": 2}}})
    assert C.stage_config(a, "train_surface_unet") != C.stage_config(b, "train_surface_unet")
    for s in ("dem_features", "datasets", "train_subsurface", "train_surface_classic", "g06_restore"):
        assert C.stage_config(a, s) == C.stage_config(b, s), s
    c = C.load_config(overrides={"surface": {"unet": {"channels": ["dem_hs315"]}}})
    assert C.stage_config(a, "datasets") != C.stage_config(c, "datasets")   # different channels: new patches


def test_vertical_override_changes_dem_products_identity_only():
    a, b = C.load_config(), C.load_config(overrides={"dem": {"vertical_unit_override": "metre"}})
    assert C.stage_config(a, "dem_inventory") != C.stage_config(b, "dem_inventory")
    assert C.stage_config(a, "dem_features") != C.stage_config(b, "dem_features")


def test_training_mode_resolution():
    assert C.training_mode(C.load_config()) == "STRICT_RESEARCH"
    assert C.training_mode(C.load_config(overrides={"dem": {"require_go": False}})) == "EXPLORATORY_TRAINING"
    assert C.training_mode(C.load_config(overrides={"dem": {"training_mode": "EXPLORATORY_TRAINING"}})) == \
        "EXPLORATORY_TRAINING"
    with pytest.raises(C.ConfigError):
        C.load_config(overrides={"dem": {"training_mode": "ANYTHING"}})


def test_tile_products_reach_drive_only_through_the_explicit_filter(tmp_path):
    local, drive = tmp_path / "l", tmp_path / "d"
    drive.mkdir()
    ws = PS.Workspace(local, drive, "sub", True)
    p = ws.path("dem_features/tiles/fp/fine_a.tif")
    p.write_bytes(b"tile")
    PS.sync_artifacts(ws, ["dem_features"])
    assert not (drive / "sub/dem_features/tiles/fp/fine_a.tif").exists()     # the default push skips tile caches
    PS.sync_artifacts(ws, ["dem_features/tiles/fp/fine_a.tif"], PS.ALL_FILES)
    assert (drive / "sub/dem_features/tiles/fp/fine_a.tif").read_bytes() == b"tile"


def test_partial_is_not_reusable(tmp_path):
    PS.write_table(tmp_path / "x/t.parquet", pd.DataFrame({"a": [1]}))
    outs = [("x/t.parquet", ("a",))]
    PS.save_manifest(tmp_path, "s", PS.PARTIAL, "fp", outs, {})
    assert not PS.validate_cache(tmp_path, "s", "fp", outs)[0]


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
