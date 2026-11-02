"""G06 contracts and configuration (synthetic; no data, no network)."""
import numpy as np
import pytest

from gems_bench.g06 import config as C
from gems_bench.g06.contracts import (ContractError, FeatureSpec, GridSpec, LeakageError, SchemaError,
                                      assert_label_free_columns, validate_feature_order, validate_grid,
                                      validate_model_input, validate_probability, validate_target_separation)


def test_feature_order_is_strict():
    validate_feature_order(["a", "b", "c"], ["a", "b", "c"])
    with pytest.raises(SchemaError):
        validate_feature_order(["a", "b", "c"], ["a", "c", "b"])
    with pytest.raises(SchemaError):
        validate_feature_order(["a", "b"], ["a", "b", "c"])


def test_model_input_schema_hash():
    validate_model_input(["x", "y"], ["x", "y"], "h1", "h1")
    with pytest.raises(SchemaError):
        validate_model_input(["x", "y"], ["x", "y"], "h1", "h2")


@pytest.mark.parametrize("col", ["y_near_truth", "target", "system_id", "near_share", "cls", "weight",
                                 "withheld_flag", "label_conf", "inner_fold"])
def test_truth_columns_refused(col):
    with pytest.raises(LeakageError):
        assert_label_free_columns(["mag_rtp_hgm_s300", col])


def test_geological_columns_allowed():
    assert_label_free_columns(["mag_rtp_hgm_s300", "struct_x_rtp__iso_grav_anom_agreew_s300", "kn_known_dist_km_mean",
                               "ev_w2_rank_mean", "seg_length_km", "tex_rtp_ent_w15"])


def test_target_separation():
    with pytest.raises(LeakageError):
        validate_target_separation(["a", "target"], ["target"])


def test_grid_and_probability():
    g = GridSpec(10, 12, "EPSG:32611", (100.0, 0.0, 5e5, 0.0, -100.0, 4e6), 100.0)
    validate_grid(g, g, 32611)
    with pytest.raises(ContractError):
        validate_grid(GridSpec(10, 13, g.crs, g.transform, 100.0), g)
    with pytest.raises(ContractError):
        validate_grid(GridSpec(10, 12, g.crs, (100.0, 0, 5e5 + 50, 0, -100.0, 4e6), 100.0), g)
    validate_probability(np.array([0.0, 0.5, 1.0, np.nan]))
    with pytest.raises(ContractError):
        validate_probability(np.array([0.2, 1.3]))


def test_feature_spec_roundtrip():
    s = FeatureSpec("mag_rtp_hgm_s300", "magnetic", "physics", "hgm", "nT/km", "rtp", "gauss", 300.0, (0, 10),
                    params={"a": 1})
    t = FeatureSpec.from_row(s.to_row())
    assert t.feature_id == s.feature_id and t.scale_m == 300.0 and t.expected_range == (0, 10)
    assert t.permitted_tracks == ("A", "B") and t.params == {"a": 1}


def test_default_config_validates():
    cfg = C.load_config()
    assert cfg["run"]["TRAIN_LIGHTGBM"] is False and cfg["run"]["PREFLIGHT"] is True
    assert len(cfg["feature_store"]["bands"]) == 19
    assert cfg["benchmarks"]["evaluation"]["primary"]["rep"] in cfg["benchmarks"]["evaluation"]["reps"]


def test_config_rejects_bad_values():
    with pytest.raises(C.ConfigError):
        C.load_config(overrides={"notebook": {"manual_upload_only": False}})
    with pytest.raises(C.ConfigError):
        C.load_config(overrides={"run": {"TRAIN_EVERYTHING": True}})
    with pytest.raises(C.ConfigError):
        C.load_config(overrides={"feature_store": {"groups": {"texture": {"windows": [4]}}}})
    with pytest.raises(C.ConfigError):
        C.load_config(overrides={"benchmarks": {"pixel_experiments": {"X": {"families": ["truth"],
                                                                             "model": "lightgbm"}}}})


def test_run_id_identity():
    cfg = C.load_config()
    src = {"features": {"sha256": "aaa"}, "faults": {"sha256": "bbb"}}
    a = C.compute_run_id(cfg, src, "c1")
    assert a.startswith("g06-") and a == C.compute_run_id(cfg, src, "c2")      # data mode: commit-agnostic
    moved = C.load_config(overrides={"notebook": {"local_root": "/elsewhere", "drive_root": "/x"}})
    assert C.compute_run_id(moved, src, "c1") == a                            # paths never change identity
    assert C.compute_run_id(cfg, {**src, "features": {"sha256": "zzz"}}, "c1") != a
    changed = C.load_config(overrides={"feature_store": {"groups": {"magnetic": {"scales_m": [100, 300]}}}})
    assert C.compute_run_id(changed, src, "c1") != a
    trained = C.load_config(overrides={"benchmarks": {"models": {"lightgbm": {"num_leaves": 63}}}})
    assert C.compute_run_id(trained, src, "c1") == a                          # training params: same feature store
    strict = C.load_config(overrides={"notebook": {"run_id_mode": "commit"}})
    assert C.compute_run_id(strict, src, "c1") != C.compute_run_id(strict, src, "c2")


def test_stage_config_scopes():
    cfg = C.load_config()
    a = C.stage_config(cfg, "train_pixel")
    b = C.stage_config(C.load_config(overrides={"benchmarks": {"models": {"lightgbm": {"num_leaves": 63}}}}),
                       "train_pixel")
    assert a != b
    assert C.stage_config(cfg, "features") == C.stage_config(
        C.load_config(overrides={"benchmarks": {"models": {"lightgbm": {"num_leaves": 63}}}}), "features")
