"""Colab orchestration infrastructure: runtime, notebook config adapter, DEM vertical units, external predictions,
SMOKE window, fusion and local staging (synthetic fixtures only)."""
import json

import numpy as np
import pytest

from gems_bench.g11 import colab_runtime as CR
from gems_bench.g11 import notebook_config as NC
from gems_bench.g11.grid import GridSpec


# ------------------------------------------------------------------ runtime
def test_detect_and_configure_runtime(cfg):
    info = CR.detect_runtime(cfg["data_root"])
    assert {"python", "cpu_count", "gpu", "ram_total_gb", "drive_mounted"} <= set(info)
    before = json.dumps({k: cfg[k] for k in ("feature_engineering", "labels", "spatial_splits")}, sort_keys=True)
    c2, adj = CR.configure_runtime(cfg, {**info, "gpu": False, "ram_total_gb": 12.0, "cpu_count": 2})
    assert c2["training"]["amp"] is False and c2["hardware"]["device"] == "cpu" and adj
    assert json.dumps({k: c2[k] for k in ("feature_engineering", "labels", "spatial_splits")}, sort_keys=True) == before
    assert c2["training"]["epochs"] == cfg["training"]["epochs"]          # scientific parameters untouched
    g, _ = CR.configure_runtime(cfg, {"gpu": True, "gpu_name": "Tesla T4", "ram_total_gb": 51, "cpu_count": 8})
    assert g["training"]["batch_size"] == 8 and g["training"]["amp"] is True
    e = CR.estimate_stage_memory("dem_aggregate", {"chunk_cells": 64, "halo_cells": 10, "res_m": 2.0})
    assert e["note"] == "ESTIMATE" and e["estimated_peak_ram_gb"] > 0


def test_stage_tracker_and_report(tmp_path):
    t = CR.StageTracker()
    with t.track("demo", log=lambda *a: None) as rec:
        _ = np.ones((1000, 1000)).sum()
    assert rec["status"] == "COMPLETE" and t.records[0]["seconds"] >= 0
    with pytest.raises(ValueError):
        with t.track("boom", log=lambda *a: None):
            raise ValueError("x")
    assert t.records[-1]["status"] == "FAILED"
    p = CR.save_runtime_report(tmp_path / "r.json", {"gpu": False}, ["a -> b"], t)
    assert json.loads(p.read_text())["stages"][0]["stage"] == "demo"


# ------------------------------------------------------------------ notebook config adapter
def test_notebook_config_validation(cfg):
    plan = NC.build_plan({"run_mode": "FULL"}, cfg)
    assert plan.steps[0] == "audit" and "submit" in plan.steps and plan.overrides["labels"]["training_target"] == "Y_BUFFER_200M"
    assert plan.overrides["models"]["xgb"]["pu_unlabeled_weight"] == 0.3
    for bad in ({"nope": 1}, {"upload_submission": True}, {"run_mode": "X"}, {"training_target_buffer_m": 150},
                {"dem_statistics_resolution_m": 3.0}, {"models": ["resnet"]}, {"resume": True, "force_recompute": True},
                {"use_gpu": "yes"}):
        with pytest.raises(NC.NotebookConfigError):
            NC.build_plan(bad, cfg)
    smoke = NC.build_plan({"run_mode": "SMOKE_TEST"}, cfg)
    assert smoke.overrides["smoke"]["enabled"] and smoke.overrides["output_root"].endswith("_SMOKE")
    assert "submit" not in smoke.steps and smoke.overrides["training"]["epochs"] == 2
    no_rank = NC.build_plan({"models": ["xgboost"]}, cfg)
    assert "candidates" not in no_rank.steps and "train_segformer" not in no_rank.steps
    assert NC.build_plan({"run_eda": False}, cfg).enabled("eda") is False


# ------------------------------------------------------------------ DEM vertical units
def _tif(path, crs, tags=None, units=None):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32", crs=crs,
                       transform=from_origin(500000, 4000000, 1, 1)) as ds:
        ds.write(np.ones((1, 4, 4), np.float32))
        if tags:
            ds.update_tags(1, **tags)
        if units:
            ds.units = (units,)
    return path


def test_vertical_unit_classification(tmp_path):
    from gems_bench.g11.dem_units import CONFLICT, UNKNOWN, VERIFIED_FT, VERIFIED_M, audit_vertical_units, classify

    assert classify([("band_units", "metre"), ("vertical_crs", "metre")])[0] == VERIFIED_M
    assert classify([("tag", "US survey foot")])[0] == VERIFIED_FT
    assert classify([("band_units", "metre"), ("tag", "foot")])[0] == CONFLICT
    assert classify([])[0] == UNKNOWN
    a = _tif(tmp_path / "plain.tif", "EPSG:32611")
    b = _tif(tmp_path / "metres.tif", "EPSG:32611", units="metre")
    c = _tif(tmp_path / "feet.tif", "EPSG:32611", tags={"VERTICAL_UNIT": "US survey foot"})
    d = _tif(tmp_path / "compound.tif", "EPSG:26911+5703")              # NAD83 / UTM 11N + NAVD88 height (metre)
    t = audit_vertical_units([a, b, c, d]).set_index("tile_id")
    assert t.loc["plain", "class"] == UNKNOWN and np.isnan(t.loc["plain", "vertical_scale"])
    assert t.loc["metres", "class"] == VERIFIED_M and t.loc["metres", "vertical_scale"] == 1.0
    assert t.loc["feet", "class"] == VERIFIED_FT and t.loc["feet", "vertical_scale"] == pytest.approx(1200 / 3937)
    assert t.loc["compound", "class"] == VERIFIED_M                      # read via GTIFF_REPORT_COMPD_CS
    from gems_bench.g11.dem_catalog import dem_files

    files, dups = dem_files([tmp_path, tmp_path])
    assert len(files) == 4 and len(dups) == 4                           # duplicates across folders reported


# ------------------------------------------------------------------ external predictions (G07-style)
def test_g07_style_array_located_by_adjacent_manifest(tmp_path):
    from gems_bench.g11.external_predictions import register_external, resolve_metadata

    master = GridSpec("EPSG:32611", (100.0, 0, 243350.0, 0, -100.0, 4508550.0), (40, 30), np.ones((40, 30), bool))
    g07 = tmp_path / "G07_GEOFORMER"
    g07.mkdir()
    a = np.random.default_rng(0).random((20, 15)).astype(np.float32)
    a[:5] = np.nan                                                      # outside the held-out support
    np.save(g07 / "T1_segformer_test_foldmasked_v2.npy", a)
    np.save(g07 / "T1_segformer_full.npy", a)
    (g07 / "manifest.json").write_text(json.dumps({"crs": "EPSG:32611",
                                                   "transform": [200.0, 0.0, 243350.0, 0.0, -200.0, 4508550.0]}))
    g, how = resolve_metadata(g07 / "T1_segformer_full.npy", a.shape)
    assert how == "ADJACENT_MANIFEST" and g["transform"][0] == 200.0
    lone = tmp_path / "lone"
    lone.mkdir()
    np.save(lone / "x_stack.npy", a)
    shifted = tmp_path / "shifted"
    shifted.mkdir()
    np.save(shifted / "y_test_foldmasked.npy", a)
    (shifted / "manifest.json").write_text(json.dumps({"crs": "EPSG:32611",
                                                       "transform": [200.0, 0.0, 243400.0, 0.0, -200.0, 4508550.0]}))
    t = register_external({"g07_arrays": [str(g07 / "T1_segformer_test_foldmasked_v2.npy"),
                                          str(g07 / "T1_segformer_full.npy"), str(lone / "x_stack.npy"),
                                          str(shifted / "y_test_foldmasked.npy")]}, master, tmp_path / "out")
    t = t.set_index("name")
    assert t.loc["T1_segformer_test_foldmasked_v2", "status"] == "COMPATIBLE"
    assert t.loc["T1_segformer_test_foldmasked_v2", "provenance"] == "OOF_FOLD_MASKED"
    assert t.loc["T1_segformer_full", "provenance"] == "FULL_DOMAIN_INFERENCE"
    assert t.loc["x_stack", "status"].startswith("EXCLUDED: UNLOCATED")
    assert "GRID_DOES_NOT_NEST" in t.loc["y_test_foldmasked", "status"]       # 50 m offset: half a master pixel
    import rasterio

    with rasterio.open(t.loc["T1_segformer_test_foldmasked_v2", "aligned_path"]) as ds:
        m = ds.read(1)
    assert np.allclose(m[10:12, 0:2], a[5, 0]) and np.isnan(m[:10]).all()     # exact 2x2 replication, support kept
    assert t.loc["T1_segformer_test_foldmasked_v2", "finite_support_px"] == 4 * np.isfinite(a).sum()


# ------------------------------------------------------------------ smoke window + cropped reads
def test_smoke_window_and_cropped_reads(synth, grid):
    from gems_bench.g11.fault_labels import label_raster
    from gems_bench.g11.grid import crop_grid, read_on_grid
    from gems_bench.g11.smoke import choose_window, dem_header_coverage

    fault = label_raster(synth["raw"] / "existing_faults.tif", grid)
    cov = dem_header_coverage(synth["tiles"], grid)
    assert cov.sum() > 0
    w = choose_window(fault, cov, grid.valid, grid.res[0], 4.0)
    r0, r1, c0, c1 = w["window"]
    assert w["fault_px_with_dem"] > 0 and (r1 - r0) == 40
    sub = crop_grid(grid, w["window"])
    a = read_on_grid(synth["raw"] / "gems-geodawn-numerical-features.tif", sub, 1)
    full = read_on_grid(synth["raw"] / "gems-geodawn-numerical-features.tif", grid, 1)
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(full[r0:r1, c0:c1]))
    assert np.array_equal(label_raster(synth["raw"] / "existing_faults.tif", sub), fault[r0:r1, c0:c1])


# ------------------------------------------------------------------ fusion
def test_fusion_fit_on_stack_select_on_validation():
    from gems_bench.g11.fusion import compare

    rng = np.random.default_rng(0)
    H = W = 60
    truth = (rng.random((H, W)) < 0.05).astype(int)
    good = np.clip(truth * 0.6 + rng.random((H, W)) * 0.4, 0, 1)
    noise = rng.random((H, W))
    valid = np.ones((H, W), bool)
    idx = np.arange(H * W)
    t, fit, maps = compare({"good": good, "noise": noise}, truth, valid, idx[::2], idx[1::2],
                           {"dem_available": (np.arange(W)[None] > 30).repeat(H, 0).astype(float)})
    ok = t[t.status == "OK"].set_index("method")
    assert {"mean", "median", "max", "best_single", "calibrated_weighted", "logistic_stacking",
            "terrain_conditioned"} <= set(ok.index)
    assert fit["best_single"] == "good"
    assert ok.validation_pr_auc.max() > 0.5 and np.isfinite(maps["logistic_stacking"]).all()


# ------------------------------------------------------------------ local staging
def test_feature_store_localize(tmp_path):
    from gems_bench.g11.features import FeatureStore

    g = GridSpec("EPSG:32611", (100.0, 0, 0, 0, -100.0, 1000.0), (10, 10), np.ones((10, 10), bool))
    st = FeatureStore(tmp_path / "drive", g)
    st.put("a", np.arange(100.0).reshape(10, 10), "geo_magnetic")
    assert st.path_of("a").parent.name == "geophysical"
    man = st.localize(["a"], tmp_path / "local")
    assert man["a"]["sha256"] and str(st.get("a").filename).startswith(str(tmp_path / "local"))
    assert float(st.get("a")[3, 4]) == 34.0
    assert st.localize(["a"], tmp_path / "local")["a"]["sha256"] == man["a"]["sha256"]     # idempotent
