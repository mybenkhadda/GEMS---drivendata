"""End-to-end smoke test of every pipeline stage on a synthetic area with several faults (no real data)."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")

H = W = 160
X0, Y0, PX = 600000.0, 4100000.0, 100.0


@pytest.fixture(scope="module")
def area(tmp_path_factory):
    import geopandas as gpd
    from rasterio.transform import from_origin
    from shapely.geometry import LineString

    from gems_bench.g11.config import load_config
    from gems_bench.g11.tests.conftest import SMALL_V2, _write

    root = tmp_path_factory.mktemp("g11area")
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(3)
    T = from_origin(X0, Y0, PX, PX)
    rr, cc = np.indices((H, W))
    x, y = X0 + PX * (cc + 0.5), Y0 - PX * (rr + 0.5)
    lines, d_all = [], np.full((H, W), np.inf)
    for k in range(10):
        x0 = X0 + rng.uniform(1000, 13000)
        y0 = Y0 - rng.uniform(1000, 13000)
        az = np.radians(rng.choice([20, 60, 120, 160]))
        L = rng.uniform(2500, 4500)
        p0, p1 = (x0, y0), (x0 + L * np.sin(az), y0 + L * np.cos(az))
        lines.append(LineString([p0, p1]))
        t = np.array([np.sin(az), np.cos(az)])
        s = np.clip((x - x0) * t[0] + (y - y0) * t[1], 0, L)
        dx, dy = x - (x0 + s * t[0]), y - (y0 + s * t[1])
        d_all = np.minimum(d_all, np.hypot(dx, dy))
    rtp = (60 * np.exp(-(d_all / 300) ** 2) + rng.normal(0, 3, (H, W))).astype(np.float32)
    grav = (4 * np.exp(-(d_all / 600) ** 2) + rng.normal(0, 0.3, (H, W))).astype(np.float32)
    elev = (1200 + 0.02 * (x - X0) + rng.normal(0, 1, (H, W))).astype(np.float32)
    _write(raw / "gems-geodawn-numerical-features.tif", np.stack([rtp, grav, elev]), T,
           ["rtp", "iso_grav_anom", "det_elev"], nodata=np.nan)
    _write(raw / "existing_faults.tif", (d_all <= 70).astype(np.uint8), T)
    _write(raw / "example_submission.tif", np.zeros((H, W), np.float32), T, nodata=np.nan)
    gpd.GeoDataFrame({"fid": range(len(lines))}, geometry=lines, crs="EPSG:32611").to_file(raw / "faults.gpkg",
                                                                                          driver="GPKG")
    cfg = load_config(overrides={
        "data_root": str(root), "output_root": str(root / "out"), "scratch_root": str(root / "scratch"),
        "geophysical_grid": {"expected_shape": [H, W], "expected_band_count": 3},
        "spatial_splits": {"block_km": 4.0, "embargo_m": 300.0, "v2": SMALL_V2},
        "feature_engineering": {"geophysics": {"scales_m": [200, 400, 800], "wavelet_bands_m": [[200, 800]]}},
        "eda": {"n_bootstrap": 20, "max_samples": 4000, "atlas_sites": 2, "bootstrap_block_km": 4.0},
        "sampling": {"pixel_samples_per_split": 15000, "balance_block_km": 2.0},
        "models": {"xgb": {"n_estimators": 40}, "segment_ranker": {"n_estimators": 40}},
        "candidate_generation": {"min_length_m": 300.0, "threshold_quantile": 0.85, "hysteresis_quantiles": [0.8, 0.9]},
        "ranking": {"length_budgets_km": [5, 15, 30]},
        "evaluation": {"n_bootstrap": 20, "bootstrap_block_km": 4.0}})
    return cfg


def test_full_pipeline_smoke(area):
    from gems_bench.g11.pipeline import G11Pipeline, StageGateError

    P = G11Pipeline(area, log=lambda *a: None)
    summ = P.audit()
    assert summ["found"]["fault_vector"] and not summ["problems"]
    assert P.ingest()["segment_source"] == "vector"
    f = P.features()
    assert f["n_predictors"] > 30
    assert P.features() == f                                    # second call is CACHED (idempotent)
    assert P.eda()["figures"] >= 2
    sp = P.splits()
    assert all(sp["pixels"][k] > 0 for k in ("train", "validation", "stack", "test"))
    with pytest.raises(StageGateError):
        P.evaluate(final=True)                                  # test metrics refused before any selection
    t = P.train(["B0", "E0", "E1"])
    assert set(t.status) >= {"TRAINED"} and (t.experiment == "E1").any()          # E1 skipped: no DEM in this area
    assert P.pred_path("E0").is_file() and (P.model_dir("E0") / "metrics.json").is_file()
    assert list((P.d("train") / "training_data").glob("E0_*/indices_train.npy"))     # persisted sample indices
    fu = P.fusion()
    assert fu.selected.any() and P.pred_path("E5").is_file()
    m = P.evaluate()
    v = m[(m.split == "validation") & (m.status == "EVALUATED")]
    assert len(v) == 3 and v.PR_AUC.notna().all() and set(v.experiment_id) == {"B0", "E0", "E5"}
    for col in ("experiment_id", "model", "features", "target_definition", "train_region", "validation_region",
                "test_region", "PR_AUC", "ROC_AUC", "calibration_error", "candidate_recall", "predicted_length_km",
                "fragmentation", "offline_DTI_proxy", "training_time", "inference_time", "peak_GPU_memory", "peak_RAM"):
        assert any(c.startswith(col) for c in m.columns), col
    assert (P.d("evaluate") / "selection.json").is_file()
    fin = P.evaluate(final=True)
    assert (fin.split == "test").any()
    man = P.infer()
    assert man["n_candidates"] > 0
    rec = P.submit()
    assert rec["validation"]["passed"] and rec["official_evaluation"]["status"] == "OFFICIAL EVALUATOR UNAVAILABLE"
    assert rec["official_DTI"] == "unavailable" and not rec["uploaded"]
    sd = P.d("submit")
    for f in ("g11_best_submission.tif", "submission_manifest.json", "submission_validation.json"):
        assert (sd / f).is_file(), f
    rep = P.report()
    for k in ("md", "html", "leaderboard"):
        assert Path(rep["paths"][k]).is_file()
    assert rep["submission"]["official_DTI"] == "unavailable"
    for f in ("01_audit/artifact_registry.parquet", "01_audit/artifact_compatibility.csv", "01_audit/audit_report.md",
              "01_audit/dem_vertical_units.csv", "02_ingestion/master_grid.json", "02_ingestion/geophysical_registry.parquet",
              "02_ingestion/fault_registry.parquet", "02_ingestion/coverage_report.json",
              "03_features/layer_manifest.parquet", "03_features/datasets/g11_features_v1/COMPLETE.json",
              "03_features/datasets/g11_features_v1/feature_schema.parquet", "recovery/feature_recovery_report.json",
              "04_eda/eda_report.html", "04_eda/feature_contrast.csv",
              "04_eda/model_complementarity.csv", "04_eda/multiscale_analysis.csv", "04_eda/eda_summary.md",
              "05_splits/v2/split_registry.parquet", "05_splits/v2/split_map.png", "05_splits/v2/leakage_report.json",
              "05_splits/v2/COMPLETE.json", "05_splits/v2/split_gates.parquet"):
        assert (P.ctx.root / f).is_file(), f
    for stage in ("audit", "ingest", "features_g11_features_v1", "eda", "splits_v2", P.skey("infer"), P.skey("submit")):
        r = P.stages.load(stage)
        assert r and r["config_hash"] == P.ctx.cfg_hash and r["source_hashes"] and r["completed_utc"]


def test_smoke_mode_subset_end_to_end(synth, tmp_path):
    """SMOKE_TEST plan on the synthetic DEM fixture: window selection, cropped grid, grid-aware reads, training."""
    from gems_bench.g11.config import deep_merge, load_config
    from gems_bench.g11.notebook_config import StepRunner, build_plan
    from gems_bench.g11.pipeline import G11Pipeline
    from gems_bench.g11.tests.conftest import SMALL_V2

    base = load_config()
    plan = build_plan({"run_mode": "SMOKE_TEST", "drive_root": str(synth["root"]), "smoke_window_km": 6.0}, base)
    ov = deep_merge(plan.overrides, {"output_root": str(tmp_path / "smoke"), "scratch_root": str(tmp_path / "scr"),
                                     "geophysical_grid": {"expected_shape": [120, 100], "expected_band_count": 3},
                                     "dem": {"allow_unknown_vertical": True, "context_scales_m": [25, 100]},
                                     "spatial_splits": {"block_km": 1.0, "embargo_m": 200.0, "v2": {
                                         **SMALL_V2, "gates": {**SMALL_V2["gates"], "require_dem_each_split": False}}},
                                     "feature_engineering": {"geophysics": {"scales_m": [200, 400]}},
                                     "eda": {"n_bootstrap": 10, "max_samples": 2000, "atlas_sites": 3},
                                     "models": {"xgb": {"n_estimators": 20}}})
    P = G11Pipeline(load_config(overrides=ov), log=lambda *a: None)
    run = StepRunner(P, plan, log=lambda *a: None)
    s = run("audit", P.audit)
    w = s["smoke_window"]
    assert w["fault_px_with_dem"] > 0 and P.grid.shape == (60, 60)
    run("ingest", P.ingest)
    assert P.mask.shape == P.grid.shape
    run("recover", P.recover)
    run("process_features", P.process_features)
    run("finalize_features", P.finalize_features)
    run("validate_features", P.validate_features)
    assert all(run.results[s]["status"] == "COMPLETE" for s in ("recover", "process_features", "finalize_features",
                                                                  "validate_features"))
    assert P.store.get("magnetic_gm_200m").shape == P.grid.shape
    assert any(n.startswith("dem100_") for n in P.store.reg.name)         # DEM aggregated on the cropped grid
    run("eda", P.eda)
    run("splits", P.splits)
    assert run.results["splits"]["status"] == "COMPLETE"
    t = P.train(["E0"])
    assert t.status.iloc[0] == "TRAINED"
    assert run("submit", lambda: 1) is None and run.results["submit"]["status"] == "SKIPPED"   # not in SMOKE plan
    pytest.importorskip("torch")
    P.cfg["sampling"].update(patch_px=12, patches_train=24, patches_validation=12)
    P.cfg["training"].update(epochs=1, batch_size=4, amp=False, grad_accum=1, loss_ablation=[])
    P.cfg["models"]["gated_fusion"].update(geo_dim=8, dem_dim=4, mode_ablation=[])
    P.cfg["models"]["terrain_encoder"].update(dem_res_m=25.0, dem_px=32)
    P.cfg["models"]["segformer"]["in_channels"] = 3
    P.cfg["inference"].update(tile_px=24, overlap_px=6)
    t4 = P.train(["E3"])
    r4 = t4.iloc[0]
    assert r4.status == "TRAINED" and r4.dem_branch and r4.fusion_mode == "gated"
    assert list((tmp_path / "scr" / "dem_mosaic").glob("dem_25m_*.tif"))           # local mosaic, not Drive reads


def test_deep_segformer_path(area, tmp_path):
    pytest.importorskip("torch")
    import copy

    from gems_bench.g11.pipeline import G11Pipeline

    cfg = copy.deepcopy(area)
    cfg["output_root"] = str(tmp_path / "deep")
    cfg["scratch_root"] = str(tmp_path / "deep_scratch")
    cfg["sampling"].update(patch_px=16, patches_train=40, patches_validation=20)
    cfg["training"].update(epochs=1, batch_size=4, amp=False, grad_accum=1, loss_ablation=["bce_boundary"])
    cfg["models"]["segformer"].update(dims=[8, 16, 24, 32], heads=[1, 2, 2, 4], decoder_dim=16, in_channels=4)
    cfg["inference"].update(tile_px=32, overlap_px=8)
    P = G11Pipeline(cfg, log=lambda *a: None)
    P.audit()
    P.ingest()
    P.features()
    P.splits()
    t = P.train(["E0", "E2"])
    row = t[t.experiment == "E2"].iloc[0]
    assert row.status == "TRAINED" and row.epochs == 1 and row.channels == 4 and row.locally_staged_channels == 4
    assert (t.experiment == "E2__bce_boundary").any() and t[t.experiment == "E2__bce_boundary"].status.iloc[0] == "TRAINED"
    assert (P.model_dir("E2") / "best.pt").is_file() or (P.model_dir("E2") / "last.pt").is_file()
    import rasterio

    with rasterio.open(P.pred_path("E2")) as ds:
        p = ds.read(1)
    v = p[P.grid.valid]
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
    t2 = P.train(["E2"])                                         # resume: cached, not retrained
    assert set(t2.status) == {"CACHED"}
