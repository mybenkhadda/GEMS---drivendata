"""Recovery-first feature pipeline (synthetic data only): interrupted writes, verified / reconstructed manifests,
orphans, duplicates, versions, resolution changes, resumable DEM chunks, split v2 gates, checkpoint restoration."""
from __future__ import annotations

import copy
import json
import shutil

import numpy as np
import pandas as pd
import pytest

MATCH = {"geophysical": "MATCH", "dem": "MATCH", "label": "MATCH"}


def _pipe(cfg, root, **over):
    from gems_bench.g11.config import deep_merge
    from gems_bench.g11.pipeline import G11Pipeline

    c = deep_merge(copy.deepcopy(cfg), {"output_root": str(root), "scratch_root": str(root.parent / f"{root.name}_scr"),
                                        "dem": {"allow_unknown_vertical": True}, **over})
    P = G11Pipeline(c, log=lambda *a: None)
    P._legacy_check = dict(MATCH)
    return P


@pytest.fixture(scope="module")
def built(synth, tmp_path_factory):
    """One complete recovery-first run (audit -> ingest -> features) on the synthetic DEM fixture."""
    root = tmp_path_factory.mktemp("rec") / "out"
    P = _pipe(synth["cfg"], root)
    P.audit()
    P.ingest()
    stats = P.features()
    return {"P": P, "root": root, "stats": stats, "cfg": P.cfg}


def test_fresh_run_finalizes_versioned_dataset(built):
    P = built["P"]
    d = P.ctx.root / "03_features" / "datasets" / "g11_features_v1"
    for f in ("COMPLETE.json", "dataset_manifest.json", "feature_schema.parquet", "label_schema.parquet",
              "valid_mask.npy", "dem_availability.npy", "feature_availability.parquet", "native_window_index.parquet",
              "fault_geometry_registry.parquet", "processing_config.json", "source_provenance.json"):
        assert (d / f).is_file(), f
    man = json.loads((d / "dataset_manifest.json").read_text())
    assert man["dem_statistics_resolution_m"] == 2.0 and not man["contains_split_assignments"]
    sch = pd.read_parquet(d / "feature_schema.parquet")
    assert not sch.name.str.startswith("Y_").any()                                  # labels kept separate
    assert sch[sch.dep_class == "dem"].layer_id.str.endswith("@r2m").all()
    assert set(np.unique(np.load(d / "dem_availability.npy"))) <= {0, 1, 2}
    assert P.validate_features("FULL").passed


def test_split_loss_and_training_changes_do_not_invalidate_features(built):
    from gems_bench.g11.config import deep_merge

    P2 = _pipe(deep_merge(built["cfg"], {"spatial_splits": {"seed": 7}, "training": {"loss": "focal", "lr": 1e-3},
                                         "models": {"xgb": {"max_depth": 3}}}), built["root"])
    before = {p: p.stat().st_mtime_ns for p in (P2.ctx.root / "03_features").rglob("*.npy")}
    assert P2.features() == built["stats"]                                          # REUSED, identical stats
    assert {p: p.stat().st_mtime_ns for p in (P2.ctx.root / "03_features").rglob("*.npy")} == before


def test_same_version_other_spec_is_refused(built):
    from gems_bench.g11.config import deep_merge
    from gems_bench.g11.pipeline import StageGateError

    P2 = _pipe(deep_merge(built["cfg"], {"dem": {"stats_resolution_m": 5.0}}), built["root"])
    with pytest.raises(StageGateError, match="another specification"):
        P2.features()


def test_resolution_change_new_version_computes_only_dem_layers(built):
    from gems_bench.g11.config import deep_merge

    P2 = _pipe(deep_merge(built["cfg"], {"dem": {"stats_resolution_m": 5.0},
                                         "features": {"version": "v_dem5", "policy": "RESUME_MISSING"}}), built["root"])
    geo_before = {p: p.stat().st_mtime_ns for p in (P2.ctx.root / "03_features" / "geophysical").glob("*.npy")}
    rep = P2.recover()
    to_do = rep.missing[rep.missing.kind == "layer"]
    assert len(to_do) and set(to_do.dep_class) == {"dem"}                          # geophysics + labels reused
    st = P2.process_features()
    assert st["complete"] and st["computed"] == len(to_do)
    P2.finalize_features()
    assert {p: p.stat().st_mtime_ns for p in (P2.ctx.root / "03_features" / "geophysical").glob("*.npy")} == geo_before
    assert list((P2.ctx.root / "03_features" / "terrain_100m" / "r5m").glob("dem100_*.npy"))
    assert list((P2.ctx.root / "03_features" / "terrain_100m" / "r2m").glob("dem100_*.npy"))   # 2 m kept apart
    sch = pd.read_parquet(P2.ctx.root / "03_features" / "datasets" / "v_dem5" / "feature_schema.parquet")
    assert sch[sch.dep_class == "dem"].layer_id.str.endswith("@r5m").all()


def test_corrupt_layer_detected_and_only_it_recomputed(built, tmp_path):
    from gems_bench.g11.pipeline import G11Pipeline

    root = tmp_path / "copy"
    shutil.copytree(built["root"], root)
    P = _pipe(built["cfg"], root, features={"policy": "RESUME_MISSING", "version": "g11_features_v2"})
    tgt = next((root / "03_features" / "geophysical").glob("magnetic_gm_*.npy"))
    data = tgt.read_bytes()
    tgt.write_bytes(data[: len(data) // 2])                                         # truncated (interrupted) write
    (root / "03_features" / "geophysical" / "gravity_coh_200m.partial.npy").write_bytes(b"\x93NUMPY")
    rep = P.recover()
    t = rep.table
    assert t[t.relpath.str.endswith(tgt.name)].status.iloc[0] == "INCOMPLETE"
    assert (t.status == "INCOMPLETE").sum() >= 2                                    # + the .partial file
    assert (t.status == "VERIFIED_COMPLETE").sum() > 20
    st = P.process_features()
    assert st["complete"] and st["missing_at_start"] == 1 and st["computed"] >= 1
    assert isinstance(P, G11Pipeline)


def _make_legacy(built, root):
    """Legacy layout: layers + feature_registry.parquet + manifests/features.json COMPLETE, no completion manifest."""
    from gems_bench.g11.registry import RunContext, StageStore

    src = built["root"] / "03_features"
    dst = root / "03_features"
    sch = pd.concat([pd.read_parquet(src / "datasets" / "g11_features_v1" / "feature_schema.parquet"),
                     pd.read_parquet(src / "datasets" / "g11_features_v1" / "label_schema.parquet")])
    for r in sch.itertuples():
        legacy_rel = r.relpath.replace("/r2m/", "/")                                 # legacy had no resolution folder
        (dst / legacy_rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / r.relpath, dst / legacy_rel)
    reg = pd.DataFrame({"name": sch.name, "family": sch.family, "scale_m": sch.scale_m, "unit": sch.unit,
                        "source": sch.source, "label_derived": sch.dep_class == "label",
                        "finite_fraction": sch.finite_fraction, "fingerprint": "legacyfp",
                        "relpath": sch.relpath.str.replace("/r2m/", "/", regex=False)})
    for d in ("01_audit", "02_ingestion"):
        shutil.copytree(built["root"] / d, root / d, dirs_exist_ok=True)
    reg_p = dst / "feature_registry.parquet"
    reg.to_parquet(reg_p, index=False)
    cfg = {**built["cfg"], "output_root": str(root)}
    ctx = RunContext(cfg)
    ctx.source_hashes = built["P"].ctx.source_hashes
    StageStore(ctx).record("features", "legacyfp", [reg_p], {"n_features": len(reg)})
    StageStore(ctx).record("ingest", "legacyfp", [], {})
    return reg


def test_legacy_layers_reconstructed_orphans_and_duplicates(built, tmp_path):
    root = tmp_path / "legacy"
    reg = _make_legacy(built, root)
    f = root / "03_features"
    one = f / reg.relpath.iloc[0]
    shutil.copyfile(one, f / one.name)                                              # duplicate at another path
    orphan = f / "geophysical" / "not_in_registry.npy"
    np.save(orphan, np.zeros((3, 3), np.float32))                                   # wrong shape, unregistered
    P = _pipe(built["cfg"], root)
    rep = P.recover()
    t = rep.table
    rec = t[t.status == "RECOVERABLE"]
    assert len(rec) == len(reg)                                                     # every legacy layer verified
    assert t[t.relpath == one.name].status.iloc[0] == "INCOMPATIBLE"                # duplicate never used
    assert t[t.relpath.str.endswith("not_in_registry.npy")].status.iloc[0] == "INCOMPLETE"
    assert (f / "layer_manifest.parquet").is_file() and (root / "recovery" / "missing_chunks.parquet").is_file()
    assert (root / "recovery" / "feature_recovery_report.json").is_file()
    st = P.process_features()
    assert st["computed"] == 0 and st["complete"]                                   # nothing recomputed
    P.finalize_features()
    assert P.store.version == "g11_features_v1"
    rep2 = P.recover()                                                              # next launch: stat-only path
    assert (rep2.table.status == "VERIFIED_COMPLETE").sum() == len(reg)
    assert rep2.counts["files_fully_read"] <= 8                                     # sampled re-hash only


def test_legacy_rejected_when_content_changed_or_config_unverifiable(built, tmp_path):
    root = tmp_path / "legacy2"
    reg = _make_legacy(built, root)
    p = root / "03_features" / reg.relpath.iloc[3]
    a = np.load(p)
    a[np.isfinite(a)] = np.nan                                                       # finite fraction now differs
    np.save(p, a)
    P = _pipe(built["cfg"], root)
    t = P.recover().table
    assert t[t.relpath == reg.relpath.iloc[3]].status.iloc[0] == "INCOMPATIBLE"
    _make_legacy(built, tmp_path / "legacy3")
    P3 = _pipe(built["cfg"], tmp_path / "legacy3")
    P3._legacy_check = {k: "UNVERIFIABLE: test" for k in MATCH}
    t3 = P3.recover().table
    assert not (t3.status == "RECOVERABLE").any()


def test_dem_engine_resume_matches_full_aggregation(synth, grid, tmp_path):
    from gems_bench.g11.dem_catalog import build_dem_catalog, coverage_100m
    from gems_bench.g11.dem_engine import (TileCache, assemble_dem_layers, chunk_plan, committed_chunks,
                                           process_missing_feature_chunks, validate_feature_chunk)
    from gems_bench.g11.dem_reader import DEMReader, aggregate_dem_to_master, dem_stat_names

    cfg = copy.deepcopy(synth["cfg"])
    cfg["dem"].update(stats_resolution_m=10.0, engine_chunk_cells=20, allow_unknown_vertical=True)
    cat = build_dem_catalog(synth["dem_dir"], grid.crs, cfg)
    cov = coverage_100m(cat, grid)["covered"] & grid.valid
    tc = TileCache(cat, 10.0, tmp_path / "cache", workers=2, log=lambda *a: None)
    tab = tc.ensure()
    assert len(tab) == 2 and not tab.get("error", pd.Series(dtype=object)).notna().any()
    assert tc.ensure().cached.all()                                                  # each tile read once
    reader = tc.reader(cfg, True)
    plan = chunk_plan(grid, cov, 20)
    cdir = tmp_path / "chunks"
    process_missing_feature_chunks(reader, grid, plan.iloc[:1], cfg, 10.0, cdir, tmp_path / "loc", "s", lambda *a: None)
    assert committed_chunks(cdir, "s") == {plan.chunk_id.iloc[0]}                    # interrupted after 1 chunk
    todo = plan[~plan.chunk_id.isin(committed_chunks(cdir, "s"))]
    process_missing_feature_chunks(reader, grid, todo, cfg, 10.0, cdir, tmp_path / "loc", "s", lambda *a: None)
    assert committed_chunks(cdir, "s") == set(plan.chunk_id)
    assert not list(cdir.glob("*.partial*"))
    full = assemble_dem_layers(cdir, plan, grid, cfg, 10.0, cov, "s")
    ref = aggregate_dem_to_master(DEMReader(cat, cfg, True), grid, cov, cfg, None, None)
    for n in ("elev_mean", "slope_mean", "relief_ctx100m"):                      # cached 10 m tiles vs direct warp:
        a, b = full[n], ref[n]                                                     # same support, equal up to the
        assert np.array_equal(np.isfinite(a), np.isfinite(b))                      # resampling at tile edges
        d = np.abs(a - b)[np.isfinite(a)]
        assert np.median(d) < 0.01 and (d > 0.05 + 1e-3 * np.abs(b[np.isfinite(a)])).mean() < 0.01
    f = next(cdir.glob("*.npz"))
    f.write_bytes(f.read_bytes()[:-10] + b"0123456789")
    with pytest.raises(RuntimeError, match="sha256"):
        assemble_dem_layers(cdir, plan, grid, cfg, 10.0, cov, "s")
    ok, probs = validate_feature_chunk({n: np.full((2, 2), np.inf, np.float32) for n in dem_stat_names(cfg)}, (2, 2),
                                       dem_stat_names(cfg))
    assert not ok and any("infinite" in p for p in probs)


def test_split_v2_nonempty_and_v1_failure_mode_gated(cfg, grid, synth):
    from gems_bench.g11 import fault_labels as FL
    from gems_bench.g11.spatial_splits import CODES, SPLITS, SplitResult
    from gems_bench.g11.splits_v2 import make_splits_v2, split_gates

    mask = FL.label_raster(synth["raw"] / "existing_faults.tif", grid, 1)
    res = make_splits_v2(grid.valid, mask, grid, cfg)
    assert res.gates.passed.all()
    for k in SPLITS:
        assert res.mask(k).sum() > 0 and (res.mask(k) & mask).sum() > 0
    bad = res.code.copy()
    bad[bad == CODES["test"]] = CODES["train"]                                        # empty test partition
    g = split_gates(SplitResult(bad, res.blocks, res.embargo_m, res.embargo_parts, res.groups), mask, grid, cfg)
    assert not g[g.check == "nonempty_test"].passed.iloc[0]


def test_split_change_reuses_features_and_namespaces_models(built):
    from gems_bench.g11.config import deep_merge

    P2 = _pipe(deep_merge(built["cfg"], {"spatial_splits": {"version": "v2b", "seed": 11}}), built["root"])
    P2.features()
    st = P2.splits()
    assert all(st["pixels"][k] > 0 for k in ("train", "validation", "stack", "test"))
    assert P2.ns.endswith("split_v2b") and "split_v2b" in str(P2.model_dir("E0"))
    assert (P2.ctx.root / "05_splits" / "v2b" / "COMPLETE.json").is_file()


def test_checkpoint_restoration_regenerates_missing_prediction(built):
    pytest.importorskip("xgboost")
    from gems_bench.g11.config import deep_merge

    P = _pipe(deep_merge(built["cfg"], {"sampling": {"pixel_samples_per_split": 4000},
                                        "models": {"xgb": {"n_estimators": 15}}}), built["root"])
    P.features()
    P.splits()
    t = P.train(["E0"])
    assert t.status.iloc[0] == "TRAINED"
    p = P.pred_path("E0")
    ref = p.read_bytes()
    p.unlink()                                                                       # lost map (disconnect)
    inv = P.predict_all()
    assert inv[inv.experiment == "E0"].status.iloc[0] == "REGENERATED" and p.is_file()
    import rasterio

    with rasterio.open(p) as ds:
        a = ds.read(1)
    from io import BytesIO

    with rasterio.MemoryFile(BytesIO(ref)) as mf, mf.open() as ds:
        np.testing.assert_allclose(a, ds.read(1), atol=1e-6, equal_nan=True)
    assert P.train(["E0"]).status.iloc[0] == "CACHED"
    from gems_bench.g11.config import deep_merge as dm

    P3 = _pipe(dm(P.cfg, {"models": {"xgb": {"n_estimators": 16}}}), built["root"])
    P3.train(["E0"])                                                                  # other config: old one archived
    assert list((P3.model_dir("E0") / "archive").glob("*/model.json"))


def test_submission_refuses_wrong_grid(tmp_path, grid):
    from gems_bench.g11.evaluation.official_adapter import (OfficialSchemaError, export_submission,
                                                            load_official_schema)
    from gems_bench.g11.features import write_geotiff

    tpl = write_geotiff(tmp_path / "tpl.tif", np.zeros(grid.shape, np.float32), grid, ["t"])
    schema = load_official_schema(tpl)
    with pytest.raises(OfficialSchemaError, match="shape"):
        export_submission(np.zeros((grid.shape[0] - 1, grid.shape[1]), np.float32), schema, tmp_path / "s.tif", {})


def test_notebook_recovery_first_defaults_and_modes(cfg):
    from gems_bench.g11 import notebook_config as NC

    p = NC.build_plan({}, cfg)
    assert p.mode == "FULL_RESUME" and p.overrides["features"] == {"policy": "REUSE_VERIFIED",
                                                                    "version": "g11_features_v1", "recovery_verify": "AUTO"}
    assert p.overrides["dem"]["stats_resolution_m"] == 10.0 and p.overrides["spatial_splits"]["version"] == "v2"
    assert p.steps[:6] == ["audit", "ingest", "recover", "process_features", "finalize_features", "validate_features"]
    for mode in ("RECOVER_ONLY", "PROCESS_FEATURES_ONLY", "FINALIZE_FEATURES_ONLY", "SPLITS_ONLY", "TRAIN_ONLY",
                 "INFER_ONLY"):
        assert NC.build_plan({"run_mode": mode}, cfg).steps
    assert "process_features" not in NC.build_plan({"run_mode": "TRAIN_ONLY"}, cfg).steps     # training never extracts
    for bad in ({"feature_policy": "ALWAYS"}, {"feature_version": "a/b"}, {"recovery_verify": "SOME"},
                {"feature_time_budget_min": -1}):
        with pytest.raises(NC.NotebookConfigError):
            NC.build_plan(bad, cfg)
    assert any("never recomputes" in n for n in NC.build_plan({"force_recompute": True, "resume": False}, cfg).notes)
