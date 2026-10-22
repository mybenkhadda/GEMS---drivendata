"""Artifact validation, split consistency, EDA install rules, feature provenance, baseline availability and
acquisition caching - on tiny synthetic rasters (no competition data, network or GPU)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

from gems_bench.artifacts import validation as V  # noqa: E402
from gems_eda.io import Grid, write_geotiff  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
H = W = 64


def grid(x0: float = 500_000.0) -> Grid:
    return Grid(Affine(100.0, 0, x0, 0, -100.0, 4_500_000.0), CRS.from_epsg(32611), H, W, 0.1, 0.1, False)


def tif(path: Path, bands: dict, g: Grid | None = None) -> Path:
    return write_geotiff(path, bands, g or grid())


def rnd(seed=0):
    return np.random.default_rng(seed).random((H, W)).astype(np.float32)


def native(g: Grid | None = None) -> dict:
    g = g or grid()
    return {"crs": g.crs, "transform": g.transform, "height": g.height, "width": g.width}


# ---------------------------------------------------------------------------- single files / alignment
def test_geotiff_checks_detect_html_pages_and_grid_mismatch(tmp_path):
    good = tif(tmp_path / "a.tif", {"probability": rnd()})
    assert V.check_geotiff(good, grid=native(), count=1).ok
    shifted = tif(tmp_path / "b.tif", {"probability": rnd()}, grid(x0=500_050.0))
    assert V.check_geotiff(shifted, grid=native()).status == V.INCOMPATIBLE
    html = tmp_path / "c.tif"
    html.write_text("<html>AccessDenied - link expired</html>" * 50)
    assert V.check_geotiff(html).status == V.INVALID
    assert V.check_geotiff(tmp_path / "missing.tif").status == V.MISSING
    assert V.check_geotiff(good, bands=("probability",)).ok
    assert V.check_geotiff(good, bands=("fused_score",)).status == V.INVALID


def test_tabular_json_npy_checks(tmp_path):
    pd.DataFrame({"seg_id": [1, 2], "length_km": [1.0, 2.0]}).to_parquet(tmp_path / "ok.parquet")
    pd.DataFrame({"seg_id": pd.Series([], dtype=int)}).to_parquet(tmp_path / "empty.parquet")
    assert V.check_parquet(tmp_path / "ok.parquet", columns=("seg_id",)).ok
    assert V.check_parquet(tmp_path / "empty.parquet").status == V.INVALID        # no empty placeholder passes
    assert V.check_parquet(tmp_path / "ok.parquet", columns=("primary_archetype",)).status == V.INVALID
    (tmp_path / "p.json").write_text(json.dumps({"n_folds": 3}))
    assert V.check_json(tmp_path / "p.json", required_keys=("n_folds",)).ok
    assert V.check_json(tmp_path / "p.json", required_keys=("seed",)).status == V.INVALID
    (tmp_path / "e.json").write_text("{}")
    assert V.check_json(tmp_path / "e.json").status == V.INVALID
    np.save(tmp_path / "f.npy", rnd())
    assert V.check_npy(tmp_path / "f.npy", shape=(H, W)).ok
    assert V.check_npy(tmp_path / "f.npy", shape=(H, W + 1)).status == V.INCOMPATIBLE
    np.save(tmp_path / "g.npy", rnd().astype(np.float64))
    assert V.check_npy(tmp_path / "g.npy", shape=(H, W)).status == V.INVALID


def test_report_status_aggregation():
    r = V.Report("x", [V.Check("a", V.OK), V.Check("b", V.MISSING)])
    assert r.status == V.PARTIAL
    assert V.Report("x", [V.Check("a", V.MISSING)]).status == V.MISSING
    assert V.Report("x", [V.Check("a", V.OK), V.Check("b", V.INCOMPATIBLE)]).status == V.INCOMPATIBLE
    assert V.Report("x").status == V.MISSING                                        # an empty set is never "ok"


# ---------------------------------------------------------------------------- splits
def _folds(g: Grid, **params):
    from gems_bench.folds import GeoFolds

    fm = np.zeros((H, W), np.int16)
    fm[:, W // 3:] = 1
    fm[:, 2 * W // 3:] = 2
    roles = np.stack([np.where(fm == k, 1, 0) for k in range(3)]).astype(np.int8)
    p = dict(n_folds=3, block_km=20.0, block_px=200, clusters_per_fold=3, buffer_km=13.0, inner_buffer_km=5.0, seed=2026)
    p.update(params)
    return GeoFolds(g, fm, roles, pd.DataFrame({"block": [0, 1, 2], "fold": [0, 1, 2]}), p)


def test_fold_folder_validation_is_content_and_parameter_based(tmp_path):
    from gems_bench.pipeline.config import DEFAULTS
    from gems_bench.pipeline.context import fold_params

    g = grid()
    exp = fold_params(DEFAULTS["validation"], 2026)
    d = _folds(g).save(tmp_path / "geo_k3_b20km_buf13km_s2026")
    assert V.validate_folds_dir(d, grid=native(g), expected=exp).ok
    assert V.validate_folds_dir(d, grid=native(grid(1e5)), expected=exp).status == V.INCOMPATIBLE
    d2 = _folds(g, clusters_per_fold=4).save(tmp_path / "other")
    rep = V.validate_folds_dir(d2, grid=native(g), expected=exp)
    assert rep.status == V.INCOMPATIBLE and "clusters_per_fold" in rep.summary()
    (d / "blocks.parquet").unlink()
    assert V.validate_folds_dir(d, grid=native(g), expected=exp).status == V.PARTIAL
    assert V.validate_folds_dir(tmp_path / "nothing", grid=native(g), expected=exp).status == V.MISSING


def test_discovery_folder_validation(tmp_path):
    from gems_bench.folds import DiscoverySplits
    from gems_bench.pipeline.config import DEFAULTS
    from gems_bench.pipeline.context import discovery_params

    g = grid()
    sm = np.full((H, W), -1, np.int32)
    sm[10, 5:30] = 0
    roles = np.zeros((3, H, W), np.int8)
    roles[0][sm == 0] = 3
    p = discovery_params(DEFAULTS["validation"], 2026)
    s = DiscoverySplits(g, sm, pd.DataFrame({"system_id": [0], "trace_km": [2.5]}), roles, p)
    d = s.save(tmp_path / "discovery_d3_link1km_s2026")
    assert V.validate_discovery_dir(d, grid=native(g), expected=p).ok
    assert V.validate_discovery_dir(d, grid=native(g), expected={**p, "link_km": 2.0}).status == V.INCOMPATIBLE


def test_split_consistency_folds_partition_the_footprint(tmp_path):
    """make_geo_folds: every valid pixel in exactly one fold; val / buffer / train disjoint per fold."""
    from gems_bench.folds import ROLE, make_geo_folds

    g = Grid(Affine(1000.0, 0, 500_000.0, 0, -1000.0, 4_500_000.0), CRS.from_epsg(32611), 120, 120, 1.0, 1.0, False)
    valid = np.ones((120, 120), bool)
    valid[:, :10] = False
    labels = np.zeros_like(valid)
    labels[20:100:7, 15:110] = True
    f = make_geo_folds(valid, labels, g, n_folds=3, block_km=20.0, clusters_per_fold=3, buffer_km=13.0, seed=2026)
    assert set(np.unique(f.fold_map[valid])) == {0, 1, 2}
    assert (f.fold_map[~valid] == -1).all()
    for k in range(3):
        val, buf, tr = f.val(k), f.buffer(k), f.train(k)
        assert not (val & buf).any() and not (val & tr).any() and not (buf & tr).any()
        assert (f.roles[k][valid] != ROLE["invalid"]).all()


# ---------------------------------------------------------------------------- EDA install rules
def _lineament_tif(path: Path):
    return tif(path, {k: rnd(i) for i, k in enumerate(("fused_score", "fused_fused", "fused_support", "response_terrain"))})


def test_eda_install_whole_groups_never_overwrites(tmp_path):
    from gems_bench.pipeline.eda_rebuild import group_states, install

    src = tmp_path / "outputs/eda_rebuild/v1"
    _lineament_tif(src / "features/lineament_maps.tif")
    eda = tmp_path / "outputs/eda"
    assert group_states(eda)["lineaments"] == V.MISSING
    installed = install(tmp_path, src, ["lineaments"], "v1")
    assert installed == ["outputs/eda/features/lineament_maps.tif"]
    man = json.loads((eda / "manifest.json").read_text())
    assert man["lineament_maps"]["origin"] == "rebuilt"
    prov = json.loads((eda / "REBUILD_PROVENANCE.json").read_text())
    assert prov["rebuilds"][0]["historical_equivalence"] is False
    before = (eda / "features/lineament_maps.tif").read_bytes()
    _lineament_tif(tmp_path / "outputs/eda_rebuild/v2/features/lineament_maps.tif")
    assert install(tmp_path, tmp_path / "outputs/eda_rebuild/v2", ["lineaments"], "v2") == []
    assert (eda / "features/lineament_maps.tif").read_bytes() == before


def test_eda_partial_group_is_reported_not_mixed(tmp_path):
    from gems_bench.pipeline.eda_rebuild import group_states

    eda = tmp_path / "outputs/eda"
    (eda / "datasets").mkdir(parents=True)
    pd.DataFrame({"seg_id": [0], "length_km": [1.0], "azimuth": [10.0], "sinuosity": [1.0]}).to_parquet(
        eda / "datasets/known_fault_segments.parquet")
    assert group_states(eda)["segments"] == V.PARTIAL


def test_eda_rebuild_skips_complete_groups(tmp_path, monkeypatch):
    from gems_bench.pipeline import eda_rebuild

    monkeypatch.setattr(eda_rebuild, "group_states", lambda eda: {"segments": "ok", "lineaments": "ok", "crossmodal": "ok"})
    rep = eda_rebuild.rebuild_minimal(tmp_path)
    assert rep["rebuilt"] == [] and rep["installed"] == []


# ---------------------------------------------------------------------------- feature provenance
def test_feature_spec_hash_and_families():
    from gems_bench.features import _spec
    from gems_bench.pipeline.stages import _family, spec_hash

    a = _spec("terrain:curvature@1km", ("G1",), "terrain", "det_elev", "curvature", 1.0)
    b = _spec("terrain:curvature@1km", ("G1",), "terrain", "det_elev", "curvature", 2.0)
    assert spec_hash(a) == spec_hash(a) and spec_hash(a) != spec_hash(b)
    assert [_family(n) for n in ("raw:rtp", "eda:fused_score", "rs:terrain:tpi@0.3km", "p5:x", "dem:coverage",
                                 "magnetic:thdr@2km")] == ["raw", "phase2_eda", "phase4_rs", "phase5_profiles", "dem",
                                                           "phase2"]


def test_required_features_follow_configuration(tmp_path, monkeypatch):
    from gems_bench.pipeline.config import PipelineConfig
    from gems_bench.pipeline.context import PipelineContext
    from gems_bench.pipeline.stages import required_features

    monkeypatch.setenv("GEMS_ROOT", "")
    cat = pd.DataFrame({"name": ["raw:rtp", "magnetic:thdr@2km", "eda:fused_score", "rs:terrain:tpi@0.3km",
                                 "rs:never:used@9km", "p5:terrain:profile_step@0.5km"]})
    cat["family"] = ["raw", "phase2", "phase2_eda", "phase4_rs", "phase4_rs", "phase5_profiles"]

    def pc(**stages):
        cfg = PipelineConfig.load(None, overrides={"paths": {"local_root": str(tmp_path), "drive_root": "",
                                                             "code_root": str(REPO)}, "stages": stages})
        return PipelineContext(cfg)

    got = required_features(pc(), cat)
    assert "rs:terrain:tpi@0.3km" in got                 # used by the P4_B_gated snapshot
    assert "rs:never:used@9km" not in got and "p5:terrain:profile_step@0.5km" not in got
    assert {"raw:rtp", "magnetic:thdr@2km", "eda:fused_score"} <= set(got)
    assert "rs:terrain:tpi@0.3km" not in required_features(pc(run_advanced_features=False), cat)


# ---------------------------------------------------------------------------- baselines: never fabricate
def _legacy_run(d: Path, status="done", models=False):
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"status": status}))
    (d / "config.json").write_text(json.dumps({"folds": "outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026"}))
    tif(d / "oof.tif", {"oof_probability": rnd()})
    if models:
        (d / "models").mkdir()
        (d / "models" / "fold0.txt").write_text("tree")


def test_baseline_status_never_invents_models(tmp_path):
    from gems_bench.pipeline import baselines as B

    t = B.status_table(tmp_path, grid=native(), n_splits=3)
    assert set(t.status) == {B.NOT_AVAILABLE}
    assert not [c for c in t.columns if "dti" in c or "score" in c]       # scores only come from the scoreboard
    _legacy_run(tmp_path / "outputs/benchmark/experiments/lgbm_G1_s2026")
    s = B.baseline_status(tmp_path, "G1", grid=native(), n_splits=3)
    assert s["status"] == B.CV_ONLY and s["checkpoints"] is False and s["origin"] == "legacy"
    for d in range(3):
        tif(tmp_path / f"outputs/benchmark/experiments/disc{d}_lgbm_G1_s2026/pred.tif", {"probability": rnd(d)})
    assert B.baseline_status(tmp_path, "G1", grid=native(), n_splits=3)["status"] == B.AVAILABLE
    assert B.baseline_status(tmp_path, "G1", grid=native(grid(1e5)), n_splits=3)["status"] == B.NOT_AVAILABLE


def test_baseline_interrupted_and_incompatible_runs(tmp_path):
    from gems_bench.pipeline import baselines as B

    _legacy_run(tmp_path / "outputs/phase4/experiments/p4_P4_B_gated_p256_s2026", status="running")
    assert B.baseline_status(tmp_path, "P4_B_gated", grid=native(), n_splits=3)["status"] == B.INTERRUPTED
    d = tmp_path / "outputs/benchmark/experiments/b0_lineament_fused_s2026"
    _legacy_run(d)
    (d / B.PROVENANCE).write_text(json.dumps({"origin": "regenerated_by_master_pipeline", "folds_sha256": "aaa"}))
    assert B.baseline_status(tmp_path, "B0_lineament", grid=native(), n_splits=3, folds_sha="aaa")["status"] == B.AVAILABLE
    assert B.baseline_status(tmp_path, "B0_lineament", grid=native(), n_splits=3, folds_sha="bbb")["status"] == B.INCOMPATIBLE


def test_legacy_spec_snapshots_and_base_spec_fallback():
    from gems_bench.pipeline.baselines import RECIPES, legacy_spec
    from gems_bench.training.spec import base_spec_dict

    for name in RECIPES:
        s = legacy_spec(name)
        assert s is not None, name
        assert not any(k in s for k in ("oof", "metrics", "dti", "summary")), name      # recipes, not results
    assert "groups" in legacy_spec("P4_B_gated") and legacy_spec("P4_B_gated")["arch"] == "dual"

    class NoLegacy:                                        # a fresh Drive: no Phase-4 run folders
        def run_dir(self, exp_id):
            raise FileNotFoundError(exp_id)

    assert base_spec_dict(NoLegacy(), "P4_B_gated")["name"] == "P4_B_gated"


# ---------------------------------------------------------------------------- acquisition
def _features_tif(path: Path):
    return tif(path, {f"b{i}": rnd(i) for i in range(19)})


def test_valid_drive_copy_is_never_downloaded_again(tmp_path, monkeypatch):
    from gems_bench.data import acquisition as A

    drive, local = tmp_path / "drive", tmp_path / "local"
    _features_tif(drive / "data/raw/geodawn/gems-geodawn-numerical-features.tif")

    def no_download(*a, **k):
        raise AssertionError("must not download a valid cached file")

    monkeypatch.setattr(A, "download", no_download)
    acq = A.Acquirer(local, drive, urls={"features": "https://example.org/x.tif"}, use_secrets=False)
    r = acq.acquire(A.BY_KEY["features"])
    assert r["status"] == "ok" and r["action"] == "staged_from_drive"
    assert acq.acquire(A.BY_KEY["features"])["action"] == "cached"


def test_alternate_location_accepted_and_missing_reports_secret(tmp_path, monkeypatch):
    from gems_bench.data import acquisition as A

    drive, local = tmp_path / "drive", tmp_path / "local"
    tif(drive / "data/raw/example_submission.tif", {"submission": rnd()})          # observed Drive layout
    monkeypatch.delenv("GEMS_URL_FAULTS", raising=False)
    acq = A.Acquirer(local, drive, use_secrets=False)
    assert acq.acquire(A.BY_KEY["submission"])["location"] == "data/raw/example_submission.tif"
    miss = acq.acquire(A.BY_KEY["faults"])
    assert miss["status"] == "missing" and "GEMS_URL_FAULTS" in miss["detail"]


def test_downloaded_url_is_never_persisted(tmp_path, monkeypatch):
    from gems_bench.data import acquisition as A

    secret = "https://dd.example.org/existing_faults.tif?X-Amz-Signature=TOPSECRET"

    def fake_download(url, dst, **k):
        tif(Path(dst), {"faults": (rnd() > 0.9).astype(np.float32)})
        return {"status": "downloaded", "bytes": Path(dst).stat().st_size, "attempts": 1, "error": ""}

    monkeypatch.setattr(A, "download", fake_download)
    drive, local = tmp_path / "drive", tmp_path / "local"
    drive.mkdir()
    acq = A.Acquirer(local, drive, urls={"faults": secret}, use_secrets=False)
    r = acq.acquire(A.BY_KEY["faults"])
    assert r["status"] == "ok" and (drive / "data/raw/faults/existing_faults.tif").exists()
    for root in (local, drive):
        txt = (root / A.MANIFEST).read_text()
        assert "TOPSECRET" not in txt and "dd.example.org" in txt


def test_url_precedence_runtime_then_env(monkeypatch):
    from gems_bench.data import acquisition as A

    f = A.BY_KEY["features"]
    monkeypatch.setenv("GEMS_URL_FEATURES", "https://env.example/f.tif")
    assert A.resolve_url(f, {}, use_secrets=False) == ("https://env.example/f.tif", "env:GEMS_URL_FEATURES")
    assert A.resolve_url(f, {"features": "https://rt.example/f.tif"}, use_secrets=False)[1] == "runtime"
    assert "Signature" not in A.mask("https://h/p/file.tif?Signature=abc")


def test_dem_url_list_canonicalised_without_touching_the_original(tmp_path):
    from gems_bench.data import acquisition as A

    drive, local = tmp_path / "drive", tmp_path / "local"
    src = drive / "data/raw/tnm_items.json"
    src.parent.mkdir(parents=True)
    src.write_text(json.dumps({"items": [{"downloadURL": "https://prd-tnm.s3.amazonaws.com/x/USGS_1M_11_x30y450_NV_A_2020.tif"}]}))
    out = A.canonicalize_dem_json(local, drive)
    assert out == local / "data/raw/dem_1m/tnm_urls.json" and out.exists()
    assert (drive / "data/raw/dem_1m/tnm_urls.json").exists() and src.exists()
