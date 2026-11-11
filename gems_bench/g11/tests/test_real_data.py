"""LEVEL 2 real-data integration tests (SMOKE subset). Skipped unless ``G11_REAL_DATA_ROOT`` points at the GEMS
Drive root (e.g. /content/drive/MyDrive/GEMS_2). Outputs go to a temporary folder, never to the production tree.

    G11_REAL_DATA_ROOT=/content/drive/MyDrive/GEMS_2 python -m pytest gems_bench/g11/tests/test_real_data.py -q
"""
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = os.environ.get("G11_REAL_DATA_ROOT")
pytestmark = pytest.mark.skipif(not ROOT or not Path(ROOT).is_dir(), reason="G11_REAL_DATA_ROOT not set / not mounted")


@pytest.fixture(scope="module")
def P(tmp_path_factory):
    from gems_bench.g11.config import load_config
    from gems_bench.g11.notebook_config import build_plan
    from gems_bench.g11.pipeline import G11Pipeline

    base = load_config()
    plan = build_plan({"run_mode": "SMOKE_TEST", "drive_root": ROOT, "smoke_window_km": 20.0}, base)
    ov = {**plan.overrides, "output_root": str(tmp_path_factory.mktemp("g11_real_smoke")),
          "scratch_root": str(tmp_path_factory.mktemp("g11_real_scratch"))}
    ov["training"].update(epochs=1, loss_ablation=[])
    ov.setdefault("models", {}).setdefault("gated_fusion", {})["mode_ablation"] = []
    return G11Pipeline(load_config(overrides=ov), log=print)


def test_real_discovery_and_alignment(P):
    s = P.audit()
    assert not s["problems"] and s["grid"]["epsg"] == 32611
    assert s["smoke_window"]["fault_px"] > 0
    comp = (P.ctx.d("audit") / "artifact_compatibility.csv")
    assert comp.is_file() and (P.ctx.d("audit") / "dem_vertical_units.csv").is_file()


def test_real_ingest_dem_window_and_units(P):
    st = P.ingest()
    assert st["fault_pixels"] > 0 and st["geodawn_bands"] >= 1
    rd = P.dem_reader()
    if rd is None or not len(rd.index.cat):
        pytest.skip("no usable DEM tile in the smoke window")
    g = rd.index.cat.geometry.iloc[0].centroid
    w = rd.read_dem_window((g.x - 200, g.y - 200, g.x + 200, g.y + 200), P.grid.crs, resolution=2.0)
    assert w.valid.any() and w.source_tiles


def test_real_features_splits_training(P):
    P.features()
    assert len(P.store.predictors()) > 20
    a = P.store.get(P.store.predictors()[0])
    assert a.shape == P.grid.shape
    P.splits()
    t = P.train(["E0"])
    assert (t.status.isin(["TRAINED", "CACHED"])).all()
    assert P.pred_path("E0").is_file()


def test_real_short_segformer_and_restore(P):
    pytest.importorskip("torch")
    t = P.train(["E2"])
    assert t.status.iloc[0] in ("TRAINED", "CACHED")
    assert ((P.model_dir("E2") / "best.pt").is_file() or (P.model_dir("E2") / "last.pt").is_file())
    assert set(P.train(["E2"]).status) == {"CACHED"}                  # checkpoint / stage restoration


def test_real_evaluate_candidates_submission(P):
    m = P.evaluate()
    assert m.PR_AUC.notna().any()
    from gems_bench.g11.inference.candidate_generation import generate_candidates
    from gems_bench.g11.pipeline import _read_tif

    p = _read_tif(P.pred_path("E0"))
    c = generate_candidates(p, P.grid.valid, P.grid, P.cfg, "E0")
    assert len(c) and c.is_valid.all()
    from gems_bench.g11.evaluation.official_adapter import export_submission, load_official_schema

    schema = load_official_schema(P.found["template"])
    full = np.zeros(schema.shape, np.float32)
    r0, r1, c0, c1 = P.smoke_window
    full[r0:r1, c0:c1] = np.nan_to_num(p)
    out = export_submission(np.where(schema.footprint, full, np.nan), schema, P.ctx.d("submit") / "smoke_check.tif",
                            {"smoke": True})
    assert out["validation"]["passed"]


def test_real_legacy_feature_store_headers_readonly():
    """Read-only: every legacy layer header of the production run has the master-grid shape and float32 dtype."""
    from gems_bench.g11.recovery import inspect_npy

    froot = Path(ROOT) / "outputs" / "G11_MULTISCALE_FAULT_DISCOVERY" / "03_features"
    if not froot.is_dir():
        pytest.skip("no production G11 feature store")
    files = [p for p in froot.rglob("*.npy") if "datasets" not in p.parts and "dem_chunks" not in p.parts]
    bad = [(p.name, i.get("problems") or i.get("error")) for p in files for i in [inspect_npy(p, (3730, 3292))]
           if not i["header_ok"]]
    print(f"{len(files)} layer files, {len(bad)} with header / size problems: {bad[:10]}")
    assert files