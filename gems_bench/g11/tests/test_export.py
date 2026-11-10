import numpy as np
import pytest

from gems_bench.g11.evaluation import official_adapter as OA


@pytest.fixture()
def schema(synth):
    return OA.load_official_schema(synth["raw"] / "example_submission.tif")


def test_schema_from_template(schema, grid):
    assert schema.shape == grid.shape and schema.count == 1
    assert np.array_equal(schema.footprint, grid.valid)


def test_export_and_validate(tmp_path, schema, grid):
    p = np.where(grid.valid, np.random.default_rng(0).random(grid.shape), np.nan).astype(np.float32)
    out = OA.export_submission(p, schema, tmp_path / "sub.tif", {"model": "synthetic", "budget_km": 10})
    assert out["validation"]["passed"]
    assert (tmp_path / "sub.sha256").is_file() and (tmp_path / "sub.json").is_file()


def test_validation_catches_bad_file(tmp_path, schema, grid):
    import rasterio

    bad = tmp_path / "bad.tif"
    with rasterio.open(bad, "w", driver="GTiff", height=grid.shape[0], width=grid.shape[1], count=1, dtype="float32",
                       crs=grid.crs, transform=grid.affine) as ds:
        a = np.full(grid.shape, 2.0, np.float32)
        a[5, 50] = np.nan
        ds.write(a, 1)
    chk = OA.validate_submission(bad, schema)
    assert not chk["passed"] and not chk["range_0_1_in_footprint"] and not chk["finite_in_footprint"]


def test_wrong_shape_refused(tmp_path, schema):
    with pytest.raises(OA.OfficialSchemaError):
        OA.export_submission(np.zeros((3, 3), np.float32), schema, tmp_path / "x.tif", {})


def test_missing_template_and_evaluator(tmp_path, cfg):
    with pytest.raises(OA.OfficialSchemaError):
        OA.load_official_schema(tmp_path / "none.tif")
    with pytest.raises(OA.OfficialEvaluatorUnavailable):
        OA.run_official_evaluator(tmp_path / "sub.tif", cfg)
    cfg["official_submission"]["evaluator_command"] = "definitely_not_installed_evaluator"
    with pytest.raises(OA.OfficialEvaluatorUnavailable):
        OA.run_official_evaluator(tmp_path / "sub.tif", cfg)


def test_proxy_dti_is_labelled(grid):
    t = np.zeros(grid.shape, bool)
    t[50, 50] = True
    p = np.where(t, 1.0, 0.0)
    d = OA.proxy_dti(p, t, grid.valid)
    assert d["dti"] > 0.99 and "proxy" in d["label"]


def test_rasterized_candidates_export(tmp_path, schema, grid):
    import geopandas as gpd
    from shapely.geometry import LineString

    from gems_bench.g11.inference.export import rasterize_segments
    from gems_bench.g11.tests.conftest import FAULT

    g = gpd.GeoDataFrame(geometry=[LineString(FAULT)], crs=grid.crs)
    r = rasterize_segments(g, np.array([0.8]), grid, grid.valid)
    assert np.nanmax(r) == pytest.approx(0.8) and np.isnan(r[~grid.valid]).all()
    assert OA.export_submission(r, schema, tmp_path / "c.tif", {})["validation"]["passed"]
