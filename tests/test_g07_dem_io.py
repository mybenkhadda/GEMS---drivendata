"""G07 DEM I/O: headers, resolution / vertical-unit contract, strict catalog status, halo windows, nodata (synthetic)."""
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from gems_bench.g06.raster_io import TileWindow  # noqa: E402
from gems_bench.g07 import dem_io as D  # noqa: E402
from gems_bench.g07.contracts import (ContractError, DEMSource, GridSpec, assert_not_100m_substitute,  # noqa: E402
                                      vertical_scale)


def write_dem(path, z, res=1.0, crs="EPSG:26911", nodata=-9999.0, units="metre", origin=(500000, 4000000)):
    prof = {"driver": "GTiff", "height": z.shape[0], "width": z.shape[1], "count": 1, "dtype": "float32", "crs": crs,
            "transform": from_origin(origin[0], origin[1], res, res), "nodata": nodata}
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(z.astype(np.float32), 1)
        if units:
            ds.update_tags(units=units)
    return path


def test_header_and_catalog_status(tmp_path):
    d = tmp_path / "data" / "raw" / "dem_1m" / "P1"
    d.mkdir(parents=True)
    write_dem(d / "USGS_1M_11_x50y400_NV_Test_2020.tif", np.ones((64, 64)))
    write_dem(d / "coarse_100m.tif", np.ones((8, 8)), res=100.0)
    T, S = D.open_dem_catalog({"catalog": {"url_lists": [], "local_dirs": ["data/raw/dem_1m"]},
                               "target_resolution_m": 1.0, "max_accepted_res_m": 3.0}, [tmp_path])
    st = dict(zip(T.tile_id, T.status))
    assert st["11_x50y400"] == "LOCAL"
    assert st["coarse_100m"] == "LOCAL_INVALID"                       # 100 m is never a high-resolution DEM
    h = D.read_header(d / "USGS_1M_11_x50y400_NV_Test_2020.tif")
    assert h["res_m"] == 1.0 and h["vertical_unit_raw"] == "metre" and h["nodata"] == -9999.0


def cat_cfg(**k):
    return {"catalog": {"url_lists": [], "local_dirs": ["data/raw/dem_1m"]}, "target_resolution_m": 1.0,
            "max_accepted_res_m": 3.0, **k}


def dem_dir(tmp_path):
    d = tmp_path / "data" / "raw" / "dem_1m"
    d.mkdir(parents=True)
    return d


def test_unknown_vertical_unit_is_invalid_until_explicitly_resolved(tmp_path):
    d = dem_dir(tmp_path)
    write_dem(d / "a.tif", np.ones((32, 32)), units=None)                  # horizontal metres say nothing about elevation
    T, S = D.open_dem_catalog(cat_cfg(), [tmp_path])
    assert S[0].status == "LOCAL_INVALID" and "UNKNOWN_VERTICAL_UNIT" in S[0].exclusion_codes
    assert "UNKNOWN_VERTICAL_UNIT" in T.exclusion_codes.iloc[0]
    v = D.source_validation_summary(T, cat_cfg())
    assert v["n_usable_local"] == 0 and v["exclusion_codes"]["UNKNOWN_VERTICAL_UNIT"] == 1
    assert any("vertical_unit_override" in x for x in v["remedy"])
    T2, S2 = D.open_dem_catalog(cat_cfg(vertical_unit_override="metre"), [tmp_path])
    assert S2[0].status == "LOCAL" and S2[0].vertical_unit_source == "override"
    assert D.source_vscale(S2[0]) == 1.0


def test_header_unit_wins_over_override(tmp_path):
    d = dem_dir(tmp_path)
    write_dem(d / "a.tif", np.ones((32, 32)), units="ft")
    _, S = D.open_dem_catalog(cat_cfg(vertical_unit_override="metre"), [tmp_path])
    assert S[0].vertical_unit == "ft" and S[0].vertical_unit_source == "tag"


def test_vertical_unit_comes_from_the_vertical_crs_only():
    wkt = ('COMPD_CS["NAD83 / UTM zone 11N + NAVD88 height",PROJCS["x",UNIT["foo_horizontal",1]],'
           'VERT_CS["NAVD88 height",VERT_DATUM["d",2005],UNIT["metre",1]]]')
    assert D.vertical_unit_from_wkt(wkt) == "metre"
    assert D.vertical_unit_from_wkt('PROJCS["x",UNIT["metre",1]]') == ""


def test_vertical_unit_from_sidecar_xml(tmp_path):
    (tmp_path / "t.xml").write_text("<a><altunits> meters </altunits></a>", encoding="utf-8")
    assert D.vertical_unit_from_sidecar(tmp_path / "t.tif") == "meters"
    assert vertical_scale("meters") == 1.0


def test_status_classes_and_reasons(tmp_path):
    d = dem_dir(tmp_path)
    write_dem(d / "USGS_1M_11_x50y400_NV_Test_2020.tif", np.ones((32, 32)), crs="EPSG:32611", origin=(500200, 4000800))
    write_dem(d / "far.tif", np.ones((32, 32)), crs="EPSG:32611", origin=(700000, 4000800))
    write_dem(d / "geographic.tif", np.ones((32, 32)), res=0.001, crs="EPSG:4326", origin=(-117, 36))
    (d / "broken.tif").write_bytes(b"not a tiff")
    urls = tmp_path / "list.json"
    urls.write_text('["https://x/Projects/P/TIFF/USGS_1M_11_x51y400_NV_Other_2021.tif"]', encoding="utf-8")
    cfg = cat_cfg()
    cfg["catalog"]["url_lists"] = ["list.json"]
    grid = GridSpec(10, 10, "EPSG:32611", (100.0, 0.0, 500000.0, 0.0, -100.0, 4001000.0), 100.0)
    T, S = D.open_dem_catalog(cfg, [tmp_path], grid)
    st = {Path(s.path).name if s.path else s.tile_id: s for s in S}
    assert st["USGS_1M_11_x50y400_NV_Test_2020.tif"].status == "LOCAL"
    assert st["far.tif"].status == "LOCAL_OUTSIDE_STUDY_AREA" and st["far.tif"].exclusion_codes == ("NO_GRID_INTERSECTION",)
    assert st["geographic.tif"].status == "LOCAL_INVALID" and "CRS_NOT_PROJECTED_METRIC" in st["geographic.tif"].exclusion_codes
    assert st["broken.tif"].status == "LOCAL_INVALID" and "UNREADABLE" in st["broken.tif"].exclusion_codes
    assert st["11_x51y400"].status == "CATALOG_ONLY" and st["11_x51y400"].path is None
    assert (T[T.status.isin(["LOCAL_INVALID", "LOCAL_OUTSIDE_STUDY_AREA"])].problems.str.len() > 0).all()
    foot = np.zeros((10, 10), bool)                                         # the tile's cells hold no data
    _, S2 = D.open_dem_catalog(cat_cfg(), [tmp_path], grid, foot)
    near = [s for s in S2 if s.tile_id == "11_x50y400"][0]
    assert near.status == "LOCAL_OUTSIDE_STUDY_AREA" and near.exclusion_codes == ("NO_FOOTPRINT_INTERSECTION",)


def test_catalog_only_never_counts_as_local(tmp_path):
    urls = tmp_path / "list.json"
    urls.write_text('["https://x/Projects/P/TIFF/USGS_1M_11_x51y400_NV_Other_2021.tif"]', encoding="utf-8")
    cfg = cat_cfg()
    cfg["catalog"]["url_lists"] = ["list.json"]
    T, S = D.open_dem_catalog(cfg, [tmp_path])
    assert [s.status for s in S] == ["CATALOG_ONLY"]
    assert D.source_validation_summary(T, cfg)["n_usable_local"] == 0


def test_halo_crosses_tile_edges_only_when_aligned(tmp_path):
    r, c = np.indices((40, 40), dtype=np.float32)
    whole = 0.5 * r + 0.25 * c
    a = write_dem(tmp_path / "a.tif", whole[:, :20], origin=(500000, 4000000))
    b = write_dem(tmp_path / "b.tif", whole[:, 20:], origin=(500020, 4000000))
    shifted = write_dem(tmp_path / "s.tif", whole[:, 20:], origin=(500020.5, 4000000))
    win = TileWindow("t", 10, 20, 10, 20, 5, 40, 20)
    with rasterio.open(a) as da, rasterio.open(b) as db, rasterio.open(shifted) as ds:
        alone = D.read_dem_window(da, win, 1, 1.0)
        joined = D.read_dem_window(da, win, 1, 1.0, neighbours=[(db, 1.0)])
        skipped = D.read_dem_window(da, win, 1, 1.0, neighbours=[(ds, 1.0)])
    assert np.isnan(alone[:, 15:]).all()                                    # beyond the tile: no data, never invented
    np.testing.assert_allclose(joined, whole[5:25, 5:25])                   # seamless: identical to one raster
    assert np.isnan(skipped[:, 15:]).all()                                  # a resampled neighbour would create a seam


def test_neighbour_selection():
    def src(tid, x, crs="EPSG:26911", res=1.0, status="LOCAL"):
        return DEMSource(tid, None, tid, "P", 2020, crs, res, (x, 0, x + 100, 100), "metre", status=status)

    s = src("a", 0)
    near = D.neighbour_sources(s, [s, src("b", 100), src("far", 900), src("utm10", 100, crs="EPSG:26910"),
                                   src("coarse", 100, res=2.0), src("bad", 100, status="LOCAL_INVALID")], 50)
    assert [x.tile_id for x in near] == ["b"]


def test_vertical_units_and_substitution_guard():
    assert vertical_scale("metre") == 1.0
    assert vertical_scale("ft") == pytest.approx(0.3048)
    with pytest.raises(ContractError):
        vertical_scale("unknown")
    s = DEMSource("t", None, "p", "P", 2020, "EPSG:26911", 100.0, (0, 0, 1, 1))
    with pytest.raises(ContractError):
        assert_not_100m_substitute(s)


def test_read_window_halo_nodata_and_block_mean(tmp_path):
    z = np.arange(40 * 40, dtype=np.float32).reshape(40, 40)
    z[5, 5] = -9999.0
    p = write_dem(tmp_path / "a.tif", z)
    with rasterio.open(p) as ds:
        wins = list(D.iter_dem_tiles(ds, 10, 3, factor=2))           # work grid 20 x 20
        cov = np.zeros((20, 20), int)
        for w in wins:
            cov[w.grid_slice] += 1
        assert (cov == 1).all()
        w = wins[0]
        a = D.read_dem_window(ds, w, 2, 1.0)
    assert a.shape == (10 + 6, 10 + 6)
    assert np.isnan(a[:3]).all() and np.isnan(a[:, :3]).all()        # beyond the raster: NaN, never 0
    blk = z[0:2, 0:2].mean()
    assert a[3, 3] == pytest.approx(blk)                             # NaN-aware 2 x 2 block mean
    assert np.isfinite(a[3 + 2, 3 + 2])                              # 3 of 4 valid -> kept (>= 50 %)


def test_block_mean_requires_half_valid():
    a = np.array([[1.0, np.nan], [np.nan, np.nan]], np.float32)
    assert np.isnan(D.block_mean(a, 2)[0, 0])
    b = np.array([[1.0, 3.0], [np.nan, np.nan]], np.float32)
    assert D.block_mean(b, 2)[0, 0] == pytest.approx(2.0)


def test_missing_coverage_reported(tmp_path):
    T, S = D.open_dem_catalog({"catalog": {"url_lists": [], "local_dirs": ["data/raw/dem_1m"]}}, [tmp_path])
    assert len(T) == 0 and S == []
