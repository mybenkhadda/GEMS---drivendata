"""G07 DEM I/O: headers, resolution / vertical-unit contract, catalog status, halo windows, nodata (synthetic)."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from gems_bench.g07 import dem_io as D  # noqa: E402
from gems_bench.g07.contracts import ContractError, DEMSource, assert_not_100m_substitute, vertical_scale  # noqa: E402


def write_dem(path, z, res=1.0, crs="EPSG:26911", nodata=-9999.0, units="metre"):
    prof = {"driver": "GTiff", "height": z.shape[0], "width": z.shape[1], "count": 1, "dtype": "float32", "crs": crs,
            "transform": from_origin(500000, 4000000, res, res), "nodata": nodata}
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
