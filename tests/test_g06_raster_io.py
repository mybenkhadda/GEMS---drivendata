"""G06 raster I/O: exact bands, windows, nodata, tile cache, writing and point reads (synthetic GeoTIFFs)."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from gems_bench.g05_1.bands import BandResolutionError  # noqa: E402
from gems_bench.g06 import raster_io as RIO  # noqa: E402
from gems_bench.g06.contracts import GridSpec  # noqa: E402

DESC = ["mag_anom - magnetic anomaly", "rtp - reduced to pole", "iso_grav_anom_hg - gravity horizontal gradient",
        "iso_grav_anom_vg - gravity vertical gradient", "iso_grav_anom - isostatic gravity anomaly",
        "iso_grav_anom_slope - slope"]


def _write(path, arrays, desc):
    prof = {"driver": "GTiff", "height": arrays.shape[1], "width": arrays.shape[2], "count": arrays.shape[0],
            "dtype": "float32", "crs": "EPSG:32611", "transform": from_origin(500000, 4000000, 100, 100)}
    with rasterio.open(path, "w", **prof) as ds:
        for i in range(arrays.shape[0]):
            ds.write(arrays[i].astype(np.float32), i + 1)
            ds.set_band_description(i + 1, desc[i])
    return path


@pytest.fixture
def src(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(len(DESC), 40, 50)).astype(np.float32)
    a[:, 0, 0] = 1e31                                   # nodata sentinel
    a[1, 5, 5] = np.nan
    for i in range(len(DESC)):
        a[i] += 100 * i                                 # band identity is visible in the values
    return _write(tmp_path / "src.tif", a, DESC), a


def test_exact_band_resolution(src):
    p, _ = src
    with rasterio.open(p) as ds:
        assert RIO.resolve_exact_band(ds, "iso_grav_anom") == 5          # NOT 3 (_hg), 4 (_vg) or 6 (_slope)
        assert RIO.resolve_exact_band(ds, "rtp") == 2
        assert RIO.resolve_exact_band(ds, "iso_grav_anom - any description") == 5
        with pytest.raises(BandResolutionError):
            RIO.resolve_exact_band(ds, "iso_grav")                       # a substring is never accepted
        with pytest.raises(BandResolutionError):
            RIO.resolve_exact_band(ds, "tmi")


def test_windows_cover_raster_once():
    cov = np.zeros((40, 50), int)
    for w in RIO.iter_windows((40, 50), 16, 5):
        cov[w.grid_slice] += 1
        assert w.core_shape == (w.row1 - w.row0, w.col1 - w.col0)
    assert (cov == 1).all()


def test_read_aligned_window_pads_and_masks(src):
    p, a = src
    with rasterio.open(p) as ds:
        w = next(iter(RIO.iter_windows((40, 50), 16, 4)))                  # top-left tile: halo outside the raster
        x = RIO.read_aligned_window(ds, [1, 2], w)
    assert x.shape == (2, 16 + 8, 16 + 8)
    assert np.isnan(x[:, :4, :]).all() and np.isnan(x[:, :, :4]).all()     # NaN beyond the edge, never 0
    core = RIO.crop_core(x, w)
    assert np.isnan(core[0, 0, 0])                                         # 1e31 nodata -> NaN
    assert np.isnan(core[1, 5, 5])
    np.testing.assert_allclose(core[0, 1:, 1:], a[0, 1:16, 1:16])


def test_tile_cache_complete_only_after_verified_write(tmp_path):
    w = RIO.TileWindow("r00000_c00000", 0, 8, 0, 8, 2, 8, 8)
    tc = RIO.TileCache(tmp_path / "tiles", ["a", "b"])
    assert not tc.complete(w)
    tc.save(w, np.ones((2, 8, 8), np.float32))
    assert tc.complete(w)
    npz, _ = tc.paths(w.tile_id)
    npz.write_bytes(npz.read_bytes()[:-10])                                 # corruption after the manifest
    assert not tc.complete(w)
    assert not RIO.TileCache(tmp_path / "tiles", ["a", "c"]).complete(w)    # other channel list
    with pytest.raises(Exception):
        tc.save(w, np.ones((3, 8, 8), np.float32))


def test_write_and_verify_alignment(tmp_path):
    g = GridSpec(20, 30, "EPSG:32611", (100.0, 0.0, 5e5, 0.0, -100.0, 4e6), 100.0)
    arr = {"f1": np.arange(600, dtype=np.float32).reshape(20, 30), "f2": np.full((20, 30), np.nan, np.float32)}
    p = RIO.write_feature_raster(tmp_path / "g.tif", arr, g, block_px=16)
    assert RIO.verify_raster_alignment(p, g, ["f1", "f2"], 32611) == []
    bad = GridSpec(20, 30, "EPSG:32611", (100.0, 0.0, 5e5 + 100, 0.0, -100.0, 4e6), 100.0)
    assert RIO.verify_raster_alignment(p, bad, ["f1", "f2"])
    rows, cols = np.array([0, 19, 7]), np.array([0, 29, 13])
    v = RIO.read_points(p, [1], rows, cols, block_px=8)[:, 0]
    np.testing.assert_array_equal(v, arr["f1"][rows, cols])


def test_group_writer_tiles_equal_full(tmp_path):
    g = GridSpec(37, 41, "EPSG:32611", (100.0, 0.0, 5e5, 0.0, -100.0, 4e6), 100.0)
    full = np.random.default_rng(1).normal(size=(2, 37, 41)).astype(np.float32)
    gw = RIO.GroupWriter(tmp_path / "w.tif", ["a", "b"], g, block_px=16)
    for w in RIO.iter_windows(g.shape, 16, 3):
        gw.write(w, full[:, w.grid_slice[0], w.grid_slice[1]])
    gw.close()
    back = RIO.read_window_bands(tmp_path / "w.tif", [1, 2], (0, 37, 0, 41))
    np.testing.assert_array_equal(back, full)
    s = RIO.compare_tile_seams(tmp_path / "w.tif", 1, 16)
    assert s["n_seams"] > 0
