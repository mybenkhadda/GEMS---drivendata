import numpy as np
import pytest

from gems_bench.g11.dem_catalog import build_dem_catalog, coverage_100m
from gems_bench.g11.dem_reader import DEMReader, DEMWindowError, aggregate_dem_to_master, detect_seams


@pytest.fixture(scope="module")
def catalog(synth, grid):
    return build_dem_catalog(synth["dem_dir"], grid.crs, synth["cfg"])


def test_catalog_priority_and_metadata(catalog):
    assert len(catalog) == 2
    assert list(catalog.year) == [2020, 2016]                     # newer first at equal valid fraction
    assert catalog.resolution_m.tolist() == [2.0, 2.0]
    assert all("UNKNOWN_VERTICAL_UNIT" in f for f in catalog.quality_flags)   # synthetic tiles carry no unit
    assert (catalog.priority_rank == [0, 1]).all()


def test_coverage_100m(catalog, grid):
    cov = coverage_100m(catalog, grid)
    assert cov["covered"].sum() == pytest.approx(30 * 20, abs=40)  # 3 km x 2 km union = 600 cells
    assert cov["count"].max() == 2


def test_reader_merges_by_priority_and_records_provenance(catalog, cfg):
    rd = DEMReader(catalog, cfg, allow_unknown_vertical=True)
    win = rd.read_dem_window((504500, 3993500, 505500, 3994500), "EPSG:32611", resolution=2.0)
    assert win.valid.all()
    assert set(win.source_tiles) == {"USGS_1M_11_x51y399_NV_B_2020", "USGS_1M_11_x50y399_NV_A_2016"}
    k_new = win.source_tiles.index("USGS_1M_11_x51y399_NV_B_2020")
    assert (win.source_index[:, -10:] == k_new).all()             # overlap filled by the preferred (newer) tile
    assert win.seam_mask.any()                                    # 0.8 m offset between tiles detected
    assert rd.read_dem_window((504500, 3993500, 505500, 3994500), "EPSG:32611", resolution=2.0) is win  # cached


def test_unknown_vertical_units_excluded_by_default(catalog, cfg):
    rd = DEMReader(catalog, cfg)
    win = rd.read_dem_window((504500, 3993500, 505500, 3994500), "EPSG:32611", resolution=2.0)
    assert not win.valid.any() and not win.source_tiles


def test_window_memory_guard(catalog, cfg):
    cfg["dem"]["max_window_mpix"] = 0.01
    with pytest.raises(DEMWindowError):
        DEMReader(catalog, cfg, True).read_dem_window((504000, 3993000, 506000, 3995000), "EPSG:32611", 2.0)


def test_seam_detection_synthetic():
    z = np.zeros((10, 10), np.float32)
    src = np.zeros((10, 10), np.int16)
    src[:, 5:] = 1
    z[:, 5:] += 2.0
    assert detect_seams(z, src, 1.0, 0.5)[:, 4:6].all()
    z2 = np.tile(np.arange(10, dtype=np.float32), (10, 1))           # smooth slope: no seam at the boundary
    assert not detect_seams(z2, src, 1.0, 0.5).any()


def test_local_mosaic_matches_direct_reads(catalog, grid, cfg, tmp_path):
    from gems_bench.g11.dem_reader import MosaicDEMReader, build_dem_mosaic

    rd = DEMReader(catalog, cfg, allow_unknown_vertical=True)
    cov = coverage_100m(catalog, grid)["covered"]
    p = build_dem_mosaic(rd, grid, cov, 25.0, tmp_path / "m.tif", chunk_cells=8, log=None)
    mr = MosaicDEMReader(p)
    b = (504400.0, 3993600.0, 505600.0, 3994800.0)                       # nested in the 25 m mosaic grid
    a1 = mr.read_dem_window(b, grid.crs).elevation
    a2 = rd.read_dem_window(b, grid.crs, resolution=25.0, resampling="average", snap=False).elevation
    m = np.isfinite(a1) & np.isfinite(a2)
    assert a1.shape == a2.shape and m.mean() > 0.9 and np.allclose(a1[m], a2[m], atol=1e-3)
    far = mr.read_dem_window((500000.0, 3999000.0, 500500.0, 3999500.0), grid.crs)
    assert not far.valid.any()                                            # outside the DEM: nodata, never zeros


def test_aggregate_chunked_equals_single_chunk(catalog, grid, cfg):
    """Chunk boundaries must not change the result (halo-aware chunking)."""
    rd = DEMReader(catalog, cfg, allow_unknown_vertical=True)
    cov = coverage_100m(catalog, grid)["covered"]
    cfg["dem"]["context_scales_m"] = [25]
    a = aggregate_dem_to_master(rd, grid, cov, cfg, progress=None, chunk_cells=8)
    b = aggregate_dem_to_master(rd, grid, cov, cfg, progress=None, chunk_cells=200)
    for k in ("elev_mean", "slope_mean", "local_relief", "roughness_mean", "relief_ctx25m"):
        m = np.isfinite(a[k]) & np.isfinite(b[k])
        assert m.sum() > 300
        assert np.allclose(a[k][m], b[k][m], atol=1e-3), k


def test_aggregate_nests_exactly(catalog, grid, cfg):
    rd = DEMReader(catalog, cfg, allow_unknown_vertical=True)
    cov = coverage_100m(catalog, grid)["covered"]
    out = aggregate_dem_to_master(rd, grid, cov, cfg, progress=None)
    m = cov & np.isfinite(out["elev_mean"])
    assert m.sum() > 400 and np.isnan(out["elev_mean"][~cov]).all()
    # plane z = 1000 + 0.01 x (+3 m on one side): cell slope ~ atan(0.01) = 0.573 deg away from the scarp
    from gems_bench.g11.tests.conftest import fault_distance_grid

    far = m & (np.abs(fault_distance_grid()) > 300)
    assert np.nanmedian(out["slope_mean"][far]) == pytest.approx(np.degrees(np.arctan(0.01)), abs=0.05)
    near = m & (np.abs(fault_distance_grid()) < 60)
    assert np.nanmedian(out["local_relief"][near]) > np.nanmedian(out["local_relief"][far]) + 1.5
    assert np.nanmax(out["valid_fraction"]) == pytest.approx(1.0)
    assert "relief_ctx100m" in out
