"""G07 native -> 100 m aggregation: exact area-weighted means, coverage, max / orientation bands, mosaic."""
import numpy as np
import pytest

pytest.importorskip("rasterio")
from rasterio.transform import Affine  # noqa: E402

from gems_bench.g07 import dem_io as D  # noqa: E402
from gems_bench.g07.contracts import GridSpec  # noqa: E402

GRID = GridSpec(10, 10, "EPSG:32611", (100.0, 0.0, 500000.0, 0.0, -100.0, 4001000.0), 100.0)


def native_block(value=3.0, rows=300, cols=250, r_cell=2, c_cell=2, res=1.0):
    T = Affine(res, 0, 500000.0 + c_cell * 100, 0, -res, 4001000.0 - r_cell * 100)
    return np.full((rows, cols), value, np.float32), T


def test_mean_and_coverage_are_exact():
    a, T = native_block()
    v = np.ones(a.shape, bool)
    win, out = D.align_dem_to_analysis_grid({"f": a, "m": a}, v, T, "EPSG:32611", GRID, {"f": D.MEAN, "m": D.MAX}, 1.0)
    r0, r1, c0, c1 = win
    cov = out["dem_coverage"]
    full = cov >= 0.999
    assert full.sum() == 3 * 2                                        # 300 x 250 m = 3 full rows x 2 full cols
    np.testing.assert_allclose(out["f"][full], 3.0, rtol=1e-6)
    part = (cov > 0) & ~full
    assert part.any() and np.allclose(cov[part], 0.5, atol=1e-6)      # the half-covered column
    np.testing.assert_allclose(out["f"][part], 3.0, rtol=1e-6)        # partial cells: mean of covered pixels only
    np.testing.assert_allclose(out["m"][cov > 0], 3.0)


def test_orientation_aggregation_is_axial():
    a, T = native_block(rows=200, cols=200)
    ang = np.full(a.shape, np.pi, np.float32)                       # doubled strike angle 180 deg (N-S strike)
    v = np.ones(a.shape, bool)
    win, out = D.align_dem_to_analysis_grid({"o": ang, "o__w": np.ones_like(a)}, v, T, "EPSG:32611", GRID,
                                            {"o": D.ORIENT}, 1.0)
    m = out["dem_coverage"] > 0.99
    np.testing.assert_allclose(out["o_cos2"][m], -1.0, atol=1e-6)
    np.testing.assert_allclose(out["o_resultant"][m], 1.0, atol=1e-6)


def test_nodata_does_not_count():
    a, T = native_block(rows=100, cols=100)
    v = np.ones(a.shape, bool)
    v[:, :50] = False
    a[:, :50] = 1000.0                                              # garbage under nodata
    win, out = D.align_dem_to_analysis_grid({"f": a}, v, T, "EPSG:32611", GRID, {"f": D.MEAN}, 1.0)
    m = out["dem_coverage"] > 0
    np.testing.assert_allclose(out["f"][m], 3.0)
    assert out["dem_coverage"][m].max() == pytest.approx(0.5)


@pytest.mark.parametrize("bounds,pad,expected", [
    ((500250, 4000550, 500450, 4000750), 0, (2, 5, 2, 5)),                 # interior
    ((500250, 4000550, 500450, 4000750), 1, (1, 6, 1, 6)),                 # padding on each side independently
    ((499950, 4000750, 500150, 4001050), 0, (0, 3, 0, 2)),                 # beyond the top-left corner
    ((500850, 3999950, 501050, 4000250), 0, (7, 10, 8, 10)),               # beyond the bottom-right corner
    ((500850, 4000750, 501050, 4001050), 0, (0, 3, 8, 10)),                # beyond the top-right corner
    ((499950, 3999950, 500150, 4000250), 0, (7, 10, 0, 2)),                # beyond the bottom-left corner
    ((500150, 4000050, 500350, 4000950), 0, (0, 10, 1, 4)),                # non-square window
    ((500850, 4000850, 501050, 4000950), 2, (0, 4, 6, 10)),                # padding clipped at two edges
])
def test_analysis_window_quadrants(bounds, pad, expected):
    assert D.analysis_window(bounds, GRID.crs, GRID, pad=pad) == expected


@pytest.mark.parametrize("bounds", [(490000, 4000500, 490100, 4000600), (501001, 4000500, 501100, 4000600),
                                    (500500, 4001001, 500600, 4001100), (500500, 3999000, 500600, 3999999)])
@pytest.mark.parametrize("pad", [0, 1])
def test_analysis_window_empty_intersection_stays_empty_with_padding(bounds, pad):
    assert D.analysis_window(bounds, GRID.crs, GRID, pad=pad) is None


def test_analysis_window_crs_transformation():
    win = D.analysis_window((500250, 4000550, 500450, 4000750), "EPSG:26911", GRID, pad=0)
    r0, r1, c0, c1 = win
    assert r0 <= 2 and r1 >= 5 and c0 <= 2 and c1 >= 5 and r1 > r0 and c1 > c0
    assert r1 - r0 <= 5 and c1 - c0 <= 5                                    # NAD83 vs WGS84 UTM differ by metres only


def test_coverage_uses_the_work_resolution_not_the_native_one(tmp_path):
    """A 1 m tile averaged to 2 m has 2500 work pixels per 100 m cell: full cells must read coverage 1, not 0.25."""
    import rasterio
    from rasterio.transform import from_origin

    from gems_bench.g07.contracts import DEMSource
    from gems_bench.g07.dem_features import process_fine_source

    r, c = np.indices((200, 200), dtype=np.float32)
    p = tmp_path / "t.tif"
    with rasterio.open(p, "w", driver="GTiff", height=200, width=200, count=1, dtype="float32", crs="EPSG:32611",
                       transform=from_origin(500200, 4000800, 1, 1), nodata=-9999.0) as ds:
        ds.write(0.05 * c, 1)
    src = DEMSource("t", None, str(p), "P", 2020, "EPSG:32611", 1.0, (500200, 4000600, 500400, 4000800), "metre")
    cfg = {"scales_m": [4, 10], "coarse_scales_m": [], "hillshade_azimuths": [315], "relief_windows_m": [10],
           "openness_radius_m": 8, "integration_factor": 2.0, "tile_px": 64, "halo_px": 8}
    out = process_fine_source(p, src, cfg, GRID, tmp_path / "w.tif", 2, {})
    with rasterio.open(out) as ds:
        cov = ds.read(list(ds.descriptions).index("dem_coverage") + 1)
        T = ds.transform
    r0, c0 = int(round((GRID.transform[5] - T.f) / 100)), int(round((T.c - GRID.transform[2]) / 100))
    inner = cov[2 - r0:4 - r0, 2 - c0:4 - c0]
    np.testing.assert_allclose(inner, 1.0, atol=0.02)


def test_mosaic_does_not_dilute_a_band_with_a_nan_tile(tmp_path):
    a1, T1 = native_block(2.0, 100, 100, 1, 1)
    v = np.ones(a1.shape, bool)
    win, out1 = D.align_dem_to_analysis_grid({"f": a1}, v, T1, "EPSG:32611", GRID, {"f": D.MEAN}, 1.0)
    p1 = D.write_window(tmp_path / "a.tif", out1, ["dem_coverage", "f"], GRID, win, {})
    out2 = {k: np.array(x) for k, x in out1.items()}
    out2["f"][:] = np.nan                                                   # this tile's guard removed the feature
    p2 = D.write_window(tmp_path / "b.tif", out2, ["dem_coverage", "f"], GRID, win, {})
    st = D.DemFeatureStore(D.mosaic_windows([p1, p2], GRID, ["dem_coverage", "f"], tmp_path / "m.tif"), 0.5)
    assert st.load(["f"])[0][1, 1] == pytest.approx(2.0)


def test_mosaic_coverage_comes_from_the_fine_tier_only(tmp_path):
    a, T = native_block(2.0, 100, 50, 1, 1)                                 # half of one cell
    v = np.ones(a.shape, bool)
    win, out = D.align_dem_to_analysis_grid({"f": a}, v, T, "EPSG:32611", GRID, {"f": D.MEAN}, 1.0)
    fine = D.write_window(tmp_path / "fine.tif", out, ["dem_coverage", "f"], GRID, win, {})
    coarse = D.write_window(tmp_path / "coarse.tif", {**out, "g": out["f"]}, ["dem_coverage", "g"], GRID, win, {})
    extra = {"dem_coverage_coarse": np.full(GRID.shape, 0.25, np.float32)}
    p = D.mosaic_windows([fine, coarse], GRID, ["dem_coverage", "f", "g"], tmp_path / "m.tif", (), [fine], extra)
    st = D.DemFeatureStore(p, 0.5)
    assert st.coverage()[1, 1] == pytest.approx(0.5, abs=1e-3)              # not 1.0: the two tiers do not add up
    assert "dem_coverage_coarse" not in st.usable() and "dem_coverage" not in st.usable()
    assert st.coverage_for_patches()[1, 1] == pytest.approx(0.25)


def test_coverage_100m_from_dem(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    z = np.ones((20, 20), np.float32)
    z[:, 15:] = np.nan
    p = tmp_path / "c.tif"
    with rasterio.open(p, "w", driver="GTiff", height=20, width=20, count=1, dtype="float32", crs="EPSG:32611",
                       transform=from_origin(500200, 4000800, 10, 10), nodata=np.nan) as ds:
        ds.write(z, 1)
    cov = D.coverage_100m_from_dem(p, GRID, block_px=8)
    assert cov[2, 2] == pytest.approx(1.0, abs=0.02)
    assert cov[2, 3] == pytest.approx(0.5, abs=0.02)                        # the NaN half of the second column
    assert cov[0, 0] == 0


def test_seam_audit_flags_a_feature_that_jumps_at_a_tile_edge(tmp_path):
    from gems_bench.g07.contracts import DEMSource

    cov = np.ones(GRID.shape, np.float32)
    flat = np.ones(GRID.shape, np.float32)
    seam = np.ones(GRID.shape, np.float32)
    seam[:, 3:5] = 10.0                                                     # a line along the tile's east boundary
    from gems_bench.g06.raster_io import write_feature_raster

    p = write_feature_raster(tmp_path / "s.tif", {"dem_coverage": cov, "dem_flat": flat, "dem_seam": seam}, GRID)
    st = D.DemFeatureStore(p, 0.5)
    src = DEMSource("t", None, "t", "P", 2020, "EPSG:32611", 1.0, (500000, 4000000, 500500, 4001000), "metre",
                    status="LOCAL")
    A = D.seam_audit(st, [src], GRID).set_index("feature")
    assert bool(A.loc["dem_seam", "flag"]) and not bool(A.loc["dem_flat", "flag"])


def test_mosaic_coverage_weighted(tmp_path):
    a1, T1 = native_block(2.0, 100, 100, 1, 1)
    a2, T2 = native_block(4.0, 100, 100, 1, 1)
    v = np.ones(a1.shape, bool)
    paths = []
    for i, (a, T) in enumerate(((a1, T1), (a2, T2))):
        win, out = D.align_dem_to_analysis_grid({"f": a}, v, T, "EPSG:32611", GRID, {"f": D.MEAN}, 1.0)
        paths.append(D.write_window(tmp_path / f"w{i}.tif", out, ["dem_coverage", "f"], GRID, win, {}))
    p = D.mosaic_windows(paths, GRID, ["dem_coverage", "f"], tmp_path / "m.tif")
    st = D.DemFeatureStore(p, 0.5)
    f = st.load(["f"])[0]
    assert f[1, 1] == pytest.approx(3.0)                              # equal coverage -> equal weights
    assert st.footprint()[1, 1] and not st.footprint()[5, 5]
