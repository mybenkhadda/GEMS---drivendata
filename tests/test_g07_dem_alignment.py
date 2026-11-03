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
