import numpy as np
import pytest

from gems_bench.g11.inference import blending as B
from gems_bench.g11.inference import candidate_generation as CG
from gems_bench.g11.inference import ranking as R
from gems_bench.g11.inference import tiling as TL
from gems_bench.g11.inference import topology as TP


def test_tiles_cover_every_pixel():
    for shape, t, o in (((100, 73), 32, 8), ((10, 10), 32, 8), ((65, 65), 32, 0)):
        cov = np.zeros(shape, int)
        for r0, r1, c0, c1 in TL.tiles(shape, t, o):
            cov[r0:r1, c0:c1] += 1
        assert (cov > 0).all()


def test_blending_is_seamless_for_constant_prediction():
    valid = np.ones((70, 90), bool)
    out = B.predict_tiled(lambda xb: np.full((len(xb), xb.shape[2], xb.shape[3]), 0.3, np.float32),
                          np.zeros((1, 70, 90), np.float32), valid, 32, 8)
    assert np.allclose(out, 0.3) and np.isfinite(out).all()
    v2 = valid.copy()
    v2[:, :5] = False
    assert np.isnan(B.predict_tiled(lambda xb: np.ones((len(xb), xb.shape[2], xb.shape[3]), np.float32),
                                    np.zeros((1, 70, 90), np.float32), v2, 32, 8)[:, :5]).all()


@pytest.fixture()
def line_prob(grid):
    from gems_bench.g11.tests.conftest import fault_distance_grid

    d = fault_distance_grid()
    p = np.exp(-(d / 120.0) ** 2).astype(np.float32)
    p[~grid.valid] = np.nan
    return p


def test_candidate_generators_find_the_line(grid, line_prob, cfg):
    cfg["candidate_generation"]["threshold_quantile"] = 0.95
    g = CG.generate_candidates(line_prob, grid.valid, grid, cfg, "synthetic")
    assert set(g.generator) >= {"threshold", "hysteresis", "ridge_nms", "skeleton"}
    assert g.is_valid.all() and (g.geometry.length >= cfg["candidate_generation"]["min_length_m"]).all()
    from gems_bench.g11.features.segment import geometry_attributes

    az = np.array([geometry_attributes(x)["azimuth_deg"] for x in g.geometry])
    assert np.median(np.abs(az - 135.0)) < 10


def test_dedup_keeps_branch_removes_duplicate(grid):
    import geopandas as gpd
    from shapely.geometry import LineString

    a = LineString([(0, 0), (0, 5000)])
    dup = LineString([(10, 100), (10, 4000)])
    branch = LineString([(0, 2500), (2500, 2500)])
    g = gpd.GeoDataFrame({"k": [0, 1, 2]}, geometry=[a, dup, branch], crs=grid.crs)
    d = TP.deduplicate(g, 0.6, 20.0)
    assert len(d) == 2 and set(d.k) == {0, 2}


def test_linking_bridges_collinear_gap_not_crossing(grid):
    import geopandas as gpd
    from shapely.geometry import LineString

    g = gpd.GeoDataFrame({"k": [0, 1, 2]}, geometry=[LineString([(0, 0), (0, 2000)]),
                                                     LineString([(0, 2400), (0, 4000)]),
                                                     LineString([(300, 1000), (2000, 1000)])], crs=grid.crs)
    out = TP.link_segments(g, 600.0, 30.0)
    assert len(out) == 2
    assert max(out.geometry.length) == pytest.approx(4000.0, abs=1)
    ids1, ids2 = TP.stable_ids(out), TP.stable_ids(out.copy())
    assert ids1 == ids2 and len(set(ids1)) == len(ids1)


def test_length_budget():
    s = np.array([0.9, 0.8, 0.7, 0.1])
    L = np.array([1000.0, 2000.0, 500.0, 100.0])
    m = R.select_by_length_budget(s, L, 3.0)
    assert m.tolist() == [True, True, False, False]
    t = R.budget_table(s, L, np.array([1, 0, 1, 0]), [1.0, 4.0])
    assert t.selected_km.tolist() == [1.0, 3.6]
