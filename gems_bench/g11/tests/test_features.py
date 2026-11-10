import numpy as np
import pytest

from gems_bench.g11.features import FeatureStore
from gems_bench.g11.features import geophysics as G
from gems_bench.g11.features import multiscale as MS
from gems_bench.g11.features import structural as SF
from gems_bench.g11.features import terrain as T
from gems_bench.g11.grid import GridSpec


def _grid(n=80, px=100.0):
    return GridSpec("EPSG:32611", (px, 0, 0, 0, -px, n * px), (n, n), np.ones((n, n), bool))


def test_constant_field_has_zero_gradient(cfg):
    g = _grid()
    fc = G.compute_geophysical_features({"magnetic": np.full(g.shape, 7.0)}, g, cfg)
    gm = fc["magnetic_gm_200m"][10:-10, 10:-10]
    assert np.nanmax(np.abs(gm)) < 1e-9


def test_ramp_gradient_units(cfg):
    g = _grid()
    x = np.arange(g.shape[1]) * 100.0
    ramp = np.tile(2.0 * x / 1000.0, (g.shape[0], 1))               # 2 units per km eastward
    fc = G.compute_geophysical_features({"magnetic": ramp}, g, cfg)
    assert np.nanmedian(fc["magnetic_dx_400m"][20:-20, 20:-20]) == pytest.approx(2.0, rel=1e-3)
    assert np.nanmedian(np.abs(fc["magnetic_dy_400m"][20:-20, 20:-20])) < 1e-6


@pytest.mark.parametrize("strike", [0.0, 45.0, 90.0, 135.0])
def test_orientation_rotation(cfg, strike):
    g = _grid(100)
    rr, cc = np.indices(g.shape)
    x, y = cc * 100.0, -rr * 100.0
    a = np.radians(strike)
    across = x * np.cos(a) - y * np.sin(a)                           # constant along the strike direction
    f = np.sin(2 * np.pi * across / 1500.0)
    fc = G.compute_geophysical_features({"magnetic": f}, g, cfg)
    c2, s2 = fc["magnetic_sc2_400m"][30:-30, 30:-30], fc["magnetic_ss2_400m"][30:-30, 30:-30]
    est = (np.degrees(np.arctan2(np.nanmean(s2), np.nanmean(c2))) / 2) % 180
    d = abs(est - strike) % 180
    assert min(d, 180 - d) < 5.0


def test_magrav_agreement_bounds(cfg):
    g = _grid(100)
    rr, cc = np.indices(g.shape)
    stripes = np.sin(2 * np.pi * cc * 100.0 / 1500.0)                 # coherent N-S structure
    rot = np.sin(2 * np.pi * rr * 100.0 / 1500.0)                     # coherent E-W structure
    same = G.compute_geophysical_features({"magnetic": stripes, "gravity": stripes}, g, cfg)["magrav_orient_agree_400m"]
    perp = G.compute_geophysical_features({"magnetic": stripes, "gravity": rot}, g, cfg)["magrav_orient_agree_400m"]
    for v in (same, perp):
        assert np.nanmax(v) <= 1 + 1e-6 and np.nanmin(v) >= -1 - 1e-6
    assert np.nanmedian(same[30:-30, 30:-30]) > 0.5                  # parallel strikes agree
    assert np.nanmedian(perp[30:-30, 30:-30]) < -0.5                 # perpendicular strikes disagree


def test_terrain_slope_on_plane(cfg):
    n, res = 200, 1.0
    rr, cc = np.indices((n, n))
    z = 0.1 * cc * res                                               # 10 % slope rising east
    tc = T.compute_terrain_features(z, np.ones(z.shape, bool), res, [5, 10], cfg)
    s = tc["slope_5m"][40:-40, 40:-40]
    assert np.nanmedian(s) == pytest.approx(np.degrees(np.arctan(0.1)), abs=0.01)
    asp = np.degrees(np.arctan2(tc["aspect_s_5m"], tc["aspect_c_5m"])) % 360
    assert np.nanmedian(asp[40:-40, 40:-40]) == pytest.approx(270.0, abs=0.5)   # downslope = west
    assert np.nanmax(np.abs(tc["profile_curvature_5m"][40:-40, 40:-40])) < 1e-6


def test_terrain_scale_support(cfg):
    ok, skipped = T.supported_scales([1, 5, 250], 1.0, (100, 100), 2.0)
    assert ok == [5.0] and len(skipped) == 2


def test_nodata_edge_produces_no_gradient(cfg):
    n = 200
    z = np.full((n, n), 50.0)
    valid = np.ones((n, n), bool)
    valid[:, 100:] = False
    z[~valid] = np.nan
    tc = T.compute_terrain_features(z, valid, 1.0, [10], cfg)
    g = tc["gradient_10m"]
    assert np.isnan(g[:, 85:100]).all()                              # guard band next to nodata
    assert np.nanmax(g) < 1e-9


def test_scarp_is_detected(cfg):
    n = 240
    rr, cc = np.indices((n, n))
    z = 3.0 * (cc > 120)
    tc = T.compute_terrain_features(z.astype(float), np.ones(z.shape, bool), 1.0, [10], cfg)
    s = tc["scarp_likelihood_10m"]
    col = np.nanargmax(np.nanmean(s[40:-40], 0))
    assert abs(col - 120) <= 3
    strike = (np.degrees(np.arctan2(np.nanmean(tc["strike_s2_10m"][40:-40, 110:130]),
                                    np.nanmean(tc["strike_c2_10m"][40:-40, 110:130]))) / 2) % 180
    assert min(strike, 180 - strike) < 5                               # N-S scarp


class _FakeReader:
    def __init__(self, step=3.0):
        self.step = step

    def read_dem_window(self, bounds, crs, resolution=1.0, halo_m=0.0, **k):
        from gems_bench.g11.dem_reader import DEMWindow

        x0, y0, x1, y1 = bounds[0] - halo_m, bounds[1] - halo_m, bounds[2] + halo_m, bounds[3] + halo_m
        w, h = int((x1 - x0) / resolution), int((y1 - y0) / resolution)
        cc, rr = np.meshgrid(np.arange(w), np.arange(h))
        x = x0 + resolution * (cc + 0.5)
        z = (100 + 0.02 * x + self.step * (x > 1000)).astype(np.float32)
        return DEMWindow(z, (resolution, 0, x0, 0, -resolution, y1), crs, np.ones(z.shape, bool),
                         np.zeros(z.shape, np.int16), ["fake"], resolution, np.zeros(z.shape, bool), [], 0)


def test_cross_fault_profile_recovers_step():
    from shapely.geometry import LineString

    seg = LineString([(1000, 0), (1000, 400)])                         # N-S line; right-hand side = east
    p = SF.sample_cross_fault_profiles(seg, _FakeReader(3.0), [-60, -40, -20, -10, 10, 20, 40, 60], 50.0, "EPSG:32611")
    f = SF.extract_cross_fault_features(p, 5.0)
    assert f["elevation_step_m"] == pytest.approx(3.0, abs=0.1)
    assert f["profile_valid_fraction"] == 1.0
    f0 = SF.extract_cross_fault_features(SF.sample_cross_fault_profiles(seg, _FakeReader(0.0), [-60, -20, 20, 60],
                                                                        50.0, "EPSG:32611"), 5.0)
    assert abs(f0["elevation_step_m"]) < 0.05 and abs(f0["profile_asymmetry"]) < 0.05   # symmetric profile
    rev = SF.extract_cross_fault_features(SF.sample_cross_fault_profiles(LineString(list(seg.coords)[::-1]),
                                                                         _FakeReader(3.0), [-60, -20, 20, 60], 50.0,
                                                                         "EPSG:32611"), 5.0)
    assert rev["elevation_step_m"] == pytest.approx(-3.0, abs=0.1)     # sign follows the line direction


def test_multimodal_missing_dem_stays_nan(cfg):
    g = _grid()
    rng = np.random.default_rng(2)
    geo = G.compute_geophysical_features({"magnetic": rng.normal(size=g.shape), "gravity": rng.normal(size=g.shape)},
                                         g, cfg).arrays
    terr = {"coh": np.full(g.shape, np.nan), "sc2": np.full(g.shape, np.nan), "ss2": np.full(g.shape, np.nan),
            "scarp": np.full(g.shape, np.nan)}
    mm = MS.compute_multimodal_alignment(geo, terr, g.valid, 400)
    assert np.isnan(mm["magnetic_terrain_orient_agree_400m"]).all()
    assert np.isnan(mm["geo_without_terrain_400m"]).all()               # missing terrain is not "no terrain"


def test_feature_store_refuses_label_derived(tmp_path):
    g = _grid(10)
    st = FeatureStore(tmp_path, g)
    st.put("f1", np.ones(g.shape), "geo")
    st.put("Y_DISTANCE", np.ones(g.shape), "target", label_derived=True)
    st.flush()
    assert FeatureStore(tmp_path, g).predictors() == ["f1"]
    with pytest.raises(ValueError):
        st.put("bad", np.ones((3, 3)), "geo")
