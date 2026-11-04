"""G07 geomorphometry: physical units, nodata-aware filters, orientation periodicity, scarps, tiled == whole."""
import numpy as np
import pytest

from gems_bench.g06.raster_io import TileWindow, iter_windows
from gems_bench.g07 import dem_features as F
from gems_bench.g07 import terrain_structure as TS

CFG = {"scales_m": [4, 10], "coarse_scales_m": [], "hillshade_azimuths": [315, 45], "relief_windows_m": [10],
       "openness_radius_m": 8, "openness_dirs": 8, "integration_factor": 2.0, "scarp_slope_deg": 12.0,
       "scarp_coherence": 0.5, "scarp_step_m": [0.5, 30.0], "lineament_min_m": 0.01, "lineament_coherence": 0.3}


def plane(slope=0.1, n=80, res=2.0):
    r, c = np.indices((n, n), dtype=np.float64)
    return slope * c * res


def test_slope_units_degrees():
    d = F.smooth_derivs(plane(1.0), 2.0, 4.0)
    s, _ = F.dem_slope(d)
    assert np.nanmedian(s[10:-10, 10:-10]) == pytest.approx(45.0, abs=0.2)
    d = F.smooth_derivs(plane(0.1), 2.0, 4.0)
    assert np.nanmedian(F.dem_slope(d)[0][10:-10, 10:-10]) == pytest.approx(np.degrees(np.arctan(0.1)), abs=0.05)


def test_hillshade_bounds_and_flat():
    d = F.smooth_derivs(np.zeros((40, 40)), 2.0, 4.0)
    hs = F.dem_hillshade(d, (315, 45))
    assert np.allclose(hs["hs315"], np.cos(np.radians(45)), atol=1e-6)
    assert np.allclose(hs["hs_range"], 0, atol=1e-9)
    d = F.smooth_derivs(plane(2.0), 2.0, 4.0)
    hs = F.dem_hillshade(d, (90, 270))
    assert (hs["hs90"] >= 0).all() and (hs["hs90"] <= 1).all()
    assert hs["hs_range"][20, 20] > 0.3                               # east-facing vs west-facing light differs


def test_plane_has_no_curvature():
    d = F.smooth_derivs(plane(0.3), 2.0, 4.0)
    cv = F.dem_curvature(d)
    assert np.nanmax(np.abs(cv["curvprof"][10:-10, 10:-10])) < 1e-8


def test_nodata_hole_creates_no_scarp():
    z = plane(0.05)
    z[30:40, 30:40] = np.nan
    f = F.dem_multiscale_features(z, 2.0, CFG, "fine")
    ring = np.zeros(z.shape, bool)
    ring[15:55, 15:55] = True
    ring[30:40, 30:40] = False
    s = f["dem_slope_s4"][ring]
    assert np.isfinite(s).any()
    assert np.nanmax(s) < 10.0              # a zero fill would create a ~50 deg cliff at the gap edge; masking: none
    assert np.nanmax(f["dem_scarpmask_s4"][ring]) == 0
    assert np.isnan(f["dem_slope_s4"][35, 35])


def test_tile_edge_invents_no_lineament():
    """A data edge through a tilted plane: inside the guard no value, beyond it no response runs along the edge. Rows and
    columns near the array border are skipped: there the halo plays this role in the real pipeline."""
    r, c = np.indices((80, 80), dtype=np.float64)
    z = 0.1 * c * 2.0 + 0.05 * r * 2.0                                      # ~6 deg: below the scarp slope threshold
    z[:, 50:] = np.nan
    f = F.dem_multiscale_features(z, 2.0, CFG, "fine")
    for s in (4, 10):
        guard = int(np.ceil(4.0 * s / 2.0))
        assert np.isnan(f[f"dem_slope_s{s}"][:, 50 - guard:50]).all()       # inside the guard: no value at all
        last = 50 - guard - 2                                               # first columns past the derivative reach
        beyond = f[f"dem_curvprof_s{s}"][21:59, 21:last]
        assert np.isfinite(beyond).all()
        assert np.max(beyond) < 1e-6                                        # the plane has no curvature, up to the guard
        assert np.max(f[f"dem_scarpmask_s{s}"][21:59, 21:last]) == 0
        assert np.max(f[f"dem_lin_s{s}"][21:59, 21:last]) == 0


def test_without_a_gap_no_guard_applies():
    f = F.dem_multiscale_features(plane(0.1), 2.0, CFG, "fine")
    assert np.isfinite(f["dem_slope_s10"][5:-5, 5:-5]).all()
    assert np.isfinite(f["dem_relief_w10"]).all()


def test_guard_can_be_disabled_for_comparison():
    z = plane(0.1)
    z[:, 50:] = np.nan
    off = F.dem_multiscale_features(z, 2.0, {**CFG, "edge_guard_sigma": 0}, "fine")
    assert np.isfinite(off["dem_slope_s10"][:, 49]).any()               # the value the guard removes by default


def test_fault_scarp_detected_with_axial_orientation():
    r, c = np.indices((80, 80), dtype=np.float64)
    z = 5.0 * (c > 40) + 0.01 * r                                       # N-S scarp, 5 m throw
    f = F.dem_multiscale_features(z, 2.0, CFG, "fine")
    assert np.nanmax(f["dem_scarp_s4"][:, 38:43]) > 2.0
    assert np.nanmax(f["dem_scarp_s4"][:, :20]) < 0.5
    st2 = f["dem_orient_s4"][40, 40]                                   # doubled strike angle of a N-S strike
    assert np.cos(st2) == pytest.approx(-1.0, abs=0.05)
    assert TS.axial(0.0, np.pi) == pytest.approx(1.0)


def test_names_match_kinds():
    z = plane(0.1)
    f = F.dem_multiscale_features(z, 2.0, CFG, "fine")
    kinds = F.feature_kinds(CFG, 2.0, "fine")
    produced = {k for k in f if not k.endswith("__w")}
    assert produced == set(kinds)
    assert set(F.aggregated_names(kinds)) >= {"dem_orient_s4_cos2", "dem_slope_s4", "dem_hs315"}


def test_tiled_equals_whole():
    rng = np.random.default_rng(0)
    r, c = np.indices((96, 88), dtype=np.float64)
    z = 3.0 * np.tanh((c - 0.5 * r - 30) / 3) + rng.normal(0, 0.05, (96, 88))
    z[50:55, 10:20] = np.nan
    halo = F.required_halo_px(CFG, 2.0, "fine")
    P = np.pad(z, halo, constant_values=np.nan)
    h, w = z.shape
    whole = F.dem_multiscale_features(P, 2.0, CFG, "fine")
    for win in iter_windows((h, w), 32, halo):
        t = P[win.row0:win.row1 + 2 * halo, win.col0:win.col1 + 2 * halo]
        ft = F.dem_multiscale_features(t, 2.0, CFG, "fine")
        rs, cs = win.core
        for k in ft:
            ref = whole[k][halo + win.row0:halo + win.row1, halo + win.col0:halo + win.col1]
            ok = np.isclose(ft[k][rs, cs], ref, rtol=1e-5, atol=1e-5, equal_nan=True)
            assert ok.all(), f"{k} differs in {win.tile_id}"
    assert isinstance(TileWindow, type)
