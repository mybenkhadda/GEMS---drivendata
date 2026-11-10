import numpy as np
import pytest

from gems_bench.g11 import fault_labels as FL
from gems_bench.g11.tests.conftest import FAULT


@pytest.fixture(scope="module")
def mask(synth, grid):
    return FL.label_raster(synth["raw"] / "existing_faults.tif", grid)


def test_vector_load_repair_dedup(synth, grid):
    g = FL.load_fault_geometries(synth["raw"] / "faults.gpkg", grid.crs)
    r = FL.repair_fault_geometries(g)
    assert len(g) == 2 and len(r) == 1                            # exact duplicate removed
    L = np.hypot(FAULT[1][0] - FAULT[0][0], FAULT[1][1] - FAULT[0][1])
    assert r.length_m.iloc[0] == pytest.approx(L)                 # metres
    assert r.azimuth_deg.iloc[0] == pytest.approx(135.0)          # NW-SE, axial clockwise from north


def test_rasterize_agrees_with_label_raster(synth, grid, mask):
    g = FL.repair_fault_geometries(FL.load_fault_geometries(synth["raw"] / "faults.gpkg", grid.crs))
    ras = FL.rasterize_faults(g, grid)
    d = FL.distance_to_fault_m(mask, grid)
    assert (d[ras] <= 150).all()                                   # vector trace lies within the label band


def test_distance_buffers_and_states(grid, mask, cfg):
    d = FL.distance_to_fault_m(mask, grid)
    assert (d[mask] == 0).all() and d.max() > 1000
    t = FL.make_targets(mask, grid.valid, grid, cfg)
    for a, b in zip(cfg["labels"]["buffers_m"][:-1], cfg["labels"]["buffers_m"][1:]):
        assert np.nansum(t[f"Y_BUFFER_{a}M"]) <= np.nansum(t[f"Y_BUFFER_{b}M"])
    assert np.isnan(t["Y_TRACE_100M"][~grid.valid]).all()
    assert "Y_CONFIDENCE" not in t                                 # never invented
    st = FL.label_states(mask, grid.valid, d, cfg)
    S = cfg["labels"]["states"]
    assert set(np.unique(st)) <= set(S.values())
    assert (st[~grid.valid] == S["OUTSIDE"]).all() and (st[grid.valid & mask] == S["FAULT"]).all()


def test_orientation_convention(grid, cfg):
    import geopandas as gpd
    from shapely.geometry import LineString

    ns = gpd.GeoDataFrame(geometry=[LineString([(505050, 3999000), (505050, 3991000)])], crs=grid.crs)
    c2, s2 = FL.orientation_raster(ns, grid)
    assert np.nanmean(c2) == pytest.approx(1.0) and np.nanmean(np.abs(s2)) < 1e-6   # azimuth 0 -> cos 0 = 1
    ew = gpd.GeoDataFrame(geometry=[LineString([(501000, 3995050), (509000, 3995050)])], crs=grid.crs)
    c2, _ = FL.orientation_raster(ew, grid)
    assert np.nanmean(c2) == pytest.approx(-1.0)                   # azimuth 90 -> cos 180 = -1


def test_segments_from_raster(grid, mask, cfg):
    segs = FL.segments_from_raster(mask, grid, 300.0, 50.0)
    assert len(segs) >= 1 and (segs.source == "raster_skeleton").all()
    assert segs.length_m.sum() > 8000                              # ~11.3 km trace, junction splits allowed
    ov = FL.segment_overlap_labels(segs, mask, grid, 300.0)
    assert (ov.overlap_fraction > 0.9).all()


def test_unsupported_geometry_rejected(tmp_path, grid):
    import geopandas as gpd
    from shapely.geometry import Point

    p = tmp_path / "pts.gpkg"
    gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=grid.crs).to_file(p, driver="GPKG")
    with pytest.raises(FL.FaultGeometryError):
        FL.load_fault_geometries(p, grid.crs)
