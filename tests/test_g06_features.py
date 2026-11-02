"""G06 feature engineering: tiled == whole-raster, masked operators, orientation periodicity, constant channels,
schema stability (synthetic fields; no data)."""
import numpy as np
import pytest

from gems_bench.g06 import feature_physics as FP
from gems_bench.g06 import feature_store as FSt
from gems_bench.g06 import feature_structure as FS
from gems_bench.g06 import feature_texture as FT
from gems_bench.g06.raster_io import TileWindow, iter_windows


def mini_cfg():
    bands = {"rtp": {"modality": "magnetic", "transform": "asinh", "scale": 100.0},
             "iso_grav_anom": {"modality": "gravity", "transform": "identity"},
             "det_elev": {"modality": "terrain", "transform": "identity"}}
    ops = ["hgm", "ocos", "osin", "lap", "hl1", "hl2", "d2n", "d2t", "curv", "edge"]
    return {"bands": bands, "grid": {"height": 80, "width": 72, "epsg": 32611, "pixel_m": 100.0, "n_bands": 3},
            "nodata": {"footprint_min_share": 0.5}, "tiling": {"tile_px": 32, "halo_px": 0, "block_px": 16},
            "groups": {
                "raw": {"enabled": True},
                "magnetic": {"enabled": True, "field": "rtp", "scales_m": [100, 300], "ops": ops},
                "gravity": {"enabled": True, "field": "iso_grav_anom", "scales_m": [100], "ops": ops},
                "multiscale": {"enabled": True, "extra_fields": ["det_elev"], "extra_scales_m": [300],
                               "extra_ops": ["hgm", "edge"],
                               "dog": {"fields": ["rtp"], "bands_m": [[100, 300]], "integration_m": [300]}},
                "texture": {"enabled": True, "sources": ["rtp"], "windows": [3, 7], "stats": ["std", "range"],
                            "mean_windows": [7], "quantile_windows": [3], "entropy_windows": [7],
                            "edge_fields": ["rtp"], "edge_windows": [7], "orientation_windows": [7],
                            "orientation_bins": 4},
                "structural": {"enabled": True, "fields": ["rtp", "iso_grav_anom"], "scales_m": [100, 300],
                               "continuity_fields": ["rtp"], "cross_modal_pairs": [["rtp", "iso_grav_anom"]],
                               "discontinuity_fields": ["rtp", "iso_grav_anom"], "discontinuity_scale_m": 100},
                "context": {"enabled": True, "regional_sigma_m": 300, "texture_fields": ["rtp"],
                            "roughness_field": "det_elev", "roughness_sigma_m": 300, "basement": "det_elev"}}}


def synthetic_fields(h=80, w=72, seed=0):
    rng = np.random.default_rng(seed)
    r, c = np.indices((h, w), dtype=np.float64)
    rtp = 200 * np.tanh((c - 0.6 * r - 20) / 4) + rng.normal(0, 5, (h, w))     # a NE-trending step (fault-like)
    grav = 3 * np.tanh((c - 0.6 * r - 22) / 6) + rng.normal(0, 0.1, (h, w))
    elev = 50 * np.sin(r / 9) + c
    for a in (rtp, grav, elev):
        a[30:34, 40:46] = np.nan                                                # interior gap
    elev[:, :3] = np.nan                                                        # band-specific missing edge
    return {"rtp": rtp, "iso_grav_anom": grav, "det_elev": elev}


def aux_of(f):
    v = f["rtp"][np.isfinite(f["rtp"])]
    return {"entropy_edges": {"rtp": list(np.unique(np.quantile(v, np.linspace(0, 1, 9)[1:-1])))},
            "median_abs": {}}


def test_tiled_equals_whole_raster():
    cfg = mini_cfg()
    names = FSt.group_names(cfg, [])
    halo = FSt.effective_halo(cfg)
    f = synthetic_fields()
    h, w = f["rtp"].shape
    padded = {k: np.pad(v, halo, constant_values=np.nan) for k, v in f.items()}
    whole = FSt.compute_tile(padded, cfg, names, [], aux_of(f), TileWindow("all", 0, h, 0, w, halo, h, w))
    for win in iter_windows((h, w), 32, halo):
        tile = {k: v[win.row0:win.row1 + 2 * halo, win.col0:win.col1 + 2 * halo] for k, v in padded.items()}
        st = FSt.compute_tile(tile, cfg, names, [], aux_of(f), win)
        for g in st:
            ref = whole[g][:, win.grid_slice[0], win.grid_slice[1]]
            bad = ~np.isclose(st[g], ref, rtol=1e-5, atol=1e-5, equal_nan=True)
            assert not bad.any(), f"{g}: {np.array(names[g])[np.unique(np.nonzero(bad)[0])][:5]} differ in " \
                                  f"{win.tile_id}"


def test_schema_is_stable_and_complete():
    cfg = mini_cfg()
    a, b = FSt.group_names(cfg, []), FSt.group_names(cfg, [])
    assert a == b
    f = synthetic_fields()
    halo = FSt.effective_halo(cfg)
    padded = {k: np.pad(v, halo, constant_values=np.nan) for k, v in f.items()}
    h, w = f["rtp"].shape
    st = FSt.compute_tile(padded, cfg, a, [], aux_of(f), TileWindow("all", 0, h, 0, w, halo, h, w))
    for g, ids in a.items():
        assert st[g].shape == (len(ids), h, w) and st[g].dtype == np.float32
    allids = [n for v in a.values() for n in v]
    assert len(allids) == len(set(allids))


def test_halo_covers_operator_support():
    cfg = mini_cfg()
    assert FSt.effective_halo(cfg) >= FS.required_halo_px(cfg["groups"]["structural"])
    assert FSt.effective_halo(cfg) >= FT.required_halo_px(cfg["groups"]["texture"])


def test_masked_smoothing_never_pulls_to_zero():
    a = np.full((40, 40), 5.0)
    a[15:25, 15:25] = np.nan
    S, _ = FP.masked_smooth(a, 3.0)
    v = S[np.isfinite(S)]
    np.testing.assert_allclose(v, 5.0, atol=1e-9)                       # a zero fill would give values < 5
    d = FP.field_at_scale(a, 300)
    assert np.nanmax(d["hgm"]) < 1e-6                                     # no artificial gradient at the gap edge


def test_constant_channel_is_flat_not_nan():
    a = np.full((30, 30), 7.0)
    out = FP.magnetic_features(a, [100, 300], field="c")
    assert np.allclose(out["mag_c_hgm_s100"], 0)
    assert np.isfinite(out["mag_c_lap_s300"]).all()
    s = FT.local_statistics(a, 7, ("std", "range"))
    assert np.allclose(s["std"], 0, atol=1e-5) and np.allclose(s["range"], 0)
    e = FT.local_entropy(a, 7, [1.0, 5.0, 10.0])
    assert np.allclose(e, 0, atol=1e-6)


def test_texture_ignores_invalid_pixels():
    a = np.full((21, 21), 2.0)
    a[10, 10] = np.nan
    a[8, 8] = np.nan
    s = FT.local_statistics(a, 5, ("mean", "min", "max"))
    assert np.isnan(s["mean"][10, 10])                                    # missing centre stays missing
    assert s["mean"][9, 9] == pytest.approx(2.0) and s["min"][9, 9] == 2.0


def test_orientation_periodicity():
    assert FS.axial_agreement(0.0, np.pi) == pytest.approx(1.0)
    assert FS.axial_agreement(0.2, 0.2 + np.pi / 2) == pytest.approx(0.0, abs=1e-12)
    r, c = np.indices((60, 60), dtype=np.float64)
    out = {}
    for name, field in (("a", np.tanh((c - 30) / 3)), ("b", -np.tanh((c - 30) / 3))):   # opposite polarity edges
        d = FP.field_at_scale(field, 100)
        st = FS.structure_tensor_features(d, 2.0)
        out[name] = (st["ocos"][30, 30], st["osin"][30, 30])
    np.testing.assert_allclose(out["a"], out["b"], atol=1e-6)            # axial: 0 and 180 deg are the same
    assert out["a"][0] == pytest.approx(1.0, abs=1e-3)                    # E-W gradient axis -> cos 2theta = 1


def test_lineament_is_coherent_and_continuous():
    r, c = np.indices((80, 80), dtype=np.float64)
    line = np.exp(-((c - 40) ** 2) / 4.0)                                 # N-S ridge
    d = FP.field_at_scale(line, 100)
    st = FS.structure_tensor_features(d, 3.0)
    ac = FS.along_cross_features(d, st["theta"])
    assert st["coh"][40, 38] > 0.8
    assert ac["along"][40, 38] > 1.0
