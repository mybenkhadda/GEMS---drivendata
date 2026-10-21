"""Phase-4 unit tests on synthetic arrays (no project data needed). Run: python -m pytest tests/test_phase4_units.py -q"""
from types import SimpleNamespace

import numpy as np
import pytest

from gems_bench import rs_features
from gems_bench.phase4_eval import mask_shape_stats

GRID = SimpleNamespace(pixel_km_x=0.1, pixel_km_y=0.1, pixel_km=0.1, shape=(64, 64))


def test_geomorphon_ridge_and_valley():
    y, x = np.mgrid[0:41, 0:41].astype(float)
    ridge = -np.abs(x - 20) * 10.0                      # N-S ridge, 10 m per pixel drop
    conv, lin = rs_features.geomorphon_outputs(rs_features.geomorphon_signs(ridge, 3, 1.0, 100.0))
    assert lin[20, 20] == 1 and conv[20, 20] > 0
    conv, lin = rs_features.geomorphon_outputs(rs_features.geomorphon_signs(-ridge, 3, 1.0, 100.0))
    assert lin[20, 20] == -1 and conv[20, 20] < 0
    plane = x * 10.0
    _, lin = rs_features.geomorphon_outputs(rs_features.geomorphon_signs(plane, 3, 1.0, 100.0))
    assert lin[20, 20] == 0


def test_prank_monotone_and_bounded():
    a = np.arange(100, dtype=np.float32).reshape(10, 10)
    v = np.ones_like(a, bool)
    r = rs_features.prank(a, v)
    assert np.all(np.diff(r.reshape(-1)) >= 0) and r.min() > 0 and r.max() == 1


def test_rs_specs_unique_and_grouped():
    bands = ["mag_anom", "rtp", "tmi", "tmi_hg", "tmi_vg", "tc", "iso_grav_anom", "iso_grav_anom_hg", "iso_grav_anom_vg",
             "iso_grav_anom_slope", "det_elev", "det_elev_slope", "depth_to_base_surf", "cond_surf", "geod_2ndinv",
             "geod_shearrate", "geod_dilaterate", "deq_n100a15", "ieq_n100a15"]
    specs = rs_features.build_rs_specs(bands)
    names = [s.name for s in specs]
    assert len(names) == len(set(names)) and len(names) > 40
    assert {s.groups[0] for s in specs} <= set(rs_features.GROUPS_RS)
    assert all(s.basis == "detrended_elevation" for s in specs if s.modality == "terrain")


def test_mask_shape_stats_line_vs_blob():
    m = np.zeros((60, 60), bool)
    m[30, 5:55] = True
    line = mask_shape_stats(m)
    b = np.zeros((60, 60), bool)
    b[20:40, 20:40] = True
    blob = mask_shape_stats(b)
    assert line["elongated_area_share"] == 1.0 and blob["elongated_area_share"] == 0.0


torch = pytest.importorskip("torch")


def test_geounet_variants_shapes():
    from gems_bench.geological_encoders import GeoUNet, count_params, matched_single_width

    x = torch.randn(2, 6, 64, 64)
    single = GeoUNet([list(range(6))], 6, enc_width=8)
    assert single(x)["seg"].shape == (2, 1, 64, 64)
    dual = GeoUNet([[0, 1], [2, 3, 4, 5]], 6, enc_width=6, dec_width=8, fusion="gated", aux_dist=True, aux_orient=True)
    out = dual.eval()(x)
    assert out["seg"].shape == (2, 1, 64, 64) and out["orient"].shape == (2, 2, 64, 64)
    assert dual.gates()[0].shape == (2, 2, 64, 64)
    ctxm = GeoUNet([list(range(6))], 6, enc_width=8, context=True, context_width=4)
    assert ctxm(x, torch.randn(2, 6, 64, 64))["seg"].shape == (2, 1, 64, 64)
    w, n = matched_single_width(count_params(GeoUNet([[0, 1], [2, 3, 4, 5]], 6, enc_width=6, dec_width=8)), 6, lo=4, hi=16)
    assert abs(n - count_params(GeoUNet([[0, 1], [2, 3, 4, 5]], 6, enc_width=6, dec_width=8))) / n < 0.25


def test_context_crop_centred():
    from gems_bench.phase4_training import context_crop

    g = torch.arange(100 * 100, dtype=torch.float32).reshape(1, 100, 100)
    c = context_crop(g, 40, 40, 16, 4)                  # patch centre (48, 48) -> pooled (12, 12)
    assert c.shape == (1, 16, 16) and c[0, 8, 8] == g[0, 12, 12]
    edge = context_crop(g, -8, -8, 16, 4)               # out-of-bounds part is zero
    assert edge[0, 0, 0] == 0


def test_aux_orientation_of_lines():
    from gems_bench.phase4_training import P4Spec, build_aux

    vis = np.zeros((64, 64), bool)
    vis[5:60, 32] = True                                # N-S line -> strike 0 -> cos 2θ = 1
    sp = P4Spec("t", ["a"], aux_dist=0.2, aux_orient=0.1)
    aux, names = build_aux(vis, GRID, sp, np.ones_like(vis))
    i = {n: k for k, n in enumerate(names)}
    assert aux[i["ovalid"], 32, 32] == 1 and aux[i["ocos"], 32, 32] > 0.9
    vis = vis.T.copy()                                  # E-W line -> strike 90 -> cos 2θ = -1
    aux, names = build_aux(vis, GRID, sp, np.ones_like(vis))
    assert aux[i["ocos"], 32, 32] < -0.9
    assert aux[i["dist"], 32, 32] == 1


def test_ssl_masks_and_loss():
    from gems_bench.geophysical_ssl import SSLSpec, make_masks, ssl_loss

    sp = SSLSpec("t", list("abcd"), block_px=8)
    mods = {"terrain": [0], "magnetic": [1, 2], "gravity": [3]}
    gen = torch.Generator().manual_seed(0)
    s, c = make_masks(3, 4, 32, sp, mods, gen, torch.device("cpu"))
    assert s.shape == (3, 32, 32) and c.shape == (3, 4) and 0.2 < s.float().mean() < 0.8
    x = torch.randn(3, 4, 32, 32)
    hide = s[:, None] | c[:, :, None, None]
    out = ssl_loss(x + 0.1, x, hide, torch.ones(4), torch.ones(4), sp)
    assert torch.isfinite(out["loss"]) and out["rec"] > 0
    sp2 = SSLSpec("t", list("abcd"), mask_mode="modality")
    s2, c2 = make_masks(5, 4, 32, sp2, mods, gen, torch.device("cpu"))
    assert not s2.any() and bool((c2.sum(1) > 0).all())
