"""M8: small synthetic checks of the physics operators, profiles, candidate geometry, Monte Carlo sampling, the
walker, rasterisation helpers and the configuration guards (no Drive, no models, no real data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================================ configuration
def test_profiles_cannot_change_evaluation_definitions():
    from gems_bench.m8.config import M8ConfigError, load_m8_config

    with pytest.raises(M8ConfigError):
        load_m8_config({"profiles": {"PILOT": {"area_fracs": [0.1]}}})


def test_mc_refuses_band_derived_operators():
    from gems_bench.m8.config import M8ConfigError, load_m8_config

    with pytest.raises(M8ConfigError):
        load_m8_config({"monte_carlo": {"families": [{"family": "magnetic", "operators": ["tilt_edge_band"],
                                                      "scales_m": [200]}]}})


def test_unknown_key_rejected_but_shared_open():
    from gems_bench.m8.config import M8ConfigError, load_m8_config

    with pytest.raises(M8ConfigError):
        load_m8_config({"generation": {"min_len_kmm": 0.3}})
    cfg = load_m8_config({"shared": {"evaluation": {"splits": [0]}}})
    assert cfg["shared"]["evaluation"]["splits"] == [0] and cfg["_profile"]["name"] == "PILOT"


def test_layer_name_parsing():
    from gems_bench.m8.layers import parse

    assert parse("magnetic:tilt_edge@400") == ("magnetic", "tilt_edge", 400.0)
    assert parse("worms:magnetic") is None


# ============================================================================ potential-field operators
def test_thdr_is_per_km():
    from gems_bench.m8.potential_fields import operator

    c = np.tile(np.arange(80, dtype=np.float32), (80, 1))
    a, _ = operator("thdr", 2.0 * c, 1.0, px_km=(0.1, 0.1))
    assert abs(float(np.median(a[20:60, 20:60])) - 20.0) < 0.5             # 2 units / 100 m = 20 / km


def test_tilt_edge_peaks_on_a_contact():
    from gems_bench.m8.potential_fields import operator

    c = np.tile(np.arange(100, dtype=np.float32), (100, 1))
    f = np.tanh((c - 50.0) / 3.0).astype(np.float32)
    a, info = operator("tilt_edge", f, 1.0, px_km=(0.1, 0.1))
    k = int(np.nanargmax(a[50, 20:80])) + 20
    assert abs(k - 50) <= 3 and info["amplitude_floor"] > 0


def test_nan_footprint_is_restored_not_propagated():
    from gems_bench.m8.potential_fields import operator

    f = np.random.default_rng(0).normal(size=(60, 60)).astype(np.float32)
    f[:10] = np.nan
    a, _ = operator("thdr", f, 2.0)
    assert np.isnan(a[:10]).all() and np.isfinite(a[15:]).all()


def test_upward_continuation_removes_short_wavelengths():
    from gems_bench.m8.potential_fields import upward

    rng = np.random.default_rng(1)
    f = rng.normal(size=(128, 128))
    g = upward(f, 1.0)
    assert np.std(np.diff(g, axis=1)) < 0.2 * np.std(np.diff(f, axis=1))


def test_axial_difference():
    from gems_bench.m8.potential_fields import axial_diff

    assert float(axial_diff(10.0, 170.0)) == pytest.approx(20.0)
    assert float(axial_diff(0.0, 90.0)) == pytest.approx(90.0)


def test_block_quantile_map_is_local():
    from gems_bench.m8.potential_fields import block_quantile_map

    a = np.zeros((40, 40), np.float32)
    a[:, 20:] = 10.0
    t = block_quantile_map(a, np.ones(a.shape, bool), 20, 0.5, min_count=10)
    assert t[5, 5] < 1.0 and t[5, 35] > 9.0


# ============================================================================ profiles
def _pc():
    return {"station_step_px": 2, "smooth_px": 5, "tangent_px": 4, "offsets_m": [100, 300, 800],
            "min_valid_share": 0.99, "continuity_threshold": 0.5}


def test_profile_contrast_sees_a_step():
    from gems_bench.m8.profiles import summarize, transect

    c = np.tile(np.arange(100), (100, 1))
    F = np.where(c >= 50, 1.0, 0.0).astype(np.float32)
    path = np.c_[np.arange(10, 90), np.full(80, 50)]
    tr = transect(F, np.ones(F.shape, bool), path, _pc())
    s = summarize(tr, 1.0, _pc(), "f")
    assert s["f_contrast_med_300m"] == pytest.approx(1.0, abs=0.05)
    assert s["f_sign_consistency"] == pytest.approx(1.0) and s["f_continuity"] == pytest.approx(1.0)


def test_normals_stay_on_one_side_of_a_curved_path():
    from gems_bench.m8.profiles import stations

    t = np.linspace(0, np.pi / 2, 60)
    path = np.c_[np.rint(50 + 30 * np.sin(t)), np.rint(50 + 30 * np.cos(t))].astype(int)
    _, n = stations(path, 2, 5, 4)
    assert (np.sum(n[1:] * n[:-1], axis=1) > 0).all()


# ============================================================================ candidates / rasterisation
def test_split_curvature_cuts_a_right_angle():
    from gems_bench.m8.candidates import split_curvature

    a = np.c_[np.full(30, 50), np.arange(20, 50)]
    b = np.c_[np.arange(49, 19, -1), np.full(30, 49)]
    out = split_curvature(np.vstack([a, b]), 60.0, 5)
    assert len(out) == 2


def test_regional_rank_and_spread():
    from gems_bench.m8.rasterize import _spread, regional_rank

    sr = np.full((20, 20), np.nan, np.float32)
    sr[5, 2:18] = np.linspace(0, 1, 16)
    r0 = regional_rank(sr, np.ones(sr.shape, bool), 10, 0.0)
    assert np.isfinite(r0[5, 2:18]).all() and np.isinf(r0[0, 0])
    core = np.zeros((20, 20), bool)
    core[5, 2:18] = True
    ids = np.where(core, 0, -1).astype(np.int32)
    q = _spread(core, ids, np.array([1]), 2)
    assert q[5, 10] == 1.0 and 0 < q[6, 10] < 1.0 and q[8, 10] == 0.0


def test_fusion_weight_simplex():
    from gems_bench.m8.ensemble import simplex

    w = simplex(0.25, (1, 1, 0))
    assert len(w) == 3 and all(abs(sum(x) - 1) < 1e-9 and x[2] == 0 and x[0] > 0 and x[1] > 0 for x in w)
    assert simplex(0.25, (1, 0, 0)) == [(1.0, 0.0, 0.0)]


# ============================================================================ signal recovery
def test_failure_reason_first_lost_stage():
    from gems_bench.m8.signal_recovery import failure_reason

    r = pd.Series({"raw": True, "gradient": True, "normalized": False, "crest": False})
    assert failure_reason(r) == ("gradient_present -> normalized_missing", "extraction_bottleneck")
    r2 = pd.Series({"raw": False, "gradient": False, "crest": False})
    assert failure_reason(r2)[1] == "information_limit_under_tested_operators"
    r3 = pd.Series({s: True for s in ("raw", "gradient", "normalized", "directional", "crest", "fragment",
                                      "persistent", "linked")} | {"ranked": False})
    assert failure_reason(r3)[1] == "ranking_bottleneck"


# ============================================================================ Monte Carlo / completion
def test_draw_distributions():
    from gems_bench.m8.montecarlo import draw

    rng = np.random.default_rng(0)
    assert 0.0 <= draw({"dist": "uniform", "lo": 0.0, "hi": 0.1}, rng) <= 0.1
    assert draw({"dist": "choice", "values": [2, 3]}, rng) in (2, 3)
    assert draw({"dist": "choice"}, rng, ["thdr", "tdx"]) in ("thdr", "tdx")


def test_walker_follows_an_evidence_ridge():
    from gems_bench.m8.completion import walk

    ev = np.zeros((100, 100), np.float32)
    ev[50, 5:] = 1.0
    strike = np.full(ev.shape, 90.0, np.float32)                          # E-W strike
    cc = {"headings_deg": [-15.0, -7.5, 0.0, 7.5, 15.0], "temperature": 0.05, "curvature_penalty": 0.15,
          "orientation_weight": 0.3, "cone_deg": 30.0, "stop_evidence": 0.35, "stop_patience_px": 6,
          "gap_attraction": 0.5}
    out = walk(50, 10, 0.0, 30, 8, ev, strike, np.ones(ev.shape, bool), np.zeros(ev.shape, bool), cc,
               np.random.default_rng(0))
    rows = np.concatenate([o["path"][:, 0] for o in out if len(o["path"])])
    assert len(rows) and np.mean(np.abs(rows - 50)) < 1.5
