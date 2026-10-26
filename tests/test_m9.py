"""M9: small synthetic checks of the configuration guards, spectral orientation conventions, corruption leakage
rules, losses, the network's shapes / channel order, scaling, tracing, inference blending, the GBM training split,
the RL environment and the Gate-1 nesting helper (no Drive, no real data, no trained model)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================================ configuration
def test_unknown_key_rejected_shared_open():
    from gems_bench.m9.config import M9ConfigError, load_m9_config

    with pytest.raises(M9ConfigError):
        load_m9_config({"tracing": {"beam_widht": 4}})
    cfg = load_m9_config({"shared": {"evaluation": {"splits": [0]}}})
    assert cfg["_profile"]["name"] == "PILOT" and cfg["_profile"]["provisional"]


def test_profiles_and_variants_cannot_touch_the_protocol():
    from gems_bench.m9.config import M9ConfigError, load_m9_config

    with pytest.raises(M9ConfigError):
        load_m9_config({"profiles": {"PILOT": {"area_fracs": [0.1]}}})
    with pytest.raises(M9ConfigError):
        load_m9_config({"variants": {"bad": {"validation": {"area_fracs": [0.1]}}}})
    with pytest.raises(M9ConfigError):
        load_m9_config({"corruption": {"mixture": {"none": 0.5, "partial": 0.4, "heavy": 0.4}}})


def test_variant_resolution_and_stages():
    from gems_bench.m9.config import SECTIONS, load_m9_config, resolve_variant, sections_of

    cfg = load_m9_config()
    v = resolve_variant(cfg, "recon_off")
    assert v["corruption"]["mixture"]["none"] == 1.0 and cfg["corruption"]["mixture"]["none"] < 1.0
    allsec = [s for st in ("PREPARE", "TRAIN", "VALIDATE", "TRACE", "RANK", "INFER", "SUBMIT") for s in sections_of(cfg, st)]
    assert sorted(allsec) == sorted(SECTIONS)


# ============================================================================ spectral conventions
def _stripes(theta_deg: float, n: int = 192, wavelength_px: float = 8.0) -> np.ndarray:
    """Linear structures striking ``theta_deg`` (clockwise from north)."""
    rr, cc = np.mgrid[0:n, 0:n].astype(float)
    t = np.radians(theta_deg)
    nr, nc = np.sin(t), np.cos(t)                      # fault-normal (row, col) of a strike t
    return np.sin(2 * np.pi * (rr * nr + cc * nc) / wavelength_px).astype(np.float32)


def test_trace_strike_is_perpendicular_to_the_wavevector():
    from gems_bench.m9.spectral import trace_strike_from_wavevector, window_spectra

    f = _stripes(0.0)                                   # north-south traces
    fc = {"window_px": 64, "stride_px": 32, "n_sectors": 8, "short_km": [0.4, 1.6], "long_km": [1.6, 6.4],
          "min_valid_frac": 0.6}
    w = window_spectra(f, np.ones(f.shape, bool), fc)
    k = np.nanmedian(w["k_azimuth"])
    strike = float(trace_strike_from_wavevector(k))
    assert min(strike, 180 - strike) < 12.0
    assert np.nanmedian(w["anisotropy"]) > 0.6


def test_wavelet_orientation_and_coherence():
    from gems_bench.m9.spectral import wavelet_layers

    f = _stripes(90.0, n=160)                            # east-west traces
    wc = {"wavelengths_km": [0.8], "n_orient": 6, "bandwidth": 0.55, "angular_sigma_deg": 20.0, "persistence_q": 0.8}
    out = wavelet_layers(f, np.ones(f.shape, bool), wc, "t")
    c2 = np.nanmedian(out["t_wav_strike_c2@0.8km"][40:120, 40:120])
    assert c2 < -0.5                                      # cos 2 x 90 deg = -1
    assert np.nanmedian(out["t_wav_coherence@0.8km"][40:120, 40:120]) > 0.5


# ============================================================================ corruption
def _line_map(n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    seg = np.zeros((n, n), bool)
    seg[10:54, 20] = True
    seg[30, 20:50] = True
    sysmap = np.where(seg, 7, -1).astype(np.int32)
    return seg, sysmap


def test_corruptor_never_invents_context_without_displacement():
    from gems_bench.m9.config import load_m9_config
    from gems_bench.m9.corruption import Corruptor

    cc = load_m9_config()["corruption"]
    cc = {**cc, "operators": {k: v for k, v in cc["operators"].items() if k != "local_displacement"},
          "mixture": {"none": 0.0, "partial": 0.0, "heavy": 1.0}}
    seg, sm = _line_map()
    c = Corruptor(cc, np.random.default_rng(0))
    for _ in range(20):
        ctx, rem, ops, mode = c(seg, sm)
        assert mode == "heavy"
        assert not (ctx & ~seg).any() and not (rem & ~seg).any() and not (ctx & rem).any()


def test_corruptor_none_mode_hides_the_map():
    from gems_bench.m9.config import load_m9_config
    from gems_bench.m9.corruption import Corruptor

    cc = {**load_m9_config()["corruption"], "mixture": {"none": 1.0, "partial": 0.0, "heavy": 0.0}}
    seg, sm = _line_map()
    ctx, rem, ops, mode = Corruptor(cc, np.random.default_rng(1))(seg, sm)
    assert mode == "none" and not ctx.any() and not rem.any() and ops == []


# ============================================================================ losses / model
def test_support_follows_the_300m_kernel():
    torch = pytest.importorskip("torch")
    from gems_bench.m9.losses import support

    p = torch.zeros(1, 1, 9, 9)
    p[0, 0, 4, 4] = 1.0
    s = support(p)[0, 0]
    assert float(s[4, 4]) == pytest.approx(1.0)
    assert float(s[4, 5]) == pytest.approx(2 / 3, abs=1e-6)
    assert float(s[4, 6]) == pytest.approx(1 / 3, abs=1e-6)
    assert float(s[4, 7]) == 0.0


def test_tversky_surrogate_perfect_and_empty():
    torch = pytest.importorskip("torch")
    from gems_bench.m9.losses import tversky_surrogate

    y = torch.zeros(1, 1, 16, 16)
    y[0, 0, 8, 2:14] = 1
    w = torch.ones_like(y)
    kern = y.clone()
    assert float(tversky_surrogate(y, y, kern, w, 0.2, 0.8)) < 1e-3
    assert float(tversky_surrogate(torch.zeros_like(y), y, kern, w, 0.2, 0.8)) > 0.99


def test_geofaultnet_shapes_and_channel_order():
    torch = pytest.importorskip("torch")
    from gems_bench.m9.model import build, net_spec

    groups = {"raw": ["raw:a", "raw:b"], "struct": ["s1"], "spectral": [], "context": ["c1"], "dem": []}
    meta = {"local": ["raw:a", "raw:b", "s1", "m9:footprint"], "pooled": ["raw:a", "raw:b", "s1", "m9:footprint", "c1"],
            "key": "k"}
    cfg = {"data": {"patch_px": 64, "context_factor": 4},
           "model": {"stem_width": 8, "widths": [8, 16, 24, 32], "context_width": 8, "fusion": "gated", "dropout": 0.0}}
    spec = net_spec(meta, groups, cfg)
    assert spec["channel_index"] == [0, 1, 3, 2]              # raw, footprint, struct
    net = build(spec)
    out = net(torch.zeros(2, 4, 64, 64), torch.zeros(2, 2, 64, 64), torch.zeros(2, 5, 64, 64))
    assert out["seg"].shape == (2, 1, 64, 64) and out["orient"].shape == (2, 2, 64, 64)
    assert out["ep"].shape == out["jn"].shape == (2, 1, 64, 64)


# ============================================================================ scaling / context
def test_fold_scaler_uses_permitted_pixels_only():
    from gems_bench.m9.dataset import FoldScaler

    cube = np.zeros((2, 40, 40), np.float16)
    cube[0, :20] = 10.0
    cube[0, 20:] = -10.0
    allowed = np.zeros((40, 40), bool)
    allowed[:20] = True
    sc = FoldScaler.fit(cube, ["a", "m9:footprint"], allowed, 10_000, 6.0, 0)
    assert sc.center[0] == pytest.approx(10.0) and sc.center[1] == 0.0 and sc.scale[1] == 1.0


def test_context_crop_strict_mask():
    from gems_bench.m9.dataset import FoldScaler, context_crop

    pooled = np.ones((1, 32, 32), np.float32)
    psc = FoldScaler(np.zeros(1, np.float32), np.ones(1, np.float32), 6.0, ["a"], 1)
    mask = np.zeros((32, 32), bool)
    mask[:, :16] = True
    c = context_crop(pooled, psc, 40, 40, 32, 4, mask)
    assert c.shape == (1, 32, 32) and c.max() == 1.0 and (c[0][:, -4:] == 0).all()


# ============================================================================ tracing / inference
def test_beam_follows_a_ridge():
    from gems_bench.m9.config import load_m9_config
    from gems_bench.m9.tracing import beam_trace, to_pixels

    tc = load_m9_config()["tracing"]
    E = np.full((80, 80), 0.02, np.float32)
    E[:, 40] = 1.0
    theta = np.zeros_like(E)                                   # strike north-south
    valid = np.ones(E.shape, bool)
    paths, sc, _ = beam_trace(E, theta, valid, np.array([[60.0, 40.0]], np.float32), np.array([0.0], np.float32),
                              {**tc, "max_len_km": 3.0}, gap_thr=0.5)
    p = to_pixels(paths[0], E.shape)
    assert len(p) > 20 and np.abs(p[:, 1] - 40).max() <= 1 and p[-1, 0] < 60
    assert (np.abs(np.diff(p, axis=0)).max(1) <= 1).all()     # 8-connected


def test_dedup_removes_duplicates():
    from gems_bench.m9.tracing import dedup

    p = np.stack([np.arange(5, 30), np.full(25, 10)], 1).astype(np.int32)
    t = pd.DataFrame({"confidence_lf": [1.0, 0.9], "uid": ["a", "b"]})
    cs = dedup([p, p.copy()], t, (40, 40), 1, 0.6, 2)
    assert len(cs) == 1 and cs.table.uid.iloc[0] == "a"


def test_tile_ramp_is_flat_on_grid_edges():
    from gems_bench.m9.infer import _ramp

    w = _ramp(20, 20, 4, (False, True, False, True))
    assert w[0, 0] == 1.0 and w[-1, -1] < 0.3 and w[10, 10] == 1.0


# ============================================================================ GBM split / choice
def test_gbm_train_mask_drops_heldout_zone_and_buffer():
    from gems_bench.gems3.official import Unit
    from gems_bench.m9.gbm import train_mask

    ev = np.zeros((100, 100), bool)
    ev[:, :40] = True
    u = Unit("disc_official", 0, (slice(0, 100), slice(0, 100)), ev, ev, None, False, None)
    T = pd.DataFrame({"task": ["disc_official"] * 4 + ["disc_official"] * 30, "zone": [0] * 4 + [1] * 30,
                      "r_full": [50] * 4 + [50] * 30, "c_full": [10] * 4 + [45] * 15 + [90] * 15,
                      "label": [1] * 4 + [1] * 15 + [0] * 15, "network_id": [str(i) for i in range(34)]})
    tr, how = train_mask(T, u, (100, 100), 1.0)
    assert not tr[:4].any() and not tr[4:19].any() and tr[19:].all()


def test_choose_prefers_the_steadier_library_within_tolerance():
    from gems_bench.m9.gbm import choose

    class M:
        cfg = {"gbm": {"choose": {"tolerance": 0.003}}}
        prim = "disc_official"

    S = pd.DataFrame({"method": ["xgboost_all", "lightgbm_all", "catboost_all"], "task": "disc_official",
                      "dti_crossfit": [0.200, 0.198, 0.150], "fold_min": [0.10, 0.17, 0.14]})
    assert choose(M(), S)["library"] == "lightgbm"


# ============================================================================ RL / gates
def test_trace_env_observation_and_expert():
    from gems_bench.m9.config import load_m9_config
    from gems_bench.m9.rl import N_OBS, TraceCore, expert_action

    rc = {**load_m9_config()["rl"], "max_gap_px": 6}
    E = np.zeros((60, 60), np.float32)
    E[:, 30] = 1.0
    tgt = E > 0
    core = TraceCore(E, np.zeros_like(E), np.ones(E.shape, bool), rc, tgt)
    o = core.reset(50, 30, 0.0)
    assert o.shape == (N_OBS,)
    total = 0.0
    for _ in range(20):
        o, r, d = core.step(expert_action(core))
        total += r
        if d:
            break
    assert total > 0 and abs(core.c - 30) <= 1


def test_nested_component_uses_other_zones():
    from gems_bench.m9.gates import nested_component

    rows = []
    for z, gain in ((0, 0.02), (1, 0.01), (2, -0.02)):
        for fs, dti in (("a", 0.2 + gain), ("b", 0.2)):
            rows.append({"task": "geo", "zone": z, "feature_set": fs, "dti": dti, "tp_w": dti * 100, "fp_w": 10.0,
                         "fn_w": (1 - dti) * 100})
    t = nested_component(pd.DataFrame(rows), "x", "a", "b", 0.003)
    assert len(t) == 3 and t.set_index("zone").loc[2, "decided_on_other_zones"]
