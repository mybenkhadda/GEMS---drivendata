"""G01 (gems_bench.m9.research): small synthetic checks of the axial orientation algebra, metre-based tolerances,
missing / NaN evidence, wavelength alignment and cache audit, geographic leakage guards, determinism, raster
persistence and the equal-area comparison (no Drive, no real data, no M9 object)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

FAM = {"mag": "magnetic", "grav": "gravity"}


def _cfg(**over):
    from gems_bench.m9.research.geology_io import load_g01_config

    return load_g01_config(over)


def _lam(x: float = 0.8):
    from gems_bench.m9.research.geology_wavelet import Wavelength

    return Wavelength(x, (("mag", f"{x:g}"), ("grav", f"{x:g}")))


def _window(n: int = 64, seed: int = 0, strike_m: float = 30.0, strike_g: float = 30.0, coh: float = 0.8,
            lams=(0.8,), drop: tuple = ()):
    """Synthetic cached-layer window + percentile references (both fields, given strikes / coherence)."""
    from gems_bench.m9.research.geology_wavelet import QuantileRef, channel, doubled_from_strike

    rng = np.random.default_rng(seed)
    win, refs = {}, {}
    for lam in lams:
        t = f"{lam:g}"
        for role, st in (("mag", strike_m), ("grav", strike_g)):
            e = rng.normal(size=(n, n)).astype(np.float32)
            c2, s2 = doubled_from_strike(np.full((n, n), st), np.full((n, n), coh))
            for k, a in (("energy", e), ("coherence", np.full((n, n), coh, np.float32)), ("strike_c2", c2),
                         ("strike_s2", s2)):
                name = channel(FAM[role], k, t)
                if (role, k) in drop:
                    continue
                win[name] = a
            en = channel(FAM[role], "energy", t)
            if en in win:
                refs[en] = QuantileRef.fit(win[en])
    return win, refs


# ============================================================================ orientation
def test_axial_agreement_invariant_under_180_rotation():
    from gems_bench.m9.research.geology_wavelet import (axial_agreement, doubled_agreement, doubled_from_strike,
                                                        unit_doubled)

    th = np.linspace(0, 179, 37)
    for dt in (0.0, 180.0, -180.0, 360.0):
        assert np.allclose(axial_agreement(th, th + dt), 1.0, atol=1e-6)
    assert np.allclose(axial_agreement(th, th + 90.0), 0.0, atol=1e-6)
    assert np.allclose(axial_agreement(10.0, 55.0), 0.5, atol=1e-6)
    c1, s1 = doubled_from_strike(th)
    c2, s2 = doubled_from_strike(th + 180.0)
    assert np.allclose(c1, c2, atol=1e-6) and np.allclose(s1, s2, atol=1e-6)
    a, b, _ = unit_doubled(c1, s1)
    c, d, _ = unit_doubled(*doubled_from_strike(th + 40.0))
    assert np.allclose(doubled_agreement(a, b, c, d), axial_agreement(th, th + 40.0), atol=1e-5)


def test_orientation_gate_uses_coherence_once():
    from gems_bench.m9.research.geology_wavelet import doubled_from_strike, orientation_gate

    shp = (8, 8)
    for coh in (0.2, 0.5, 1.0):
        cm, sm = doubled_from_strike(np.full(shp, 20.0), np.full(shp, coh))
        cg, sg = doubled_from_strike(np.full(shp, 200.0), np.full(shp, coh))       # same axis, +180 deg
        o = orientation_gate(cm, sm, cg, sg)
        assert np.allclose(o["A"], 1.0, atol=1e-5)                                  # A is coherence-free
        assert np.allclose(o["gate"], 1.0, atol=1e-5)
        cp, sp = doubled_from_strike(np.full(shp, 110.0), np.full(shp, coh))       # perpendicular
        o = orientation_gate(cm, sm, cp, sp)
        assert np.allclose(o["A"], 0.0, atol=1e-5)
        assert np.allclose(o["kappa"], coh, atol=1e-5)                              # sqrt(coh * coh), not coh^2
        assert np.allclose(o["gate"], 1.0 - coh, atol=1e-5)
    z = np.zeros(shp, np.float32)                                                    # no orientation: no penalty
    assert np.allclose(orientation_gate(z, z, z, z)["gate"], 1.0)


def test_builder_invariant_when_gravity_strike_rotated_180():
    from gems_bench.m9.research.geology_wavelet import build_evidence

    cfg = _cfg()
    w1, r1 = _window(strike_g=30.0)
    w2, r2 = _window(strike_g=210.0)
    a = build_evidence(w1, r1, [_lam()], cfg, (100.0, 100.0))
    b = build_evidence(w2, r2, [_lam()], cfg, (100.0, 100.0))
    for k in ("G01-D", "G01-E", "comp:orient_gate"):
        assert np.allclose(a[k], b[k], atol=1e-5, equal_nan=True)


# ============================================================================ tolerance in metres
def test_tolerance_metres_not_pixels():
    from gems_bench.m9.research.geology_wavelet import meters_to_px, soft_tolerant_max, tolerance_m

    assert meters_to_px(300, (100, 100)) == (3, 3)
    assert meters_to_px(300, (50, 50)) == (6, 6)
    assert meters_to_px(300, (100, 50)) == (3, 6)                  # anisotropic grid: per-axis conversion
    with pytest.raises(ValueError):
        meters_to_px(300, (0, 100))
    tc = {"per_wavelength": 0.25, "min_m": 200.0, "max_m": 1200.0}
    assert tolerance_m(0.4, tc) == 200.0 and tolerance_m(1.6, tc) == 400.0 and tolerance_m(10.0, tc) == 1200.0
    vals = []
    for px in (100.0, 50.0):
        n = int(4000 / px)
        a = np.zeros((n, n), np.float32)
        c = n // 2
        a[c, c] = 1.0
        t = soft_tolerant_max(a, 300.0, (px, px), n_rings=4, reach=2.0)
        off = lambda m: t[c, c + int(round(m / px))]                # noqa: E731
        vals.append([off(0), off(300), off(400), off(600), off(900)])
    assert np.allclose(vals[0], vals[1], atol=1e-6)                  # same metres -> same evidence on both grids
    assert vals[0][0] == 1.0 and vals[0][1] == pytest.approx(np.exp(-0.5), abs=1e-6) and vals[0][4] == 0.0
    assert all(x >= y for x, y in zip(vals[0], vals[0][1:]))         # monotone decay with distance
    assert np.array_equal(soft_tolerant_max(a, 0.0, (50, 50)), a)


def test_disk_footprint_is_euclidean():
    from gems_bench.m9.research.geology_wavelet import soft_tolerant_max

    a = np.zeros((41, 41), np.float32)
    a[20, 20] = 1.0
    sq = soft_tolerant_max(a, 500.0, (100, 100), n_rings=1, reach=1.0, footprint="square")
    dk = soft_tolerant_max(a, 500.0, (100, 100), n_rings=1, reach=1.0, footprint="disk")
    assert sq[25, 25] > 0 and dk[25, 25] == 0                         # corner at 707 m > 500 m: only the square sees it
    assert dk[20, 25] == sq[20, 25] > 0


# ============================================================================ missing / NaN evidence
def test_missing_gravity_field_degrades_gracefully():
    from gems_bench.m9.research.geology_wavelet import build_evidence

    cfg = _cfg()
    drop = tuple(("grav", k) for k in ("energy", "coherence", "strike_c2", "strike_s2"))
    win, refs = _window(drop=drop)
    ev = build_evidence(win, refs, [_lam()], cfg, (100.0, 100.0))
    assert "G01-A" in ev and "G01-B" not in ev
    assert np.allclose(ev["G01-C"], ev["G01-A"], equal_nan=True)     # no gravity: C reduces to magnetic evidence
    assert np.allclose(ev["G01-D"], ev["G01-A"], equal_nan=True)
    assert "comp:J" not in ev


def test_missing_magnetic_pixels_are_never_penalties():
    from gems_bench.m9.research.geology_wavelet import build_evidence, channel

    cfg = _cfg()
    win, refs = _window()
    e = win[channel("magnetic", "energy", "0.8")]
    e[:, :20] = np.nan                                                # magnetic survey gap
    ev = build_evidence(win, refs, [_lam()], cfg, (100.0, 100.0))
    assert np.isnan(ev["comp:J"][:, :20]).all()                       # co-location undefined, not zero
    assert np.isfinite(ev["G01-C"][:, :20]).all()                     # gravity evidence still scores the gap
    assert np.allclose(ev["G01-C"][:, :20], ev["G01-B"][:, :20])
    assert np.isnan(ev["G01-A"][:, :20]).all()


def test_nan_boundary_and_interior_mask():
    from gems_bench.m9.research.geology_metrics import equal_area_mask, interior_mask

    v = np.zeros((40, 40), bool)
    v[5:35, 5:35] = True
    inner = interior_mask(v, 300.0, (100, 100), touches=(False, False, False, False))
    assert inner[10:30, 10:30].all() and not inner[5:7, 5:35].any() and not (inner & ~v).any()
    assert inner[7, 10:30].all()                                      # exactly 300 m from the edge: interior
    full = np.ones((10, 10), bool)
    assert not interior_mask(full, 200.0, (100, 100), touches=(True, True, True, True))[0].any()
    assert interior_mask(full, 200.0, (100, 100), touches=(False, False, False, False)).all()
    s = np.random.default_rng(1).random((40, 40)).astype(np.float32)
    s[10:20, 10:20] = np.nan
    sel = equal_area_mask(s, inner, 0.5)
    assert not (sel & ~inner).any() and not (sel & ~np.isfinite(s)).any()


# ============================================================================ wavelength alignment / cache audit
def _names(fam, lams, kinds=("energy", "coherence", "strike_c2", "strike_s2")):
    from gems_bench.m9.research.geology_wavelet import channel

    return [channel(fam, k, t) for t in lams for k in kinds]


def test_wavelength_alignment():
    from gems_bench.m9.research.geology_io import channel_audit

    names = _names("magnetic", ["0.4", "0.8", "1.6"]) + _names("gravity", ["0.4", "0.81", "3.2"])
    T, lams = channel_audit(names, FAM, "auto", 0.02)
    assert [w.lam_km for w in lams] == [0.4, 0.8]
    assert dict(lams[1].tags) == {"mag": "0.8", "grav": "0.81"}       # within 2 %: same scale, own tags
    un = T[T.status == "unaligned"]
    assert set(zip(un.role, un.lam_km)) == {("mag", 1.6), ("grav", 3.2)}
    _, l2 = channel_audit(names, FAM, [0.8, 6.4], 0.02)
    assert [w.lam_km for w in l2] == [0.8]
    T3, _ = channel_audit(names, FAM, [0.8, 6.4], 0.02)
    assert (T3.status == "requested_missing").sum() == 1


def test_duplicate_and_missing_cached_channels():
    from gems_bench.m9.research.geology_io import channel_audit
    from gems_bench.m9.research.geology_wavelet import channel

    names = _names("magnetic", ["0.4"]) + [channel("magnetic", "energy", "0.40")] + \
        _names("gravity", ["0.4"], kinds=("energy", "coherence"))
    T, lams = channel_audit(names, FAM)
    assert len(lams) == 1 and lams[0].tag("mag") == "0.4"
    assert (T.status == "duplicate").sum() == 1
    miss = T[T.status == "missing"]
    assert set(zip(miss.role, miss.kind)) == {("grav", "strike_c2"), ("grav", "strike_s2")}
    _, none = channel_audit(_names("magnetic", ["0.4"]), FAM)        # gravity absent: nothing usable
    assert none == []


def test_layer_cache_names_round_trip(tmp_path):
    from gems_bench.m9.features import LayerCache
    from gems_bench.m9.research.geology_io import names_from_stems, xmod_names

    lc = LayerCache(tmp_path)
    names = ["m9:spec:magnetic_wav_energy@0.4km", "m9:spec:gravity_wav_strike_c2@3.2km",
             "m9:xmod:mag_grav_tol@300m_1km", "m9:xmod:mag_grav_orient@2km", "m9:xmod:mag_lcn@1km"]
    stems = [lc.path(n).stem for n in names] + ["m9_spec_magnetic_fft_anisotropy"]
    assert names_from_stems(stems) == sorted(names)
    assert xmod_names(names, ["^m9:xmod:mag_grav_tol@", "^m9:xmod:mag_grav_orient@"]) == \
        ["m9:xmod:mag_grav_orient@2km", "m9:xmod:mag_grav_tol@300m_1km"]


# ============================================================================ leakage
def test_reference_geography_disjoint_from_evaluation():
    from gems_bench.m9.research.geology_io import LeakageError
    from gems_bench.m9.research.geology_metrics import assert_disjoint, full_mask, reference_mask

    H, W = 30, 30
    valid = np.ones((H, W), bool)
    val = np.zeros((H, W), bool)
    val[:, :10] = True
    buf = np.zeros((H, W), bool)
    buf[:, 10:12] = True
    ez = np.zeros((H, W), bool)
    ez[20:, :] = True
    ctx = SimpleNamespace(data=SimpleNamespace(valid=valid),
                          folds=SimpleNamespace(train=lambda k: ~val & ~buf),
                          splits=SimpleNamespace(eval_zone=lambda d: ez))
    ref = reference_mask(ctx, "geo", 0)
    assert_disjoint(ref, val, "geo")
    assert not (ref & buf).any()
    ref_d = reference_mask(ctx, "disc_official", 0)
    assert_disjoint(ref_d, ez, "disc")
    sl = (slice(18, 30), slice(0, 30))
    ev_full = full_mask((H, W), sl, np.ones((12, 30), bool))           # evaluation crop reaches outside the zone
    with pytest.raises(LeakageError):
        assert_disjoint(ref_d, ev_full, "disc crop")


def test_scores_never_receive_labels():
    import inspect

    from gems_bench.m9.research.geology_wavelet import build_evidence

    params = set(inspect.signature(build_evidence).parameters)
    assert not params & {"labels", "truth", "faults", "y", "target"}


# ============================================================================ determinism / persistence
def test_deterministic_output():
    from gems_bench.m9.research.geology_metrics import equal_area_mask
    from gems_bench.m9.research.geology_wavelet import build_evidence

    cfg = _cfg()
    lams = [_lam(0.4), _lam(0.8)]
    w, r = _window(lams=(0.4, 0.8), seed=3)
    a = build_evidence(w, r, lams, cfg, (100.0, 100.0))
    b = build_evidence(w, r, lams, cfg, (100.0, 100.0))
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k], equal_nan=True)
    tie = np.ones((10, 10), np.float32)
    m1 = equal_area_mask(tie, np.ones((10, 10), bool), 0.1)
    assert m1.sum() == 10 and m1[0].all()                             # ties: lowest flat index first


def test_raster_save_reload_round_trip(tmp_path):
    from gems_bench.m9.research.geology_io import G01Store

    st = G01Store(tmp_path / "g01_test", fingerprint="abc")
    rng = np.random.default_rng(0)
    a = rng.random((16, 12)).astype(np.float32)
    a[0, 0] = np.nan
    m = rng.random((16, 12)) > 0.5
    st.save_raster({"G01-E": a, "comp:P_mag": a, "domain": m}, "rasters/u.npz", {"window": [0, 16, 0, 12]},
                   dtype="float32")
    arrs, meta = st.load_raster("rasters/u.npz")
    assert set(arrs) == {"G01-E", "comp:P_mag", "domain"}             # ':' restored from the stored key map
    assert np.array_equal(arrs["G01-E"], a, equal_nan=True) and arrs["domain"].dtype == bool
    assert np.array_equal(arrs["domain"], m) and meta["window"] == [0, 16, 0, 12]
    st.save_raster({"x": a}, "rasters/h.npz", {}, dtype="float16")
    h, _ = st.load_raster("rasters/h.npz")
    assert h["x"].dtype == np.float16 and np.allclose(h["x"], a, atol=1e-3, equal_nan=True)
    st.save_raster({"x": np.zeros_like(a)}, "rasters/h.npz", {}, dtype="float16")   # write-once: kept, not replaced
    h2, _ = st.load_raster("rasters/h.npz")
    assert np.allclose(h2["x"], a, atol=1e-3, equal_nan=True)
    st.flush_manifest()
    man = st.load_json("manifest.json")
    assert "rasters/u.npz" in man["artifacts"] and man["fingerprint"] == "abc"


def test_resume_markers_and_tables(tmp_path):
    from gems_bench.m9.research.geology_io import G01Store

    st = G01Store(tmp_path / "g01_x", fingerprint="f")
    assert not st.is_done("unit_geo_0")
    st.mark_done("unit_geo_0", {"n": 1})
    assert st.is_done("unit_geo_0")
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    st.save_table(df, "t", "tables", parquet=False)
    assert st.load_table("t", "tables").equals(df)


# ============================================================================ equal-area comparison / metrics
def test_equal_area_budget_is_identical_across_candidates():
    from gems_bench.m9.research.geology_metrics import equal_area_k, equal_area_mask

    rng = np.random.default_rng(7)
    dom = rng.random((50, 50)) > 0.3
    a = rng.random((50, 50)).astype(np.float32)
    b = (a ** 3).astype(np.float32)                                    # monotone transform: same selection
    c = rng.random((50, 50)).astype(np.float32)
    k = equal_area_k(dom, 0.05)
    sa, sb, sc = (equal_area_mask(s, dom, 0.05) for s in (a, b, c))
    assert sa.sum() == sb.sum() == sc.sum() == k
    assert np.array_equal(sa, sb)
    few = np.full((50, 50), np.nan, np.float32)
    few[0, :5] = 1.0
    dom2 = np.ones((50, 50), bool)
    assert equal_area_mask(few, dom2, 0.05).sum() == 5                # shortfall recorded via n_finite, never padded


def test_distance_tolerant_counts_by_hand():
    from gems_bench.m9.research.geology_metrics import block_index, summarize, unit_block_counts

    n = 20
    dom = np.ones((n, n), bool)
    truth = np.zeros((n, n), bool)
    truth[:, 10] = True
    cols = np.arange(n)[None, :].repeat(n, 0)
    score = -np.abs(cols - 12).astype(np.float32)                      # the top 5 % is exactly column 12
    blk, ids = block_index((slice(0, n), slice(0, n)), (n, n), 10)
    B = unit_block_counts({"s": score}, dom, truth, (100.0, 100.0), {"s": [0.05]}, [0.0, 100.0, 200.0], blk, ids,
                          tags={"task": "geo", "zone": 0})
    S = summarize(B, ["candidate", "tol_m"]).set_index("tol_m")
    assert S.loc[0.0, "n_pred"] == 20 and S.loc[100.0, "precision"] == 0.0
    assert S.loc[200.0, "precision"] == 1.0 and S.loc[200.0, "recall"] == 1.0
    assert S.loc[200.0, "enrichment"] == pytest.approx(1.0 / (5 / 20))   # cols 8..12 lie within 200 m of truth
    assert set(B.block) == set(ids.tolist()) and len(ids) == 4


def test_paired_bootstrap_identical_maps_and_nested_choice():
    from gems_bench.m9.research.geology_metrics import nested_choice, paired_block_bootstrap

    rng = np.random.default_rng(0)
    rows = []
    for z in range(3):
        for b in range(6):
            c = {"n_pred": 10, "n_pred_hit": int(rng.integers(0, 10)), "n_truth": 8,
                 "n_truth_hit": int(rng.integers(0, 8)), "n_domain": 100, "n_domain_near": 20}
            for name in ("x", "y"):
                rows.append({"task": "geo", "zone": z, "block": b, "candidate": name, **c})
            rows.append({"task": "geo", "zone": z, "block": b, "candidate": "w",
                         **{**c, "n_pred_hit": min(c["n_pred_hit"] + 2, 10)}})
    B = pd.DataFrame(rows)
    r = paired_block_bootstrap(B, "x", "y", "f1", n_boot=50, seed=1)
    assert r["delta"] == 0.0 and r["ci_lo"] == 0.0 and r["ci_hi"] == 0.0
    r2 = paired_block_bootstrap(B, "w", "y", "precision", n_boot=50, seed=1)
    assert r2["delta"] > 0 and r2["p_delta_positive"] == 1.0
    N = nested_choice(B, {"fam": ["w", "y"]}, "precision")
    assert set(N.chosen) == {"w"} and len(N) == 3


def test_quantile_reference():
    from gems_bench.m9.research.geology_wavelet import QuantileRef

    r = QuantileRef.fit(np.arange(1000, dtype=float))
    p = r(np.array([-5.0, 0.0, 499.5, 999.0, 2000.0, np.nan], np.float32))
    assert p[0] == 0.0 and p[-2] == 1.0 and np.isnan(p[-1]) and abs(p[2] - 0.5) < 0.01
    c = QuantileRef.fit(np.full(100, 3.0))
    assert np.allclose(c(np.array([3.0], np.float32)), 0.5)
    assert np.isnan(QuantileRef.fit(np.array([1.0, np.nan]))(np.array([1.0], np.float32))).all()
    assert QuantileRef.from_dict(r.to_dict()).n == r.n


def test_config_guards():
    from gems_bench.m9.research.geology_io import G01ConfigError

    cfg = _cfg()
    assert cfg["candidates"] == ["G01-A", "G01-B", "G01-C", "G01-D", "G01-E", "G01-F"]
    with pytest.raises(G01ConfigError):
        _cfg(tolerance={"min_mm": 1})
    with pytest.raises(G01ConfigError):
        _cfg(candidates=["G01-Z"])
    with pytest.raises(G01ConfigError):
        _cfg(evaluation={"primary_area_frac": 0.07})
