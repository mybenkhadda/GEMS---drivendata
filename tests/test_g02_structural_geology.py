"""G02 (gems_bench.m9.research): synthetic checks of the rotated-frame geometry and map conventions, displacement
(step vs oblique fabric, sign, rotation and 180-degree invariance), truncation (endpoints facing a line, data-edge
exclusion), the lineament network (junctions, colinear continuation, coordinates), hypothesis maps, leakage guards of
the nested choices, invalid data, determinism, decisions, the G01 review and saved-artifact reload. No Drive, no real
data, no M9 object. (Authored in code-only mode: not executed yet.)"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

PX = (100.0, 100.0)


def _cfg(**over):
    from gems_bench.m9.research.g02_io import load_g02_config

    return load_g02_config({"mode": "FULL_EVALUATION", **over})


def _fabric(n=160, strike=70.0, fault_col=None, offset_px=0, seed=0):
    """Straight, NON-periodic bands of azimuth ``strike``; east of ``fault_col`` the block is moved north by
    ``offset_px`` pixels (apparent sinistral offset on a north-striking line)."""
    rng = np.random.default_rng(seed)
    g = ndimage.gaussian_filter1d(rng.normal(size=4 * n), 2.0)
    r, c = np.indices((n, n), dtype=np.float64)
    if fault_col is not None:
        r = np.where(c >= fault_col, r + offset_px, r)
    E, N = c * PX[1], -r * PX[0]
    b = np.radians(strike)
    s = E * np.cos(b) - N * np.sin(b)                            # coordinate across the bands
    return np.interp(s / 100.0 + 2 * n, np.arange(4 * n), g).astype(np.float32)


# ============================================================================ frame / conventions
def test_offset_px_and_frame_conventions():
    from gems_bench.m9.research.geology_displacement import frame_vectors, offset_px

    assert np.allclose(offset_px(100.0, 0.0, PX), (-1.0, 0.0))     # north = up = row - 1
    assert np.allclose(offset_px(100.0, 90.0, PX), (0.0, 1.0))     # east = col + 1
    (tE, tN), (nE, nN) = frame_vectors(0.0)
    assert np.allclose((tE, tN, nE, nN), (0.0, 1.0, 1.0, 0.0), atol=1e-12)
    for phi in (0.0, 33.0, 90.0, 147.0):
        (tE, tN), (nE, nN) = frame_vectors(phi)
        assert abs(tE * nE + tN * nN) < 1e-12 and abs(np.hypot(tE, tN) - 1) < 1e-12


def test_rotframe_round_trip_and_row_direction():
    from gems_bench.m9.research.geology_displacement import RotFrame

    r, c = np.indices((64, 64), dtype=np.float32)
    a = np.sin(r / 9.0) + np.cos(c / 7.0)
    for phi in (0.0, 30.0, 90.0, 135.0):
        fr = RotFrame(a.shape, PX, phi, 100.0)
        back = fr.from_rot(fr.to_rot(a))
        inner = (slice(8, -8), slice(8, -8))
        assert np.nanmax(np.abs(back[inner] - a[inner])) < 0.05
    fr = RotFrame(a.shape, PX, 0.0, 100.0)
    R = fr.to_rot(-r)                                              # value increases northwards
    assert np.nanmean(np.diff(R, axis=0)) == pytest.approx(1.0, abs=0.05)   # rows run along +t (north)


def test_select_by_azimuth_is_axial():
    from gems_bench.m9.research.geology_displacement import select_by_azimuth

    maps = {0.0: np.zeros((2, 2)), 90.0: np.ones((2, 2))}
    th = np.array([[179.0, 1.0], [91.0, np.nan]])
    out = select_by_azimuth(maps, th)
    assert out[0, 0] == 0 and out[0, 1] == 0 and out[1, 0] == 1 and np.isnan(out[1, 1])


def test_dog_peaks_at_wavelength():
    from gems_bench.m9.research.geology_displacement import dog_sigmas_m

    lam = 1600.0
    s1, s2 = dog_sigmas_m(lam, 2.0)
    k = np.linspace(1e-5, 0.02, 20000)
    resp = np.exp(-0.5 * (s1 * k) ** 2) - np.exp(-0.5 * (s2 * k) ** 2)
    assert 2 * np.pi / k[np.argmax(resp)] == pytest.approx(lam, rel=0.02)


def test_window_sum_matches_brute_force():
    from gems_bench.m9.research.geology_truncation import window_sum

    a = np.random.default_rng(1).random((7, 9)).astype(np.float32)
    S = window_sum(a, -2, -1, 1)
    for j in range(9):
        lo, hi = max(j - 2, 0), max(j - 1 + 1, 0)
        assert np.allclose(S[:, j], a[:, lo:hi].sum(1) if hi > lo else 0.0, atol=1e-5)


# ============================================================================ displacement
def _disp(field, cfg, valid=None):
    from gems_bench.m9.research.geology_displacement import displacement_evidence

    v = np.ones(field.shape, bool) if valid is None else valid
    return displacement_evidence(field, v, {}, [0.8], PX, cfg)


def test_displacement_detects_step_offset_and_sign():
    cfg = _cfg(orientation={"n_azimuths": 4})
    d = _disp(_fabric(fault_col=80, offset_px=3), cfg)
    near = d["score"][30:130, 77:84]
    far = d["score"][30:130, 15:35]
    assert np.nanmean(near) > 3 * max(np.nanmean(far), 1e-4)
    strong = near > np.nanpercentile(near, 75)
    off = d["offset_m"][30:130, 77:84][strong]
    assert 100.0 <= np.nanmedian(off) <= 500.0                     # +300 m (east block north = apparent sinistral)
    az = d["azimuth"][30:130, 77:84][strong]
    assert np.mean(np.abs(((az + 90) % 180) - 90) < 1) > 0.5        # line found at azimuth 0 (north)


def test_continuous_oblique_fabric_is_not_displacement():
    cfg = _cfg(orientation={"n_azimuths": 4})
    faulted = _disp(_fabric(fault_col=80, offset_px=3), cfg)["score"][30:130, 77:84]
    intact = _disp(_fabric(), cfg)["score"][30:130, 77:84]
    assert np.nanmean(faulted) > 2 * max(np.nanmean(intact), 1e-4)


def test_displacement_sign_invariant_to_line_flip():
    from gems_bench.m9.research.geology_displacement import RotFrame, bandpass, disp_scale, displacement_rot

    cfg = _cfg()
    f = bandpass(_fabric(fault_col=80, offset_px=3), np.ones((160, 160), bool), 0.8, PX)
    sc = disp_scale(0.8, PX, cfg["displacement"])
    meds = []
    for phi in (0.0, 180.0):
        fr = RotFrame(f.shape, PX, phi, sc.step_m)
        res = displacement_rot(fr.to_rot(f), None, sc, cfg["displacement"])
        meds.append(np.nanmedian(fr.from_rot(res["offset_m"])[40:120, 79:82]))
    assert np.sign(meds[0]) == np.sign(meds[1]) and abs(meds[0] - meds[1]) < 150.0


def test_displacement_rotation_equivariance():
    cfg = _cfg(orientation={"n_azimuths": 4})                     # tested azimuths closed under 90-degree rotation
    f = _fabric(fault_col=80, offset_px=3)
    a = _disp(f, cfg)["score"]
    b = _disp(np.rot90(f).copy(), cfg)["score"]
    x, y = np.rot90(a)[20:-20, 20:-20].ravel(), b[20:-20, 20:-20].ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    assert np.corrcoef(x[ok], y[ok])[0, 1] > 0.7


def test_displacement_invalid_data_and_determinism():
    cfg = _cfg(orientation={"n_azimuths": 2})
    f = _fabric(fault_col=80, offset_px=3)
    valid = np.ones(f.shape, bool)
    valid[:, :40] = False
    f = np.where(valid, f, np.nan)
    d1, d2 = _disp(f, cfg, valid), _disp(f, cfg, valid)
    assert np.isnan(d1["score"][:, :40]).all()
    for k in d1:
        np.testing.assert_array_equal(d1[k], d2[k])


# ============================================================================ truncation
def _terminated_ridges(n=160, end_col=80):
    r, c = np.indices((n, n))
    E = 0.5 + 0.45 * np.exp(-0.5 * ((r % 10) - 5.0) ** 2)
    E = np.where(c >= end_col, 0.3, E).astype(np.float32)
    strike = np.full((n, n), 90.0, np.float32)                     # E-W ridges
    return E, strike


def test_skeleton_endpoints_and_junctions():
    from gems_bench.m9.research.geology_truncation import skeleton_nodes

    s = np.zeros((20, 20), bool)
    s[10, 2:18] = True
    ep, jn = skeleton_nodes(s)
    assert ep.sum() == 2 and not jn.any()
    s[2:10, 10] = True                                             # T junction
    ep, jn = skeleton_nodes(s)
    assert ep.sum() == 3 and jn[9:12, 9:12].any()


def test_truncation_highlights_line_where_ridges_end():
    from gems_bench.m9.research.geology_truncation import ridge_skeleton, truncation_evidence

    cfg = _cfg(orientation={"n_azimuths": 4})
    E, st = _terminated_ridges()
    valid = np.ones(E.shape, bool)
    sk = ridge_skeleton(E, st, valid, PX, 0.8, 500.0)
    assert sk[:, 10:70].any() and not sk[:, 85:].any()
    t = truncation_evidence({0.8: sk}, {0.8: E}, {0.8: st}, valid, PX, cfg)
    assert np.nanmean(t["score"][30:130, 76:84]) > 2 * max(np.nanmean(t["score"][30:130, 30:50]), 1e-3)
    assert np.nanmax(t["edge_endpoints"][:, :5]) > 0                 # endpoints at the data edge are set aside


# ============================================================================ network
def _network_evidence(n=160):
    """Deterministic: smooth low ramp + 3-pixel-wide lines (centre 1.0, shoulders 0.6) so the area-fraction
    hysteresis thresholds fall on the lines and no background speckle is attached."""
    r, c = np.indices((n, n))
    e = (0.3 * (r + c) / (2.0 * n)).astype(np.float32)

    def hline(row, c0, c1):
        e[row, c0:c1] = 1.0
        e[row - 1, c0:c1] = np.maximum(e[row - 1, c0:c1], 0.6)
        e[row + 1, c0:c1] = np.maximum(e[row + 1, c0:c1], 0.6)

    hline(40, 20, 141)                                             # horizontal line
    e[10:101, 80] = 1.0                                            # vertical line crossing it (junction)
    e[10:101, 79] = np.maximum(e[10:101, 79], 0.6)
    e[10:101, 81] = np.maximum(e[10:101, 81], 0.6)
    hline(140, 20, 61)                                             # colinear pair with a 900 m gap
    hline(140, 70, 111)
    return e


def test_network_junctions_continuation_and_raster():
    from gems_bench.m9.research.geology_network import build_network

    cfg = _cfg()
    e = _network_evidence()
    valid = np.ones(e.shape, bool)
    net = build_network(e, valid, PX, cfg)
    S = net["segments"]
    assert len(S) >= 4
    assert (S.n_junctions > 0).any()
    assert S.straightness.max() > 0.95
    pair = S[(S.row0 - 140).abs().le(2) & (S.row1 - 140).abs().le(2)]
    assert len(pair) >= 2 and pair.continuation_kind.str.contains("colinear").any()
    assert (pair.continuation > 0).all()                           # supported, but isolated segments are still scored
    assert (S.score >= 0).all()
    r = net["raster"]
    assert np.isfinite(r).all() and np.nanmax(r) <= S.score.max() + 1e-6


def test_network_isolated_segment_keeps_most_of_its_score():
    cfg = _cfg()
    cw = float(cfg["network"]["continuation_weight"])
    assert 0 <= cw < 1 and (1 - cw) >= 0.75


def test_segment_features_coordinates():
    from gems_bench.m9.research.geology_network import build_network, segment_features

    cfg = _cfg()
    net = build_network(_network_evidence(), np.ones((160, 160), bool), PX, cfg)
    rc_to_xy = lambda r, c: (500000.0 + (np.asarray(c) + 0.5) * 100.0, 4000000.0 - (np.asarray(r) + 0.5) * 100.0)  # noqa: E731
    feats = segment_features(net["graph"], net["segments"], rc_to_xy, r0=1000, c0=2000)
    assert feats
    p = net["graph"]["paths"][feats[0]["properties"]["segment"]]
    x0, y0 = feats[0]["coords"][0]
    assert x0 == pytest.approx(500000.0 + (p["cols"][0] + 2000 + 0.5) * 100.0)
    assert y0 == pytest.approx(4000000.0 - (p["rows"][0] + 1000 + 0.5) * 100.0)
    json.dumps(feats[0]["properties"])                              # JSON-serialisable properties


# ============================================================================ hypotheses
def test_hypotheses_fault_vs_ambiguous():
    from gems_bench.m9.research.geology_hypotheses import hypotheses
    from gems_bench.m9.research.geology_wavelet import doubled_from_strike

    cfg = _cfg(orientation={"n_azimuths": 2})
    n = 80
    c2, s2 = doubled_from_strike(np.zeros((n, n)), np.ones((n, n)))
    disp = np.zeros((n, n), np.float32)
    disp[:, 38:42] = 0.8
    one = np.ones((n, n), np.float32)
    inp = {"L": 0.9 * one, "disp": disp, "trunc": 0 * one, "straight": one, "persist": one,
           "vectors": [(c2, s2, one)], "raw_pct": {}, "raw": {}, "raw_iqr": {}}
    h = hypotheses(inp, np.ones((n, n), bool), PX, cfg)
    for k in ("hyp:fault", "hyp:contact", "hyp:intrusive", "hyp:ambiguous", "G02-E"):
        v = h[k][np.isfinite(h[k])]
        assert (v >= -1e-6).all() and (v <= 1 + 1e-6).all()
    assert (h["hyp:dominant"][30:50, 39:41] == 0).all()            # displacement on a lineament -> fault
    assert (h["hyp:dominant"][30:50, 10:20] == 3).all()            # strong lineament, no disruption -> ambiguous
    assert h["G02-E"][40, 40] > h["G02-E"][40, 15]


# ============================================================================ leakage / nested choices
def _blocks(rows):
    from gems_bench.m9.research.geology_metrics import COUNTS

    out = []
    for task, zone, cand, hit in rows:
        out.append({"task": task, "zone": zone, "candidate": cand, "area_frac": 0.03, "tol_m": 300.0, "block": 0,
                    "n_pred": 100, "n_pred_hit": hit, "n_truth": 100, "n_truth_hit": hit, "n_domain": 1000,
                    "n_domain_near": 100})
    return pd.DataFrame(out)[["task", "zone", "candidate", "area_frac", "tol_m", "block", *COUNTS]]


def test_nested_virtual_never_uses_the_held_out_zone():
    from gems_bench.m9.research.g02_metrics import nested_virtual

    B = _blocks([("geo", 0, "v1", 90), ("geo", 0, "v2", 10), ("geo", 1, "v1", 10), ("geo", 1, "v2", 50),
                 ("geo", 2, "v1", 10), ("geo", 2, "v2", 50)])
    rows, ch = nested_virtual(B, ["v1", "v2"], "VB", "f1", 0.03, 300.0)
    assert ch.set_index("zone").chosen[0] == "v2"                  # v1 wins zone 0 but zone 0 cannot choose for itself
    assert int(rows[rows.zone == 0].n_pred_hit.iloc[0]) == 10


def test_nested_virtual_excludes_overlapping_zones():
    from gems_bench.m9.research.g02_metrics import exclusions, nested_virtual, unit_overlap

    m = np.zeros((16, 16), bool)
    m[:8] = True
    fps = {("geo", 0): {"r0": 0, "c0": 0, "stride": 1, "mask": m}, ("geo", 1): {"r0": 4, "c0": 0, "stride": 1, "mask": m},
           ("geo", 2): {"r0": 100, "c0": 0, "stride": 1, "mask": m}}
    ov = unit_overlap(fps)
    assert len(ov) == 1 and {int(ov.zone_a.iloc[0]), int(ov.zone_b.iloc[0])} == {0, 1}
    ex = exclusions(ov)
    B = _blocks([("geo", 0, "v1", 10), ("geo", 0, "v2", 10), ("geo", 1, "v1", 90), ("geo", 1, "v2", 10),
                 ("geo", 2, "v1", 10), ("geo", 2, "v2", 60)])
    _, ch = nested_virtual(B, ["v1", "v2"], "VB", "f1", 0.03, 300.0, ex)
    assert ch.set_index("zone").chosen[0] == "v2"                  # zone 1 overlaps zone 0: only zone 2 may choose


def test_reference_geography_disjoint_guard():
    from gems_bench.m9.research.geology_io import LeakageError
    from gems_bench.m9.research.geology_metrics import assert_disjoint

    a = np.zeros((5, 5), bool)
    b = np.zeros((5, 5), bool)
    a[0, 0] = b[0, 0] = True
    with pytest.raises(LeakageError):
        assert_disjoint(a, b, "test")


def test_tie_noise_is_deterministic_and_small():
    from gems_bench.m9.research.g02_metrics import tie_noise

    a, b = tie_noise((8, 8), 7, 1e-6), tie_noise((8, 8), 7, 1e-6)
    np.testing.assert_array_equal(a, b)
    assert a.max() < 1e-6 and a.min() >= 0


def test_lazy_scores_membership_does_not_compute():
    from gems_bench.m9.research.g02_experiment import LazyScores, fusion_name, parse_fusion

    def boom(k):
        raise AssertionError("computed")
    L = LazyScores(["a", "b"], boom)
    assert "a" in L and "z" not in L and len(L) == 2
    assert parse_fusion(fusion_name("G01-E", "G02-B", 0.25)) == ("G01-E", "G02-B", 0.25)
    assert parse_fusion("G01-E") is None


# ============================================================================ decisions
def _H(verdicts, mode_zones=5):
    rows = []
    for task, ref, v, p in verdicts:
        rows.append({"family": "G02-A", "kind": "incremental", "task": task, "candidate": f"F:{ref}+G02-A",
                     "reference": ref, "verdict": v, "p_delta_positive": p, "delta": 0.01 if p > 0.5 else -0.01,
                     "n_zones": mode_zones})
    return pd.DataFrame(rows)


def test_decide_rules():
    from gems_bench.m9.research.g02_metrics import decide

    cfg = _cfg()
    dc = cfg["decision"]
    keep = decide(_H([("disc_official", "G01-E", "SUPPORTED", 0.97), ("geo", "G01-E", "INCONCLUSIVE", 0.7)]),
                  "FULL_EVALUATION", dc, ["G02-A"])
    assert keep.recommendation.iloc[0] == "KEEP"
    fast = decide(_H([("disc_official", "G01-E", "SUPPORTED", 0.97)]), "FAST_RESEARCH", dc, ["G02-A"])
    assert fast.recommendation.iloc[0] == "UNVERIFIED" and fast.provisional.iloc[0] == "KEEP"
    rej = decide(_H([("disc_official", "G01-E", "REFUTED", 0.02), ("geo", "G01-E", "INCONCLUSIVE", 0.4)]),
                 "FULL_EVALUATION", dc, ["G02-A"])
    assert rej.recommendation.iloc[0] == "REJECT"
    few = decide(_H([("disc_official", "G01-E", "SUPPORTED", 0.97)], mode_zones=2), "FULL_EVALUATION", dc, ["G02-A"])
    assert few.recommendation.iloc[0] == "UNVERIFIED"
    none = decide(pd.DataFrame(columns=["family", "kind"]), "FULL_EVALUATION", dc, ["G02-B"])
    assert none.recommendation.iloc[0] == "UNVERIFIED"


# ============================================================================ configuration
def test_config_modes_and_validation():
    from gems_bench.m9.research.g02_io import G02ConfigError, load_g02_config

    fast = load_g02_config({"mode": "FAST_RESEARCH"})
    full = load_g02_config({"mode": "FULL_EVALUATION"})
    assert fast["wavelengths_km"] == [0.8, 1.6] and full["wavelengths_km"] == "auto"
    assert fast["evaluation"]["max_units_per_task"] == 2 and full["evaluation"]["max_units_per_task"] is None
    with pytest.raises(G02ConfigError):
        load_g02_config({"mode": "FULL_EVALUATION", "not_a_key": 1})
    with pytest.raises(G02ConfigError):
        load_g02_config({"mode": "FULL_EVALUATION", "evaluation": {"ablation_area_fracs": [0.05]}})
    with pytest.raises(G02ConfigError):
        load_g02_config({"mode": "BOTH"})


# ============================================================================ G01 review
def test_g01_review_absent_and_present(tmp_path):
    from gems_bench.m9.research.g01_review import review_g01

    rev = review_g01(None)
    assert not rev["available"]
    assert rev["questions"].answer.str.contains("ABSENT").all()
    assert set(rev["decision_table"]["Keep / Reject / Investigate"]) == {"UNVERIFIED"}
    root = tmp_path / "g01_x"
    (root / "tables").mkdir(parents=True)
    pd.DataFrame([{"hypothesis": "G01-D", "task": t, "candidate": "G01-D", "reference": "G01-C", "delta": -0.02,
                   "p_delta_positive": 0.03, "zone_win_share": 0.2, "n_zones": 5, "verdict": "REFUTED"}
                  for t in ("geo", "disc_official")]).to_csv(root / "tables" / "hypothesis_contrasts.csv", index=False)
    rev = review_g01(root)
    D = rev["decision_table"].set_index("Component")
    assert D.loc["axial strike compatibility", "Keep / Reject / Investigate"] == "Reject"
    assert D.loc["multiscale persistence", "Keep / Reject / Investigate"] == "UNVERIFIED"
    assert not (root / "processed").exists()                       # the review never writes into a G01 run


# ============================================================================ artifacts
def test_store_stream_raster_and_vectors_reload(tmp_path):
    from gems_bench.m9.research.g02_io import G02Store

    st = G02Store(tmp_path / "g02_test", fingerprint="abc")
    for d in ("config", "processed", "rasters", "lineaments", "tables", "figures", "reports", "manifests"):
        assert (st.root / d).is_dir()
    a = np.random.default_rng(0).random((12, 10)).astype(np.float32)
    m = a > 0.5
    arrs = {"G02-A": a, "hyp:dominant": np.full((12, 10), 2, np.int8), "domain": m}
    st.save_raster_stream(list(arrs), arrs.__getitem__, "rasters/units/geo_0.npz", {"task": "geo"})
    got, meta = st.load_raster_keys("rasters/units/geo_0.npz", ["G02-A", "domain", "hyp:dominant"])
    assert meta["task"] == "geo" and got["G02-A"].dtype == np.float16
    assert np.allclose(got["G02-A"], a, atol=1e-3) and (got["domain"] == m).all() and (got["hyp:dominant"] == 2).all()
    assert set(st.raster_keys("rasters/units/geo_0.npz")) == set(arrs)
    loaded, _ = st.load_raster("rasters/units/geo_0.npz")         # G01 reader is compatible
    assert set(loaded) == set(arrs)
    feats = [{"coords": [(0.0, 0.0), (100.0, 50.0)], "properties": {"segment": 0, "score": 0.5}}]
    st.save_vectors(feats, "lineaments/geo_0_G01-E", 32611)
    fc = json.loads((st.root / "lineaments" / "geo_0_G01-E.geojson").read_text())
    assert fc["features"][0]["geometry"]["coordinates"][1] == [100.0, 50.0]
    assert fc["crs"]["properties"]["name"] == "EPSG:32611"
    st.mark_done("unit_geo_0")
    st.flush_manifest()
    assert st.is_done("unit_geo_0") and (st.root / "manifests" / "manifest.json").exists()


def test_mapstore_resume(tmp_path):
    from gems_bench.m9.research.g02_experiment import MapStore

    ms = MapStore(tmp_path / "tmp_geo_0", (10, 10))
    ms.put("G02-A", (slice(0, 5), slice(None)), np.ones((5, 10), np.float32))
    ms.put("ridge[mag]@0.8km", (slice(0, 5), slice(None)), np.ones((5, 10), bool))
    ms.mark_tile({(0, 0)})
    ms.close()
    ms2 = MapStore(tmp_path / "tmp_geo_0", (10, 10))
    assert ms2.tiles_done() == {(0, 0)}
    assert np.isnan(ms2["G02-A"][6, 0]) and ms2["G02-A"][0, 0] == 1
    assert ms2["ridge[mag]@0.8km"].dtype == np.uint8
