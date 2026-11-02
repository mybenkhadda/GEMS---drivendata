"""Synthetic self-tests of the pure G05.1 functions (run by stage 00 in Colab; also the fixtures of
tests/test_g05_1_*.py). No real data, no engine."""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from . import bands as BD
from . import config as CF
from . import displacement as DP
from . import export as KX
from . import geometry as GM
from . import graph as GR
from . import rankers as RK
from . import recall as RC
from . import registry as RG
from . import selection as SL

DC = {"side_px": [3, 6], "max_shift_px": 8, "min_length_px": 15, "smoothing_sigma_px": [0.0, 2.0],
      "control_offset_px": 12}
GC = {"max_gap_km": 2.0, "max_angle_deg": 25.0, "branch_km": 0.6, "branch_angle_deg": [20, 90], "scale_km": 1.0,
      "regional_radius_km": 10.0}
PC = {"min_disc_gain": 0.005, "recommend_disc_gain": 0.010, "max_worst_zone_drop": 0.01, "max_geo_drop": 0.005,
      "min_extra_systems": 1, "require": ["provenance_verified", "upstream_honesty_verified",
                                          "export_parity_verified"]}


# =============================================================================== synthetic builders
def line_lab(shape=(100, 100), segs=(((50, 20), (50, 80)),)) -> np.ndarray:
    """Label raster with one straight 1-px segment per ((r0, c0), (r1, c1)) (labels 1..n)."""
    lab = np.zeros(shape, np.int32)
    for k, ((r0, c0), (r1, c1)) in enumerate(segs, start=1):
        n = max(abs(r1 - r0), abs(c1 - c0)) + 1
        rr = np.round(np.linspace(r0, r1, n)).astype(int)
        cc = np.round(np.linspace(c0, c1, n)).astype(int)
        lab[rr, cc] = k
    return lab


def shifted_field(shape=(100, 100), row=50, shift=4, period=3.0) -> np.ndarray:
    """sin(c / period) above ``row``, the same pattern shifted by ``shift`` columns below it (a lateral offset)."""
    r, c = np.indices(shape, dtype=float)
    return np.where(r < row, np.sin(c / period), np.sin((c - shift) / period)).astype(np.float32)


def decision_frames(gain=0.02, zone_delta=0.0, net_new=2):
    """(S, P, ref_S, ref_P) fixed-policy frames for ``selection.promotion``."""
    ref_S = pd.DataFrame([{"scenario": "V1", "model": "A_submitted_W2", "dti": 0.25}])
    S = pd.DataFrame([{"scenario": "V1", "model": "S1", "dti": 0.25 + gain}])
    ref_P = pd.DataFrame([{"scenario": "V1", "unit": f"V1:{z}", "zone": z, "dti_heldout": 0.25,
                           "systems": "1;2"} for z in range(3)])
    new = ";".join(str(10 + i) for i in range(net_new))
    P = pd.DataFrame([{"scenario": "V1", "unit": f"V1:{z}", "zone": z, "dti_heldout": 0.25 + zone_delta,
                       "systems": "1;2" + (";" + new if z == 0 and net_new else "")} for z in range(3)])
    return S, P, ref_S, ref_P


# =============================================================================== tests
def t_bands_exact():
    av = ["iso_grav_anom_hg - horizontal gradient", "iso_grav_anom - isostatic anomaly", "rtp - reduced to pole"]
    assert BD.resolve(["iso_grav_anom"], av) == {"iso_grav_anom": 2}
    assert BD.resolve(["iso_grav_anom_hg", "rtp"], av) == {"iso_grav_anom_hg": 1, "rtp": 3}
    for bad in (["iso_grav"], ["tmi_hg"]):
        try:
            BD.resolve(bad, av)
            raise AssertionError(f"{bad} resolved by a fuzzy match")
        except BD.BandResolutionError:
            pass
    try:
        BD.resolve(["rtp"], ["rtp - a", "rtp - b"])
        raise AssertionError("duplicate identifier accepted")
    except BD.BandResolutionError:
        pass


def t_geometry_line():
    lab = line_lab()
    g = GM.geometry_table(lab).iloc[0]
    assert abs(g.chain_length_km - 6.0) < 1e-9, g.chain_length_km
    assert abs(g.strike_deg - 90.0) < 1e-6, g.strike_deg
    assert g.linearity > 0.999 and abs(g.tortuosity - 1.0) < 1e-9 and g.n_endpoints == 2


def t_geometry_diagonal():
    lab = line_lab(segs=(((10, 10), (20, 20)),))
    g = GM.geometry_table(lab).iloc[0]
    assert abs(g.chain_length_km - 10 * np.sqrt(2) * 0.1) < 1e-9
    assert abs(g.strike_deg - 135.0) < 1e-6, g.strike_deg       # row down + col right = SE-NW axis (135)


def t_graph_continuity():
    lab = line_lab(segs=(((50, 10), (50, 40)), ((50, 45), (50, 75)), ((5, 90), (30, 90))))
    G = GM.geometry_table(lab)
    D = GR.edges(G, GM.segment_pixels(lab), GC)
    cont = D[D.kind == "continuity"]
    pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(cont.src, cont.dst)}
    assert (1, 2) in pairs, pairs
    assert not any(3 in p for p in pairs), pairs
    F = GR.node_features(G, D, {1: 0.9, 2: 0.5, 3: 0.1}, GC)
    f1 = F.set_index("label")
    assert f1.loc[1, "g_component_size"] == 2 and f1.loc[3, "g_isolated"]


def t_displacement_shift():
    lab = line_lab()
    rr, cc = np.nonzero(lab == 1)
    fs = DP.smoothed(shifted_field(), DC["smoothing_sigma_px"])
    e = DP.segment_evidence(fs, rr, cc, DC)
    assert e["status"] == "MEASURED" and abs(e["shift_px"]) == 4, e
    assert e["consistent"] and e["excess"] > 0.3, e


def t_displacement_control():
    lab = line_lab()
    rr, cc = np.nonzero(lab == 1)
    r, c = np.indices((100, 100), dtype=float)
    fs = DP.smoothed(np.sin(c / 3.0).astype(np.float32), DC["smoothing_sigma_px"])
    e = DP.segment_evidence(fs, rr, cc, DC)
    assert e["status"] == "MEASURED" and e["shift_px"] == 0 and not e["consistent"], e
    short = DP.segment_evidence(fs, rr[:10], cc[:10], DC)
    assert short["status"] == "UNRESOLVABLE_SHORT"


def t_recall_union():
    E = pd.DataFrame({"W2": [0.9, 0.1, np.nan], "CNN": [0.2, 0.8, np.nan]})
    u = RC.union_max_rank(E, ["W2", "CNN"])
    assert np.allclose(u[:2], [0.9, 0.8]) and np.isnan(u[2])
    r = RC.unit_rank([3.0, np.nan, 1.0, 2.0])
    assert np.allclose(r[[0, 2, 3]], [1.0, 1 / 3, 2 / 3]) and np.isnan(r[1])
    c = RC.constructions({"experts": {"W2": [], "CNN": []}, "unions": {"Full": ["W2", "CNN"]}, "quota_rule": True})
    assert c["Full|quota"][0] == "union_quota" and "Full-minus-CNN" in c and "Full-minus-W2" not in c


def t_labels():
    T = pd.DataFrame({"y_near_truth": [0.8, 0.3, 0.05, 0.0], "w2_unit_rank": [0.5, 0.5, 0.95, 0.1],
                      "n_px": [10, 10, 10, 10]})
    L = RK.make_labels(T, {"positive_share": 0.5, "ambiguous_share": 0.1, "hard_negative_w2_q": 0.9,
                           "hard_negative_weight": 1.0, "unlabeled_weight": 0.3})
    assert list(L.cls) == ["positive", "ambiguous", "hard_negative", "unlabeled"]
    assert list(L.y) == [1, 0, 0, 0] and L.w.iloc[1] == 0 and abs(L.w.iloc[3] - 3.0) < 1e-9


def t_leakage_guard():
    try:
        RG.assert_label_free(["mean_w2", "y_near_truth"])
        raise AssertionError("truth column accepted as a feature")
    except RG.LeakageError:
        pass
    R = pd.DataFrame({c: [0.0] for c in ("mean_w2", "w2_unit_rank", "mean_cnn_b0", "y_near_truth", "g_support")})
    f = RG.feature_columns(R)
    assert "mean_w2" not in f and "w2_unit_rank" not in f and "y_near_truth" not in f and "g_support" in f
    assert "g_support" not in RG.feature_columns(R, include_graph=False)


def t_promotion():
    ok = {"provenance_verified": True, "upstream_honesty_verified": True}
    geo = ({"dti": 0.13}, {"dti": 0.13})
    S, P, rS, rP = decision_frames()
    d = SL.promotion("S1", "STRICT_LABEL_FREE", S, P, rS, rP, *geo, PC, ok)
    assert d["status"] == SL.PROMOTE and d["deferred_gates"], d
    d = SL.promotion("G1", "SCREENING_ONLY", S, P, rS, rP, *geo, PC, ok)
    assert d["status"] == SL.PROMISING, d
    S2, P2, rS2, rP2 = decision_frames(gain=0.002)
    assert SL.promotion("S1", "STRICT_LABEL_FREE", S2, P2, rS2, rP2, *geo, PC, ok)["status"] == SL.RETAIN
    S3, P3, rS3, rP3 = decision_frames(zone_delta=-0.05)
    assert SL.promotion("S1", "STRICT_LABEL_FREE", S3, P3, rS3, rP3, *geo, PC, ok)["status"] == SL.RETAIN
    d = SL.promotion("S1", "STRICT_LABEL_FREE", S, P, rS, rP, {"dti": 0.10}, {"dti": 0.13}, PC, ok)
    assert d["status"] == SL.RETAIN, d
    d = SL.promotion("S1", "STRICT_LABEL_FREE", S, P, rS, rP, *geo, PC, {"provenance_verified": True})
    assert d["status"] == SL.RETAIN, d


def t_partial_map():
    seg = np.array([[1.0, 1.0, 2.0, np.nan]])
    w2 = np.array([[0.2, 0.9, 0.1, 0.99]])
    m = SL.partial_map(seg, w2)
    assert m[0, 1] > m[0, 0] and m[0, 2] > m[0, 1] and np.isnan(m[0, 3])


def t_zone_rank():
    s = np.array([[np.nan, 1.0], [2.0, 3.0]])
    dom = np.array([[True, True], [True, False]])
    z = KX.zone_rank(s, dom)
    assert z[0, 0] == 0.0 and np.isnan(z[1, 1]) and z[1, 0] > z[0, 1]


def t_run_id():
    cfg = {"version": "g051.1", "inputs": {"a": 1}, "target_systems": {0: [1]}, "rankers": {"seed": 1}}
    a = CF.run_id(cfg, {}, "main")
    assert a == CF.run_id({**cfg, "rankers": {"seed": 2}}, {}, "main")          # stage-only setting
    assert a != CF.run_id({**cfg, "inputs": {"a": 2}}, {}, "main") and a.startswith("g051-")


def t_strict_blend():
    T = pd.DataFrame({"unit": ["u"] * 3, "w2_unit_rank": [0.2, 0.5, 0.9], "g_support": [3.0, 1.0, np.nan]})
    s = RK.strict_blend(T, 0.25)
    assert np.isclose(s[0], 0.75 * 0.2 + 0.25 * 1.0) and np.isclose(s[1], 0.75 * 0.5 + 0.25 * 0.5)
    assert np.isclose(s[2], 0.9)                       # no evidence: W2 alone, never a penalty


TESTS = [("bands_exact_identifier", True, t_bands_exact), ("geometry_line", True, t_geometry_line),
         ("geometry_diagonal", True, t_geometry_diagonal), ("graph_continuity", True, t_graph_continuity),
         ("displacement_shift", True, t_displacement_shift), ("displacement_control", True, t_displacement_control),
         ("recall_union", True, t_recall_union), ("labels", True, t_labels), ("leakage_guard", True, t_leakage_guard),
         ("promotion_gates", True, t_promotion), ("partial_segments", True, t_partial_map),
         ("zone_rank", True, t_zone_rank), ("run_id", True, t_run_id), ("strict_blend", True, t_strict_blend)]


def run_selftests() -> pd.DataFrame:
    rows = []
    for name, crit, fn in TESTS:
        try:
            fn()
            rows.append({"test": name, "status": "PASS", "critical": crit, "detail": ""})
        except Exception as e:                          # noqa: BLE001 - every failure is a row
            rows.append({"test": name, "status": "FAIL", "critical": crit,
                         "detail": f"{type(e).__name__}: {e} | {traceback.format_exc()[-400:]}"})
    return pd.DataFrame(rows)
