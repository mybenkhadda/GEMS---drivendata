"""Synthetic runtime self-tests of the EDA's pure functions (stage 00; also run by tests/test_geological_eda.py).
Nothing here reads competition data: every array is constructed in the test."""
from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import core as K
from . import data as DA


# =============================================================================== synthetic fixtures
class _Stack:
    def __init__(self, grid, bands: dict):
        self.grid, self._b = grid, bands
        self.names = list(bands)
        self.valid = np.ones(grid.shape, bool)

    def __getitem__(self, k):
        return self._b[k]


def synthetic_grid(h: int = 200, w: int = 240):
    from affine import Affine

    from gems_eda.io import Grid

    return Grid(Affine(100.0, 0.0, 500_000.0, 0.0, -100.0, 4_000_000.0), "EPSG:32611", h, w, 0.1, 0.1, False)


def synthetic_data(h: int = 200, w: int = 240, row: int = 100) -> DA.EDAData:
    """A straight E-W mapped fault at ``row`` with a magnetic step across it and a flat gravity field."""
    from gems_eda.enhance import Enhancer
    from gems_eda.morphology import build_fault_network

    g = synthetic_grid(h, w)
    lab = np.zeros((h, w), bool)
    lab[row, 20:w - 20] = True
    rr = np.arange(h)[:, None] * np.ones((1, w))
    rtp = np.where(rr < row, 0.0, 10.0).astype(np.float32) + 0.01 * np.arange(w)[None, :]
    bands = {"rtp": rtp, "iso_grav_anom": np.zeros((h, w), np.float32),
             "det_elev": (0.1 * rr).astype(np.float32)}
    return DA.EDAData(_Stack(g, bands), lab, np.ones((h, w), bool), build_fault_network(lab, g, (3, 3)), Enhancer(g))


class _FakeE:
    """Minimum of an ExhaustiveEDA the registry needs: a workspace, the config, a stage name."""

    def __init__(self, root: Path):
        from gems_bench.leaderboard_focus.session import Workspace

        self.ws = Workspace(root)
        self.c = K.eda_config({"figures": {"format": ["png", "svg"], "dpi": 60}})
        self.current_stage = "selftest"


# =============================================================================== tests
def t_cliffs_delta():
    assert K.cliffs_delta([2, 3, 4], [0, 1]) == 1.0 and K.cliffs_delta([0, 1], [2, 3]) == -1.0
    assert abs(K.cliffs_delta([1, 2, 3], [1, 2, 3])) < 1e-12
    assert abs(K.auc([1, np.nan, 3], [2]) - 0.5) < 1e-12
    return "delta = 2 AUC - 1; NaNs dropped"


def t_status_rule():
    assert K.status_of(0.3, 0.2, 0.4, 0.1) == K.SUPPORTED
    assert K.status_of(-0.3, -0.4, -0.2, 0.1) == K.SUPPORTED
    assert K.status_of(0.0, -0.05, 0.05, 0.1) == K.NOT_SUPPORTED
    assert K.status_of(0.1, -0.02, 0.25, 0.1) == K.INCONCLUSIVE
    assert K.status_of(0.5, np.nan, np.nan, 0.1) == K.INCONCLUSIVE
    return "CI vs +-min_effect band"


def t_cluster_bootstrap_grouping():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 30)
    f = rng.normal(1.0, 1.0, len(g)) + np.repeat(rng.normal(0, 1, 20), 30)   # system-level shared noise
    c = rng.normal(0.0, 1.0, len(g)) + np.repeat(rng.normal(0, 1, 20), 30)
    e = K.cluster_effect(f, g, c, g, 200, 1)
    assert e["n_groups"] == 20 and e["ci_lo"] < e["effect"] < e["ci_hi"]
    few = K.cluster_effect(f[:90], g[:90], c[:90], g[:90], 50, 1)            # 3 groups -> no CI
    assert few["n_groups"] == 3 and np.isnan(few["ci_lo"])
    assert K.effect_result(few, 0.1).status == K.INCONCLUSIVE
    a = K.cluster_effect(f, g, c, g, 200, 7)
    b = K.cluster_effect(f, g, c, g, 200, 7)
    assert a == b
    return f"20 systems: delta {e['effect']:+.2f} [{e['ci_lo']:+.2f}, {e['ci_hi']:+.2f}]; < 5 systems -> no CI; seeded"


def t_robustness():
    R = pd.DataFrame({"effect": [0.3, 0.25, -0.2, 0.4]})
    r = K.robustness(R, 0.3, 0.1)
    assert r["n_regions"] == 4 and abs(r["same_sign_share"] - 0.75) < 1e-12 and r["stability"] == "STABLE"
    assert K.robustness(pd.DataFrame({"effect": []}), 0.3, 0.1)["stability"] == "NOT_EVALUATED"
    return "same-sign share and label"


def t_circular_stats():
    m, R = K.axial_mean_deg([179.0, 1.0])
    assert min(m, 180 - m) < 1e-6 and R > 0.99
    assert float(K.axial_diff_deg(179.0, 1.0)) == 2.0 and float(K.axial_diff_deg(0.0, 90.0)) == 90.0
    return "axial mean across the 0/180 wrap; axial difference <= 90"


def t_robust_zscore():
    a = np.array([[1.0, 2.0, 3.0, 1000.0]], np.float32)
    z, med, mad = DA.zscore(a, np.ones(a.shape, bool))
    assert med == 2.5 and mad > 0 and abs(z[0, 1]) < 1
    return "median / MAD scaling resists outliers"


def t_coordinates():
    g = synthetic_grid()
    x, y = g.rc_to_xy(np.array([0]), np.array([0]))
    assert float(x[0]) == 500_050.0 and float(y[0]) == 3_999_950.0
    e, n = g.rc_to_km(np.array([10]), np.array([20]))
    assert float(e[0]) == 2.0 and float(n[0]) == -1.0
    assert abs(g.pixel_km - 0.1) < 1e-12
    return "pixel centres in UTM; local km; 100 m pixels"


def t_profile_sampling():
    off = DA.offsets(1.0)
    assert len(off) == 21 and off[10] == 0.0 and off[0] == -1.0
    a = np.tile(np.arange(100, dtype=np.float32), (50, 1))                      # value = column
    S = pd.DataFrame({"r": [25], "c": [50], "dr": [0.0], "dc": [10.0]})          # normal along +col, 10 px / km
    v = DA.along_normal(a, S, off)
    assert np.allclose(v[0], 50 + 10 * off)
    b = a.copy()
    b[25, 52:56] = np.nan
    w = DA.Sampler(b)(np.array([25.0]), np.array([53.5]))
    assert np.isnan(w[0])
    ang = np.zeros((3, 3), np.float32)
    ang[1, 1], ang[1, 2] = 179.0, 1.0
    assert DA.Sampler(ang, order=0)(np.array([1.0]), np.array([1.4]))[0] == 179.0
    return "km offsets x px-per-km normal; NaN propagation; nearest sampling for angles"


def t_rowcorr():
    x = np.tile(np.arange(10.0), (3, 1))
    y = np.vstack([np.arange(10.0), -np.arange(10.0), np.r_[np.arange(5.0), [np.nan] * 5]])
    r = DA.rowcorr(x, y)
    assert abs(r[0] - 1) < 1e-12 and abs(r[1] + 1) < 1e-12 and np.isnan(r[2])
    return "row-wise Pearson with min_n"


def t_station_metrics_step():
    off = DA.offsets(5.0)
    cube = np.where(off < 0, 0.0, 4.0)[None, None, :].astype(np.float32)
    S = pd.DataFrame({"sample_id": [0], "kind": ["fault"], "system": [1], "region": [0], "domain": [0]})
    M = DA.station_metrics(cube, S, ["sig"], off)
    assert abs(M.abs_step.iloc[0] - 4.0) < 1e-6 and abs(M.grad_peak_offset_km.iloc[0]) <= 0.1
    return f"|step| {M.abs_step.iloc[0]:.2f}, gradient peak at {M.grad_peak_offset_km.iloc[0]:+.1f} km"


def t_tangent_normal():
    D = synthetic_data()
    st = DA.fault_stations(D, K.eda_config({"sampling": {"profile_half_width_km": 2}}), np.random.default_rng(0))
    assert len(st) and (st.system > 0).all()
    assert np.allclose(np.abs(st.dr.to_numpy()), 10.0, atol=1.0) and np.allclose(st.dc.to_numpy(), 0.0, atol=1.0)
    return f"{len(st)} stations on an E-W fault; normal is N-S (|dr| = 10 px / km)"


def t_matched_controls():
    D = synthetic_data()
    c = K.eda_config({"sampling": {"profile_half_width_km": 2}})
    rng = np.random.default_rng(0)
    st = DA.fault_stations(D, c, rng)
    strata = np.zeros(D.grid.shape, np.int16)
    C = DA.matched_controls(D, st, strata, 3, 2.0, 6.0, 22, rng)
    assert len(C) == 3 * len(st)
    assert C.dist_known_km.between(2.0, 6.0).all()
    assert (C.system.to_numpy() == st.system.to_numpy()[C.matched_to.to_numpy()]).all()
    assert (C.match_level != "global").mean() > 0.5
    return f"{len(C)} controls in [2, 6] km, systems inherited, levels {C.match_level.value_counts().to_dict()}"


def t_lineament_identity():
    D = synthetic_data()
    crest = {"magnetic": D.labels.copy(), "terrain": np.zeros(D.grid.shape, bool)}
    crest["terrain"][30, 40:120] = True
    L, lab = DA.lineaments(D, crest, 1.0)
    assert len(L) == 2 and set(np.unique(lab)) == {0, 1, 2}
    m = L[L.status == "mapped"]
    u = L[L.status == "unmapped"]
    assert len(m) == 1 and len(u) == 1 and m.families.iloc[0] == "magnetic" and u.families.iloc[0] == "terrain"
    assert abs(m.azimuth.iloc[0] - 90) < 1 and m.n_families.iloc[0] == 1
    for r in L.itertuples():
        assert int(lab[r.r_mid, r.c_mid]) == r.lineament_id or (lab == r.lineament_id).sum() == r.n_px
    return "label-free lineaments keep their id, family and mapping status; E-W strike = 90 deg"


def t_error_taxonomy():
    from .parts_hi import error_class, fp_class

    assert error_class(0.6, 0.9, 0.1) == "detected"
    assert error_class(0.2, 0.7, 0.1) == "localisation_failure"
    assert error_class(0.1, 0.2, 0.95) == "missed_ranked_high"
    assert error_class(0.1, 0.2, 0.3) == "detection_failure"
    assert error_class(0.1, 0.2, np.nan) == "detection_failure"
    assert fp_class(1.0, 0.0) == "near_fp_0.3_2km" and fp_class(5.0, 0.8) == "far_fp_with_lineament"
    assert fp_class(5.0, 0.1) == "far_fp_no_evidence"
    return "forensic class rules"


def t_dti_formula():
    from gems_bench.geomoe.evaluate import dti

    tp, fp, fn = 30.0, 50.0, 70.0
    assert abs(dti(tp, fp, fn) - tp / (tp + 0.2 * fp + 0.8 * fn + 1e-6)) < 1e-9
    return "pooled DTI used by F087 equals the engine's"


def t_config_guards():
    try:
        K.eda_config({"models": {"train_models": True}})
        raise AssertionError("training was not refused")
    except K.EDAConfigError:
        pass
    try:
        K.eda_config({"mode": "TRAIN"})
        raise AssertionError("bad mode accepted")
    except K.EDAConfigError:
        pass
    q = K.eda_config({"mode": "quick_look"})
    assert q["mode"] == "QUICK_LOOK" and q["n_boot"] == K.PROFILES["QUICK_LOOK"]["n_boot"]
    assert K.eda_config({"mode": "RESUME", "reuse_cached": False})["reuse_cached"] is True
    return "no training; modes validated; QUICK_LOOK profile"


def t_provenance_and_missing_modality():
    for fam, b in DA.PRIMARY.items():
        assert b in DA.PROVENANCE or ":" in b, f"no provenance for {b}"
    from .parts_efg import fam_cols

    X = pd.DataFrame(columns=["rtp|abs_step", "iso_grav_anom|abs_step", "magnetic:structure|side_corr"])
    assert fam_cols(X, "terrain") == [] and len(fam_cols(X, "magnetic")) == 2
    return "primary bands carry provenance; a missing family yields no columns (figure skipped, not faked)"


def t_figure_export_and_catalog():
    with tempfile.TemporaryDirectory() as d:
        E = _FakeE(Path(d))
        reg = K.Registry(E)
        from . import viz as V

        p = V.plt()
        fig, ax = p.subplots()
        ax.plot([0, 1], [0, 1])
        s = K.Spec("F999", "t", "test", "Z", "q", "c", "w", "m", {K.SUPPORTED: "yes."}, {K.SUPPORTED: "do."}, "l", 1)
        reg.save(s, fig, pd.DataFrame({"x": [1]}), K.Result(K.SUPPORTED, "measured 1", {"effect": np.float64(0.3)}, 1))
        reg.skip(K.Spec("F998", "u", "skip", "Z", "q", "c", "w", "m", {}, {}, "l"), "no data")
        j = E.ws.read_json("figures/interpretations/F999.json")
        C = E.ws.load_table("figures/FIGURE_CATALOG.csv")
        assert j["measured"]["status"] == K.SUPPORTED and j["interpretation"].startswith("yes.")
        assert j["measured"]["effect"] == 0.3
        assert E.ws.path("figures/F999_t.png").exists() and E.ws.path("figures/F999_t.svg").exists()
        assert E.ws.path("figures/data/F999_t.csv").exists() and not E.ws.path("figures/F998_u.png").exists()
        assert set(C.figure_id) == {"F998", "F999"} and C.set_index("figure_id").status["F998"] == K.NOT_AVAILABLE
    return "PNG + SVG + data + interpretation from the Result; skipped figure has no image"


def t_annotation_persistence():
    from gems_bench.reset import atlas as AT

    cat = pd.DataFrame({"candidate_id": ["GDE000_A", "GDE001_B"], "review_split": ["TRAIN", "EVAL"],
                        "geometry_wkt": ["", ""]})
    R0 = pd.DataFrame(columns=list(AT.REVIEW_FIELDS))
    r = {"candidate_id": "GDE000_A", "interpretation": "POSSIBLE_FAULT", "confidence": "MEDIUM",
         "review_mode": "EVIDENCE_ONLY", "reviewer": "t", "evidence_sources": "magnetic"}
    R1, _ = AT.append_reviews(R0, [r], cat)
    R2, _ = AT.append_reviews(R1, [{**r, "interpretation": "LIKELY_CONTACT"}], cat)
    R3, _ = AT.append_reviews(R2, [{**r, "interpretation": "LIKELY_CONTACT"}], cat)
    assert len(R2) == 2 and len(R3) == 2 and R2.iloc[0].interpretation == "POSSIBLE_FAULT"
    assert AT.latest_reviews(R2).iloc[0].interpretation == "LIKELY_CONTACT"
    return "append-only; repeats skipped; latest review wins for display only"


def t_synthesis_rules():
    from .synthesis import hypothesis_status, pick_executive

    assert hypothesis_status([K.SUPPORTED, K.DESCRIPTIVE]) == "PROMOTE_TO_EXPERIMENT"
    assert hypothesis_status([K.NOT_SUPPORTED]) == "DO_NOT_PRIORITISE"
    assert hypothesis_status([K.SUPPORTED, K.NOT_SUPPORTED]) == "MIXED_EVIDENCE"
    assert hypothesis_status(["NOT_RUN", K.NOT_AVAILABLE]) == "NOT_TESTED"
    C = pd.DataFrame({"figure_id": [f"F{i:03d}" for i in range(40)] + ["D1"],
                      "status": [K.SUPPORTED, K.DESCRIPTIVE, K.NOT_AVAILABLE, K.INCONCLUSIVE] * 10 + [K.DESCRIPTIVE],
                      "priority": [1, 2] * 20 + [1], "png": ["x.png"] * 41, "title": ["t"] * 41})
    X = pick_executive(C)
    assert 10 <= len(X) <= 15 and not (X.status == K.NOT_AVAILABLE).any() and not X.figure_id.str.startswith("D").any()
    return "register statuses; executive selection 10-15, no dashboards / unavailable figures"


TESTS = [("cliffs_delta", True, t_cliffs_delta), ("status_rule", True, t_status_rule),
         ("cluster_bootstrap_grouping", True, t_cluster_bootstrap_grouping), ("robustness", True, t_robustness),
         ("circular_stats", True, t_circular_stats), ("robust_zscore", True, t_robust_zscore),
         ("coordinates", True, t_coordinates), ("profile_sampling", True, t_profile_sampling),
         ("rowcorr", True, t_rowcorr), ("station_metrics_step", True, t_station_metrics_step),
         ("tangent_normal", True, t_tangent_normal), ("matched_controls", True, t_matched_controls),
         ("lineament_identity", True, t_lineament_identity), ("error_taxonomy", True, t_error_taxonomy),
         ("dti_formula", True, t_dti_formula), ("config_guards", True, t_config_guards),
         ("provenance_missing_modality", True, t_provenance_and_missing_modality),
         ("figure_export_catalog", True, t_figure_export_and_catalog),
         ("annotation_persistence", True, t_annotation_persistence), ("synthesis_rules", True, t_synthesis_rules)]


def run_selftests() -> pd.DataFrame:
    rows = []
    for name, critical, fn in TESTS:
        try:
            rows.append({"test": name, "critical": critical, "status": "PASS", "detail": fn()})
        except Exception as e:
            rows.append({"test": name, "critical": critical, "status": "FAIL",
                         "detail": f"{type(e).__name__}: {e} | {traceback.format_exc()[-600:]}"})
    return pd.DataFrame(rows)
