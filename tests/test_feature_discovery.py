"""Tests of gems_eda.feature_discovery (synthetic only: small arrays and the Layer Lab demo raster). Authored, not
executed at authoring time."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
LAB = REPO / "streamlit" / "gems_layer_lab"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from gems_eda.feature_discovery import catalog as CAT  # noqa: E402
from gems_eda.feature_discovery import metrics as M  # noqa: E402


# ------------------------------------------------------------------ metrics
def _line_case():
    f = np.zeros((40, 40), bool)
    f[20, 5:35] = True
    x = np.random.default_rng(0).normal(size=(40, 40)).astype(np.float32)
    x[20, 5:35] += 6
    return x, f, np.ones((40, 40), bool)


def test_association_both_directions_and_budget_on_full_population():
    x, f, v = _line_case()
    a = M.association_both(x, f, v, (0.02,))
    assert a["status"] == "OK" and a["auroc_high"] > 0.95 and a["auroc_low"] == pytest.approx(1 - a["auroc_high"])
    assert a["recall_2pct_high"] > 0.9 and a["recall_2pct_low"] < 0.1
    assert a["ap_lift_high"] > a["ap_lift_low"] and a["better_direction_EXPLORATORY"] == "high"
    b = M.association_both(-x, f, v, (0.02,))
    assert b["auroc_low"] == pytest.approx(a["auroc_high"])          # sign twin: same information, reversed


def test_insufficient_never_zero():
    x = np.ones((20, 20), np.float32)
    f = np.zeros((20, 20), bool)
    f[5, 5] = True
    r = M.association_both(x, f, np.ones_like(f))
    assert r["status"] == M.INSUFFICIENT and "auroc_high" not in r


def test_recovery_exact_300m_components_and_localisation():
    x, f, v = _line_case()
    lab, n = M.fault_components(f)
    d_km = (np.abs(np.arange(40) - 20)[:, None] * np.ones((1, 40)) * 0.1).astype(np.float32)
    shifted = np.roll(x, 2, axis=0)                                    # response 200 m off the trace
    r = M.recovery(shifted, f, v, 30 / 1600, lab, n, dist_km=d_km)
    assert r["recall_exact"] < 0.2 and r["recall_300m"] > 0.8 and r["component_recall"] == 1.0
    assert r["selected_within_300m"] > 0.8 and 0 < r["localization_index"] <= 1


def test_unique_components_vs_reference():
    f = np.zeros((30, 30), bool)
    f[5, 2:12] = True
    f[20, 15:25] = True
    lab, n = M.fault_components(f)
    feat = np.zeros((30, 30), np.float32)
    feat[20, 15:25] = 1
    ref = np.zeros((30, 30), np.float32)
    ref[5, 2:12] = 1
    u = M.unique_components(feat, ref, f, np.ones_like(f), 10 / 900, lab, n, radius_px=1)
    assert u["feature_only"] == 1 and u["reference_only"] == 1 and u["both"] == 0


def test_regional_blocks_support_and_maps():
    x, f, v = _line_case()
    bid, nbr, nbc = M.block_index(x.shape, 20, 20)
    T = M.regional(x, f, v, bid, 0.05, min_pos=5, min_neg=50)
    ok = T[T.status == "OK"]
    assert set(ok.block) == {2, 3} and (T.status == M.INSUFFICIENT).sum() == 2
    Z = M.block_map(T, "auroc_global_direction", nbr, nbc)
    assert np.isnan(Z[0]).all() and np.isfinite(Z[1]).all()


def test_distance_response_and_class():
    f = np.zeros((140, 20), bool)                                      # rows up to 11 km from the trace
    f[30, :] = True
    d = (np.abs(np.arange(140) - 30)[:, None] * np.ones((1, 20)) * 0.1).astype(np.float32)
    noise = np.random.default_rng(3).uniform(0, 0.05, d.shape).astype(np.float32)
    x = np.exp(-d / 0.1).astype(np.float32) + noise                    # peaks on the trace
    T = M.distance_response(x, d, np.ones_like(f), f)
    assert T.band.iloc[0] == M.DIST_LABELS[0] and T["median"].iloc[0] > 0.95 and T.n.iloc[-1] > 0
    assert M.response_class(T)["response_class"].startswith("LOCALIZES")
    y = np.exp(-np.abs(d - 1.0) / 0.2).astype(np.float32) + noise     # peaks 1 km away
    assert M.response_class(M.distance_response(y, d, np.ones_like(f), f))["response_class"].startswith("BROAD")


def test_matched_controls_sign():
    x, f, v = _line_case()
    cov = np.random.default_rng(1).normal(size=x.shape).astype(np.float32)
    r = M.matched_controls(x, f, cov, v, tile_px=20, min_dist_px=4)
    assert r["matched_delta"] > 0.8 and r["n_strata"] >= 1


def test_fold_masks_and_frozen_direction():
    roles = np.zeros((20, 40), np.int8)
    roles[:, 20:] = 1
    roles[:, 18:22] = 2
    dev, ev = M.fold_masks(roles, np.ones(roles.shape, bool))
    assert not (dev & ev).any() and not dev[:, 18:22].any() and not ev[:, 18:22].any()
    x = np.random.default_rng(2).normal(size=roles.shape).astype(np.float32)
    f = np.zeros(roles.shape, bool)
    f[::3, :] = True
    x[:, :18][f[:, :18]] += 3                                         # dev: high values at faults
    x[:, 22:][f[:, 22:]] -= 3                                         # eval: reversed
    r = M.heldout_feature(x, f, dev, ev)
    assert r["direction"] == "high" and r["eval_auroc"] < 0.5         # never flipped by evaluation labels


@pytest.mark.parametrize("row,cls", [
    ({"numerically_reliable": False}, "REJECT"),
    ({"numerically_reliable": True, "heldout_folds": 3, "heldout_folds_ci_above_half": 0,
      "regional_same_polarity_share": 0.3}, "REJECT"),
    ({"numerically_reliable": True, "max_abs_spearman_vs_earlier": 0.97, "most_similar": "rtp",
      "unique_component_share_vs_best_reference": 0.0}, "REDUNDANT"),
    ({"numerically_reliable": True, "heldout_folds": 3, "heldout_folds_ci_above_half": 2,
      "heldout_direction_consistent": True, "regional_same_polarity_share": 0.8,
      "unique_component_share_vs_best_reference": 0.05}, "PROMOTE"),
    ({"numerically_reliable": True}, "EXPERIMENTAL"),
])
def test_classification_rules(row, cls):
    assert M.classify(row, M.DEFAULT_THRESHOLDS)[0] == cls


# ------------------------------------------------------------------ catalog
def test_catalog_compiles_with_the_19_bands():
    from lab import featurelang as FL
    from lab.rasters import FALLBACK_BANDS

    prog = FL.compile_program(CAT.CATALOG, FL.Env(set(FALLBACK_BANDS)))
    names = prog.names
    for k in ("neg_rtp", "R20_basement_terrain_contrast", "R25_magnetic_edge", "R29_magnetic_multiscale",
              "R33_magnetic_tilt_abs", "R42_terrain_ridge"):
        assert k in names
    assert prog.statement("R20_basement_terrain_contrast").aoi_dependent          # robust_z without frozen stats
    assert not prog.statement("R25_magnetic_edge").aoi_dependent
    assert all(r in CAT.CATALOG for r in CAT.REFERENCES)


def test_formulas_cannot_read_labels():
    from lab import featurelang as FL

    with pytest.raises(FL.FeatureError):
        FL.compile_program("x = faults * 2", FL.Env({"rtp"}, {"faults": "target"}))


# ------------------------------------------------------------------ end to end on the synthetic demo raster
@pytest.fixture(scope="module")
def demo_root(tmp_path_factory):
    import rasterio
    from rasterio.transform import from_origin

    from demo.create_demo import generate_demo

    root = tmp_path_factory.mktemp("gems2")
    feats, _ = generate_demo(root)
    with rasterio.open(feats) as s:
        H, W = s.height, s.width
    fold = np.zeros((4, H, W), np.float32)
    strips = np.array_split(np.arange(W), 3)
    for k, cols in enumerate(strips):
        fold[0][:, cols] = k
        role = np.zeros((H, W), np.float32)
        role[:, cols] = 1
        lo, hi = cols[0], cols[-1]
        role[:, max(lo - 5, 0):lo] = 2
        role[:, hi + 1:hi + 6] = 2
        fold[1 + k] = role
    p = root / "outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026/fold_map.tif"
    p.parent.mkdir(parents=True)
    with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=4, dtype="float32", crs="EPSG:32611",
                       transform=from_origin(243350, 4508550, 100, 100)) as d:
        d.write(fold)
    return root


def test_end_to_end_full_mode_on_demo(demo_root):
    from gems_eda.feature_discovery import FeatureDiscoveryExperiment

    cfg = {"gems_root": str(demo_root), "mode": "FULL", "n_boot": 20, "complementarity_top_k": 4,
           "tiles_km": [5, 10], "min_pos": 10, "min_neg": 50}
    E = FeatureDiscoveryExperiment(cfg)
    rep = E.load()
    assert rep["checks"]["epsg_32611"] and rep["checks"]["res_100m"] and not rep["checks"]["shape"]  # demo grid
    raw = E.global_association(E.E.inputs, section="raw_fault_association")
    assert {"auroc_high", "auroc_low"} <= set(raw.columns)
    E.register_formulas("my_mag_ratio = ratio(gradmag(rtp, sigma=3), local_std(rtp, 5), 0.05)")
    done = E.compute_features()
    assert "my_mag_ratio" in done and E.compute_features() == []      # second call: all reused by hash
    G = E.global_association()
    sel = ["neg_rtp", "R25_magnetic_edge", "my_mag_ratio", "grav_broad", "R20_basement_terrain_contrast"]
    E.fault_distance_responses(sel)
    E.regional_association(sel, G)
    E.fault_recovery(sel)
    E.unique_recovery(sel)
    H = E.heldout_validation(sel)
    assert set(H.fold) <= {0, 1, 2}
    R = E.rank_and_classify()
    assert set(R["class"]) <= {"PROMOTE", "EXPERIMENTAL", "REDUNDANT", "REJECT"}
    md = E.synthesis()
    assert "## 7." in md and (E.store.root / "reports" / "FEATURE_DISCOVERY_SYNTHESIS.md").exists()
    for name in ("global_fault_association", "regional_fault_association", "feature_polarity_comparison",
                 "feature_scale_comparison", "fault_distance_responses", "fault_recovery_by_budget",
                 "unique_fault_recovery", "feature_complementarity", "feature_recommendations"):
        assert (E.store.root / "tables" / f"{name}.csv").exists(), name
    for figname in ("auroc_ap_polarity", "recall_vs_budget", "fault_distance_response", "region_feature_heatmap",
                    "complementarity_matrix"):
        assert (E.store.root / "figures" / f"{figname}.png").exists() and (E.store.root / "figures" /
                                                                          f"{figname}.svg").exists(), figname
    assert (E.store.root / "formulas" / "frozen_catalog.py").exists()          # robust_z statistics frozen
    E2 = FeatureDiscoveryExperiment(cfg)
    E2.custom = E.custom
    E2.load()
    assert E2._reuse("05_global", ["tables/global_fault_association.csv"], False)
    tifs = E.export_features(["R25_magnetic_edge"])
    import rasterio

    with rasterio.open(tifs[0]) as d, rasterio.open(E.p("features")) as s:
        assert d.crs == s.crs and d.transform == s.transform and (d.height, d.width) == (s.height, s.width)
    names = E.publish_to_layer_lab(["R28_normalized_magnetic_edge"])
    assert "R28_normalized_magnetic_edge" in names
    assert isinstance(pd.read_csv(E.store.root / "tables" / "feature_recommendations.csv"), pd.DataFrame)
