"""G07 models: directional augmentation consistency, proximity regressor, U-Net head, candidate features, utility."""
import numpy as np
import pandas as pd
import pytest

from gems_bench.g06.supervision import build_orientation_targets
from gems_bench.g07 import models_candidates as MC
from gems_bench.g07 import models_surface as MS
from gems_bench.g07.contracts import LeakageError


def line(angle_deg, n=48):
    """A straight line through the centre at ``angle_deg`` (from east, counter-clockwise, north up)."""
    m = np.zeros((n, n), bool)
    t = np.linspace(-18, 18, 200)
    a = np.radians(angle_deg)
    rr, cc = np.round(n / 2 - t * np.sin(a)).astype(int), np.round(n / 2 + t * np.cos(a)).astype(int)
    m[rr, cc] = True
    return m


@pytest.mark.parametrize("k,mirror", [(1, False), (2, False), (3, True), (0, True)])
def test_augmentation_keeps_orientation_consistent(k, mirror):
    y = line(30)
    o = build_orientation_targets(y, 2.0, 0.3)
    O = np.stack([np.nan_to_num(o["orient_cos2"]), np.nan_to_num(o["orient_sin2"]), np.isfinite(o["orient_cos2"])])
    x = np.zeros((1, *y.shape), np.float32)
    _, ya, _, oa = MS.augment(x, y.astype(np.float32), np.ones(y.shape, np.float32), O.astype(np.float32),
                              np.random.default_rng(0), k=k, mirror=mirror)
    ref = build_orientation_targets(ya > 0.5, 2.0, 0.3)                      # recompute on the transformed label
    m = (oa[2] > 0) & np.isfinite(ref["orient_cos2"])
    assert m.sum() > 10
    np.testing.assert_allclose(oa[0][m], ref["orient_cos2"][m], atol=0.15)
    np.testing.assert_allclose(oa[1][m], ref["orient_sin2"][m], atol=0.15)


def test_proximity_regressor_bounds():
    pytest.importorskip("lightgbm")
    from gems_bench.g07.models_subsurface import train_fault_distance_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3)).astype(np.float32)
    t = np.clip(np.exp(-np.abs(X[:, 0])), 0, 1).astype(np.float32)
    inner = (np.arange(600) % 3).astype(np.int8)
    m, _, info = train_fault_distance_model(X, t, np.ones(600, np.float32), {"n_estimators": 40, "num_leaves": 7,
                                                                             "early_stopping_rounds": 5}, 0, inner)
    p = m.predict_proba(X)
    assert p.shape == (600, 2) and (p >= 0).all() and (p <= 1).all() and info.get("best_iteration")


def test_unet_has_centreline_and_orientation_heads():
    torch = pytest.importorskip("torch")
    net = MS.build_surface_unet(4, base=4, depth=2)
    out = net(torch.zeros(2, 5, 32, 32))
    assert tuple(out.shape) == (2, 3, 32, 32)


def test_geology_only_candidates_refuse_w2():
    T = pd.DataFrame({"geo_mean_dem_scarp_s10": [1.0], "ev_w2_rank_mean": [0.5], "ev_subsurface_rank_mean": [0.2],
                      "on_w2": [1.0], "seg_length_km": [1.0], "target": [1], "system_id": [3]})
    g = MC.candidate_feature_columns(T, w2_aware=False)
    assert "ev_w2_rank_mean" not in g and "on_w2" not in g and "ev_subsurface_rank_mean" in g
    assert "target" not in g and "system_id" not in g
    assert "ev_w2_rank_mean" in MC.candidate_feature_columns(T, w2_aware=True)
    with pytest.raises(LeakageError):
        MC.candidate_feature_columns(T, w2_aware=False, families=["model_evidence"])


def test_utility_rewards_new_relevant_area_only():
    lab = np.zeros((20, 20), np.int32)
    lab[2, 2:12] = 1                                                        # new area
    lab[10, 2:12] = 2                                                       # overlaps the base
    base = np.zeros((20, 20), bool)
    base[10, :] = True
    S = pd.DataFrame({"label": [1, 2], "p": [0.9, 0.9]})
    U = MC.estimate_candidate_utility(lab, S, base, np.ones((20, 20), bool), 0.25, 1.0).set_index("label")
    assert U.loc[1, "utility"] > 0 and U.loc[2, "n_new"] == 0
    S2 = pd.DataFrame({"label": [1, 2], "p": [0.1, 0.1]})
    assert MC.estimate_candidate_utility(lab, S2, base, np.ones((20, 20), bool), 0.25).set_index("label").loc[
        1, "utility"] < 0
