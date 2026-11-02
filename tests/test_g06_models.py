"""G06 models: pixel fit / save / load parity, raster prediction bounds, segment cross-fit isolation, segment maps,
meta strictness (synthetic)."""
import numpy as np
import pandas as pd
import pytest

from gems_bench.g06 import models_meta as MM
from gems_bench.g06 import models_pixel as MP
from gems_bench.g06 import models_segment as MS
from gems_bench.g06.contracts import LeakageError, SchemaError


class FakeStore:
    schema_hash = "h"

    def __init__(self, arrays):
        self.arrays, self.names = arrays, list(arrays)

    def load(self, names, window=None):
        h, w = next(iter(self.arrays.values())).shape
        r0, r1, c0, c1 = window or (0, h, 0, w)
        return np.stack([self.arrays[n][r0:r1, c0:c1] for n in names]).astype(np.float32)


def data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n) > 0.8).astype(np.int8)
    X[rng.random(n) < 0.05, 2] = np.nan
    return X, y, np.ones(n, np.float32)


@pytest.mark.parametrize("kind", ["random_forest", "lightgbm"])
def test_fit_save_load_parity(tmp_path, kind):
    pytest.importorskip("lightgbm" if kind == "lightgbm" else "sklearn")
    X, y, w = data()
    params = {"n_estimators": 30, "max_depth": 4, "min_samples_leaf": 5, "n_jobs": 1} if kind == "random_forest" \
        else {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7, "early_stopping_rounds": 10, "n_jobs": 1}
    inner = (np.arange(len(y)) % 3).astype(np.int8)
    est, imp, info = MP.fit_pixel_model(kind, X, y, w, params, 0, inner, np.zeros(len(y), bool))
    m = MP.PixelModel(kind, est, ["a", "b", "c"], "h", imp)
    p = m.predict_proba(X)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    m.save(tmp_path / "m")
    m2 = MP.PixelModel.load(tmp_path / "m")
    np.testing.assert_allclose(m2.predict_proba(X), p, rtol=0, atol=1e-7)
    if kind == "random_forest":
        assert imp is not None and np.isfinite(imp).all()
    else:
        assert info.get("best_iteration") is not None


def test_predict_raster_bounds_and_schema(tmp_path):
    pytest.importorskip("lightgbm")
    X, y, w = data()
    est, imp, _ = MP.fit_pixel_model("lightgbm", X, y, w, {"n_estimators": 20, "num_leaves": 7, "n_jobs": 1}, 0)
    m = MP.PixelModel("lightgbm", est, ["a", "b", "c"], "h", imp)
    rng = np.random.default_rng(1)
    st = FakeStore({k: rng.normal(size=(40, 50)).astype(np.float32) for k in ("a", "b", "c")})
    mask = np.ones((20, 30), bool)
    mask[:5] = False
    p = MP.predict_pixel_raster(m, st, (10, 30, 10, 40), mask, tile_px=8)
    assert np.isnan(p[:5]).all() and np.isfinite(p[5:]).all()
    assert (p[5:] >= 0).all() and (p[5:] <= 1).all()
    direct = m.predict_proba(st.load(["a", "b", "c"], (10, 30, 10, 40))[:, mask].T)
    np.testing.assert_allclose(p[mask], direct, atol=1e-7)                   # tiled == direct
    bad = MP.PixelModel("lightgbm", est, ["a", "c", "b"], "h", imp)
    st2 = FakeStore({k: st.arrays[k] for k in ("a", "b")})
    with pytest.raises(SchemaError):
        MP.predict_pixel_raster(bad, st2, (0, 10, 0, 10), np.ones((10, 10), bool))
    st.schema_hash = "other"
    with pytest.raises(SchemaError):
        MP.predict_pixel_raster(m, st, (0, 10, 0, 10), np.ones((10, 10), bool))


def seg_table(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sc in ("V1",):
        for z in range(3):
            n = 120
            f = rng.normal(size=n)
            rows.append(pd.DataFrame({"unit": f"{sc}:{z}", "scenario": sc, "zone": z, "label": np.arange(1, n + 1),
                                      "geo_mean_x": f, "seg_length_km": 1.0, "label_conf": 1.0,
                                      "kn_known_dist_km_mean": rng.random(n), "ev_w2_rank_mean": rng.random(n),
                                      "target": (f + rng.normal(0, 0.3, n) > 0).astype(float),
                                      "r_full": z * 10 + 1, "c_full": 1, "near_share": 0.0, "cls": "x"}))
    return pd.concat(rows, ignore_index=True)


def zone_masks():
    m = {}
    for z in range(3):
        a = np.zeros((40, 5), bool)
        a[z * 10:(z + 1) * 10] = True
        m[f"V1:{z}"] = a
    return m


def test_segment_crossfit_ignores_heldout_truth():
    pytest.importorskip("lightgbm")
    T = seg_table()
    prm = {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 5}
    s1, info = MS.crossfit_segment_model("S1", {"track": "S1", "model": "lightgbm"}, T, zone_masks(), prm, 0,
                                         progress=None)
    T2 = T.copy()
    m = T2.unit == "V1:0"
    T2.loc[m, "target"] = 1 - T2.loc[m, "target"]                             # permute held-out truth
    s2, _ = MS.crossfit_segment_model("S1", {"track": "S1", "model": "lightgbm"}, T2, zone_masks(), prm, 0,
                                      progress=None)
    a, b = s1[s1.unit == "V1:0"].score.to_numpy(), s2[s2.unit == "V1:0"].score.to_numpy()
    np.testing.assert_allclose(a, b)                                          # held-out scores unchanged
    assert info["strictness"] == "SCREENING_ONLY"                             # no nested context supplied
    assert "ev_w2_rank_mean" not in info["features"]                          # S1 never sees W2 evidence


def test_segment_feature_tracks():
    T = seg_table()
    s1 = MS.segment_feature_columns(T, "S1")
    s2 = MS.segment_feature_columns(T, "S2")
    assert "ev_w2_rank_mean" not in s1 and "ev_w2_rank_mean" in s2
    assert "target" not in s2 and "near_share" not in s2


def test_segment_to_raster_preserves_pixel_order():
    lab = np.zeros((10, 10), np.int32)
    lab[2, 2:6] = 1
    pix = np.arange(100, dtype=np.float32).reshape(10, 10)
    ev = np.ones((10, 10), bool)
    out = MS.segment_to_raster(lab, pd.DataFrame({"label": [1], "score": [0.9]}), pix, ev, 0.5)
    outside = ~(lab > 0)
    o = out[outside]
    assert (np.diff(o) > 0).all()                                             # pixel ranking kept outside
    inside = out[lab > 0]
    assert (np.diff(inside) > 0).all()                                        # and inside the candidate
    with pytest.raises(ValueError):
        MS.segment_to_raster(lab, pd.DataFrame({"label": [1], "score": [0.9]}), None, ev)


def test_meta_strictness_and_rank_average():
    assert MM.strictness_of("rank_average", ["w2", "g06_pixel"], True) == "STRICT_LABEL_FREE"
    assert MM.strictness_of("residual_lgbm", ["w2", "g06_pixel"], True) == "SCREENING_ONLY"
    assert MM.strictness_of("moe", ["g06_pixel"], True) == "STRICT_NESTED"
    assert MM.strictness_of("moe", ["g06_pixel"], False) == "SCREENING_ONLY"
    ev = np.ones((4, 4), bool)
    R = {"w2": np.full((4, 4), 0.2, np.float32), "g06_pixel": np.full((4, 4), 0.6, np.float32)}
    s = MM.MetaModel("rank_average", ["w2", "g06_pixel"], weights={"w2": 0.5, "g06_pixel": 0.5}).score(R, ev)
    np.testing.assert_allclose(s, 0.4)
    with pytest.raises(LeakageError):
        MM.train_regularized_meta_model("moe", pd.DataFrame({"w2": [0.1], "target": [1], "y": [1], "w": [1.0]}),
                                        ["w2", "target"], {}, 0)
