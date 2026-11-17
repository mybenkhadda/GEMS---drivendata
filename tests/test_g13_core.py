"""G13 regression suite: raster, features, splits, models, metric, submission (synthetic fixtures only).

Nothing here touches the competition data; real-data milestones are checked by the notebook's smoke cells.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage

from gems_bench.g13 import contracts as C
from gems_bench.g13 import signal as S
from gems_bench.g13.config import G13Config
from gems_bench.g13.synthetic import make, small_config


# =============================================================================== fixtures
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("g13data")
    paths = make(root)
    return root, paths


@pytest.fixture(scope="module")
def data(synth, tmp_path_factory):
    _, paths = synth
    return C.load(paths, tmp_path_factory.mktemp("g13cache"))


@pytest.fixture(scope="module")
def splits(data):
    from gems_bench.g13.splits import build_splits

    cfg = small_config("/tmp/unused", "/tmp/unused")
    return build_splits(data, cfg)


def _raster(path, a, crs="EPSG:32611", res=100.0, desc=()):
    a = np.asarray(a, np.float32)
    a = a[None] if a.ndim == 2 else a
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=a.shape[2],
        height=a.shape[1],
        count=a.shape[0],
        dtype="float32",
        crs=crs,
        transform=from_origin(0, 1000 * res, res, res),
        nodata=np.nan,
    ) as d:
        d.write(a)
        for i, n in enumerate(desc, 1):
            d.set_band_description(i, n)
    return path


# =============================================================================== raster contracts
def test_grid_checks_reject_mismatch(tmp_path):
    a = _raster(tmp_path / "a.tif", np.zeros((10, 10)))
    b = _raster(tmp_path / "b.tif", np.zeros((10, 11)))
    c = _raster(tmp_path / "c.tif", np.zeros((10, 10)), crs="EPSG:4326")
    d = _raster(tmp_path / "d.tif", np.zeros((10, 10)), res=30.0)
    with rasterio.open(a) as A, rasterio.open(b) as B:
        with pytest.raises(ValueError, match="shape"):
            C.check_grid(A, B)
    with rasterio.open(c) as X:
        with pytest.raises(ValueError, match="EPSG"):
            C.check_grid(X)
    with rasterio.open(d) as X:
        with pytest.raises(ValueError, match="100"):
            C.check_grid(X)


def test_discovery_by_content_and_ambiguity(synth, tmp_path):
    root, paths = synth
    found = C.discover_inputs(root)
    assert found["raw"] == paths["raw"] and found["known"] == paths["known"]
    # a file with the official name but the wrong content is rejected
    bad = tmp_path / "x"
    bad.mkdir()
    _raster(bad / C.OFFICIAL_NAMES["raw"], np.zeros((3, 10, 10)), desc=("a", "b", "c"))
    with pytest.raises(FileNotFoundError):
        C.discover_inputs(bad)
    with pytest.raises(ValueError):
        C.discover_inputs(root, {"raw": str(bad / C.OFFICIAL_NAMES["raw"])})


def test_load_masks_nodata_and_descriptions(data):
    assert data.shape == (400, 400) and data.names[1] == "rtp"
    assert not data.template_valid[:4].any() and not data.valid[:, :3].any()
    assert data.labels.sum() > 500 and not (data.labels & ~data.valid).any()
    cond = data.band("cond_surf")
    assert np.isnan(cond[300:340, 20:120]).all()
    assert data.coverage["cond_surf"] < data.coverage["rtp"]
    b = data.band("rtp")
    b[0, 0] = 1.0  # band() returns a writable copy, never the cache
    assert np.isnan(data.band("rtp")[0, 0])


def test_inventory_tables(data, synth, tmp_path):
    from gems_bench.g13.cache import Workspace

    ws = Workspace(tmp_path / "w")
    res = C.run_inventory(data, synth[1], ws)
    inv = res["band_inventory"]
    assert len(inv) == 19 and set(inv.columns) >= {"name", "modality", "finite_frac_in_template", "p50", "meaning"}
    assert inv.set_index("name").loc["cond_surf", "missing_px_in_template"] > 0
    assert res["summary"]["fault_px"] == int(data.labels.sum())


# =============================================================================== signal / features
def test_gaussian_gradient_on_plane_and_orientation():
    yy, xx = np.indices((64, 64)).astype(np.float32)
    plane = 2.0 * xx - 3.0 * (-yy)  # f = 2 east + (-3) north  (north = -row)
    gx, gy = S.gaussian_gradient(plane, 2.0)
    assert np.allclose(gx[20:44, 20:44], 2.0, atol=1e-3)
    assert np.allclose(gy[20:44, 20:44], -3.0, atol=1e-3)
    # a step across a north-south line -> gradient east -> strike north-south (pi/2)
    edge = np.tanh((xx - 32) / 2)
    coh, th, _ = S.structure_tensor(edge, 1.5, 3.0)
    assert coh[32, 32] > 0.9 and abs(th[32, 32] - np.pi / 2) < 0.05
    # rotate 90 deg: E-W line -> strike 0 (axial: 0 == pi)
    coh, th, _ = S.structure_tensor(edge.T.copy(), 1.5, 3.0)
    assert min(th[32, 32], np.pi - th[32, 32]) < 0.05


def test_axial_invariance():
    a = np.array([0.1, 1.0, 3.0])
    assert np.allclose(S.axial_diff(a, a + np.pi), 0, atol=1e-6)
    c1, s1 = S.axial_encode(a)
    c2, s2 = S.axial_encode(a + np.pi)
    assert np.allclose(c1, c2, atol=1e-6) and np.allclose(s1, s2, atol=1e-6)
    assert abs(S.axial_diff(np.array([0.0]), np.array([np.pi / 2]))[0] - np.pi / 2) < 1e-6


def test_spectral_derivatives_signs_and_contact_edges():
    yy, xx = np.indices((128, 128)).astype(np.float32)
    v = np.ones((128, 128), bool)
    sf = S.SpectralField(3.0 * xx, v, 100.0)  # 3 units / px east = 0.03 / m
    assert np.allclose(sf.dx()[40:88, 40:88], 0.03, atol=2e-3)
    sf = S.SpectralField(-2.0 * yy, v, 100.0)  # increases northwards (row up) by 2 / px
    assert np.allclose(sf.dy()[40:88, 40:88], 0.02, atol=2e-3)
    contact = np.tanh((xx - 64) / 3).astype(np.float32)
    pf = S.potential_field_products(contact, v, heights_m=(500, 2000))
    row = pf["thd"][64]
    assert abs(int(np.argmax(row[20:108])) + 20 - 64) <= 1  # THD maximum over the contact
    assert np.isfinite(pf["nss"]).all() and (pf["nss"] >= 0).all()
    # upward continuation smooths: lower THD peak at larger height
    assert pf["uc2000_thd"][64].max() < pf["uc500_thd"][64].max()


def test_nodata_does_not_leak_and_fill_keeps_valid():
    x = np.ones((80, 80), np.float32)
    v = np.ones((80, 80), bool)
    v[:, 38:42] = False
    x[~v] = 1e9
    f = S.fill_nodata(x, v)
    assert np.allclose(f, 1.0, atol=1e-4)
    pf = S.potential_field_products(np.where(v, x, np.nan), v, heights_m=(500,))
    assert np.isnan(pf["thd"][:, 38:42]).all()
    assert np.nanmax(np.abs(pf["vd"])) < 1e-3  # a constant field has no edges anywhere
    assert np.allclose(S.masked_smooth(x, v, 3)[v], 1.0, atol=1e-4)


def test_windowed_apply_matches_full_array():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 260)).astype(np.float32)
    f = lambda a: ndimage.gaussian_filter(a, 3.0, truncate=3.0)  # noqa: E731  support 9 px
    full = f(x)
    tiled = S.windowed_apply(f, x, block=64, halo=16)
    assert np.allclose(full[20:-20, 20:-20], tiled[20:-20, 20:-20], atol=1e-5)


def test_monogenic_and_gabor_shapes(data):
    x = data.band("rtp")
    m = S.monogenic(x, data.valid & np.isfinite(x), (400, 1600))
    assert set(m) >= {"phase_congruency", "mono400_amp"}
    pc = m["phase_congruency"][data.valid]
    assert np.nanmin(pc) >= 0 and np.nanmax(pc) <= 1
    g = S.gabor_bank(x, data.valid & np.isfinite(x), (800,), 4)
    assert np.nanmax(g["gabor800_dircoh"]) <= 1.0 + 1e-5


def test_shifted_correlation_recovers_lag():
    from gems_bench.g13.features import shifted_edge_correlation

    yy, xx = np.indices((100, 100)).astype(np.float32)
    a = np.exp(-((xx - 50) ** 2) / 4)
    b = np.exp(-((xx - 53) ** 2) / 4)  # 3 px east of a
    th = np.full(a.shape, np.pi / 2, np.float32)  # N-S strike -> normal points east
    lag, corr = shifted_edge_correlation(a, b, th, np.ones(a.shape, bool))
    # lag is measured along n = (-sin theta, cos theta) = west for a N-S strike: b 3 px east -> -0.3 km
    assert abs(lag[50, 50] + 0.3) < 0.11 and corr[50, 50] > 0.9


def test_features_label_free_schema_and_cache(data, tmp_path):
    from gems_bench.g13.features import build_store

    cfg = small_config(tmp_path, tmp_path, "FEATURE_DISCOVERY", families=("magnetic", "cross_modal", "validity"))
    s1 = build_store(data, cfg, tmp_path / "f1", log=lambda *a: None)
    names = s1.names(include_raw=False)
    assert any(n.startswith("cross_modal:magn_grav") for n in names)
    # labels are not an input: flipping every label leaves every feature identical
    import copy

    d2 = copy.copy(data)
    d2.labels = ~data.labels & data.valid
    s2 = build_store(d2, cfg, tmp_path / "f2", log=lambda *a: None)
    assert names == s2.names(include_raw=False)
    for n in names[:15]:
        assert np.array_equal(s1.get(n), s2.get(n), equal_nan=True)
    # cache: rebuilding reuses; a changed scale set invalidates
    msgs = []
    build_store(data, cfg, tmp_path / "f1", log=msgs.append)
    assert all("reused" in m for m in msgs)
    cfg2 = small_config(tmp_path, tmp_path, "FEATURE_DISCOVERY", families=("magnetic",), scales_m=(200, 800))
    msgs = []
    build_store(data, cfg2, tmp_path / "f1", log=msgs.append)
    assert "reused" not in msgs[0]
    # physical() undoes the robust scaling
    n = "magnetic:rtp:coh_s800"
    a = s1.physical(n)
    assert np.nanmin(a[data.valid]) >= -1e-2 and np.nanmax(a[data.valid]) <= 1.01


def test_common_source_pairs_flagged(data, tmp_path):
    from gems_bench.g13.features import build_store

    cfg = small_config(tmp_path, tmp_path, "FEATURE_DISCOVERY", families=("cross_modal",))
    s = build_store(data, cfg, tmp_path / "f", log=lambda *a: None)
    reg = s.registry.set_index("name")
    assert bool(reg.loc["cross_modal:grav_dept:axial_alignment", "common_source"])
    assert not bool(reg.loc["cross_modal:magn_grav:axial_alignment", "common_source"])


# =============================================================================== splits
def test_splits_disjoint_embargo_and_fault_exclusion(splits):
    from gems_bench.g13.splits import assert_valid, audit

    sp = splits
    assert_valid(sp)
    a = audit(sp)
    cv = a[a.partition.str.startswith("cv")]
    assert (cv.train_val_overlap == 0).all() and (cv.train_test_overlap == 0).all()
    assert (cv.min_train_to_val_km > 0).all()
    disc = a[a.partition.str.startswith("disc")]
    assert (disc.supervised_withheld_px == 0).all() and (disc.withheld_in_test_px == 0).all()
    assert (disc.withheld_px > 0).all()
    for k in sp.cv_folds:
        assert (sp.train(k) & sp.labels).sum() > 20


def test_splits_deterministic(data):
    from gems_bench.g13.splits import build_splits

    cfg = small_config("/tmp/u", "/tmp/u")
    a, b = build_splits(data, cfg), build_splits(data, cfg)
    assert np.array_equal(a.fold_map, b.fold_map) and np.array_equal(a.disc_roles, b.disc_roles)
    assert a.test_fold == b.test_fold


def test_units_use_official_masking(splits):
    us = splits.units(("geo", "disc_official"))
    disc = [u for u in us if u.task == "disc_official"]
    assert disc and all(u.exclude is not None and not u.supports_tp for u in disc)
    for u in disc:
        assert not (u.truth & u.exclude).any()  # withheld truth never inside the known mask


# =============================================================================== models / sampling
def test_sampling_strategies(splits):
    from gems_bench.g13.models import distance_px, draw_sample

    sp = splits
    tr = sp.train(sp.cv_folds[0])
    d = distance_px(sp.labels)
    for s in ("standard", "distance_stratified", "soft_negative"):
        smp = draw_sample(tr, sp.labels, np.random.default_rng(0), strategy=s, max_positives=500)
        flat = tr.ravel()
        assert flat[smp.ids].all()
        dd = d.ravel()[smp.ids]
        assert (dd[smp.y == 1] <= 1).all() and (dd[smp.y == 0] > 3).all()  # ignore ring excluded
        assert (smp.w > 0).all() and (smp.w <= 1).all()
    smp = draw_sample(tr, sp.labels, np.random.default_rng(0), strategy="soft_negative", max_positives=500)
    near = d.ravel()[smp.ids] < 10
    assert smp.w[(smp.y == 0) & near].mean() < smp.w[(smp.y == 0) & ~near].mean()
    ign = np.zeros_like(tr)
    ign[:, :200] = True
    smp = draw_sample(tr, sp.labels, np.random.default_rng(0), ignore=ign, max_positives=500)
    assert not ign.ravel()[smp.ids].any()


def test_model_determinism_reload_and_chunked_parity(data, splits, tmp_path):
    from gems_bench.g13.features import FeatureStore
    from gems_bench.g13.models import draw_sample, fit, load_model, predict_mask, predict_proba, save_model

    st = FeatureStore(tmp_path / "fs", data)
    names = st.names(include_raw=True)
    sp = splits
    cfg = small_config(tmp_path, tmp_path)
    smp = draw_sample(sp.train(sp.cv_folds[0]), sp.labels, np.random.default_rng(1), max_positives=800)
    X = st.matrix(smp.ids, names)
    assert np.isnan(X).any()  # missing features stay NaN (native missing handling)
    for kind in ("xgboost", "lightgbm"):
        m1, m2 = fit(kind, cfg, X, smp.y, smp.w, 3), fit(kind, cfg, X, smp.y, smp.w, 3)
        p1, p2 = predict_proba(m1, X), predict_proba(m2, X)
        assert np.array_equal(p1, p2)
        assert (p1 >= 0).all() and (p1 <= 1).all()
        save_model(m1, kind, tmp_path / f"m_{kind}", names, {})
        m3, meta = load_model(tmp_path / f"m_{kind}")
        assert meta["features"] == names
        assert np.allclose(predict_proba(m3, X), p1, atol=1e-6)
        mask = data.valid & (np.indices(data.shape)[0] < 120)
        a = predict_mask(m1, st, names, mask, chunk_rows=17)
        b = predict_mask(m1, st, names, mask, chunk_rows=128)
        assert np.array_equal(a, b, equal_nan=True) and np.isnan(a[~mask]).all()
        ids = np.flatnonzero(mask.ravel())[:500]
        assert np.allclose(a.ravel()[ids], predict_proba(m1, st.matrix(ids, names)), atol=1e-6)


# =============================================================================== metric / evaluation
def _line(sh=(60, 60), r=30):
    a = np.zeros(sh, bool)
    a[r, 10:50] = True
    return a


def test_metric_cases():
    from gems_bench.metric import dti_components

    t = _line()
    assert dti_components(t.astype(float), t).dti > 0.999  # perfect
    assert dti_components(np.zeros(t.shape), t).dti == 0  # empty
    c1 = dti_components(np.roll(t, 1, 0).astype(float), t)  # shifted 100 m: partial credit
    c3 = dti_components(np.roll(t, 3, 0).astype(float), t)  # 300 m: no credit
    assert 0.3 < c1.dti < 0.9 and c3.tp == 0
    thick = ndimage.binary_dilation(t, iterations=2).astype(float)
    assert dti_components(thick, t).dti < 1 and dti_components(thick, t).fp > 0
    fp = (t | _line(r=5)).astype(float)
    assert dti_components(fp, t).dti < 1
    half = t.copy()
    half[:, 30:] = False
    # 20 truth pixels hit exactly + kernel support 2/3 and 1/3 for the next two (max-based TP)
    assert dti_components(half.astype(float), t).tp == pytest.approx(20 + 2 / 3 + 1 / 3)
    ev = np.ones(t.shape, bool)
    ev[:, :5] = False
    out = np.zeros(t.shape)
    out[:, :5] = 1
    assert dti_components(out, t, eval_mask=ev).fp == 0  # prediction outside eval region
    k = np.zeros(t.shape, bool)
    k[10, 10:50] = True
    p = (t | k).astype(float)  # known pixels masked pixel-exactly
    c = dti_components(p, t, exclude_mask=k, exclude_supports_tp=False)
    assert c.dti > 0.999
    half_p = t * 0.5
    assert dti_components(half_p, t).tp == pytest.approx(0.5 * t.sum())  # continuous scaling


def test_crossfit_and_pooled_vs_zone():
    import pandas as pd

    from gems_bench.g13.evaluation import crossfit, pooled_dti

    rows = []
    for z in range(3):
        for p, (tp, fp, fn) in {"a": (10, 5, 10), "b": (12, 20, 8)}.items():
            rows.append({"zone": z, "k": p, "tp_w": tp + z, "fp_w": fp, "fn_w": fn})
    t = pd.DataFrame(rows)
    per, summ = crossfit(t, ["k"])
    assert len(per) == 3 and summ["optimism"] >= -1e-12
    # pooled DTI over zones != mean of zone DTIs (pooling is the official aggregation)
    zone_mean = per.dti_heldout.mean()
    assert abs(pooled_dti(per) - zone_mean) >= 0


def test_block_components_sum_to_unit(data, splits):
    from gems_bench.g13.evaluation import block_components, bootstrap_dti, paired_bootstrap

    u = splits.units(("disc_official",))[0]
    rng = np.random.default_rng(0)
    score = rng.random(data.shape).astype(np.float32) * 0.2 + data.labels * 0.7
    bl = block_components(score, u, None, block_px=50)
    c = u.components(u.crop(score))
    assert bl.tp_w.sum() == pytest.approx(c.tp, rel=1e-6) and bl.fp_w.sum() == pytest.approx(c.fp, rel=1e-6)
    b = bootstrap_dti(bl, 50)
    assert b["ci_lo"] <= b["dti"] <= b["ci_hi"]
    pb = paired_bootstrap(bl, bl, 50)
    assert pb["delta"] == pytest.approx(0) and pb["ci_lo"] == pytest.approx(0)


def test_synthetic_sensitivity_shapes():
    from gems_bench.g13.dti_opt import synthetic_sensitivity

    s = synthetic_sensitivity()
    w = s[s.experiment == "width_px"].dti.to_numpy()
    assert w[0] > 0.99 and np.all(np.diff(w) < 0)  # wider ribbons always cost here
    d = s[s.experiment == "displacement_px"].dti.to_numpy()
    assert d[0] > d[1] > d[2] and d[3] == pytest.approx(0, abs=1e-9)
    r = s[s.experiment == "recall"].dti.to_numpy()
    assert np.all(np.diff(r) > 0)


def test_representation_render_respects_known_and_budget(splits, data):
    from gems_bench.g13.dti_opt import render

    u = [x for x in splits.units(("disc_official",))][0]
    s = np.random.default_rng(0).random(u.ev.shape).astype(np.float32)
    spec = {"rep": {"select": "top"}, "frac": 0.02, "ring_px": 1}
    q = render(spec, s, u.ev, u.known, None, u.scored_mask())
    assert not (q[u.known] > 0).any()
    from gems_bench.gems3.representation import _disk

    ring = ndimage.binary_dilation(u.known, structure=_disk(1)) & ~u.known
    assert not (q[ring] > 0).any()
    assert (q > 0).sum() <= np.ceil(0.02 * (u.scored_mask() & ~ring).sum()) + 1


# =============================================================================== submission
def test_submission_write_validate_and_reject(data, tmp_path):
    from gems_bench.g13.cache import Workspace
    from gems_bench.g13.submission import export_candidate, validate, write

    ws = Workspace(tmp_path / "w")
    q = np.random.default_rng(0).random(data.shape).astype(np.float32)
    rec = export_candidate("cand", q, data, ws, {"model": "m"}, extra_names=("predicted_faults.tif",))
    v = rec["validation"]
    assert v["valid"], v["checks_failed"]
    with rasterio.open(v["path"]) as src:
        a = src.read(1)
        assert src.count == 1 and src.dtypes[0] == "float32" and src.crs.to_epsg() == 32611
        with rasterio.open(data.template_path) as t:
            assert src.transform == t.transform and (src.height, src.width) == (t.height, t.width)
    assert np.isnan(a[~data.template_valid]).all() and np.isfinite(a[data.template_valid]).all()
    assert (a[data.labels] == 0).all() and np.nanmax(a) <= 1 and np.nanmin(a) >= 0
    assert len(v["sha256"]) == 64 and (tmp_path / "w" / "submissions" / "predicted_faults.tif").exists()
    json.loads((tmp_path / "w" / "submissions" / "cand.json").read_text())
    # degenerate prediction rejected
    p = write(np.zeros(data.shape), data, tmp_path / "zero.tif")
    assert "degenerate" in " ".join(validate(p, data)["checks_failed"])
    # NaN injected inside the template is detected
    with rasterio.open(v["path"]) as src:
        prof, arr = src.profile, src.read(1)
    arr[200, 200] = np.nan
    with rasterio.open(tmp_path / "bad.tif", "w", **prof) as dst:
        dst.write(arr, 1)
    assert "NaN inside" in " ".join(validate(tmp_path / "bad.tif", data)["checks_failed"])


def test_config_validation():
    with pytest.raises(KeyError, match="did you mean"):
        G13Config.from_dict({"n_estimator": 10})
    with pytest.raises(ValueError):
        G13Config.from_mode("NOPE")
    c = G13Config.from_mode("FAST_BASELINE")
    assert c.active_stages()[0] == "inventory" and "submission" in c.active_stages()
    assert c.fingerprint("seed") != G13Config.from_mode("FAST_BASELINE", seed=1).fingerprint("seed")
