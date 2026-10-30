"""Synthetic tests of the Feature Discovery Studio (lab.ops / featurelang / plugins / library / engine / association /
complementarity / tournament / transects / export_native / integration). Small arrays and the packaged synthetic
demo raster only. Authored without being executed."""
from __future__ import annotations

import json
import textwrap

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

from demo.create_demo import generate_demo
from lab import association as A
from lab import complementarity as CP
from lab import featurelang as FL
from lab import ops as O
from lab import plugins as PL
from lab import tournament as TO
from lab import transects as TR
from lab.engine import Engine
from lab.library import FeatureLibrary, LibraryError


# ------------------------------------------------------------------ helpers
def env(layers=("a", "b"), **kw):
    return FL.Env(set(layers), kw.get("forbidden", {}), kw.get("library", {}), kw.get("plugins", {}))


def run(src, arrays, **kw):
    e = env(tuple(arrays), **kw)
    prog = FL.compile_program(src, e)
    vals = dict(arrays)
    shape = next(iter(arrays.values())).shape
    for st in prog.statements:
        vals[st.name] = FL.run_statement(st, prog, lambda k: vals[k], e, shape)
    return prog, vals


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("gems")
    feats, faults = generate_demo(root)
    return root, str(feats), str(faults)


def engine(demo, tmp_path, extras=None):
    root, feats, faults = demo
    lib = FeatureLibrary(tmp_path / "registry.json")
    return Engine(feats, faults, extras or [], lib, plugin_dirs=[PL.PLUGIN_DIR])


# ------------------------------------------------------------------ parsing / safety
@pytest.mark.parametrize("bad", [
    "x = __import__('os').system('x')", "import os", "x = a.__class__", "x = a[0]", "x = [v for v in a]",
    "for i in range(3):\n    x = a", "x = (lambda q: q)(a)", "x = eval('a')", "x = open('f')", "x = unknown + 1",
    "a = b + 1", "x = sum(a)", "x = a ** b", "x = a ** 9", "x = gaussian(a, sigma=b)", "x = f'{a}'",
    "x = gaussian(*[a])", "x = gaussian(a, 999)", "x, y = a, b", "x = a\nx = b",
])
def test_rejects_unsafe_or_invalid(bad):
    with pytest.raises(FL.FeatureError):
        FL.compile_program(bad, env())


def test_target_and_prediction_are_not_formula_inputs():
    e = env(("a",), forbidden={"faults": "target", "w2": "prediction"})
    for src in ("x = faults * 2", "x = gaussian(w2, 2)"):
        with pytest.raises(FL.FeatureError, match="leakage"):
            FL.compile_program(src, e)


def test_named_chained_multilayer():
    a = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    b = np.full((3, 4), 2.0, np.float32)
    prog, v = run("r = ratio(a, b, eps=0.01)\ns = r * 2 + 1\nm = where(a > 5, s, 0)\nn = normdiff(a, b)", {"a": a,
                                                                                                          "b": b})
    np.testing.assert_allclose(v["s"], a / 2 * 2 + 1, rtol=1e-6)
    np.testing.assert_allclose(v["m"], np.where(a > 5, a + 1, 0), rtol=1e-6)
    assert prog.statement("s").locals == {"r"} and prog.raw_inputs == {"a", "b"}


def test_params_literal_and_clamping():
    src = 'PARAMS = {"sigma": {"min": 1, "max": 10, "step": 1, "default": 3}}\ng = gaussian(a, sigma=sigma)'
    p = FL.compile_program(src, env(("a",)), {"sigma": 50})
    assert p.values["sigma"] == 10 and p.statement("g").halo == int(np.ceil(4 * 10)) + 2
    with pytest.raises(FL.FeatureError):
        FL.compile_program('PARAMS = {"s": {"min": 1, "max": 2}}\nx = a', env(("a",)))
    with pytest.raises(FL.FeatureError):
        FL.compile_program('PARAMS = {"s": {"min": 1, "max": 2, "step": 1, "default": __import__("os")}}\nx = a',
                           env(("a",)))


def test_halo_accumulates_through_nesting():
    p = FL.compile_program("x = gaussian(gaussian(a, 2), 3)\ny = highpass(x, 1)", env(("a",)))
    assert p.statement("x").halo == (8 + 2) + (12 + 2) and p.statement("y").halo == p.statement("x").halo + 4 + 2


# ------------------------------------------------------------------ numerics / NaN / division
def test_invalid_math_gives_nan_not_values():
    a = np.array([[-1.0, 0.0, 4.0]], np.float32)
    _, v = run("l = log(a)\ns = sqrt(a)", {"a": a})
    assert np.isnan(v["l"][0, :2]).all() and np.isclose(v["l"][0, 2], np.log(4))
    assert np.isnan(v["s"][0, 0]) and v["s"][0, 2] == 2
    d = FL.diagnose(v["l"], np.isfinite(a))
    assert any("became NaN" in w for w in d["warnings"])


def test_division_near_zero_is_bounded_and_reported():
    a = np.ones((4, 4), np.float32)
    b = np.zeros((4, 4), np.float32)
    rec = O.set_recorder(O.Recorder())
    r = O.ratio(a, b, 0.01)
    s = O.safe_divide(a, b, 0.01)
    assert np.isfinite(r).all() and np.abs(r).max() <= 100 + 1e-3 and (s == 0).all()
    assert rec.division[0]["floored_share"] == 1.0
    _, v = run("q = a / b", {"a": a, "b": b})
    assert FL.diagnose(v["q"])["n_inf"] == 16
    O.set_recorder(None)


def test_nan_propagation_in_filters():
    a = np.ones((40, 40), np.float32)
    a[10:14, 10:14] = np.nan
    for f in (O.gaussian(a, 3), O.local_std(a, 3), O.gradmag(a, 2), O.highpass(a, 3), O.median_filter(a, 5)):
        assert np.isnan(f[11, 11]) and np.isfinite(f[30, 30])
    assert abs(O.gaussian(a, 3)[30, 30] - 1) < 1e-5


def test_comparison_with_nan_is_nan_not_false():
    a = np.array([[np.nan, 1.0, 3.0]], np.float32)
    _, v = run("m = where(a > 2, 1, 0)", {"a": a})
    assert np.isnan(v["m"][0, 0]) and v["m"][0, 1] == 0 and v["m"][0, 2] == 1


def test_axial_orientation_conventions():
    r, c = np.mgrid[0:64, 0:64]
    ew = np.sin(r / 3.0).astype(np.float32)            # stripes varying north-south -> E-W structures
    s = O.orientation(ew, 4)[32, 32]
    assert min(abs(s - 90), 180 - abs(s - 90)) < 5
    ns = np.sin(c / 3.0).astype(np.float32)
    s2 = O.orientation(ns, 4)[32, 32]
    assert min(s2, 180 - s2) < 5 and 0 <= s2 < 180
    assert O.axial_agreement(np.float32(0), np.float32(180)) == pytest.approx(1)
    assert O.axial_difference(np.float32(170), np.float32(10)) == pytest.approx(20)
    gx = O.gradient_x(np.tile(np.arange(20, dtype=np.float32), (20, 1)), 0)
    gy = O.gradient_y(np.tile(np.arange(20, dtype=np.float32)[:, None], (1, 20)), 0)
    assert np.allclose(gx[5:-5, 5:-5], 1) and np.allclose(gy[5:-5, 5:-5], -1)    # rows run south


def test_aoi_dependent_flag_and_freeze():
    a = np.random.default_rng(0).normal(5, 2, (30, 30)).astype(np.float32)
    src = "z = zscore(a)\nk = robust_z(gaussian(a, 2))"
    e = env(("a",))
    prog = FL.compile_program(src, e)
    assert prog.statement("z").aoi_dependent and prog.statement("k").aoi_dependent
    rec = O.set_recorder(O.Recorder())
    vals = {"a": a}
    for st in prog.statements:
        vals[st.name] = FL.run_statement(st, prog, lambda k: vals[k], e, a.shape)
    new, done = FL.freeze_normalization(src, rec.normalization)
    O.set_recorder(None)
    p2 = FL.compile_program(new, e)
    assert len(done) == 2 and not p2.statement("z").aoi_dependent and "center=" in new and "scale=" in new


# ------------------------------------------------------------------ plugins
def test_plugin_registration_and_call(tmp_path):
    (tmp_path / "my_plugin.py").write_text(textwrap.dedent('''
        from lab.plugins import register_feature
        from lab.ops import gradmag

        @register_feature(name="double_edge", inputs=["x"], parameters={"sigma": 2.0}, halo_px=lambda p: 4 * p["sigma"])
        def double_edge(layers, sigma=2.0):
            return 2 * gradmag(layers["x"], sigma)

        @register_feature(name="bad_shape", inputs=["x"], parameters={})
        def bad_shape(layers):
            return layers["x"][:2]
    '''))
    reg, errs = PL.load_plugins([tmp_path])
    assert not errs and {"double_edge", "bad_shape"} <= set(reg)
    a = np.random.default_rng(1).normal(size=(20, 20)).astype(np.float32)
    prog, v = run("e = double_edge(a, sigma=3)", {"a": a}, plugins=reg)
    np.testing.assert_allclose(v["e"], 2 * O.gradmag(a, 3), rtol=1e-6, equal_nan=True)
    assert prog.statement("e").halo == 12
    with pytest.raises(FL.FeatureError):
        FL.compile_program("e = double_edge(a, sigma=3, zz=1)", env(("a",), plugins=reg))
    with pytest.raises(FL.FeatureError):
        FL.compile_program("e = double_edge(a, a)", env(("a",), plugins=reg))
    with pytest.raises(PL.PluginError):
        run("e = bad_shape(a)", {"a": a}, plugins=reg)


def test_example_plugins_load():
    reg, errs = PL.load_plugins([PL.PLUGIN_DIR])
    assert not errs and {"custom_disruption", "edge_persistence", "fabric_agreement"} <= set(reg)


# ------------------------------------------------------------------ library
def test_library_versions_cycles_rename_and_reload(tmp_path):
    L = FeatureLibrary(tmp_path / "r.json")
    L.upsert("f1", "gradmag(rtp, 3)", raw_inputs=["rtp"], depends_on=[])
    L.upsert("f2", "f1 * 2", raw_inputs=["rtp"], depends_on=["f1"])
    e = L.upsert("f1", "gradmag(rtp, 4)", raw_inputs=["rtp"], depends_on=[])
    assert e["version"] == 2 and e["history"][0]["formula"] == "gradmag(rtp, 3)"
    with pytest.raises(LibraryError, match="circular"):
        L.upsert("f1", "f2 + 1", raw_inputs=["rtp"], depends_on=["f2"])
    with pytest.raises(LibraryError):
        L.delete("f1")                                   # f2 depends on it
    L.rename("f1", "edge")
    assert L.get("f2")["formula"] == "edge * 2" and L.get("f2")["depends_on"] == ["edge"]
    L2 = FeatureLibrary(tmp_path / "r.json")
    assert set(L2.features) == {"edge", "f2"} and L2.closure(["f2"]) == ["edge", "f2"]
    data = L2.export_json(["f2"])
    L3 = FeatureLibrary(tmp_path / "other.json")
    assert L3.import_json(data) == {"edge": "added", "f2": "added"}
    bad = json.loads(data)
    bad["features"]["edge"]["depends_on"] = ["f2"]
    with pytest.raises(LibraryError):
        FeatureLibrary(tmp_path / "x.json").import_json(json.dumps(bad))


# ------------------------------------------------------------------ engine / cache / halo
def test_engine_cache_invalidation_and_saved_feature(demo, tmp_path):
    E = engine(demo, tmp_path)
    src = ('PARAMS = {"sigma": {"min": 1, "max": 10, "step": 1, "default": 2}}\n'
           "g = gradmag(rtp, sigma=sigma)\nt = highpass(det_elev, sigma=5)\nf = abs(t) * g")
    w = Window(100, 100, 120, 120)
    r1 = E.evaluate(E.compile(src), w)
    assert set(r1.computed) == {"g", "t", "f"}
    r2 = E.evaluate(E.compile(src, {"sigma": 3}), w)
    assert set(r2.computed) == {"g", "f"} and r2.cache_hits == 1          # t reused, only dependents recomputed
    E.library.upsert("saved_g", "gradmag(rtp, 3)", raw_inputs=["rtp"], depends_on=[], halo_px=15)
    r3 = E.evaluate(E.compile("h = saved_g * 2"), w)
    np.testing.assert_allclose(r3.arrays["h"], 2 * r2.arrays["g"], rtol=1e-5, equal_nan=True)


def test_window_halo_equals_full_grid(demo, tmp_path):
    E = engine(demo, tmp_path)
    src = "x = ridge_strength(gaussian(rtp, 3), 2)\ny = local_std(highpass(det_elev, 4), 2)"
    prog = E.compile(src)
    full = E.evaluate(prog, Window(0, 0, E.W, E.H))
    w = Window(150, 120, 90, 80)
    part = E.evaluate(prog, w)
    for k in ("x", "y"):
        ref = full.arrays[k][120:200, 150:240]
        np.testing.assert_allclose(part.arrays[k], ref, atol=1e-5, rtol=1e-5, equal_nan=True)


def test_shape_mismatch_rejected():
    a = np.ones((5, 5), np.float32)
    e = env(("a",))
    prog = FL.compile_program("x = a + 1", e)
    with pytest.raises(FL.FeatureError):
        FL.run_statement(prog.statement("x"), prog, lambda k: np.ones((4, 4), np.float32), e, (5, 5))


# ------------------------------------------------------------------ association
def test_auroc_ap_prevalence_and_recall():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    s = np.array([10, 9, 8, 1, 1, 1, 1, 1, 1, 1], float)
    assert A.auroc(s, y) == 1.0 and A.auroc(-s, y) == 0.0
    assert A.average_precision(s, y) == pytest.approx(1.0)
    assert A.recall_at(s, y, 0.2) == 1.0 and A.recall_at(s, y, 0.1) == 0.5
    x = np.zeros((20, 20), np.float32)
    f = np.zeros((20, 20), bool)
    f[5, :] = True
    x[5, :] = 1
    m = A.metrics(x, f, budgets=(0.05,))
    assert m["auroc"] == 1.0 and m["ap_lift"] == pytest.approx(1 / m["prevalence"]) and m["point_biserial"] > 0.99
    lo = A.metrics(-x, f, direction="low")
    assert lo["auroc"] == 1.0 and lo["association_sign"] == -1


def test_frozen_direction_uses_development_pixels_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 80)).astype(np.float32)
    f = np.zeros((40, 80), bool)
    f[::4, :40] = True
    x[f] += 3                                                     # positive association on the west (dev) half
    f[::4, 40:] = True                                            # east (eval) half: faults but inverted signal
    x[:, 40:][f[:, 40:]] -= 6
    dev, ev = A.split_masks(x.shape, "halves_ew", buffer_px=2)
    T = A.leaderboard({"x": x}, f, direction_mode="frozen", dev_mask=dev, eval_mask=ev)
    assert T.direction[0] == "high" and T.auroc[0] < 0.5          # eval labels never flipped the direction


def test_tiles_insufficient_support():
    x = np.random.default_rng(1).normal(size=(60, 60)).astype(np.float32)
    f = np.zeros((60, 60), bool)
    f[5:8, 5:30] = True
    T = A.tile_table(x, f, "grid", 3, min_pos=20, min_neg=100)
    assert (T.status == A.INSUFFICIENT).sum() >= 7 and (T[T.tile_row == 0].iloc[0].status == "OK")


def test_negative_control_and_bootstrap_shapes():
    x = np.random.default_rng(2).normal(size=(80, 80)).astype(np.float32)
    f = np.zeros((80, 80), bool)
    f[40, 5:75] = True
    x[39:42, :] += 5
    nc = A.negative_control(x, f, n=8, min_shift_px=10)
    assert nc["auroc"] > nc["null_p95"] and nc["exceeds_null_p95"]
    bb = A.block_bootstrap(x, f, block_px=10, n_boot=30)
    assert bb["auroc_ci"][0] <= nc["auroc"] <= bb["auroc_ci"][1] + 1e-9


def test_proximity_profile_bands():
    f = np.zeros((50, 50), bool)
    f[25, :] = True
    r = np.abs(np.arange(50) - 25)[:, None] * np.ones((1, 50))
    P = A.proximity_profile(r.astype(np.float32), f, (slice(0, 50), slice(0, 50)), 100.0)
    assert P.iloc[0]["band"] == "on fault" and P.iloc[0]["median"] == 0 and P["median"].dropna().is_monotonic_increasing


# ------------------------------------------------------------------ complementarity / tournament / transects
def test_complementarity_categories():
    f = np.zeros((30, 30), bool)
    f[5, 2:12] = True                                             # fault A: baseline finds it
    f[20, 15:25] = True                                           # fault B: only the feature finds it
    base = np.zeros((30, 30), np.float32)
    base[5, 2:12] = 1
    feat = np.zeros((30, 30), np.float32)
    feat[20, 15:25] = 1
    T, cat, comp = CP.compare(base, feat, f, budget=10 / 900, radius_px=1)
    assert set(comp.category) == {"baseline only", "feature only"}
    assert (cat[20, 15:25] == 3).all() and (cat[5, 2:12] == 2).all()
    assert T.attrs["unique_feature_components"] == 1 and "NOT DTI" in T.attrs["note"]


def test_tournament_validate_selects_on_dev_and_resumes(demo, tmp_path):
    E = engine(demo, tmp_path)
    tpl = ('PARAMS = {"sigma": {"min": 1, "max": 8, "step": 1, "default": 2}}\n'
           "feat = gradmag(det_elev_slope, sigma=sigma)")
    w = Window(0, 0, 200, 200)
    f = E.faults(w)
    folds = TO.block_folds(f.shape, 50, 2)
    out = TO.run(E, tpl, "feat", {"sigma": [1, 4]}, w, f, "VALIDATE", folds, 3, 0.01, tmp_path / "t")
    sel = out["summary"]["selected_per_fold"]
    R = out["results"]
    for p in sel:
        k = p["fold"]
        assert p["dev_auroc"] == R[f"fold{k}_dev_auroc"].max()   # chosen by development score only
    again = TO.run(E, tpl, "feat", {"sigma": [1, 4]}, w, f, "VALIDATE", folds, 3, 0.01, tmp_path / "t")
    assert len(again["results"]) == 2                             # resumed, not duplicated
    with pytest.raises(ValueError):
        TO.run(E, tpl, "feat", {"sigma": [50]}, w, f, "EXPLORE", None, 0, 0.01, None)


def test_transect_sampling_and_markers():
    a = np.tile(np.arange(10, dtype=np.float32), (10, 1))
    f = np.zeros((10, 10), bool)
    f[:, 5] = True
    T, marks = TR.transect({"a": a}, (5, 0), (5, 9), 1.0, 100.0, f)
    np.testing.assert_allclose(T.a, np.arange(10))
    assert marks == [pytest.approx(0.5)]


# ------------------------------------------------------------------ export / integration
def test_native_export_alignment_and_values(demo, tmp_path):
    from lab.export_native import export_features

    E = engine(demo, tmp_path)
    E.library.upsert("smooth_rtp", "gaussian(rtp, 3)", raw_inputs=["rtp"], depends_on=[], halo_px=14)
    man = export_features(E, ["smooth_rtp"], tmp_path / "exports", tile=128)
    with rasterio.open(man["layers"]["smooth_rtp"]) as d, rasterio.open(E.feature_tif) as s:
        assert d.crs == s.crs and d.transform == s.transform and (d.height, d.width) == (s.height, s.width)
        got = d.read(1)
    full = E.evaluate(None, Window(0, 0, E.W, E.H), library_targets=["smooth_rtp"]).arrays["smooth_rtp"]
    np.testing.assert_allclose(got, full, atol=1e-5, equal_nan=True)
    with pytest.raises(FileExistsError):
        export_features(E, ["smooth_rtp"], tmp_path / "exports", tile=128)
    E.library.upsert("z_rtp", "zscore(rtp)", raw_inputs=["rtp"], depends_on=[], aoi_dependent=True)
    with pytest.raises(ValueError, match="AOI-dependent"):
        export_features(E, ["z_rtp"], tmp_path / "exports2")


def test_integration_api_without_streamlit(demo, tmp_path):
    from lab.integration import FeatureStore

    root = demo[0]
    fs = FeatureStore(root)
    fs.library.upsert("edge", "gradmag(det_elev, 2)", raw_inputs=["det_elev"], depends_on=[], halo_px=11)
    a = fs.compute("edge", window=(10, 10, 50, 60))
    assert a.shape == (50, 60) and a.dtype == np.float32
    assert fs.list()[0]["name"] == "edge"
    import lab.integration as I

    assert "streamlit" not in open(I.__file__, encoding="utf-8").read().split('"""', 2)[2]
