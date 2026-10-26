"""GEMS3: official-convention scoring, representations, nested selection, submission export, candidates, oracle,
map completion, DEM NoData, config validation and stage resume / failure isolation.

Tiny synthetic rasters only (no competition data). Run later with:  python -m pytest tests/test_gems3.py -q
Geometry: 100 m pixels, R = 300 m, so k(100 m) = 2/3, k(200 m) = 1/3, k(300 m) = 0.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from gems_bench.metric import MaskedScorer, dti_components

H = W = 40


def hline(r, c0=5, c1=35, shape=(H, W)):
    a = np.zeros(shape, bool)
    a[r, c0:c1] = True
    return a


def vline(c, r0=5, r1=35, shape=(H, W)):
    a = np.zeros(shape, bool)
    a[r0:r1, c] = True
    return a


# ============================================================================ 1. official DTI parity
def _cases():
    from gems_bench.evaluation.parity import parity_cases

    return parity_cases()


@pytest.mark.parametrize("supports", [True, False])
def test_fast_scorer_matches_masked_scorer_and_reference(supports):
    from gems_bench.evaluation.parity import reference_dti
    from gems_bench.gems3.official import FastScorer

    for c in _cases():
        t = c["truth"]
        ev = c.get("eval_mask", np.ones_like(t))
        ex = c.get("exclude_mask")
        p = np.asarray(c["pred"], np.float64)
        a = MaskedScorer(t, ev, ex, exclude_supports_tp=supports).components(p)
        b = FastScorer(t, ev, ex, exclude_supports_tp=supports).components(p)
        assert b.tp == pytest.approx(a.tp, abs=1e-9), c["case"]
        assert b.fp == pytest.approx(a.fp, abs=1e-9), c["case"]
        assert b.dti == pytest.approx(a.dti, abs=1e-9), c["case"]
        if supports:
            ref = reference_dti(c["pred"], t, eval_mask=ev, exclude_mask=ex)
            assert b.dti == pytest.approx(ref["dti"], abs=1e-9), c["case"]


def test_max_based_tp_is_not_additive():
    """Two predictions next to one truth pixel give TP = max, not sum."""
    from gems_bench.gems3.official import FastScorer

    t = np.zeros((H, W), bool)
    t[20, 20] = True
    p = np.zeros((H, W))
    p[20, 21] = p[20, 19] = 1.0
    c = FastScorer(t, np.ones_like(t)).components(p)
    assert c.tp == pytest.approx(2 / 3)
    assert c.fp == pytest.approx(2 * (1 - 2 / 3))


@pytest.mark.parametrize("off,k", [(1, 2 / 3), (2, 1 / 3), (3, 0.0)])
def test_kernel_boundary_fast_scorer(off, k):
    from gems_bench.gems3.official import FastScorer

    t = hline(20)
    p = hline(20 + off).astype(float)
    c = FastScorer(t, np.ones_like(t)).components(p)
    assert c.tp == pytest.approx(30 * k, abs=1e-9)
    assert c.fp == pytest.approx(30 * (1 - k), abs=1e-9)


# ============================================================================ 2. known-fault masking
def test_masked_predictions_have_no_effect_under_official_reading():
    from gems_bench.p3.curves import BinaryCurve

    known, new = hline(20), hline(21)
    rng = np.random.default_rng(0)
    base = rng.random((H, W)) * 0.3
    other = base.copy()
    other[known] = 1.0                                       # change predictions on masked pixels only
    for p in (base, other):
        assert np.isfinite(dti_components(p, new, exclude_mask=known, exclude_supports_tp=False).dti)
    a = dti_components(base, new, exclude_mask=known, exclude_supports_tp=False)
    b = dti_components(other, new, exclude_mask=known, exclude_supports_tp=False)
    assert a.tp == pytest.approx(b.tp) and a.fp == pytest.approx(b.fp)
    s1 = MaskedScorer(new, None, known, exclude_supports_tp=False)
    assert s1.components(base).dti == pytest.approx(s1.components(other).dti)
    c1 = BinaryCurve(base, new, np.ones_like(new), known, exclude_supports_tp=False).at([0.2, 0.5])
    c2 = BinaryCurve(other, new, np.ones_like(new), known, exclude_supports_tp=False).at([0.2, 0.5])
    assert np.allclose(c1.tp_w, c2.tp_w) and np.allclose(c1.fp_w, c2.fp_w)


def test_masking_audit_cases():
    from gems_bench.gems3.official import masking_audit

    m = masking_audit().set_index(["case", "convention"]).dti
    # predicting only the known trace next to a 100 m correction
    assert m[("predict_known_only", "official_pixel_exact_no_support")] == pytest.approx(0.0)
    assert m[("predict_known_only", "pixel_exact_with_support")] == pytest.approx(20 / (20 + 0.8 * 10), abs=1e-4)
    # a ring 200 m off a known trace, far from new truth: penalised officially, forgiven by the 300 m dilation
    assert m[("ring_far_from_new", "official_pixel_exact_no_support")] == pytest.approx(30 / (30 + 0.2 * 30), abs=1e-4)
    assert m[("ring_far_from_new", "excl_dilated_300m_with_support")] == pytest.approx(1.0, abs=1e-4)
    # known pixels filled with 1: neutral under the official mask, FP if known faults were not masked
    assert m[("known_fill_plus_new", "official_pixel_exact_no_support")] == pytest.approx(1.0, abs=1e-4)
    assert m[("known_fill_plus_new", "known_is_fp")] < 1.0


def test_adjacent_known_and_new_fault_official_unit():
    """New truth adjacent to a known trace stays scorable; the known pixel itself is never truth."""
    from gems_bench.gems3.official import FastScorer

    known, new = hline(20), hline(21)
    s = FastScorer(new | known, np.ones_like(new), known, exclude_supports_tp=False)
    assert s.n == int(new.sum())
    c = s.components(new.astype(float))
    assert c.dti == pytest.approx(1.0, abs=1e-6)


# ============================================================================ 3. tiny context + units
@dataclass
class _Grid:
    height: int = H
    width: int = W
    pixel_km_x: float = 0.1
    pixel_km_y: float = 0.1
    pixel_km: float = 0.1

    @property
    def shape(self):
        return (self.height, self.width)


@dataclass
class _Data:
    labels: np.ndarray
    valid: np.ndarray
    grid: _Grid = field(default_factory=_Grid)

    @property
    def shape(self):
        return self.labels.shape


class _Splits:
    def __init__(self, labels):
        self.n_splits = 2
        self.wh = [vline(10) & labels, vline(30) & labels]
        sm = np.full(labels.shape, -1, np.int32)
        lab, _ = ndimage.label(labels, structure=np.ones((3, 3), bool))
        sm[labels] = lab[labels] - 1
        self.system_map = sm
        self.systems = pd.DataFrame({"system_id": np.arange(lab.max()), "trace_km": [3.0] * lab.max()})

    def withheld(self, d):
        return self.wh[d]

    def eval_zone(self, d):
        return ndimage.distance_transform_edt(~self.wh[d]) <= 8


class _Folds:
    def __init__(self):
        fm = np.full((H, W), -1, np.int16)
        fm[:, :19], fm[:, 21:] = 0, 1
        self.fold_map = fm
        self.n_folds = 2


class TinyCtx:
    def __init__(self):
        lab = vline(10) | vline(30) | hline(20, 12, 28)
        self.data = _Data(lab, np.ones((H, W), bool))
        self.folds = _Folds()
        self.splits = _Splits(lab)
        self.fold_ids = [0, 1]
        rng = np.random.default_rng(1)
        d = ndimage.distance_transform_edt(~lab)
        self._oof = {m: np.clip(np.exp(-d / s) + 0.3 * rng.random((H, W)), 0, 1).astype(np.float32)
                     for m, s in (("A", 1.5), ("B", 3.0))}

    def oof(self, m):
        return self._oof[m]

    def discovery_pred(self, m, d):
        return self._oof[m]


def test_units_follow_the_contract():
    from gems_bench.gems3.official import build_units

    ctx = TinyCtx()
    us = {u.key: u for u in build_units(ctx, ["geo", "disc_official", "disc_excl"])}
    u = us["disc_official:0"]
    assert u.supports_tp is False
    assert np.array_equal(u.exclude, u.crop(ctx.data.labels & ~ctx.splits.withheld(0)))      # pixel-exact
    assert not (u.known & u.truth).any()                                                       # held-out never visible
    e = us["disc_excl:0"]
    assert e.supports_tp and e.exclude.sum() > u.exclude.sum()                                 # dilated
    assert us["geo:0"].exclude is None and us["geo:0"].known is None


# ============================================================================ 4. representations
def test_top_mask_area_and_range():
    from gems_bench.gems3.representation import apply, top_mask

    s = np.arange(H * W, dtype=float).reshape(H, W) / (H * W)
    ev = np.ones((H, W), bool)
    m = top_mask(s, ev, 0.05)
    assert m.sum() == int(np.ceil(0.05 * H * W))
    q = apply({"select": "top", "post": [["soft_top", {"v_min": 0.5}]]}, s, ev, 0.05)
    assert q.min() >= 0 and q.max() <= 1 and q[m].min() >= 0.5 and (q[~m] == 0).all()


def test_crest_lines_are_thin_and_budgeted():
    from gems_bench.gems3.representation import crest_map, crest_top

    d = ndimage.distance_transform_edt(~hline(20))
    s = np.exp(-d / 3.0)
    ev = np.ones((H, W), bool)
    c = crest_top(s, ev, 0.02, crest_map(s, ev, 1.0))
    assert c.sum() <= int(np.ceil(0.02 * H * W)) + 1
    assert c[20].sum() >= 0.8 * c.sum()                         # the crest is the centre line


def test_dilation_geometry():
    from gems_bench.gems3.representation import oriented_dilate, op_dilate, op_skeleton

    m = np.zeros((H, W), bool)
    m[20, 20] = True
    assert op_dilate(m.astype(np.float32), {"radius_px": 1}, {}).sum() == 5            # disk r = 1: a plus
    line = hline(20)
    across = oriented_dilate(line, across_px=1, along_px=0)
    assert across[19, 10:30].all() and across[21, 10:30].all()                       # widened across strike
    assert not across[20, :3].any()                                                  # not along it
    along = oriented_dilate(line, across_px=0, along_px=2)
    assert along[19:22, 3:5].any()                                                   # extended past the tip
    assert not along[:18].any() and not along[23:].any()                             # never more than 2 px off-line
    assert along[20, 10:30].all() and not along[18, 10:30].any()                     # interior: along only
    w3 = op_skeleton(ndimage.binary_dilation(line, iterations=2).astype(np.float32), {"width_px": 1}, {})
    assert 2 <= w3[:, 20].sum() <= 3


def test_known_ring_and_fill():
    from gems_bench.gems3.representation import op_known_fill, op_known_ring

    known = hline(20)
    q = np.ones((H, W), np.float32)
    r = op_known_ring(q, {"ring_px": 2}, {"known": known})
    assert (r[known] == 1).all() and (r[18:23, 10][~known[18:23, 10]] == 0).all() and r[25, 10] == 1
    f = op_known_fill(np.zeros((H, W), np.float32), {}, {"known": known})
    assert (f[known] == 1).all() and f.sum() == known.sum()


def test_with_ring_inserted_before_soft_ops():
    from gems_bench.gems3.representation import with_ring

    rep = {"select": "crest", "post": [["dilate", {"radius_px": 1}], ["two_level", {"halo_px": 1}]]}
    assert [n for n, _ in with_ring(rep, 2)["post"]] == ["dilate", "known_ring", "two_level"]
    assert [n for n, _ in with_ring(rep, 0)["post"]] == ["dilate", "two_level"]


# ============================================================================ 5. nested selection
def _toy_table():
    rows = []
    for z in range(3):
        for f in (0.01, 0.05):
            tp = 10 + (5 if f == 0.05 else 0) + z
            rows.append({"zone": z, "frac": f, "tp_w": tp, "fp_w": 20 * f * 100, "fn_w": 30 - tp, "n_truth": 30})
    return pd.DataFrame(rows)


def test_nested_select_ignores_the_held_out_zone_labels():
    from gems_bench.gems3.metric_opt import nested_select

    t = _toy_table()
    per, _ = nested_select(t, ["frac"])
    t2 = t.copy()
    t2.loc[(t2.zone == 0) & (t2.frac == 0.01), "tp_w"] = 30           # zone 0 now prefers 0.01 in-sample
    t2.loc[(t2.zone == 0) & (t2.frac == 0.01), "fn_w"] = 0
    per2, _ = nested_select(t2, ["frac"])
    assert per.set_index("zone").frac[0] == per2.set_index("zone").frac[0]   # its own choice is unchanged


def test_nested_select_respects_allowed_candidates():
    from gems_bench.gems3.metric_opt import nested_select

    t = _toy_table()
    allowed = {z: {"0.01"} for z in range(3)}
    per, _ = nested_select(t, ["frac"], allowed)
    assert (per.frac == 0.01).all()


def test_metric_opt_end_to_end_on_tiny_context():
    from gems_bench.gems3.metric_opt import run
    from gems_bench.gems3.official import build_units

    ctx = TinyCtx()
    cfg = {"primary_task": "disc_official", "metric_opt": {
        "fusion_weights": [0.0, 0.5, 1.0], "area_fracs": [0.02, 0.05, 0.1], "thresholds": [0.3, 0.6],
        "n_weights_stage2": 1, "refine_factors": [0.8, 1.0], "ridge_sigma_px": 1.0, "raw_models": True,
        "known_ring_px": [1],
        "representations": {"binary": {"select": "top"}, "crest_1px": {"select": "crest"}}}}
    out = run(ctx, build_units(ctx, ["geo", "disc_official"]), ["A", "B"], cfg)
    assert set(out["choice"]) == {"geo", "disc_official"}
    assert "ring" in out["choice"]["disc_official"]["params_all"]
    assert out["paired"].dti_heldout.between(0, 1).all()
    assert (out["stage2"].area_frac <= 1).all()


# ============================================================================ 6. submission export
class _SubData:
    def __init__(self, tmp):
        from affine import Affine
        from rasterio.crs import CRS

        self.valid = np.ones((H, W), bool)
        self.valid[:3] = False
        self.template_valid = np.ones((H, W), bool)
        self.template_valid[:1] = False
        self.labels = hline(20)
        self.template_profile = {"driver": "GTiff", "height": H, "width": W, "count": 1, "dtype": "float32",
                                 "crs": CRS.from_epsg(32611), "transform": Affine(100, 0, 300000, 0, -100, 4500000)}
        self.grid = None

    def template_matches(self):
        return True


def test_finalise_template_policy():
    from gems_bench.gems3.submission import finalise

    d = _SubData(None)
    q = np.full((H, W), np.nan, np.float32)
    q[10, 10] = 1.7
    a = finalise(q, d, "template")
    assert np.isnan(a[0]).all() and (a[1:3] == 0).all() and a[10, 10] == 1.0 and np.isfinite(a[3:]).all()
    b = finalise(q, d, "nan")
    assert np.isnan(b[:3]).all()


def test_submission_geotiff_compliance(tmp_path):
    from gems_bench.gems3.submission import finalise, validate
    from gems_bench.submission import write_submission

    d = _SubData(tmp_path)
    q = np.zeros((H, W), np.float32)
    q[d.labels] = 1.0
    p = write_submission(finalise(q, d), d, tmp_path / "s.tif")
    res = validate(p, d)
    assert res["dtype"] == "float32" and res["epsg"] == 32611 and not res["checks_failed"]
    bad = q.copy()
    bad[5, 5] = np.nan                                                       # NaN inside the footprint
    p2 = write_submission(np.where(d.valid, bad, np.nan).astype(np.float32), d, tmp_path / "b.tif")
    with pytest.raises(AssertionError):
        validate(p2, d)


# ============================================================================ 7. candidates + oracle
def test_tip_extension_points_away_from_the_trace():
    from gems_bench.gems3.candidates import tip_paths

    known = hline(20, 10, 20)
    ps = tip_paths(known, np.ones((H, W), bool), [0.5], tail_px=4)
    cols = sorted(int(p[:, 1].mean()) for p, _ in ps)
    assert len(ps) == 2 and cols[0] < 10 and cols[1] >= 20
    assert not any(known[p[:, 0], p[:, 1]].any() for p, _ in ps)


def test_fuse_merges_duplicates_and_keeps_provenance():
    from gems_bench.gems3.candidates import CandidateSet, fuse

    a = np.argwhere(hline(20))
    b = np.argwhere(hline(21, 6, 34))
    c = np.argwhere(vline(5))
    raw = CandidateSet(pd.DataFrame({"source": ["mag", "grav", "mag"], "kind": ["feature"] * 3, "scale_m": [300.0] * 3}),
                       [a, b, c])
    out = fuse(raw, (H, W), {"dedup_radius_px": 2, "dedup_overlap": 0.7, "agreement_radius_px": 2})
    assert len(out) == 2                                         # the grav line 100 m off the mag line is a duplicate
    assert set(out.table.sources.iloc[0].split(";")) == {"grav", "mag"}   # longest first: the E-W mag line owns it
    assert out.table.sources.iloc[1] == "mag"
    assert out.table.agree_grav.iloc[0] >= 0.9


def test_oracle_recall_perfect_and_empty():
    from gems_bench.gems3.candidates import CandidateSet
    from gems_bench.gems3.official import build_units
    from gems_bench.gems3.oracle import coverage

    ctx = TinyCtx()
    u = build_units(ctx, ["disc_official"])[0]
    cfg = {"oracle": {"system_recovered_support": 0.5, "system_recovered_frac": 0.5, "short_system_km": 2.0}}
    G = u.eval_truth()
    perfect = CandidateSet(pd.DataFrame({"source": ["x"], "sources": ["x"]}), [np.argwhere(G)])
    c = coverage(ctx, u, perfect, cfg)
    assert c["recall_w"] == pytest.approx(1.0) and c["system_recall"] == pytest.approx(1.0) and c["dti"] > 0.99
    empty = CandidateSet(pd.DataFrame(columns=["source", "sources"]), [])
    e = coverage(ctx, u, empty, cfg)
    assert e["recall_w"] == 0 and e["system_recall"] == 0


# ============================================================================ 8. map completion
def test_corruption_removes_only_label_pixels_and_is_seeded():
    from gems_bench.gems3.map_completion import CLASSES, corrupt

    lab = vline(10, 2, 38) | hline(30, 15, 35) | hline(5, 20, 24)
    mc = {"sample_frac": 0.5, "short_fault_km": 1.0, "trim_endpoint_km": [0.3], "delete_segment_km": [0.6],
          "shift_px": 1}
    d1, r1, p1 = corrupt(lab, np.random.default_rng(3), mc)
    d2, r2, _ = corrupt(lab, np.random.default_rng(3), mc)
    assert np.array_equal(d1, d2)
    for c in CLASSES:
        assert not (r1[c] & d1).any()                                # removed pixels are not in the damaged map
        assert not (r1[c] & ~lab).any()                              # and were mapped faults
    assert set(p1["class"]) == set(CLASSES)


def test_systems_split_before_corruption():
    from gems_bench.gems3.map_completion import systems_by_fold

    sm = np.full((H, W), -1, np.int32)
    sm[vline(10)], sm[vline(30)] = 0, 1
    t = systems_by_fold(sm, _Folds().fold_map).set_index("system").fold
    assert t[0] == 0 and t[1] == 1


# ============================================================================ 9. DEM NoData
def test_dem_block_features_keep_nodata_as_nan():
    from gems_bench.dem.morphology import MorphConfig, block_features

    z = np.tile(np.linspace(0, 50, 64, dtype=np.float32), (64, 1))
    z[:10, :10] = np.nan
    f = block_features(z, 1.0, MorphConfig())
    assert (f["valid"][:10, :10] == 0).all() and f["valid"][20:, 20:].all()
    assert np.isnan(f["slope_mean"][:10, :10]).all() and np.isfinite(f["slope_mean"][20:, 20:]).all()


# ============================================================================ 10. config + stage resume
def test_config_rejects_unknown_keys_and_retraining():
    from gems_bench.gems3.config import GEMS3ConfigError, load_gems3_config

    with pytest.raises(GEMS3ConfigError):
        load_gems3_config({"metric_opt": {"fusion_weigths": [0.5]}})
    with pytest.raises(GEMS3ConfigError):
        load_gems3_config({"retrain_existing_cnns": True})
    with pytest.raises(GEMS3ConfigError):
        load_gems3_config({"metric_opt": {"representations": {"x": {"select": "top", "post": [["blur", {}]]}}}})
    cfg = load_gems3_config({"metric_opt": {"representations": {"x": {"select": "crest"}}}})
    assert cfg["primary_task"] == "disc_official"


def test_stage_failure_isolated_and_reuse(tmp_path):
    from gems_bench.gems3.run import GEMS3Research

    g3 = GEMS3Research(None, {}, local_root=tmp_path)
    calls = []

    def boom():
        raise ValueError("synthetic failure")

    def ok():
        calls.append(1)
        g3.ws.save_table(pd.DataFrame({"a": [1]}), "t", "audit")
        return "done"

    assert g3._stage("bad", "fp0", boom) is None
    assert g3.log[-1]["outcome"] == "FAILED"
    assert g3._stage("good", "fp1", ok, load=lambda: "reloaded") == "done"
    assert g3._stage("good", "fp1", ok, load=lambda: "reloaded") == "reloaded" and len(calls) == 1
    assert g3._stage("good", "fp2", ok) == "done" and len(calls) == 2                 # new fingerprint recomputes
    with pytest.raises(ValueError):
        g3._stage("strict", "fp3", boom, optional=False)


def test_ranker_drops_training_rows_inside_the_scored_eval_zone():
    pytest.importorskip("lightgbm")
    from gems_bench.gems3.official import build_units
    from gems_bench.gems3.ranker import lgbm_scores

    ctx = TinyCtx()
    us = build_units(ctx, ["disc_official"])
    rng = np.random.default_rng(0)
    n = 120
    t = pd.DataFrame({"task": "disc_official", "zone": np.repeat([0, 1], n // 2), "cand_id": np.arange(n),
                      "r_full": rng.integers(0, H, n), "c_full": rng.integers(0, W, n), "label": rng.integers(0, 2, n),
                      "f1": rng.random(n), "f2": rng.random(n)})
    cfg = {"ranker": {"exclude_eval_buffer_km": 0.0, "negative_weight": 0.5,
                      "lgbm": {"n_estimators": 5, "num_leaves": 4, "min_child_samples": 2, "seed": 1}}}
    _, audit = lgbm_scores(t, us, (H, W), cfg)
    a = audit.set_index("zone")
    for u in us:
        ez = np.zeros((H, W), bool)
        ez[u.sl] = u.ev
        other = t[t.zone != u.zone]
        assert a.loc[u.zone, "n_dropped_inside_eval"] == int(ez[other.r_full, other.c_full].sum())
