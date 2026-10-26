"""M7 + shared submission fixes: small synthetic checks (no Drive, no models)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

H, W = 40, 40


def _data():
    valid = np.ones((H, W), bool)
    valid[:3] = False
    tv = np.ones((H, W), bool)
    tv[:1] = False
    tv[10, :5] = False                                  # template NaN INSIDE the data footprint
    lab = np.zeros((H, W), bool)
    lab[20, 5:35] = True
    return SimpleNamespace(valid=valid, template_valid=tv, labels=lab)


# ============================================================================ shared submission fixes
def test_finalise_never_finite_where_template_nan():
    from gems_bench.gems3.submission import finalise

    d = _data()
    q = np.ones((H, W), np.float32)
    a = finalise(q, d, "template")
    assert np.isnan(a[~d.template_valid]).all()          # includes row 10 inside the footprint
    assert (a[d.template_valid & ~d.valid] == 0).all()
    assert np.isfinite(a[d.template_valid]).all()
    b = finalise(q, d, "nan")
    assert np.isnan(b[~d.template_valid | ~d.valid]).all()


def test_eligible_mask_excludes_known_ring_and_template():
    from gems_bench.gems3.submission import eligible_mask, ring_zone

    d = _data()
    e0 = eligible_mask(d, 0)
    assert not (e0 & d.labels).any() and not (e0 & ~d.template_valid).any() and not (e0 & ~d.valid).any()
    e2 = eligible_mask(d, 2)
    assert not (e2 & ring_zone(d.labels, 2)).any() and e2.sum() < e0.sum()
    legacy = eligible_mask(d, 2, "evaluation_zone")
    assert (legacy & d.labels).any()


def test_apply_budget_counts_eligible_pixels_only():
    from gems_bench.gems3 import representation as R
    from gems_bench.gems3.submission import eligible_mask

    d = _data()
    s = np.where(d.labels, 1.0, np.random.default_rng(0).random((H, W)) * 0.5).astype(np.float32)
    el = eligible_mask(d, 0)
    q = R.apply({"select": "top"}, s, d.valid, 0.05, eligible=el)
    assert not (q[d.labels] > 0).any()                    # known pixels (highest scores) consume no budget
    assert abs(int((q > 0).sum()) - int(np.ceil(0.05 * el.sum()))) <= 1


def test_area_ledger_reports_masked_pixels():
    from gems_bench.gems3.submission import area_ledger

    d = _data()
    q = np.zeros((H, W), np.float32)
    q[20, 5:10] = 1                                       # on known
    q[30, 5:10] = 1                                       # eligible
    r = area_ledger(q, d, 0)
    assert r["on_known_px"] == 5 and r["on_eligible_px"] == 5


# ============================================================================ protocol
def _unit():
    from gems_bench.gems3.official import _unit as mk

    d = _data()
    G = np.zeros((H, W), bool)
    G[30, 5:35] = True
    return mk("disc_official", 0, d.valid, G, d.labels, False, d.labels), d


def test_unit_eligible_and_protocol_keys():
    from gems_bench.m7.protocol import ProtocolMismatch, assert_comparable, eligible, protocol_key

    u, d = _unit()
    t = u.crop(d.template_valid)
    e = eligible(u, t, 1)
    assert not (e & u.exclude).any()
    k1, k2 = protocol_key(u, t, "eligible", True), protocol_key(u, t, "evaluation_zone", True)
    assert k1 != k2
    a = pd.DataFrame([{"task": "disc_official", "zone": 0, "protocol_key": k1}])
    b = pd.DataFrame([{"task": "disc_official", "zone": 0, "protocol_key": k2}])
    assert_comparable(a, a)
    with pytest.raises(ProtocolMismatch):
        assert_comparable(a, b)


def test_block_components_sum_to_components():
    from gems_bench.m7.protocol import block_components

    u, _ = _unit()
    q = np.zeros(u.ev.shape, np.float32)
    q[np.nonzero(u.eval_truth())] = 1.0
    q[5:8, 5:8] = 1.0
    c = u.components(q)
    b = block_components(u, q, 10)
    assert np.isclose(b.tp_w.sum(), c.tp) and np.isclose(b.fp_w.sum(), c.fp)


# ============================================================================ generator / rasterisation
def test_edge_response_finds_a_step():
    from gems_bench.m7.generator import response

    a = np.zeros((60, 60), np.float32)
    a[:, 30:] = 10.0
    r = response(a, np.ones(a.shape, bool), 2.0, "edge")
    cols = np.nonzero(r["nms"][30])[0]
    assert len(cols) and np.all(np.abs(cols - 29.5) <= 2)
    assert abs(((r["strike_deg"][30, 29] + 90) % 180) - 90) < 20          # N-S strike


def test_fuse_merges_duplicates_and_keeps_provenance():
    from gems_bench.gems3.candidates import CandidateSet
    from gems_bench.m7.generator import fuse

    p = np.c_[np.arange(5, 30), np.full(25, 10)]
    t = pd.DataFrame({"uid": ["a", "b"], "source": ["s1", "s2"], "modality": ["magnetic", "gravity"],
                      "kind": ["feature", "feature"], "scale_m": [200.0, 400.0]})
    out = fuse(CandidateSet(t, [p, p[:20]]), (40, 40), {"dedup_radius_px": 2, "dedup_overlap": 0.7,
                                                         "agreement_radius_px": 2})
    assert len(out) == 1 and out.table.n_sources.iloc[0] == 2 and out.table.evidence_class.iloc[0] == "multi_evidence"


def test_select_core_respects_budget_and_heldout_roundtrip(tmp_path):
    from gems_bench.m7.rasterize import load_heldout, save_heldout, select_core
    from gems_bench.m7.workspace import M7Workspace

    sr = np.full((H, W), -np.inf, np.float32)
    sr[10, :] = np.linspace(0, 1, W)
    el = np.ones((H, W), bool)
    core = select_core(sr, el, 10 / el.sum())
    assert core.sum() == 10 and core[10, -1]
    ws = M7Workspace(tmp_path, {"save_parquet": False})
    u = SimpleNamespace(key="geo:0")
    save_heldout(ws, "m", u, core.astype(np.float32), "k1")
    q, k = load_heldout(ws, "m", u, with_key=True)
    assert k == "k1" and (q > 0).sum() == 10


# ============================================================================ configuration
def test_config_rejects_unknown_keys_and_retraining():
    from gems_bench.m7.config import M7ConfigError, load_m7_config

    cfg = load_m7_config()
    assert cfg["evaluation"]["budget_domain"] == "eligible"
    with pytest.raises(M7ConfigError):
        load_m7_config({"evalution": {}})
    with pytest.raises(M7ConfigError):
        load_m7_config({"retrain_existing_cnns": True})
    with pytest.raises(M7ConfigError):
        load_m7_config({"probes": {"max_probes": 5}})
