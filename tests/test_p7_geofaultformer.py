"""Synthetic tests of P7 GeoFaultFormer (config, attention, ConvGRU, model, losses, augmentation, leakage guards,
checkpoint contract, evaluation helpers, tracer, stitching, export recipe). No data, GPU or Drive access.
Written in the authoring session; NOT executed there (the brief forbade any local execution)."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gems_bench.p7 import augment as AU
from gems_bench.p7 import config as CF
from gems_bench.p7 import data as DA
from gems_bench.p7 import evaluate as EV
from gems_bench.p7 import targets as TG
from gems_bench.p7 import tracer as TR
from gems_bench.p7 import w2 as W2


# ----------------------------------------------------------------------------------------------- configuration
def test_config_loads_and_ladder_is_bounded():
    cfg = CF.load_p7_config()
    assert set(CF.trained_experiments(cfg)) >= {"T1", "T1_cnn", "T2", "T3"}
    assert len(cfg["ladder"]["experiments"]) <= cfg["ladder"]["max_experiments"]
    t1, t2, t3 = (CF.resolve_experiment(cfg, n) for n in ("T1", "T2", "T3"))
    assert t1["model"]["attention"]["enabled"] and not t1["model"]["heads"]["center"]
    assert t2["model"]["heads"]["center"] and t2["model"]["heads"]["orient"]
    assert t3["model"]["convgru"]["enabled"]
    assert not CF.resolve_experiment(cfg, "T1_cnn")["model"]["attention"]["enabled"]


def test_loss_on_disabled_head_is_refused():
    with pytest.raises(CF.P7ConfigError):
        CF.load_p7_config(experiments={"experiments": {"T1": {"overrides": {"loss": {"weights": {"center": 0.5}}}}}})


def test_too_many_experiments_refused():
    extra = {f"X{i}": {"kind": "train", "hypothesis": "x", "parent": None, "overrides": {}} for i in range(10)}
    with pytest.raises(CF.P7ConfigError):
        CF.load_p7_config(experiments={"experiments": extra})


def test_semantic_view_ignores_runtime_knobs():
    cfg = CF.load_p7_config()
    e = CF.resolve_experiment(cfg, "T1")
    f = copy.deepcopy(e)
    f["training"].update(micro_batch=4, grad_accum=4, infer_batch=64, ckpt_sync_min=1, max_runtime_min=30)
    assert CF.semantic_view(e) == CF.semantic_view(f)
    g = copy.deepcopy(e)
    g["training"]["lr"] = 0.01
    assert CF.semantic_view(e) != CF.semantic_view(g)


def test_micro_split_and_presets():
    assert CF.micro_split(16, 16) == (16, 1)
    assert CF.micro_split(16, 8) == (8, 2)
    assert CF.micro_split(16, 6) == (4, 4)
    g = CF.load_p7_config()["gpu"]
    assert CF.gpu_preset(g, {"cuda_available": False})[0] == "CPU"
    assert CF.gpu_preset(g, {"cuda_available": True, "total_gb": 22.5})[0] == "L4"
    assert CF.gpu_preset(g, {"cuda_available": True, "total_gb": 40.0})[0] == "A100"
    assert CF.gpu_preset(g, {"cuda_available": True, "total_gb": 15.0})[0] == "T4"


def test_experiment_id_changes_with_fingerprint():
    assert CF.experiment_id("T1", "abcdef0123456789") == "p7_t1_abcdef0123"
    assert CF.experiment_id("T1", "abcdef0123456789") != CF.experiment_id("T1", "bbcdef0123456789")


# ----------------------------------------------------------------------------------------------- attention
def test_window_partition_roundtrip():
    torch = pytest.importorskip("torch")
    from gems_bench.p7.attention import window_partition, window_reverse

    x = torch.randn(2, 16, 24, 5)
    w = window_partition(x, 8)
    assert w.shape == (2 * 2 * 3, 64, 5)
    assert torch.equal(window_reverse(w, 8, 16, 24), x)


def test_window_mask_padding_and_shift():
    pytest.importorskip("torch")
    from gems_bench.p7.attention import MASK_VALUE, window_mask

    assert window_mask(16, 16, 8, 0) is None
    m = window_mask(12, 12, 8, 0)                      # padded to 16: right / bottom windows have pad keys
    assert m.shape == (4, 64, 64)
    assert float(m[0].abs().max()) == 0.0              # top-left window: no padding
    cols = np.arange(64) % 8 + 8                       # window 1 covers columns 8..15
    pad_keys = np.flatnonzero(cols >= 12)
    assert (m[1][:, pad_keys] == MASK_VALUE).all()
    s = window_mask(16, 16, 8, 4)
    assert (s[3] == MASK_VALUE).any() and float(s[0].abs().max()) == 0.0


def test_shifted_window_never_attends_across_wraparound():
    torch = pytest.importorskip("torch")
    from gems_bench.p7.attention import SwinBlock

    torch.manual_seed(0)
    blk = SwinBlock(16, 4, 8, 4).eval()
    x = torch.randn(1, 16, 16, 16)
    x2 = x.clone()
    x2[..., -1, -1] += 10.0                            # shares a rolled window with (0, 0), different region
    with torch.no_grad():
        y1, y2 = blk(x), blk(x2)
    assert torch.allclose(y1[..., 0, 0], y2[..., 0, 0], atol=1e-5)


def test_attention_handles_maps_smaller_than_window():
    torch = pytest.importorskip("torch")
    from gems_bench.p7.attention import AttentionStage

    st = AttentionStage(32, 2, 4, 8).eval()
    with torch.no_grad():
        y = st(torch.randn(2, 32, 5, 7))
    assert y.shape == (2, 32, 5, 7) and torch.isfinite(y).all()


# ----------------------------------------------------------------------------------------------- ConvGRU
def test_convgru_zero_steps_is_identity_and_states_update():
    torch = pytest.importorskip("torch")
    from gems_bench.p7.convgru import ConvGRURefiner

    g = ConvGRURefiner(8, 16, 3, 3)
    x = torch.randn(2, 8, 12, 12)
    assert torch.equal(g(x, steps=0), x)
    assert torch.allclose(g(x), x)                     # zero-initialised output projection
    g.capture = True
    g(x)
    assert len(g.states) == 3 and not torch.allclose(g.states[0], g.states[-1])


# ----------------------------------------------------------------------------------------------- model
def _spec(C=7, P=64, **model_over):
    cfg = CF.load_p7_config()
    e = CF.resolve_experiment(cfg, "T1")
    e["model"] = CF.merge(e["model"], model_over)
    e["data"]["patch_px"] = P
    from gems_bench.p7.model import net_spec

    return net_spec([f"c{i}" for i in range(C)], {"magnetic": list(range(C // 2)), "other": list(range(C // 2, C))},
                    e, "k")


@pytest.mark.parametrize("C,P", [(3, 64), (17, 64), (9, 72), (5, 100)])
def test_model_forward_shapes_variable_channels_and_sizes(C, P):
    torch = pytest.importorskip("torch")
    from gems_bench.p7.model import build

    m = build(_spec(C, P)).eval()
    with torch.no_grad():
        out = m(torch.randn(2, C, P, P))
    assert set(out) == {"seg"} and out["seg"].shape == (2, 1, P, P)


def test_all_heads_grouped_fusion_and_convgru():
    torch = pytest.importorskip("torch")
    from gems_bench.p7.model import build

    heads = {h: True for h in CF.HEADS}
    m = build(_spec(6, 64, heads=heads, fusion={"mode": "grouped"}, convgru={"enabled": True})).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 6, 64, 64))
    assert out["orient"].shape == (1, 2, 64, 64)
    assert {"seg", "center", "orient", "dist", "ep", "jn"} == set(out)


def test_model_recurrent_bypass_reproduces_non_recurrent_path():
    torch = pytest.importorskip("torch")
    from gems_bench.p7.model import build

    torch.manual_seed(1)
    m = build(_spec(5, 64, convgru={"enabled": True, "steps": 3})).eval()
    x = torch.randn(1, 5, 64, 64)
    with torch.no_grad():
        a = m(x)["seg"]
        m.gru.steps = 0
        b = m(x)["seg"]
    assert torch.allclose(a, b, atol=1e-6)


def test_pure_cnn_path_has_no_attention_parameters():
    pytest.importorskip("torch")
    from gems_bench.p7.model import build, param_breakdown

    p_cnn = param_breakdown(build(_spec(5, 64, attention={"enabled": False})))
    p_att = param_breakdown(build(_spec(5, 64)))
    assert p_cnn["attn"] == 0 and p_att["attn"] > 0


# ----------------------------------------------------------------------------------------------- losses
def _batch(torch, B=2, P=32, pos=False, weight=1.0):
    z = torch.zeros(B, 1, P, P)
    seg = z.clone()
    if pos:
        seg[:, :, P // 2, :] = 1.0
    return {"x": torch.randn(B, 3, P, P), "w": torch.full((B, 1, P, P), weight), "seg": seg, "kern": seg.clone(),
            "center": seg.clone(), "ocos": z.clone(), "osin": z.clone(), "ovalid": seg.clone(), "dist": z + 1,
            "ep": z.clone(), "jn": z.clone(), "topo_valid": z + 1, "valid": z + 1}


@pytest.mark.parametrize("pos,weight", [(False, 0.0), (False, 1.0), (True, 1.0)])
def test_losses_are_finite_on_empty_targets(pos, weight):
    torch = pytest.importorskip("torch")
    from gems_bench.p7 import losses as LS

    lc = CF.load_p7_config()["loss"]
    lc = CF.merge(lc, {"weights": {"center": 0.5, "orient": 0.2, "distance": 0.1, "topology": 0.1, "endpoint": 0.1}})
    b = _batch(torch, pos=pos, weight=weight)
    out = {k: torch.randn(2, n, 32, 32, requires_grad=True) for k, n in
           (("seg", 1), ("center", 1), ("orient", 2), ("dist", 1), ("ep", 1), ("jn", 1))}
    loss, parts = LS.compute(out, b, lc)
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in parts.values())
    loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in out.values())


def test_axial_orientation_loss_is_axial():
    torch = pytest.importorskip("torch")
    from gems_bench.p7.losses import axial_orientation

    th = torch.tensor(0.3)
    c, s = torch.cos(2 * th).view(1, 1, 1, 1), torch.sin(2 * th).view(1, 1, 1, 1)
    v = torch.ones(1, 1, 1, 1)
    same = axial_orientation(torch.cat([c, s], 1), c, s, v)
    flipped = axial_orientation(torch.stack([torch.cos(2 * (th + np.pi)), torch.sin(2 * (th + np.pi))]).view(1, 2, 1, 1),
                                c, s, v)
    ortho = axial_orientation(torch.cat([-c, -s], 1), c, s, v)
    assert float(same) < 1e-5 and float(flipped) < 1e-5 and abs(float(ortho) - 2.0) < 1e-5


# ----------------------------------------------------------------------------------------------- augmentation
def _line_mask(H=64, angle_deg=30.0):
    from skimage.draw import line

    m = np.zeros((H, H), bool)
    t = np.radians(angle_deg)
    r0, c0 = H // 2 - int(20 * np.sin(t)), H // 2 - int(20 * np.cos(t))
    r1, c1 = H // 2 + int(20 * np.sin(t)), H // 2 + int(20 * np.cos(t))
    rr, cc = line(r0, c0, r1, c1)
    m[rr, cc] = True
    return m


@pytest.mark.parametrize("angle", [0.0, 45.0])             # exactly digitisable lines: cos (0) and sin (45) signs
@pytest.mark.parametrize("op", ["rot90", "rot180", "rot270", "flip_lr", "flip_ud"])
def test_orientation_targets_follow_geometric_transforms(op, angle):
    from gems_bench.p6.boost import axial_orientation

    m = _line_mask(angle_deg=angle)
    th = axial_orientation(m)
    c, s = np.cos(2 * th), np.sin(2 * th)
    x = np.stack([m.astype(np.float32)] * 2)
    xo, t = AU.apply_geometric(x, {"ocos": c, "osin": s, "seg": m.astype(np.float32)}, op)
    th2 = axial_orientation(t["seg"] > 0.5)
    on = (t["seg"] > 0.5) & np.isfinite(th2) & np.isfinite(t["ocos"])
    assert on.sum() > 10
    assert np.allclose(t["ocos"][on], np.cos(2 * th2[on]), atol=0.15)
    assert np.allclose(t["osin"][on], np.sin(2 * th2[on]), atol=0.15)
    assert np.array_equal(xo[0] > 0.5, t["seg"] > 0.5)          # inputs and targets stay aligned


def test_photometric_keeps_footprint_and_alignment():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 16, 16)).astype(np.float32)
    x[4] = 1.0
    valid = np.ones((16, 16), bool)
    y, ops = AU.photometric(x, valid, {"a": [0, 1], "b": [2, 3, 4]}, AU.PRESETS["structural"], rng, protect=[4])
    assert np.array_equal(y[4], x[4]) and y.shape == x.shape
    assert "scale" in ops and "noise" in ops


def test_conservative_preset_draws_only_rot180():
    rng = np.random.default_rng(1)
    ops = {o for _ in range(200) for o in AU.sample_ops(rng, AU.PRESETS["conservative"])}
    assert ops <= {"rot180"}


# ----------------------------------------------------------------------------------------------- leakage guards
def _tu(kind="cv"):
    from gems_bench.m9.dataset import TrainUnit

    H = W = 40
    allowed = np.zeros((H, W), bool)
    allowed[:, :20] = True
    inner = np.zeros((H, W), bool)
    inner[:10, 20:30] = True
    pred = np.zeros((H, W), bool)
    pred[:, 30:] = True
    lab = np.zeros((H, W), bool)
    lab[20, 2:18] = True
    lab[20, 32:38] = True
    return TrainUnit("geo_f0", kind, 0, allowed, allowed.copy(), lab & allowed, inner, pred), lab


def test_targets_derive_from_visible_labels_only():
    tu, lab = _tu()
    t = {"seg": tu.visible.astype(np.uint8), "weight": tu.allowed.astype(np.float16),
         "center": TG.centerline_target(tu.visible, 1)}
    r = TG.audit_targets(tu, t)
    assert r["ok"] and r["center_outside_visible_px"] == 0
    t["seg"] = lab.astype(np.uint8)                    # held-out fold label leaked into a target
    assert not TG.audit_targets(tu, t)["ok"]
    d = TG.distance_target(tu.visible, 10)
    assert float(d[20, 10]) == 0.0 and float(d.max()) == 1.0
    assert TG.distance_target(np.zeros((4, 4), bool)).min() == 1.0


def test_overlap_audit_flags_heldout_visibility():
    tu, lab = _tu()
    stub = SimpleNamespace(cfg={"data": {"patch_px": 8}}, ctx=None)
    ok = DA.overlap_audit(stub, [tu])
    assert bool(ok.ok.iloc[0])
    bad = copy.copy(tu)
    bad.visible = lab
    assert not bool(DA.overlap_audit(stub, [bad]).ok.iloc[0])


def test_channel_contract_and_label_free_allow_list():
    reg = DA.channel_registry(["raw:rtp", "raw:iso_grav_anom", "m9:footprint"],
                              [{"name": "map:B0_lineament", "source": "label_free_map", "label_free": True,
                                "full_region": True}], CF.load_p7_config()["inputs"]["modality_patterns"])
    assert list(reg.modality[:3]) == ["magnetic", "gravity", "footprint"]
    DA.assert_contract(reg)
    g = DA.modality_groups(reg)
    assert sum(len(v) for v in g.values()) == 4
    bad = pd.concat([reg, pd.DataFrame([{"index": 4, "name": "oof:P5_B0", "modality": "other", "source": "oof",
                                         "label_free": False, "full_region": True}])])
    with pytest.raises(DA.LeakageError):
        DA.assert_contract(bad)
    cfg = CF.load_p7_config({"inputs": {"label_free_maps": {"enabled": True, "allow": ["P5_B0_gated"]}}})
    with pytest.raises(DA.LeakageError):
        DA.extra_specs(cfg)


# ----------------------------------------------------------------------------------------------- checkpoints
def test_inner_status_and_selection_rule():
    from gems_bench.p7.train import better, discovery_epochs, inner_status

    assert inner_status(None, None, False) == "NOT_CONFIGURED"
    assert inner_status(None, 0, True) == "NO_ELIGIBLE_VALIDATION"
    assert inner_status(float("nan"), 50, True) == "INVALID_NUMERIC"
    assert inner_status(0.2, 50, True) == "VALID"
    assert not better("INVALID_NUMERIC", float("nan"), -np.inf)
    assert not better("NO_ELIGIBLE_VALIDATION", 0.9, 0.1)
    assert better("VALID", 0.3, 0.2)
    recs = [{"kind": "cv", "best_epoch": 8, "selection": "best_valid_inner"},
            {"kind": "cv", "best_epoch": 12, "selection": "best_valid_inner"},
            {"kind": "cv", "best_epoch": 3, "selection": "last_epoch (no VALID inner epoch: flagged)"}]
    assert discovery_epochs(recs, 5, 20) == 10
    assert discovery_epochs([], 5, 20) == 20


def test_checkpoint_resume_refuses_other_spec(tmp_path):
    pytest.importorskip("torch")
    from gems_bench.training.checkpoint import CheckpointManager

    a = CheckpointManager(tmp_path, "geo_f0", spec_hash="aaa", policy="best_and_latest")
    a.save({"model": {}, "epoch": 1, "best": 0.1, "best_ep": 1, "best_state": None}, 1)
    assert a.latest()["spec_hash"] == "aaa"
    b = CheckpointManager(tmp_path, "geo_f0", spec_hash="bbb", policy="best_and_latest")
    with pytest.raises(RuntimeError):
        b.latest()


# ----------------------------------------------------------------------------------------------- evaluation
def _grid():
    rows = []
    for z in (0, 1, 2):
        for c, bl, sys in (("p7_T1", 1.0, "1;2;3"), ("w2", 1.0, "1;2"), ("w2", 0.5, "1"), ("cnn_b0", np.nan, "2")):
            for rep in ("crest_1px",):
                for f in (0.01, 0.03):
                    tp = 10.0 + z + (5 if c == "p7_T1" else 0) + 10 * f
                    rows.append({"task": "disc_official", "zone": z, "candidate": c, "blend": bl, "rep": rep,
                                 "frac": f, "tp_w": tp, "fp_w": 100.0, "fn_w": 40.0 - tp, "n_truth": 40.0,
                                 "n_systems": 4, "n_recovered": len(sys.split(";")), "systems": sys,
                                 "area_frac_eligible": f, "length_km": 1.0, "n_segments": 3, "short_share": 0.1,
                                 "area_px": 100})
                    rows.append({**rows[-1], "task": "geo", "systems": None, "n_systems": np.nan,
                                 "n_recovered": np.nan})
    return pd.DataFrame(rows)


def test_recall_table_counts_new_systems_against_reference():
    T = EV.recall_table(_grid(), ["p7_T1"], [0.01, 0.03], "disc_official",
                        {"w2": 1.0, "cnn_b0": None})
    p = T[(T.candidate == "p7_T1") & np.isclose(T.frac, 0.03)].iloc[0]
    assert p.systems_recovered == 9 and p.systems_total == 12
    assert p.new_vs_w2 == 3 and p.new_vs_cnn_b0 == 6


def test_policy_summary_is_nested_and_complete():
    ev = CF.load_p7_config()["evaluation"]
    s, P = EV.policy_summary(_grid(), "p7_T1", ev, "disc_official")
    assert np.isfinite(s["disc_dti"]) and np.isfinite(s["geo_dti"])
    assert set(P.zone) == {0, 1, 2} and s["system_recall"] == pytest.approx(0.75)


def test_hypothesis_and_promotion_verdicts():
    a = {"disc_dti": 0.26, "disc_worst": 0.20}
    b = {"disc_dti": 0.25, "disc_worst": 0.21}
    assert EV.hypothesis_verdict(a, b)["verdict"] == "SUPPORTED"
    assert EV.hypothesis_verdict({"disc_dti": 0.252, "disc_worst": 0.2}, b)["verdict"] == "INCONCLUSIVE"
    assert EV.hypothesis_verdict(None, b)["verdict"] == "INCOMPLETE"
    pc = CF.load_p7_config()["promotion"]
    ok = {k: True for k in ("no_leakage", "schema_compatible", "parity", "deployable_policy", "export_valid")}
    w2 = {"disc_dti": 0.22, "disc_worst": 0.19}
    good = {"disc_dti": 0.31, "disc_worst": 0.20, "geo_dti": 0.18}
    assert EV.promotion_verdict(good, w2, pc, 0, ok)["decision"] == "PROMOTE"
    low_geo = {**good, "geo_dti": 0.11}
    assert EV.promotion_verdict(low_geo, w2, pc, 0, ok)["decision"] == "REJECT"
    assert EV.promotion_verdict(low_geo, w2, pc, 2, ok)["decision"] == "AUXILIARY"
    assert EV.promotion_verdict(good, w2, pc, 0, {**ok, "parity": False})["decision"] == "REJECT"


# ----------------------------------------------------------------------------------------------- W2 schema
def test_p7_w2_schema_extends_frozen_list_and_checks_provenance():
    sets = W2.feature_lists(["cnn_b0", "cnn_b1", "mag_hgrad"], {"p7_fault", "p7_center", "x"},
                            ["p7_fault", "p7_center", "p7_orient_coh"])
    assert sets["w2_ctrl"] == ["cnn_b0", "cnn_b1", "mag_hgrad"]
    assert sets["w2_p7"][:3] == sets["w2_ctrl"] and sets["w2_p7"][3:] == ["p7_fault", "p7_center"]
    u = SimpleNamespace(task="disc_official", zone=1, key="disc_official:1")
    W2.assert_p7_provenance({"protocol_key": "k", "unit": "disc_s1", "labels_stored": False}, u, "k")
    with pytest.raises(W2.LeakageError):
        W2.assert_p7_provenance({"protocol_key": "k", "unit": "disc_s0", "labels_stored": False}, u, "k")
    with pytest.raises(W2.LeakageError):
        W2.assert_p7_provenance({"protocol_key": "other", "unit": "disc_s1", "labels_stored": False}, u, "k")


# ----------------------------------------------------------------------------------------------- tracer
TC = CF.load_p7_config()["tracer"]


def _straight(step=2.0, p_stop=0.0):
    return lambda pos, head, t: (head, step, p_stop)


def test_tracer_stops_at_boundary_known_gap_and_length():
    H = W = 60
    inside = np.ones((H, W), bool)
    ev = np.ones((H, W), np.float32)
    p, why, _ = TR.rollout(_straight(), np.array([30.0, 30.0]), np.array([0.0, 1.0]), ev, inside, None,
                           {**TC, "max_length_px": 1000, "max_steps": 1000})
    assert why == "boundary" and p[-1, 1] < W
    known = np.zeros((H, W), bool)
    known[:, 40] = True
    assert TR.rollout(_straight(1.0), np.array([30.0, 30.0]), np.array([0.0, 1.0]), ev, inside, known, TC)[1] == "known"
    weak = ev.copy()
    weak[:, 36:] = 0.0
    p, why, unsup = TR.rollout(_straight(1.0), np.array([30.0, 30.0]), np.array([0.0, 1.0]), weak, inside, None, TC)
    assert why == "gap" and p[-1, 1] < 36 + 1 and unsup > 0
    p, why, _ = TR.rollout(_straight(2.0), np.array([30.0, 5.0]), np.array([0.0, 1.0]), ev, inside, None,
                           {**TC, "max_length_px": 10})
    assert why == "max_length"
    assert TR.rollout(_straight(p_stop=0.9), np.array([30.0, 30.0]), np.array([0.0, 1.0]), ev, inside, None, TC)[1] \
        == "model_stop"


def test_tracer_detects_loops():
    ev = np.ones((50, 50), np.float32)
    inside = np.ones((50, 50), bool)
    ang = {"a": 0.0}

    def circle(pos, head, t):
        ang["a"] += np.pi / 3
        return np.array([np.sin(ang["a"]), np.cos(ang["a"])]), 2.0, 0.0

    _, why, _ = TR.rollout(circle, np.array([25.0, 25.0]), np.array([0.0, 1.0]), ev, inside, None,
                           {**TC, "loop_radius_px": 1.5, "max_steps": 200})
    assert why == "loop"


def test_skeleton_paths_resample_and_geometric_link():
    m = np.zeros((40, 40), bool)
    m[20, 2:15] = True
    m[20, 20:35] = True
    paths = TR.skeleton_paths(m)
    assert len(paths) == 2
    q = TR.resample(paths[0], 2.0)
    assert np.allclose(np.diff(q[:, 1]).__abs__(), 2.0, atol=1e-5)
    add = TR.geometric_link(m, np.ones_like(m), 8, 20)
    assert add[20, 15:20].all() and not (add & m).any()
    drop = np.zeros_like(m)
    drop[:, 25:] = True
    tr = TR.trajectory_set(paths, 1.0, 3, drop)
    assert all((np.rint(t[:, 1]) < 25).all() for t in tr)


# ----------------------------------------------------------------------------------------------- stitching
def test_full_region_stitching_is_seam_free(tmp_path):
    torch = pytest.importorskip("torch")
    from gems_bench.m9.dataset import FoldScaler
    from gems_bench.p7.infer import full_region, predict_mask

    class Pointwise(torch.nn.Module):
        head_keys = ["seg"]

        def forward(self, x, m=None, xc=None):
            return {"seg": x[:, :1]}

    rng = np.random.default_rng(0)
    cube = rng.standard_normal((2, 90, 110)).astype(np.float32)
    valid = np.ones((90, 110), bool)
    valid[:5] = False
    sc = FoldScaler(np.zeros(2, np.float32), np.ones(2, np.float32), 100.0, ["a", "b"], 1)
    kw = dict(P=32, stride=16, device=torch.device("cpu"), amp=False, amp_dtype=None, batch=8)
    out = full_region([(Pointwise(), sc, None)], cube, None, valid, tmp_path, tile_px=40, halo_px=8, log=lambda m: None,
                      **kw)
    expect = 1.0 / (1.0 + np.exp(-cube[0]))
    assert np.allclose(out["seg"][valid], expect[valid], atol=2e-3)
    assert np.isnan(out["seg"][~valid]).all()
    again = full_region([(Pointwise(), sc, None)], cube, None, valid, tmp_path, tile_px=40, halo_px=8,
                        log=lambda m: None, **kw)                  # resumes from saved tiles
    assert np.allclose(again["seg"][valid], out["seg"][valid])
    pm = predict_mask([(Pointwise(), sc, None)], cube, None, valid, **kw)
    assert np.allclose(pm["seg"][valid], expect[valid], atol=1e-5)


# ----------------------------------------------------------------------------------------------- export
def test_export_recipe_single_selection_contract():
    from gems_bench.p6 import export_parity as EP

    pol = {"selection_id": "p7_selection", "space": "p7_T1", "objective": "disc", "rep": "crest_1px", "frac": 0.015,
           "blend": 1.0}
    rec = EP.build_recipe(candidate="p7_T1", kind="p7_zone_stitched", policy=pol,
                          model={"experiment_id": "p7_t1_x", "features": ["a", "b"]}, sources={"geo:0": "sha"},
                          feature_schema=["a", "b"], preprocessing={"eligibility": "eligible", "known": "masked"})
    assert EP.plan_audit(rec)["status"] == "PASS"
    with pytest.raises(EP.PolicyMismatch):
        EP.build_recipe(candidate="p7_T1", kind="k", policy={**pol, "frac_source": "geo"}, model={"features": ["a"]},
                        sources={}, feature_schema=["a"], preprocessing={})
