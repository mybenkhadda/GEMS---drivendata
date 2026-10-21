"""Phase-5 scientific components on synthetic inputs (no project data, CPU only)."""
from types import SimpleNamespace

import numpy as np
import pytest

from gems_bench.calibration.area_policy import top_fraction
from gems_bench.dem.inventory import parse_url
from gems_bench.evaluation.parity import parity_cases, reference_dti, run_parity
from gems_bench.geology.orientation import axial_angle, axial_difference, axial_vector


def test_local_metric_matches_independent_reference():
    t = run_parity()
    assert t.pass_reference.all(), t[~t.pass_reference]


def test_kernel_boundary_is_zero_at_radius():
    c = {x["case"]: x for x in parity_cases()}["offset_3px"]
    r = reference_dti(c["pred"], c["truth"])
    assert r["tp_w"] == pytest.approx(0.0)                          # k(300 m) = 0


def test_axial_algebra():
    c, s = axial_vector(np.array([10.0, 190.0]))
    assert np.allclose(c[0], c[1]) and np.allclose(s[0], s[1])      # θ and θ + 180° identical
    assert axial_difference(5.0, 175.0) == pytest.approx(10.0)
    assert axial_angle(*axial_vector(np.array([37.0])))[0] == pytest.approx(37.0, abs=1e-4)


def test_top_fraction_area():
    p = np.arange(100, dtype=float).reshape(10, 10)
    v = np.ones_like(p, bool)
    q = top_fraction(p, v, 0.1)
    assert q.sum() == 10 and q[9].sum() == 10


def test_dem_url_parsing():
    u = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/NV_WesternNV_2021_D21/TIFF/"
         "USGS_1M_11_x30y435_NV_WesternNV_2021_D21.tif")
    r = parse_url(u)
    assert r["parse_ok"] and r["zone"] == 11 and r["tile_id"] == "11_x30y435"
    assert r["project"] == "NV_WesternNV_2021_D21" and r["west"] == 300_000 and r["north"] == 4_350_000


def test_profile_step_detects_a_fault_like_step():
    from gems_bench.features.profiles import profile_output

    H = W = 40
    S = np.zeros((H, W), np.float32)
    S[:, 21:] = 1.0                                                  # N-S step centred on column 20 (strike 0°)
    S[:, 20] = 0.5
    v = {0: S}
    for k in range(1, 4):
        v[k] = np.roll(S, -k, axis=1)
        v[-k] = np.roll(S, k, axis=1)
    sm = {"v": v, "loc_std": np.full_like(S, 0.5)}
    step = profile_output(sm, "step", 3)
    asym = profile_output(sm, "asymmetry", 3)
    sign = profile_output(sm, "sign_change", 3)
    assert step[20, 20] > step[20, 5]
    assert asym[20, 20] > 0.99                                       # pure odd profile: step / offset-like
    assert sign[20, 20] < -0.99                                      # monotonic ramp, not a ridge


torch = pytest.importorskip("torch")


def test_nnpu_reduces_to_tversky_without_prior():
    from gems_bench.losses.pu import nnpu_dti_surrogate
    from gems_bench.p3.cnn3 import dti_surrogate

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 1, 32, 32, generator=g)
    truth = torch.zeros(2, 1, 32, 32)
    truth[:, :, 16, 4:28] = 1
    kern = truth.clone()
    w = torch.ones_like(truth)
    a = nnpu_dti_surrogate(logits, truth, kern, w, prior=0.0, bce_weight=0.0)
    b = dti_surrogate(logits, truth, kern, w, bce_weight=0.0)
    assert torch.allclose(a, b, atol=1e-6)
    hi = nnpu_dti_surrogate(logits, truth, kern, w, prior=0.05, bce_weight=0.0)
    assert hi <= a + 1e-6                                           # forgiving hidden positives lowers FP


def test_geostructurenet_parity_shapes_and_capacity():
    from gems_bench.geological_encoders import GeoUNet, count_params
    from gems_bench.models.geostructurenet import GeoStructureNet

    groups = [[0, 1, 2], [3, 4]]
    ref = GeoUNet(groups, 5, enc_width=8, dec_width=8, fusion="gated", depth=3)
    gsn = GeoStructureNet(groups, 5, enc_width=8, dec_width=8, fusion="gated", depth=3)
    assert count_params(ref) == count_params(gsn)                   # parity configuration = Phase-4 gated encoder
    x = torch.randn(2, 5, 64, 64)
    for fusion in ("attention", "crossscale", "concat"):
        for lr in ("none", "dilated", "fourier", "pyramid"):
            m = GeoStructureNet(groups, 5, enc_width=8, dec_width=8, fusion=fusion, long_range=lr, depth=3,
                                aux_dist=True, aux_orient=True)
            out = m(x)
            assert out["seg"].shape == (2, 1, 64, 64) and out["orient"].shape == (2, 2, 64, 64)
    m = GeoStructureNet(groups, 5, enc_width=8, dec_width=8, context_idx=[4], depth=3)
    assert m(x, torch.randn(2, 5, 64, 64))["seg"].shape == (2, 1, 64, 64)
    assert 0 < m.context_weight() < 0.2                              # context starts nearly switched off


def test_geometry_losses_finite_and_axial():
    from gems_bench.losses import geometry as G

    p = torch.rand(1, 1, 32, 32, requires_grad=True)
    t = torch.zeros(1, 1, 32, 32)
    t[..., 16, :] = 1
    w = torch.ones_like(t)
    o = torch.randn(1, 2, 32, 32, requires_grad=True)
    for v in (G.cl_dice(p, t, t, w), G.thinness(p, w), G.continuity(p, o, w)):
        assert torch.isfinite(v)
    c, s = torch.ones_like(t), torch.zeros_like(t)
    perfect = torch.cat([c, s], 1) * 5
    assert G.orientation_loss(perfect, c, s, w) < 0.01
    assert G.orientation_loss(-perfect, c, s, w) > 1.9              # 90° axial error -> maximal loss


def test_trainer_state_roundtrip(tmp_path):
    from gems_bench.training.checkpoint import CheckpointManager

    ck = CheckpointManager(tmp_path, "cv_fold0", keep=2, spec_hash="h")
    for ep in (1, 2, 3):
        ck.save({"epoch": ep, "model": {"w": torch.ones(1) * ep}}, ep)
    assert len(ck.resume_files()) == 2
    assert ck.latest()["epoch"] == 3
    with pytest.raises(RuntimeError):
        CheckpointManager(tmp_path, "cv_fold0", spec_hash="other").latest()


def test_background_weights_label_free_guard():
    from gems_bench.losses.pu import background_weights, resolve_evidence

    ev = np.random.default_rng(0).random((20, 20)).astype(np.float32)
    w = background_weights(ev, np.ones((20, 20), bool), lam=0.5, q0=0.8, w_min=0.2)
    assert w.min() >= 0.5 - 1e-6 and w.max() == 1.0
    with pytest.raises(ValueError):
        resolve_evidence(SimpleNamespace(), "map:P4_B_gated")         # supervised maps are refused
