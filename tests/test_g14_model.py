"""G14 torch tests: tensor shape contracts, fusion masks, decoder output, losses, checkpoint metadata, inference edges.

NEVER EXECUTED by the authoring session. Skipped automatically when torch is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gems_bench.g14 import losses as Lo  # noqa: E402
from gems_bench.g14.model import GeoFaultFormer, ModalityFusion, align_center  # noqa: E402

from test_g14_core import small_cfg  # noqa: E402  (tests/ is on sys.path under pytest's prepend import mode)


def _batch(B=2, C=6, M=3, L=32, R=64, G=64):
    return {"local": torch.randn(B, C, L, L), "local_mask": torch.ones(B, M, L, L),
            "regional": torch.randn(B, C, R, R), "regional_mask": torch.ones(B, M, R, R),
            "global": torch.randn(B, C, G, G), "global_mask": torch.ones(B, M, G, G)}


GROUPS = {"magnetic": [0, 1], "gravity": [2, 3], "terrain": [4, 5]}
MODS = ["magnetic", "gravity", "terrain"]


@pytest.mark.parametrize("enc", ["unet", "resnet", "convnext"])
@pytest.mark.parametrize("fusion", ["concat", "gated", "cross_attention"])
def test_forward_shapes(enc, fusion):
    cfg = small_cfg(model={"local_encoder": enc, "fusion": fusion, "heads": ("fault", "orientation", "connectivity")})
    m = GeoFaultFormer(cfg, GROUPS, MODS).eval()
    with torch.no_grad():
        out = m(_batch())
    assert out["fault"].shape == (2, 1, 32, 32)                     # output on the 100 m local grid
    assert out["orientation"].shape == (2, 2, 32, 32)
    assert torch.allclose(out["orientation"].norm(dim=1), torch.ones(2, 32, 32), atol=1e-3)
    assert out["connectivity"].shape == (2, 2, 32, 32)
    assert all(a.shape == (2, 1, 32, 32) for a in out["aux"])


def test_local_only_and_mit_variants():
    cfg = small_cfg(model={"regional_encoder": "none", "global_encoder": "none", "context_stages": (),
                           "bottleneck_cross_attention": False, "film_global": False})
    out = GeoFaultFormer(cfg, GROUPS, MODS)(_batch())
    assert out["fault"].shape == (2, 1, 32, 32)
    cfg = small_cfg(model={"regional_encoder": "mit"})
    out = GeoFaultFormer(cfg, GROUPS, MODS)(_batch())
    assert out["fault"].shape == (2, 1, 32, 32)


def test_gated_fusion_ignores_unavailable_modality():
    f = ModalityFusion(GROUPS, {m: i for i, m in enumerate(MODS)}, 8, 8, "gated").eval()
    x = torch.randn(1, 6, 16, 16)
    mk = torch.ones(1, 3, 16, 16)
    mk[:, 1] = 0                                                     # gravity unavailable everywhere
    _, d = f(x, mk)
    w = d["modality_weights"]
    assert torch.all(w[:, 1] == 0) and torch.allclose(w.sum(1), torch.ones(1, 16, 16), atol=1e-5)
    x2 = x.clone()
    x2[:, 2:4] = 1e3                                                 # garbage in the unavailable channels
    y1, _ = f(x, mk)
    y2, _ = f(x2, mk)
    assert torch.allclose(y1, y2, atol=1e-5)


def test_align_center_extracts_central_fraction():
    feat = torch.zeros(1, 1, 8, 8)
    feat[..., 2:6, 2:6] = 1.0
    y = align_center(feat, 0.5, (4, 4))
    assert torch.all(y > 0.99)


def test_losses_finite_with_empty_positives_and_masks():
    B, H = 2, 32
    logit = torch.randn(B, H, H, requires_grad=True)
    z = torch.zeros(B, H, H)
    tg = {"loss_mask": torch.ones(B, H, H), "trace": z, "neg_weight": torch.ones(B, H, H), "soft": z,
          "reliable_neg": torch.ones(B, H, H), "cos2": z, "sin2": z, "orient_w": z, "endpoint": z, "junction": z}
    cfg = small_cfg(loss={"aux_weights": {"orientation": 0.0, "connectivity": 0.0, "dti_surrogate": 0.1,
                                          "deep_supervision": 0.0}})
    total, logs = Lo.CompositeLoss(cfg)({"fault": logit[:, None]}, tg)
    assert torch.isfinite(total) and "soft_cldice" not in logs       # skipped without positives
    total.backward()
    assert torch.isfinite(logit.grad).all()
    tg["loss_mask"] = torch.zeros(B, H, H)                           # nothing supervised
    total, _ = Lo.CompositeLoss(cfg)({"fault": logit[:, None]}, tg)
    assert torch.isfinite(total)


def test_dti_surrogate_matches_official_on_binary():
    from gems_bench.metric import dti_components

    t = np.zeros((40, 40), bool)
    t[20, 5:35] = True
    for shift in (0, 1, 2):
        p = np.roll(t, shift, 0).astype(np.float32)
        ref = dti_components(p, t, crop=False).dti
        s = 1 - float(Lo.dti_surrogate(torch.from_numpy(p)[None], torch.from_numpy(t.astype(np.float32))[None],
                                       torch.ones(1, 40, 40)))
        assert abs(ref - s) < 1e-3, (shift, ref, s)


def test_soft_skeleton_thin_line():
    x = torch.zeros(1, 20, 20)
    x[0, 8:12, 2:18] = 1.0
    sk = Lo.soft_skeleton(x, 5)
    assert sk.sum() < x.sum() and sk[0, 9:11, 4:16].sum() > 0


def test_checkpoint_hash_refusal(tmp_path):
    from gems_bench.g14.train import Checkpoint

    ck = Checkpoint(tmp_path, "geo_1", "hashA", "schemaA")
    ck.save("latest", {"x": 1})
    assert Checkpoint(tmp_path, "geo_1", "hashA", "schemaA").load("latest")["x"] == 1
    with pytest.raises(RuntimeError):
        Checkpoint(tmp_path, "geo_1", "hashB", "schemaA").load("latest")


def test_tile_origins_cover_mask_and_hann_positive():
    from gems_bench.g14.inference import hann2d, tile_origins

    m = np.ones((100, 130), bool)
    o = tile_origins(m, 32, 4, 8)
    cov = np.zeros_like(m)
    for t, l in o:
        cov[max(t + 4, 0):t + 28, max(l + 4, 0):l + 28] = True
        assert (t + 16) % 8 == 0 and (l + 16) % 8 == 0                # centres on the lattice
    assert cov.all()
    assert (hann2d(32, 4) > 0).all()
