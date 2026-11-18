"""G14.1 torch tests: objectives, gradient diagnostics, architectures, channel views.

WRITTEN BUT NEVER EXECUTED. Skipped automatically when torch is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gems_bench.g14_1 import losses as L  # noqa: E402


def _tg(B=2, H=32, pos=True):
    z = torch.zeros(B, H, H)
    p = z.clone()
    if pos:
        p[:, 16, 4:28] = 1
    return {"positive": p, "neg_weight": 1 - p, "loss_mask": torch.ones(B, H, H), "soft": p.clone(),
            "reliable_neg": 1 - p}


@pytest.mark.parametrize("name", L.OBJECTIVES)
@pytest.mark.parametrize("pos", [True, False])
def test_every_objective_is_finite_and_differentiable(name, pos):
    logit = torch.randn(2, 32, 32, requires_grad=True)
    terms, w = L.objective(name, logit, _tg(pos=pos), radius_px=3.0)
    tot = L.total(terms, w)
    assert torch.isfinite(tot)
    tot.backward()
    assert torch.isfinite(logit.grad).all()


def test_gradient_diagnostics_report_norms_and_conflicts():
    logit = torch.randn(2, 32, 32, requires_grad=True)
    terms, w = L.objective("g14_original", logit, _tg(), radius_px=3.0)
    d = L.diagnostics(terms, w, logit, _tg())
    assert any(k.startswith("gradnorm_") for k in d) and any(k.startswith("cos_") for k in d)
    assert d["empty_positive_batch"] is False


def test_kernel_radius_is_resolution_aware():
    assert L.kernel_radius_px(100.0) == pytest.approx(3.0)
    assert L.kernel_radius_px(50.0) == pytest.approx(6.0)


@pytest.mark.parametrize("arch", ["n1_large_rf", "n2_local_regional", "n5_directional", "n5_isotropic", "g14_local"])
def test_architectures_share_the_output_contract(arch):
    from gems_bench.g14_1 import architectures as A
    from test_g14_core import small_cfg     # tests/ is on sys.path under pytest's prepend import mode

    class View:
        modalities = ["magnetic", "gravity", "terrain"]
        n_channels = 6

        def modality_index(self):
            return {"magnetic": [0, 1], "gravity": [2, 3], "terrain": [4, 5]}

    cfg = A.g14_cfg_for(arch, small_cfg())
    m = A.build(arch, cfg, View()).eval()
    b = {"local": torch.randn(2, 6, 32, 32), "local_mask": torch.ones(2, 3, 32, 32),
         "regional": torch.randn(2, 6, 64, 64), "regional_mask": torch.ones(2, 3, 64, 64),
         "global": torch.randn(2, 6, 64, 64), "global_mask": torch.ones(2, 3, 64, 64)}
    with torch.no_grad():
        o = m(b)
    assert o["fault"].shape == (2, 1, 32, 32)


def test_directional_kernels_are_normalised_lines():
    from gems_bench.g14_1.architectures import _line_kernel

    for a in (0, 45, 90, 135):
        k = _line_kernel(9, a)
        assert float(k.sum()) == pytest.approx(1.0) and int((k > 0).sum()) >= 9
