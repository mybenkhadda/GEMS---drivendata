"""Consistency-regularised semi-supervision (Phase-5 Experiment B3) - run only if B1 / B2 justify it.

Mean-teacher: an EMA copy of the network predicts on a second view of the same patch; the student is
pulled towards it on *unlabeled* pixels only. Physically valid views on these inputs:

* integer translation (shifts the whole stack; the prediction is shifted back before comparison and the
  wrapped border is masked) - valid for every channel;
* small Gaussian noise on standardised inputs - valid for every channel;
* NOT 90° rotations or mirror flips: several channels are directional (hillshade azimuths, directional
  derivatives, Gabor orientations, strike-dependent features), so rotating the raster without rotating
  those quantities would create physically impossible inputs. The 180° rotation is already applied by the
  sampler (strike-preserving; ``rot180``).
"""
from __future__ import annotations

import copy


class EMATeacher:
    def __init__(self, model, decay: float = 0.99):
        import torch

        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._torch = torch

    def update(self, student) -> None:
        with self._torch.no_grad():
            for t, s in zip(self.model.state_dict().values(), student.state_dict().values()):
                if t.dtype.is_floating_point:
                    t.mul_(self.decay).add_(s.detach(), alpha=1 - self.decay)
                else:
                    t.copy_(s)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd)


def shifted_view(x, rng, max_shift: int = 8, noise: float = 0.05):
    """(view, (dy, dx)) with a random integer roll and input noise."""
    import torch

    dy, dx = (int(v) for v in rng.integers(-max_shift, max_shift + 1, size=2))
    v = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
    if noise:
        v = v + noise * torch.randn_like(v)
    return v, (dy, dx)


def consistency_loss(student_logits, teacher_logits, shift: tuple[int, int], truth, weight):
    """MSE between student p and the teacher's p rolled back, on unlabeled, non-wrapped pixels."""
    import torch

    dy, dx = shift
    pt = torch.roll(torch.sigmoid(teacher_logits.float()), shifts=(-dy, -dx), dims=(-2, -1))
    ps = torch.sigmoid(student_logits.float())
    H, W = ps.shape[-2:]
    m = torch.zeros_like(ps)
    m[..., abs(dy):H - abs(dy), abs(dx):W - abs(dx)] = 1.0
    m = m * (1 - truth) * weight
    return (((ps - pt.detach()) ** 2) * m).sum() / m.sum().clamp_min(1.0)


def rampup(epoch: int, length: int) -> float:
    """Sigmoid-shaped ramp-up (Laine & Aila) from 0 to 1 over ``length`` epochs."""
    import math

    if length <= 0:
        return 1.0
    t = min(max(epoch, 0), length) / length
    return float(math.exp(-5.0 * (1.0 - t) ** 2))
