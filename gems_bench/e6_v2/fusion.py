"""Gated fusion of the encoder branches at every level.

TwoWayGate (reference model):  w = softmax([s_T, s_F]),  fused = w_T P_T(t) + w_F P_F(f)
MaskedThreeWayGate (E6):       s_T, s_F as above, computed from (t, f) only;
                               s_D = conv([t, f, A(d)]) + log(clamp(m_l, eps)),  s_D = -inf where m_l == 0
                               w = softmax([s_T, s_F, s_D]),  fused = w_T P_T(t) + w_F P_F(f) + w_D P_D(A(d))
``m_l`` is the measured DEM coverage average-pooled to the level resolution. Where m_l == 0 the DEM weight is
exactly 0 and (s_T, s_F) are the reference logits, so the three-way gate equals the reference gate exactly
when initialised from it. The softmax runs in float32 under AMP.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-4


class TwoWayGate(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj_t = nn.Conv2d(width, width, 1)
        self.proj_f = nn.Conv2d(width, width, 1)
        self.logit_tf = nn.Conv2d(2 * width, 2, 1)

    def forward(self, t, f):
        w = torch.softmax(self.logit_tf(torch.cat([t, f], 1)).float(), 1).to(t.dtype)
        return w[:, :1] * self.proj_t(t) + w[:, 1:2] * self.proj_f(f), w


class MaskedThreeWayGate(nn.Module):
    def __init__(self, width: int, dem_width: int, dem_bias: float):
        super().__init__()
        self.proj_t = nn.Conv2d(width, width, 1)
        self.proj_f = nn.Conv2d(width, width, 1)
        self.logit_tf = nn.Conv2d(2 * width, 2, 1)
        self.adapt_d = nn.Conv2d(dem_width, width, 1)
        self.proj_d = nn.Conv2d(width, width, 1)
        self.logit_d = nn.Conv2d(3 * width, 1, 1)
        nn.init.zeros_(self.logit_d.weight)
        nn.init.constant_(self.logit_d.bias, dem_bias)

    def forward(self, t, f, d, m):
        a = self.adapt_d(d)
        s_tf = self.logit_tf(torch.cat([t, f], 1)).float()
        s_d = self.logit_d(torch.cat([t, f, a], 1)).float() + torch.log(m.float().clamp_min(EPS))
        s_d = torch.where(m > 0, s_d, torch.full_like(s_d, float("-inf")))
        w = torch.softmax(torch.cat([s_tf, s_d], 1), 1)
        w = w.to(t.dtype)
        fused = w[:, :1] * self.proj_t(t) + w[:, 1:2] * self.proj_f(f) + w[:, 2:3] * self.proj_d(a)
        return fused, w

    @torch.no_grad()
    def load_from_two_way(self, g: TwoWayGate):
        for name in ("proj_t", "proj_f", "logit_tf"):
            getattr(self, name).load_state_dict(getattr(g, name).state_dict())


def pool_mask(m: torch.Tensor, level: int) -> torch.Tensor:
    return m if level == 0 else F.avg_pool2d(m, 2 ** level)
