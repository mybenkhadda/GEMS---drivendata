"""Spatial ConvGRU refinement (H3): iterative, weight-shared refinement of a fused feature map. Not a temporal model.

    h_0 = tanh(W_0 * x)                                  learned state initialisation from the input map
    z_t = sigma(W_z * [x, h_{t-1}]),  r_t = sigma(W_r * [x, h_{t-1}])
    h~_t = tanh(W_h * [x, r_t . h_{t-1}]),  h_t = (1 - z_t) . h_{t-1} + z_t . h~_t
    y = x + P(h_T)                                       residual integration; P is zero-initialised

Every step widens the receptive field by the kernel, so weak / fragmented lineament responses can borrow support
along the trace. ``steps = 0`` returns ``x`` unchanged (exact bypass = the non-recurrent model), and because ``P`` starts
at zero the recurrent model also starts as that model. ``checkpoint = True`` recomputes each step in the backward
pass (gradient checkpointing). ``capture = True`` keeps every h_t (CPU copies) for the refinement figure.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hidden: int, kernel: int = 3):
        super().__init__()
        p = kernel // 2
        self.gates = nn.Conv2d(in_ch + hidden, 2 * hidden, kernel, padding=p)
        self.cand = nn.Conv2d(in_ch + hidden, hidden, kernel, padding=p)
        self.hidden = hidden

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1.0 - z) * h + z * c


class ConvGRURefiner(nn.Module):
    def __init__(self, channels: int, hidden: int = 64, kernel: int = 3, steps: int = 3, checkpoint: bool = False):
        super().__init__()
        self.steps, self.checkpoint = int(steps), bool(checkpoint)
        self.init = nn.Conv2d(channels, hidden, 1)
        self.cell = ConvGRUCell(channels, hidden, kernel)
        self.out = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.capture = False
        self.states: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        n = self.steps if steps is None else int(steps)
        self.states = []
        if n <= 0:
            return x
        h = torch.tanh(self.init(x))
        for _ in range(n):
            if self.checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint

                h = checkpoint(self.cell, x, h, use_reentrant=False)
            else:
                h = self.cell(x, h)
            if self.capture:
                self.states.append(h.detach().float().cpu())
        return x + self.out(h)
