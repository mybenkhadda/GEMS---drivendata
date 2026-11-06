"""Output heads. Only ``fault`` is used at inference; proximity and orientation are training-time auxiliaries.

fault        1 logit
proximity    sigmoid -> min(d, 300 m) / 300 m
orientation  tanh    -> (cos 2t, sin 2t) of the nearest mapped strike (cosine loss within 300 m)
"""
from __future__ import annotations

import torch
from torch import nn


class Heads(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fault = nn.Conv2d(width, 1, 1)
        self.proximity = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 1, 1))
        self.orientation = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 2, 1))
        nn.init.constant_(self.fault.bias, -4.0)       # start near the low fault prior

    def forward(self, x: torch.Tensor, aux: bool = True) -> dict:
        out = {"logit": self.fault(x)}
        if aux:
            out["proximity"] = torch.sigmoid(self.proximity(x))
            out["orientation"] = torch.tanh(self.orientation(x))
        return out
