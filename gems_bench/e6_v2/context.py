"""Coarse pooled context branch and the dilated bottleneck.

Context input: the 8 x 8 mean of the normalised terrain + field stacks (``datasets/context.npy``), cropped to
``context_extent`` x the patch extent around the patch (4 x 25.6 km = 102 km for 256 px patches) - one pooled
pixel per bottleneck pixel. A small dilated network reads it and the central crop that lies under the patch is
returned at bottleneck resolution.
"""
from __future__ import annotations

import torch
from torch import nn

from .encoders import ResBlock, gn


class ContextBranch(nn.Module):
    def __init__(self, cin: int, width: int, extent: int):
        super().__init__()
        self.extent = extent
        self.net = nn.Sequential(nn.Conv2d(cin, width, 3, padding=1, bias=False), gn(width), nn.SiLU(),
                                 ResBlock(width, width, 1), ResBlock(width, width, 2), ResBlock(width, width, 4))

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        y = self.net(ctx)
        n = ctx.shape[-1] // self.extent
        o = (ctx.shape[-1] - n) // 2
        return y[..., o:o + n, o:o + n]


class Bottleneck(nn.Module):
    def __init__(self, width: int, ctx_width: int, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(width, width, d) for d in dilations])
        self.merge = nn.Sequential(nn.Conv2d(width + ctx_width, width, 1, bias=False), gn(width), nn.SiLU())

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.merge(torch.cat([self.blocks(x), ctx], 1))
