"""Residual encoders: 4 levels at 1, 1/2, 1/4, 1/8 of the patch (256 -> 128 -> 64 -> 32 px for 256 px patches),
widths w, 2w, 4w, 8w. GroupNorm (batch statistics are unreliable at batch 8 and differ between train / eval)."""
from __future__ import annotations

import torch
from torch import nn


def gn(c: int) -> nn.GroupNorm:
    g = 8
    while c % g:
        g //= 2
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False)
        self.n1 = gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False)
        self.n2 = gn(cout)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return self.act(y + self.skip(x))


class Encoder(nn.Module):
    def __init__(self, cin: int, width: int, levels: int = 4):
        super().__init__()
        self.widths = [width * 2 ** i for i in range(levels)]
        self.stem = nn.Sequential(nn.Conv2d(cin, self.widths[0], 3, padding=1, bias=False), gn(self.widths[0]),
                                  nn.SiLU())
        blocks = [ResBlock(self.widths[0], self.widths[0])]
        for i in range(1, levels):
            blocks.append(nn.Sequential(nn.Conv2d(self.widths[i - 1], self.widths[i], 3, stride=2, padding=1,
                                                  bias=False), gn(self.widths[i]), nn.SiLU(),
                                        ResBlock(self.widths[i], self.widths[i])))
        self.levels = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        feats = []
        for blk in self.levels:
            x = blk(x)
            feats.append(x)
        return feats
