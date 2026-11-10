"""Lightweight DEM patch encoder (native-resolution terrain branch). Input: (B, 1+k, H, W) normalised elevation
(+ optional derived channels) and a validity mask; masked pixels are zeroed and the mask is a second channel so
the network can tell missing terrain from flat terrain."""
from __future__ import annotations

import torch
from torch import nn


class TerrainEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, channels=(16, 32, 64), embed_dim: int = 64):
        super().__init__()
        layers, c = [], in_channels + 1
        for co in channels:
            layers += [nn.Conv2d(c, co, 3, 2, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                       nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
            c = co
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv2d(c, embed_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, dem, valid):
        x = torch.cat([dem * valid, valid], 1)
        return self.proj(self.body(x))           # (B, embed_dim, H / 2^k, W / 2^k)
