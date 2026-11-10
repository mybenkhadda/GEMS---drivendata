"""Two-branch multiresolution fusion: 100 m geophysics encoder x native DEM terrain encoder.

The DEM branch is resampled to the 100 m patch grid (adaptive average pooling over the same ground footprint)
and multiplied by the per-sample availability ``modality[:, 1]`` - where no 1 m DEM exists the model reduces
exactly to its geophysics path (``geo_only`` behaviour), so it works everywhere.
Modes: ``geo_only`` | ``dem_only`` | ``concat`` | ``gated`` | ``cross_attention``.
Heads: fault logit (1), optional auxiliary orientation (cos 2a, sin 2a) and distance.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .terrain_encoder import TerrainEncoder

MODES = ("geo_only", "dem_only", "concat", "gated", "cross_attention")


class GeoEncoder(nn.Module):
    def __init__(self, cin, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, dim // 2, 3, 1, 1), nn.BatchNorm2d(dim // 2), nn.ReLU(inplace=True),
                                 nn.Conv2d(dim // 2, dim, 3, 1, 2, dilation=2), nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
                                 nn.Conv2d(dim, dim, 3, 1, 4, dilation=4), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class GatedFusionNet(nn.Module):
    def __init__(self, geo_channels: int, mode: str = "gated", geo_dim: int = 128, dem_dim: int = 64,
                 dem_channels: int = 1, aux: bool = True, heads: int = 4):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode {mode} not in {MODES}")
        self.mode = mode
        self.geo = GeoEncoder(geo_channels, geo_dim)
        self.dem = TerrainEncoder(dem_channels, embed_dim=dem_dim)
        self.dem_proj = nn.Conv2d(dem_dim, geo_dim, 1)
        self.concat = nn.Conv2d(2 * geo_dim, geo_dim, 1)
        self.gate = nn.Sequential(nn.Conv2d(2 * geo_dim + 1, geo_dim, 1), nn.Sigmoid())
        self.attn = nn.MultiheadAttention(geo_dim, heads, batch_first=True)
        self.head = nn.Conv2d(geo_dim, 1, 1)
        self.aux = aux
        if aux:
            self.orient = nn.Conv2d(geo_dim, 2, 1)
            self.dist = nn.Conv2d(geo_dim, 1, 1)

    def forward(self, x, dem=None, dem_valid=None, modality=None):
        g = self.geo(x)
        b, _, h, w = g.shape
        if dem is None or self.mode == "geo_only":
            f = g
        else:
            has = (modality[:, 1] if modality is not None else torch.ones(b, device=x.device)).view(b, 1, 1, 1)
            d = self.dem_proj(F.adaptive_avg_pool2d(self.dem(dem, dem_valid), (h, w))) * has
            if self.mode == "dem_only":
                f = d
            elif self.mode == "concat":
                f = self.concat(torch.cat([g, d], 1))
            elif self.mode == "gated":
                a = self.gate(torch.cat([g, d, has.expand(b, 1, h, w)], 1)) * has
                f = g + a * d
            else:   # cross_attention: geo tokens attend to DEM tokens at 1/4 resolution; no DEM -> geo only
                q = g.flatten(2).transpose(1, 2)
                kv = F.adaptive_avg_pool2d(d, (max(h // 4, 1), max(w // 4, 1))).flatten(2).transpose(1, 2)
                att, _ = self.attn(q, kv, kv)
                f = g + att.transpose(1, 2).reshape(b, -1, h, w) * has
        out = {"logits": self.head(f)}
        if self.aux:
            out["orientation"] = torch.tanh(self.orient(f))
            out["distance"] = F.softplus(self.dist(f))
        return out
