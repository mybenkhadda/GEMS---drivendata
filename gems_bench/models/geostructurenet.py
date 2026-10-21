"""GeoStructureNet: the Phase-4 gated dual encoder, extended in controlled, individually ablatable steps.

Starting point = ``geological_encoders.GeoUNet`` (P4_B_gated: terrain + subsurface encoders, gated fusion at
every level, shared decoder; geographic DTI 0.158, the best Phase-4 model). GeoStructureNet keeps its
Encoder / Decoder / GatedFuse blocks unchanged, so ``fusion="gated", long_range="none", context=None`` with
the same widths reproduces P4_B_gated's architecture (the parity control of Experiment E).

Ablation axes (one changed at a time; parameter counts matched with ``models.factory.match_width``)
---------------------------------------------------------------------------------------------------
branches     terrain (+ morphometry, optional DEM aggregates) | potential field (magnetic, gravity, basement,
             structural derivatives) | optional context (coarse regional strain / seismicity / geology)
fusion       ``concat`` | ``gated`` (Phase 4) | ``attention`` (per-pixel softmax over modalities, separately
             for ``heads`` channel groups) | ``crossscale`` (gated fusion that also sees every branch's next
             coarser level: deep / smooth potential-field sources inform shallow terrain detail)
long_range   ``none`` | ``dilated`` (residual 3x3 convs, dilation 1-2-4-8, at the bottleneck) |
             ``fourier`` (small spectral block: rfft2 -> 1x1 conv -> irfft2, the LaMa FFC global branch; a
             research hypothesis, not the default) | ``pyramid`` (pooled bins 1, 2, 4)
context      coarse branch on the pooled 4x-extent crop, restricted to the context channels and entered
             through a learned scalar gate initialised near zero, so regional context cannot dominate local
             fault evidence unless it earns its weight (``context_gate`` is logged)
heads        seg (+ dist, + orient): the geometry-aware multi-task decoder of Experiment C

Tile artefacts: ``long_range`` changes the receptive field inside a patch only; shifted-tile consistency is
measured by ``inference.full_raster.shifted_tile_consistency``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geological_encoders import ConcatFuse, ContextBranch, Decoder, Encoder, GatedFuse, conv_block


class AttentionFuse(nn.Module):
    """Modality-wise attention: per pixel and per channel group, softmax weights over the branches."""

    def __init__(self, cins: list[int], cout: int, heads: int = 4):
        super().__init__()
        while cout % heads:
            heads -= 1
        self.heads, self.nb = heads, len(cins)
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, cout, 1, bias=False), nn.BatchNorm2d(cout)) for c in cins])
        hid = max(sum(cins) // 4, 16)
        self.score = nn.Sequential(nn.Conv2d(sum(cins), hid, 1), nn.ReLU(inplace=True), nn.Conv2d(hid, self.nb * heads, 1))
        self.last_gate = None

    def forward(self, feats):
        B, _, H, W = feats[0].shape
        g = torch.softmax(self.score(torch.cat(feats, 1)).view(B, self.nb, self.heads, H, W), dim=1)
        if not self.training:
            self.last_gate = g.mean(2).detach()                        # (B, nb, H, W) for gate maps
        out = 0
        for i, (p, f) in enumerate(zip(self.proj, feats)):
            v = p(f)
            C = v.shape[1]
            out = out + (g[:, i].repeat_interleave(C // self.heads, dim=1) * v)
        return F.relu(out)


class CrossScaleFuse(nn.Module):
    """Gated fusion over every branch at level l and every branch's level l+1 (upsampled)."""

    def __init__(self, cins: list[int], cins_next: list[int] | None, cout: int):
        super().__init__()
        self.has_next = cins_next is not None
        self.fuse = GatedFuse(list(cins) + (list(cins_next) if self.has_next else []), cout)

    def forward(self, feats, nxt=None):
        if self.has_next and nxt is not None:
            size = feats[0].shape[-2:]
            feats = list(feats) + [F.interpolate(n, size=size, mode="bilinear", align_corners=False) for n in nxt]
        return self.fuse(feats)

    @property
    def last_gate(self):
        return self.fuse.last_gate


class DilatedContext(nn.Module):
    def __init__(self, c: int, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 3, padding=d, dilation=d, bias=False),
                                                   nn.BatchNorm2d(c), nn.ReLU(inplace=True)) for d in dilations])

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return x


class FourierUnit(nn.Module):
    """Global receptive field in one layer (LaMa FFC spectral transform, simplified). Runs in fp32."""

    def __init__(self, c: int, hidden_frac: float = 0.5):
        super().__init__()
        h = max(int(c * hidden_frac), 8)
        self.reduce = nn.Sequential(nn.Conv2d(c, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True))
        self.spec = nn.Sequential(nn.Conv2d(2 * h, 2 * h, 1, bias=False), nn.BatchNorm2d(2 * h), nn.ReLU(inplace=True))
        self.expand = nn.Conv2d(h, c, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))                      # residual starts as identity

    def forward(self, x):
        y = self.reduce(x)
        H, W = y.shape[-2:]
        with torch.autocast(device_type=y.device.type, enabled=False):
            f = torch.fft.rfft2(y.float(), norm="ortho")
            z = self.spec(torch.cat([f.real, f.imag], 1))
            re, im = z.chunk(2, 1)
            y = torch.fft.irfft2(torch.complex(re, im), s=(H, W), norm="ortho")
        return x + self.gamma * self.expand(y.to(x.dtype))


class PyramidContext(nn.Module):
    def __init__(self, c: int, bins=(1, 2, 4)):
        super().__init__()
        r = max(c // len(bins), 8)
        self.bins = bins
        self.stages = nn.ModuleList([nn.Sequential(nn.Conv2d(c, r, 1, bias=False), nn.BatchNorm2d(r), nn.ReLU(inplace=True))
                                     for _ in bins])
        self.out = nn.Sequential(nn.Conv2d(c + r * len(bins), c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))

    def forward(self, x):
        H, W = x.shape[-2:]
        ps = [F.interpolate(s(F.adaptive_avg_pool2d(x, b)), (H, W), mode="bilinear", align_corners=False)
              for b, s in zip(self.bins, self.stages)]
        return self.out(torch.cat([x] + ps, 1))


LONG_RANGE = {"none": None, "dilated": DilatedContext, "fourier": FourierUnit, "pyramid": PyramidContext}


class GeoStructureNet(nn.Module):
    def __init__(self, groups: list[list[int]], n_channels: int, *, enc_width: int = 24, dec_width: int = 32,
                 fusion: str = "gated", long_range: str = "none", context_idx: list[int] | None = None,
                 context_width: int = 16, context_factor: int = 4, depth: int = 5, heads: int = 4,
                 aux_dist: bool = False, aux_orient: bool = False):
        super().__init__()
        if fusion not in ("concat", "gated", "attention", "crossscale"):
            raise ValueError(f"unknown fusion {fusion}")
        if long_range not in LONG_RANGE:
            raise ValueError(f"unknown long_range {long_range}")
        self.groups = [list(g) for g in groups]
        for i, g in enumerate(self.groups):
            self.register_buffer(f"idx{i}", torch.tensor(g, dtype=torch.long), persistent=False)
        self.encoders = nn.ModuleList([Encoder(len(g), enc_width, depth) for g in self.groups])
        dws = [dec_width * 2 ** i for i in range(depth)]
        ws = [[e.widths[l] for e in self.encoders] for l in range(depth)]
        self.fusion = fusion
        if fusion == "concat":
            self.fuse = nn.ModuleList([ConcatFuse(ws[l], dws[l]) for l in range(depth)])
        elif fusion == "gated":
            self.fuse = nn.ModuleList([GatedFuse(ws[l], dws[l]) for l in range(depth)])
        elif fusion == "attention":
            self.fuse = nn.ModuleList([AttentionFuse(ws[l], dws[l], heads) for l in range(depth)])
        else:
            self.fuse = nn.ModuleList([CrossScaleFuse(ws[l], ws[l + 1] if l + 1 < depth else None, dws[l])
                                       for l in range(depth)])
        lr = LONG_RANGE[long_range]
        self.long_range = lr(dws[-1]) if lr is not None else None
        self.context = None
        if context_idx:
            self.register_buffer("ctx_idx", torch.tensor(context_idx, dtype=torch.long), persistent=False)
            self.context = ContextBranch(len(context_idx), context_width, dws[-1], context_factor)
            self.context_gate = nn.Parameter(torch.tensor(-2.0))     # sigmoid(-2) = 0.12 at initialisation
        self.decoder = Decoder(dws)
        self.head = nn.Conv2d(dws[0], 1, 1)
        self.dist_head = nn.Conv2d(dws[0], 1, 1) if aux_dist else None
        self.orient_head = nn.Conv2d(dws[0], 2, 1) if aux_orient else None

    def encode(self, x):
        per = [enc(x.index_select(1, getattr(self, f"idx{i}"))) for i, enc in enumerate(self.encoders)]
        levels = list(zip(*per))
        if self.fusion == "crossscale":
            return [f(list(levels[l]), list(levels[l + 1]) if l + 1 < len(levels) else None)
                    for l, f in enumerate(self.fuse)]
        return [f(list(lv)) for f, lv in zip(self.fuse, levels)]

    def forward(self, x, ctx=None):
        feats = self.encode(x)
        if self.long_range is not None:
            feats[-1] = self.long_range(feats[-1])
        if self.context is not None:
            if ctx is None:
                raise ValueError("this model has a context branch: pass the pooled context crop")
            c = self.context(ctx.index_select(1, self.ctx_idx), feats[-1].shape[-2:])
            feats[-1] = feats[-1] + torch.sigmoid(self.context_gate) * c
        y = self.decoder(feats)
        out = {"seg": self.head(y)}
        if self.dist_head is not None:
            out["dist"] = self.dist_head(y)
        if self.orient_head is not None:
            out["orient"] = self.orient_head(y)
        return out

    def encoder_parameters(self):
        return [p for e in self.encoders for p in e.parameters()]

    def gates(self) -> list:
        return [getattr(f, "last_gate", None) for f in self.fuse]

    def context_weight(self) -> float | None:
        return float(torch.sigmoid(self.context_gate)) if self.context is not None else None
