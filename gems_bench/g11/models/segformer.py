"""Compact SegFormer (Mix Transformer encoder + all-MLP decoder) for 100 m geophysical patches.

The G07 "T1 SegFormer" code is not in this repository; :func:`load_compatible_weights` loads any external
state dict whose keys AND shapes match and reports the coverage - weights are never assumed compatible.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device) >= self.p
        return x * keep / (1 - self.p)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, cin, dim, k, s):
        super().__init__()
        self.proj = nn.Conv2d(cin, dim, k, s, k // 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


class EfficientAttention(nn.Module):
    def __init__(self, dim, heads, sr):
        super().__init__()
        self.h, self.scale = heads, (dim // heads) ** -0.5
        self.q, self.kv, self.proj = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.sr = nn.Conv2d(dim, dim, sr, sr) if sr > 1 else None
        self.norm = nn.LayerNorm(dim) if sr > 1 else None

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.h, c // self.h).transpose(1, 2)
        y = x
        if self.sr is not None and h >= self.sr.kernel_size[0] and w >= self.sr.kernel_size[1]:
            y = self.sr(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
            y = self.norm(y)                     # (maps smaller than the reduction ratio attend without reduction)
        k, v = self.kv(y).reshape(b, -1, 2, self.h, c // self.h).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)
        return self.proj((a @ v).transpose(1, 2).reshape(b, n, c))


class MixFFN(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        hid = dim * ratio
        self.fc1, self.dw, self.fc2 = nn.Linear(dim, hid), nn.Conv2d(hid, hid, 3, 1, 1, groups=hid), nn.Linear(hid, dim)

    def forward(self, x, h, w):
        x = self.fc1(x)
        b, n, c = x.shape
        x = self.dw(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(x))


class Block(nn.Module):
    def __init__(self, dim, heads, sr, dp):
        super().__init__()
        self.n1, self.att, self.n2, self.ffn, self.dp = nn.LayerNorm(dim), EfficientAttention(dim, heads, sr), \
            nn.LayerNorm(dim), MixFFN(dim), DropPath(dp)

    def forward(self, x, h, w):
        x = x + self.dp(self.att(self.n1(x), h, w))
        return x + self.dp(self.ffn(self.n2(x), h, w))


class SegFormer(nn.Module):
    """``forward(x) -> logits (B, 1, H, W)``; ``features(x)`` -> list of 4 stage maps (for fusion models)."""

    def __init__(self, in_channels: int, dims=(32, 64, 160, 256), depths=(2, 2, 2, 2), heads=(1, 2, 5, 8),
                 srs=(8, 4, 2, 1), decoder_dim: int = 128, drop_path: float = 0.1, out_channels: int = 1):
        super().__init__()
        self.embeds, self.stages, self.norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        cin = in_channels
        dps = torch.linspace(0, drop_path, sum(depths)).tolist()
        j = 0
        for i, d in enumerate(dims):
            self.embeds.append(OverlapPatchEmbed(cin, d, 7 if i == 0 else 3, 4 if i == 0 else 2))
            self.stages.append(nn.ModuleList([Block(d, heads[i], srs[i], dps[j + k]) for k in range(depths[i])]))
            self.norms.append(nn.LayerNorm(d))
            j += depths[i]
            cin = d
        self.lin = nn.ModuleList([nn.Conv2d(d, decoder_dim, 1) for d in dims])
        self.fuse = nn.Sequential(nn.Conv2d(4 * decoder_dim, decoder_dim, 1, bias=False), nn.BatchNorm2d(decoder_dim),
                                  nn.ReLU(inplace=True))
        self.head = nn.Conv2d(decoder_dim, out_channels, 1)
        self.decoder_dim = decoder_dim

    def features(self, x):
        outs = []
        for emb, blocks, norm in zip(self.embeds, self.stages, self.norms):
            t, h, w = emb(x)
            for blk in blocks:
                t = blk(t, h, w)
            x = norm(t).transpose(1, 2).reshape(t.shape[0], -1, h, w)
            outs.append(x)
        return outs

    def decode(self, feats, size):
        s0 = feats[0].shape[-2:]
        ups = [F.interpolate(l(f), size=s0, mode="bilinear", align_corners=False) for l, f in zip(self.lin, feats)]
        y = self.fuse(torch.cat(ups, 1))
        return y, F.interpolate(self.head(y), size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        _, logits = self.decode(self.features(x), x.shape[-2:])
        return logits


def load_compatible_weights(model: nn.Module, state_or_path) -> dict:
    """Load matching (name, shape) tensors only; returns {'loaded', 'skipped_shape', 'missing', 'coverage'}."""
    state = torch.load(state_or_path, map_location="cpu", weights_only=False) if isinstance(state_or_path, (str,)) \
        or hasattr(state_or_path, "__fspath__") else state_or_path
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    own = model.state_dict()
    ok = {k: v for k, v in state.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
    model.load_state_dict({**own, **ok})
    return {"loaded": len(ok), "skipped_shape": sorted(k for k in state if k in own and k not in ok),
            "missing": sorted(k for k in own if k not in ok), "coverage": len(ok) / max(len(own), 1)}
