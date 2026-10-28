"""Shifted-window self-attention on convolutional feature maps (H1: local context at 1/8 and 1/16).

    x (B, C, H, W) -> pad to a multiple of the window -> [cyclic shift] -> windows of ws x ws tokens -> multi-head
    attention + relative position bias + additive mask -> reverse -> [unshift] -> crop -> residual MLP

Masking (``window_mask``): one additive (n_windows, N, N) mask per feature-map size combines
* padding: keys on padded positions get MASK_VALUE, so real tokens never attend to zeros added by the padding;
* shifted windows: after the cyclic shift a window may hold tokens from opposite borders of the map; keys from a
  different region than the query get MASK_VALUE, so nothing attends across the wrap-around.
MASK_VALUE is finite (-1e4 stays finite in fp16 / bf16): a fully padded query row softmaxes to a uniform average and
is cropped away, never NaN.

Attention never spans the whole patch: cost is O(H W ws^2). ``capture = True`` stores the softmax weights of the
last forward pass (explicit path) for the attention figure; these are genuine model weights, unlike gradient
saliency, which the figure labels separately.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_VALUE = -1.0e4


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) with H, W multiples of ws -> (B * nW, ws * ws, C), windows in row-major order."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def window_reverse(win: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """Inverse of ``window_partition``: (B * nW, ws * ws, C) -> (B, H, W, C)."""
    C = win.shape[-1]
    B = win.shape[0] // ((H // ws) * (W // ws))
    x = win.view(B, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, C)


def pad_amount(n: int, ws: int) -> int:
    return (ws - n % ws) % ws


def window_mask(H: int, W: int, ws: int, shift: int, device=None) -> torch.Tensor | None:
    """Additive (nW, N, N) mask for an UNPADDED map of H x W padded to the window grid, shifted by ``shift``.
    Returns None when neither padding nor shifting needs masking."""
    ph, pw = pad_amount(H, ws), pad_amount(W, ws)
    if shift == 0 and ph == 0 and pw == 0:
        return None
    Hp, Wp = H + ph, W + pw
    valid = torch.zeros(Hp, Wp, device=device)
    valid[:H, :W] = 1.0
    region = torch.zeros(Hp, Wp, device=device)
    if shift > 0:
        cnt = 0
        for hs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            for wsl in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
                region[hs, wsl] = cnt
                cnt += 1
        valid = torch.roll(valid, shifts=(-shift, -shift), dims=(0, 1))
        # the region map is defined on the SHIFTED grid (Swin convention): labels split the wrapped strips
    rw = window_partition(region[None, :, :, None], ws)[..., 0]           # (nW, N)
    vw = window_partition(valid[None, :, :, None], ws)[..., 0]            # (nW, N)
    same = rw[:, :, None] == rw[:, None, :]
    key_ok = vw[:, None, :] > 0.5
    return torch.where(same & key_ok, torch.zeros((), device=device), torch.full((), MASK_VALUE, device=device))


class RelativePositionBias(nn.Module):
    """Learned bias per head for every relative offset inside a window ((2 ws - 1)^2 entries)."""

    def __init__(self, ws: int, heads: int):
        super().__init__()
        self.ws, self.heads = ws, heads
        self.table = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        nn.init.trunc_normal_(self.table, std=0.02)
        c = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)   # (2, N)
        rel = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + (ws - 1)                             # (N, N, 2)
        self.register_buffer("index", rel[..., 0] * (2 * ws - 1) + rel[..., 1], persistent=False)

    def forward(self) -> torch.Tensor:
        N = self.ws * self.ws
        return self.table[self.index.view(-1)].view(N, N, self.heads).permute(2, 0, 1).contiguous()   # (h, N, N)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, ws: int):
        super().__init__()
        self.heads, self.ws = heads, ws
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.bias = RelativePositionBias(ws, heads)
        self.capture = False
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """x: (B * nW, N, C); mask: (nW, N, N) additive or None."""
        Bn, N, C = x.shape
        qkv = self.qkv(x).view(Bn, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                   # (Bn, h, N, d)
        bias = self.bias().unsqueeze(0).to(q.dtype)                        # (1, h, N, N)
        if mask is not None:                                               # windows are batch-major (b * nW + w)
            nW = mask.shape[0]
            m = mask.to(q.dtype)[None, :, None]                             # (1, nW, 1, N, N)
            bias = (bias.unsqueeze(1) + m).expand(Bn // nW, nW, self.heads, N, N).reshape(Bn, self.heads, N, N)
        if self.capture or not hasattr(F, "scaled_dot_product_attention"):
            a = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5) + bias
            a = a.softmax(-1)
            if self.capture:
                self.last_weights = a.detach().float().cpu()
            out = a @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.expand(Bn, -1, -1, -1))
        return self.proj(out.transpose(1, 2).reshape(Bn, N, C))


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = 1.0 - self.p
        m = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * m / keep


class SwinBlock(nn.Module):
    """Pre-norm window attention + MLP on (B, C, H, W); ``shift`` is disabled when the map fits in one window."""

    def __init__(self, dim: int, heads: int, ws: int, shift: int, mlp_ratio: float = 2.0, drop_path: float = 0.0):
        super().__init__()
        self.ws, self.shift = int(ws), int(shift)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, self.ws)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))
        self.dp = DropPath(drop_path)
        self._masks: dict = {}

    def _mask(self, H: int, W: int, shift: int, device) -> torch.Tensor | None:
        k = (H, W, shift, str(device))
        if k not in self._masks:
            self._masks[k] = window_mask(H, W, self.ws, shift, device)
        return self._masks[k]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.ws
        ph, pw = pad_amount(H, ws), pad_amount(W, ws)
        Hp, Wp = H + ph, W + pw
        shift = self.shift if min(Hp, Wp) > ws else 0
        t = x.permute(0, 2, 3, 1)                                          # (B, H, W, C)
        h = self.n1(t)
        if ph or pw:
            h = F.pad(h, (0, 0, 0, pw, 0, ph))
        if shift:
            h = torch.roll(h, shifts=(-shift, -shift), dims=(1, 2))
        w = self.attn(window_partition(h, ws), self._mask(H, W, shift, x.device))
        h = window_reverse(w, ws, Hp, Wp)
        if shift:
            h = torch.roll(h, shifts=(shift, shift), dims=(1, 2))
        t = t + self.dp(h[:, :H, :W])
        t = t + self.dp(self.mlp(self.n2(t)))
        return t.permute(0, 3, 1, 2).contiguous()


class AttentionStage(nn.Module):
    """``depth`` blocks alternating regular (shift 0) and shifted (ws // 2) windows."""

    def __init__(self, dim: int, depth: int, heads: int, ws: int, mlp_ratio: float = 2.0, drop_path: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, heads, ws, 0 if i % 2 == 0 else ws // 2, mlp_ratio, drop_path)
                                     for i in range(int(depth))])

    def set_capture(self, on: bool) -> None:
        for b in self.blocks:
            b.attn.capture = bool(on)
            if not on:
                b.attn.last_weights = None

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x
