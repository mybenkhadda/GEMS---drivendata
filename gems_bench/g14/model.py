"""GeoFaultFormer (Phase 5): modality encoders -> adaptive fusion -> local CNN (100 m) + regional transformer (200 m)
+ global token encoder (800 m) -> aligned local-global fusion -> U-Net decoder -> heads, on the 100 m target grid.

Every encoder is implemented here (no network access, no implicit weight download). ``pretrained_*`` paths load a
local state dict with ``strict=False`` and report missing / unexpected keys; nothing is fetched.

Shapes (defaults): local (B, C, 128, 128), regional (B, C, 256, 256), global (B, C, 256, 256), masks (B, M, ., .).
Output ``fault`` logits (B, 1, 128, 128) on the local grid.

Attention or gate weights are returned for diagnostics only; they are not evidence of physical causality.

UNTESTED: written without execution (shape contracts are covered by tests/test_g14_model.py, not yet run).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _gn(c: int) -> nn.GroupNorm:
    g = 8 if c % 8 == 0 else (4 if c % 4 == 0 else 1)
    return nn.GroupNorm(g, c)


class ConvBlock(nn.Module):
    def __init__(self, ci: int, co: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ci, co, 3, stride, 1, bias=False), _gn(co), nn.GELU(),
                                  nn.Conv2d(co, co, 3, 1, 1, bias=False), _gn(co))
        self.skip = nn.Identity() if ci == co and stride == 1 else nn.Sequential(
            nn.Conv2d(ci, co, 1, stride, bias=False), _gn(co))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


# =============================================================================== modality encoders + fusion
class ModalityEncoder(nn.Module):
    """One stem per modality: (its channels + its availability mask) -> ``dim`` features, zeroed where unavailable."""

    def __init__(self, n_in: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(n_in + 1, dim, 3, 1, 1, bias=False), _gn(dim), nn.GELU(),
                                 nn.Conv2d(dim, dim, 3, 1, 1, bias=False), _gn(dim), nn.GELU())

    def forward(self, x, mask):
        return self.net(torch.cat([x * mask, mask], 1)) * mask


class ModalityFusion(nn.Module):
    """concat | gated (sample- and pixel-dependent softmax over AVAILABLE modalities) | cross_attention."""

    def __init__(self, groups: dict[str, list[int]], mask_index: dict[str, int], dim: int, out_dim: int, mode: str):
        super().__init__()
        self.names = list(groups)
        self.groups = groups
        self.mask_index = mask_index
        self.mode = mode
        self.enc = nn.ModuleDict({m: ModalityEncoder(len(groups[m]), dim) for m in self.names})
        M = len(self.names)
        if mode == "concat":
            self.out = nn.Sequential(nn.Conv2d(M * dim, out_dim, 1), _gn(out_dim), nn.GELU())
        elif mode == "gated":
            self.gate = nn.Conv2d(M * dim + M, M, 1)
            self.out = nn.Sequential(nn.Conv2d(dim, out_dim, 1), _gn(out_dim), nn.GELU())
        elif mode == "cross_attention":
            self.q = nn.Parameter(torch.zeros(dim))
            self.k = nn.Conv2d(dim, dim, 1)
            self.v = nn.Conv2d(dim, dim, 1)
            self.out = nn.Sequential(nn.Conv2d(dim, out_dim, 1), _gn(out_dim), nn.GELU())
        else:
            raise ValueError(mode)
        self.dim = dim

    def forward(self, x, masks):
        feats, avail = [], []
        for m in self.names:
            mk = masks[:, self.mask_index[m]:self.mask_index[m] + 1]
            feats.append(self.enc[m](x[:, self.groups[m]], mk))
            avail.append(mk)
        A = torch.cat(avail, 1)                                          # (B, M, H, W) availability in [0, 1]
        diag = {}
        if self.mode == "concat":
            y = self.out(torch.cat(feats, 1))
        elif self.mode == "gated":
            logits = self.gate(torch.cat(feats + [A], 1)).float()
            logits = logits.masked_fill(A < 0.05, -1e4)
            w = torch.softmax(logits, 1) * (A.amax(1, keepdim=True) >= 0.05)
            y = self.out(sum(w[:, i:i + 1].to(feats[i].dtype) * feats[i] for i in range(len(feats))))
            diag["modality_weights"] = w.detach()
        else:
            Fm = torch.stack(feats, 1)                                   # (B, M, D, H, W)
            B, M, D, H, W = Fm.shape
            flat = Fm.reshape(B * M, D, H, W)
            k = self.k(flat).reshape(B, M, D, H, W)
            v = self.v(flat).reshape(B, M, D, H, W)
            q = self.q.view(1, 1, D, 1, 1) + (Fm * A.unsqueeze(2)).sum(1, keepdim=True) / A.sum(1, keepdim=True).clamp(
                min=1e-3).unsqueeze(2)
            s = (q * k).sum(2).float() / math.sqrt(D)                     # (B, M, H, W)
            s = s.masked_fill(A < 0.05, -1e4)
            a = torch.softmax(s, 1) * (A.amax(1, keepdim=True) >= 0.05)
            y = self.out((a.unsqueeze(2).to(v.dtype) * v).sum(1))
            diag["modality_attention"] = a.detach()
        return y, diag


# =============================================================================== local encoders
class ConvNeXtBlock(nn.Module):
    def __init__(self, c: int, layer_scale: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, 1, 3, groups=c)
        self.ln = nn.LayerNorm(c)
        self.pw1, self.pw2 = nn.Linear(c, 4 * c), nn.Linear(4 * c, c)
        self.gamma = nn.Parameter(layer_scale * torch.ones(c))

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.pw2(F.gelu(self.pw1(self.ln(y)))) * self.gamma
        return x + y.permute(0, 3, 1, 2)


class LocalEncoder(nn.Module):
    """Features at strides 1, 2, 4, 8, 16 with channels [w0, w1, w2, w3, w3]."""

    def __init__(self, cin: int, widths, kind: str = "unet", convnext_depths=(2, 2, 6, 2)):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.kind = kind
        self.stem = ConvBlock(cin, w0)
        chans = [w1, w2, w3, w3]
        prev = w0
        self.stages = nn.ModuleList()
        for i, c in enumerate(chans):
            if kind == "convnext":
                down = nn.Sequential(_gn(prev), nn.Conv2d(prev, c, 2, 2))
                blocks = [ConvNeXtBlock(c) for _ in range(convnext_depths[i])]
                self.stages.append(nn.Sequential(down, *blocks))
            elif kind == "resnet":
                self.stages.append(nn.Sequential(ConvBlock(prev, c, 2), ConvBlock(c, c)))
            else:                                                        # lightweight U-Net encoder
                self.stages.append(nn.Sequential(nn.MaxPool2d(2), ConvBlock(prev, c)))
            prev = c
        self.channels = [w0] + chans

    def forward(self, x):
        feats = [self.stem(x)]
        for s in self.stages:
            feats.append(s(feats[-1]))
        return feats


# =============================================================================== regional transformer (Swin-style)
def _windows(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def _unwindows(w, ws, B, H, W):
    C = w.shape[-1]
    x = w.view(B, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, C)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, ws: int, heads: int):
        super().__init__()
        self.ws, self.heads = ws, heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        nn.init.trunc_normal_(self.bias, std=0.02)
        c = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + (ws - 1)
        self.register_buffer("rel_index", rel[..., 0] * (2 * ws - 1) + rel[..., 1], persistent=False)

    def forward(self, x, mask=None):
        Bw, N, C = x.shape
        qkv = self.qkv(x).view(Bw, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * (C // self.heads) ** -0.5) @ k.transpose(-2, -1)
        attn = attn + self.bias[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(Bw // nw, nw, self.heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, N, N)
        attn = torch.softmax(attn.float(), -1).to(v.dtype)
        return self.proj((attn @ v).transpose(1, 2).reshape(Bw, N, C))


class SwinBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ws: int, shift: int, res: int):
        super().__init__()
        self.ws, self.shift = (res, 0) if res <= ws else (ws, shift)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.ws, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        if self.shift:
            img = torch.zeros(1, res, res, 1)
            cnt = 0
            for hs in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
                for wsl in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
                    img[:, hs, wsl, :] = cnt
                    cnt += 1
            mw = _windows(img, self.ws).squeeze(-1)
            m = mw.unsqueeze(1) - mw.unsqueeze(2)
            self.register_buffer("mask", m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0), persistent=False)
        else:
            self.mask = None

    def forward(self, x):                                    # x: (B, H, W, C)
        B, H, W, C = x.shape
        y = self.n1(x)
        if self.shift:
            y = torch.roll(y, (-self.shift, -self.shift), (1, 2))
        y = _unwindows(self.attn(_windows(y, self.ws), self.mask), self.ws, B, H, W)
        if self.shift:
            y = torch.roll(y, (self.shift, self.shift), (1, 2))
        x = x + y
        return x + self.mlp(self.n2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.red = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.red(self.norm(x))


class SwinEncoder(nn.Module):
    """Hierarchical shifted-window transformer; returns (B, C_i, H_i, W_i) per stage."""

    def __init__(self, cin: int, size: int, embed: int, depths, heads, ws: int, patch: int):
        super().__init__()
        self.embed = nn.Conv2d(cin, embed, patch, patch)
        self.norm0 = nn.LayerNorm(embed)
        res, dim = size // patch, embed
        self.stages, self.merges, self.channels = nn.ModuleList(), nn.ModuleList(), []
        for i, (d, h) in enumerate(zip(depths, heads)):
            self.stages.append(nn.ModuleList([SwinBlock(dim, h, ws, (ws // 2) if j % 2 else 0, res) for j in range(d)]))
            self.channels.append(dim)
            if i < len(depths) - 1:
                self.merges.append(PatchMerging(dim))
                dim, res = dim * 2, res // 2

    def forward(self, x):
        x = self.norm0(self.embed(x).permute(0, 2, 3, 1))
        outs = []
        for i, blocks in enumerate(self.stages):
            for b in blocks:
                x = b(x)
            outs.append(x.permute(0, 3, 1, 2).contiguous())
            if i < len(self.merges):
                x = self.merges[i](x)
        return outs


class MiTEncoder(nn.Module):
    """SegFormer-style Mix Transformer: overlapping patch embeddings, spatial-reduction attention, Mix-FFN."""

    def __init__(self, cin: int, embed: int, depths, heads, sr=(8, 4, 2, 1)):
        super().__init__()
        self.channels, self.stages = [], nn.ModuleList()
        prev = cin
        for i, (d, h) in enumerate(zip(depths, heads)):
            dim = embed * 2 ** i
            k, s = (7, 4) if i == 0 else (3, 2)
            pe = nn.Conv2d(prev, dim, k, s, k // 2)
            blocks = nn.ModuleList([_MiTBlock(dim, h, sr[min(i, len(sr) - 1)]) for _ in range(d)])
            self.stages.append(nn.ModuleList([pe, nn.LayerNorm(dim), blocks, nn.LayerNorm(dim)]))
            self.channels.append(dim)
            prev = dim

    def forward(self, x):
        outs = []
        for pe, n0, blocks, n1 in self.stages:
            x = pe(x)
            B, C, H, W = x.shape
            t = n0(x.flatten(2).transpose(1, 2))
            for b in blocks:
                t = b(t, H, W)
            x = n1(t).transpose(1, 2).reshape(B, C, H, W)
            outs.append(x)
        return outs


class _MiTBlock(nn.Module):
    def __init__(self, dim, heads, sr):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.sr = nn.Conv2d(dim, dim, sr, sr) if sr > 1 else None
        self.nsr = nn.LayerNorm(dim) if sr > 1 else None
        self.fc1, self.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
        self.dw = nn.Conv2d(4 * dim, 4 * dim, 3, 1, 1, groups=4 * dim)

    def forward(self, t, H, W):
        y = self.n1(t)
        kv = y
        if self.sr is not None:
            B, N, C = y.shape
            kv = self.nsr(self.sr(y.transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2))
        t = t + self.attn(y, kv, kv, need_weights=False)[0]
        y = self.fc1(self.n2(t))
        B, N, C4 = y.shape
        y = self.dw(y.transpose(1, 2).reshape(B, C4, H, W)).flatten(2).transpose(1, 2)
        return t + self.fc2(F.gelu(y))


# =============================================================================== global context encoder
class GlobalTokenEncoder(nn.Module):
    """Compressed tokens of the 205 km view (no full-resolution global attention): strided CNN -> K pooled tokens ->
    small transformer. Returns tokens (B, K, D) and a summary vector (B, D)."""

    def __init__(self, cin: int, dim: int, n_tokens: int, layers: int):
        super().__init__()
        self.cnn = nn.Sequential(ConvBlock(cin, dim // 4, 2), ConvBlock(dim // 4, dim // 2, 2), ConvBlock(dim // 2, dim, 2),
                                 ConvBlock(dim, dim, 2))
        g = int(round(math.sqrt(n_tokens)))
        self.grid = g
        self.pos = nn.Parameter(torch.zeros(1, g * g, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, 2 * dim, 0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        f = F.adaptive_avg_pool2d(self.cnn(x), self.grid)
        t = self.tf(f.flatten(2).transpose(1, 2) + self.pos)
        return t, t.mean(1)


# =============================================================================== alignment / fusion helpers
def align_center(feat: torch.Tensor, frac: float, out_hw: tuple[int, int]) -> torch.Tensor:
    """Exact centred crop of the fraction ``frac`` of a coarser view, resampled to ``out_hw`` (bilinear).

    With centres on the shared lattice the local window is exactly the central ``frac`` of the coarser view, so the
    affine grid [[frac, 0, 0], [0, frac, 0]] samples precisely its footprint."""
    B, C = feat.shape[:2]
    theta = torch.zeros(B, 2, 3, device=feat.device, dtype=torch.float32)
    theta[:, 0, 0] = frac
    theta[:, 1, 1] = frac
    grid = F.affine_grid(theta, (B, C, *out_hw), align_corners=False)
    return F.grid_sample(feat.float(), grid, mode="bilinear", padding_mode="border", align_corners=False).to(feat.dtype)


class ContextInjection(nn.Module):
    """x <- x + sigmoid(g([x, P(ctx)])) * P(ctx), ctx aligned to x's footprint."""

    def __init__(self, c_local: int, c_ctx: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(c_ctx, c_local, 1), _gn(c_local), nn.GELU())
        self.gate = nn.Conv2d(2 * c_local, c_local, 1)
        nn.init.constant_(self.gate.bias, -2.0)              # start mostly local; context earns its weight

    def forward(self, x, ctx, frac):
        p = self.proj(align_center(ctx, frac, x.shape[-2:]))
        g = torch.sigmoid(self.gate(torch.cat([x, p], 1)))
        return x + g * p, g.detach().mean()


class TokenCrossAttention(nn.Module):
    """Local bottleneck tokens attend to regional + global context tokens (residual, pre-norm)."""

    def __init__(self, dim: int, ctx_dims: list[int], heads: int = 4):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(c, dim) for c in ctx_dims])
        self.view_emb = nn.Parameter(torch.zeros(len(ctx_dims), dim))
        self.nq, self.nk = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x, ctx_tokens: list[torch.Tensor]):
        B, C, H, W = x.shape
        q = x.flatten(2).transpose(1, 2)
        kv = torch.cat([p(t) + self.view_emb[i] for i, (p, t) in enumerate(zip(self.projs, ctx_tokens))], 1)
        y, w = self.attn(self.nq(q), self.nk(kv), self.nk(kv), need_weights=True)
        return (q + y).transpose(1, 2).reshape(B, C, H, W), w.detach()


class FiLM(nn.Module):
    def __init__(self, dim: int, c: int):
        super().__init__()
        self.lin = nn.Linear(dim, 2 * c)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, z):
        g, b = self.lin(z).chunk(2, -1)
        return x * (1 + g[..., None, None]) + b[..., None, None]


# =============================================================================== the model
class GeoFaultFormer(nn.Module):
    def __init__(self, cfg, channel_groups: dict[str, list[int]], modalities: list[str]):
        super().__init__()
        m, d = cfg.model, cfg.data
        self.cfg = cfg
        self.modalities = modalities
        mask_index = {name: i for i, name in enumerate(modalities)}
        groups = {k: v for k, v in channel_groups.items() if k in mask_index and v}
        n_ch = sum(len(v) for v in channel_groups.values())
        w = m.local_widths
        self.fusion = ModalityFusion(groups, mask_index, m.modality_dim, w[0], m.fusion)
        self.local = LocalEncoder(w[0], w, m.local_encoder, m.convnext_depths)
        self.frac_regional = d.local.extent_m / d.regional.extent_m
        self.frac_global = d.local.extent_m / d.global_.extent_m
        ctx_in = n_ch + len(modalities)
        self.regional = None
        if m.regional_encoder == "swin":
            self.regional = SwinEncoder(ctx_in, d.regional.size_px, m.swin_embed, m.swin_depths, m.swin_heads,
                                        m.swin_window, m.swin_patch)
        elif m.regional_encoder == "mit":
            self.regional = MiTEncoder(ctx_in, m.swin_embed, m.swin_depths, m.swin_heads)
        self.globalenc = GlobalTokenEncoder(ctx_in, m.global_dim, m.global_tokens, m.global_layers) \
            if m.global_encoder == "tokens" else None
        enc_ch = self.local.channels                         # strides 1, 2, 4, 8, 16
        self.inject = nn.ModuleDict()
        if self.regional is not None:
            for s in m.context_stages:
                j = min(s, len(self.regional.channels) - 1)
                self.inject[str(s)] = ContextInjection(enc_ch[s + 1], self.regional.channels[j])
        ctx_dims = ([self.regional.channels[-1]] if self.regional is not None else []) + \
                   ([m.global_dim] if self.globalenc is not None else [])
        self.xattn = TokenCrossAttention(enc_ch[-1], ctx_dims) if m.bottleneck_cross_attention and ctx_dims else None
        # decoder (U-Net): 16 -> 8 -> 4 -> 2 -> 1
        dec = []
        for i in range(4, 0, -1):
            dec.append(ConvBlock(enc_ch[i] + enc_ch[i - 1], enc_ch[i - 1]))
        self.dec = nn.ModuleList(dec)
        self.film = nn.ModuleList([FiLM(m.global_dim, enc_ch[i - 1]) for i in range(4, 0, -1)]) \
            if m.film_global and self.globalenc is not None else None
        self.drop = nn.Dropout2d(m.dropout) if m.dropout > 0 else nn.Identity()
        self.head_fault = nn.Conv2d(enc_ch[0], 1, 1)
        nn.init.constant_(self.head_fault.bias, -3.0)        # sparse prior: faults are ~1 % of pixels
        self.head_orient = nn.Conv2d(enc_ch[0], 2, 1) if "orientation" in m.heads else None
        self.head_conn = nn.Conv2d(enc_ch[0], 2, 1) if "connectivity" in m.heads else None
        self.head_logvar = nn.Conv2d(enc_ch[0], 1, 1) if m.uncertainty == "heteroscedastic" else None
        self.aux = nn.ModuleList([nn.Conv2d(enc_ch[2], 1, 1), nn.Conv2d(enc_ch[3], 1, 1)]) if m.deep_supervision else None
        self._mc = False

    def enable_mc_dropout(self, on: bool = True) -> None:
        """Keep dropout active at inference (MC-dropout epistemic uncertainty)."""
        self._mc = on

    def forward(self, batch: dict) -> dict:
        x, xm = batch["local"], batch["local_mask"]
        diag = {}
        z, fd = self.fusion(x, xm)
        diag.update(fd)
        feats = self.local(z)
        reg = None
        if self.regional is not None:
            reg = self.regional(torch.cat([batch["regional"], batch["regional_mask"]], 1))
            for s, mod in self.inject.items():
                s = int(s)
                j = min(s, len(reg) - 1)
                feats[s + 1], g = mod(feats[s + 1], reg[j], self.frac_regional)
                diag[f"context_gate_stage{s}"] = g
        gtok = gsum = None
        if self.globalenc is not None:
            gtok, gsum = self.globalenc(torch.cat([batch["global"], batch["global_mask"]], 1))
        if self.xattn is not None:
            toks = []
            if reg is not None:
                toks.append(reg[-1].flatten(2).transpose(1, 2))
            if gtok is not None:
                toks.append(gtok)
            feats[-1], aw = self.xattn(feats[-1], toks)
            diag["bottleneck_attention"] = aw
        y = feats[-1]
        aux_logits = []
        for i, blk in enumerate(self.dec):
            skip = feats[3 - i]
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, skip], 1))
            if self.film is not None:
                y = self.film[i](y, gsum)
            if self.aux is not None and i in (0, 1):                     # strides 8 and 4
                aux_logits.append(self.aux[1 - i](y))
        if self._mc and isinstance(self.drop, nn.Dropout2d):
            y = F.dropout2d(y, self.drop.p, training=True)
        else:
            y = self.drop(y)
        out = {"fault": self.head_fault(y), "diagnostics": diag}
        H, W = x.shape[-2:]
        if self.aux is not None:
            out["aux"] = [F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False) for a in aux_logits]
        if self.head_orient is not None:
            o = torch.tanh(self.head_orient(y))
            out["orientation"] = o / o.norm(dim=1, keepdim=True).clamp(min=1e-4)
        if self.head_conn is not None:
            out["connectivity"] = self.head_conn(y)
        if self.head_logvar is not None:
            out["logvar"] = self.head_logvar(y).clamp(-6, 4)
        return out


def build_model(cfg, inputs) -> GeoFaultFormer:
    """Model for the channel layout of ``inputs`` (multiscale.UnitInputs); loads optional local weights."""
    groups = inputs.modality_index()
    model = GeoFaultFormer(cfg, groups, inputs.modalities)
    for path, part in ((cfg.model.pretrained_local, model.local), (cfg.model.pretrained_regional, model.regional)):
        if path:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            res = part.load_state_dict(sd, strict=False)
            print(f"[g14.model] loaded {path}: missing {len(res.missing_keys)}, unexpected {len(res.unexpected_keys)}")
    return model


def count_parameters(model: nn.Module) -> dict:
    out = {"total": sum(p.numel() for p in model.parameters())}
    for name in ("fusion", "local", "regional", "globalenc", "xattn", "dec"):
        mod = getattr(model, name, None)
        out[name] = sum(p.numel() for p in mod.parameters()) if mod is not None else 0
    return out
