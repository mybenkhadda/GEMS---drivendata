"""Architecture variants for the local-global reconsideration (Phase N). Native PyTorch, no downloads.

``g14_local``            G14 GeoFaultFormer, local view only (= T1 geometry)                     control
``n1_large_rf``          local CNN + ASPP (dilations 1, 2, 4, 8 at stride 16 => ~ 12 km receptive field)   N1
``n2_local_regional``    local CNN + regional CNN branch, aligned gated injection (cheap transformer control) N2
``g14_regional_swin``    G14 GeoFaultFormer with the regional Swin encoder, no global                         N3
``g14_full``             G14 GeoFaultFormer with regional Swin + global tokens                                N4
``n5_directional``       local CNN + directional line attention over 8 orientations                           N5
``n5_isotropic``         the same network with an isotropic conv in place of the directional block (control)
(N6 graph reasoning is post-processing: ``g14.graph`` via ``experiments``.)

Every variant consumes the same batch dict as G14 (``local``, ``local_mask``, ``regional``, ...) and returns
``{"fault": (B,1,H,W) logits, "aux": [...]}`` so the corrected trainer, losses and inference are shared: the
mandatory comparison (identical features, supervision, folds, masks, optimisation) only changes this module.

UNTESTED: written without execution.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..g14.model import ContextInjection, ConvBlock, LocalEncoder, ModalityFusion, _gn

ARCHS = ("g14_local", "n1_large_rf", "n2_local_regional", "g14_regional_swin", "g14_full", "n5_directional",
         "n5_isotropic")


class Decoder(nn.Module):
    """U-Net decoder 16 -> 8 -> 4 -> 2 -> 1 with two auxiliary logits (strides 8 and 4)."""

    def __init__(self, enc_ch: list[int], dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([ConvBlock(enc_ch[i] + enc_ch[i - 1], enc_ch[i - 1]) for i in range(4, 0, -1)])
        self.aux = nn.ModuleList([nn.Conv2d(enc_ch[3], 1, 1), nn.Conv2d(enc_ch[2], 1, 1)])
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(enc_ch[0], 1, 1)
        nn.init.constant_(self.head.bias, -3.0)

    def forward(self, feats: list[torch.Tensor], out_hw) -> dict:
        y = feats[-1]
        aux = []
        for i, blk in enumerate(self.blocks):
            skip = feats[3 - i]
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, skip], 1))
            if i < 2:
                aux.append(F.interpolate(self.aux[i](y), size=out_hw, mode="bilinear", align_corners=False))
        return {"fault": self.head(self.drop(y)), "aux": aux}


class ASPP(nn.Module):
    def __init__(self, c: int, rates=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 3, 1, r, dilation=r, bias=False), _gn(c),
                                                     nn.GELU()) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1), nn.GELU())
        self.out = nn.Sequential(nn.Conv2d(c * (len(rates) + 1), c, 1, bias=False), _gn(c), nn.GELU())

    def forward(self, x):
        ys = [b(x) for b in self.branches]
        ys.append(self.pool(x).expand_as(x))
        return x + self.out(torch.cat(ys, 1))


def _line_kernel(length: int, angle_deg: float) -> torch.Tensor:
    k = torch.zeros(length, length)
    c = (length - 1) / 2
    a = math.radians(angle_deg)
    for t in torch.linspace(-c, c, 4 * length):
        r, q = int(round(c - float(t) * math.sin(a))), int(round(c + float(t) * math.cos(a)))
        if 0 <= r < length and 0 <= q < length:
            k[r, q] = 1.0
    return k / k.sum()


class DirectionalLineAttention(nn.Module):
    """Per-pixel softmax over ``n_dirs`` oriented line aggregations (fixed depthwise line kernels, learnable mixing):
    evidence is integrated along candidate strike directions rather than isotropically."""

    def __init__(self, c: int, n_dirs: int = 8, length: int = 9, isotropic: bool = False):
        super().__init__()
        self.isotropic = isotropic
        if isotropic:
            self.conv = nn.Sequential(nn.Conv2d(c, c, length, 1, length // 2, groups=c, bias=False),
                                      nn.Conv2d(c, c, 1), _gn(c), nn.GELU())
            return
        ks = torch.stack([_line_kernel(length, 180.0 * i / n_dirs) for i in range(n_dirs)])   # (D, L, L)
        self.register_buffer("kernels", ks[:, None], persistent=False)
        self.n_dirs, self.length = n_dirs, length
        self.score = nn.Conv2d(c, n_dirs, 3, 1, 1)
        self.mix = nn.Sequential(nn.Conv2d(c, c, 1), _gn(c), nn.GELU())

    def forward(self, x):
        if self.isotropic:
            return x + self.conv(x)
        B, C, H, W = x.shape
        a = torch.softmax(self.score(x).float(), 1).to(x.dtype)           # (B, D, H, W)
        acc = 0
        for d in range(self.n_dirs):
            k = self.kernels[d:d + 1].to(x.dtype).expand(C, 1, self.length, self.length)
            g = F.conv2d(x, k, padding=self.length // 2, groups=C)
            acc = acc + a[:, d:d + 1] * g
        return x + self.mix(acc)


class _LocalBase(nn.Module):
    def __init__(self, cfg, groups, modalities):
        super().__init__()
        m = cfg.model
        mask_index = {n: i for i, n in enumerate(modalities)}
        groups = {k: v for k, v in groups.items() if k in mask_index and v}
        self.fusion = ModalityFusion(groups, mask_index, m.modality_dim, m.local_widths[0], m.fusion)
        self.local = LocalEncoder(m.local_widths[0], m.local_widths, m.local_encoder, m.convnext_depths)
        self.dec = Decoder(self.local.channels, m.dropout)
        self._mc = False

    def enable_mc_dropout(self, on: bool = True):
        self._mc = on
        for mod in self.modules():
            if isinstance(mod, nn.Dropout2d):
                mod.train(on)

    def encode(self, batch):
        z, _ = self.fusion(batch["local"], batch["local_mask"])
        return self.local(z)


class LargeRFNet(_LocalBase):
    def __init__(self, cfg, groups, modalities):
        super().__init__(cfg, groups, modalities)
        self.aspp = ASPP(self.local.channels[-1])

    def forward(self, batch):
        f = self.encode(batch)
        f[-1] = self.aspp(f[-1])
        return self.dec(f, batch["local"].shape[-2:])


class LocalRegionalCNN(_LocalBase):
    def __init__(self, cfg, groups, modalities, n_ch: int, stages=(2, 3)):
        super().__init__(cfg, groups, modalities)
        m, d = cfg.model, cfg.data
        self.reg_stem = nn.Sequential(nn.Conv2d(n_ch + len(modalities), m.local_widths[0], 3, 1, 1), _gn(m.local_widths[0]),
                                      nn.GELU())
        self.reg = LocalEncoder(m.local_widths[0], m.local_widths, "resnet")
        self.frac = d.local.extent_m / d.regional.extent_m
        self.stages = stages
        self.inject = nn.ModuleDict({str(s): ContextInjection(self.local.channels[s + 1], self.reg.channels[-1])
                                     for s in stages})

    def forward(self, batch):
        f = self.encode(batch)
        r = self.reg(self.reg_stem(torch.cat([batch["regional"], batch["regional_mask"]], 1)))[-1]
        for s in self.stages:
            f[s + 1], _ = self.inject[str(s)](f[s + 1], r, self.frac)
        return self.dec(f, batch["local"].shape[-2:])


class DirectionalNet(_LocalBase):
    def __init__(self, cfg, groups, modalities, isotropic: bool = False):
        super().__init__(cfg, groups, modalities)
        ch = self.local.channels
        self.dir2 = DirectionalLineAttention(ch[2], isotropic=isotropic)
        self.dir3 = DirectionalLineAttention(ch[3], isotropic=isotropic)

    def forward(self, batch):
        f = self.encode(batch)
        f[2], f[3] = self.dir2(f[2]), self.dir3(f[3])
        return self.dec(f, batch["local"].shape[-2:])


def g14_cfg_for(arch: str, base_cfg):
    """G14 config variant (model section) for the GeoFaultFormer-based architectures."""
    common = {"heads": ("fault",), "uncertainty": "none"}
    loss = {"aux_weights": {"orientation": 0.0, "connectivity": 0.0}}       # the fault head only
    if arch in ("g14_local", "n1_large_rf", "n2_local_regional", "n5_directional", "n5_isotropic"):
        return base_cfg.updated({"loss": loss, "model": {**common, "regional_encoder": "none",
                                                         "global_encoder": "none", "context_stages": (),
                                                         "bottleneck_cross_attention": False, "film_global": False}})
    if arch == "g14_regional_swin":
        return base_cfg.updated({"loss": loss, "model": {**common, "regional_encoder": "swin",
                                                         "global_encoder": "none", "film_global": False}})
    if arch == "g14_full":
        return base_cfg.updated({"loss": loss, "model": {**common, "regional_encoder": "swin",
                                                         "global_encoder": "tokens"}})
    raise KeyError(arch)


def build(arch: str, cfg, view) -> nn.Module:
    """Model for ``arch`` on the channel layout of ``view`` (training.ChannelView or g14 UnitInputs)."""
    from ..g14.model import GeoFaultFormer

    groups, mods = view.modality_index(), view.modalities
    if arch in ("g14_local", "g14_regional_swin", "g14_full"):
        return GeoFaultFormer(cfg, groups, mods)
    if arch == "n1_large_rf":
        return LargeRFNet(cfg, groups, mods)
    if arch == "n2_local_regional":
        return LocalRegionalCNN(cfg, groups, mods, view.n_channels)
    if arch in ("n5_directional", "n5_isotropic"):
        return DirectionalNet(cfg, groups, mods, isotropic=arch == "n5_isotropic")
    raise KeyError(f"unknown architecture {arch!r}; known {ARCHS}")


def count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
