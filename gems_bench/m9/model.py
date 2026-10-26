"""Section 06: GeoFaultNet, one multi-resolution, multi-task network (the only new architecture family of M9).

    local patch (P x P, 100 m)                       low-resolution context (P x P cells of f x f px = f*P extent)
      raw stem | structural stem | spectral stem       pooled local + regional context channels (strain,
      | DEM stem | map-context stem (damaged map + flag)  seismicity, smooth regional fields)
            \\ gated per-pixel fusion /                         |
       residual U-Net encoder (P, P/2, P/4, P/8, P/16) <-- centre crop at P/16 + FiLM from the pooled context vector
       decoder with skips (P/8 ... P)
            |-> seg       dense fault evidence (logit)
            |-> orient    axial orientation (cos 2t, sin 2t) (tanh in the loss / at inference)
            |-> endpoint  endpoint heat (logit)
            |-> junction  branch / junction heat (logit)

Design notes
* Modality stems keep raw fields, structural evidence, spectral descriptors and the optional DEM separate before a
  learned per-pixel gate decides how much each contributes (no forced agreement between modalities).
* The map-context stem sees the (possibly damaged) known-fault map and a constant flag (1 = a map was given, 0 = no
  map): "no context" therefore never reads as "no faults here".
* The context branch sees a 4x wider area at 4x coarser resolution; its centre crop is aligned with the local patch
  at the bottleneck (16 native px per cell), so regional descriptors modulate, but never localise, faults.
* Every input is label-free except the map-context channel (visible labels only; see corruption.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = {"seg": 1, "orient": 2, "ep": 1, "jn": 1}


def _block(i: int, o: int, drop: float = 0.0) -> nn.Module:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                         nn.Dropout2d(drop) if drop > 0 else nn.Identity())


class ResBlock(nn.Module):
    def __init__(self, i: int, o: int, drop: float = 0.0):
        super().__init__()
        self.body = _block(i, o, drop)
        self.skip = nn.Conv2d(i, o, 1, bias=False) if i != o else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class Stem(nn.Module):
    def __init__(self, c_in: int, w: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.GELU(),
                                 nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.GELU())

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    """Per-pixel sigmoid gate per modality stem, then a 1x1 projection (``concat`` = no gate)."""

    def __init__(self, n: int, w: int, out: int, gated: bool = True):
        super().__init__()
        self.n, self.gated = n, gated
        self.gate = nn.Conv2d(n * w, n, 1) if gated else None
        self.proj = nn.Sequential(nn.Conv2d(n * w, out, 1, bias=False), nn.BatchNorm2d(out), nn.GELU())

    def forward(self, feats: list):
        x = torch.cat(feats, 1)
        if self.gated:
            g = torch.sigmoid(self.gate(x))
            x = torch.cat([f * g[:, i:i + 1] for i, f in enumerate(feats)], 1)
        return self.proj(x)


class ContextEncoder(nn.Module):
    """Pooled-cube encoder: stride 16 / f, returns (feature map, global vector)."""

    def __init__(self, c_in: int, w: int, factor: int):
        super().__init__()
        n_down = int(round(math.log2(16 / factor)))
        layers = [Stem(c_in, w)]
        for _ in range(n_down):
            layers += [nn.Conv2d(w, w, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(w), nn.GELU(), ResBlock(w, w)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        f = self.net(x)
        return f, f.mean((2, 3))


class GeoFaultNet(nn.Module):
    def __init__(self, group_sizes: dict, n_pool: int, *, patch_px: int, factor: int, channel_index=None,
                 stem_width: int = 16, widths=(32, 64, 128, 192), context_width: int = 48, fusion: str = "gated",
                 dropout: float = 0.1):
        super().__init__()
        self.groups = [(g, int(n)) for g, n in group_sizes.items() if int(n) > 0]   # stem order, footprint in "raw"
        n_in = sum(n for _, n in self.groups)
        idx = list(range(n_in)) if channel_index is None else list(channel_index)
        self.register_buffer("idx", torch.as_tensor(idx, dtype=torch.long), persistent=False)
        self.P, self.f = int(patch_px), int(factor)
        self.stems = nn.ModuleDict({g: Stem(n, stem_width) for g, n in self.groups})
        self.ctx_stem = Stem(2, stem_width)
        w0, w1, w2, w3 = widths
        self.fuse = GatedFusion(len(self.groups) + 1, stem_width, w0, gated=fusion == "gated")
        self.enc0 = ResBlock(w0, w0)
        self.enc1 = ResBlock(w0, w1)
        self.enc2 = ResBlock(w1, w2)
        self.enc3 = ResBlock(w2, w3, dropout)
        self.bott = ResBlock(w3, w3, dropout)
        self.pool = nn.MaxPool2d(2)
        self.cenc = ContextEncoder(n_pool, context_width, self.f) if n_pool > 0 else None
        cw = context_width if self.cenc is not None else 0
        self.cw = cw
        self.cfuse = nn.Sequential(nn.Conv2d(w3 + cw, w3, 1, bias=False), nn.BatchNorm2d(w3), nn.GELU())
        self.film = nn.Linear(cw, 2 * w3) if cw else None
        self.up3, self.dec3 = nn.Conv2d(w3, w3, 1), ResBlock(w3 + w3, w3)
        self.up2, self.dec2 = nn.Conv2d(w3, w2, 1), ResBlock(w2 + w2, w2)
        self.up1, self.dec1 = nn.Conv2d(w2, w1, 1), ResBlock(w1 + w1, w1)
        self.up0, self.dec0 = nn.Conv2d(w1, w0, 1), ResBlock(w0 + w0, w0)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.heads = nn.ModuleDict({h: nn.Sequential(nn.Conv2d(w0, w0, 3, padding=1), nn.GELU(), nn.Conv2d(w0, k, 1))
                                    for h, k in HEADS.items()})
        with torch.no_grad():                                          # rare positives: start near p = 0.02
            for h in ("seg", "ep", "jn"):
                self.heads[h][-1].bias.fill_(-4.0)

    def _ctx_crop(self, cf, size: int):
        n = cf.shape[-1]
        o = (n - size) // 2
        c = cf[..., o:o + size, o:o + size]
        return c if c.shape[-1] == size else F.interpolate(c, size=(size, size), mode="bilinear", align_corners=False)

    def forward(self, x, m, xc=None) -> dict:
        """x: (B, C_local, P, P) in CUBE order; m: (B, 2, P, P) map context + flag; xc: (B, C_pool, P, P)."""
        x = x.index_select(1, self.idx)
        feats, i = [], 0
        for g, n in self.groups:
            feats.append(self.stems[g](x[:, i:i + n]))
            i += n
        feats.append(self.ctx_stem(m))
        e0 = self.enc0(self.fuse(feats))
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        if self.cenc is not None and xc is not None:
            cf, cv = self.cenc(xc)
            b = self.cfuse(torch.cat([b, self._ctx_crop(cf, b.shape[-1])], 1))
            g, s = self.film(cv).chunk(2, 1)
            b = b * (1 + torch.tanh(g)[..., None, None]) + s[..., None, None]
        elif self.cenc is not None:
            b = self.cfuse(torch.cat([b, b.new_zeros(b.shape[0], self.cw, *b.shape[2:])], 1))
        d = self.dec3(torch.cat([self.up3(F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)), e3], 1))
        d = self.dec2(torch.cat([self.up2(F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)), e2], 1))
        d = self.dec1(torch.cat([self.up1(F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)), e1], 1))
        d = self.dec0(torch.cat([self.up0(F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)), e0], 1))
        d = self.drop(d)
        return {h: head(d) for h, head in self.heads.items()}


def group_sizes(meta: dict, groups: dict) -> dict:
    """Channel counts per stem in cube order (the footprint channel joins the raw stem)."""
    sizes = {"raw": len(groups["raw"]) + 1, "struct": len(groups["struct"]), "spectral": len(groups["spectral"]),
             "dem": len(groups["dem"])}
    local = list(meta["local"])
    order = groups["raw"] + groups["struct"] + groups["spectral"] + groups["dem"] + ["m9:footprint"]
    if local != list(dict.fromkeys(order)):
        raise ValueError("cube channel order differs from the group order: rebuild the cube (section 04)")
    return sizes


def reorder_index(meta: dict, groups: dict) -> list[int]:
    """Index of the cube channels in stem order (raw..., footprint, struct..., spectral..., dem...)."""
    pos = {n: i for i, n in enumerate(meta["local"])}
    order = groups["raw"] + ["m9:footprint"] + groups["struct"] + groups["spectral"] + groups["dem"]
    return [pos[n] for n in order]


def build(spec: dict) -> GeoFaultNet:
    """``spec`` = net_spec(...) (stored verbatim in every checkpoint)."""
    mc = spec["model"]
    return GeoFaultNet(spec["group_sizes"], int(spec["n_pool"]), patch_px=int(spec["patch_px"]),
                       factor=int(spec["factor"]), channel_index=spec["channel_index"],
                       stem_width=int(mc["stem_width"]), widths=tuple(mc["widths"]),
                       context_width=int(mc["context_width"]), fusion=mc["fusion"], dropout=float(mc["dropout"]))


def net_spec(meta: dict, groups: dict, vcfg: dict) -> dict:
    """Everything needed to rebuild the network and its inputs (JSON-serialisable)."""
    return {"group_sizes": group_sizes(meta, groups), "channel_index": reorder_index(meta, groups),
            "n_pool": len(meta["pooled"]), "patch_px": int(vcfg["data"]["patch_px"]),
            "factor": int(vcfg["data"]["context_factor"]), "model": dict(vcfg["model"]), "cube_key": meta["key"],
            "local": list(meta["local"]), "pooled": list(meta["pooled"]), "groups": groups}


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
