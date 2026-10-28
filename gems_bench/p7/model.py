"""GeoFaultFormer: residual CNN encoder + shifted-window attention + optional ConvGRU + U-Net decoder + heads.

    x (B, C, P, P)  aligned geophysical patch, C discovered from the feature registry (never hard-coded)
      fusion:  early    one stem on every channel (default)
               grouped  one stem per modality group (magnetic / gravity / geological / other) + learned per-pixel
                        sigmoid gate per group, then a 1x1 projection
      e0 (1/1, stem_width) -> e1 (1/2) -> e2 (1/4) -> e3 (1/8) -> e4 (1/16)      residual stages, GroupNorm
      attention stages after the configured encoder stages (default 3 and 4); identity when disabled (pure-CNN path)
      decoder: e4 -> up + e3 -> up + e2 [-> ConvGRU refinement] -> up + e1 -> up + e0 -> d0 (full resolution)
      heads on d0: seg (fault logit), center (centerline logit), orient (cos 2t, sin 2t), dist (normalised distance
                   logit), ep / jn (endpoint / junction logits); only the enabled heads exist

Design notes
* GroupNorm (not BatchNorm): micro-batches of 8 or 16 on different GPUs train the same model (presets change only
  the micro-batch / accumulation split).
* Upsampling is ``interpolate(size = skip size)``: any patch size works; stride-2 convolutions with padding 1 give
  ceil(H / 2), so odd sizes are safe. Zero padding at the patch border is down-weighted at inference by the Hann
  blend of overlapping patches (m9.infer), so tiles have no hard seams.
* No pretrained RGB weights: the geophysical channels have no RGB analogue (documented choice, not an omission).
* ``forward(x, m=None, xc=None)`` keeps the M9 call signature; P7 uses no map context and no pooled context, so
  every input is label-free.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionStage
from .convgru import ConvGRURefiner

HEAD_CHANNELS = {"fault": ("seg", 1), "center": ("center", 1), "orient": ("orient", 2), "distance": ("dist", 1),
                 "endpoint": ("ep", 1), "junction": ("jn", 1)}
LOGIT_BIAS = {"seg": -4.0, "center": -4.0, "ep": -4.0, "jn": -4.0}       # rare positives: start near p = 0.02


def gn(c: int) -> nn.GroupNorm:
    g = 8
    while c % g:
        g //= 2
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    def __init__(self, i: int, o: int, stride: int = 1, drop: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), gn(o), nn.GELU(),
                                  nn.Conv2d(o, o, 3, 1, 1, bias=False), gn(o),
                                  nn.Dropout2d(drop) if drop > 0 else nn.Identity())
        self.skip = nn.Identity() if (i == o and stride == 1) else nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False),
                                                                                gn(o))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


def stem(c_in: int, w: int) -> nn.Module:
    return nn.Sequential(nn.Conv2d(c_in, w, 3, padding=1, bias=False), gn(w), nn.GELU(),
                         nn.Conv2d(w, w, 3, padding=1, bias=False), gn(w), nn.GELU())


class GroupedFusion(nn.Module):
    """Modality stems + per-pixel gate per group (no forced agreement between modalities) + 1x1 projection."""

    def __init__(self, groups: dict[str, list[int]], w: int):
        super().__init__()
        self.names = [g for g, idx in groups.items() if len(idx)]
        for g in self.names:
            self.register_buffer(f"idx_{g}", torch.as_tensor(groups[g], dtype=torch.long), persistent=False)
        self.stems = nn.ModuleDict({g: stem(len(groups[g]), w) for g in self.names})
        n = len(self.names)
        self.gate = nn.Conv2d(n * w, n, 1)
        self.proj = nn.Sequential(nn.Conv2d(n * w, w, 1, bias=False), gn(w), nn.GELU())
        self.last_gate: torch.Tensor | None = None

    def forward(self, x):
        feats = [self.stems[g](x.index_select(1, getattr(self, f"idx_{g}"))) for g in self.names]
        f = torch.cat(feats, 1)
        g = torch.sigmoid(self.gate(f))
        self.last_gate = g.detach()
        return self.proj(torch.cat([a * g[:, i:i + 1] for i, a in enumerate(feats)], 1))


class GeoFaultFormer(nn.Module):
    def __init__(self, in_channels: int, *, groups: dict[str, list[int]] | None = None, stem_width: int = 32,
                 widths=(32, 64, 128, 192), depths=(1, 1, 2, 2), dropout: float = 0.1, fusion: str = "early",
                 attention: dict | None = None, convgru: dict | None = None, heads: dict | None = None):
        super().__init__()
        sw, ws = int(stem_width), [int(w) for w in widths]
        self.fusion_mode = fusion
        self.fuse = GroupedFusion(groups, sw) if fusion == "grouped" and groups else stem(int(in_channels), sw)
        enc, prev = [], sw
        for s, (w, d) in enumerate(zip(ws, depths), start=1):
            blocks = [ResBlock(prev, w, 2, dropout if s >= 3 else 0.0)]
            blocks += [ResBlock(w, w, 1, dropout if s >= 3 else 0.0) for _ in range(int(d) - 1)]
            enc.append(nn.Sequential(*blocks))
            prev = w
        self.enc = nn.ModuleList(enc)
        at = attention or {}
        scales = set(int(s) for s in at.get("scales", [])) if at.get("enabled") else set()
        self.attn = nn.ModuleDict({str(s): AttentionStage(ws[s - 1], int(at["depth"]), int(at["heads"]),
                                                          int(at["window"]), float(at.get("mlp_ratio", 2.0)),
                                                          float(at.get("drop_path", 0.0)))
                                   for s in sorted(scales)})
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for s in (3, 2, 1):                                   # decoder levels 1/8, 1/4, 1/2
            self.up.append(nn.Conv2d(ws[s], ws[s - 1], 1))
            self.dec.append(ResBlock(2 * ws[s - 1], ws[s - 1]))
        self.up0, self.dec0 = nn.Conv2d(ws[0], sw, 1), ResBlock(2 * sw, sw)
        g = convgru or {}
        self.gru_scale = int(g.get("scale", 2)) if g.get("enabled") else None
        self.gru = ConvGRURefiner(ws[self.gru_scale - 1], int(g.get("hidden", 64)), int(g.get("kernel", 3)),
                                  int(g.get("steps", 3)), bool(g.get("checkpoint", False))) \
            if self.gru_scale else None
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        hd = heads or {"fault": True}
        self.head_keys = [HEAD_CHANNELS[h][0] for h in HEAD_CHANNELS if hd.get(h)]
        self.heads = nn.ModuleDict({HEAD_CHANNELS[h][0]: nn.Sequential(nn.Conv2d(sw, sw, 3, padding=1), nn.GELU(),
                                                                       nn.Conv2d(sw, HEAD_CHANNELS[h][1], 1))
                                    for h in HEAD_CHANNELS if hd.get(h)})
        with torch.no_grad():
            for k, b in LOGIT_BIAS.items():
                if k in self.heads:
                    self.heads[k][-1].bias.fill_(b)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def set_capture(self, on: bool) -> None:
        for a in self.attn.values():
            a.set_capture(on)
        if self.gru is not None:
            self.gru.capture = bool(on)

    def forward(self, x, m=None, xc=None) -> dict:
        e = [self.fuse(x)]
        for s, blk in enumerate(self.enc, start=1):
            f = blk(e[-1])
            if str(s) in self.attn:
                f = self.attn[str(s)](f)
            e.append(f)
        d = e[4]
        for i, s in enumerate((3, 2, 1)):
            d = self.dec[i](torch.cat([self.up[i](self._up(d, e[s])), e[s]], 1))
            if self.gru is not None and s == self.gru_scale:
                d = self.gru(d)
        d = self.dec0(torch.cat([self.up0(self._up(d, e[0])), e[0]], 1))
        d = self.drop(d)
        return {k: h(d) for k, h in self.heads.items()}


# =============================================================================== spec / build
def net_spec(channel_names: list[str], groups: dict[str, list[int]], ecfg: dict, cube_key: str) -> dict:
    """Everything needed to rebuild the network and its inputs (JSON-serialisable; stored in every checkpoint)."""
    mc = ecfg["model"]
    return {"in_channels": len(channel_names), "channels": list(channel_names), "groups": groups,
            "patch_px": int(ecfg["data"]["patch_px"]), "cube_key": cube_key,
            "model": {k: mc[k] for k in ("stem_width", "widths", "depths", "dropout", "fusion", "attention", "convgru",
                                         "heads", "uncertainty")}}


def build(spec: dict) -> GeoFaultFormer:
    mc = spec["model"]
    return GeoFaultFormer(int(spec["in_channels"]), groups=spec.get("groups"), stem_width=int(mc["stem_width"]),
                          widths=tuple(mc["widths"]), depths=tuple(mc["depths"]), dropout=float(mc["dropout"]),
                          fusion=mc["fusion"]["mode"], attention=mc["attention"], convgru=mc["convgru"],
                          heads=mc["heads"])


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def param_breakdown(model) -> dict:
    """Trainable parameters per top-level block (parameter-budget comparisons across rungs)."""
    out = {}
    for n, mod in model.named_children():
        out[n] = int(sum(p.numel() for p in mod.parameters() if p.requires_grad))
    out["total"] = count_params(model)
    return out


def profile_memory(spec: dict, micro_batch: int, device, amp_dtype=None, steps: int = 3) -> dict:
    """Peak GPU memory and seconds per training step of one micro-batch on SYNTHETIC tensors (no data). Colab only:
    the authoring session never calls it."""
    model = build(spec).to(device).train()
    P, C = int(spec["patch_px"]), int(spec["in_channels"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(micro_batch, C, P, P, device=device)
    use_amp = amp_dtype is not None and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    times = []
    for _ in range(int(steps)):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(x)
        loss = sum(v.float().mean() for v in out.values())
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    rec = {"micro_batch": int(micro_batch), "patch_px": P, "in_channels": C, "n_params": count_params(model),
           "params": param_breakdown(model), "step_s_median": float(sorted(times)[len(times) // 2]),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.type == "cuda" else None}
    del model, opt, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec
