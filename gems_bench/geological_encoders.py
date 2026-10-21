"""Geology-aware encoder-decoders for Phase 4 (all return {"seg": logits, ...}; output = one fault raster).

Building blocks mirror the Phase-3 built-in U-Net (gems_bench/cnn.py): double 3x3 conv + BN + ReLU blocks,
max-pool down, transposed-conv up, 5 levels with widths w, 2w, 4w, 8w, 16w. ``GeoUNet`` with one channel
group and ``dec_width == enc_width`` is that U-Net, so Architecture A (single encoder) and the matched-capacity
control share code with the dual encoder.

Architectures
-------------
A  single encoder on all channels (``groups=[all]``).
B  dual encoder: terrain and subsurface branches, fused at every level by ``concat`` (concat + 1x1 conv) or
   ``gated`` (per-pixel softmax over branches; the gate is kept in eval mode for evidence maps), then one
   shared decoder. Compare against ``matched_single_width`` (same parameter count) to separate capacity
   from the branch structure.
C  low-resolution context branch: the stack average-pooled by ``context_factor`` (4) and cropped to the patch
   size, i.e. 4x the patch extent; its encoder output (centre cells + global mean) is fused into the bottleneck.
D  auxiliary heads (training-only): fault proximity (sigmoid) and local strike as (cos 2θ, sin 2θ) (tanh).

``MaskedAutoencoder`` (SSL) uses the same ``Encoder`` so pretrained weights load into ``GeoUNet.encoders[0]``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    def __init__(self, c_in: int, width: int, depth: int = 5):
        super().__init__()
        self.widths = [width * 2 ** i for i in range(depth)]
        ws = self.widths
        self.blocks = nn.ModuleList([conv_block(c_in, ws[0])] + [conv_block(ws[i], ws[i + 1]) for i in range(depth - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feats = []
        for i, b in enumerate(self.blocks):
            if i:
                x = self.pool(x)
            x = b(x)
            feats.append(x)
        return feats


class Decoder(nn.Module):
    def __init__(self, widths: list[int]):
        super().__init__()
        n = len(widths)
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(n - 1))])
        self.dec = nn.ModuleList([conv_block(2 * widths[i], widths[i]) for i in reversed(range(n - 1))])

    def forward(self, feats):
        x = feats[-1]
        for up, dec, s in zip(self.up, self.dec, reversed(feats[:-1])):
            x = dec(torch.cat([up(x), s], 1))
        return x


class ConcatFuse(nn.Module):
    def __init__(self, cins: list[int], cout: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(sum(cins), cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, feats):
        return self.proj(torch.cat(feats, 1))


class GatedFuse(nn.Module):
    """out = ReLU(sum_b g_b * BN(1x1(f_b))), g = softmax over branches of a 1x1 conv of the concatenation."""

    def __init__(self, cins: list[int], cout: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, cout, 1, bias=False), nn.BatchNorm2d(cout)) for c in cins])
        self.gate = nn.Conv2d(sum(cins), len(cins), 1)
        self.last_gate = None

    def forward(self, feats):
        g = torch.softmax(self.gate(torch.cat(feats, 1)), dim=1)
        if not self.training:
            self.last_gate = g.detach()
        out = sum(g[:, i:i + 1] * p(f) for i, (p, f) in enumerate(zip(self.proj, feats)))
        return F.relu(out)


class ContextBranch(nn.Module):
    """Encoder over the 1/factor-resolution context crop; returns features at the main bottleneck size."""

    def __init__(self, c_in: int, width: int, out_ch: int, factor: int, depth: int = 4):
        super().__init__()
        ws = [width * min(2 ** i, 4) for i in range(depth)]
        layers, c = [], c_in
        for w in ws:
            layers += [conv_block(c, w), nn.MaxPool2d(2)]
            c = w
        self.body = nn.Sequential(*layers)
        self.factor = factor
        self.proj = nn.Sequential(nn.Conv2d(2 * c, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, ctx, size):
        f = self.body(ctx)
        n = f.shape[-1]
        m = max(int(round(n / self.factor)), 1)
        a = (n - m) // 2
        centre = F.interpolate(f[..., a:a + m, a:a + m], size=tuple(size), mode="bilinear", align_corners=False)
        glob = f.mean((-2, -1), keepdim=True).expand(-1, -1, *size)
        return self.proj(torch.cat([centre, glob], 1))


class GeoUNet(nn.Module):
    def __init__(self, groups: list[list[int]], n_channels: int, enc_width: int = 32, dec_width: int | None = None,
                 fusion: str = "concat", depth: int = 5, context: bool = False, context_width: int = 16,
                 context_factor: int = 4, aux_dist: bool = False, aux_orient: bool = False):
        super().__init__()
        self.groups = [list(g) for g in groups]
        self._identity = [g == list(range(n_channels)) for g in self.groups]
        for i, g in enumerate(self.groups):
            self.register_buffer(f"idx{i}", torch.tensor(g, dtype=torch.long), persistent=False)
        self.encoders = nn.ModuleList([Encoder(len(g), enc_width, depth) for g in self.groups])
        dec_width = dec_width or enc_width
        dws = [dec_width * 2 ** i for i in range(depth)]
        if len(self.groups) == 1 and dec_width == enc_width:
            self.fuse = None
        else:
            Fuse = GatedFuse if fusion == "gated" else ConcatFuse
            self.fuse = nn.ModuleList([Fuse([e.widths[l] for e in self.encoders], dws[l]) for l in range(depth)])
        self.context = ContextBranch(n_channels, context_width, dws[-1], context_factor) if context else None
        self.ctx_fuse = nn.Sequential(nn.Conv2d(2 * dws[-1], dws[-1], 1, bias=False), nn.BatchNorm2d(dws[-1]),
                                      nn.ReLU(inplace=True)) if context else None
        self.decoder = Decoder(dws)
        self.head = nn.Conv2d(dws[0], 1, 1)
        self.dist_head = nn.Conv2d(dws[0], 1, 1) if aux_dist else None
        self.orient_head = nn.Conv2d(dws[0], 2, 1) if aux_orient else None

    def encode(self, x):
        per = [enc(x if self._identity[i] else x.index_select(1, getattr(self, f"idx{i}")))
               for i, enc in enumerate(self.encoders)]
        if self.fuse is None:
            return list(per[0])
        return [f(list(level)) for f, level in zip(self.fuse, zip(*per))]

    def forward(self, x, ctx=None):
        feats = self.encode(x)
        if self.context is not None:
            if ctx is None:
                raise ValueError("this model has a context branch: pass the context crop")
            c = self.context(ctx, feats[-1].shape[-2:])
            feats[-1] = self.ctx_fuse(torch.cat([feats[-1], c], 1))
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
        """Last eval-mode gate tensors per level (gated fusion only)."""
        return [f.last_gate for f in (self.fuse or []) if isinstance(f, GatedFuse)]


class MaskedAutoencoder(nn.Module):
    """SSL model: same Encoder as GeoUNet, light U-Net decoder, reconstructs all channels (+ optional extras)."""

    def __init__(self, c_in: int, enc_width: int, depth: int = 5, n_out: int | None = None):
        super().__init__()
        self.encoder = Encoder(c_in, enc_width, depth)
        self.decoder = Decoder(self.encoder.widths)
        self.head = nn.Conv2d(enc_width, n_out or c_in, 1)

    def forward(self, x):
        return self.head(self.decoder(self.encoder(x)))


# ---------------------------------------------------------------------------
# construction helpers
# ---------------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def group_indices(channels: list[str], groups: dict[str, list[str]]) -> list[list[int]]:
    pos = {c: i for i, c in enumerate(channels)}
    out = []
    for name, chans in groups.items():
        missing = [c for c in chans if c not in pos]
        if missing:
            raise KeyError(f"group {name}: channels not in the stack: {missing}")
        out.append([pos[c] for c in chans])
    return out


def build_geo_model(spec) -> GeoUNet:
    """GeoUNet from a phase4_training.P4Spec."""
    n = len(spec.channels)
    if spec.arch == "dual":
        groups = group_indices(spec.channels, spec.groups)
        dec = spec.dec_width
    else:
        groups, dec = [list(range(n))], spec.enc_width
    return GeoUNet(groups, n, enc_width=spec.enc_width, dec_width=dec, fusion=spec.fusion, context=spec.context,
                   context_factor=spec.context_factor, aux_dist=spec.aux_dist > 0, aux_orient=spec.aux_orient > 0)


def matched_single_width(target_params: int, n_channels: int, lo: int = 12, hi: int = 64, **kw) -> tuple[int, int]:
    """Encoder width of a single-encoder GeoUNet whose parameter count is closest to ``target_params``."""
    best = None
    for w in range(lo, hi + 1):
        n = count_params(GeoUNet([list(range(n_channels))], n_channels, enc_width=w, **kw))
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (w, n)
    return best


def load_encoder_weights(model: GeoUNet, path, channels: list[str], device="cpu", group: int = 0) -> dict:
    """Load an SSL encoder (geophysical_ssl checkpoint) into ``model.encoders[group]``; channels must match."""
    obj = torch.load(path, map_location=device, weights_only=False)
    if list(obj["channels"]) != list(channels):
        raise ValueError(f"SSL encoder channels differ from the fine-tuning stack ({path})")
    model.encoders[group].load_state_dict(obj["encoder"], strict=True)
    return {"path": str(path), "ssl_exp": obj.get("exp_id"), "ssl_epochs": obj.get("epochs"), "strict": True}
