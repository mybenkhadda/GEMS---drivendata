"""GeoStructureNet E6 V2.

ReferenceNet (dual encoder, trained from scratch, then frozen and saved immutable):
    terrain encoder + field encoder -> TwoWayGate per level -> dilated bottleneck (+ pooled context)
    -> U-Net decoder -> heads
E6Net (same components + half-width DEM encoder, MaskedThreeWayGate per level), initialised from the reference.
E6System = E6Net + frozen ReferenceNet with the exact output-level fallback

    logit(x) = E6Net(x)        where dem_coverage(x) > 0
             = ReferenceNet(x) where dem_coverage(x) == 0

so every pixel without measured DEM coverage is predicted by the frozen reference, bit for bit.
"""
from __future__ import annotations

import torch
from torch import nn

from .context import Bottleneck, ContextBranch
from .decoder import Decoder
from .encoders import Encoder
from .fusion import MaskedThreeWayGate, TwoWayGate, pool_mask
from .heads import Heads


def model_spec(cfg: dict, schema: dict) -> dict:
    m = cfg["model"]
    g = schema["groups"]
    return {"terrain_in": len(g["terrain"]), "field_in": len(g["field"]), "dem_in": len(g["dem"]),
            "ctx_in": len(g["terrain"]) + len(g["field"]), "levels": m["levels"], "terrain_width": m["terrain_width"],
            "field_width": m["field_width"], "dem_width": m["dem_width"], "dilations": list(m["dilations"]),
            "context_width": m["context_width"], "context_extent": m["context_extent"],
            "dem_gate_bias": m["dem_gate_bias"]}


class _Shared(nn.Module):
    def __init__(self, s: dict):
        super().__init__()
        if s["terrain_width"] != s["field_width"]:
            raise ValueError("terrain_width must equal field_width (the gates mix them per channel)")
        self.enc_t = Encoder(s["terrain_in"], s["terrain_width"], s["levels"])
        self.enc_f = Encoder(s["field_in"], s["field_width"], s["levels"])
        self.widths = self.enc_t.widths
        self.context = ContextBranch(s["ctx_in"], s["context_width"], s["context_extent"])
        self.bottleneck = Bottleneck(self.widths[-1], s["context_width"], s["dilations"])
        self.decoder = Decoder(self.widths)
        self.heads = Heads(self.widths[0])

    def _tail(self, fused: list[torch.Tensor], ctx: torch.Tensor, aux: bool) -> dict:
        bottom = self.bottleneck(fused[-1], self.context(ctx))
        return self.heads(self.decoder(bottom, fused), aux)


class ReferenceNet(_Shared):
    def __init__(self, s: dict):
        super().__init__(s)
        self.gates = nn.ModuleList([TwoWayGate(w) for w in self.widths])

    def forward(self, terrain, field, ctx, aux: bool = True) -> dict:
        ft, ff = self.enc_t(terrain), self.enc_f(field)
        fused, ws = [], []
        for g, t, f in zip(self.gates, ft, ff):
            x, w = g(t, f)
            fused.append(x)
            ws.append(w)
        out = self._tail(fused, ctx, aux)
        out["gates"] = ws
        return out


class E6Net(_Shared):
    def __init__(self, s: dict):
        super().__init__(s)
        self.enc_d = Encoder(s["dem_in"], s["dem_width"], s["levels"])
        self.gates = nn.ModuleList([MaskedThreeWayGate(w, wd, s["dem_gate_bias"])
                                    for w, wd in zip(self.widths, self.enc_d.widths)])

    def forward(self, terrain, field, dem, gate_cov, ctx, aux: bool = True) -> dict:
        ft, ff, fd = self.enc_t(terrain), self.enc_f(field), self.enc_d(dem)
        fused, ws = [], []
        for lvl, (g, t, f, d) in enumerate(zip(self.gates, ft, ff, fd)):
            x, w = g(t, f, d, pool_mask(gate_cov, lvl))
            fused.append(x)
            ws.append(w)
        out = self._tail(fused, ctx, aux)
        out["gates"] = ws
        return out

    @torch.no_grad()
    def init_from_reference(self, ref: ReferenceNet) -> list[str]:
        """Copy every shared parameter; gates take the reference (s_T, s_F) path. Returns the copied keys."""
        copied = []
        for name in ("enc_t", "enc_f", "context", "bottleneck", "decoder", "heads"):
            getattr(self, name).load_state_dict(getattr(ref, name).state_dict(), strict=True)
            copied += [f"{name}.{k}" for k in getattr(ref, name).state_dict()]
        for i, (g3, g2) in enumerate(zip(self.gates, ref.gates)):
            g3.load_from_two_way(g2)
            copied += [f"gates.{i}.{k}" for k in g2.state_dict()]
        return copied

    def dem_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith("enc_d.") or (n.startswith("gates.") and any(k in n for k in ("adapt_d", "proj_d",
                                                                                         "logit_d"))):
                yield n, p


class E6System(nn.Module):
    """E6Net + frozen reference, exact fallback where coverage == 0 (inference and training)."""

    def __init__(self, e6: E6Net, reference: ReferenceNet):
        super().__init__()
        self.e6 = e6
        self.reference = reference
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.reference.eval()
        return self

    def forward(self, terrain, field, dem, cov, gate_cov, ctx, aux: bool = True) -> dict:
        out = self.e6(terrain, field, dem, gate_cov, ctx, aux)
        with torch.no_grad():
            ref = self.reference(terrain, field, ctx, aux=False)["logit"]
        covered = cov > 0
        out["logit_e6"] = out["logit"]
        out["logit"] = torch.where(covered, out["logit"], ref.to(out["logit"].dtype))
        out["logit_reference"] = ref
        return out


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
