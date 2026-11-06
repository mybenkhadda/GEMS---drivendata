"""Warm-started E6 masked 3-way extension of the actual GEMS GeoStructureNet.

With missing DEM, predictions are pixelwise identical to the baseline even
when neighboring pixels have coverage. DEM mask enters gates only.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from ..geological_encoders import Encoder


class MaskedThreeWayGate(nn.Module):
    def __init__(self, n_dem, out_channels):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(n_dem, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels))
        self.logit = nn.Conv2d(n_dem, 1, 1)
        nn.init.zeros_(self.logit.weight)
        nn.init.constant_(self.logit.bias, -2.0)
        self.last_gate = None

    def forward(self, terrain, field, dem, coverage, original):
        scores2 = original.gate(torch.cat((terrain, field), dim=1))
        m = F.interpolate(coverage.float(), size=scores2.shape[-2:], mode="area").clamp(0, 1)
        log_dem = self.logit(dem).float() + torch.log(m.clamp_min(1e-12))
        log_dem = log_dem.masked_fill(m == 0, float("-inf"))
        weights = torch.softmax(torch.cat((scores2.float(), log_dem), dim=1), dim=1)
        if not self.training:
            self.last_gate = weights.detach()
        terrain_p = original.proj[0](terrain)
        field_p = original.proj[1](field)
        dem_p = self.project(dem)
        return F.relu((weights[:, 0:1] * terrain_p.float() +
                       weights[:, 1:2] * field_p.float() +
                       weights[:, 2:3] * dem_p.float()).to(terrain_p.dtype))


class GeoStructureNetE6(nn.Module):
    """Three-way DEM specialist with an unchanged E1/B0 two-way baseline."""

    def __init__(self, baseline, n_dem_channels, dem_width=16, modality_dropout=.30):
        super().__init__()
        if getattr(baseline, "fusion", None) != "gated" or len(baseline.encoders) != 2:
            raise ValueError("E6 requires a two-branch GeoStructureNet with gated fusion")
        if not 0 <= modality_dropout < 1:
            raise ValueError("modality_dropout must be in [0,1)")
        if int(n_dem_channels) < 1:
            raise ValueError("n_dem_channels must be positive")
        self.baseline = baseline
        self.n_dem_channels = int(n_dem_channels)
        self.modality_dropout = float(modality_dropout)
        self.dem_encoder = Encoder(n_dem_channels, dem_width, depth=len(baseline.fuse))
        self.dem_fuse = nn.ModuleList([
            MaskedThreeWayGate(self.dem_encoder.widths[i], baseline.head.in_channels * 2 ** i)
            for i in range(len(baseline.fuse))
        ])

    def warm_start(self, path, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=False)
        weights = state.get("model", state.get("model_state_dict", state))
        if not isinstance(weights, dict):
            raise ValueError("Checkpoint missing a model state_dict")
        if any(k.startswith("module.") for k in weights):
            weights = {k.removeprefix("module."): v for k, v in weights.items()}
        self.baseline.load_state_dict(weights, strict=True)
        return {"source_checkpoint": str(path), "strict_load": True}

    def freeze_baseline(self):
        for p in self.baseline.parameters():
            p.requires_grad_(False)
        for p in self.dem_encoder.parameters():
            p.requires_grad_(True)
        for p in self.dem_fuse.parameters():
            p.requires_grad_(True)
        self.baseline.eval()

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if mode and not any(p.requires_grad for p in self.baseline.parameters()):
            self.baseline.eval()
        return self

    def encoder_parameters(self):
        return list(self.dem_encoder.parameters())

    def gates(self):
        return [g.last_gate for g in self.dem_fuse]

    def context_weight(self):
        return self.baseline.context_weight()

    def _decode(self, levels, ctx):
        feats = list(levels)
        b = self.baseline
        if b.long_range is not None:
            feats[-1] = b.long_range(feats[-1])
        if b.context is not None:
            if ctx is None:
                raise ValueError("Baseline uses E5 context; pass the original context tensor")
            z = b.context(ctx.index_select(1, b.ctx_idx), feats[-1].shape[-2:])
            feats[-1] = feats[-1] + torch.sigmoid(b.context_gate) * z
        y = b.decoder(feats)
        out = {"seg": b.head(y)}
        if b.dist_head is not None:
            out["dist"] = b.dist_head(y)
        if b.orient_head is not None:
            out["orient"] = b.orient_head(y)
        return out

    def forward(self, x, ctx=None, dem=None, dem_mask=None):
        b = self.baseline
        branches = [enc(x.index_select(1, getattr(b, f"idx{i}")))
                    for i, enc in enumerate(b.encoders)]
        original = [fuser([branches[0][i], branches[1][i]])
                    for i, fuser in enumerate(b.fuse)]
        base_output = self._decode(original, ctx)
        if dem is None or dem_mask is None:
            return base_output
        if dem.ndim != 4 or dem.shape[1] != self.n_dem_channels or dem.shape[0] != x.shape[0]:
            raise ValueError("DEM shape must be B x n_dem_channels x H x W")
        if dem_mask.shape != (x.shape[0], 1, *x.shape[-2:]) or dem.shape[-2:] != x.shape[-2:]:
            raise ValueError("DEM coverage and data must match base 100 m patch size")
        m = torch.nan_to_num(dem_mask, nan=0, posinf=0, neginf=0).clamp(0, 1).to(x.dtype)
        if self.training and self.modality_dropout:
            dropped = (torch.rand((x.shape[0], 1, 1, 1), device=x.device) < self.modality_dropout)
            m = m * (~dropped).to(m.dtype)
        if not torch.any(m > 0):
            return base_output
        dem_clean = torch.nan_to_num(dem, nan=0, posinf=0, neginf=0) * m
        dem_feats = self.dem_encoder(dem_clean)
        fusion = [gate(branches[0][i], branches[1][i], dem_feats[i], m, b.fuse[i])
                  for i, gate in enumerate(self.dem_fuse)]
        dem_output = self._decode(fusion, ctx)
        # Decoder receptive fields cross tile/coverage borders. The final
        # hard fallback enforces exact per-pixel baseline parity at m=0.
        on_dem = m > 0
        return {key: torch.where(on_dem, dem_output[key], base_output[key])
                for key in base_output}
