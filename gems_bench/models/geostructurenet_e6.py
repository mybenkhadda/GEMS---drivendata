"""E6 masked three-encoder GeoStructureNet.

Runs under the existing Phase-5 trainer/inference API: forward(x, ctx=None).
The last input channels are 100 m DEM summaries and a raw coverage mask.
The original terrain/potential channel indices and two-way gate remain intact.
At missing DEM pixels, the output is exactly the baseline output.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..geological_encoders import Encoder
from .geostructurenet import GeoStructureNet


class MaskedThreeWayGate(nn.Module):
    def __init__(self, in_dem: int, out_c: int, initial_bias=-2.0):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_dem, out_c, 1, bias=False),
                                  nn.BatchNorm2d(out_c))
        self.score_dem = nn.Conv2d(in_dem, 1, 1)
        nn.init.zeros_(self.score_dem.weight)
        nn.init.constant_(self.score_dem.bias, initial_bias)
        self.last_gate = None

    def forward(self, terrain, potential, dem, coverage, original):
        scores2 = original.gate(torch.cat((terrain, potential), dim=1))
        m = F.interpolate(coverage.float(), size=scores2.shape[-2:], mode="area").clamp_(0, 1)
        dem_logit = self.score_dem(dem).float() + m.clamp_min(1e-12).log()
        dem_logit = dem_logit.masked_fill(m <= 0, float("-inf"))
        weights = torch.softmax(torch.cat((scores2.float(), dem_logit), dim=1), dim=1)
        if not self.training:
            self.last_gate = weights.detach()
        ft = original.proj[0](terrain)
        fp = original.proj[1](potential)
        fd = self.proj(dem)
        return F.relu((weights[:, :1] * ft.float() +
                       weights[:, 1:2] * fp.float() +
                       weights[:, 2:3] * fd.float()).to(ft.dtype))


class GeoStructureNetE6(nn.Module):
    """E6 accepts a full P5 input stack: original channels, DEM features, coverage."""

    def __init__(self, channels: list[str], groups: dict[str, list[str]], *,
                 enc_width=24, dec_width=32, dem_width=16, long_range="none",
                 aux_dist=True, aux_orient=True, dem_dropout=0.30,
                 freeze_warm_epochs=3):
        super().__init__()
        if not 0 <= dem_dropout < 1:
            raise ValueError("Invalid E6 DEM modality dropout")
        if len(set(channels)) != len(channels):
            raise ValueError("Duplicate E6 stack channels")
        terrain = list(groups.get("terrain") or [])
        potential = list(groups.get("potential") or groups.get("subsurface") or [])
        dem = list(groups.get("dem") or [])
        mask_name = "dem:dem_coverage"
        if not terrain or not potential or not dem or mask_name not in channels:
            raise ValueError("E6 needs terrain, potential, dem and dem:dem_coverage")
        if any(n not in channels for n in [*terrain, *potential, *dem]):
            raise ValueError("E6 group includes unregistered channel")
        if mask_name in [*terrain, *potential, *dem]:
            raise ValueError("Coverage cannot enter a prediction encoder")
        if set(terrain) & set(potential) or set(terrain) & set(dem) or set(potential) & set(dem):
            raise ValueError("E6 modality groups must be disjoint")
        self._group_names = {"terrain": terrain, "potential": potential, "dem": dem}
        self._widths = (int(enc_width), int(dec_width), str(long_range))
        base_names = list(dict.fromkeys([*terrain, *potential]))
        self.register_buffer("base_idx", torch.tensor([channels.index(n) for n in base_names], dtype=torch.long),
                             persistent=False)
        self.register_buffer("dem_idx", torch.tensor([channels.index(n) for n in dem], dtype=torch.long),
                             persistent=False)
        self.coverage_idx = channels.index(mask_name)
        group_idx = [[base_names.index(n) for n in terrain],
                     [base_names.index(n) for n in potential]]
        self.baseline = GeoStructureNet(group_idx, len(base_names), enc_width=enc_width,
                                        dec_width=dec_width, fusion="gated",
                                        long_range=long_range, aux_dist=bool(aux_dist),
                                        aux_orient=bool(aux_orient))
        depth = len(self.baseline.fuse)
        self.dem_encoder = Encoder(len(dem), dem_width, depth=depth)
        self.dem_fuse = nn.ModuleList(
            [MaskedThreeWayGate(self.dem_encoder.widths[i], dec_width * 2**i)
             for i in range(depth)])
        self.modality_dropout = float(dem_dropout)
        self.freeze_warm_epochs = int(freeze_warm_epochs)
        self._frozen = False
        self._warm_started = False

    def warm_start(self, checkpoint, device="cpu"):
        """Load genuine E1 fold checkpoint, not a different fold's model."""
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        self._check_e1_schema(ck.get("spec") if isinstance(ck, dict) else None, checkpoint)
        state = ck.get("state_dict", ck.get("model", ck.get("model_state_dict", ck)))
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        loaded = self.baseline.load_state_dict(state, strict=False)
        # E1 had a score head only. New E6 proximity/orientation heads are
        # intentionally initialized fresh; every other baseline weight must match.
        allowed = ("dist_head.", "orient_head.")
        bad = [key for key in loaded.missing_keys if not key.startswith(allowed)]
        if bad or loaded.unexpected_keys:
            raise RuntimeError(f"E1/E6 warm-start baseline schema mismatch: missing={bad} "
                               f"unexpected={loaded.unexpected_keys}")
        self._warm_started = True
        return {"checkpoint": str(checkpoint), "matched_e1": True,
                "new_auxiliary_head_keys": loaded.missing_keys}

    def _check_e1_schema(self, spec: dict | None, checkpoint) -> None:
        """Refuse an E1 checkpoint whose encoders would receive other channels. Matching tensor shapes are not enough:
        two groups of equal size in another order, or one channel replaced by another, load without any error."""
        if not spec:
            raise RuntimeError(f"{checkpoint}: no spec stored with the E1 model; its channel order cannot be verified")
        groups = {k: list(v) for k, v in (spec.get("gsn_groups") or spec.get("groups") or {}).items() if k != "context"}
        ordered = list(groups.values())
        want = [self._group_names["terrain"], self._group_names["potential"]]
        problems = []
        if ordered != want:
            problems.append(f"encoder groups {list(groups)} with {[len(g) for g in ordered]} channels differ from E6 "
                            f"terrain / potential ({[len(g) for g in want]}) or their order")
        if spec.get("model") not in ("geostructurenet", None):
            problems.append(f"model {spec.get('model')} is not a GeoStructureNet")
        if (spec.get("gsn_groups") or {}).get("context"):
            problems.append("E1 checkpoint has a context branch; E6 baseline has none")
        enc, dec, lr = self._widths
        if int(spec.get("enc_width", enc)) != enc or int(spec.get("dec_width", dec)) != dec:
            problems.append(f"widths {spec.get('enc_width')}/{spec.get('dec_width')} != {enc}/{dec}")
        if str(spec.get("gsn_long_range", lr)) != lr or str(spec.get("gsn_fusion", "gated")) != "gated":
            problems.append(f"fusion {spec.get('gsn_fusion')} / long_range {spec.get('gsn_long_range')} != gated / {lr}")
        if problems:
            raise RuntimeError(f"E1 warm start refused ({checkpoint}): " + "; ".join(problems))

    def freeze_baseline(self):
        self._frozen = True
        for p in self.baseline.parameters():
            p.requires_grad_(False)
        self.baseline.eval()

    def unfreeze_all(self):
        self._frozen = False
        for p in self.parameters():
            p.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if mode and self._frozen:
            self.baseline.eval()
        return self

    def encoder_parameters(self):
        return [p for e in self.baseline.encoders for p in e.parameters()] + \
               list(self.dem_encoder.parameters())

    def gates(self):
        return [f.last_gate for f in self.dem_fuse]

    def context_weight(self):
        return self.baseline.context_weight()

    def _decode(self, features, ctx):
        b = self.baseline
        feats = list(features)
        if b.long_range is not None:
            feats[-1] = b.long_range(feats[-1])
        if b.context is not None:
            if ctx is None:
                raise ValueError("E6 context tensor missing")
            z = b.context(ctx.index_select(1, b.ctx_idx), feats[-1].shape[-2:])
            feats[-1] = feats[-1] + torch.sigmoid(b.context_gate) * z
        y = b.decoder(feats)
        out = {"seg": b.head(y)}
        if b.dist_head is not None:
            out["dist"] = b.dist_head(y)
        if b.orient_head is not None:
            out["orient"] = b.orient_head(y)
        return out

    def forward(self, x, ctx=None):
        if x.shape[1] <= self.coverage_idx:
            raise ValueError("E6 stack missing DEM coverage channel")
        base = x.index_select(1, self.base_idx)
        m = torch.nan_to_num(x[:, self.coverage_idx:self.coverage_idx+1], nan=0,
                             posinf=0, neginf=0).clamp(0, 1)
        if self.training and self.modality_dropout:
            keep = (torch.rand((len(x), 1, 1, 1), device=x.device) >= self.modality_dropout)
            m = m * keep.to(m.dtype)
        # Only a gate/availability constraint; NEVER passed to an encoder.
        branch = [enc(base.index_select(1, getattr(self.baseline, f"idx{i}")))
                  for i, enc in enumerate(self.baseline.encoders)]
        original = [f([branch[0][i], branch[1][i]]) for i, f in enumerate(self.baseline.fuse)]
        baseline_out = self._decode(original, ctx)
        if not torch.any(m > 0):
            return baseline_out
        dem = torch.nan_to_num(x.index_select(1, self.dem_idx), nan=0, posinf=0, neginf=0) * m
        dem_parts = self.dem_encoder(dem)
        fused = [gate(branch[0][i], branch[1][i], dem_parts[i], m, self.baseline.fuse[i])
                 for i, gate in enumerate(self.dem_fuse)]
        enhanced = self._decode(fused, ctx)
        # Enforce pixelwise parity even where the fused decoder receptive field
        # crosses a DEM boundary. No probability calibration at missing pixels.
        on = m > 0
        return {k: torch.where(on, enhanced[k], baseline_out[k]) for k in baseline_out}
