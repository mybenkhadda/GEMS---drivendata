"""Model construction from a ``training.spec.P5Spec`` (Phase-4 GeoUNet or GeoStructureNet) + capacity matching."""
from __future__ import annotations


def _group_lists(spec) -> tuple[list[list[int]], list[int] | None]:
    from ..geological_encoders import group_indices

    groups = dict(spec.gsn_groups or spec.groups)
    ctx_names = groups.pop("context", None)
    idx = group_indices(spec.channels, groups)
    ctx_idx = [spec.channels.index(c) for c in ctx_names] if ctx_names else None
    return idx, ctx_idx


def build_model5(spec):
    """``spec.model``: "geounet" -> geological_encoders.build_geo_model (Phase-4, unchanged);
    "geostructurenet" -> models.geostructurenet.GeoStructureNet."""
    if spec.model == "geounet":
        from ..geological_encoders import build_geo_model

        return build_geo_model(spec)
    if spec.model == "geostructurenet":
        from .geostructurenet import GeoStructureNet

        groups, ctx_idx = _group_lists(spec)
        return GeoStructureNet(groups, len(spec.channels), enc_width=spec.enc_width, dec_width=spec.dec_width,
                               fusion=spec.gsn_fusion, long_range=spec.gsn_long_range, context_idx=ctx_idx,
                               context_factor=spec.context_factor, heads=spec.gsn_heads,
                               aux_dist=spec.aux_dist > 0, aux_orient=spec.aux_orient > 0)
    raise ValueError(f"unknown model {spec.model}")


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def match_width(spec, target_params: int, lo: int = 12, hi: int = 48) -> tuple[int, int]:
    """Encoder width (decoder width scaled in proportion) whose parameter count is closest to the target.

    Used so that every Experiment-E variant is compared with P4_B_gated at (approximately) equal capacity:
    an improvement must not be explainable by parameter count alone.
    """
    from dataclasses import replace

    ratio = spec.dec_width / max(spec.enc_width, 1)
    best = None
    for w in range(lo, hi + 1):
        s = replace(spec, enc_width=w, dec_width=max(int(round(w * ratio)), 8))
        n = count_params(build_model5(s))
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (w, n)
    return best
