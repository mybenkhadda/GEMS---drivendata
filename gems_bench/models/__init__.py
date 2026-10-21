"""Models: GeoStructureNet (Phase 5), factory over Phase-4 GeoUNet / GeoStructureNet, catalogue of baselines.

Torch is imported lazily: ``from gems_bench.models import zoo`` works without torch.
"""
from . import zoo

__all__ = ["zoo"]
