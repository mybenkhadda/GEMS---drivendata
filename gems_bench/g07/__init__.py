"""G07 - high-resolution multiscale fault discovery.

Four coordinated systems on top of the frozen W2 and the G06 feature store:

    A  surface discovery      USGS 3DEP 1 m DEM -> native-resolution geomorphology -> 100 m evidence, surface models
    B  subsurface discovery   G06 feature store + improved supervision (system-balanced PU, hard negatives, distance)
    C  hypothesis ranking     label-free candidate pools (W2 / surface / subsurface / CNN / potential fields), learned
                              relevance with geometric positive augmentation, calibrated confidence
    D  DTI-aware selection    exact-k budgets with deterministic label-free tie-breaking, W2-preserving portfolio,
                              strict promotion + export parity

The first milestone is the DEM FEASIBILITY go / no-go (coverage, licensing, discriminative signal, incremental
recall). Heavy dependencies are imported lazily. Nothing here uploads a submission. Docs: ``docs/g07/``.
"""
from __future__ import annotations

G07_VERSION = "g07.1"
G07_CODE_REV = "g07.1-r2"

__all__ = ["G07_VERSION", "G07_CODE_REV", "load_config", "validate_config", "compute_run_id", "G07Pipeline"]


def __getattr__(name: str):
    if name in ("load_config", "validate_config", "compute_run_id"):
        from . import config

        return getattr(config, name)
    if name == "G07Pipeline":
        from .pipeline import G07Pipeline

        return G07Pipeline
    raise AttributeError(name)
