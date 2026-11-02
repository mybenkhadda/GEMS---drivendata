"""G06 - geological feature store and multi-task ML benchmark.

Transforms the 19-band GeoDAWN raster into a reusable, fold-safe geological feature store (tiled GeoTIFF groups +
Parquet registry), builds pixel / patch / segment / connection datasets with explicit label scope, and benchmarks
classical pixel models, geological segment classifiers, a compact U-Net and cross-fitted meta-models against the
frozen W2 under the canonical official-proxy protocol (``gems3.official`` + ``m7.protocol``).

Entry points (heavy dependencies are imported lazily, never at package import)::

    from gems_bench.g06 import load_config, G06Pipeline
    cfg = load_config(overrides={"notebook": CONFIG, "run": RUN})
    P = G06Pipeline(cfg, RUN, HEAD)
    P.run()

Docs: ``docs/g06/``. Nothing in this package uploads a submission.
"""
from __future__ import annotations

G06_VERSION = "g06.1"
G06_CODE_REV = "g06.1-r1"

__all__ = ["G06_VERSION", "G06_CODE_REV", "load_config", "validate_config", "compute_run_id", "G06Pipeline"]


def __getattr__(name: str):
    if name in ("load_config", "validate_config", "compute_run_id"):
        from . import config

        return getattr(config, name)
    if name == "G06Pipeline":
        from .pipeline import G06Pipeline

        return G06Pipeline
    raise AttributeError(name)
