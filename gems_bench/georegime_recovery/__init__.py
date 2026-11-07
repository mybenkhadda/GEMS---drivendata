"""GEMS GeoRegime V2 - discovery recovery, full-area validation and geological error atlas.

Inference-only diagnostics of the EXISTING GeoRegime V1 checkpoints (no training, no new architecture).
Entry point for ``notebooks/colab/GEMS_GEOREGIME_RECOVERY_V2.ipynb``::

    from gems_bench.georegime_recovery import RecoveryRun
    RUN = RecoveryRun(CFG)
"""
from .run import RecoveryRun

__all__ = ["RecoveryRun"]
