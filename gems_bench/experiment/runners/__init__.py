"""Experiment kinds -> runner classes (each implements prepare / train / evaluate / export)."""
from __future__ import annotations


def get_runner(kind: str):
    if kind == "zoo":
        from .zoo import ZooRunner as R
    elif kind == "parity":
        from .audit import ParityRunner as R
    elif kind == "splits":
        from .audit import SplitsRunner as R
    elif kind == "cnn5":
        from .cnn5 import Cnn5Runner as R
    elif kind == "lgbm5":
        from .lgbm5 import Lgbm5Runner as R
    elif kind == "ensemble5":
        from .ensemble5 import Ensemble5Runner as R
    elif kind == "dem_pilot":
        from .dem import DemPilotRunner as R
    elif kind == "submission":
        from .submission import SubmissionRunner as R
    else:
        raise ValueError(f"unknown experiment kind {kind!r}")
    return R


KINDS = ("zoo", "parity", "splits", "cnn5", "lgbm5", "ensemble5", "dem_pilot", "submission")
