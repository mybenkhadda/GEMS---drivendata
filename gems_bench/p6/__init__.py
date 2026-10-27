"""P6 fast track to the leaderboard (docs/p6/P6_FAST_TRACK.md, configs/p6/fast_track.yaml).

W0 ``probes``      leaderboard probes A-D, SHA-keyed log, interpretation
W1 ``seeds``       multi-seed rank averaging + honest 180 deg TTA of B0 / B1
W2 ``stack``       stacked LightGBM ranker (honest CNN ranks + G02 + potential fields + permitted known geometry)
W3 ``throughput``  M9 GeoFaultNet epoch-time breakdown and stop / go
W4 ``dem_pilot``   1 m DEM pilot gate (M7 processing reused)
W5 ``completion``  near-known completion (only after W0 says so)
W6 ``select``      ordered selection among gate-passing candidates

Orchestration: ``P6FastTrack(m9)`` in ``run``.
"""
from .run import P6_VERSION, BudgetExceeded, P6FastTrack, load_p6_config

__all__ = ["P6_VERSION", "BudgetExceeded", "P6FastTrack", "load_p6_config"]
