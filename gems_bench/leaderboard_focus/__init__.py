"""GEMS leaderboard focus, Milestone 1: recover, validate and export the strongest existing W2-based champion.

Thin adapters over P6 (``p6.full_benchmark``, ``p6.final_submission``, ``p6.export_parity``); no model is trained or
re-run. Notebooks: ``notebooks/leaderboard_focus/``; docs: ``docs/leaderboard_focus/``.
"""
from .config import M1_CODE_REV, M1_VERSION

__all__ = ["M1_CODE_REV", "M1_VERSION"]
