"""GEMS leaderboard focus, Milestone 1: recover, validate and export the strongest existing W2-based champion.

Thin adapters over P6 (``p6.full_benchmark``, ``p6.final_submission``, ``p6.export_parity``); no model is trained or
re-run. Entry point: ``workflow.Runner`` driven by ``notebooks/leaderboard_focus/GEMS_M1_CHAMPION_TO_SUBMISSION.ipynb``;
docs: ``docs/leaderboard_focus/``.
"""
from .config import M1_CODE_REV, M1_VERSION

__all__ = ["M1_CODE_REV", "M1_VERSION"]
