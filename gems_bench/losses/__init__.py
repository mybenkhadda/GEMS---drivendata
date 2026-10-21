"""Phase-5 objectives: positive-unlabeled supervision (B), geometry-aware multi-task losses (C), consistency (B3)."""
from .compose import AUX_ORDER5, build_targets5, loss5

__all__ = ["AUX_ORDER5", "build_targets5", "loss5"]
