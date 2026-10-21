"""Model complementarity and the fault-recovery atlas (which withheld systems each model recovers)."""
from __future__ import annotations

import pandas as pd

from .. import phase4_eval
from ..p3 import discovery3


def recovery_matrix(ctx, sc: dict, frac: float = 0.05, policy: str = "excl", thr: float = 0.5) -> pd.DataFrame:
    """One row per withheld system, one column per model: weighted recall at a matched budget (+ recovered flag)."""
    per = discovery3.systems_at_budget(ctx, sc["disc_objs"], frac, policy=policy)
    m = per.pivot_table(index=["split", "system_id", "trace_km"], columns="model", values="weighted_recall")
    m["n_models_recovering"] = (m >= thr).sum(1)
    return m.reset_index()


def complementarity(ctx, maps: dict, frac: float = 0.05, refs=("G1", "P4_B_gated"),
                    support: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pairwise table, per-model table) from ``phase4_eval.complementarity``; pass ``sc['support']`` to reuse."""
    refs = tuple(r for r in refs if r in maps)
    return phase4_eval.complementarity(ctx, maps, frac, refs=refs, support=support)
