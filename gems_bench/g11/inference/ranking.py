"""Apply a ranker score and select candidates under a total-length budget (controlled predicted length)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_by_length_budget(scores: np.ndarray, lengths_m: np.ndarray, budget_km: float) -> np.ndarray:
    """Greedy by descending score until the cumulative length would exceed ``budget_km``; returns a bool mask."""
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    cum = np.cumsum(lengths_m[order]) / 1000.0
    take = np.zeros(len(scores), bool)
    take[order[cum <= budget_km]] = True
    return take


def budget_table(scores, lengths_m, labels, budgets_km) -> pd.DataFrame:
    rows = []
    for b in budgets_km:
        m = select_by_length_budget(scores, lengths_m, b)
        tp_len = float(lengths_m[m & (labels > 0)].sum()) / 1000.0
        rows.append({"budget_km": b, "n_selected": int(m.sum()), "selected_km": float(lengths_m[m].sum()) / 1000.0,
                     "positive_km": tp_len, "length_precision": tp_len / max(lengths_m[m].sum() / 1000.0, 1e-9),
                     "positive_recall": float((m & (labels > 0)).sum() / max((labels > 0).sum(), 1))})
    return pd.DataFrame(rows)
