"""Stage J: ablations and explainability (pure helpers; the refits run in ``runner.sJ_ablation``).

* Family ablation: the same cross-fitted model refitted with one registry family removed (``registry.ABLATION_MAP``),
  evaluated at the model's fixed-policy cell; the delta vs the full model is descriptive (one run, no CI).
* Contributions: LightGBM ``pred_contrib`` (TreeSHAP) of the residual correction on held-out rows; mean |contribution|
  per feature and per family; per target system (segments within 300 m, post-hoc) for the atlas reading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FAMILY


def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """(n, features) TreeSHAP contributions of the correction (bias column dropped)."""
    C = np.asarray(model.predict(X, pred_contrib=True), float)
    return pd.DataFrame(C[:, :X.shape[1]], columns=list(X.columns), index=X.index)


def importance(C: pd.DataFrame) -> pd.DataFrame:
    m = C.abs().mean().rename("mean_abs_contrib").reset_index().rename(columns={"index": "feature"})
    m["family"] = m.feature.map(FAMILY).fillna("other")
    m["signed_mean"] = C.mean().to_numpy()
    return m.sort_values("mean_abs_contrib", ascending=False).reset_index(drop=True)


def family_importance(I: pd.DataFrame) -> pd.DataFrame:
    g = I.groupby("family").mean_abs_contrib.sum().rename("mean_abs_contrib").reset_index()
    g["share"] = g.mean_abs_contrib / max(float(g.mean_abs_contrib.sum()), 1e-12)
    return g.sort_values("share", ascending=False).reset_index(drop=True)


def ablation_deltas(S: pd.DataFrame, full: str) -> pd.DataFrame:
    """Per scenario: DTI of each ablated model minus the full model (fixed-policy summaries)."""
    rows = []
    for sc, g in S.groupby("scenario"):
        f = g[g.model == full]
        if not len(f):
            continue
        for r in g[g.model != full].itertuples():
            rows.append({"scenario": sc, "model": r.model, "removed": r.model.split("-minus-")[-1],
                         "dti": r.dti, "dti_full": float(f.dti.iloc[0]), "delta_vs_full": r.dti - float(f.dti.iloc[0]),
                         "reading": "family carried signal" if r.dti < float(f.dti.iloc[0]) else
                         "no measurable contribution (single run)"})
    return pd.DataFrame(rows)
