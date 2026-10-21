"""Experiment F4: controlled DEM benchmark on identical DEM-covered regions.

Comparisons (configs/experiments/f4_*.yaml): G1 vs G1 + DEM aggregates (LightGBM, NaN outside coverage is
native "missing" for LightGBM) and P4_B_gated vs P4_B_gated + DEM channels (zero-filled outside coverage plus
an explicit ``dem:coverage`` indicator channel). Training uses the whole map; *evaluation* of every model in a
comparison uses the same restricted context in which ``valid`` = data footprint & coverage >= ``min_coverage``.

Missing-coverage strategy is explicit: no DEM model is ever scored on the full competition raster. A
full-raster DEM model would need a coverage-aware fallback (e.g. the non-DEM champion outside coverage),
which is only worth building if F4 shows a compelling, fold-stable gain.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd


def coverage_mask(ctx, min_coverage: float = 0.9) -> np.ndarray:
    name = "dem:coverage"
    if name not in ctx.store.specs:
        raise FileNotFoundError("no DEM mosaic registered: run notebook 05 (dem.morphology.mosaic) first")
    if not ctx.store.has(name):
        ctx.store.build([name], verbose=False)
    return np.nan_to_num(np.asarray(ctx.store.get(name))) >= min_coverage


def covered_context(ctx, covered: np.ndarray):
    """Context copy whose scoring footprint is restricted to DEM-covered pixels (training is unaffected)."""
    return replace(ctx, data=replace(ctx.data, valid=ctx.data.valid & covered), _maps=dict(ctx._maps))


def coverage_report(ctx, covered: np.ndarray, min_label_px: int = 300) -> pd.DataFrame:
    """Per geographic fold and discovery split: covered area and labels -> is the comparison informative?"""
    rows = []
    for k in ctx.fold_ids:
        v = ctx.folds.val(k) & ctx.data.valid
        rows.append({"unit": f"fold{k}", "covered_km2": float((v & covered).sum() * 0.01),
                     "label_px": int((ctx.data.labels & v & covered).sum())})
    for d in range(ctx.splits.n_splits):
        w = ctx.splits.withheld(d)
        rows.append({"unit": f"disc{d}", "covered_km2": float((ctx.splits.eval_zone(d) & covered).sum() * 0.01),
                     "label_px": int((w & covered).sum())})
    df = pd.DataFrame(rows)
    df["informative"] = df.label_px >= min_label_px
    return df
