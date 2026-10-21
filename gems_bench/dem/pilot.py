"""Experiment F2: balanced, reproducible pilot tile selection (selection never looks at DEM content).

Each candidate tile (in the study area, one URL per tile) is described on the native grid by:
label density, short mapped faults missed by G1, withheld systems missed by the reference CNN, model
disagreement (G1 vs gated dual encoder rank difference), and 3 x 3 region. The pilot then takes

* quota tiles for each scientific question (short faults missed by G1, systems missed by the CNN, high
  disagreement), drawn at random among qualifying tiles - not the "best looking" ones;
* control tiles without mapped faults (false-positive behaviour of DEM features);
* stratified random tiles (region x label-density tertile) to keep the sample representative.

The reason for every selection is recorded; the seed makes the pilot reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rasterio.transform import rowcol


def _window(ctx, r):
    t = ctx.data.grid.transform
    r0, c0 = rowcol(t, r.west, r.north)
    r1, c1 = rowcol(t, r.east, r.south)
    H, W = ctx.data.shape
    return max(min(r0, r1), 0), min(max(r0, r1), H), max(min(c0, c1), 0), min(max(c0, c1), W)


def tile_descriptors(ctx, tiles: pd.DataFrame, *, short_missed: np.ndarray | None = None,
                     cnn_missed: np.ndarray | None = None, maps: dict | None = None) -> pd.DataFrame:
    """Per tile: label density, counts of missed-short / missed-system pixels, mean rank disagreement, region."""
    from ..geo_eval import region_map
    from ..p3.transforms import quantile_map

    reg = region_map(ctx.data)
    dis = None
    if maps and len(maps) >= 2:
        a, b = list(maps.values())[:2]
        v = ctx.data.valid
        dis = np.abs(quantile_map(a, v, None) - quantile_map(b, v, None))
    rows = []
    for r in tiles.itertuples():
        r0, r1, c0, c1 = _window(ctx, r)
        if r1 <= r0 or c1 <= c0:
            continue
        v = ctx.data.valid[r0:r1, c0:c1]
        lab = ctx.data.labels[r0:r1, c0:c1]
        rr = reg[r0:r1, c0:c1]
        row = {"tile_id": r.tile_id, "url": r.url, "valid_frac": float(v.mean()), "label_density": float(lab[v].mean()) if v.any() else 0.0,
               "region": int(np.bincount(rr[rr >= 0]).argmax()) if (rr >= 0).any() else -1}
        row["short_missed_px"] = int(short_missed[r0:r1, c0:c1].sum()) if short_missed is not None else 0
        row["cnn_missed_px"] = int(cnn_missed[r0:r1, c0:c1].sum()) if cnn_missed is not None else 0
        row["disagreement"] = float(np.nanmean(dis[r0:r1, c0:c1][v])) if dis is not None and v.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def select_pilot(desc: pd.DataFrame, *, n_total: int = 24, quotas: dict | None = None, seed: int = 2026) -> pd.DataFrame:
    q = {"short_missed": 5, "cnn_missed": 5, "disagreement": 4, "control": 4, **(quotas or {})}
    rng = np.random.default_rng(seed)
    d = desc[desc.valid_frac >= 0.3].copy()
    chosen: dict[str, str] = {}

    def take(pool: pd.DataFrame, n: int, reason: str):
        pool = pool[~pool.tile_id.isin(chosen)]
        if len(pool) == 0 or n <= 0:
            return
        for t in pool.tile_id.to_numpy()[rng.permutation(len(pool))[:n]]:
            chosen[t] = reason

    take(d[d.short_missed_px > 0], q["short_missed"], "short faults missed by G1")
    take(d[d.cnn_missed_px > 0], q["cnn_missed"], "withheld systems missed by the reference CNN")
    if d.disagreement.notna().any():
        take(d[d.disagreement >= d.disagreement.quantile(0.8)], q["disagreement"], "high G1 / gated-encoder disagreement")
    take(d[d.label_density == 0], q["control"], "control: no mapped faults")
    rest = n_total - len(chosen)
    if rest > 0:
        d["density_tertile"] = pd.qcut(d.label_density.rank(method="first"), 3, labels=False)
        strata = list(d.groupby(["region", "density_tertile"]))
        rng.shuffle(strata)
        i = 0
        while rest > 0 and i < 10 * len(strata):
            _, g = strata[i % len(strata)]
            before = len(chosen)
            take(g, 1, "stratified random (region x label-density tertile)")
            rest -= len(chosen) - before
            i += 1
    out = d[d.tile_id.isin(chosen)].copy()
    out["reason"] = out.tile_id.map(chosen)
    return out.reset_index(drop=True)


def missed_short_faults(ctx, g1: np.ndarray, frac: float = 0.05, max_km: float = 2.0) -> np.ndarray:
    """Label pixels on traces < ``max_km`` that G1's matched-budget mask does not reach within 300 m."""
    from scipy import ndimage

    from .. import phase4_eval
    from ..geology.skeleton import trace_strata

    s = trace_strata(ctx.data.labels, ctx.data.grid)
    short = s["len_lt1"] | s["len_1_2"] if max_km <= 2.0 else s["len_lt1"] | s["len_1_2"] | s["len_2_5"]
    mask = phase4_eval.fold_masks(ctx, g1, frac)
    d = ndimage.distance_transform_edt(~mask, sampling=(100.0, 100.0))
    return short & (d > 300)


def missed_withheld(ctx, model: str, frac: float = 0.05) -> np.ndarray:
    """Withheld-system pixels (any split) not reached by ``model``'s discovery mask within 300 m."""
    from scipy import ndimage

    from ..p3.budget import binary_mask

    out = np.zeros(ctx.data.shape, bool)
    for d in range(ctx.splits.n_splits):
        p = ctx.discovery_pred(model, d)
        if p is None:
            continue
        ez = ctx.splits.eval_zone(d) & ctx.data.valid
        m = binary_mask(np.where(ez, p, np.nan), ez, None, frac)
        out |= ctx.splits.withheld(d) & (ndimage.distance_transform_edt(~m, sampling=(100.0, 100.0)) > 300)
    return out
