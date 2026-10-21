"""Pseudo-discovery evaluation: can a model rediscover fault systems withheld from supervision?

This is a controlled stress test of missing-label generalisation, *not* an unbiased estimate
of the hidden test score. It is nevertheless the closest internal proxy to the competition,
whose test labels are expert-mapped faults that are absent from the USGS training labels.

Scoring for split d:
* ``G``          = withheld fault pixels of split d;
* ``eval_mask``  = evaluation zone (within ``eval_km`` of withheld systems);
* ``exclude``    = retained (training-visible) labels dilated by R — predictions on faults the
                   model was trained on are neither credit nor penalty ("excl" variant);
                   the "noexcl" variant counts them as false positives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..data import NativeData
from ..folds import DiscoverySplits
from ..metric import clean_prediction, dti_components, local_weighted_max


def _dilate_m(mask: np.ndarray, data: NativeData, metres: float) -> np.ndarray:
    d = ndimage.distance_transform_edt(~mask, sampling=(data.grid.pixel_km_y * 1000, data.grid.pixel_km_x * 1000))
    return d <= metres


def evaluate_split(pred: np.ndarray, data: NativeData, splits: DiscoverySplits, d: int, model: str,
                   threshold: float = 0.5, tolerances=(100.0, 200.0, 300.0), extra_min_km: float = 1.0
                   ) -> tuple[dict, pd.DataFrame]:
    G = splits.withheld(d)
    ev = splits.eval_zone(d) & data.valid
    retained = data.labels & ~G
    excl = _dilate_m(retained, data, 300.0) & ~G
    c_ex = dti_components(pred, G, eval_mask=ev, exclude_mask=excl)
    c_no = dti_components(pred, G | (retained & ev), eval_mask=ev)  # all labels as truth, nothing excluded
    p = clean_prediction(pred)
    p[~data.valid] = 0
    px = (data.grid.pixel_km_y * 1000, data.grid.pixel_km_x * 1000)
    M = local_weighted_max(p, 300.0, px)
    hit = (p >= threshold) & data.valid
    d_hit = ndimage.distance_transform_edt(~hit, sampling=px) if hit.any() else np.full(p.shape, np.inf)
    sys_ids = splits.system_map[G]
    df = pd.DataFrame({"system_id": sys_ids, "M": M[G], "p": p[G],
                       **{f"hit{t:g}": d_hit[G] <= t for t in tolerances}})
    per_sys = df.groupby("system_id").agg(n_px=("M", "size"), weighted_recall=("M", "mean"),
                                          mean_p_on_trace=("p", "mean"),
                                          **{f"recall@{t:g}m": (f"hit{t:g}", "mean") for t in tolerances}).reset_index()
    # systems also has an n_px column (whole system); keep the scored-pixel count as n_px
    per_sys = per_sys.merge(splits.systems.rename(columns={"n_px": "system_n_px"}), on="system_id", how="left")
    per_sys["model"], per_sys["split"] = model, d
    # extra confident structures in the eval zone, unsupported by any mapped fault
    dist_all = ndimage.distance_transform_edt(~data.labels, sampling=(data.grid.pixel_km_y, data.grid.pixel_km_x))
    extra = hit & ev & (dist_all > extra_min_km)
    summary = {"model": model, "split": d, "n_systems": int(per_sys.system_id.nunique()),
               "withheld_px": int(G.sum()),
               "dti_excl_retained": c_ex.dti, "tp_w": c_ex.tp, "fp_w": c_ex.fp, "fn_w": c_ex.fn,
               "dti_no_exclusion": c_no.dti,
               "weighted_recall": c_ex.tp / max(c_ex.n_truth, 1),
               **{f"recall@{t:g}m": float(df[f"hit{t:g}"].mean()) for t in tolerances},
               "mean_p_on_withheld": float(df.p.mean()),
               "mean_p_eval_background": float(p[ev & (dist_all > extra_min_km)].mean()),
               "extra_confident_km2": float(extra.sum() * data.grid.pixel_km_x * data.grid.pixel_km_y),
               "system_recall_spread": float(per_sys.weighted_recall.std(ddof=0))}
    return summary, per_sys


def sensitivity_tables(per_sys: pd.DataFrame, seg_raster: np.ndarray | None = None,
                       seg_table: pd.DataFrame | None = None, splits: DiscoverySplits | None = None) -> pd.DataFrame:
    """Withheld-system recall by length bin, region and (when segments are available) dominant strike."""
    df = per_sys.copy()
    df["length_bin"] = pd.cut(df.trace_km, [0, 10, 20, 50, np.inf], labels=["5-10", "10-20", "20-50", ">50"])
    if seg_raster is not None and seg_table is not None and splits is not None:
        m = (seg_raster >= 0) & (splits.system_map >= 0)
        pairs = pd.DataFrame({"system_id": splits.system_map[m], "seg_id": seg_raster[m]}).drop_duplicates()
        pairs = pairs.merge(seg_table[["seg_id", "azimuth", "length_km"]], on="seg_id")
        az = pairs.groupby("system_id").apply(
            lambda g: float(np.degrees(np.angle(np.average(np.exp(2j * np.radians(g.azimuth)), weights=g.length_km))) / 2 % 180))
        df = df.merge(az.rename("system_strike").reset_index(), on="system_id", how="left")
        df["strike_sector"] = pd.cut(df.system_strike, np.arange(0, 181, 30), include_lowest=True).astype(str)
    rows = []
    for col in ("length_bin", "region_name", "strike_sector"):
        if col not in df:
            continue
        for (model, val), g in df.groupby(["model", col], observed=True):
            rows.append({"model": model, "stratum": col, "value": str(val), "n_systems": len(g),
                         "trace_km": float(g.trace_km.sum()),
                         "weighted_recall": float(np.average(g.weighted_recall, weights=g.n_px)),
                         "recall@300m": float(np.average(g["recall@300m"], weights=g.n_px))})
    return pd.DataFrame(rows)
