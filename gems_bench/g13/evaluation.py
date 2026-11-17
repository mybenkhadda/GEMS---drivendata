"""Exact DTI evaluation, cross-fitted selection, stratified diagnostics and spatial uncertainty (sections 19-20).

Scoring always goes through ``gems_bench.metric`` / ``gems_bench.gems3.official`` (tested against an independent
KD-tree reference in ``tests/test_gems3.py``). Three kinds of numbers are kept apart in every table:

* ``sampled_*``   classification metrics on sampled pixels (diagnostics),
* ``proxy_dti``   full-raster DTI against withheld *known* faults (offline proxy),
* ``public_lb``   only ever read from the manual leaderboard log (never computed here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..metric import DTI_ALPHA, DTI_BETA, DTI_EPS, sampled_auc_ap


def dti_of(tp: float, fp: float, fn: float) -> float:
    return float(tp / (tp + DTI_ALPHA * fp + DTI_BETA * fn + DTI_EPS))


def pooled_dti(df: pd.DataFrame) -> float:
    return dti_of(df.tp_w.sum(), df.fp_w.sum(), df.fn_w.sum())


# =============================================================================== per-unit scoring
def continuous_components(score: np.ndarray, units) -> pd.DataFrame:
    """Components of the map as submitted (continuous values, no area budget) on every unit."""
    rows = []
    for u in units:
        c = u.components(u.crop(score))
        rows.append(
            {
                "task": u.task,
                "zone": u.zone,
                "tp_w": c.tp,
                "fp_w": c.fp,
                "fn_w": c.fn,
                "n_truth": c.n_truth,
                "pred_mass": c.pred_mass,
                "dti": c.dti,
            }
        )
    return pd.DataFrame(rows)


def area_table(score: np.ndarray, units, fracs) -> pd.DataFrame:
    """Binary top-a components per unit and area fraction (fast all-threshold curve, official masking)."""
    rows = []
    for u in units:
        df = u.curve(u.crop(score)).at_area(fracs)
        df.insert(0, "zone", u.zone)
        df.insert(0, "task", u.task)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def crossfit(table: pd.DataFrame, params: list[str], zone_col: str = "zone") -> tuple[pd.DataFrame, dict]:
    """Pick ``params`` for each zone on the *other* zones (pooled DTI); score the choice on the zone itself.

    Deterministic tie-break: the first parameter row in sorted order. The identity / smallest change should be
    listed first by the caller so that ties prefer the simpler option."""
    zones = sorted(table[zone_col].unique())
    keys = table[params].astype(str).agg("|".join, axis=1)
    t = table.assign(_key=keys.values)
    agg = t.groupby([zone_col, "_key"], sort=False)[["tp_w", "fp_w", "fn_w"]].sum().reset_index()
    first = t.drop_duplicates("_key").set_index("_key")[params]
    rows = []
    for z in zones:
        other = agg[agg[zone_col] != z].groupby("_key", sort=False)[["tp_w", "fp_w", "fn_w"]].sum()
        if other.empty:
            other = agg.groupby("_key", sort=False)[["tp_w", "fp_w", "fn_w"]].sum()
        s = other.tp_w / (other.tp_w + DTI_ALPHA * other.fp_w + DTI_BETA * other.fn_w + DTI_EPS)
        best = s.index[int(np.argmax(s.to_numpy()))]
        held = agg[(agg[zone_col] == z) & (agg._key == best)]
        rows.append(
            {
                zone_col: z,
                **first.loc[best].to_dict(),
                "dti_other": float(s.loc[best]),
                "tp_w": float(held.tp_w.sum()),
                "fp_w": float(held.fp_w.sum()),
                "fn_w": float(held.fn_w.sum()),
                "dti_heldout": dti_of(held.tp_w.sum(), held.fp_w.sum(), held.fn_w.sum()),
            }
        )
    per = pd.DataFrame(rows)
    alls = agg.groupby("_key", sort=False)[["tp_w", "fp_w", "fn_w"]].sum()
    s_all = alls.tp_w / (alls.tp_w + DTI_ALPHA * alls.fp_w + DTI_BETA * alls.fn_w + DTI_EPS)
    top = s_all.index[int(np.argmax(s_all.to_numpy()))]
    summary = {
        "dti_crossfit": pooled_dti(per),
        "dti_insample": float(s_all.loc[top]),
        "optimism": float(s_all.loc[top]) - pooled_dti(per),
        "zone_min": float(per.dti_heldout.min()),
        "zone_max": float(per.dti_heldout.max()),
        "params_all": first.loc[top].to_dict(),
        "n_zones": len(zones),
    }
    return per, summary


def model_scorecard(name: str, oof: np.ndarray | None, disc_maps: dict | None, sp, fracs) -> tuple[dict, dict]:
    """Geo (OOF on CV folds) and disc_official (per-split maps) DTI at cross-fitted area budgets.

    Returns (scorecard row, raw tables). ``disc_maps``: split -> full-grid score of the model trained without the
    withheld systems of that split."""
    row, tables = {"model": name}, {}
    if oof is not None:
        units = sp.units(("geo",))
        tab = area_table(oof, units, fracs)
        per, summ = crossfit(tab, ["target_frac"])
        tables["geo_area"], tables["geo_choice"] = tab, per
        row.update(
            geo_proxy_dti=summ["dti_crossfit"],
            geo_zone_min=summ["zone_min"],
            geo_optimism=summ["optimism"],
            geo_frac=summ["params_all"]["target_frac"],
        )
        cont = continuous_components(oof, units)
        row["geo_proxy_dti_continuous"] = pooled_dti(cont)
    if disc_maps:
        units = [u for u in sp.units(("disc_official",)) if u.zone in disc_maps]
        tabs = [area_table(disc_maps[u.zone], [u], fracs) for u in units]
        tab = pd.concat(tabs, ignore_index=True)
        per, summ = crossfit(tab, ["target_frac"])
        tables["disc_area"], tables["disc_choice"] = tab, per
        row.update(
            disc_proxy_dti=summ["dti_crossfit"],
            disc_zone_min=summ["zone_min"],
            disc_optimism=summ["optimism"],
            disc_frac=summ["params_all"]["target_frac"],
        )
        cont = pd.concat([continuous_components(disc_maps[u.zone], [u]) for u in units])
        row["disc_proxy_dti_continuous"] = pooled_dti(cont)
    return row, tables


# =============================================================================== stratified diagnostics
def per_truth_support(score: np.ndarray, unit, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """(truth (row, col) on the full grid, TP support of each truth pixel) for the top-``frac`` binary mask."""
    c = unit.curve(unit.crop(score))
    t = c.threshold_for_area([frac])[0]
    return c.truth_rc, c.per_truth_tp(t)


def stratified_recall(score: np.ndarray, units, fracs_by_zone: dict, strata: dict[str, np.ndarray]) -> pd.DataFrame:
    """Weighted recall (mean TP support of truth pixels) per stratum. ``strata``: name -> int raster of bins."""
    rows = []
    for u in units:
        rc, sup = per_truth_support(score, u, fracs_by_zone.get(u.zone, 0.02))
        if not len(sup):
            continue
        for sname, raster in strata.items():
            b = raster[rc[:, 0], rc[:, 1]]
            for v in np.unique(b):
                m = b == v
                rows.append(
                    {
                        "task": u.task,
                        "zone": u.zone,
                        "stratum": sname,
                        "bin": int(v),
                        "n_truth": int(m.sum()),
                        "tp_w": float(sup[m].sum()),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["task", "stratum", "bin"])[["n_truth", "tp_w"]].sum().reset_index()
    g["recall_w"] = g.tp_w / g.n_truth.clip(lower=1)
    return g


def length_strata(
    labels: np.ndarray, sysmap: np.ndarray, systems: pd.DataFrame, bins_km=(0, 2, 5, 10, 25, 1e9)
) -> np.ndarray:
    """Bin of the fault-system trace length of every label pixel (-1 elsewhere)."""
    lut = np.full(int(sysmap.max()) + 2, -1, np.int16)
    for sid, km in zip(systems.system_id, systems.trace_km):
        lut[int(sid)] = int(np.searchsorted(bins_km, km, side="right") - 1)
    out = np.full(labels.shape, -1, np.int16)
    m = sysmap >= 0
    out[m] = lut[sysmap[m]]
    return out


def proximity_strata(truth: np.ndarray, known: np.ndarray, bins_m=(0, 300, 1000, 3000, 1e9)) -> np.ndarray:
    """Distance of every pixel to the nearest *other* known fault, binned (proximity-to-known evaluation)."""
    d = (
        ndimage.distance_transform_edt(~(known & ~truth), sampling=(100.0, 100.0))
        if (known & ~truth).any()
        else np.full(truth.shape, 1e9)
    )
    return (np.searchsorted(bins_m, d, side="right") - 1).astype(np.int16)


# =============================================================================== spatial uncertainty
def block_components(score: np.ndarray, unit, frac: float | None, block_px: int = 200) -> pd.DataFrame:
    """TP/FP/FN contributions per spatial block (blocks of ``block_px`` on the full grid) for one unit.

    TP and FN are attributed to the block of each truth pixel, FP to the block of each predicted pixel, so the
    block sums equal the unit components (tested) and blocks can be bootstrap-resampled."""
    p = np.asarray(unit.crop(score), np.float64)
    c = unit.curve(p)
    if frac is not None:
        t = c.threshold_for_area([frac])[0]
        q = (np.nan_to_num(p, nan=-1) >= t).astype(np.float64) * unit.ev
    else:
        q = np.clip(np.nan_to_num(p), 0, 1) * unit.ev
    sc = unit.scorer
    q = q.copy()
    q[sc.zero] = 0.0
    sup = sc.support(q, cleaned=True)
    r0, c0 = unit.sl[0].start, unit.sl[1].start
    gr, gc = sc.g_rc
    bid_t = ((gr + r0) // block_px) * 10_000 + (gc + c0) // block_px
    rr, cc = np.nonzero(q > 0)
    fpv = q[rr, cc] * sc.fp_w[rr, cc]
    bid_p = ((rr + r0) // block_px) * 10_000 + (cc + c0) // block_px
    t = pd.DataFrame({"block": bid_t, "tp_w": sup, "fn_w": 1.0 - sup}).groupby("block").sum()
    f = pd.DataFrame({"block": bid_p, "fp_w": fpv}).groupby("block").sum()
    out = t.join(f, how="outer").fillna(0.0).reset_index()
    out.insert(0, "zone", unit.zone)
    out.insert(0, "task", unit.task)
    return out


def bootstrap_dti(blocks: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """Spatial block bootstrap of the pooled DTI (blocks resampled within each zone)."""
    rng = np.random.default_rng(seed)
    groups = [g[["tp_w", "fp_w", "fn_w"]].to_numpy() for _, g in blocks.groupby(["task", "zone"])]
    vals = []
    for _ in range(n_boot):
        s = np.zeros(3)
        for g in groups:
            s += g[rng.integers(0, len(g), len(g))].sum(0)
        vals.append(dti_of(*s))
    vals = np.asarray(vals)
    tot = blocks[["tp_w", "fp_w", "fn_w"]].sum()
    return {
        "dti": dti_of(tot.tp_w, tot.fp_w, tot.fn_w),
        "ci_lo": float(np.percentile(vals, 2.5)),
        "ci_hi": float(np.percentile(vals, 97.5)),
        "boot_sd": float(vals.std()),
        "n_blocks": int(len(blocks)),
    }


def paired_bootstrap(blocks_a: pd.DataFrame, blocks_b: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """Paired block bootstrap of DTI(b) - DTI(a) on identical resampled blocks (same task / zones)."""
    key = ["task", "zone", "block"]
    m = blocks_a.merge(blocks_b, on=key, how="outer", suffixes=("_a", "_b")).fillna(0.0)
    rng = np.random.default_rng(seed)
    groups = [g for _, g in m.groupby(["task", "zone"])]
    diffs = []
    for _ in range(n_boot):
        sa, sb = np.zeros(3), np.zeros(3)
        for g in groups:
            i = rng.integers(0, len(g), len(g))
            sa += g[["tp_w_a", "fp_w_a", "fn_w_a"]].to_numpy()[i].sum(0)
            sb += g[["tp_w_b", "fp_w_b", "fn_w_b"]].to_numpy()[i].sum(0)
        diffs.append(dti_of(*sb) - dti_of(*sa))
    d = np.asarray(diffs)
    a = m[["tp_w_a", "fp_w_a", "fn_w_a"]].sum().to_numpy()
    b = m[["tp_w_b", "fp_w_b", "fn_w_b"]].sum().to_numpy()
    return {
        "delta": dti_of(*b) - dti_of(*a),
        "ci_lo": float(np.percentile(d, 2.5)),
        "ci_hi": float(np.percentile(d, 97.5)),
        "p_le_0": float((d <= 0).mean()),
        "n_boot": n_boot,
    }


def concentration(blocks: pd.DataFrame, top: int = 3) -> float:
    """Share of the total TP carried by the ``top`` blocks: tests whether a gain depends on few fault segments."""
    t = blocks.tp_w.sort_values(ascending=False)
    return float(t.head(top).sum() / max(t.sum(), 1e-12))


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for many model-selection comparisons."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def sampled_classification(score: np.ndarray, truth: np.ndarray, mask: np.ndarray, seed: int = 0) -> dict:
    """Sampled ROC-AUC / AP on near-truth (<= 100 m) vs far (>= 1 km) pixels: a *diagnostic* population."""
    out = sampled_auc_ap(score, truth, mask, np.random.default_rng(seed), n=200_000)
    return {f"sampled_{k}": v for k, v in out.items()}
