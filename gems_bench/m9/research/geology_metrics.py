"""G01 evaluation: equal-area budgets, distance-tolerant precision / recall, enrichment, spatial-block bootstrap.

Separate from the official competition score by design (the official-style DTI of the same binary masks is reported
in its own table, ``official_style``, for orientation only).

Protocol (per held-out unit = one geographic fold or one discovery split)
    domain   the unit's eligible budget domain (``m9.elig``: evaluation zone, template finite, known faults excluded on
             disc_official) AND interior pixels (>= ``boundary_margin_m`` from the data edge). Identical for every
             candidate and baseline.
    budget   exactly k = round(frac x |domain|) pixels per candidate (``equal_area_mask``: deterministic tie-break by
             pixel index; a candidate with fewer finite pixels than k selects all of them and the shortfall is
             recorded). Thresholds come from the area fraction only, never from labels.
    truth    the unit's scored truth (``u.eval_truth()``) inside the interior.
    metrics  at each distance d (metres, Euclidean on the actual pixel size):
                 precision@d = |selected within d of truth| / |selected|
                 recall@d    = |truth within d of a selected pixel| / |truth|
                 enrichment@d = precision@d / (share of the domain within d of truth)   (random expectation = 1)
             and the precision ratio against the simple-edge baseline at the same budget.
    blocks   every count is kept per spatial block (``block_km``); pooled numbers are sums of counts; uncertainty is a
             paired block bootstrap (blocks resampled within each zone, the same draw for both maps). Pixels are never
             treated as independent samples. Selections are fixed (conditional on the maps).
    nested   any choice among variants (wavelength, tolerance) is made on the OTHER zones (``nested_choice``).

Leakage guards: scores are label-free (the builder has no label argument); percentile references are fitted on the
unit's training geography and ``assert_disjoint`` refuses any overlap with the evaluation zone; budgets are area
fractions; no post-processing is tuned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .geology_io import LeakageError

COUNTS = ("n_pred", "n_pred_hit", "n_truth", "n_truth_hit", "n_domain", "n_domain_near")


# =============================================================================== geography
def reference_mask(ctx, task: str, zone: int) -> np.ndarray:
    """Full-grid reference geography of one held-out unit (where label-free percentile tables may be fitted).

    geo: the fold's training role (inner validation included; the buffer and the validation zone excluded).
    disc_*: every valid pixel outside the split's evaluation zone (which contains the withheld systems + ring)."""
    valid = np.asarray(ctx.data.valid, bool)
    if task == "geo":
        return np.asarray(ctx.folds.train(int(zone)), bool) & valid
    return valid & ~np.asarray(ctx.splits.eval_zone(int(zone)), bool)


def full_mask(shape, sl, crop: np.ndarray) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[sl] = np.asarray(crop, bool)
    return m


def assert_disjoint(reference: np.ndarray, evaluation: np.ndarray, what: str = "") -> None:
    n = int((np.asarray(reference, bool) & np.asarray(evaluation, bool)).sum())
    if n:
        raise LeakageError(f"{what}: reference geography overlaps the evaluation zone on {n} pixels")


def interior_mask(valid: np.ndarray, margin_m: float, pixel_m, touches=(True, True, True, True)) -> np.ndarray:
    """Valid pixels at least ``margin_m`` metres from invalid pixels. ``touches`` = (top, bottom, left, right): sides
    of the window that are the GRID edge (beyond them counts as invalid; other sides are interior context)."""
    v = np.asarray(valid, bool)
    if margin_m <= 0:
        return v.copy()
    pad = [(1 if touches[0] else 0, 1 if touches[1] else 0), (1 if touches[2] else 0, 1 if touches[3] else 0)]
    vp = np.pad(v, pad, mode="constant", constant_values=False)
    if vp.all():                                               # no invalid pixel in reach: every pixel is interior
        return v.copy()
    d = ndimage.distance_transform_edt(vp, sampling=(float(pixel_m[0]), float(pixel_m[1])))
    d = d[pad[0][0]:pad[0][0] + v.shape[0], pad[1][0]:pad[1][0] + v.shape[1]]
    return v & (d >= float(margin_m))


def block_index(sl, shape_crop, block_px: int) -> tuple[np.ndarray, np.ndarray]:
    """(compact block index per crop pixel, global block ids) for blocks of ``block_px`` on the FULL grid."""
    r0, c0 = sl[0].start or 0, sl[1].start or 0
    rr = (np.arange(shape_crop[0]) + r0) // block_px
    cc = (np.arange(shape_crop[1]) + c0) // block_px
    gid = rr[:, None].astype(np.int64) * 100000 + cc[None, :]
    uniq, inv = np.unique(gid.ravel(), return_inverse=True)
    return inv.reshape(shape_crop).astype(np.int32), uniq


# =============================================================================== selection + distances
def equal_area_k(domain: np.ndarray, frac: float) -> int:
    return int(round(float(frac) * int(np.asarray(domain, bool).sum())))


def equal_area_mask(score: np.ndarray, domain: np.ndarray, frac: float) -> np.ndarray:
    """Exactly k = round(frac |domain|) highest-scoring finite domain pixels (ties: lower flat index first)."""
    domain = np.asarray(domain, bool)
    s = np.asarray(score, np.float32)
    k = equal_area_k(domain, frac)
    idx = np.flatnonzero((domain & np.isfinite(s)).ravel())
    out = np.zeros(domain.shape, bool)
    if k <= 0 or not idx.size:
        return out
    v = s.ravel()[idx]
    order = np.lexsort((idx, -v.astype(np.float64)))           # primary: score desc; secondary: index asc
    out.ravel()[idx[order[:min(k, idx.size)]]] = True
    return out


def distance_m(mask: np.ndarray, pixel_m) -> np.ndarray:
    """Euclidean distance (metres) to the nearest True pixel; +inf everywhere when the mask is empty."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return ndimage.distance_transform_edt(~mask, sampling=(float(pixel_m[0]), float(pixel_m[1]))).astype(np.float32)


def unit_block_counts(scores: dict, domain: np.ndarray, truth: np.ndarray, pixel_m, fracs: dict, tols: list,
                      blk: np.ndarray, blk_ids: np.ndarray, tags: dict | None = None) -> pd.DataFrame:
    """Per (candidate, area fraction, distance, block): the six counts. ``fracs``: {candidate: [fractions]}."""
    domain = np.asarray(domain, bool)
    truth = np.asarray(truth, bool)
    nb = len(blk_ids)
    d_truth = distance_m(truth, pixel_m)
    bd = blk[domain]
    bt = blk[truth]
    n_dom = np.bincount(bd, minlength=nb)
    n_tru = np.bincount(bt, minlength=nb)
    near = {float(t): np.bincount(blk[domain & (d_truth <= float(t))], minlength=nb) for t in tols}
    rows = []
    for name, s in scores.items():
        n_fin = int((domain & np.isfinite(np.asarray(s, np.float32))).sum())
        for f in fracs.get(name, []):
            sel = equal_area_mask(s, domain, f)
            k = equal_area_k(domain, f)
            d_sel = distance_m(sel, pixel_m)
            n_sel = np.bincount(blk[sel], minlength=nb)
            for t in tols:
                t = float(t)
                hit = np.bincount(blk[sel & (d_truth <= t)], minlength=nb)
                thit = np.bincount(blk[truth & (d_sel <= t)], minlength=nb)
                keep = (n_dom > 0) | (n_tru > 0)
                for b in np.flatnonzero(keep):
                    rows.append((name, float(f), t, int(blk_ids[b]), int(n_sel[b]), int(hit[b]), int(n_tru[b]),
                                 int(thit[b]), int(n_dom[b]), int(near[t][b]), k, n_fin))
    df = pd.DataFrame(rows, columns=["candidate", "area_frac", "tol_m", "block", *COUNTS, "k_budget", "n_finite"])
    for c, v in (tags or {}).items():
        df[c] = v
    return df


# =============================================================================== pooled metrics
def _ratios(n_pred, n_pred_hit, n_truth, n_truth_hit, n_domain, n_domain_near) -> dict:
    prec = n_pred_hit / n_pred if n_pred else np.nan
    rec = n_truth_hit / n_truth if n_truth else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (np.isfinite(prec) and np.isfinite(rec) and prec + rec > 0) else np.nan
    base = n_domain_near / n_domain if n_domain else np.nan
    enr = prec / base if (np.isfinite(prec) and base and np.isfinite(base)) else np.nan
    return {"precision": prec, "recall": rec, "f1": f1, "random_precision": base, "enrichment": enr}


def summarize(blocks: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Pooled counts (sums over blocks) -> precision / recall / F1 / enrichment, grouped by ``by``."""
    if not len(blocks):
        return pd.DataFrame()
    g = blocks.groupby(by, dropna=False)[list(COUNTS)].sum().reset_index()
    m = pd.DataFrame([_ratios(*r) for r in g[list(COUNTS)].itertuples(index=False, name=None)])
    return pd.concat([g, m], axis=1)


def versus_edge(summary: pd.DataFrame, keys: list[str], ref: str = "edge") -> pd.DataFrame:
    """Precision ratio of every candidate against the simple-edge baseline at the same budget / distance."""
    if not len(summary) or ref not in set(summary.candidate):
        return summary.assign(precision_vs_edge=np.nan)
    e = summary[summary.candidate == ref][keys + ["precision"]].rename(columns={"precision": "edge_precision"})
    out = summary.merge(e, on=keys, how="left")
    out["precision_vs_edge"] = out.precision / out.edge_precision
    return out


METRICS = ("precision", "recall", "f1", "enrichment")


def paired_block_bootstrap(blocks: pd.DataFrame, cand: str, ref: str, metric: str = "f1", n_boot: int = 500,
                           seed: int = 2026, strata: tuple = ("task", "zone")) -> dict:
    """Paired spatial bootstrap of ``metric(cand) - metric(ref)``; ``blocks`` holds ONE (area_frac, tol_m)."""
    a = blocks[blocks.candidate == cand].set_index([*strata, "block"])[list(COUNTS)]
    b = blocks[blocks.candidate == ref].set_index([*strata, "block"])[list(COUNTS)]
    idx = a.index.union(b.index)
    res = {"candidate": cand, "reference": ref, "metric": metric}
    if not len(idx) or not len(a) or not len(b):
        return {**res, "delta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_delta_positive": np.nan, "n_blocks": 0}
    A = a.reindex(idx, fill_value=0).to_numpy(np.float64)
    B = b.reindex(idx, fill_value=0).to_numpy(np.float64)
    labels = pd.factorize(np.array([repr(t[:-1]) for t in idx]))[0] if len(strata) else np.zeros(len(idx), int)
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]

    def val(S):
        return _ratios(*S)[metric]

    obs = val(A.sum(0)) - val(B.sum(0))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        sel = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        draws[i] = val(A[sel].sum(0)) - val(B[sel].sum(0))
    draws = draws[np.isfinite(draws)]
    return {**res, "delta": float(obs), "ci_lo": float(np.quantile(draws, 0.05)) if draws.size else np.nan,
            "ci_hi": float(np.quantile(draws, 0.95)) if draws.size else np.nan,
            "p_delta_positive": float((draws > 0).mean()) if draws.size else np.nan, "n_blocks": int(len(idx)),
            "n_boot": int(n_boot)}


def bootstrap_table(blocks: pd.DataFrame, candidates: list[str], refs: list[str], frac: float, tol: float,
                    n_boot: int, seed: int, metrics=METRICS) -> pd.DataFrame:
    rows = []
    for task, g in blocks[(blocks.area_frac == float(frac)) & (blocks.tol_m == float(tol))].groupby("task"):
        have = set(g.candidate)
        for r in refs:
            if r not in have:
                continue
            for c in candidates:
                if c == r or c not in have:
                    continue
                for m in metrics:
                    rows.append({"task": task, "area_frac": float(frac), "tol_m": float(tol),
                                 **paired_block_bootstrap(g, c, r, m, n_boot, seed)})
    return pd.DataFrame(rows)


def per_block_uncertainty(blocks: pd.DataFrame, by: list[str], metric: str = "f1", n_boot: int = 500,
                          seed: int = 2026) -> pd.DataFrame:
    """Block-bootstrap 90 % interval of a pooled metric per group (zones as strata)."""
    rows = []
    rng = np.random.default_rng(seed)
    for key, g in blocks.groupby(by):
        S = g[list(COUNTS)].to_numpy(np.float64)
        labels = (pd.factorize(np.array([f"{a}|{b}" for a, b in zip(g.task, g.zone)]))[0]
                  if {"task", "zone"} <= set(g) else np.zeros(len(g), int))
        groups = [np.flatnonzero(labels == x) for x in np.unique(labels)]
        d = []
        for _ in range(int(n_boot)):
            sel = np.concatenate([x[rng.integers(0, len(x), len(x))] for x in groups])
            d.append(_ratios(*S[sel].sum(0))[metric])
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "metric": metric, "value": _ratios(*S.sum(0))[metric],
                     "ci_lo": float(np.quantile(d, 0.05)) if d.size else np.nan,
                     "ci_hi": float(np.quantile(d, 0.95)) if d.size else np.nan})
    return pd.DataFrame(rows)


def nested_choice(per_zone: pd.DataFrame, family: dict, metric: str = "f1") -> pd.DataFrame:
    """Leave-one-zone-out choice among variants: for each (task, zone) pick the variant with the best POOLED metric
    on the task's OTHER zones; report that variant's held-out metric. ``per_zone`` = ``summarize(blocks, [task, zone,
    candidate, ...])`` restricted to one budget / distance; ``family`` = {family name: [variant candidates]}."""
    rows = []
    for fam, variants in family.items():
        sub = per_zone[per_zone.candidate.isin(variants)]
        for task, g in sub.groupby("task"):
            for z in sorted(g.zone.unique()):
                other = g[g.zone != z]
                if not len(other):
                    continue
                pooled = summarize(other, ["candidate"]).set_index("candidate")[metric]
                if pooled.dropna().empty:
                    continue
                best = pooled.dropna().idxmax()
                held = g[(g.zone == z) & (g.candidate == best)]
                val = summarize(held, ["candidate"])[metric].iloc[0] if len(held) else np.nan
                rows.append({"family": fam, "task": task, "zone": int(z), "chosen": best, "metric": metric,
                             "held_out_value": val, "chosen_on_other_zones_value": float(pooled[best])})
    return pd.DataFrame(rows)


def official_style_rows(u, masks: dict, elig: np.ndarray, pkey: str) -> list[dict]:
    """Official-style components of each binary mask (separate table; NOT the G01 evaluation)."""
    from ...m7.protocol import score_row

    rows = []
    for (name, frac), q in masks.items():
        rows.append(score_row(u, np.asarray(q, np.float32), elig, pkey, method=name, area_frac=float(frac)))
    return rows


def pooled_official(rows: pd.DataFrame) -> pd.DataFrame:
    from ...m7.protocol import dti_of

    if not len(rows):
        return rows
    return (rows.groupby(["task", "method", "area_frac"])
            .apply(lambda g: pd.Series({"dti_pooled": dti_of(g),
                                        "worst_zone_dti": float(g.dti.min()) if "dti" in g else np.nan,
                                        "n_zones": int(len(g))}))
            .reset_index())
