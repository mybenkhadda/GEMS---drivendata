"""Efficient, balanced, leakage-aware training samples.

ONE sample pool is drawn once for every training visibility, so features are read from the store once:

    label        mapped-fault pixels (ALL labels), thinned to one pixel per ``label_spacing_px`` cell and capped
                 per system; for a given fit only the rows on VISIBLE faults are kept (rows on hidden faults are
                 dropped, so the over-sampling of fault pixels never reveals where hidden faults are)
    background   uniform within ``block_km`` spatial blocks, allocation proportional to footprint area
                 (label-independent: hidden faults appear here at their natural rate - that is the PU setting)
    evidence     high label-free evidence (a store feature above its footprint quantile): hard-negative candidates
                 when far from visible faults (label-independent)

Every row keeps its stratum and inclusion probability. Weights (``compute_sample_weights``): positives are
reweighted per fault SYSTEM (equal / sqrt(length) / length) so a long system cannot dominate by pixel count;
unlabelled rows get the PU weight, likely-background rows full weight, hard negatives their own weight;
background weights are partially balanced across macro regions; the background total equals ``neg_to_pos`` x the
positive total. Deterministic seeds everywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .supervision import IGNORE, LIKELY_BACKGROUND, POSITIVE, UNLABELED


def sample_spatial_blocks(shape, block_km: float) -> np.ndarray:
    from .splits import block_ids

    return block_ids(shape, block_km)


def thin_positions(flat_idx: np.ndarray, shape, spacing_px: int, rng) -> np.ndarray:
    """At most one (random) pixel per ``spacing_px`` x ``spacing_px`` cell; sorted flat indices."""
    if spacing_px <= 1 or not flat_idx.size:
        return np.sort(flat_idx)
    r, c = np.unravel_index(flat_idx, shape)
    ncols = (shape[1] + spacing_px - 1) // spacing_px
    cell = (r // spacing_px).astype(np.int64) * ncols + (c // spacing_px)
    perm = rng.permutation(flat_idx.size)
    _, first = np.unique(cell[perm], return_index=True)
    return np.sort(flat_idx[perm[first]])


def sample_positive_pixels(labels: np.ndarray, system_map: np.ndarray, footprint: np.ndarray, rng, cfg: dict
                           ) -> pd.DataFrame:
    """Label stratum (all mapped faults; per-fit visibility is applied later)."""
    lab = np.asarray(labels, bool) & np.asarray(footprint, bool)
    idx = thin_positions(np.flatnonzero(lab), lab.shape, int(cfg.get("label_spacing_px", 3)), rng)
    sid = np.asarray(system_map).ravel()[idx]
    df = pd.DataFrame({"flat": idx, "system_id": sid})
    cap = int(cfg.get("per_system_cap", 2000))
    n_sys = pd.Series(np.asarray(system_map).ravel()[np.flatnonzero(lab)]).value_counts()
    keep = []
    for s, g in df.groupby("system_id", sort=True):
        keep.append(g if len(g) <= cap else g.sample(cap, random_state=int(rng.integers(1 << 31))))
    df = pd.concat(keep, ignore_index=True) if keep else df
    mx = int(cfg.get("label_max", 120000))
    if len(df) > mx:
        df = df.sample(mx, random_state=int(rng.integers(1 << 31)))
    cnt = df.groupby("system_id").flat.transform("size")
    df["incl_prob"] = (cnt / df.system_id.map(n_sys).astype(float)).astype(np.float32)
    df["stratum"] = "label"
    return df.sort_values("flat").reset_index(drop=True)


def sample_background_pixels(footprint: np.ndarray, blocks: np.ndarray, rng, cfg: dict) -> pd.DataFrame:
    """Background stratum: block-stratified uniform sample of the footprint (label-independent)."""
    fp = np.asarray(footprint, bool).ravel()
    b = np.asarray(blocks).ravel()
    idx = np.flatnonzero(fp)
    order = np.argsort(b[idx], kind="stable")
    idx = idx[order]
    bb = b[idx]
    ub, start, cnt = np.unique(bb, return_index=True, return_counts=True)
    total = int(cfg.get("background_total", 300000))
    alloc = np.maximum(np.round(total * cnt / cnt.sum()).astype(int), int(cfg.get("background_min_per_block", 20)))
    alloc = np.minimum(alloc, cnt)
    rows = []
    for u, s, c, a in zip(ub, start, cnt, alloc):
        pick = idx[s + rng.choice(c, int(a), replace=False)] if a < c else idx[s:s + c]
        rows.append(pd.DataFrame({"flat": pick, "block": int(u), "incl_prob": np.float32(a / c)}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["flat", "block", "incl_prob"])
    df["stratum"] = "background"
    return df.sort_values("flat").reset_index(drop=True)


def sample_hard_negatives(evidence: np.ndarray, footprint: np.ndarray, rng, cfg: dict) -> pd.DataFrame:
    """Evidence stratum: pixels whose label-free evidence exceeds its footprint quantile (label-independent)."""
    fp = np.asarray(footprint, bool)
    e = np.asarray(evidence, np.float64)
    ok = fp & np.isfinite(e)
    if not ok.any():
        return pd.DataFrame(columns=["flat", "incl_prob", "stratum"])
    thr = np.quantile(e[ok], float(cfg.get("evidence_quantile", 0.95)))
    cand = np.flatnonzero((ok & (e >= thr)).ravel())
    n = min(int(cfg.get("evidence_total", 60000)), cand.size)
    pick = np.sort(rng.choice(cand, n, replace=False)) if n < cand.size else cand
    return pd.DataFrame({"flat": pick, "incl_prob": np.float32(n / max(cand.size, 1)), "stratum": "evidence"})


def build_pool(labels, system_map, footprint, evidence, cfg: dict) -> pd.DataFrame:
    """The deduplicated pool: one row per pixel with stratum membership flags (primary stratum: label > evidence >
    background) and the inclusion probability of the primary stratum."""
    rng = np.random.default_rng(int(cfg.get("seed", 2026)))
    shape = np.asarray(footprint).shape
    blocks = sample_spatial_blocks(shape, float(cfg.get("block_km", 20.0)))
    P = sample_positive_pixels(labels, system_map, footprint, rng, cfg)
    B = sample_background_pixels(footprint, blocks, rng, cfg)
    E = sample_hard_negatives(evidence, footprint, rng, cfg) if evidence is not None else \
        pd.DataFrame(columns=["flat", "incl_prob", "stratum"])
    allp = pd.concat([P[["flat", "incl_prob", "stratum"]], E[["flat", "incl_prob", "stratum"]],
                      B[["flat", "incl_prob", "stratum"]]], ignore_index=True)
    flags = pd.crosstab(allp.flat, allp.stratum).reindex(columns=["label", "evidence", "background"], fill_value=0)
    prim = allp.drop_duplicates("flat", keep="first").set_index("flat")
    out = pd.DataFrame({"flat": flags.index.to_numpy()})
    out["stratum"] = prim.loc[out.flat, "stratum"].to_numpy()
    out["incl_prob"] = prim.loc[out.flat, "incl_prob"].to_numpy().astype(np.float32)
    for s in ("label", "evidence", "background"):
        out[f"in_{s}"] = flags[s].to_numpy() > 0
    r, c = np.unravel_index(out.flat.to_numpy(), shape)
    out["row"], out["col"] = r.astype(np.int32), c.astype(np.int32)
    out["block"] = np.asarray(blocks)[r, c]
    out["row_id"] = np.arange(len(out), dtype=np.int64)
    return out


def compute_sample_weights(T: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Weights for one fit's labelled pool (columns ``cls``, ``stratum``, ``system_id``, ``row``, ``col``)."""
    cls = T["cls"].to_numpy()
    w = np.zeros(len(T), np.float64)
    pos = cls == POSITIVE
    if pos.any():
        sid = T.loc[pos, "system_id"].to_numpy()
        n_rows = pd.Series(sid).map(pd.Series(sid).value_counts()).to_numpy().astype(float)
        size = T.loc[pos, "system_px"].to_numpy().astype(float) if "system_px" in T else n_rows
        mode = str(cfg.get("system_weight", "sqrt"))
        ws = np.ones_like(size) if mode == "equal" else (np.sqrt(np.maximum(size, 1)) if mode == "sqrt"
                                                          else np.maximum(size, 1))
        w[pos] = ws / n_rows
        w[pos] /= w[pos].sum()
    bg = np.isin(cls, (UNLABELED, LIKELY_BACKGROUND))
    if bg.any():
        base = np.where(cls == UNLABELED, float(cfg.get("unlabeled_weight", 0.3)),
                        float(cfg.get("likely_background_weight", 1.0)))
        hard = (T["stratum"].to_numpy() == "evidence") & (T["dist_km"].to_numpy() >= float(
            cfg.get("hard_negative_min_km", 1.0))) if "dist_km" in T else np.zeros(len(T), bool)
        base = np.where(hard, float(cfg.get("hard_negative_weight", 1.5)), base)
        alpha = float(cfg.get("region_balance_alpha", 0.5))
        if alpha > 0:
            rk = float(cfg.get("region_km", 60.0))
            reg = (T["row"].to_numpy() // int(rk / 0.1)) * 10000 + (T["col"].to_numpy() // int(rk / 0.1))
            rc = pd.Series(reg[bg]).value_counts()
            share = pd.Series(reg).map(rc).to_numpy().astype(float)
            base = np.where(bg, base * (rc.mean() / np.maximum(share, 1.0)) ** alpha, base)
        w[bg] = base[bg]
        tot_pos = w[pos].sum() if pos.any() else 1.0
        w[bg] *= float(cfg.get("neg_to_pos", 1.0)) * tot_pos / max(w[bg].sum(), 1e-12)
    w[cls == IGNORE] = 0.0
    return w.astype(np.float32)
