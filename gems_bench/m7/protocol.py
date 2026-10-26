"""One evaluation protocol for every M7 number.

* Units are the GEMS3 units (``gems3.official.build_units``): ``geo`` (truth = all mapped faults of the held-out fold),
  ``disc_official`` (truth = withheld systems, retained faults masked pixel-exactly, masked predictions ignored),
  ``disc_retained_fp`` (retained faults not masked: predictions on them are FP). Scoring is the verified FastScorer.
* Budget domain (``eligible``): the area fraction is counted and selected on pixels the scorer counts and the
  submission can carry: ``ev`` & template finite & not excluded (pixel-exact known faults on ``disc_official``) &
  not inside the optional suppression ring. The ring is an experiment, NOT an organizer exclusion.
* Protocol keys: every result row carries a hash of (task, zone, evaluation mask, scored truth, exclusion, TP
  convention, budget domain, template). ``assert_comparable`` refuses to compare rows whose keys differ for the
  same (task, zone) - e.g. a DEM-covered sub-zone against a full zone, or another budget domain.
* Area ledgers record the positive area after every masking step, as shares of ``ev`` and of the eligible domain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..gems3.official import Unit, build_units
from ..gems3.submission import ring_zone
from ..metric import clean_prediction
from .workspace import mask_hash

COMP = ["tp_w", "fp_w", "fn_w", "n_truth"]


class ProtocolMismatch(ValueError):
    """Two results that do not share folds / masks / budget domain / TP convention were about to be compared."""


def dti_of(df: pd.DataFrame) -> float:
    if not len(df):
        return np.nan
    tp, fp, fn = float(df.tp_w.sum()), float(df.fp_w.sum()), float(df.fn_w.sum())
    return tp / (tp + 0.2 * fp + 0.8 * fn + 1e-6)


def m7_units(ctx, cfg: dict, tasks=None) -> list[Unit]:
    """GEMS3 units for the configured tasks, restricted to the configured folds / splits."""
    ev = cfg["evaluation"]
    tasks = list(tasks or ev["tasks"])
    us = build_units(ctx, tasks)
    keep_f = None if ev["folds"] == "all" else set(int(x) for x in ev["folds"])
    keep_s = None if ev["splits"] == "all" else set(int(x) for x in ev["splits"])
    out = []
    for u in us:
        if u.task == "geo" and keep_f is not None and u.zone not in keep_f:
            continue
        if u.is_disc and keep_s is not None and u.zone not in keep_s:
            continue
        if not u.ev.any():
            continue
        out.append(u)
    return out


def template_crop(ctx, u: Unit) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(ctx.data.template_valid, bool)[u.sl])


def eligible(u: Unit, template: np.ndarray | None, ring_px: int = 0, domain: str = "eligible",
             include_template: bool = True) -> np.ndarray:
    """Budget domain of one unit (cropped)."""
    m = u.ev.copy()
    if include_template and template is not None:
        m &= template
    if domain == "evaluation_zone":
        return m
    if u.exclude is not None:
        m &= ~u.exclude
    if ring_px and u.known is not None:
        m &= ~ring_zone(u.known, int(ring_px))
    return m


def protocol_key(u: Unit, template: np.ndarray | None, domain: str, include_template: bool) -> str:
    from ..pipeline.m6 import fingerprint

    return fingerprint("m7-protocol", u.task, int(u.zone), mask_hash(u.ev), mask_hash(u.eval_truth()),
                       mask_hash(u.exclude), bool(u.supports_tp), domain, bool(include_template),
                       mask_hash(template) if include_template else "no-template")


def protocol_table(units: list[Unit], templates: dict, domain: str, include_template: bool) -> pd.DataFrame:
    rows = []
    for u in units:
        t = templates.get(u.key)
        rows.append({"task": u.task, "zone": u.zone, "unit": u.key,
                     "protocol_key": protocol_key(u, t, domain, include_template),
                     "eval_px": int(u.ev.sum()), "scored_truth_px": int(u.eval_truth().sum()),
                     "excluded_px": int(u.exclude.sum()) if u.exclude is not None else 0,
                     "eligible_px_ring0": int(eligible(u, t, 0, domain, include_template).sum()),
                     "supports_tp": bool(u.supports_tp), "budget_domain": domain,
                     "include_template": bool(include_template)})
    return pd.DataFrame(rows)


def assert_comparable(*tables: pd.DataFrame, by=("task", "zone")) -> None:
    """Every (task, zone) must carry exactly one protocol key across all tables (raises ProtocolMismatch)."""
    parts = [t[list(by) + ["protocol_key"]] for t in tables if t is not None and len(t) and "protocol_key" in t]
    if not parts:
        return
    allk = pd.concat(parts, ignore_index=True).drop_duplicates()
    n = allk.groupby(list(by)).protocol_key.nunique()
    bad = n[n > 1]
    if len(bad):
        raise ProtocolMismatch("incompatible protocols for " + ", ".join(f"{k}" for k in bad.index) +
                               ": different evaluation masks / exclusions / budget domain / template; results "
                               "from these rows cannot be compared or pooled")


# =============================================================================== area accounting
def unit_ledger(q: np.ndarray, u: Unit, elig: np.ndarray, step: str, ring_px: int = 0) -> dict:
    a = np.nan_to_num(np.asarray(q, np.float32), nan=0.0) > 0
    ring = ring_zone(u.known, int(ring_px)) if (ring_px and u.known is not None) else np.zeros(a.shape, bool)
    n_ev, n_el = max(int(u.ev.sum()), 1), max(int(elig.sum()), 1)
    exc = u.exclude if u.exclude is not None else np.zeros(a.shape, bool)
    return {"task": u.task, "zone": u.zone, "step": step, "positive_px": int((a & u.ev).sum()),
            "frac_of_ev": float((a & u.ev).sum() / n_ev), "frac_of_eligible": float((a & elig).sum() / n_el),
            "on_excluded_px": int((a & exc).sum()), "on_ring_px": int((a & ring).sum()),
            "outside_ev_px": int((a & ~u.ev).sum())}


def score_row(u: Unit, q: np.ndarray, elig: np.ndarray, pkey: str, **tags) -> dict:
    """Official components of one prediction + its area accounting (the row every table is built from)."""
    q = np.where(u.ev, np.nan_to_num(q, nan=0.0), 0.0).astype(np.float32)
    c = u.components(q)
    pos = q > 0
    return {"task": u.task, "zone": u.zone, **tags, **c.as_dict(), "protocol_key": pkey,
            "eval_px": int(u.ev.sum()), "eligible_px": int(elig.sum()), "area_px": int(pos.sum()),
            "area_frac_ev": float(pos.sum() / max(int(u.ev.sum()), 1)),
            "area_frac_eligible": float((pos & elig).sum() / max(int(elig.sum()), 1)),
            "masked_area_px": int((pos & ~elig).sum())}


# =============================================================================== systems
def system_ids(ctx, u: Unit) -> np.ndarray:
    """System id of every scored truth pixel (FastScorer order); -1 where unknown."""
    gr, gc = u.scorer.g_rc
    return np.asarray(ctx.splits.system_map)[u.sl][gr, gc]


def system_hits(ctx, u: Unit, q: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Per truth system of the unit: scored pixels, mean TP support, hit share, recovered flag."""
    ev = cfg["evaluation"]
    sup = u.scorer.support(np.asarray(q, np.float64))
    if not len(sup):
        return pd.DataFrame(columns=["system_id", "n_px", "support_mean", "hit_share", "recovered"])
    df = pd.DataFrame({"system_id": system_ids(ctx, u), "sup": sup, "hit": sup >= ev["system_recovered_support"]})
    df = df[df.system_id >= 0]
    g = df.groupby("system_id").agg(n_px=("sup", "size"), support_mean=("sup", "mean"), hit_share=("hit", "mean"))
    g["recovered"] = g.hit_share >= ev["system_recovered_frac"]
    return g.reset_index()


# =============================================================================== spatial blocks
def block_components(u: Unit, q: np.ndarray, block_px: int) -> pd.DataFrame:
    """Per spatial block (full-grid block index): TP, FP, FN contributions of one prediction (sums = components)."""
    s = u.scorer
    p = clean_prediction(np.where(u.ev, np.nan_to_num(q, nan=0.0), 0.0))
    p[s.zero] = 0.0
    r0, c0 = u.sl[0].start, u.sl[1].start
    fpx = p * s.fp_w
    rr, cc = np.nonzero(fpx > 0)
    nb_c = 100000
    fb = ((rr + r0) // block_px) * nb_c + (cc + c0) // block_px
    fp = pd.Series(fpx[rr, cc]).groupby(fb).sum()
    out = pd.DataFrame({"fp_w": fp})
    if s.n:
        gr, gc = s.g_rc
        sup = s.support(p, cleaned=True)
        tb = ((gr + r0) // block_px) * nb_c + (gc + c0) // block_px
        t = pd.DataFrame({"b": tb, "tp_w": sup, "fn_w": 1.0 - sup, "n_truth": 1}).groupby("b").sum()
        out = out.join(t, how="outer")
    out = out.fillna(0.0)
    for c in COMP:
        if c not in out:
            out[c] = 0.0
    out.index.name = "block"
    return out.reset_index().assign(task=u.task, zone=u.zone)


def paired_block_bootstrap(blocks: pd.DataFrame, method: str, ref: str, n_boot: int = 500, seed: int = 2026) -> dict:
    """Paired spatial bootstrap of the pooled-DTI difference ``method - ref``: blocks are resampled with replacement
    inside each zone, the same draw for both methods. ``blocks`` has columns method, zone, block + components."""
    rng = np.random.default_rng(seed)
    a = blocks[blocks.method == method].set_index(["zone", "block"])[COMP]
    b = blocks[blocks.method == ref].set_index(["zone", "block"])[COMP]
    idx = a.index.union(b.index)
    a, b = a.reindex(idx, fill_value=0.0), b.reindex(idx, fill_value=0.0)
    zones = idx.get_level_values(0).to_numpy()
    groups = [np.flatnonzero(zones == z) for z in np.unique(zones)]
    A, B = a.to_numpy(), b.to_numpy()

    def d(sa, sb):
        da = sa[0] / (sa[0] + 0.2 * sa[1] + 0.8 * sa[2] + 1e-6)
        db = sb[0] / (sb[0] + 0.2 * sb[1] + 0.8 * sb[2] + 1e-6)
        return da - db

    obs = d(A.sum(0), B.sum(0))
    if not len(idx):
        return {"method": method, "reference": ref, "delta_dti": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p_delta_positive": np.nan, "n_blocks": 0}
    draws = []
    for _ in range(int(n_boot)):
        sel = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups if len(g)])
        draws.append(d(A[sel].sum(0), B[sel].sum(0)))
    draws = np.asarray(draws)
    return {"method": method, "reference": ref, "delta_dti": float(obs), "ci_lo": float(np.quantile(draws, 0.05)),
            "ci_hi": float(np.quantile(draws, 0.95)), "p_delta_positive": float((draws > 0).mean()),
            "n_blocks": int(len(idx)), "n_boot": int(n_boot)}


def restrict_unit(u: Unit, mask_crop: np.ndarray, task_suffix: str) -> Unit:
    """Same crop, evaluation mask restricted (e.g. to DEM-covered pixels): a NEW protocol (different key)."""
    ev = u.ev & np.asarray(mask_crop, bool)
    return Unit(f"{u.task}{task_suffix}", u.zone, u.sl, ev, u.truth, u.exclude, u.supports_tp, u.known)


def distance_km(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return (ndimage.distance_transform_edt(~mask, sampling=(0.1, 0.1))).astype(np.float32)
