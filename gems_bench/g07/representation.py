"""Versioned prediction selection (representation policy ``g07.rep.1``).

EXACT-K. ``gems3.representation.top_mask`` keeps every pixel with ``score >= threshold``: with tied scores (saturated
probabilities, constant maps - the all-positive XGBoost case) it selects far more than the budget. ``exact_k_mask``
selects EXACTLY k = ceil(frac x |eligible|) eligible pixels (or all finite ones when fewer exist, reported as a
shortfall), ordered by score, then an optional label-free secondary score, then a deterministic hash of the FULL-GRID
pixel index (seeded splitmix64: identical for a pixel whatever the crop). No label is read. Every selection returns a
report: requested / actual fraction, cutoff score, pixels tied at the cutoff, how many the legacy rule would have
selected, selected score mass. The legacy selector stays available (``tie_break='legacy'``) for historical
reproduction only.

Other policies: post operations of ``gems3.representation`` applied after exact-k; ``soft_corridor`` (continuous
confidence with a decaying halo); ``variable_width_representation`` (width from localisation uncertainty);
``candidate_portfolio_selection`` (W2-preserving: an exact-k W2 base, whole candidates added by label-free utility,
the remainder filled in W2 order - the total is exactly k).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

POLICY_VERSION = "g07.rep.1"
_M1, _M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def pixel_hash(rows: np.ndarray, cols: np.ndarray, width_full: int, seed: int = 2026) -> np.ndarray:
    """splitmix64 of the full-grid pixel index (uint64; label-free, crop-invariant)."""
    with np.errstate(over="ignore"):
        x = (np.asarray(rows, np.uint64) * np.uint64(width_full) + np.asarray(cols, np.uint64)
             + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15))
        x = (x ^ (x >> np.uint64(30))) * _M1
        x = (x ^ (x >> np.uint64(27))) * _M2
        return x ^ (x >> np.uint64(31))


def budget_k(frac: float, n_eligible: int) -> int:
    return max(int(np.ceil(float(frac) * int(n_eligible))), 1) if n_eligible else 0


def exact_k_mask(score: np.ndarray, eligible: np.ndarray, frac: float | None = None, k: int | None = None, *,
                 secondary: np.ndarray | None = None, offset=(0, 0), width_full: int | None = None,
                 tie_break: str = "secondary_then_hash", seed: int = 2026) -> tuple[np.ndarray, dict]:
    """(bool mask with exactly k eligible pixels, report). ``offset`` = full-grid (row, col) of the crop origin."""
    s = np.asarray(score, np.float64)
    el = np.asarray(eligible, bool)
    n_el = int(el.sum())
    k = budget_k(frac, n_el) if k is None else int(k)
    cand = el & np.isfinite(s)
    idx = np.flatnonzero(cand.ravel())
    out = np.zeros(s.shape, bool)
    rep = {"policy": POLICY_VERSION, "tie_break": tie_break, "requested_frac": float(frac) if frac is not None else
           (k / n_el if n_el else np.nan), "k": int(k), "n_eligible": n_el, "n_finite_eligible": int(idx.size)}
    if not idx.size or k <= 0:
        return out, {**rep, "actual_px": 0, "actual_frac": 0.0, "cutoff": np.nan, "n_tied_at_cutoff": 0,
                     "legacy_px": 0, "selected_mass": 0.0, "shortfall": int(k)}
    v = s.ravel()[idx]
    if tie_break == "legacy":
        kk = min(k, v.size)
        thr = np.partition(v, v.size - kk)[v.size - kk]
        sel = idx[v >= thr]
    else:
        keys = []
        r, c = np.unravel_index(idx, s.shape)
        if tie_break in ("secondary_then_hash", "hash"):
            keys.append(pixel_hash(r + int(offset[0]), c + int(offset[1]), int(width_full or s.shape[1]), seed))
        if tie_break == "secondary_then_hash" and secondary is not None:
            sv = np.asarray(secondary, np.float64).ravel()[idx]
            keys.append(-np.nan_to_num(sv, nan=-np.inf))
        keys.append(-v)                                         # primary key last (np.lexsort convention)
        order = np.lexsort(tuple(keys))
        sel = idx[order[:min(k, idx.size)]]
    out.ravel()[sel] = True
    sv = s.ravel()[sel]
    cutoff = float(sv.min()) if sv.size else np.nan
    rep.update(actual_px=int(sel.size), actual_frac=float(sel.size / n_el) if n_el else 0.0, cutoff=cutoff,
               n_tied_at_cutoff=int((v == cutoff).sum()), n_tied_selected=int((sv == cutoff).sum()),
               legacy_px=int((v >= cutoff).sum()), selected_mass=float(np.clip(sv, 0, None).sum()),
               shortfall=int(max(k - sel.size, 0)))
    return out, rep


def apply_policy(rep: dict, score: np.ndarray, ev: np.ndarray, frac: float, *, eligible: np.ndarray,
                 known: np.ndarray | None = None, secondary=None, offset=(0, 0), width_full=None,
                 tie_break: str = "secondary_then_hash", seed: int = 2026) -> tuple[np.ndarray, dict]:
    """``{select, post}`` representation with exact-k for ``select='top'`` (crest / raw delegate to gems3)."""
    from ..gems3 import representation as R

    sel = rep.get("select", "top")
    dom = np.asarray(eligible, bool) & np.asarray(ev, bool)
    if sel == "top":
        m, report = exact_k_mask(score, dom, frac, secondary=secondary, offset=offset, width_full=width_full,
                                 tie_break=tie_break, seed=seed)
        q = m.astype(np.float32)
        aux = {"score": score, "ev": ev, "known": known}
        for name, prm in rep.get("post") or []:
            q = R.POST_OPS[name](q, prm or {}, aux)
        q = np.where(ev, q, 0.0).astype(np.float32)
    else:
        q = R.apply(rep, score, ev, frac, known=known, eligible=dom)
        report = {"policy": "gems3." + sel, "tie_break": "n/a", "requested_frac": frac}
        if sel == "raw":                     # a ready-made selection (portfolio): audit its area like any exact-k row
            n, k = int(((q > 0) & dom).sum()), budget_k(frac, int(dom.sum()))
            report.update(policy=POLICY_VERSION + ":as_selected", k=k, actual_px=n, shortfall=max(k - n, 0),
                          n_tied_at_cutoff=0)
    report["final_px"] = int((q > 0).sum())
    report["final_frac_eligible"] = float(((q > 0) & dom).sum() / max(int(dom.sum()), 1))
    report["final_mass"] = float(q.sum())
    return q, report


# =============================================================================== continuous representations
def soft_corridor(core: np.ndarray, confidence: np.ndarray | None, halo_px: float, halo_value: float = 0.5,
                  v_min: float = 0.5, ev: np.ndarray | None = None) -> np.ndarray:
    """Core pixels valued by confidence in [v_min, 1]; a Gaussian-decaying halo (scale ``halo_px``) at
    ``halo_value`` x the nearest core value; 0 elsewhere (and outside ``ev``)."""
    core = np.asarray(core, bool)
    out = np.zeros(core.shape, np.float32)
    if not core.any():
        return out
    c = np.ones(core.shape, np.float32) if confidence is None else np.clip(np.nan_to_num(confidence), 0, 1)
    val = v_min + (1 - v_min) * c
    out[core] = val[core]
    if halo_px > 0:
        d, (ir, ic) = ndimage.distance_transform_edt(~core, return_indices=True)
        halo = (~core) & (d <= 3 * halo_px)
        out[halo] = (halo_value * val[ir, ic] * np.exp(-0.5 * (d / halo_px) ** 2))[halo]
    if ev is not None:
        out = np.where(ev, out, 0.0)
    return out.astype(np.float32)


def variable_width_representation(core: np.ndarray, width_px: np.ndarray, max_px: int = 3) -> np.ndarray:
    """Binary map: each core pixel dilated by its own (rounded, capped) localisation width."""
    from ..postprocess.topology import _disk

    core = np.asarray(core, bool)
    w = np.clip(np.round(np.nan_to_num(width_px)), 0, max_px).astype(int)
    out = core.copy()
    for r in range(1, max_px + 1):
        m = core & (w >= r)
        if m.any():
            out |= ndimage.binary_dilation(m, structure=_disk(r))
    return out


# =============================================================================== W2-preserving portfolio
def candidate_portfolio_selection(w2: np.ndarray, eligible: np.ndarray, frac: float, cand_lab: np.ndarray,
                                  utility: pd.DataFrame, base_share: float = 0.85, min_utility: float = 0.0,
                                  offset=(0, 0), width_full=None, seed: int = 2026) -> tuple[np.ndarray, dict]:
    """Exactly k pixels: the top ``base_share`` k of W2, then whole candidates (label-free utility order, positive
    utility only, eligible pixels not already selected), then W2 order for the remainder."""
    el = np.asarray(eligible, bool)
    k = budget_k(frac, int(el.sum()))
    kb = int(np.floor(base_share * k))
    base, rb = exact_k_mask(w2, el, k=kb, offset=offset, width_full=width_full, seed=seed)
    sel = base.copy()
    lab = np.asarray(cand_lab, np.int32)
    added, added_px = [], 0
    if len(utility) and lab.max() > 0:
        U = utility[utility.utility > min_utility].sort_values(["utility", "label"], ascending=[False, True])
        sizes = np.bincount(lab[el & ~sel].ravel(), minlength=int(lab.max()) + 1)
        for r in U.itertuples():
            lb = int(r.label)
            if lb <= 0 or lb >= sizes.size:
                continue
            m = (lab == lb) & el & ~sel
            n = int(m.sum())
            if not n or int(sel.sum()) + n > k:
                continue
            sel |= m
            added.append(lb)
            added_px += n
    rest = k - int(sel.sum())
    if rest > 0:
        fill, _ = exact_k_mask(np.where(sel, -np.inf, w2), el & ~sel, k=rest, offset=offset, width_full=width_full,
                               seed=seed)
        sel |= fill
    return sel, {"policy": POLICY_VERSION + ":portfolio", "k": k, "base_px": int(base.sum()), "added_candidates":
                 len(added), "candidate_px": added_px, "w2_fill_px": int(max(rest, 0)), "actual_px": int(sel.sum()),
                 "base_report": rb}


def compare_representation_policies(vu, maps: dict, policies: dict, frac: float, proto: dict, tie_break: str,
                                    seed: int = 2026, width_full: int | None = None) -> pd.DataFrame:
    """Official components of every (map, policy) on one unit at a fixed requested fraction (matched budget)."""
    from ..m7.protocol import score_row
    from .evaluation import dti

    u = vu.unit
    rows = []
    for mname, s in maps.items():
        for pname, rep in policies.items():
            q, rp = apply_policy(rep, s, u.ev, frac, eligible=proto["elig"], known=vu.known,
                                 offset=(u.sl[0].start, u.sl[1].start), width_full=width_full, tie_break=tie_break,
                                 seed=seed)
            r = score_row(u, q, proto["elig"], proto["pkey"], model=mname, policy=pname, frac=frac, unit=vu.key,
                          scenario=vu.scenario)
            r.update({f"sel_{k}": v for k, v in rp.items() if not isinstance(v, dict)})
            r["dti"] = dti(r["tp_w"], r["fp_w"], r["fn_w"])
            rows.append(r)
    return pd.DataFrame(rows)
