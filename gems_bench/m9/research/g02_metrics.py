"""G02 evaluation on top of the G01 protocol (``geology_metrics``: equal-area masks, distance-tolerant counts per
spatial block, paired block bootstrap). Added here:

nested virtual candidates (``nested_virtual``)
    B1, B2 and G02-F are CHOICES among variants. For held-out unit (task, zone) the variant with the best pooled
    metric on the task's OTHER zones -- excluding zones whose evaluation footprint overlaps this one
    (``unit_overlap``) -- is chosen, and that variant's counts in this zone are copied under the virtual name. No
    choice ever sees the zone it is scored on.
contrasts (``contrast_table``)
    paired block bootstrap of metric(candidate) - metric(reference) at the primary budget / distance + share of zones
    won, classified SUPPORTED / REFUTED / INCONCLUSIVE with the thresholds of ``decision`` (fixed before the run).
complementarity (``complement_counts``)
    truth pixels MISSED by a reference selection (farther than d) and recovered by a candidate selection, and the
    precision of the candidate pixels that lie away from the reference selection ("new" picks). The random map at the
    same budget gives the chance level.
strata (``truth_strata`` / ``strata_counts``)
    recall by truth-trace length class. Lengths come from connected components of the skeletonised LABEL raster (the
    competition labels carry no segment attributes): a trace crossing another is one component.
redundancy (``rank_correlations``)
    label-free Spearman correlation between maps on sampled domain pixels.
lineament geometry (``lineament_metrics``)
    evaluated separately from pixel ranking: segments ranked by score are taken until a length budget (km per
    100 km^2 of domain); length precision, truth recall and orientation agreement at each distance.
recommendations (``decide``)
    KEEP / INVESTIGATE / REJECT / UNVERIFIED from the incremental contrasts (never from map appearance).
"""
from __future__ import annotations

import fnmatch

import numpy as np
import pandas as pd
from scipy import ndimage

from . import geology_metrics as GM

VERDICTS = ("SUPPORTED", "REFUTED", "INCONCLUSIVE", "UNAVAILABLE")
RECOMMENDATIONS = ("KEEP", "INVESTIGATE", "REJECT", "UNVERIFIED")


# =============================================================================== selection helpers
def tie_noise(shape, seed: int, scale: float) -> np.ndarray:
    """Seeded uniform noise in [0, scale): added to EVERY map of a unit (same draw) so ties never select by index."""
    return (np.random.default_rng(int(seed)).random(shape, dtype=np.float32) * np.float32(scale)).astype(np.float32)


def match_family(names, patterns) -> list[str]:
    names = list(names)
    return sorted({n for n in names for p in patterns if fnmatch.fnmatchcase(n, p)})


# =============================================================================== overlap (leakage guard)
def footprint(u, stride: int) -> dict:
    """Downsampled evaluation footprint of a unit in FULL-grid coordinates."""
    r0, c0 = u.sl[0].start or 0, u.sl[1].start or 0
    ev = np.asarray(u.ev, bool)
    return {"r0": int(r0), "c0": int(c0), "stride": int(stride), "mask": ev[::stride, ::stride].copy()}


def _fp_set(fp: dict) -> set:
    s = fp["stride"]
    rr, cc = np.nonzero(fp["mask"])
    return set(zip(((rr * s + fp["r0"]) // s).tolist(), ((cc * s + fp["c0"]) // s).tolist()))


def unit_overlap(fps: dict) -> pd.DataFrame:
    """Pairs of units (any task) whose evaluation footprints share downsampled cells. ``fps``: {(task, zone): fp}."""
    keys = sorted(fps)
    sets = {k: _fp_set(fps[k]) for k in keys}
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            n = len(sets[a] & sets[b])
            if n:
                rows.append({"task_a": a[0], "zone_a": int(a[1]), "task_b": b[0], "zone_b": int(b[1]),
                             "shared_cells": n, "stride": fps[a]["stride"]})
    return pd.DataFrame(rows, columns=["task_a", "zone_a", "task_b", "zone_b", "shared_cells", "stride"])


def exclusions(overlap: pd.DataFrame) -> dict:
    """{(task, zone): {zones of the SAME task that may not choose for it}} (overlapping evaluation footprints)."""
    ex: dict = {}
    for r in overlap.itertuples():
        if r.task_a != r.task_b:
            continue
        ex.setdefault((r.task_a, int(r.zone_a)), set()).add(int(r.zone_b))
        ex.setdefault((r.task_b, int(r.zone_b)), set()).add(int(r.zone_a))
    return ex


# =============================================================================== nested virtual candidates
def nested_virtual(B: pd.DataFrame, variants: list[str], name: str, metric: str, f0: float, d0: float,
                   excl: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(block rows of the virtual candidate ``name``, choice table). See module docstring."""
    excl = excl or {}
    variants = [v for v in variants if v in set(B.candidate)]
    if not variants:
        return B.iloc[0:0].copy(), pd.DataFrame(columns=["virtual", "task", "zone", "chosen", "n_other_zones"])
    sel = B[(B.area_frac == float(f0)) & (B.tol_m == float(d0)) & B.candidate.isin(variants)]
    rows, choices = [], []
    for task, g in sel.groupby("task"):
        zones = sorted(int(z) for z in g.zone.unique())
        for z in zones:
            others = [o for o in zones if o != z and o not in excl.get((task, z), set())]
            if not others:
                choices.append({"virtual": name, "task": task, "zone": z, "chosen": None, "n_other_zones": 0})
                continue
            pooled = GM.summarize(g[g.zone.isin(others)], ["candidate"]).set_index("candidate")[metric].dropna()
            if pooled.empty:
                choices.append({"virtual": name, "task": task, "zone": z, "chosen": None,
                                "n_other_zones": len(others)})
                continue
            best = str(pooled.sort_index().idxmax())               # ties: alphabetical first (deterministic)
            choices.append({"virtual": name, "task": task, "zone": z, "chosen": best, "n_other_zones": len(others),
                            "chosen_value_other_zones": float(pooled[best])})
            rows.append(B[(B.task == task) & (B.zone == z) & (B.candidate == best)].assign(candidate=name,
                                                                                           chosen=best))
    out = pd.concat(rows, ignore_index=True) if rows else B.iloc[0:0].copy()
    return out, pd.DataFrame(choices)


# =============================================================================== contrasts
def contrast_table(B: pd.DataFrame, Z: pd.DataFrame, pairs: list[tuple], f0: float, d0: float, dc: dict,
                   n_boot: int, seed: int) -> pd.DataFrame:
    """``pairs``: [(family, kind, candidate, reference, question)]. One row per (pair, task)."""
    m = dc["metric"]
    have = set(B.candidate)
    Bs = B[(B.area_frac == float(f0)) & (B.tol_m == float(d0))]
    Zs = Z[(Z.area_frac == float(f0)) & (Z.tol_m == float(d0))]
    rows = []
    for fam, kind, c, r, q in pairs:
        for task, g in Bs.groupby("task"):
            base = {"family": fam, "kind": kind, "task": task, "candidate": c, "reference": r, "question": q,
                    "metric": m, "area_frac": float(f0), "tol_m": float(d0)}
            if c not in have or r not in have:
                rows.append({**base, "verdict": "UNAVAILABLE"})
                continue
            bt = GM.paired_block_bootstrap(g, c, r, m, int(n_boot), int(seed))
            zc = Zs[(Zs.task == task) & (Zs.candidate == c)].set_index("zone")[m]
            zr = Zs[(Zs.task == task) & (Zs.candidate == r)].set_index("zone")[m]
            j = pd.concat([zc, zr], axis=1, keys=["c", "r"]).dropna()
            win = float((j.c > j.r).mean()) if len(j) else np.nan
            p = bt["p_delta_positive"]
            if np.isfinite(p) and bt["delta"] > 0 and p >= float(dc["min_p_positive"]) and np.isfinite(win) and \
                    win >= float(dc["min_zone_win_share"]):
                v = "SUPPORTED"
            elif np.isfinite(p) and bt["delta"] < 0 and p <= 1 - float(dc["min_p_positive"]):
                v = "REFUTED"
            else:
                v = "INCONCLUSIVE"
            rows.append({**base, **{k: bt[k] for k in ("delta", "ci_lo", "ci_hi", "p_delta_positive", "n_blocks")},
                         "zone_win_share": win, "n_zones": int(len(j)), "verdict": v})
    return pd.DataFrame(rows)


def per_zone_deltas(Z: pd.DataFrame, pairs: list[tuple], f0: float, d0: float, metric: str) -> pd.DataFrame:
    """Per-zone metric difference candidate - reference (figure 14)."""
    Zs = Z[(Z.area_frac == float(f0)) & (Z.tol_m == float(d0))]
    rows = []
    for fam, kind, c, r, q in pairs:
        for (task, zone), g in Zs.groupby(["task", "zone"]):
            a = g[g.candidate == c][metric]
            b = g[g.candidate == r][metric]
            if len(a) and len(b):
                rows.append({"family": fam, "kind": kind, "candidate": c, "reference": r, "task": task,
                             "zone": int(zone), "metric": metric, "candidate_value": float(a.iloc[0]),
                             "reference_value": float(b.iloc[0]), "delta": float(a.iloc[0] - b.iloc[0])})
    return pd.DataFrame(rows)


# =============================================================================== complementarity
def complement_counts(scores: dict, refs: list[str], cands: list[str], domain: np.ndarray, truth: np.ndarray, pixel_m,
                      frac: float, tol: float, blk: np.ndarray, blk_ids: np.ndarray, tags: dict) -> pd.DataFrame:
    """Per (reference, candidate, block): truth missed by the reference and recovered by the candidate; candidate
    picks away from the reference selection and how many of them are near truth."""
    domain, truth = np.asarray(domain, bool), np.asarray(truth, bool)
    nb = len(blk_ids)
    d_truth = GM.distance_m(truth, pixel_m)
    sels = {n: GM.equal_area_mask(scores[n], domain, frac) for n in set(refs) | set(cands) if n in scores}
    dsel = {n: GM.distance_m(s, pixel_m) for n, s in sels.items()}
    rows = []
    for r in refs:
        if r not in sels:
            continue
        missed = truth & (dsel[r] > tol)
        away_r = dsel[r] > tol
        n_missed = np.bincount(blk[missed], minlength=nb)
        for c in cands:
            if c not in sels or c == r:
                continue
            rec = np.bincount(blk[missed & (dsel[c] <= tol)], minlength=nb)
            new = sels[c] & away_r
            n_new = np.bincount(blk[new], minlength=nb)
            n_new_hit = np.bincount(blk[new & (d_truth <= tol)], minlength=nb)
            keep = np.flatnonzero((n_missed > 0) | (n_new > 0))
            for b in keep:
                rows.append((r, c, int(blk_ids[b]), int(n_missed[b]), int(rec[b]), int(n_new[b]), int(n_new_hit[b])))
    df = pd.DataFrame(rows, columns=["reference", "candidate", "block", "n_missed_by_ref", "n_recovered",
                                     "n_new_picks", "n_new_hits"])
    df["area_frac"], df["tol_m"] = float(frac), float(tol)
    for k, v in tags.items():
        df[k] = v
    return df


def summarize_complement(C: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if not len(C):
        return pd.DataFrame()
    g = C.groupby(by)[["n_missed_by_ref", "n_recovered", "n_new_picks", "n_new_hits"]].sum().reset_index()
    g["recovered_share"] = g.n_recovered / g.n_missed_by_ref.where(g.n_missed_by_ref > 0)
    g["new_pick_precision"] = g.n_new_hits / g.n_new_picks.where(g.n_new_picks > 0)
    if "random" in set(g.candidate):
        keys = [k for k in by if k != "candidate"]
        rnd = g[g.candidate == "random"][keys + ["recovered_share", "new_pick_precision"]].rename(
            columns={"recovered_share": "random_recovered_share", "new_pick_precision": "random_new_pick_precision"})
        g = g.merge(rnd, on=keys, how="left")
        g["recovery_vs_random"] = g.recovered_share / g.random_recovered_share
    return g


# =============================================================================== truth-length strata
def truth_strata(truth: np.ndarray, pixel_m, edges_km) -> tuple[np.ndarray, list[str]]:
    """(stratum index per pixel, -1 off truth; labels) by the length of each truth trace component."""
    from skimage.morphology import skeletonize

    t = np.asarray(truth, bool)
    out = np.full(t.shape, -1, np.int16)
    edges = [float(e) for e in edges_km]
    labels = [f"{a:g}-{b:g}km" if b < 1e8 else f">={a:g}km" for a, b in zip(edges, edges[1:])]
    if not t.any():
        return out, labels
    lab, n = ndimage.label(t, structure=np.ones((3, 3), bool))
    sk = skeletonize(t)
    length_km = np.bincount(lab[sk], minlength=n + 1).astype(np.float64) * float(np.mean(pixel_m)) / 1000.0
    k = np.clip(np.searchsorted(edges, length_km, side="right") - 1, 0, len(labels) - 1)
    out[t] = k[lab[t]]
    return out, labels


def strata_counts(scores: dict, names: list[str], domain, truth, strata: np.ndarray, labels: list[str], pixel_m,
                  frac: float, tol: float, tags: dict) -> pd.DataFrame:
    rows = []
    for n in names:
        if n not in scores:
            continue
        d_sel = GM.distance_m(GM.equal_area_mask(scores[n], domain, frac), pixel_m)
        for k, lab in enumerate(labels):
            m = np.asarray(truth, bool) & (strata == k)
            rows.append({"candidate": n, "stratum": lab, "n_truth": int(m.sum()), "n_truth_hit": int((m & (d_sel <= tol)).sum()),
                         "area_frac": float(frac), "tol_m": float(tol), **tags})
    return pd.DataFrame(rows)


# =============================================================================== redundancy
def rank_correlations(scores: dict, names: list[str], domain, n: int, seed: int, tags: dict) -> pd.DataFrame:
    idx = np.flatnonzero(np.asarray(domain, bool).ravel())
    if idx.size > n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    names = [x for x in names if x in scores]
    R = {}
    for x in names:
        v = np.asarray(scores[x], np.float32).ravel()[idx]
        R[x] = pd.Series(v).rank().to_numpy()
    rows = []
    for i, a in enumerate(names):
        for b in names[i:]:
            ok = np.isfinite(R[a]) & np.isfinite(R[b])
            rho = float(np.corrcoef(R[a][ok], R[b][ok])[0, 1]) if ok.sum() > 10 else np.nan
            rows.append({"a": a, "b": b, "spearman": rho, "n": int(ok.sum()), **tags})
    return pd.DataFrame(rows)


# =============================================================================== lineament geometry
def lineament_metrics(seg_lab: np.ndarray, S: pd.DataFrame, truth, domain, pixel_m, budgets, tols, tags: dict) -> pd.DataFrame:
    from .geology_hypotheses import terrain_strike
    from .geology_wavelet import axial_agreement

    domain, truth = np.asarray(domain, bool), np.asarray(truth, bool)
    py, px = float(pixel_m[0]), float(pixel_m[1])
    area_km2 = domain.sum() * py * px / 1e6
    rows = []
    if not truth.any() or not domain.any():
        return pd.DataFrame(rows)
    d_truth, (ir, ic) = ndimage.distance_transform_edt(~truth, sampling=(py, px), return_indices=True)
    t_strike, _ = terrain_strike(truth.astype(np.float32), np.ones(truth.shape, bool), 150.0, pixel_m)
    near_strike = t_strike[ir, ic]
    order = S.sort_values("score", ascending=False) if len(S) else S
    az_img = np.full(seg_lab.shape, np.nan, np.float32)
    if len(S):
        lut = np.full(int(seg_lab.max()) + 1, np.nan, np.float32)
        lut[S.segment.to_numpy() + 1] = S.azimuth.to_numpy(np.float32)
        az_img = lut[seg_lab]
    for b in budgets:
        Lb = float(b) * area_km2 / 100.0 * 1000.0
        if len(order):
            cum = order.length_m.cumsum().to_numpy()
            take = order.segment.to_numpy()[(cum - order.length_m.to_numpy()) < Lb]
        else:
            take = np.array([], int)
        chosen = np.isin(seg_lab, take + 1) & domain
        d_sel = GM.distance_m(chosen, pixel_m)
        for t in tols:
            t = float(t)
            nsel = int(chosen.sum())
            hit = chosen & (d_truth <= t)
            agree = axial_agreement(az_img[hit], near_strike[hit]) if hit.any() else np.array([])
            rows.append({"budget_km_per_100km2": float(b), "tol_m": t, "n_segments": int(len(take)),
                         "chosen_length_km": float(order.length_m[order.segment.isin(take)].sum() / 1000.0)
                         if len(order) else 0.0,
                         "length_precision": hit.sum() / nsel if nsel else np.nan,
                         "truth_recall": float((truth & (d_sel <= t)).sum() / truth.sum()),
                         "random_precision": float((domain & (d_truth <= t)).sum() / domain.sum()),
                         "orientation_agreement": float(np.nanmean(agree)) if np.isfinite(agree).any() else np.nan,
                         **tags})
    return pd.DataFrame(rows)


# =============================================================================== recommendations
def decide(H: pd.DataFrame, mode: str, dc: dict, families: list[str]) -> pd.DataFrame:
    """KEEP / INVESTIGATE / REJECT / UNVERIFIED per G02 family from its INCREMENTAL contrasts (base + X vs base)."""
    rows = []
    pt = dc["primary_task"]
    for fam in families:
        h = H[(H.family == fam) & (H.kind == "incremental")] if len(H) else H
        avail = h[h.verdict != "UNAVAILABLE"] if len(h) else h
        notes = []
        if not len(avail):
            prov, notes = "UNVERIFIED", ["no incremental contrast computed"]
        else:
            nz = avail[avail.task == pt].n_zones.max() if (avail.task == pt).any() else 0
            if not nz or nz < int(dc["min_zones"]):
                prov = "UNVERIFIED"
                notes.append(f"{int(nz or 0)} held-out zones on {pt} (< {dc['min_zones']})")
            else:
                ok_bases = []
                for ref, g in avail.groupby("reference"):
                    sup = (g[g.task == pt].verdict == "SUPPORTED").any()
                    ref_other = (g[g.task != pt].verdict == "REFUTED").any()
                    if sup and not ref_other:
                        ok_bases.append(ref)
                any_sup = (avail.verdict == "SUPPORTED").any()
                any_ref = (avail.verdict == "REFUTED").any()
                low = (avail.p_delta_positive.fillna(0.5) < 0.5).all()
                if ok_bases:
                    prov = "KEEP"
                    notes.append(f"supported on {pt} over base(s) {ok_bases}, not refuted elsewhere")
                elif not any_sup and (any_ref or low):
                    prov = "REJECT"
                    notes.append("no supported increment; refuted or P(delta > 0) < 0.5 everywhere")
                else:
                    prov = "INVESTIGATE"
                    notes.append("mixed / inconclusive increments")
        rec = prov if mode == "FULL_EVALUATION" else "UNVERIFIED"
        if mode != "FULL_EVALUATION":
            notes.append(f"{mode}: exploratory subset, never a final verdict (provisional: {prov})")
        best = avail.sort_values("p_delta_positive", ascending=False).head(1) if len(avail) else avail
        rows.append({"family": fam, "recommendation": rec, "provisional": prov, "mode": mode,
                     "best_delta": float(best.delta.iloc[0]) if len(best) else np.nan,
                     "best_p_delta_positive": float(best.p_delta_positive.iloc[0]) if len(best) else np.nan,
                     "best_reference": best.reference.iloc[0] if len(best) else None,
                     "best_task": best.task.iloc[0] if len(best) else None, "notes": "; ".join(notes)})
    return pd.DataFrame(rows)
