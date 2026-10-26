"""Milestone D (first step): synthetic incomplete maps + a deterministic endpoint-extension baseline.

The learning target changes from P(mapped fault | data) to P(missing fault | data, damaged map). This module only
builds the corruption protocol and scores the simplest non-neural completer; a learned completer (gradient-boosted
candidate ranker on damaged maps, then a small CNN conditioned on the damaged map) comes only if this shows headroom.

Corruptions (seeded, one class per removed pixel; applied to the mapped-fault raster)::

    interior_gap    remove a disk of diameter L (``delete_segment_km``) centred on a random interior skeleton pixel
    endpoint        remove label pixels within L (``trim_endpoint_km``) of a random skeleton endpoint
    short_fault     remove whole 8-connected faults shorter than ``short_fault_km`` (a sampled share of them)
    shifted         move a sampled fault component by ``shift_px`` (target = the original trace)

Evaluation: truth = removed pixels of one class; the damaged map is the known mask (pixel-exact, its predictions
ignored: the official convention); evaluation zone = valid pixels within 5 km of the removed pixels; zones =
geographic folds, so the extension length is chosen on the other folds (``nested_select``).

Leakage note: the deterministic baseline has no trainable parameter except L, selected across folds. A learned
completer must split fault *systems* geographically BEFORE corruption (``systems_by_fold``) and never sees an
intact held-out system as input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..postprocess.topology import _disk
from .candidates import tip_paths
from .metric_opt import nested_select

CLASSES = ("interior_gap", "endpoint", "short_fault", "shifted")


def systems_by_fold(system_map: np.ndarray, fold_map: np.ndarray) -> pd.DataFrame:
    """Majority geographic fold of every fault system (split systems before any corruption)."""
    m = (system_map >= 0) & (fold_map >= 0)
    df = pd.DataFrame({"system": system_map[m], "fold": fold_map[m]})
    return df.groupby("system").fold.agg(lambda s: int(s.mode().iloc[0])).rename("fold").reset_index()


def _remove_disk(dmg: np.ndarray, removed: np.ndarray, r: int, c: int, rad: int) -> None:
    """In place: label pixels of ``dmg`` within ``rad`` px of (r, c) move to ``removed`` (clipped at the edges)."""
    h, w = dmg.shape
    r0, r1, c0, c1 = max(r - rad, 0), min(r + rad + 1, h), max(c - rad, 0), min(c + rad + 1, w)
    d = _disk(rad)[rad - (r - r0):rad + (r1 - r), rad - (c - c0):rad + (c1 - c)]
    win = dmg[r0:r1, c0:c1] & d
    removed[r0:r1, c0:c1] |= win
    dmg[r0:r1, c0:c1] &= ~win


def corrupt(labels: np.ndarray, rng: np.random.Generator, mc: dict) -> tuple[np.ndarray, dict, pd.DataFrame]:
    """(damaged map, {class: removed-pixel mask}, protocol rows)."""
    from ..geology.skeleton import component_length_km, endpoints, skeletonize

    frac = float(mc.get("sample_frac", 0.15))
    dmg = labels.copy()
    removed = {c: np.zeros(labels.shape, bool) for c in CLASSES}
    rows = []
    sk = skeletonize(labels)
    ep = endpoints(sk)
    lab, n = ndimage.label(labels, structure=np.ones((3, 3), bool))
    comp_km = component_length_km(skeletonize(labels), 0.1)
    # short faults: whole components below the length threshold
    short_ids = np.unique(lab[(comp_km > 0) & (comp_km < mc["short_fault_km"])])
    short_ids = short_ids[short_ids > 0]
    pick = rng.choice(short_ids, int(round(frac * len(short_ids))), replace=False) if len(short_ids) else []
    m = np.isin(lab, pick) & dmg
    removed["short_fault"] |= m
    dmg &= ~m
    rows.append({"class": "short_fault", "n_events": len(pick), "removed_px": int(m.sum())})
    # endpoints
    eps = np.argwhere(ep & dmg)
    sel = eps[rng.choice(len(eps), int(round(frac * len(eps))), replace=False)] if len(eps) else []
    for r, c in sel:
        _remove_disk(dmg, removed["endpoint"], r, c, int(round(float(rng.choice(mc["trim_endpoint_km"])) / 0.1)))
    rows.append({"class": "endpoint", "n_events": len(sel), "removed_px": int(removed["endpoint"].sum())})
    # interior gaps on long traces
    interior = np.argwhere(sk & dmg & ~ndimage.binary_dilation(ep, iterations=5) & (comp_km >= 3.0))
    k = int(round(frac * len(interior) / 50.0))                   # ~ one gap per 5 km of eligible trace
    sel = interior[rng.choice(len(interior), min(k, len(interior)), replace=False)] if len(interior) else []
    for r, c in sel:                                               # disk of diameter L (radius L / 2)
        _remove_disk(dmg, removed["interior_gap"], r, c, int(round(float(rng.choice(mc["delete_segment_km"])) / 0.2)))
    rows.append({"class": "interior_gap", "n_events": len(sel), "removed_px": int(removed["interior_gap"].sum())})
    # shifts
    lab2, n2 = ndimage.label(dmg, structure=np.ones((3, 3), bool))
    ids = rng.choice(np.arange(1, n2 + 1), int(round(frac * n2 / 3)), replace=False) if n2 else []
    s = int(mc["shift_px"])
    for i in ids:
        m = lab2 == i
        moved = np.roll(m, (s, s), axis=(0, 1))
        removed["shifted"] |= m & ~moved
        dmg = (dmg & ~m) | moved
    rows.append({"class": "shifted", "n_events": len(ids), "removed_px": int(removed["shifted"].sum())})
    for c in CLASSES:
        removed[c] &= ~dmg
    return dmg, removed, pd.DataFrame(rows)


def endpoint_extension(damaged: np.ndarray, valid: np.ndarray, L_km: float, tail_px: int = 5) -> np.ndarray:
    out = np.zeros(damaged.shape, np.float32)
    for p, _ in tip_paths(damaged, valid, [L_km], tail_px):
        out[p[:, 0], p[:, 1]] = 1.0
    return out


def evaluate(ctx, cfg: dict) -> dict[str, pd.DataFrame]:
    from .official import _unit

    mc = cfg["map_completion"]
    rng = np.random.default_rng(mc["seed"])
    lab, valid, fm = ctx.data.labels, ctx.data.valid, ctx.folds.fold_map
    proto, rows = [], []
    for real in range(int(mc["n_realisations"])):
        dmg, removed, pr = corrupt(lab, rng, mc)
        proto.append(pr.assign(realisation=real))
        preds = {L: endpoint_extension(dmg, valid, L) for L in mc["extension_km"]}
        for cls in CLASSES:
            G = removed[cls]
            if not G.any():
                continue
            near = ndimage.distance_transform_edt(~G, sampling=(0.1, 0.1)) <= 5.0
            for k in ctx.fold_ids:
                ev = near & valid & (fm == k)
                if not (G & ev).any():
                    continue
                u = _unit("map_completion", k, ev, G, dmg, False, dmg)
                for L, q in preds.items():
                    c = u.components(u.crop(q))
                    rows.append({"realisation": real, "class": cls, "zone": k, "L_km": L, **c.as_dict()})
        print(f"[gems3] map completion realisation {real}: done")
    res = pd.DataFrame(rows)
    sel = []
    for cls, g in res.groupby("class") if len(res) else []:
        g = g.groupby(["zone", "L_km"])[["tp_w", "fp_w", "fn_w", "n_truth"]].sum().reset_index()
        per, s = nested_select(g, ["L_km"])
        sel.append({"class": cls, "method": "endpoint_extension", "dti_crossfit": s["dti_crossfit"],
                    "recall_crossfit": float(per.tp_w.sum() / max(per.n_truth.sum(), 1)) if len(per) else np.nan,
                    "L_all_zones": s["params_all"]["L_km"], "fold_min": s["fold_min"]})
    return {"protocol": pd.concat(proto, ignore_index=True), "results": res, "selection": pd.DataFrame(sel)}
