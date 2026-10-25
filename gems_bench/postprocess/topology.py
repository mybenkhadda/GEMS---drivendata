"""M6.6: topology-aware postprocessing of binary fault masks (research hypotheses, not assumed improvements).

Operations act on one boolean mask (the per-fold top-a mask of a model) and never read labels:

``remove_small``          drop 8-connected components below ``min_px``
``close_gaps``            isotropic morphological closing (disk radius ``radius_px``)
``directional_close``     closing with line elements of ``length_px`` at ``n_dirs`` orientations; a closed pixel is
                          kept only where the local mask orientation agrees with the element (+- ``tol_deg``)
``link_endpoints``        curvature-constrained linking: skeleton endpoints within ``max_gap_px`` are joined by a
                          straight segment only if both endpoint tangents and the joining direction agree within
                          ``max_angle_deg``
``skeleton_width``        skeletonize, then re-dilate to a uniform width (``width_px``)
``min_length``            drop components whose skeleton is shorter than ``min_km``
``orientation_filter``    drop components whose internal orientation coherence is below ``min_coherence`` (blobs)

Every operation is deterministic (no randomness, ties broken by raster order) and is applied inside each fold
zone separately (``apply_per_zone``) so nothing propagates across held-out boundaries.

``ablation`` scores each operation and parameter on every fold with the shared ``FoldScorers`` and selects
(area fraction, parameter) for held-out fold k on the other folds only (``crossfit_select``). The identity is
always a candidate, so selection can decline to postprocess. Short faults are fragile: every row reports the
< 2 km recall change next to the DTI change.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import ndimage

_S8 = np.ones((3, 3), bool)


# =============================================================================== primitive operations
def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _line(length: int, angle_deg: float) -> np.ndarray:
    """Line structuring element; angle in degrees counter-clockwise from the +column axis."""
    h = max(int(length) // 2, 1)
    se = np.zeros((2 * h + 1, 2 * h + 1), bool)
    t = np.radians(angle_deg)
    for s in np.linspace(-h, h, 4 * h + 1):
        se[int(round(h - s * np.sin(t))), int(round(h + s * np.cos(t)))] = True
    return se


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=_S8)
    if n == 0:
        return mask.copy()
    cnt = np.bincount(lab.ravel())
    keep = cnt >= int(min_px)
    keep[0] = False
    return keep[lab]


def close_gaps(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.copy()
    return ndimage.binary_closing(mask, structure=_disk(int(radius_px)), border_value=0) | mask


def orientation_field(mask: np.ndarray, sigma_px: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """(angle deg CCW from +column in [0, 180), coherence in [0, 1]) of a binary mask's structure tensor.

    The angle is the dominant *along-structure* direction (perpendicular to the mean gradient)."""
    f = ndimage.gaussian_filter(mask.astype(np.float32), 1.0)
    gy, gx = np.gradient(f)
    jxx = ndimage.gaussian_filter(gx * gx, sigma_px)
    jyy = ndimage.gaussian_filter(gy * gy, sigma_px)
    jxy = ndimage.gaussian_filter(gx * gy, sigma_px)
    grad_dir = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))           # gradient direction (row down)
    along = (-(grad_dir + 90.0)) % 180.0                                  # flip row axis -> CCW, rotate to strike
    lam = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = np.where(jxx + jyy > 1e-12, lam / (jxx + jyy + 1e-12), 0.0)
    return along.astype(np.float32), np.clip(coh, 0, 1).astype(np.float32)


def _axial_diff(a, b):
    d = np.abs((np.asarray(a) - np.asarray(b)) % 180.0)
    return np.minimum(d, 180.0 - d)


def directional_close(mask: np.ndarray, length_px: int, n_dirs: int = 8, tol_deg: float = 22.5,
                      sigma_px: float = 2.0) -> np.ndarray:
    if length_px <= 1 or not mask.any():
        return mask.copy()
    ang, coh = orientation_field(mask, sigma_px)
    out = mask.copy()
    for i in range(n_dirs):
        a = 180.0 * i / n_dirs
        closed = ndimage.binary_closing(mask, structure=_line(length_px, a), border_value=0)
        ok = (_axial_diff(ang, a) <= tol_deg) & (coh > 0.2)
        out |= closed & ~mask & ok
    return out


def skeleton(mask: np.ndarray) -> np.ndarray:
    from ..geology.skeleton import skeletonize

    return skeletonize(mask)


def skeleton_width(mask: np.ndarray, width_px: int) -> np.ndarray:
    sk = skeleton(mask)
    return ndimage.binary_dilation(sk, structure=_disk(max(int(width_px), 0))) if width_px > 0 else sk


def min_length(mask: np.ndarray, min_km: float, pixel_km: float) -> np.ndarray:
    from ..geology.skeleton import component_length_km

    lab, n = ndimage.label(mask, structure=_S8)
    if n == 0:
        return mask.copy()
    sk = skeleton(mask)
    L = component_length_km(sk, pixel_km)
    comp_len = ndimage.maximum(L, lab, index=np.arange(1, n + 1))
    keep = np.r_[False, np.asarray(comp_len) >= float(min_km)]
    return keep[lab]


def component_coherence(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(label raster, coherence per component) from the pixel-coordinate covariance (1 = a line, 0 = a disc)."""
    lab, n = ndimage.label(mask, structure=_S8)
    if n == 0:
        return lab, np.zeros(0)
    rr, cc = np.nonzero(lab)
    L = lab[rr, cc]
    cnt = np.bincount(L, minlength=n + 1)[1:].astype(float)
    mr, mc = np.bincount(L, rr, n + 1)[1:] / cnt, np.bincount(L, cc, n + 1)[1:] / cnt
    vrr = np.bincount(L, rr * rr, n + 1)[1:] / cnt - mr ** 2
    vcc = np.bincount(L, cc * cc, n + 1)[1:] / cnt - mc ** 2
    vrc = np.bincount(L, rr * cc, n + 1)[1:] / cnt - mr * mc
    lam = np.sqrt((vrr - vcc) ** 2 + 4 * vrc ** 2)
    coh = np.where(vrr + vcc > 1e-9, lam / (vrr + vcc + 1e-9), 1.0)       # single pixels count as coherent
    return lab, coh


def orientation_filter(mask: np.ndarray, min_coherence: float, min_px: int = 4) -> np.ndarray:
    """Drop components (>= ``min_px``) whose shape is isotropic; tiny components are left to ``remove_small``."""
    lab, coh = component_coherence(mask)
    if not len(coh):
        return mask.copy()
    cnt = np.bincount(lab.ravel())[1:]
    keep = np.r_[False, (coh >= float(min_coherence)) | (cnt < int(min_px))]
    return keep[lab]


def _endpoint_tangents(sk: np.ndarray, tail_px: int) -> list[tuple[int, int, float, int]]:
    """(row, col, outward direction deg CCW from +col, component) for every skeleton endpoint."""
    from ..geology.skeleton import endpoints

    lab, _ = ndimage.label(sk, structure=_S8)
    eps = np.argwhere(endpoints(sk))
    out = []
    for r, c in eps:                                       # argwhere order = raster order (deterministic)
        r0, r1, c0, c1 = max(r - tail_px, 0), r + tail_px + 1, max(c - tail_px, 0), c + tail_px + 1
        win = (lab[r0:r1, c0:c1] == lab[r, c])
        yy, xx = np.nonzero(win)
        if len(yy) < 2:
            continue
        dy, dx = (r - (yy.mean() + r0)), (c - (xx.mean() + c0))           # from the tail centroid to the tip
        out.append((int(r), int(c), float(np.degrees(np.arctan2(-dy, dx)) % 360.0), int(lab[r, c])))
    return out


def link_endpoints(mask: np.ndarray, max_gap_px: int, max_angle_deg: float = 30.0, tail_px: int = 4) -> np.ndarray:
    """Join endpoint pairs of different components when the gap is short and the path keeps its direction."""
    from scipy.spatial import cKDTree
    from skimage.draw import line as draw_line

    if max_gap_px <= 0 or not mask.any():
        return mask.copy()
    sk = skeleton(mask)
    eps = _endpoint_tangents(sk, tail_px)
    out = mask.copy()
    if len(eps) < 2:
        return out
    used, cand = set(), []
    tree = cKDTree(np.array([(e[0], e[1]) for e in eps], float))
    for i, j in sorted(map(tuple, tree.query_pairs(r=float(max_gap_px), output_type="ndarray"))):
        r1, c1, t1, l1 = eps[i]
        r2, c2, t2, l2 = eps[j]
        gap = float(np.hypot(r2 - r1, c2 - c1))
        if l1 == l2 or gap == 0:
            continue
        dir12 = float(np.degrees(np.arctan2(-(r2 - r1), c2 - c1)) % 360.0)
        d1 = abs((dir12 - t1 + 180) % 360 - 180)                           # tip 1 must point towards tip 2
        d2 = abs(((dir12 + 180) % 360 - t2 + 180) % 360 - 180)             # tip 2 must point towards tip 1
        if d1 <= max_angle_deg and d2 <= max_angle_deg:
            cand.append((gap, i, j))
    for gap, i, j in sorted(cand):                         # shortest gaps first; each endpoint linked once
        if i in used or j in used:
            continue
        rr, cc = draw_line(eps[i][0], eps[i][1], eps[j][0], eps[j][1])
        out[rr, cc] = True
        used.update((i, j))
    return out


OPERATIONS = {
    "remove_small": lambda m, p, px_km: remove_small(m, p["min_px"]),
    "close_gaps": lambda m, p, px_km: close_gaps(m, p["radius_px"]),
    "directional_close": lambda m, p, px_km: directional_close(m, p["length_px"], p.get("n_dirs", 8),
                                                               p.get("tol_deg", 22.5)),
    "link_endpoints": lambda m, p, px_km: link_endpoints(m, p["max_gap_px"], p.get("max_angle_deg", 30.0)),
    "skeleton_width": lambda m, p, px_km: skeleton_width(m, p["width_px"]),
    "min_length": lambda m, p, px_km: min_length(m, p["min_km"], px_km),
    "orientation_filter": lambda m, p, px_km: orientation_filter(m, p["min_coherence"], p.get("min_px", 4)),
}

DEFAULT_GRID = {
    "remove_small": [{"min_px": 3}, {"min_px": 6}, {"min_px": 12}],
    "close_gaps": [{"radius_px": 1}, {"radius_px": 2}],
    "directional_close": [{"length_px": 5}, {"length_px": 9}],
    "link_endpoints": [{"max_gap_px": 4}, {"max_gap_px": 8}],
    "skeleton_width": [{"width_px": 1}, {"width_px": 2}],
    "min_length": [{"min_km": 0.5}, {"min_km": 1.0}],
    "orientation_filter": [{"min_coherence": 0.3}, {"min_coherence": 0.5}],
}


def apply_ops(mask: np.ndarray, ops: list[tuple[str, dict]], pixel_km: float) -> np.ndarray:
    out = np.asarray(mask, bool)
    for name, prm in ops:
        out = OPERATIONS[name](out, prm, pixel_km)
    return out


def apply_per_zone(mask: np.ndarray, zones: np.ndarray, valid: np.ndarray, ops_by_zone: dict, pixel_km: float) -> np.ndarray:
    """Apply ``ops_by_zone[z]`` inside each zone separately (cropped to the zone's bounding box)."""
    out = np.zeros(mask.shape, bool)
    for z, ops in ops_by_zone.items():
        zm = valid & (zones == z)
        if not zm.any():
            continue
        rr, cc = np.nonzero(zm)
        sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
        res = apply_ops(mask[sl] & zm[sl], ops, pixel_km) & zm[sl]
        out[sl] |= res
    return out


# =============================================================================== topology metrics
def topology_metrics(mask: np.ndarray, pixel_km: float) -> dict:
    from ..geology.skeleton import branch_points, endpoints

    lab, n = ndimage.label(mask, structure=_S8)
    sk = skeleton(mask) if mask.any() else mask
    sk_km = float(sk.sum() * pixel_km * 1.12)
    _, coh = component_coherence(mask)
    cnt = np.bincount(lab.ravel())[1:] if n else np.zeros(0)
    return {"area_px": int(mask.sum()), "n_components": int(n), "skeleton_km": sk_km,
            "fragmentation_per_100km": float(n / max(sk_km, 1e-9) * 100.0) if n else 0.0,
            "n_endpoints": int(endpoints(sk).sum()) if n else 0, "n_junctions": int(branch_points(sk).sum()) if n else 0,
            "orientation_coherence_area_w": float(np.average(coh, weights=cnt)) if n else np.nan}


# =============================================================================== nested ablation
def _identity_key() -> str:
    return json.dumps({"op": "identity"})


def ablation(ctx, model: str, p: np.ndarray, *, fracs=(0.02, 0.05, 0.10), grid: dict | None = None,
             scorers=None) -> dict:
    """Per-fold components of every (operation, parameter, area fraction) + nested selection per operation.

    Returns {"table": long components table, "selection": per (operation, held-out fold) choice,
    "summary": per operation cross-fitted DTI and the identity baseline, "choices": {op: {zone: ops list}}}.
    """
    from ..p3.budget import binary_mask
    from ..p3.curves import crossfit_select
    from ..p3.transforms import FoldScorers

    grid = grid or DEFAULT_GRID
    v, fm, lab = ctx.data.valid, ctx.folds.fold_map, ctx.data.labels
    pkm = ctx.data.grid.pixel_km
    scorers = scorers or FoldScorers(lab, fm, v)
    rows = []
    base = {f: binary_mask(p, v, fm, f) for f in fracs}
    for f, m in base.items():
        rows += scorers.table(m.astype(np.float32), op="identity", param=_identity_key(), target_frac=f)
        for op, plist in grid.items():
            for prm in plist:
                q = apply_per_zone(m, fm, v, {z: [(op, prm)] for z in ctx.fold_ids}, pkm)
                rows += scorers.table(q.astype(np.float32), op=op, param=json.dumps({"op": op, **prm}), target_frac=f)
    table = pd.DataFrame(rows).assign(model=model)
    sel_rows, summ, choices = [], [], {}
    ident = table[table.op == "identity"]
    per_i, s_i = crossfit_select(ident, ["target_frac", "param"])
    for op in ["identity"] + list(grid):
        t = ident if op == "identity" else pd.concat([ident, table[table.op == op]], ignore_index=True)
        per, s = crossfit_select(t, ["target_frac", "param"])
        zones = sorted(t.zone.unique())
        choices[op] = {}
        for r in per.itertuples():
            prm = json.loads(r.param)
            name = prm.pop("op")
            choices[op][int(r.zone)] = {"frac": float(r.target_frac), "ops": [] if name == "identity" else [(name, prm)]}
            sel_rows.append({"model": model, "operation": op, "heldout_zone": int(r.zone),
                             "selection_zones": ",".join(str(z) for z in zones if z != r.zone),
                             "chosen_frac": float(r.target_frac), "chosen_param": r.param,
                             "dti_heldout": r.dti_heldout, "uses_heldout_labels": False})
        summ.append({"model": model, "operation": op, "dti_crossfit": s["dti_crossfit"], "fold_min": s["fold_min"],
                     "delta_vs_identity": s["dti_crossfit"] - s_i["dti_crossfit"],
                     "n_folds_applied": int(sum(1 for c in choices[op].values() if c["ops"]))})
    return {"table": table, "selection": pd.DataFrame(sel_rows), "summary": pd.DataFrame(summ), "choices": choices}


def stitched_mask(ctx, p: np.ndarray, choice: dict) -> np.ndarray:
    """Per-fold mask at the fold's chosen fraction with the fold's chosen operations (all chosen on other folds)."""
    from ..p3.budget import binary_mask

    v, fm = ctx.data.valid, ctx.folds.fold_map
    out = np.zeros(ctx.data.shape, bool)
    for z, c in choice.items():
        zm = v & (fm == z)
        base = binary_mask(np.where(zm, p, np.nan), zm, None, c["frac"])
        out |= apply_per_zone(base, fm, v, {z: c["ops"]}, ctx.data.grid.pixel_km)
    return out


def combine_choices(choices: dict, ops: list[str]) -> dict:
    """Sequential combination of several operations, each with its own per-fold (dev-chosen) parameters; the
    area fraction is the first operation's choice."""
    zones = sorted(next(iter(choices.values())).keys())
    out = {}
    for z in zones:
        seq = []
        for op in ops:
            seq += choices[op][z]["ops"]
        out[z] = {"frac": choices[ops[0]][z]["frac"], "ops": seq}
    return out
