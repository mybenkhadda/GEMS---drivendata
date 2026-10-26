"""Output representations: how a score map becomes the submitted raster (no labels are read here).

A representation is ``{select, post}``. ``select`` picks the core pixels inside the evaluation zone, ``post`` is
an ordered list of ``[operation, params]``; binary operations act on the current support (q > 0), the soft ones
(``two_level``, ``soft_top``) must come last.

Selectors (``frac`` = share of the zone's evaluation pixels, so equal ``frac`` means equal core area)::

    top     the top-``frac`` pixels of the score                                        (A4 / A10)
    crest   the top-``frac`` *crest* pixels: non-maximum suppression across the Hessian ridge normal of the
            lightly smoothed score, i.e. 1-px centre lines of the predicted corridors     (A6)
    raw     the score itself, clipped to [0, 1], no area budget                         (A0 / A1)

Post operations::

    skeleton           skeletonise, re-dilate to 2 w + 1 px                              (A6 / widths)
    dilate             isotropic disk dilation                                           (A7)
    oriented           line dilation across and / or along the local strike              (A8)
    confidence_width   pixels above the ``hi_quantile`` of the selected scores get ``extra_px`` more width (A9)
    two_level          core 1, halo of ``halo_px`` at ``halo_value``                     (soft vs binary)
    soft_top           core value v_min + (1 - v_min) rank (soft output on a binary support)
    known_ring         0 within ``ring_px`` of known faults, known pixels themselves untouched (official FP)
    known_fill         1 on known-fault pixels (neutral under the official mask; tests the other reading)

Theory note (p3/curves.py): for a fixed truth set the DTI-optimal map is binary, so soft operations can only
win through uncertainty about the truth; whether they do is an empirical question answered on held-out zones.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..postprocess.topology import _axial_diff, _disk, _line, orientation_field, skeleton

SELECTORS = ("top", "crest", "raw")


def rank01(p: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """Average rank of ``p`` inside ``ev`` scaled to [0, 1]; 0 outside (label-free, per zone)."""
    from scipy.stats import rankdata

    out = np.zeros(p.shape, np.float32)
    v = np.nan_to_num(np.asarray(p, np.float64)[ev], nan=0.0)
    if v.size:
        out[ev] = ((rankdata(v, method="average") - 1) / max(v.size - 1, 1)).astype(np.float32)
    return out


def fuse(r0: np.ndarray, r1: np.ndarray, w: float) -> np.ndarray:
    """w R0 + (1 - w) R1 of two per-zone rank maps (w = weight of the first member)."""
    return (w * r0 + (1.0 - w) * r1).astype(np.float32)


def _k(frac: float, ev: np.ndarray) -> int:
    return max(int(np.ceil(frac * int(ev.sum()))), 1)


def top_mask(score: np.ndarray, ev: np.ndarray, frac: float) -> np.ndarray:
    s = np.where(ev, np.nan_to_num(score, nan=-np.inf), -np.inf)
    v = s[ev]
    if not v.size:
        return np.zeros(ev.shape, bool)
    k = min(_k(frac, ev), v.size)
    thr = np.partition(v, v.size - k)[v.size - k]
    return ev & (s >= thr) & np.isfinite(s)


def crest_map(score: np.ndarray, ev: np.ndarray, sigma_px: float = 1.0) -> np.ndarray:
    """Pixels that are local maxima of the smoothed score across the ridge normal (inside ``ev``)."""
    from ..p3.lineaments import hessian_ridge

    s = np.where(ev, np.nan_to_num(score, nan=0.0), 0.0).astype(np.float32)
    ps = ndimage.gaussian_filter(s, sigma_px) if sigma_px > 0 else s
    strength, nr, nc = hessian_ridge(ps, max(sigma_px, 0.7))
    rr, cc = np.indices(ps.shape, dtype=np.float32)
    fwd = ndimage.map_coordinates(ps, [rr + nr, cc + nc], order=1, mode="nearest")
    bwd = ndimage.map_coordinates(ps, [rr - nr, cc - nc], order=1, mode="nearest")
    return ev & (strength > 0) & (ps >= fwd) & (ps >= bwd)


def crest_top(score: np.ndarray, ev: np.ndarray, frac: float, crest: np.ndarray) -> np.ndarray:
    """Top-``frac`` (share of ``ev``) crest pixels by score; all crest pixels if there are fewer."""
    return top_mask(score, ev & crest, min(1.0, _k(frac, ev) / max(int((ev & crest).sum()), 1)))


# =============================================================================== post operations
def _bin(q):
    return np.asarray(q) > 0


def op_skeleton(q, prm, a):
    sk = skeleton(_bin(q))
    w = int(prm.get("width_px", 0))
    return (ndimage.binary_dilation(sk, structure=_disk(w)) if w > 0 else sk).astype(np.float32)


def op_dilate(q, prm, a):
    r = int(prm.get("radius_px", 1))
    m = _bin(q)
    return (ndimage.binary_dilation(m, structure=_disk(r)) if r > 0 else m).astype(np.float32)


def oriented_dilate(mask: np.ndarray, across_px: int = 1, along_px: int = 0, n_dirs: int = 8,
                    sigma_px: float = 2.0) -> np.ndarray:
    """Dilate every pixel with line elements across (normal) and / or along its local strike."""
    if not mask.any() or (across_px <= 0 and along_px <= 0):
        return mask.copy()
    ang, _ = orientation_field(mask, sigma_px)
    out = mask.copy()
    half = 90.0 / n_dirs
    for i in range(n_dirs):
        a = 180.0 * i / n_dirs
        sel = mask & (_axial_diff(ang, a) <= half)
        if not sel.any():
            continue
        if across_px > 0:
            out |= ndimage.binary_dilation(sel, structure=_line(2 * across_px + 1, a + 90.0))
        if along_px > 0:
            out |= ndimage.binary_dilation(sel, structure=_line(2 * along_px + 1, a))
    return out


def op_oriented(q, prm, a):
    return oriented_dilate(_bin(q), int(prm.get("across_px", 1)), int(prm.get("along_px", 0)),
                           int(prm.get("n_dirs", 8))).astype(np.float32)


def op_confidence_width(q, prm, a):
    m = _bin(q)
    if not m.any():
        return m.astype(np.float32)
    s = np.nan_to_num(a["score"], nan=0.0)
    thr = np.quantile(s[m], float(prm.get("hi_quantile", 0.5)))
    hi = m & (s >= thr)
    extra = int(prm.get("extra_px", 1))
    return (m | ndimage.binary_dilation(hi, structure=_disk(extra))).astype(np.float32)


def op_two_level(q, prm, a):
    m = _bin(q)
    halo = ndimage.binary_dilation(m, structure=_disk(int(prm.get("halo_px", 1)))) & ~m
    return np.where(m, 1.0, np.where(halo, float(prm.get("halo_value", 0.5)), 0.0)).astype(np.float32)


def op_soft_top(q, prm, a):
    m = _bin(q)
    out = np.zeros(m.shape, np.float32)
    if m.any():
        vmin = float(prm.get("v_min", 0.5))
        out[m] = vmin + (1.0 - vmin) * rank01(a["score"], m)[m]
    return out


def op_known_ring(q, prm, a):
    known = a.get("known")
    r = int(prm.get("ring_px", 1))
    if known is None or r <= 0 or not known.any():
        return q
    ring = ndimage.binary_dilation(known, structure=_disk(r)) & ~known
    return np.where(ring, 0.0, q).astype(np.float32)


def op_known_fill(q, prm, a):
    known = a.get("known")
    return q if known is None else np.where(known, 1.0, q).astype(np.float32)


POST_OPS = {"skeleton": op_skeleton, "dilate": op_dilate, "oriented": op_oriented,
            "confidence_width": op_confidence_width, "two_level": op_two_level, "soft_top": op_soft_top,
            "known_ring": op_known_ring, "known_fill": op_known_fill}
SOFT_OPS = ("two_level", "soft_top")


def apply(rep: dict, score: np.ndarray, ev: np.ndarray, frac: float | None = None, *, known: np.ndarray | None = None,
          crest: np.ndarray | None = None, sigma_px: float = 1.0) -> np.ndarray:
    """Prediction raster (float32 in [0, 1], 0 outside ``ev``) of one representation."""
    sel = rep.get("select", "top")
    if sel == "raw":
        q = np.clip(np.nan_to_num(score, nan=0.0), 0.0, 1.0).astype(np.float32)
    elif sel == "top":
        q = top_mask(score, ev, frac).astype(np.float32)
    elif sel == "crest":
        c = crest if crest is not None else crest_map(score, ev, sigma_px)
        q = crest_top(score, ev, frac, c).astype(np.float32)
    else:
        raise KeyError(sel)
    aux = {"score": score, "ev": ev, "known": known}
    for name, prm in rep.get("post") or []:
        q = POST_OPS[name](q, prm or {}, aux)
    return np.where(ev, q, 0.0).astype(np.float32)


def with_ring(rep: dict, ring_px: int) -> dict:
    """``rep`` followed by a known-fault suppression ring (inserted before any soft operation)."""
    post = [list(x) for x in rep.get("post") or []]
    if ring_px <= 0:
        return {"select": rep.get("select", "top"), "post": post}
    i = next((j for j, (n, _) in enumerate(post) if n in SOFT_OPS), len(post))
    post.insert(i, ["known_ring", {"ring_px": int(ring_px)}])
    return {"select": rep.get("select", "top"), "post": post}


def family(name: str, rep: dict) -> str:
    """Report grouping: width / dilation / soft / area."""
    ops = [n for n, _ in rep.get("post") or []]
    if any(o in ("dilate", "oriented") for o in ops):
        return "dilation"
    if any(o in SOFT_OPS for o in ops):
        return "soft"
    if rep.get("select") == "crest" or any(o in ("skeleton", "confidence_width") for o in ops):
        return "width"
    return "area"
