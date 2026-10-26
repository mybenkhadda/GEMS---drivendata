"""Milestone E: coherent fault *candidates* instead of independent pixels (no neural training).

Sources (all rasters already cached; nothing is rebuilt)::

    feature   label-free feature-store layers (|robust z| of magnetic / gravity / conductivity / terrain edges and
              curvature): multiscale crest lines (p3.lineaments.ridge_mask) -> branches -> ordered paths.
              Computed ONCE on the full grid and reused by every unit.
    model     crest lines of an honest model map per unit (geo: OOF, disc: the split's discovery map, full: the
              fold-average full-region map).
    tip       straight extensions of the endpoints of the faults KNOWN to the unit (disc: retained faults; full: all
              mapped faults; geo: none, its truth is the fold's own mapped faults).

Fusion: candidates are processed longest first; one that lies >= ``dedup_overlap`` within ``dedup_radius_px`` of
already kept candidates is a duplicate and only adds its source to the owner's provenance (``sources``).
Cross-modal agreement: share of a candidate's pixels within ``agreement_radius_px`` of each other source's raw
candidates (``agree_<source>``) and the number of sources agreeing on >= half of it (``n_agree``).

Paths are stored in the unit's crop coordinates (``Unit.sl``); ``to_full`` converts them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from ..postprocess.topology import _disk, _endpoint_tangents
from .official import Unit


@dataclass
class CandidateSet:
    table: pd.DataFrame
    paths: list = field(default_factory=list)          # (n, 2) int arrays, crop coordinates

    def __len__(self) -> int:
        return len(self.paths)

    def raster(self, shape, values=None) -> np.ndarray:
        """Max-rasterised candidates (1-px lines); ``values`` per candidate (default 1)."""
        out = np.zeros(shape, np.float32)
        vals = np.ones(len(self.paths), np.float32) if values is None else np.asarray(values, np.float32)
        for p, v in zip(self.paths, vals):
            np.maximum.at(out, (p[:, 0], p[:, 1]), v)
        return out

    def subset(self, idx) -> "CandidateSet":
        idx = list(idx)
        return CandidateSet(self.table.iloc[idx].reset_index(drop=True), [self.paths[i] for i in idx])

    # -- persistence -----------------------------------------------------------
    def to_npz(self) -> dict:
        lens = np.array([len(p) for p in self.paths], np.int64)
        rc = np.concatenate(self.paths).astype(np.int32) if self.paths else np.zeros((0, 2), np.int32)
        return {"rc": rc, "offsets": np.r_[0, np.cumsum(lens)]}

    @classmethod
    def from_npz(cls, table: pd.DataFrame, z) -> "CandidateSet":
        rc, off = z["rc"], z["offsets"]
        return cls(table.reset_index(drop=True), [rc[off[i]:off[i + 1]] for i in range(len(off) - 1)])


def to_full(path: np.ndarray, u: Unit) -> np.ndarray:
    return path + np.array([u.sl[0].start, u.sl[1].start])


# =============================================================================== generators
def crest_paths(layer: np.ndarray, valid: np.ndarray, *, scale_m: float, top_frac: float, min_len_km: float,
                pixel_m: float = 100.0) -> list[np.ndarray]:
    from ..p3.lineaments import ridge_mask, split_branches, trace

    if not valid.any():
        return []
    sk, _ = ridge_mask(layer, valid, scale_m=scale_m, top_frac=top_frac, pixel_m=pixel_m)
    lab, _, _ = split_branches(sk)
    out = []
    min_px = max(int(np.ceil(min_len_km * 1000.0 / (pixel_m * 1.12))), 2)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        loc = np.argwhere(lab[sl] == i) + np.array([sl[0].start, sl[1].start])
        if len(loc) >= min_px:
            out.append(trace(loc))
    return out


def tip_paths(known: np.ndarray, valid: np.ndarray, lengths_km, tail_px: int = 5, pixel_m: float = 100.0) -> list:
    """(paths, lengths) of straight endpoint extensions of the known traces, clipped to ``valid``."""
    from skimage.draw import line as draw_line

    from ..geology.skeleton import skeletonize

    if known is None or not known.any():
        return []
    sk = skeletonize(known)
    h, w = known.shape
    out = []
    for r, c, ang, _ in _endpoint_tangents(sk, tail_px):
        t = np.radians(ang)
        for L in lengths_km:
            n = int(round(L * 1000.0 / pixel_m))
            r1, c1 = int(round(r - n * np.sin(t))), int(round(c + n * np.cos(t)))
            rr, cc = draw_line(r, c, r1, c1)
            keep = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            rr, cc = rr[keep][1:], cc[keep][1:]                  # drop the endpoint itself (it is known)
            ok = valid[rr, cc] & ~known[rr, cc]
            if ok.sum() >= 2:
                out.append((np.c_[rr[ok], cc[ok]], float(L)))
    return out


def feature_layer(ctx, src: dict) -> np.ndarray | None:
    if not ctx.store.has(src["feature"]):
        return None
    a = np.asarray(ctx.store.get(src["feature"]), np.float32)
    return np.abs(a) if src.get("absolute", True) else a


def label_free(ctx, cfg: dict) -> tuple[CandidateSet, pd.DataFrame]:
    """Feature-layer candidates on the full grid (computed once; cached by the caller)."""
    cc = cfg["candidates"]
    rows, paths, log = [], [], []
    for src in cc["sources"]:
        a = feature_layer(ctx, src)
        if a is None:
            log.append({"source": src["name"], "feature": src["feature"], "status": "missing in feature store", "n": 0})
            continue
        ps = crest_paths(a, ctx.data.valid & np.isfinite(a), scale_m=src["scale_m"], top_frac=src["top_frac"],
                         min_len_km=cc["min_len_km"])
        for p in ps:
            rows.append({"source": src["name"], "kind": "feature", "scale_m": src["scale_m"]})
            paths.append(p)
        log.append({"source": src["name"], "feature": src["feature"], "status": "ok", "n": len(ps)})
        print(f"[gems3] candidates {src['name']}: {len(ps)}")
    return CandidateSet(pd.DataFrame(rows), paths), pd.DataFrame(log)


def unit_candidates(u: Unit, lf: CandidateSet, model_scores: dict, cfg: dict) -> CandidateSet:
    """Label-free candidates clipped to the unit + its model crest lines + tip extensions, then fused."""
    cc = cfg["candidates"]
    min_px = max(int(np.ceil(cc["min_len_km"] / 0.112)), 2)
    off = np.array([u.sl[0].start, u.sl[1].start])
    h, w = u.ev.shape
    rows, paths = [], []
    for r, p in zip(lf.table.itertuples(), lf.paths):
        q = p - off
        ok = (q[:, 0] >= 0) & (q[:, 0] < h) & (q[:, 1] >= 0) & (q[:, 1] < w)
        q = q[ok]
        q = q[u.ev[q[:, 0], q[:, 1]]] if len(q) else q
        if len(q) >= min_px:
            rows.append({"source": r.source, "kind": r.kind, "scale_m": r.scale_m})
            paths.append(q)
    for src in cc.get("model_sources") or []:
        s = model_scores.get(src["model"])
        if s is None:
            continue
        for p in crest_paths(np.nan_to_num(s, nan=0.0), u.ev, scale_m=src["scale_m"], top_frac=src["top_frac"],
                             min_len_km=cc["min_len_km"]):
            rows.append({"source": src["name"], "kind": "model", "scale_m": src["scale_m"]})
            paths.append(p)
    if u.known is not None and u.task != "geo":
        for p, L in tip_paths(u.known, u.ev, cc["tip_extension_km"], cc["tip_tail_px"]):
            rows.append({"source": f"tip{L:g}km", "kind": "tip", "scale_m": np.nan})
            paths.append(p)
    raw = CandidateSet(pd.DataFrame(rows), paths)
    return fuse(raw, u.ev.shape, cc)


def fuse(raw: CandidateSet, shape, cc: dict) -> CandidateSet:
    """Duplicate removal with provenance, then cross-modal agreement columns."""
    if not len(raw):
        return CandidateSet(pd.DataFrame(columns=["cand_id", "source", "kind", "sources", "n_sources"]), [])
    rad = int(cc["dedup_radius_px"])
    disk = np.argwhere(_disk(rad)) - rad
    h, w = shape
    order = np.argsort([-len(p) for p in raw.paths], kind="stable")
    claimed = np.zeros(shape, np.int32)                     # kept-candidate index + 1 within the radius
    keep, prov = [], {}
    for i in order:
        p = raw.paths[i]
        own = claimed[p[:, 0], p[:, 1]]
        if (own > 0).mean() >= cc["dedup_overlap"]:
            owner = int(np.bincount(own[own > 0]).argmax()) - 1
            prov[owner].add(raw.table.source.iloc[i])
            continue
        k = len(keep)
        keep.append(i)
        prov[k] = {raw.table.source.iloc[i]}
        pts = (p[:, None, :] + disk[None, :, :]).reshape(-1, 2)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < h) & (pts[:, 1] >= 0) & (pts[:, 1] < w)]
        free = claimed[pts[:, 0], pts[:, 1]] == 0
        claimed[pts[free, 0], pts[free, 1]] = k + 1
    out = raw.subset(keep)
    out.table["cand_id"] = np.arange(len(out))
    out.table["sources"] = [";".join(sorted(prov[k])) for k in range(len(out))]
    out.table["n_sources"] = [len(prov[k]) for k in range(len(out))]
    # agreement with every source's raw candidates
    arad = _disk(int(cc["agreement_radius_px"]))
    for s in sorted(raw.table.source.unique()):
        m = np.zeros(shape, bool)
        for j in np.flatnonzero(raw.table.source.to_numpy() == s):
            m[raw.paths[j][:, 0], raw.paths[j][:, 1]] = True
        m = ndimage.binary_dilation(m, structure=arad)
        out.table[f"agree_{s}"] = [float(m[p[:, 0], p[:, 1]].mean()) for p in out.paths]
    ag = [c for c in out.table.columns if c.startswith("agree_")]
    own_src = out.table.source.to_numpy()
    out.table["n_agree"] = [int(sum(out.table.at[i, c] >= 0.5 for c in ag if c != f"agree_{own_src[i]}"))
                            for i in range(len(out))]
    return out


def geometry_table(cs: CandidateSet) -> pd.DataFrame:
    from ..p3.lineaments import geometry

    g = [geometry(p) if len(p) > 1 else {} for p in cs.paths]
    df = pd.DataFrame(g)
    df["n_px"] = [len(p) for p in cs.paths]
    df["r_mid"] = [int(p[len(p) // 2, 0]) for p in cs.paths]
    df["c_mid"] = [int(p[len(p) // 2, 1]) for p in cs.paths]
    return df
