"""Section 05: synthetic fault-map corruption (conditional reconstruction) and the CV-C benchmark sets.

Training (``Corruptor``, applied per patch to VISIBLE labels only)
    mode ~ mixture {none, partial, heavy}:
        none      no map context (context channel 0, flag 0): independent discovery, the network must find every
                  visible fault from the geophysics alone (this is also "full removal of map context");
        partial   each eligible element damaged with probability ``partial_strength`` x operator weight;
        heavy     the same with ``heavy_strength``.
    operators (on the patch's visible map):
        endpoint_truncation    remove a disk of L km around a skeleton endpoint
        interior_removal       remove a disk of L / 2 km around an interior skeleton pixel (gaps)
        segment_omission       remove a whole 8-connected trace
        branch_omission        remove a branch (< max_km) attached to a junction
        local_displacement     shift a trace by 1-2 px across strike (context shows the WRONG position; the target is
                               the original trace)
        whole_system_removal   remove every pixel of one fault system (system ids from the discovery split map)
    The context channel shows the damaged map; ``removed`` = visible pixels the context no longer shows (the L6
    reconstruction target). Seg targets are always the full visible map, so damaged and intact geometry are both
    supervised, while pixels SHOWN as context get the low ``context_pixel_weight`` (copying earns little).
    Leakage: only visible labels (inside the unit's permitted geography, ignored systems removed) ever enter the
    context; the input features are label-free; nothing else carries the original geometry.

Evaluation (``benchmark_sets``): M7's corruption classes (``m7.corruption.corrupt_fold``) on the systems of each
TRAINED geographic fold, systems assigned to folds BEFORE damage (``gems3.map_completion.systems_by_fold``). The
model of fold k never saw these systems; context = the damaged map; truth = removed pixels of one class; evaluation
zone = valid pixels of fold k's validation zone within ``eval_radius_km`` of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

PX_KM = 0.1
_S8 = np.ones((3, 3), bool)


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _remove_disk(ctx: np.ndarray, r: int, c: int, rad: int) -> None:
    h, w = ctx.shape
    r0, r1, c0, c1 = max(r - rad, 0), min(r + rad + 1, h), max(c - rad, 0), min(c + rad + 1, w)
    d = _disk(rad)[rad - (r - r0):rad + (r1 - r), rad - (c - c0):rad + (c1 - c)]
    ctx[r0:r1, c0:c1] &= ~d


class Corruptor:
    def __init__(self, cc: dict, rng: np.random.Generator):
        self.cc, self.rng = cc, rng
        mx = cc["mixture"]
        self.modes = ["none", "partial", "heavy"]
        self.p = np.array([float(mx[m]) for m in self.modes])
        self.p = self.p / self.p.sum()
        self.ops = cc["operators"]

    def _prob(self, strength: float, op: str) -> float:
        return float(np.clip(strength * float(self.ops.get(op, {}).get("weight", 0.0)), 0.0, 1.0))

    def __call__(self, seg: np.ndarray, sysmap: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], str]:
        """(context shown, removed, operations applied, mode)."""
        mode = self.modes[int(self.rng.choice(3, p=self.p))]
        if mode == "none" or not seg.any():
            return np.zeros(seg.shape, bool), np.zeros(seg.shape, bool), [], mode
        s = float(self.cc["partial_strength"] if mode == "partial" else self.cc["heavy_strength"])
        ctx = seg.copy()
        ops: list[str] = []
        rng = self.rng
        if "whole_system_removal" in self.ops:
            ids = np.unique(sysmap[seg])
            ids = ids[ids >= 0]
            pk = ids[rng.random(len(ids)) < self._prob(s, "whole_system_removal")]
            if len(pk):
                ctx &= ~np.isin(sysmap, pk)
                ops.append("whole_system_removal")
        if "segment_omission" in self.ops and ctx.any():
            lab, n = ndimage.label(ctx, structure=_S8)
            pk = np.flatnonzero(rng.random(n) < self._prob(s, "segment_omission")) + 1
            if len(pk):
                ctx &= ~np.isin(lab, pk)
                ops.append("segment_omission")
        if ctx.any() and any(o in self.ops for o in ("branch_omission", "endpoint_truncation", "interior_removal")):
            from ..geology.skeleton import endpoints, skeletonize

            sk = skeletonize(ctx)
            if "branch_omission" in self.ops:
                from ..p3.lineaments import split_branches

                lb, _, junction = split_branches(sk, 2)
                if junction.any():
                    jz = ndimage.binary_dilation(junction, structure=_S8, iterations=2)
                    mx = float(self.ops["branch_omission"].get("max_km", 3.0))
                    hit = False
                    for i, sl in enumerate(ndimage.find_objects(lb), start=1):
                        if sl is None or rng.random() >= self._prob(s, "branch_omission"):
                            continue
                        b = np.zeros(ctx.shape, bool)
                        b[sl] = lb[sl] == i
                        if b.sum() * 0.112 > mx or not (ndimage.binary_dilation(b, structure=_S8) & jz).any():
                            continue
                        ctx &= ~(ndimage.binary_dilation(b, structure=_S8) & ~jz)
                        hit = True
                    if hit:
                        ops.append("branch_omission")
            if "endpoint_truncation" in self.ops:
                eps = np.argwhere(endpoints(sk) & ctx)
                sel = eps[rng.random(len(eps)) < self._prob(s, "endpoint_truncation")] if len(eps) else eps
                lo, hi = self.ops["endpoint_truncation"].get("km", [0.3, 1.5])
                for r, c in sel:
                    _remove_disk(ctx, int(r), int(c), max(int(round(rng.uniform(lo, hi) / PX_KM)), 1))
                if len(sel):
                    ops.append("endpoint_truncation")
            if "interior_removal" in self.ops:
                ep = ndimage.binary_dilation(endpoints(sk), iterations=4)
                inner = np.argwhere(sk & ctx & ~ep)
                n = int(rng.binomial(max(len(inner) // 40, 1), self._prob(s, "interior_removal"))) if len(inner) else 0
                lo, hi = self.ops["interior_removal"].get("km", [0.5, 2.0])
                for r, c in inner[rng.choice(len(inner), min(n, len(inner)), replace=False)] if n else []:
                    _remove_disk(ctx, int(r), int(c), max(int(round(rng.uniform(lo, hi) / PX_KM / 2.0)), 1))
                if n:
                    ops.append("interior_removal")
        removed = seg & ~ctx
        if "local_displacement" in self.ops and ctx.any():
            lab, n = ndimage.label(ctx, structure=_S8)
            pk = np.flatnonzero(rng.random(n) < self._prob(s, "local_displacement")) + 1
            lo, hi = self.ops["local_displacement"].get("px", [1, 2])
            for i in pk:
                m = lab == i
                d = int(rng.integers(int(lo), int(hi) + 1))
                dr, dc = [(d, 0), (0, d), (-d, 0), (0, -d)][int(rng.integers(4))]
                moved = np.roll(m, (dr, dc), axis=(0, 1))
                removed |= m & ~moved
                ctx = (ctx & ~m) | moved
            if len(pk):
                ops.append("local_displacement")
        return ctx, removed & seg, ops, mode


# =============================================================================== audit samples (section 05)
def audit_samples(m9, builder, n: int) -> tuple[dict, pd.DataFrame]:
    """``n`` deterministic training samples of one builder: inputs are not stored, only label-derived planes."""
    rows, planes = [], {"seg": [], "context": [], "removed": []}
    for i in range(n):
        x, xc, m, rem, t = builder.sample()
        planes["seg"].append((t["seg"] > 0))
        planes["context"].append(m[0] > 0)
        planes["removed"].append(rem)
        rows.append({"sample": i, "flag": float(m[1].max()), "seg_px": int((t["seg"] > 0).sum()),
                     "context_px": int((m[0] > 0).sum()), "removed_px": int(rem.sum()),
                     "context_on_non_visible_px": int(((m[0] > 0) & ~(t["seg"] > 0)).sum())})
    arr = {k: np.packbits(np.stack(v).astype(bool), axis=-1) for k, v in planes.items()}
    return arr, pd.DataFrame(rows)


# =============================================================================== CV-C benchmark sets
def benchmark_sets(m9) -> tuple[dict, pd.DataFrame]:
    """{fold k: {"damaged": bool map, "removed": {class: bool map}}} for the trained folds + protocol rows."""
    from ..gems3.map_completion import systems_by_fold
    from ..m7.corruption import CLASSES, corrupt_fold

    ctx = m9.ctx
    cc = m9.cfg["corruption"]["eval"]
    lab = np.asarray(ctx.data.labels, bool)
    sm = np.asarray(ctx.splits.system_map)
    sysf = systems_by_fold(sm, ctx.folds.fold_map)
    rng = np.random.default_rng(int(m9.cfg["seed"]) + 909)
    sets, proto = {}, []
    for k in m9.train_folds():
        dmg, removed, rows = corrupt_fold(lab, sm, sysf, int(k), rng, cc)
        sets[int(k)] = {"damaged": dmg, "removed": {c: removed[c] for c in CLASSES}}
        proto += rows
    return sets, pd.DataFrame(proto)


def benchmark_units(m9, k: int, sets: dict) -> list:
    """One evaluation unit per corruption class of fold k (official convention: damaged map masked pixel-exactly,
    its predictions ignored), restricted to fold k's VALIDATION zone (the only geography the fold model never saw)."""
    from ..gems3.official import _unit

    cc = m9.cfg["corruption"]["eval"]
    valid = np.asarray(m9.ctx.data.valid, bool)
    val = m9.ctx.folds.val(k) & valid
    dmg = sets[k]["damaged"]
    out = []
    for cls, G in sets[k]["removed"].items():
        G = G & val
        if not G.any():
            continue
        zone = (ndimage.distance_transform_edt(~G, sampling=(0.1, 0.1)) <= float(cc["eval_radius_km"])) & val
        out.append(_unit(f"cvc_{cls}", int(k), zone, G, dmg, False, dmg))
    return out


def save_sets(ws, sets: dict) -> None:
    for k, s in sets.items():
        arr = {"damaged": np.packbits(s["damaged"], axis=-1), "shape": np.array(s["damaged"].shape)}
        for c, m in s["removed"].items():
            arr[f"removed__{c}"] = np.packbits(m, axis=-1)
        ws.save_npz(arr, f"corruption/benchmark/fold{k}.npz")


def load_sets(ws) -> dict:
    out = {}
    for p in sorted((ws.root / "corruption" / "benchmark").glob("fold*.npz")):
        z = np.load(p)
        shp = tuple(z["shape"])
        unpack = (lambda a: np.unpackbits(a, axis=-1)[:, :shp[1]].astype(bool))
        out[int(p.stem[4:])] = {"damaged": unpack(z["damaged"]),
                                "removed": {k.split("__", 1)[1]: unpack(z[k]) for k in z.files if k.startswith("removed__")}}
    return out
