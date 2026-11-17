"""Leakage-resistant geographic validation for G14 (Phase 3).

``UnitSystem`` gives every experiment the same units, masks and evaluation definitions:

* backend ``g13``: ``gems_bench.g13.splits.Splits`` (geographic folds with a 13 km embargo, an untouched TEST fold,
  whole fault-system discovery splits). Built before any patch is extracted.
* backend ``phase5``: the Phase-5 context folds / discovery splits used by P6 W2, M9 and P7, so the strongest
  existing maps (T0) are scored on exactly the same units. There is no locked TEST fold in this backend: the report
  says so and the final claim must then come from the g13 backend.

Protocols
---------
``spatial_cv`` / ``spatial_cv_embargo``  geographic folds (embargo = the folds' buffer, in metres)
``leave_one_region_out``                 each CV fold is a region; identical to spatial CV with one region per fold
``fault_system_holdout``                 discovery splits (whole linked systems withheld from supervision)
``leave_one_regime_out``                 regimes fitted label-free on CV training pixels (G13 regimes), each regime
                                          held out with a metric embargo; the TEST exclusion is never trained on

Context windows: regional / global inputs are label-free, so a training window may *see* held-out geography as
input; it never receives held-out labels (losses are masked to the unit's training pixels) and the known-fault
context is computed from the unit's permitted map only. ``strict_context_embargo`` additionally drops training
centres whose regional window reaches the held-out zone.

UNTESTED: written without execution.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage


@dataclass
class TrainUnit:
    task: str           # geo | disc_official | regime
    zone: int
    train: np.ndarray   # pixels whose labels may supervise (bool, full grid)
    inner_val: np.ndarray   # early-stopping pixels inside the training geography (never the held-out zone)
    labels: np.ndarray  # supervision visible to this unit
    ignore: np.ndarray  # pixels with no negative supervision (discovery ignore ring)
    permitted_known: np.ndarray   # known-fault map the unit may use as input context
    heldout: np.ndarray  # evaluation zone of the unit

    @property
    def key(self) -> str:
        return f"{self.task}_{self.zone}"


class UnitSystem:
    def __init__(self, backend: str, data, *, g13_splits=None, ctx=None, regimes: np.ndarray | None = None,
                 embargo_m: float = 13000.0, pixel_m: float = 100.0):
        self.backend, self.data, self.embargo_m, self.pixel_m = backend, data, embargo_m, pixel_m
        self.sp, self.ctx, self.regimes = g13_splits, ctx, regimes
        if backend == "g13" and g13_splits is None:
            raise ValueError("g13 backend needs gems_bench.g13.splits.Splits")
        if backend == "phase5" and ctx is None:
            raise ValueError("phase5 backend needs a Phase-5 context (gems_bench.phase5_context.load_context5)")
        if backend == "phase5":
            lab5, val5 = np.asarray(ctx.data.labels, bool), np.asarray(ctx.data.valid, bool)
            if lab5.shape != data.shape:
                raise ValueError(f"Phase-5 grid {lab5.shape} differs from the G13 GeoData grid {data.shape}")
            self.phase5_mismatch = {"labels_px": int((lab5 != data.labels).sum()),
                                    "valid_px": int((val5 != data.valid).sum())}
            if self.phase5_mismatch["labels_px"]:
                raise ValueError(f"Phase-5 labels differ from G13 labels on {self.phase5_mismatch['labels_px']} px: "
                                 "the two loaders disagree; refuse to mix units")

    # ------------------------------------------------------------------ geography
    @property
    def labels(self) -> np.ndarray:
        return self.data.labels

    @property
    def cv_zones(self) -> list[int]:
        return list(self.sp.cv_folds) if self.backend == "g13" else list(self.ctx.fold_ids)

    @property
    def n_disc(self) -> int:
        return self.sp.n_splits if self.backend == "g13" else int(self.ctx.splits.n_splits)

    def test_exclusion(self) -> np.ndarray:
        if self.backend == "g13":
            return self.sp.test_exclusion()
        return np.zeros(self.data.shape, bool)

    @property
    def has_locked_test(self) -> bool:
        return self.backend == "g13"

    def _fold_masks(self, k: int):
        if self.backend == "g13":
            from ..folds import ROLE

            r = self.sp.roles[k]
            train = self.sp.cv_region() & (r == ROLE["train"])
            inner = self.sp.cv_region() & (r == ROLE["inner_val"])
            return train, inner, self.sp.val(k) & self.data.valid
        f = self.ctx.folds
        return f.inner_train(k) & self.data.valid, f.inner_val(k) & self.data.valid, f.val(k) & self.data.valid

    # ------------------------------------------------------------------ units
    def geo_unit(self, k: int) -> TrainUnit:
        train, inner, held = self._fold_masks(k)
        lab = self.labels
        if self.backend == "g13":
            permitted = lab & ~(self.sp.val(k) | self.sp.buffer(k)) & ~self.test_exclusion()
        else:
            permitted = lab & (self.ctx.folds.fold_map != k)
        return TrainUnit("geo", k, train, inner, lab & (train | inner), np.zeros_like(lab), permitted, held)

    def disc_unit(self, d: int) -> TrainUnit:
        if self.backend == "g13":
            sp = self.sp
            region = sp.cv_region()
            vis = sp.disc_labels(d)
            withheld, ignore, ev = sp.withheld(d), sp.ignore(d), sp.eval_zone(d)
        else:
            s = self.ctx.splits
            region = self.data.valid
            withheld, ignore, ev = s.withheld(d), s.ignore(d), s.eval_zone(d) & self.data.valid
            vis = self.labels & ~withheld
        # inner validation: a deterministic 1-in-5 block of 20 km cells of the training geography
        inner = region & _block_select(self.data.shape, 200, d) & ~ignore
        train = region & ~inner & ~ignore
        return TrainUnit("disc_official", d, train, inner, vis & (train | inner), ignore, vis, ev)

    def regime_units(self) -> list[TrainUnit]:
        if self.regimes is None:
            raise ValueError("leave_one_regime_out needs a label-free regime raster (G13 regimes.npy)")
        out = []
        emb = int(round(self.embargo_m / self.pixel_m))
        base = self.data.valid & ~self.test_exclusion()
        for r in sorted(set(np.unique(self.regimes)) - {-1}):
            held = base & (self.regimes == r)
            if held.sum() < 1000:
                continue
            buf = ndimage.distance_transform_edt(~held) <= emb
            region = base & ~buf
            inner = region & _block_select(self.data.shape, 200, int(r))
            train = region & ~inner
            lab = self.labels
            out.append(TrainUnit("regime", int(r), train, inner, lab & (train | inner), np.zeros_like(lab),
                                 lab & ~held & ~buf, held))
        return out

    def units_for(self, protocol: str) -> list[TrainUnit]:
        if protocol in ("spatial_cv", "spatial_cv_embargo", "leave_one_region_out"):
            return [self.geo_unit(k) for k in self.cv_zones]
        if protocol == "fault_system_holdout":
            return [self.disc_unit(d) for d in range(self.n_disc)]
        if protocol == "leave_one_regime_out":
            return self.regime_units()
        raise KeyError(protocol)

    def final_unit(self) -> TrainUnit:
        """Submission model: trains on everything except the locked TEST fold (+ buffer); all mapped faults are context
        at inference (they are a test-time input)."""
        ex = self.test_exclusion()
        region = self.data.valid & ~ex
        inner = region & _block_select(self.data.shape, 200, 99)
        lab = self.labels
        return TrainUnit("final", 0, region & ~inner, inner, lab & region, np.zeros_like(lab), lab & ~ex,
                         self.data.valid & ex)

    # ------------------------------------------------------------------ official evaluation units
    def eval_units(self, tasks=("geo", "disc_official")) -> list:
        if self.backend == "g13":
            return self.sp.units(tasks)
        from ..gems3.official import build_units

        return build_units(self.ctx, tasks)

    def test_unit(self):
        return self.sp.test_unit() if self.backend == "g13" else None

    # ------------------------------------------------------------------ audits
    def audit(self, units: list[TrainUnit], context_radius_px: int = 0) -> pd.DataFrame:
        rows = []
        ex = self.test_exclusion()
        for u in units:
            d_held = ndimage.distance_transform_edt(~u.heldout) * self.pixel_m / 1000 if u.heldout.any() else None
            r = {"unit": u.key, "train_px": int(u.train.sum()), "inner_val_px": int(u.inner_val.sum()),
                 "heldout_px": int(u.heldout.sum()), "train_pos": int((u.labels & u.train).sum()),
                 "train_heldout_overlap": int((u.train & u.heldout).sum()),
                 "train_test_overlap": int((u.train & ex).sum()),
                 "inner_heldout_overlap": int((u.inner_val & u.heldout).sum()),
                 "min_train_to_heldout_km": float(d_held[u.train].min()) if d_held is not None and u.train.any() else np.nan,
                 "heldout_truth_supervised": int((u.labels & self._truth_of(u)).sum()),
                 "permitted_known_in_heldout": int((u.permitted_known & u.heldout & self._truth_of(u)).sum())}
            if context_radius_px and d_held is not None:
                r["train_px_with_context_reaching_heldout"] = int((u.train & (d_held * 1000 / self.pixel_m
                                                                              <= context_radius_px)).sum())
            rows.append(r)
        return pd.DataFrame(rows)

    def _withheld(self, u: TrainUnit) -> np.ndarray:
        if self.backend == "g13":
            return self.sp.withheld(u.zone)
        return self.ctx.splits.withheld(u.zone)

    def _truth_of(self, u: TrainUnit) -> np.ndarray:
        if u.task == "disc_official":
            return self._withheld(u)
        return self.labels & u.heldout

    def assert_no_leakage(self, units: list[TrainUnit]) -> None:
        ex = self.test_exclusion()
        for u in units:
            assert not (u.train & u.heldout).any(), f"{u.key}: training pixels inside the held-out zone"
            assert not (u.inner_val & u.heldout).any(), f"{u.key}: inner validation inside the held-out zone"
            if u.task != "final":
                assert not ((u.train | u.inner_val) & ex).any(), f"{u.key}: trains on the locked TEST partition"
            assert not (u.permitted_known & self._truth_of(u)).any(), f"{u.key}: held-out truth in the known context"
            if u.task == "disc_official":
                assert not (u.labels & self._withheld(u)).any(), f"{u.key}: withheld systems supervised"
            assert (u.labels & u.train).sum() > 0, f"{u.key}: no positive supervision"

    # ------------------------------------------------------------------ manifest
    def manifest(self, units: list[TrainUnit], protocol: str, extra: dict | None = None) -> dict:
        h = hashlib.sha256()
        for u in units:
            for a in (u.train, u.inner_val, u.heldout, u.labels, u.permitted_known):
                h.update(np.packbits(a).tobytes())
        t = self.data.transform
        return {"backend": self.backend, "protocol": protocol, "has_locked_test": self.has_locked_test,
                "embargo_m": self.embargo_m, "crs": str(self.data.crs), "transform": list(t)[:6],
                "shape": list(self.data.shape), "units": [u.key for u in units], "mask_digest": h.hexdigest()[:16],
                "test_exclusion_px": int(self.test_exclusion().sum()),
                "g13_params": getattr(self.sp, "params", None) if self.sp is not None else None,
                "created_unix": time.time(), **(extra or {})}


def _block_select(shape, block_px: int, salt: int) -> np.ndarray:
    """Deterministic 1-in-5 selection of square blocks (inner validation inside a training geography)."""
    h, w = shape
    r, c = np.indices(((h + block_px - 1) // block_px, (w + block_px - 1) // block_px))
    sel = ((r * 7 + c * 13 + salt * 3) % 5) == 0
    return np.kron(sel, np.ones((block_px, block_px), bool))[:h, :w]


# =============================================================================== fault-system grouping
def fault_system_groups(labels: np.ndarray, link_m: float = 1000.0, angle_deg: float = 30.0,
                        pixel_m: float = 100.0) -> tuple[np.ndarray, pd.DataFrame]:
    """Orientation-aware grouping of mapped-fault components into systems.

    Components (8-connected) are linked when their gap is <= ``link_m`` AND either they touch (gap <= 1 px) or their
    principal strikes differ by <= ``angle_deg`` and the gap direction is roughly collinear with both. Distance alone
    never merges two faults. Returns (system id per label pixel, -1 elsewhere; component table)."""
    lab, n = ndimage.label(labels, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.full(labels.shape, -1, np.int32), pd.DataFrame()
    objs = ndimage.find_objects(lab)
    comps = []
    for i, sl in enumerate(objs, 1):
        rr, cc = np.nonzero(lab[sl] == i)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        X = np.c_[cc, -rr].astype(float)
        if len(X) >= 3:
            _, _, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
            strike = float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180)
        else:
            strike = np.nan
        comps.append({"comp": i, "n_px": len(rr), "strike_deg": strike, "r": rr, "c": cc})
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    link_px = link_m / pixel_m
    pad = int(np.ceil(link_px)) + 2
    H, W = labels.shape
    for comp, sl in zip(comps, objs):
        i = comp["comp"]
        # work inside the component's bounding box expanded by the link distance (cheap on the full grid)
        r0, r1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
        c0, c1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
        win = lab[r0:r1, c0:c1]
        m = win == i
        others = np.where(m, 0, win)
        if not (others > 0).any():
            continue
        # gap from every pixel of this component to the nearest pixel of each other component
        _, (jr, jc) = ndimage.distance_transform_edt(others == 0, return_indices=True)
        mr, mc = np.nonzero(m)
        nr, nc = jr[mr, mc], jc[mr, mc]
        gaps = np.hypot(mr - nr, mc - nc)
        nb = others[nr, nc]
        cand = {}
        for j, g in zip(nb, gaps):
            if j and j != i and g < cand.get(int(j), 1e9):
                cand[int(j)] = float(g)
        for j, gap in cand.items():
            if gap > link_px:
                continue
            a, b = comps[i - 1], comps[j - 1]
            ok = gap <= 1.5
            if not ok and np.isfinite(a["strike_deg"]) and np.isfinite(b["strike_deg"]):
                dang = abs(((a["strike_deg"] - b["strike_deg"]) + 90) % 180 - 90)
                ok = dang <= angle_deg
            if ok:
                parent[find(i)] = find(j)
    roots = {i: find(i) for i in range(1, n + 1)}
    ids = {r: k for k, r in enumerate(sorted(set(roots.values())))}
    lut = np.full(n + 1, -1, np.int32)
    for i, r in roots.items():
        lut[i] = ids[r]
    sysmap = lut[lab]
    table = pd.DataFrame([{"comp": c["comp"], "n_px": c["n_px"], "strike_deg": c["strike_deg"],
                           "system": int(lut[c["comp"]])} for c in comps])
    return sysmap, table


def system_crossing_report(sysmap: np.ndarray, units: list[TrainUnit]) -> pd.DataFrame:
    """Systems with pixels both in a unit's supervised labels and in its held-out truth (should be none for disc)."""
    rows = []
    for u in units:
        sup = np.unique(sysmap[u.labels & (u.train | u.inner_val) & (sysmap >= 0)])
        held = np.unique(sysmap[u.heldout & (sysmap >= 0)])
        rows.append({"unit": u.key, "systems_supervised": len(sup), "systems_in_heldout": len(held),
                     "systems_in_both": int(len(np.intersect1d(sup, held)))})
    return pd.DataFrame(rows)


def save_manifest(man: dict, audit: pd.DataFrame, root: Path) -> list[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "fold_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    audit.to_csv(root / "fold_audit.csv", index=False)
    return [root / "fold_manifest.json", root / "fold_audit.csv"]
