"""Validation units and training visibilities without geological label leakage.

Validation units reuse the GEMS units exactly (``geomoe.scenarios.build`` on the attached engine):

    V1  disc_official   withheld complete fault systems of discovery split d (PRIMARY)
    V3  geo             geographic fold k (all mapped faults of the fold are truth)       (secondary)
    V2  v2_net          artificially hidden network elements inside fold k                (diagnostic)

A TRAINING VISIBILITY (``fit_id``) states which labels a model may see and where its training rows may come from::

    V1:d      visible = labels - withheld(d)                  region = footprint (deployment analogue: every
                                                              visible fault everywhere, hidden ones unlabelled)
                                                              or footprint - (eval zone d + buffer)
    V3:k      visible = labels outside fold k                 region = fold-k TRAIN role (fold + buffer removed)
    V1:j|k    visible = labels - withheld(j) - withheld(k)    nested inner fit of outer j, predicts unit V1:k
    V3:j|k    visible = labels outside folds j and k          nested inner fit of outer j, predicts unit V3:k
    FULL      visible = all labels                            region = footprint (full-region research only)

A fit may predict only the units listed in ``predicts``; ``verify_fold_isolation`` checks explicitly that none of the
truth pixels of those units (nor of the outer unit of a nested fit) is visible, that geographic training regions do
not touch the evaluation zone, and that the permitted known-fault context of a unit never contains its truth.
Inner folds for early stopping are spatial blocks of the training region, balanced on visible positives, with a
buffer that removes inner-validation pixels close to another inner fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage

from .contracts import LeakageError, SplitSpec


@dataclass
class ValidationUnit:
    scenario: str
    zone: int
    unit: object                # scored gems3 Unit
    base: object                # unit whose honest W2 / CNN maps are the sources
    known: np.ndarray           # cropped permitted known faults (never truth)
    sysmap: np.ndarray          # cropped system / event id on truth pixels, -1 elsewhere

    @property
    def key(self) -> str:
        return f"{self.scenario}:{self.zone}"

    @property
    def sl(self) -> tuple:
        return self.unit.sl

    def eval_truth(self) -> np.ndarray:
        return self.unit.eval_truth()

    def full(self, crop: np.ndarray, shape) -> np.ndarray:
        out = np.zeros(shape, bool)
        out[self.sl] = np.asarray(crop, bool)
        return out


# =============================================================================== loading
def load_geographic_folds(engine) -> dict:
    f = engine.ctx.folds
    return {"fold_map": np.asarray(f.fold_map), "roles": np.asarray(f.roles), "n_folds": int(f.n_folds),
            "params": dict(getattr(f, "params", {}) or {})}


def load_discovery_splits(engine) -> dict:
    s = engine.ctx.splits
    return {"splits": s, "n_splits": int(s.n_splits), "system_map": np.asarray(s.system_map),
            "systems": getattr(s, "systems", pd.DataFrame()), "params": dict(getattr(s, "params", {}) or {})}


def build_validation_units(engine, scenarios=("V1", "V3", "V2")) -> list[ValidationUnit]:
    from ..geomoe import scenarios as SC

    shim = SimpleNamespace(bst=engine.bst, ctx=engine.ctx, cfg={"v2": engine.v2_cfg})
    out = []
    for su in SC.build(shim):
        if su.scenario in scenarios:
            out.append(ValidationUnit(su.scenario, int(su.zone), su.unit, su.base, np.asarray(su.known, bool),
                                      np.asarray(su.sysmap, np.int32)))
    return out


# =============================================================================== visibilities
@dataclass
class Visibility:
    fit_id: str
    scenario: str
    outer: int | None
    inner: int | None
    visible: np.ndarray
    hidden: np.ndarray
    region: np.ndarray
    visible_rule: str
    region_rule: str
    predicts: tuple

    def spec(self) -> SplitSpec:
        from ..m7.workspace import mask_hash

        return SplitSpec(self.fit_id, self.scenario, self.outer, self.inner, self.visible_rule, self.region_rule,
                         mask_hash(self.visible), mask_hash(self.region), self.predicts)


def parse_fit_id(fit_id: str) -> tuple[str, int | None, int | None]:
    if fit_id == "FULL":
        return "FULL", None, None
    sc, rest = fit_id.split(":")
    if "|" in rest:
        j, k = rest.split("|")
        return sc, int(j), int(k)
    return sc, int(rest), None


def fit_ids(n_splits: int, n_folds: int, scenarios, nested: bool, nested_scenarios=("V1", "V3")) -> list[str]:
    out = []
    if "V1" in scenarios:
        out += [f"V1:{d}" for d in range(n_splits)]
    if "V3" in scenarios or "V2" in scenarios:
        out += [f"V3:{k}" for k in range(n_folds)]
    if nested:
        if "V1" in nested_scenarios:
            out += [f"V1:{j}|{k}" for j in range(n_splits) for k in range(n_splits) if j != k]
        if "V3" in nested_scenarios:
            out += [f"V3:{j}|{k}" for j in range(n_folds) for k in range(n_folds) if j != k]
    return out


def _dilate_km(mask: np.ndarray, km: float) -> np.ndarray:
    if km <= 0:
        return np.asarray(mask, bool)
    return ndimage.distance_transform_edt(~np.asarray(mask, bool), sampling=(0.1, 0.1)) <= km


def build_training_visibility(engine, fit_id: str, val_cfg: dict, scenarios=("V1", "V3", "V2")) -> Visibility:
    """Visible labels / hidden labels / training region of one fit (full-grid boolean arrays)."""
    from ..folds import ROLE

    lab, valid = engine.labels(), engine.valid()
    sc, j, k = parse_fit_id(fit_id)
    S, Fo = engine.ctx.splits, engine.ctx.folds
    mode = str(val_cfg.get("v1_training_region", "all_visible"))
    buf = float(val_cfg.get("v1_eval_buffer_km", 2.0))
    if sc == "FULL":
        hidden = np.zeros(lab.shape, bool)
        return Visibility(fit_id, "FULL", None, None, lab.copy(), hidden, valid.copy(), "all labels", "footprint",
                          ())
    if sc == "V1":
        zones = [j] if k is None else [j, k]
        hidden = np.zeros(lab.shape, bool)
        for z in zones:
            hidden |= S.withheld(z)
        region = valid.copy()
        rr = "footprint"
        if mode == "outside_eval_zone":
            for z in zones:
                region &= ~_dilate_km(S.eval_zone(z), buf)
            rr = f"footprint - (eval zone(s) {zones} + {buf} km)"
        pred = (f"V1:{j}",) if k is None else (f"V1:{k}",)
        return Visibility(fit_id, "V1", j, k, lab & ~hidden, hidden, region,
                          f"labels - withheld{zones}", rr, pred)
    if sc == "V3":
        zones = [j] if k is None else [j, k]
        fm = np.asarray(Fo.fold_map)
        hidden = lab & np.isin(fm, zones)
        region = valid.copy()
        for z in zones:
            region &= np.asarray(Fo.roles[z]) == ROLE["train"]
        tgt = j if k is None else k
        pred = tuple(x for x in (f"V3:{tgt}", f"V2:{tgt}") if x.split(":")[0] in scenarios)
        return Visibility(fit_id, "V3", j, k, lab & ~hidden, hidden, region, f"labels outside folds {zones}",
                          f"TRAIN role of folds {zones} (val + buffer removed)", pred)
    raise ValueError(f"unknown fit id {fit_id!r}")


def verify_fold_isolation(engine, vis: Visibility, units: dict, outer_unit=None) -> list[dict]:
    """Explicit pixel tests (PASS / FAIL) for one visibility and the units it predicts."""
    shape = engine.shape
    rows = []

    def row(check, ok, detail=""):
        rows.append({"fit_id": vis.fit_id, "check": check, "status": "PASS" if ok else "FAIL", "detail": detail})

    row("visible and hidden labels are disjoint", not (vis.visible & vis.hidden).any())
    for key in vis.predicts:
        vu = units.get(key)
        if vu is None:
            row(f"predicted unit {key} exists", False, "unit absent")
            continue
        t = vu.full(vu.eval_truth(), shape)
        n = int((t & vis.visible).sum())
        row(f"{key}: no scored truth pixel is visible to the fit", n == 0, f"{n} visible truth px")
        if vu.scenario in ("V3", "V2"):
            ev = vu.full(vu.unit.ev, shape)
            n2 = int((vis.region & ev).sum())
            row(f"{key}: training region does not touch the evaluation zone", n2 == 0, f"{n2} px overlap")
        kn = int((vu.known & vu.eval_truth()).sum())
        row(f"{key}: permitted known context excludes the unit truth", kn == 0, f"{kn} px")
    if outer_unit is not None:
        t = outer_unit.full(outer_unit.eval_truth(), shape)
        n = int((t & vis.visible).sum())
        row(f"nested: outer unit {outer_unit.key} truth invisible to the inner fit", n == 0, f"{n} px")
    return rows


def assert_isolated(rows: list[dict]) -> None:
    bad = [r for r in rows if r["status"] != "PASS"]
    if bad:
        raise LeakageError(f"fold isolation failed: {bad[:3]}")


# =============================================================================== inner folds
def block_ids(shape, block_km: float) -> np.ndarray:
    """int32 id of the ``block_km`` square block of every pixel (row-major)."""
    h, w = (int(x) for x in shape)
    b = max(int(round(float(block_km) / 0.1)), 1)
    nbc = (w + b - 1) // b
    return ((np.arange(h, dtype=np.int32)[:, None] // b) * nbc + (np.arange(w, dtype=np.int32)[None, :] // b))


def make_inner_folds(region: np.ndarray, visible: np.ndarray, block_km: float, n_inner: int, seed: int
                     ) -> np.ndarray:
    """int8 inner-fold id per pixel of ``region`` (-1 elsewhere): spatial blocks assigned greedily so every inner
    fold gets a similar number of visible positive pixels (ties broken by a seeded shuffle)."""
    region = np.asarray(region, bool)
    h, w = region.shape
    bid = block_ids((h, w), block_km)
    nb = int(bid.max()) + 1
    pos = np.bincount(bid[region & visible], minlength=nb)
    area = np.bincount(bid[region], minlength=nb)
    blocks = np.flatnonzero(area > 0)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(blocks)
    blocks = blocks[np.argsort(-pos[blocks], kind="stable")]
    load_pos, load_area = np.zeros(n_inner), np.zeros(n_inner)
    lut = np.full(nb, -1, np.int8)
    for x in blocks:
        f = int(np.lexsort((load_area, load_pos))[0])
        lut[x] = f
        load_pos[f] += pos[x]
        load_area[f] += area[x]
    out = np.where(region, lut[bid], -1).astype(np.int8)
    return out


def inner_buffer_mask(inner: np.ndarray, buffer_km: float) -> np.ndarray:
    """Pixels within ``buffer_km`` of a DIFFERENT inner fold (excluded from inner validation rows)."""
    out = np.zeros(inner.shape, bool)
    if buffer_km <= 0:
        return out
    for f in np.unique(inner[inner >= 0]):
        other = (inner >= 0) & (inner != f)
        if not other.any():
            continue
        d = ndimage.distance_transform_edt(~other, sampling=(0.1, 0.1))
        out |= (inner == f) & (d <= buffer_km)
    return out


# =============================================================================== tables
def unit_table(units: list[ValidationUnit]) -> pd.DataFrame:
    from ..m7.workspace import mask_hash

    rows = []
    for vu in units:
        t = vu.eval_truth()
        ids = np.unique(vu.sysmap[t & (vu.sysmap >= 0)]) if t.any() else []
        rows.append({"unit": vu.key, "scenario": vu.scenario, "zone": vu.zone, "task": vu.unit.task,
                     "base": f"{vu.base.task}:{vu.base.zone}", "row0": vu.sl[0].start, "row1": vu.sl[0].stop,
                     "col0": vu.sl[1].start, "col1": vu.sl[1].stop, "eval_px": int(vu.unit.ev.sum()),
                     "scored_truth_px": int(t.sum()), "n_systems": len(ids), "known_px": int(vu.known.sum()),
                     "truth_hash": mask_hash(t), "known_hash": mask_hash(vu.known)})
    return pd.DataFrame(rows)


def visibility_row(vis: Visibility) -> dict:
    s = vis.spec()
    return {"fit_id": vis.fit_id, "scenario": vis.scenario, "outer": vis.outer, "inner": vis.inner,
            "visible_px": int(vis.visible.sum()), "hidden_px": int(vis.hidden.sum()),
            "region_px": int(vis.region.sum()), "visible_rule": vis.visible_rule, "region_rule": vis.region_rule,
            "visible_hash": s.visible_hash, "region_hash": s.region_hash, "predicts": ";".join(vis.predicts)}
