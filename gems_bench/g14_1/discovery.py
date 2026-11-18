"""Corrected validation protocols, conditional discovery supervision and separated targets (Phases D5, E, F).

Conditional discovery (M2) in one line:  Y_target = Y_visible \\ Y_context.

* The unit's visible fault systems inside its training geography are grouped (orientation-aware, ``g14.splits.
  fault_system_groups``) and partitioned, per epoch, into CONTEXT systems (allowed as input: distance, density,
  tip, strike channels) and TARGET systems (supervised, hidden from every input channel).
* Faults visible to the unit outside its training geography (other permitted geography) are context only.
* Inner-validation faults are never context during training and are the targets of the inner evaluation, where the
  context is every training-geography fault (what the official test looks like: all known faults visible, new
  faults hidden).
* Pixel-exact context faults carry no loss (the official scorer masks them); pixels around them are ordinary
  candidates, so painting the shoulders of known faults is penalised like any other false positive.
* Reliable negatives: farther than ``reliable_negative_m`` from every visible fault (context and target); the
  ``ignore_ring_m`` around target systems has no negative supervision (positive-unlabelled caution).

Separated targets (S03): training targets come only from labels inside the (embargoed) training region, inner
targets only from labels inside the inner region; a label-free ``boundary_m`` embargo removes training pixels next
to inner blocks, so no inner label shapes a training distance transform, ignore ring or soft target.

All distances are in metres and converted with the real pixel size.

UNTESTED: written without execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

PROTOCOL_TASKS = {"geo_only": ("geo",), "disc_only": ("disc_official",), "geo_and_disc": ("geo", "disc_official")}


# =============================================================================== protocol consistency
def check_protocol(protocol: str, primary_task: str, available_maps: dict[str, int] | None = None,
                   available_checkpoints: dict[str, int] | None = None, submission: bool = False) -> pd.DataFrame:
    """Every (protocol, task, primary task, artifacts) inconsistency as a row with an actionable fix.

    ``available_maps`` / ``available_checkpoints``: {task: number of units with a map / checkpoint}."""
    rows = []
    tasks = PROTOCOL_TASKS[protocol]

    def add(level, msg, fix):
        rows.append({"level": level, "issue": msg, "fix": fix})

    if primary_task not in tasks:
        add("ERROR", f"primary task {primary_task} not produced by protocol {protocol}",
            "use protocol 'geo_and_disc' (or 'disc_only') for a discovery-calibrated submission")
    am = available_maps or {}
    ac = available_checkpoints or {}
    for t in tasks:
        if am and am.get(t, 0) == 0:
            add("WARN", f"protocol {protocol} expects {t} maps; none found",
                f"train / predict the {t} units (stage R10/R11) or restore them from Drive")
    if submission and am.get(primary_task, 0) == 0:
        if primary_task == "disc_official" and am.get("geo", 0):
            add("PROVISIONAL", "no discovery maps: only a geographic calibration is possible",
                "label the submission PROVISIONAL_GEO_CALIBRATION; produce discovery maps for a final candidate")
        else:
            add("ERROR", f"no {primary_task} maps for calibration", "run held-out prediction for the primary task")
    if ac and ac.get("final", 0) == 0 and submission:
        add("INFO", "no final-unit checkpoint", "a full map must already exist, or train the final unit (mode TRAIN)")
    if not rows:
        add("OK", "protocol, tasks and artifacts are consistent", "")
    return pd.DataFrame(rows)


# =============================================================================== distances
def px(metres: float, pixel_m: float) -> int:
    return int(round(metres / pixel_m))


def dist_px(mask: np.ndarray) -> np.ndarray:
    return ndimage.distance_transform_edt(~mask).astype(np.float32) if mask.any() else \
        np.full(mask.shape, 1e6, np.float32)


# =============================================================================== separated targets
def embargoed_train(unit, boundary_m: float, pixel_m: float) -> np.ndarray:
    """Training region minus a label-free band of ``boundary_m`` around the inner-validation region."""
    b = px(boundary_m, pixel_m)
    if b <= 0 or not unit.inner_val.any():
        return unit.train.copy()
    near = ndimage.distance_transform_edt(~unit.inner_val) <= b
    return unit.train & ~near


def separate_targets(unit, target_labels: np.ndarray, kernel_radius_px: float, boundary_m: float, pixel_m: float):
    """(train_region, train_targets, inner_targets). ``target_labels``: the labels this unit may supervise
    (all visible for M1 / M3, target systems for M2). Inner targets always come from the unit's inner faults."""
    from ..g14.multiscale import unit_targets

    region = embargoed_train(unit, boundary_m, pixel_m)
    tr = unit_targets(target_labels & region, kernel_radius_px)
    iv = unit_targets(unit.labels & unit.inner_val, kernel_radius_px)
    return region, tr, iv


def target_leak_check(unit, region: np.ndarray, tr_targets: dict, ring_px: float) -> dict:
    """Assert-style audit: no inner label within ``ring_px`` of a supervised training pixel."""
    iv = unit.labels & unit.inner_val
    d = dist_px(iv)
    return {"unit": unit.key, "train_px_within_ring_of_inner_fault": int((region & (d <= ring_px)).sum()),
            "inner_label_px_in_train_targets": int(((tr_targets["trace"] > 0) & unit.inner_val).sum())}


# =============================================================================== conditional discovery
@dataclass
class ConditionalDiscoveryUnit:
    """Wraps a ``g14.splits.TrainUnit`` with a per-epoch context / target partition of its training systems."""
    base: object
    sysmap: np.ndarray                     # system id per visible training label pixel, -1 elsewhere
    systems: np.ndarray                    # system ids inside the training geography
    system_px: dict
    outside_context: np.ndarray            # permitted faults outside train + inner (context only, never targets)
    context_fraction: float = 0.5
    stratify_by_length: bool = True
    history: list = field(default_factory=list)

    @classmethod
    def build(cls, unit, link_m: float, angle_deg: float, pixel_m: float, context_fraction: float = 0.5):
        from ..g14.splits import fault_system_groups

        vis_train = unit.labels & unit.train
        sysmap, _ = fault_system_groups(vis_train, link_m, angle_deg, pixel_m)
        ids, counts = np.unique(sysmap[sysmap >= 0], return_counts=True)
        outside = unit.permitted_known & ~(unit.train | unit.inner_val)
        return cls(unit, sysmap, ids, dict(zip(ids.tolist(), counts.tolist())), outside, context_fraction)

    @property
    def key(self) -> str:
        return f"cond_{self.base.key}"

    def draw(self, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """(context map, target map, record). Systems are shuffled within length strata so both sides keep a
        similar length distribution; at least one system goes to each side."""
        rng = np.random.default_rng(seed)
        ids = self.systems.copy()
        if ids.size < 2:
            raise ValueError(f"{self.base.key}: fewer than two fault systems in the training geography")
        if self.stratify_by_length:
            sizes = np.array([self.system_px[int(i)] for i in ids])
            order = np.argsort(sizes)
            ids = ids[order]
            ctx_sel = np.zeros(ids.size, bool)
            for s in range(0, ids.size, 4):                      # strata of 4 consecutive sizes
                blk = np.arange(s, min(s + 4, ids.size))
                k = int(round(self.context_fraction * blk.size))
                ctx_sel[rng.choice(blk, k, replace=False)] = True
        else:
            ctx_sel = rng.random(ids.size) < self.context_fraction
        if ctx_sel.all():
            ctx_sel[rng.integers(ids.size)] = False
        if not ctx_sel.any():
            ctx_sel[rng.integers(ids.size)] = True
        ctx_ids, tgt_ids = ids[ctx_sel], ids[~ctx_sel]
        ctx = np.isin(self.sysmap, ctx_ids) | self.outside_context
        tgt = np.isin(self.sysmap, tgt_ids)
        rec = {"seed": seed, "n_context_systems": int(ctx_ids.size), "n_target_systems": int(tgt_ids.size),
               "context_px": int(ctx.sum()), "target_px": int(tgt.sum()),
               "overlap_px": int((ctx & tgt).sum())}
        assert rec["overlap_px"] == 0, "context and target systems overlap"
        self.history.append(rec)
        return ctx, tgt, rec

    def eval_context(self) -> np.ndarray:
        """Context for inner / held-out evaluation: every visible fault of the training geography + outside
        permitted faults (inner faults are hidden: they are the inner targets)."""
        return (self.base.labels & self.base.train) | self.outside_context


def supervision_masks(region: np.ndarray, valid: np.ndarray, context: np.ndarray, target: np.ndarray,
                      ignore: np.ndarray, pixel_m: float, ignore_ring_m: float = 300.0,
                      reliable_negative_m: float = 1000.0, policy: str = "distance_masked",
                      neg_weight: float = 1.0) -> dict:
    """Full-grid supervision masks of one draw.

    ``policy``:
      all_unlabeled     every non-target pixel in the region is a negative (weight ``neg_weight``)
      distance_masked   no negative within ``ignore_ring_m`` of a target system (default)
      reduced_weight    distance_masked, negatives weighted ``neg_weight`` (< 1)
      pu                distance_masked; the loss treats negatives as unlabelled (nnPU)
      hard_negative     distance_masked; the sampler over-samples model-selected hard windows
    Pixel-exact context faults never carry loss."""
    r_ign, r_rel = ignore_ring_m / pixel_m, reliable_negative_m / pixel_m
    dt_t = dist_px(target)
    dt_all = dist_px(target | context)
    loss = region & valid & ~context
    pos = target & loss
    if policy == "all_unlabeled":
        neg = loss & ~target & ~ignore
    else:
        neg = loss & ~target & ~ignore & (dt_t > r_ign)
    w = np.where(neg, neg_weight if policy == "reduced_weight" else 1.0, 0.0).astype(np.float32)
    return {"loss_mask": loss.astype(np.float32), "positive": pos.astype(np.float32), "neg_weight": w,
            "reliable_neg": (neg & (dt_all > r_rel)).astype(np.float32), "dist_target_px": dt_t, "policy": policy}


def describe_draws(cu: ConditionalDiscoveryUnit) -> pd.DataFrame:
    return pd.DataFrame(cu.history)


# =============================================================================== corrected unit families
def geophysics_only(unit):
    """M1: same geography, no known-fault context at all (permitted map emptied), training labels = training region."""
    from ..g14.splits import TrainUnit

    return TrainUnit(unit.task, unit.zone, unit.train, unit.inner_val, unit.labels & unit.train, unit.ignore,
                     np.zeros_like(unit.permitted_known), unit.heldout)


def with_context(unit, context: np.ndarray, labels: np.ndarray | None = None):
    """A copy of ``unit`` whose permitted known map is ``context`` (and optionally other supervised labels)."""
    from ..g14.splits import TrainUnit

    return TrainUnit(unit.task, unit.zone, unit.train, unit.inner_val, unit.labels if labels is None else labels,
                     unit.ignore, context, unit.heldout)


def protocol_units(system, protocol: str) -> list:
    """G14 UnitSystem units for a corrected protocol (geo and / or discovery), in a stable order."""
    out = []
    if "geo" in PROTOCOL_TASKS[protocol]:
        out += [system.geo_unit(k) for k in system.cv_zones]
    if "disc_official" in PROTOCOL_TASKS[protocol]:
        out += [system.disc_unit(d) for d in range(system.n_disc)]
    return out


def corrected_final_context(data, exclude_test: bool = False, test_exclusion: np.ndarray | None = None) -> np.ndarray:
    """Context for the submission model: EVERY mapped fault (a test-time input). G14's final unit dropped the locked
    TEST fold (S10); that is only appropriate for a locked-TEST evaluation map, never for the official file."""
    if exclude_test and test_exclusion is not None:
        return data.labels & ~test_exclusion
    return data.labels.copy()


def discovery_split_audit(system, protocol: str) -> pd.DataFrame:
    """Per unit: geometry, overlaps and the shortcut status of the ORIGINAL G14 construction."""
    from .leakage import target_context_overlap

    rows = []
    for u in protocol_units(system, protocol):
        a = system.audit([u]).iloc[0].to_dict()
        rows.append({**a, **target_context_overlap(u)})
    return pd.DataFrame(rows)
