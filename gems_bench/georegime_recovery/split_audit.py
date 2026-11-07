"""Section 05 - reconstruct the exact V1 split masks and verify them against every checkpoint.

The masks come from the checked-out ``gems_bench.georegime.data.geo_masks`` / ``discovery_masks`` with the
buffers stored in the V1 run's config.json; the full model trained on ``bank.valid``. A checkpoint is accepted
only if ``sha256(train_mask.tobytes())[:12]`` equals its ``contract['train_mask_sha']`` - the same
expression V1 used when it wrote the contract. Nothing is replaced: a mismatch stops the milestone.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from .cache import array_sha, atomic_write_csv, atomic_write_geotiff, atomic_write_json, sha256_file
from .contracts import SplitProvenanceError, StageResult, check_schema

BITS = {"train": 0, "zone": 1, "truth": 2, "v1_exclude": 3, "training_visible": 4}


@dataclass
class Unit:
    name: str
    protocol: str              # GEOGRAPHIC | WITHHELD_SYSTEM | FULL
    train: np.ndarray          # V1 training mask (supervision region)
    zone: np.ndarray           # V1 evaluation zone (narrow for discovery)
    truth: np.ndarray          # held mapped faults (scoring truth)
    v1_exclude: np.ndarray     # V1 scorer exclusion (retained labels for discovery, none for geographic)
    labels: np.ndarray         # all mapped labels (bank.labels)

    @property
    def training_visible(self) -> np.ndarray:
        return self.labels & self.train

    @property
    def all_non_held(self) -> np.ndarray:
        return self.labels & ~self.truth

    def exclusion(self, kind: str) -> np.ndarray:
        if kind == "training_visible":
            return self.training_visible
        if kind == "all_non_held":
            return self.all_non_held
        if kind == "v1":
            return self.v1_exclude
        raise ValueError(kind)


def train_mask_sha(mask: np.ndarray) -> str:
    return hashlib.sha256(mask.tobytes()).hexdigest()[:12]


def reconstruct(bank, src_cfg: dict, full_unit: str = "full") -> dict:
    from gems_bench.georegime.data import discovery_masks, geo_masks

    units = {}
    for name, train, zone, truth, exclude in geo_masks(bank.valid, bank.labels, int(src_cfg["geo_buffer_px"])):
        units[name] = Unit(name, "GEOGRAPHIC", train, zone, truth, np.asarray(exclude, bool), bank.labels)
    for name, train, zone, truth, exclude in discovery_masks(bank.valid, bank.labels, int(src_cfg["discovery_buffer_px"])):
        units[name] = Unit(name, "WITHHELD_SYSTEM", train, zone, truth, np.asarray(exclude, bool), bank.labels)
    units[full_unit] = Unit(full_unit, "FULL", bank.valid, bank.valid, bank.labels,
                            np.zeros_like(bank.labels), bank.labels)
    return units


def components(mask: np.ndarray):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), np.uint8))
    return lab, n


def _component_geometry(lab, ids, transform, px=100.0):
    """PCA length / linearity / strike of each labelled component (axial azimuth clockwise from north)."""
    a, c, e, f = transform.a, transform.c, transform.e, transform.f
    out = {}
    objs = ndimage.find_objects(lab)
    for i in ids:
        sl = objs[i - 1]
        if sl is None:
            continue
        rr, cc = np.nonzero(lab[sl] == i)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        E, N = c + a * (cc + 0.5), f + e * (rr + 0.5)
        if len(rr) >= 3:
            X = np.stack([E - E.mean(), N - N.mean()], 1)
            w, v = np.linalg.eigh(np.cov(X.T))
            d = v[:, 1]
            proj = X @ d
            length = float((proj.max() - proj.min() + px) / 1e3)
            lin = float(w[1] / (w[0] + w[1] + 1e-12))
            strike = float(np.degrees(np.arctan2(d[0], d[1])) % 180.0)
        else:
            length, lin, strike = len(rr) * px / 1e3, np.nan, np.nan
        out[i] = {"n_px": int(len(rr)), "length_km": length, "linearity": lin, "strike_deg": strike,
                  "centroid_e": float(E.mean()), "centroid_n": float(N.mean())}
    return out


def audit(run) -> StageResult:
    P, bank = run.paths, run.bank
    units = run.units
    rows, frag, overlap = [], [], []
    labels_lab, _ = components(bank.labels)
    mismatches = []
    manifest = {"source": {"module": "gems_bench/georegime/data.py",
                           "module_sha256": sha256_file(run.code_root / "gems_bench/georegime/data.py"),
                           "geo_buffer_px": run.src_cfg["geo_buffer_px"],
                           "discovery_buffer_px": run.src_cfg["discovery_buffer_px"],
                           "connectivity": "8-connected mapped components (ndimage.label, 3x3)",
                           "discovery_zone": "dilate(held, max(30, 3*buffer)) & valid  (3 km for buffer 8)"},
                "units": {}}
    outs = []
    for name, U in units.items():
        sha = train_mask_sha(U.train)
        want = run.ckpt.get(name, {}).get("train_mask_sha")
        match = want is not None and sha == want
        if not match:
            mismatches.append(f"{name}: reconstructed {sha} vs checkpoint {want}")
        tv = U.training_visible
        buffer_labels = U.labels & ~U.truth & ~U.train
        r = {"unit": name, "protocol": U.protocol, "train_mask_sha_reconstructed": sha,
             "train_mask_sha_checkpoint": want, "match": bool(match), "train_px": int(U.train.sum()),
             "zone_px": int(U.zone.sum()), "truth_px": int(U.truth.sum()), "exclude_px": int(U.v1_exclude.sum()),
             "training_visible_px": int(tv.sum()), "truth_inside_train_px": int((U.truth & U.train).sum()),
             "zone_inside_train_px": int((U.zone & U.train).sum()),
             "unseen_non_held_labels_px": int(buffer_labels.sum()),
             "eligible_px": int(bank.valid.sum())}
        if U.protocol == "WITHHELD_SYSTEM":
            ids = np.unique(labels_lab[U.truth])
            ids = ids[ids > 0]
            d_tv = ndimage.distance_transform_edt(~tv) * 100.0
            mind = ndimage.minimum(d_tv, labels_lab, index=ids)
            geo = _component_geometry(labels_lab, ids, bank.transform)
            for i, m in zip(ids, np.atleast_1d(mind)):
                g = geo.get(int(i), {})
                frag.append({"unit": name, "component": int(i), "n_px": g.get("n_px"),
                             "min_dist_to_train_visible_m": float(m), "length_km": g.get("length_km"),
                             "linearity": g.get("linearity"), "strike_deg": g.get("strike_deg"),
                             "centroid_e": g.get("centroid_e"), "centroid_n": g.get("centroid_n")})
            md = np.atleast_1d(mind)
            r.update(n_held_components=int(len(ids)),
                     held_components_within_1km_of_visible=int((md <= 1000).sum()),
                     held_components_within_3km_of_visible=int((md <= 3000).sum()))
        rows.append(r)
        bits = sum((m.astype(np.uint8) << b) for m, b in ((U.train, 0), (U.zone, 1), (U.truth, 2),
                                                          (U.v1_exclude, 3), (tv, 4)))
        outs.append(atomic_write_geotiff(P.d("01_split_audit", "masks") / f"{name}_masks.tif", bits.astype(np.uint8),
                                         bank.profile, ["bitmask: 1 train, 2 zone, 4 truth, 8 v1_exclude, 16 training_visible"],
                                         nodata=None))
        manifest["units"][name] = {"protocol": U.protocol, "train_mask_sha": sha, "checkpoint_train_mask_sha": want,
                                   "match": bool(match),
                                   "array_sha16": {k: array_sha(getattr(U, k)) for k in ("train", "zone", "truth", "v1_exclude")},
                                   "pixels": {k: int(getattr(U, k).sum()) for k in ("train", "zone", "truth", "v1_exclude")}}
    names = [n for n in units if units[n].protocol != "FULL"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = units[a], units[b]
            for q, x in (("zone_overlap", A.zone & B.zone), ("truth_overlap", A.truth & B.truth),
                         ("truth_a_in_train_b", A.truth & B.train), ("truth_b_in_train_a", B.truth & A.train)):
                overlap.append({"unit_a": a, "unit_b": b, "quantity": q, "pixels": int(x.sum())})
    geo = [units[n] for n in names if units[n].protocol == "GEOGRAPHIC"]
    if geo:
        cover = np.sum([g.zone for g in geo], axis=0)
        overlap.append({"unit_a": "geo*", "unit_b": "", "quantity": "eligible_not_in_any_geo_zone",
                        "pixels": int((bank.valid & (cover == 0)).sum())})
        overlap.append({"unit_a": "geo*", "unit_b": "", "quantity": "eligible_in_multiple_geo_zones",
                        "pixels": int((cover > 1).sum())})
    disc = [units[n] for n in names if units[n].protocol == "WITHHELD_SYSTEM"]
    if disc:
        held_any = np.sum([d.truth for d in disc], axis=0)
        overlap.append({"unit_a": "disc*", "unit_b": "", "quantity": "mapped_px_never_held",
                        "pixels": int((bank.labels & (held_any == 0)).sum())})
    unit_table = pd.DataFrame(rows)
    frag_t = pd.DataFrame(frag)
    ov = pd.DataFrame(overlap)
    check_schema("checkpoint_split_validation", unit_table)
    check_schema("split_overlap", ov)
    check_schema("holdout_fragmentation", frag_t)
    d = P.d("01_split_audit")
    outs += [atomic_write_csv(d / "checkpoint_split_validation.csv", unit_table),
             atomic_write_csv(d / "split_overlap.csv", ov),
             atomic_write_csv(d / "holdout_fragmentation.csv", frag_t),
             atomic_write_json(d / "original_masks_manifest.json", manifest)]
    fr = frag_t
    stats = {"all_match": not mismatches, "mismatches": mismatches,
             "n_held_components": int(len(fr)),
             "share_held_within_1km_of_visible": float((fr.min_dist_to_train_visible_m <= 1000).mean()) if len(fr) else None,
             "share_held_within_3km_of_visible": float((fr.min_dist_to_train_visible_m <= 3000).mean()) if len(fr) else None,
             "median_held_length_km": float(fr.length_km.median()) if len(fr) else None}
    run.split_verified = not mismatches
    res = StageResult("audit_original_splits", "VERIFIED" if not mismatches else "MISMATCH",
                      {"unit_table": unit_table, "overlap": ov, "fragmentation": frag_t}, stats, outs,
                      ["discovery groups are 8-connected raster components balanced by pixel count; they are NOT "
                       "verified to be geologically independent fault systems"])
    if mismatches:
        raise SplitProvenanceError("Reconstructed V1 training masks do not match the checkpoints - the old splits "
                                   "cannot be reproduced; a new split needs retraining (separate milestone):\n  - "
                                   + "\n  - ".join(mismatches))
    return res
