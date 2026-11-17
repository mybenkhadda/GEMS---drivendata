"""Spatially rigorous validation design (section 18), built on ``gems_bench.folds``.

Partitions (all geographically distinct, separated by an embargo buffer)::

    TEST      one geographic fold, untouched until the final once-only evaluation (section 37)
    CV k      the other folds; for fold k the model trains on CV-region pixels outside fold k and its buffer
    selection cross-fitted on the CV folds: a choice for fold k is made on the other CV folds only
    stacking  fusion / gating models are trained on out-of-fold predictions of the other CV folds
    DISC d    whole mapped-fault *systems* (linked within 1 km) withheld from supervision inside the CV region;
              scored with retained faults masked pixel-exactly and their predictions ignored
              (``disc_official``, the hidden-test analogue documented in docs/GEMS_OFFICIAL_METRIC_AUDIT.md)

The withheld systems of a discovery split are removed from positive supervision and their ``ignore`` ring gets
weight 0 (they are *not* turned into negatives). No label-derived feature exists in G13, so no feature can leak them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from ..folds import DROLE, ROLE, make_discovery_splits, make_geo_folds
from ..gems3.official import _unit
from ..metric import OFFICIAL_EXCLUDE_SUPPORTS_TP
from .cache import atomic_save_npy, atomic_save_table, atomic_write_json, combine


@dataclass
class Splits:
    fold_map: np.ndarray  # int16, -1 invalid
    roles: np.ndarray  # int8 (n_folds, h, w) gems_bench.folds.ROLE codes
    test_fold: int
    disc_roles: np.ndarray  # int8 (n_splits, h, w) gems_bench.folds.DROLE codes
    sysmap: np.ndarray  # int32 fault-system id per label pixel, -1 elsewhere
    systems: pd.DataFrame
    valid: np.ndarray
    labels: np.ndarray
    params: dict

    # ------------------------------------------------------------------ geography
    @property
    def n_folds(self) -> int:
        return int(self.roles.shape[0])

    @property
    def cv_folds(self) -> list[int]:
        return [k for k in range(self.n_folds) if k != self.test_fold]

    def val(self, k: int) -> np.ndarray:
        return self.roles[k] == ROLE["val"]

    def buffer(self, k: int) -> np.ndarray:
        return self.roles[k] == ROLE["buffer"]

    def test(self) -> np.ndarray:
        return self.val(self.test_fold)

    def test_exclusion(self) -> np.ndarray:
        """TEST pixels and their embargo buffer: never used for training, selection or stacking."""
        return self.val(self.test_fold) | self.buffer(self.test_fold)

    def cv_region(self) -> np.ndarray:
        return self.valid & ~self.test_exclusion()

    def train(self, k: int) -> np.ndarray:
        """Training pixels for CV fold ``k``: CV region minus fold k and its buffer."""
        r = self.roles[k]
        return self.cv_region() & ((r == ROLE["train"]) | (r == ROLE["inner_val"]) | (r == ROLE["inner_buffer"]))

    def train_final(self) -> np.ndarray:
        """Training pixels for the once-only TEST evaluation and for the submission model."""
        return self.cv_region()

    # ------------------------------------------------------------------ discovery
    @property
    def n_splits(self) -> int:
        return int(self.disc_roles.shape[0])

    def withheld(self, d: int) -> np.ndarray:
        return self.disc_roles[d] == DROLE["withheld"]

    def ignore(self, d: int) -> np.ndarray:
        return self.disc_roles[d] >= DROLE["ignore"]

    def eval_zone(self, d: int) -> np.ndarray:
        return (self.disc_roles[d] >= DROLE["eval_zone"]) & self.cv_region()

    def disc_train(self, d: int) -> np.ndarray:
        return self.cv_region()

    def disc_labels(self, d: int) -> np.ndarray:
        """Supervision visible to a model of discovery split ``d`` (withheld systems removed)."""
        return self.labels & ~self.withheld(d)

    # ------------------------------------------------------------------ evaluation units
    def units(self, tasks=("geo", "disc_official")) -> list:
        """``gems3.official.Unit`` objects for CV folds and discovery splits (official masking conventions)."""
        out = []
        for task in tasks:
            if task == "geo":
                for k in self.cv_folds:
                    out.append(_unit("geo", k, self.valid & self.val(k), self.labels, None, True, None))
            elif task == "disc_official":
                for d in range(self.n_splits):
                    G = self.withheld(d)
                    retained = self.labels & ~G
                    out.append(
                        _unit("disc_official", d, self.eval_zone(d), G, retained, OFFICIAL_EXCLUDE_SUPPORTS_TP, retained)
                    )
            else:
                raise KeyError(task)
        return out

    def test_unit(self):
        return _unit("geo_test", self.test_fold, self.valid & self.test(), self.labels, None, True, None)

    # ------------------------------------------------------------------ io
    def save(self, root: Path) -> list[Path]:
        root = Path(root)
        outs = [
            atomic_save_npy(root / "fold_map.npy", self.fold_map),
            atomic_save_npy(root / "roles.npy", self.roles),
            atomic_save_npy(root / "disc_roles.npy", self.disc_roles),
            atomic_save_npy(root / "sysmap.npy", self.sysmap),
            atomic_save_table(root / "systems.csv", self.systems),
            atomic_write_json(root / "params.json", {**self.params, "test_fold": self.test_fold}),
        ]
        return outs

    @classmethod
    def load(cls, root: Path, valid: np.ndarray, labels: np.ndarray) -> "Splits":
        import json

        root = Path(root)
        p = json.loads((root / "params.json").read_text())
        return cls(
            np.load(root / "fold_map.npy"),
            np.load(root / "roles.npy"),
            int(p["test_fold"]),
            np.load(root / "disc_roles.npy"),
            np.load(root / "sysmap.npy"),
            pd.read_csv(root / "systems.csv"),
            valid,
            labels,
            p,
        )


def build_splits(data, cfg) -> Splits:
    grid = data.grid
    geo = make_geo_folds(
        data.valid, data.labels, grid, n_folds=cfg.n_geo_folds, block_km=cfg.block_km, buffer_km=cfg.buffer_km, seed=cfg.seed
    )
    # TEST = the fold whose fault density is closest to the median fold density (representative, deterministic)
    dens = [
        float((data.labels & (geo.fold_map == k)).sum() / max((geo.fold_map == k).sum(), 1)) for k in range(cfg.n_geo_folds)
    ]
    test_fold = int(np.argsort(np.abs(np.array(dens) - np.median(dens)))[0])
    test_ex = (geo.roles[test_fold] == ROLE["val"]) | (geo.roles[test_fold] == ROLE["buffer"])
    cv = data.valid & ~test_ex
    disc = make_discovery_splits(
        data.labels & cv,
        cv,
        grid,
        n_splits=cfg.discovery_splits,
        link_km=cfg.discovery_link_km,
        frac_km=cfg.discovery_frac_km,
        ignore_km=cfg.discovery_ignore_km,
        eval_km=cfg.discovery_eval_km,
        seed=cfg.seed,
    )
    params = {
        "geo": geo.params,
        "discovery": disc.params,
        "fold_density": dens,
        "hash": combine(
            cfg.fingerprint("seed", "n_geo_folds", "block_km", "buffer_km"), int(data.labels.sum()), int(data.valid.sum())
        ),
    }
    return Splits(
        geo.fold_map, geo.roles, test_fold, disc.roles, disc.system_map, disc.systems, data.valid, data.labels, params
    )


# =============================================================================== audits
def audit(sp: Splits, pixel_km: float = 0.1) -> pd.DataFrame:
    """Disjointness, embargo distance, positive counts and fault components crossing fold boundaries."""
    rows = []
    test = sp.test()
    for k in sp.cv_folds:
        tr, va = sp.train(k), sp.val(k)
        d_val = ndimage.distance_transform_edt(~va) * pixel_km
        d_test = ndimage.distance_transform_edt(~test) * pixel_km
        comp, n = ndimage.label(sp.labels, structure=np.ones((3, 3), bool))
        in_tr = np.unique(comp[sp.labels & tr])
        in_va = np.unique(comp[sp.labels & va])
        crossing = np.intersect1d(in_tr[in_tr > 0], in_va[in_va > 0])
        rows.append(
            {
                "partition": f"cv{k}",
                "train_px": int(tr.sum()),
                "val_px": int(va.sum()),
                "train_pos": int((tr & sp.labels).sum()),
                "val_pos": int((va & sp.labels).sum()),
                "train_val_overlap": int((tr & va).sum()),
                "train_test_overlap": int((tr & test).sum()),
                "min_train_to_val_km": float(d_val[tr].min()) if tr.any() else np.nan,
                "min_train_to_test_km": float(d_test[tr].min()) if tr.any() and test.any() else np.nan,
                "fault_components_crossing_train_val": int(len(crossing)),
                "note": "crossing components are split by the embargo buffer; buffer pixels are never trained",
            }
        )
    for d in range(sp.n_splits):
        wh, ev = sp.withheld(d), sp.eval_zone(d)
        rows.append(
            {
                "partition": f"disc{d}",
                "withheld_px": int(wh.sum()),
                "eval_px": int(ev.sum()),
                "retained_in_eval_px": int((sp.labels & ~wh & ev).sum()),
                "withheld_in_test_px": int((wh & sp.test_exclusion()).sum()),
                "supervised_withheld_px": int((sp.disc_labels(d) & wh).sum()),
                "n_systems": int((sp.systems.get("discovery_split", pd.Series(dtype=int)) == d).sum()),
            }
        )
    rows.append(
        {"partition": "test", "val_px": int(test.sum()), "val_pos": int((test & sp.labels).sum()), "test_fold": sp.test_fold}
    )
    return pd.DataFrame(rows)


def assert_valid(sp: Splits, min_pos: int = 20) -> None:
    """Hard checks (section 35, spatial validation tests). Raises AssertionError."""
    test = sp.test()
    for k in sp.cv_folds:
        tr, va = sp.train(k), sp.val(k)
        assert not (tr & va).any(), f"train/val overlap in fold {k}"
        assert not (tr & test).any(), f"train/test overlap in fold {k}"
        assert not (tr & sp.buffer(k)).any(), f"training on the embargo buffer of fold {k}"
        assert (tr & sp.labels).sum() >= min_pos, f"fold {k}: too few training positives"
    for d in range(sp.n_splits):
        assert not (sp.disc_labels(d) & sp.withheld(d)).any(), f"withheld systems supervised in split {d}"
        assert not (sp.eval_zone(d) & sp.test_exclusion()).any(), f"discovery eval zone touches TEST in split {d}"
