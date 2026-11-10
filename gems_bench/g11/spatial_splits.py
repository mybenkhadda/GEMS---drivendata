"""Geographically disjoint train / validation / stack / test partitions with embargo, plus leakage checks.

Blocks of ``block_km`` are grouped with every mapped fault system they touch (8-connected components of the
fault mask dilated by the embargo) so one fault system never spans incompatible splits; groups are assigned
greedily (largest first) to the split with the largest remaining quota (deterministic for a seed). Pixels within
``embargo_m`` of a different split are EMBARGO (never trained or scored).

The embargo is chosen BEFORE any test result from the configured components:
max(feature receptive field, DEM window halo, model patch context, candidate link radius, autocorrelation range).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

SPLITS = ("train", "validation", "stack", "test")
CODES = {"none": 0, "train": 1, "validation": 2, "stack": 3, "test": 4, "embargo": 5}


class LeakageError(RuntimeError):
    pass


def embargo_width_m(cfg: dict) -> tuple[float, dict]:
    ss = cfg["spatial_splits"]
    if ss.get("embargo_m"):
        return float(ss["embargo_m"]), {"configured": float(ss["embargo_m"])}
    fe = cfg["feature_engineering"]
    parts = {
        "feature_receptive_field_m": 4.0 * max(fe["geophysics"]["scales_m"]) / 2.0,
        "dem_halo_m": 2.0 * max(cfg["dem"]["context_scales_m"]),
        "patch_context_m": cfg["sampling"]["patch_px"] * 100.0 / 2.0,
        "candidate_link_radius_m": float(cfg["candidate_generation"]["link_radius_m"]),
        "autocorrelation_range_m": float(ss["autocorrelation_range_m"]),
    }
    return max(parts.values()), parts


@dataclass
class SplitResult:
    code: np.ndarray            # uint8 per pixel (CODES)
    blocks: pd.DataFrame        # block / group / split table
    embargo_m: float
    embargo_parts: dict
    groups: np.ndarray          # int32 group id per pixel (-1 outside)

    def mask(self, name: str) -> np.ndarray:
        return self.code == CODES[name]


def make_splits(valid: np.ndarray, faults: np.ndarray, grid, cfg: dict) -> SplitResult:
    ss = cfg["spatial_splits"]
    px = grid.res[0]
    B = max(1, int(round(ss["block_km"] * 1000 / px)))
    emb, parts = embargo_width_m(cfg)
    emb_px = int(np.ceil(emb / px))
    h, w = valid.shape
    nbc = -(-w // B)
    ri, ci = np.indices(valid.shape, dtype=np.int32)
    block = (ri // B) * nbc + ci // B
    del ri, ci
    # fault systems: components of the fault mask dilated by the embargo; blocks they touch are merged
    sysmask = ndimage.binary_dilation(faults, iterations=max(emb_px // 2, 1)) if faults.any() else faults
    lab, n = ndimage.label(sysmask, structure=np.ones((3, 3), bool))
    nb = int(block.max()) + 1
    parent = np.arange(nb)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if n:
        pairs = pd.DataFrame({"s": lab[sysmask & valid], "b": block[sysmask & valid]}).drop_duplicates()
        for _, g in pairs.groupby("s"):
            bs = g.b.to_numpy()
            for b2 in bs[1:]:
                ra, rb = find(bs[0]), find(b2)
                if ra != rb:
                    parent[rb] = ra
    root = np.array([find(b) for b in range(nb)])
    vcount = np.bincount(block[valid], minlength=nb)
    gsize = pd.Series(vcount).groupby(root).sum()
    gsize = gsize[gsize > 0]
    rng = np.random.default_rng(int(ss["seed"]))
    order = gsize.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).sort_values(ascending=False, kind="stable")
    total = float(gsize.sum())
    quota = {k: ss["fractions"][k] * total for k in SPLITS}
    got = {k: 0.0 for k in SPLITS}
    assign = {}
    for g, sz in order.items():
        k = max(SPLITS, key=lambda s: (quota[s] - got[s]) / max(quota[s], 1))
        assign[g] = k
        got[k] += sz
    gsplit = np.array([CODES[assign.get(root[b], "train")] if vcount[b] else 0 for b in range(nb)], np.uint8)
    code = np.where(valid, gsplit[block], 0).astype(np.uint8)
    # embargo: pixels within emb of a pixel of another split
    emb_mask = np.zeros(valid.shape, bool)
    for k in SPLITS:
        m = code == CODES[k]
        if m.any():
            near_other = ndimage.distance_transform_edt(~((code != CODES[k]) & (code > 0) & (code != CODES["embargo"])),
                                                        sampling=grid.res[::-1]) <= emb
            emb_mask |= m & near_other
    code[emb_mask] = CODES["embargo"]
    groups = np.where(valid, root[block], -1).astype(np.int32)
    bt = pd.DataFrame({"block": np.arange(nb), "group": root, "valid_px": vcount,
                       "split": [assign.get(root[b], "none") if vcount[b] else "none" for b in range(nb)],
                       "row0": (np.arange(nb) // nbc) * B, "col0": (np.arange(nb) % nbc) * B})
    return SplitResult(code, bt, emb, parts, groups)


# ------------------------------------------------------------------------------------------- leakage checks
def check_disjoint(res: SplitResult) -> list[str]:
    errs = []
    for k in SPLITS:
        for j in SPLITS:
            if k < j and (res.mask(k) & res.mask(j)).any():
                errs.append(f"{k} and {j} overlap")
    return errs


def check_embargo(res: SplitResult, grid) -> list[str]:
    errs = []
    for k in SPLITS:
        m = res.mask(k)
        others = np.zeros_like(m)
        for j in SPLITS:
            if j != k:
                others |= res.mask(j)
        if m.any() and others.any():
            d = ndimage.distance_transform_edt(~others, sampling=grid.res[::-1])
            dmin = float(d[m].min())
            if dmin < res.embargo_m:
                errs.append(f"{k}: nearest other-split pixel at {dmin:.0f} m < embargo {res.embargo_m:.0f} m")
    return errs


def check_patch_leakage(patch_windows: pd.DataFrame, res: SplitResult) -> list[str]:
    """Each patch (row0, col0, size, split) must contain only pixels of its own split (embargo allowed)."""
    errs = []
    for r in patch_windows.itertuples():
        sub = res.code[r.row0:r.row0 + r.size, r.col0:r.col0 + r.size]
        bad = set(np.unique(sub)) - {0, CODES["embargo"], CODES[r.split]}
        if bad:
            errs.append(f"patch at ({r.row0},{r.col0}) of {r.split} touches split codes {sorted(bad)}")
    return errs


def check_segment_groups(segments: pd.DataFrame) -> list[str]:
    """A fault group id must appear in only one split (``group``, ``split`` columns)."""
    g = segments.groupby("group").split.nunique()
    return [f"group {k} appears in {v} splits" for k, v in g[g > 1].items()]


def check_predictor_names(names, label_derived_names) -> list[str]:
    bad = sorted(set(names) & set(label_derived_names))
    return [f"label-derived feature used as predictor: {b}" for b in bad]


def check_oof_provenance(prediction_meta: dict, split: str) -> list[str]:
    """An array may be treated as out-of-fold for ``split`` only if its provenance says so explicitly."""
    prov = prediction_meta.get("provenance") or {}
    stated = (prov.get("split") or prediction_meta.get("split") or "")
    if prediction_meta.get("alignment_confidence") == "UNLOCATED":
        return [f"{prediction_meta.get('path')}: unlocated prediction cannot be used"]
    if stated.lower() not in ("oof", split.lower(), "test" if split == "test" else "__never__"):
        return [f"{prediction_meta.get('path')}: provenance split '{stated}' is not out-of-fold for {split}"]
    return []


class TrainOnlyNormalizer:
    """Robust (median / IQR) normaliser that can only be fitted on train-split pixels."""

    def __init__(self):
        self.center = self.scale = None
        self.fitted_on = None

    def fit(self, X: np.ndarray, split_codes: np.ndarray):
        if np.any(split_codes != CODES["train"]):
            raise LeakageError("normaliser must be fitted on train-split samples only")
        q25, q50, q75 = np.nanpercentile(X, [25, 50, 75], axis=0)
        self.center, self.scale = q50, np.where(q75 - q25 > 1e-12, q75 - q25, 1.0)
        self.fitted_on = "train"
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.center is None:
            raise LeakageError("normaliser not fitted")
        return (X - self.center) / self.scale


def run_leakage_checks(res: SplitResult, grid, predictor_names=(), label_derived=(), patches=None,
                       segments=None) -> pd.DataFrame:
    rows = []
    for name, errs in (("disjoint", check_disjoint(res)), ("embargo", check_embargo(res, grid)),
                       ("predictors", check_predictor_names(predictor_names, label_derived)),
                       ("patches", check_patch_leakage(patches, res) if patches is not None else []),
                       ("segment_groups", check_segment_groups(segments) if segments is not None else [])):
        rows.append({"check": name, "passed": not errs, "errors": "; ".join(errs[:20]), "n_errors": len(errs)})
    return pd.DataFrame(rows)


def save_splits(res: SplitResult, grid, out_dir, ctx) -> list:
    from .features import write_geotiff
    from .registry import atomic_write_parquet

    outs = [atomic_write_parquet(out_dir / "split_registry.parquet", res.blocks)]
    for k in SPLITS:
        outs.append(write_geotiff(out_dir / f"{'validation' if k == 'validation' else k}_mask.tif",
                                  res.mask(k).astype(np.uint8), grid, [f"{k} split"], nodata=None))
    outs.append(write_geotiff(out_dir / "embargo_mask.tif", res.mask("embargo").astype(np.uint8), grid, ["embargo"],
                              nodata=None))
    outs.append(ctx.write_manifest(out_dir / "split_manifest.json", {
        "embargo_m": res.embargo_m, "embargo_components": res.embargo_parts, "codes": CODES,
        "pixels": {k: int(res.mask(k).sum()) for k in list(SPLITS) + ["embargo"]},
        "block_km": ctx.cfg["spatial_splits"]["block_km"], "fractions": ctx.cfg["spatial_splits"]["fractions"],
        "selected_before_test": True}))
    return outs
