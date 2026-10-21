"""Experiment A3: stronger spatial validation designs, saved in the Phase-2 file formats.

Every design is written with ``GeoFolds.save`` / ``DiscoverySplits.save``, so ``GeoFolds.load`` /
``DiscoverySplits.load`` and all existing scorers (budget, discovery3, phase4_eval) use them unchanged:
point a context at them with ``with_splits`` / ``with_folds``.

Designs
-------
``system_kfold``       every eligible fault system (pieces linked within 1 km) withheld exactly once over K
                       splits, balanced by trace length and region (complete withheld-system discovery).
``length_stratified``  withhold only systems of one trace-length stratum (e.g. < 2 km): direct test of the
                       hardest, short-fault regime. Systems are small, so splits withhold many of them.
``region_loro``        leave-one-region-out geographic folds (3 x 3 regions) with a buffer >= feature support
                       and seeded inner-validation blocks: generalisation to an unseen geological province.
``hard_subregions``    reporting strata only: blocks where the reference models score worst. Never used for
                       training, feature selection or model selection (they are defined from model errors).

Leakage rule: training-side preprocessing, thresholds and stacking are fitted inside the training zone of
each split; precomputed label-independent rasters are allowed (transductive inputs, documented in
docs/VALIDATION_PROTOCOL.md) provided the buffer covers their support (``buffer_audit``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..folds import DROLE, ROLE, DiscoverySplits, GeoFolds, _within, fault_systems


def _roles_for(withheld_ids, sysmap, labels, valid, grid, ignore_km, eval_km):
    wh = np.isin(sysmap, list(withheld_ids)) & labels
    retained = labels & ~wh
    r = np.zeros(labels.shape, np.int8)
    r[_within(wh, eval_km, grid) & valid] = DROLE["eval_zone"]
    r[_within(wh, ignore_km, grid) & ~retained] = DROLE["ignore"]
    r[wh] = DROLE["withheld"]
    return r


def system_kfold(ctx, *, k: int = 5, link_km: float = 1.0, min_system_km: float = 1.0, max_system_frac: float = 0.05,
                 ignore_km: float = 1.0, eval_km: float = 10.0, seed: int = 2026, length_range=None,
                 name: str | None = None) -> DiscoverySplits:
    """K disjoint withheld-system splits covering every eligible system once (optionally one length stratum)."""
    d = ctx.data
    sysmap, systems = fault_systems(d.labels, d.grid, link_km)
    total = systems.trace_km.sum()
    lo, hi = length_range or (min_system_km, max_system_frac * total)
    elig = systems[(systems.trace_km >= lo) & (systems.trace_km < hi) & (systems.trace_km <= max_system_frac * total)].copy()
    rng = np.random.default_rng(seed)
    elig = elig.assign(rand=rng.random(len(elig))).sort_values(["region", "rand"])
    load, assign = np.zeros(k), {}
    for sid, km in sorted(zip(elig.system_id, elig.trace_km), key=lambda t: -t[1]):   # longest first, least loaded
        j = int(np.argmin(load))
        assign[sid] = j
        load[j] += km
    systems["discovery_split"] = systems.system_id.map(assign).fillna(-1).astype(int)
    roles = np.stack([_roles_for([s for s, j in assign.items() if j == i], sysmap, d.labels, d.valid, d.grid,
                                 ignore_km, eval_km) for i in range(k)])
    params = dict(design="length_stratified" if length_range else "system_kfold", n_splits=k, link_km=link_km,
                  length_range_km=list(length_range) if length_range else [lo, hi], ignore_km=ignore_km,
                  eval_km=eval_km, seed=seed, withheld_km=load.tolist(), total_trace_km=float(total),
                  n_systems_withheld=len(assign))
    return DiscoverySplits(d.grid, sysmap, systems, roles, params)


def length_stratified(ctx, lo_km: float = 0.0, hi_km: float = 2.0, k: int = 3, **kw) -> DiscoverySplits:
    return system_kfold(ctx, k=k, length_range=(lo_km, hi_km), min_system_km=lo_km, **kw)


def region_loro(ctx, *, region_grid=(3, 3), buffer_km: float = 13.0, inner_val_frac: float = 0.15,
                block_km: float = 20.0, min_val_label_px: int = 500, seed: int = 2026) -> GeoFolds:
    """Leave-one-region-out folds (regions with too few labels are merged into training only)."""
    from ..geo_eval import region_map

    d = ctx.data
    reg = region_map(d, region_grid)
    ids = [int(r) for r in np.unique(reg[reg >= 0]) if (d.labels & (reg == r)).sum() >= min_val_label_px]
    rng = np.random.default_rng(seed)
    bpx = int(round(block_km / d.grid.pixel_km))
    H, W = d.shape
    roles = np.full((len(ids), H, W), ROLE["invalid"], np.int8)
    fold_map = np.full((H, W), -1, np.int16)
    rows = []
    for k, r in enumerate(ids):
        val = (reg == r) & d.valid
        buf = _within(val, buffer_km, d.grid) & ~val & d.valid
        rest = d.valid & ~val & ~buf
        ro = roles[k]
        ro[rest] = ROLE["train"]
        # inner validation: random training blocks (+ inner buffer), never adjacent to the held-out region
        bl = [(r0, c0) for r0 in range(0, H, bpx) for c0 in range(0, W, bpx) if rest[r0:r0 + bpx, c0:c0 + bpx].mean() > 0.5]
        pick = rng.permutation(len(bl))[:max(1, int(round(inner_val_frac * len(bl))))]
        iv = np.zeros((H, W), bool)
        for i in pick:
            r0, c0 = bl[i]
            iv[r0:r0 + bpx, c0:c0 + bpx] = True
        iv &= rest
        ib = _within(iv, buffer_km / 2, d.grid) & rest & ~iv
        ro[ib] = ROLE["inner_buffer"]
        ro[iv] = ROLE["inner_val"]
        ro[buf] = ROLE["buffer"]
        ro[val] = ROLE["val"]
        fold_map[val] = k
        rows.append({"fold": k, "region": r, "val_px": int(val.sum()), "val_label_px": int((d.labels & val).sum()),
                     "inner_val_px": int(iv.sum())})
    params = dict(design="region_loro", region_grid=list(region_grid), buffer_km=buffer_km, inner_val_frac=inner_val_frac,
                  block_km=block_km, seed=seed, regions=ids)
    return GeoFolds(d.grid, fold_map, roles, pd.DataFrame(rows), params)


def hard_subregions(ctx, maps: dict[str, np.ndarray], block_km: float = 30.0, worst_q: float = 0.25) -> pd.DataFrame:
    """Blocks in the worst quartile of DTI for *every* reference map (reporting strata only)."""
    from ..geo_eval import block_dti

    tabs = []
    for m, p in maps.items():
        b = block_dti(p, ctx.data, block_km)
        tabs.append(b[["r0", "r1", "c0", "c1", "dti"]].rename(columns={"dti": f"dti_{m}"}))
    t = tabs[0]
    for x in tabs[1:]:
        t = t.merge(x, on=["r0", "r1", "c0", "c1"])
    cols = [c for c in t.columns if c.startswith("dti_")]
    t["hard"] = np.all([t[c] <= t[c].quantile(worst_q) for c in cols], axis=0)
    return t


def buffer_audit(ctx, channels: list[str], buffer_km: float) -> pd.DataFrame:
    """Feature support vs fold buffer (leakage through spatial smoothing of inputs near validation labels)."""
    rows = []
    for c in channels:
        s = ctx.store.specs.get(c)
        sup = float(getattr(s, "support_km", np.nan)) if s else np.nan
        rows.append({"feature": c, "support_km": sup, "buffer_km": buffer_km, "ok": bool(sup <= buffer_km) if sup == sup else None})
    return pd.DataFrame(rows)


def save_design(obj, root: Path, name: str) -> Path:
    return obj.save(Path(root) / name)


def with_splits(ctx, directory: Path):
    """Context copy scored on another discovery-split design (``DiscoverySplits.load``)."""
    from dataclasses import replace

    return replace(ctx, splits=DiscoverySplits.load(Path(directory), ctx.data.grid))


def with_folds(ctx, directory: Path):
    from dataclasses import replace

    return replace(ctx, folds=GeoFolds.load(Path(directory), ctx.data.grid))
