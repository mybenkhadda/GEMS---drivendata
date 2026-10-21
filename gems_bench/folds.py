"""Geographically honest validation.

Two split families that answer different questions and are never mixed:

1. **Geographic holdout** (``GeoFolds``) — the valid footprint is tiled into square blocks,
   blocks are clustered into contiguous groups (k-means on block centroids), and groups are
   assigned to K folds balancing mapped-fault pixels and area. For fold k:

   * ``val``    = fold-k pixels (scored);
   * ``buffer`` = pixels within ``buffer_km`` of val that are not val — never used in training
     (no labels, no CNN patch may touch them, no GBM sample is drawn there);
   * ``train``  = every other valid pixel;
   * ``inner_val`` = one training cluster, separated from inner training by ``inner_buffer_km``,
     used only for early stopping / checkpoint choice (never the scored fold).

   CNN training patches must lie entirely inside ``train`` (checked with an integral image),
   so no training receptive field overlaps the validation area. Label-free input features
   (filters of the rasters) are computed once on the full grid; filters wider than the buffer
   do see validation-area *inputs* (never labels) — ``features.leakage_audit`` lists them.

2. **Pseudo-discovery holdout** (``DiscoverySplits``) — complete fault systems (mapped pieces
   linked within ``system_link_km``) are withheld from positive supervision. Their pixels plus
   an ``ignore_km`` ring get weight 0 (they are *not* turned into negatives). Input rasters stay
   available everywhere. The model is scored on rediscovering the withheld systems.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from gems_eda.io import Grid, write_geotiff
from gems_eda.morphology import region_ids, region_name

ROLE = {"invalid": -1, "train": 0, "val": 1, "buffer": 2, "inner_val": 3, "inner_buffer": 4}


def _within(mask: np.ndarray, km: float, grid: Grid) -> np.ndarray:
    """Pixels within ``km`` of ``mask`` (including mask)."""
    if km <= 0:
        return mask.copy()
    d = ndimage.distance_transform_edt(~mask, sampling=(grid.pixel_km_y, grid.pixel_km_x))
    return d <= km


def skeleton_px(labels: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    return skeletonize(labels)


# ---------------------------------------------------------------------------
# geographic folds
# ---------------------------------------------------------------------------
@dataclass
class GeoFolds:
    grid: Grid
    fold_map: np.ndarray               # int16, -1 outside valid
    roles: np.ndarray                  # int8 (K, H, W) with ROLE codes
    blocks: pd.DataFrame
    params: dict
    directory: Path | None = None

    @property
    def n_folds(self) -> int:
        return self.roles.shape[0]

    def val(self, k: int) -> np.ndarray:
        return self.roles[k] == ROLE["val"]

    def train(self, k: int, include_inner_val: bool = True) -> np.ndarray:
        r = self.roles[k]
        m = r == ROLE["train"]
        if include_inner_val:
            m |= (r == ROLE["inner_val"]) | (r == ROLE["inner_buffer"])
        return m

    def inner_train(self, k: int) -> np.ndarray:
        return self.roles[k] == ROLE["train"]

    def inner_val(self, k: int) -> np.ndarray:
        return self.roles[k] == ROLE["inner_val"]

    def buffer(self, k: int) -> np.ndarray:
        return self.roles[k] == ROLE["buffer"]

    # -- persistence ----------------------------------------------------
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        layers = {"fold_id": np.where(self.fold_map >= 0, self.fold_map, np.nan)}
        for k in range(self.n_folds):
            layers[f"role_fold{k}"] = np.where(self.roles[k] >= 0, self.roles[k], np.nan)
        write_geotiff(directory / "fold_map.tif", layers, self.grid)
        self.blocks.to_parquet(directory / "blocks.parquet", index=False)
        (directory / "fold_params.json").write_text(json.dumps({**self.params, "roles": ROLE}, indent=2),
                                                    encoding="utf-8")
        self.directory = directory
        return directory

    @classmethod
    def load(cls, directory: Path, grid: Grid) -> "GeoFolds":
        with rasterio.open(directory / "fold_map.tif") as src:
            a = src.read()
        fold_map = np.where(np.isfinite(a[0]), a[0], -1).astype(np.int16)
        roles = np.where(np.isfinite(a[1:]), a[1:], -1).astype(np.int8)
        params = json.loads((directory / "fold_params.json").read_text(encoding="utf-8"))
        params.pop("roles", None)
        return cls(grid, fold_map, roles, pd.read_parquet(directory / "blocks.parquet"), params, directory)


def make_geo_folds(valid: np.ndarray, labels: np.ndarray, grid: Grid, *, n_folds: int = 3,
                   block_km: float = 20.0, clusters_per_fold: int = 3, buffer_km: float = 13.0,
                   inner_buffer_km: float = 5.0, seed: int = 2026) -> GeoFolds:
    from sklearn.cluster import KMeans

    h, w = valid.shape
    bpx = max(int(round(block_km / grid.pixel_km)), 8)
    rr, cc = np.indices((h, w))
    br, bc = rr // bpx, cc // bpx
    nbc = int(bc.max()) + 1
    bid = (br * nbc + bc).astype(np.int32)
    nb = int(bid.max()) + 1
    v_px = np.bincount(bid[valid], minlength=nb)
    l_px = np.bincount(bid[valid & labels], minlength=nb)
    r_sum = np.bincount(bid[valid], weights=rr[valid], minlength=nb)
    c_sum = np.bincount(bid[valid], weights=cc[valid], minlength=nb)
    keep = np.flatnonzero(v_px > 0)
    blocks = pd.DataFrame({"block": keep, "br": keep // nbc, "bc": keep % nbc, "valid_px": v_px[keep],
                           "label_px": l_px[keep], "r_cent": r_sum[keep] / v_px[keep],
                           "c_cent": c_sum[keep] / v_px[keep]})
    n_clusters = min(n_folds * clusters_per_fold, len(blocks))
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    blocks["cluster"] = km.fit_predict(blocks[["r_cent", "c_cent"]].to_numpy(), sample_weight=blocks.valid_px)

    # greedy balance: clusters in decreasing label count go to the fold with the smallest load
    cl = blocks.groupby("cluster").agg(valid_px=("valid_px", "sum"), label_px=("label_px", "sum")).reset_index()
    cl["load"] = cl.label_px / max(cl.label_px.sum(), 1) + cl.valid_px / cl.valid_px.sum()
    cl = cl.sort_values(["load", "cluster"], ascending=[False, True])
    fold_load = np.zeros(n_folds)
    assign = {}
    for row in cl.itertuples():
        f = int(np.argmin(fold_load))
        assign[row.cluster] = f
        fold_load[f] += row.load
    blocks["fold"] = blocks.cluster.map(assign).astype(int)

    lut = np.full(nb, -1, np.int16)
    lut[blocks.block.to_numpy()] = blocks.fold.to_numpy()
    fold_map = np.where(valid, lut[bid], -1).astype(np.int16)
    cmap = np.full(nb, -1, np.int32)
    cmap[blocks.block.to_numpy()] = blocks.cluster.to_numpy()
    cluster_map = np.where(valid, cmap[bid], -1)

    roles = np.full((n_folds, h, w), ROLE["invalid"], np.int8)
    for k in range(n_folds):
        val = fold_map == k
        near = _within(val, buffer_km, grid) & valid & ~val
        r = roles[k]
        r[valid] = ROLE["train"]
        r[near] = ROLE["buffer"]
        r[val] = ROLE["val"]
        # inner validation: the training cluster with the median label load
        tr_clusters = blocks[blocks.fold != k].groupby("cluster").label_px.sum().sort_values()
        if len(tr_clusters) >= 2:
            ic = int(tr_clusters.index[len(tr_clusters) // 2])
            ival = (cluster_map == ic) & (r == ROLE["train"])
            ibuf = _within(ival, inner_buffer_km, grid) & (r == ROLE["train"]) & ~ival
            r[ibuf] = ROLE["inner_buffer"]
            r[ival] = ROLE["inner_val"]
            blocks.loc[blocks.cluster == ic, f"inner_val_fold{k}"] = True
    params = dict(n_folds=n_folds, block_km=block_km, block_px=bpx, clusters_per_fold=clusters_per_fold,
                  buffer_km=buffer_km, inner_buffer_km=inner_buffer_km, seed=seed)
    return GeoFolds(grid, fold_map, roles, blocks, params)


def fold_summary(folds: GeoFolds, labels: np.ndarray, valid: np.ndarray, region_grid=(3, 3)) -> pd.DataFrame:
    """Per fold x role: pixels, area, label pixels, approx. trace km, EDA-region distribution."""
    g = folds.grid
    skel = skeleton_px(labels)
    rr, cc = np.nonzero(valid)
    reg = np.full(valid.shape, -1, np.int16)
    reg[rr, cc] = region_ids(rr, cc, g, region_grid)
    rows = []
    for k in range(folds.n_folds):
        for name, code in ROLE.items():
            if code < 0:
                continue
            m = folds.roles[k] == code
            if not m.any():
                continue
            regions = np.bincount(reg[m], minlength=region_grid[0] * region_grid[1])
            rows.append({
                "fold": k, "role": name, "px": int(m.sum()),
                "km2": float(m.sum() * g.pixel_km_x * g.pixel_km_y),
                "label_px": int((labels & m).sum()),
                "trace_km_approx": float((skel & m).sum() * g.pixel_km * 1.1),   # 1.1 ~ diagonal steps
                "regions": json.dumps({region_name(i, region_grid): int(n) for i, n in enumerate(regions) if n}),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# pseudo-discovery splits
# ---------------------------------------------------------------------------
DROLE = {"none": 0, "eval_zone": 1, "ignore": 2, "withheld": 3}


@dataclass
class DiscoverySplits:
    grid: Grid
    system_map: np.ndarray             # int32 system id on label pixels, -1 elsewhere
    systems: pd.DataFrame              # one row per system
    roles: np.ndarray                  # int8 (D, H, W) with DROLE codes
    params: dict
    directory: Path | None = None

    @property
    def n_splits(self) -> int:
        return self.roles.shape[0]

    def withheld(self, d: int) -> np.ndarray:
        return self.roles[d] == DROLE["withheld"]

    def ignore(self, d: int) -> np.ndarray:
        """Withheld traces + ring: weight 0 in training (never negatives)."""
        return self.roles[d] >= DROLE["ignore"]

    def eval_zone(self, d: int) -> np.ndarray:
        return self.roles[d] >= DROLE["eval_zone"]

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        layers = {"system_id": np.where(self.system_map >= 0, self.system_map, np.nan)}
        for d in range(self.n_splits):
            layers[f"discovery{d}"] = self.roles[d].astype(np.float32)
        write_geotiff(directory / "discovery_map.tif", layers, self.grid)
        self.systems.to_csv(directory / "fault_systems.csv", index=False)
        (directory / "discovery_params.json").write_text(json.dumps({**self.params, "roles": DROLE}, indent=2),
                                                        encoding="utf-8")
        self.directory = directory
        return directory

    @classmethod
    def load(cls, directory: Path, grid: Grid) -> "DiscoverySplits":
        with rasterio.open(directory / "discovery_map.tif") as src:
            a = src.read()
        sm = np.where(np.isfinite(a[0]), a[0], -1).astype(np.int32)
        roles = np.nan_to_num(a[1:], nan=0).astype(np.int8)
        params = json.loads((directory / "discovery_params.json").read_text(encoding="utf-8"))
        params.pop("roles", None)
        return cls(grid, sm, pd.read_csv(directory / "fault_systems.csv"), roles, params, directory)


def fault_systems(labels: np.ndarray, grid: Grid, link_km: float, region_grid=(3, 3)) -> tuple[np.ndarray, pd.DataFrame]:
    """Connected fault systems: label pixels linked through gaps < ``link_km``."""
    linked = _within(labels, link_km / 2.0, grid)
    comp, n = ndimage.label(linked, structure=np.ones((3, 3), bool))
    sysmap = np.where(labels, comp - 1, -1).astype(np.int32)
    skel = skeleton_px(labels)
    ids = sysmap[labels]
    n_px = np.bincount(ids, minlength=n)
    sk = np.bincount(sysmap[skel & labels], minlength=n)
    rr, cc = np.nonzero(labels)
    r_m = np.bincount(ids, weights=rr, minlength=n) / np.maximum(n_px, 1)
    c_m = np.bincount(ids, weights=cc, minlength=n) / np.maximum(n_px, 1)
    x, y = grid.rc_to_xy(r_m, c_m)
    df = pd.DataFrame({"system_id": np.arange(n), "n_px": n_px, "trace_km": sk * grid.pixel_km * 1.1,
                       "r_cent": r_m, "c_cent": c_m, "x_cent": x, "y_cent": y,
                       "region": region_ids(r_m, c_m, grid, region_grid)})
    df["region_name"] = [region_name(int(r), region_grid) for r in df.region]
    return sysmap, df[df.n_px > 0].reset_index(drop=True)


def make_discovery_splits(labels: np.ndarray, valid: np.ndarray, grid: Grid, *, n_splits: int = 3,
                          link_km: float = 1.0, frac_km: float = 0.12, min_system_km: float = 5.0,
                          max_system_frac: float = 0.05, ignore_km: float = 1.0, eval_km: float = 10.0,
                          seed: int = 2026) -> DiscoverySplits:
    """Withhold ~``frac_km`` of mapped trace length per split as whole systems (disjoint across splits).

    Systems are drawn region-stratified (round-robin over regions in seeded order) so that each split
    spans several regions; systems larger than ``max_system_frac`` of all trace are never withheld
    (they would remove most of the training signal of a region).
    """
    sysmap, systems = fault_systems(labels, grid, link_km)
    total = systems.trace_km.sum()
    rng = np.random.default_rng(seed)
    elig = systems[(systems.trace_km >= min_system_km) & (systems.trace_km <= max_system_frac * total)].copy()
    elig["rand"] = rng.random(len(elig))
    queues = {r: list(g.sort_values("rand").system_id) for r, g in elig.groupby("region")}
    order = []
    while any(queues.values()):
        for r in sorted(queues, key=lambda r: rng.random()):
            if queues[r]:
                order.append(queues[r].pop(0))
    target = frac_km * total
    assign, load = {}, np.zeros(n_splits)
    for sid in order:
        d = int(np.argmin(load))
        if load[d] >= target:
            break
        assign[sid] = d
        load[d] += float(systems.loc[systems.system_id == sid, "trace_km"].iloc[0])
    systems["discovery_split"] = systems.system_id.map(assign).fillna(-1).astype(int)

    roles = np.zeros((n_splits, *labels.shape), np.int8)
    for d in range(n_splits):
        ids = systems.system_id[systems.discovery_split == d].to_numpy()
        wh = np.isin(sysmap, ids) & labels
        retained = labels & ~wh
        ign = _within(wh, ignore_km, grid) & ~retained
        ev = _within(wh, eval_km, grid) & valid
        r = roles[d]
        r[ev] = DROLE["eval_zone"]
        r[ign] = DROLE["ignore"]
        r[wh] = DROLE["withheld"]
    params = dict(n_splits=n_splits, link_km=link_km, frac_km=frac_km, min_system_km=min_system_km,
                  max_system_frac=max_system_frac, ignore_km=ignore_km, eval_km=eval_km, seed=seed,
                  withheld_km=load.tolist(), total_trace_km=float(total))
    return DiscoverySplits(grid, sysmap, systems, roles, params)


def discovery_summary(splits: DiscoverySplits, labels: np.ndarray) -> pd.DataFrame:
    g = splits.grid
    rows = []
    for d in range(splits.n_splits):
        s = splits.systems[splits.systems.discovery_split == d]
        rows.append({"split": d, "n_systems": len(s), "withheld_trace_km": s.trace_km.sum(),
                     "withheld_px": int(splits.withheld(d).sum()),
                     "ignore_px": int(splits.ignore(d).sum()),
                     "eval_zone_km2": float(splits.eval_zone(d).sum() * g.pixel_km_x * g.pixel_km_y),
                     "train_positive_px": int((labels & ~splits.ignore(d)).sum()),
                     "regions": json.dumps(s.region_name.value_counts().to_dict()),
                     "system_ids": json.dumps(s.system_id.tolist())})
    return pd.DataFrame(rows)
