"""G07 datasets (Parquet tables + lazily loaded arrays; features and labels separate).

* ``CombinedStore`` - one FeatureStore-compatible view over the G06 store and the DEM 100 m store (name routing,
  combined schema hash) so the G06 sampling / prediction code serves every family.
* Surface patches (``datasets/surface/``): DEM read through a ``WarpedVRT`` ALIGNED to the 100 m analysis grid at the
  surface work resolution (cells subdivide exactly), channels computed by ``dem_features`` with a halo, stored as a
  float16 memmap (N, C, P, P) + a patch index (analysis-cell origin, DEM coverage, fold / evaluation-zone shares).
  Targets are rasterised per training visibility at train time from the visible 100 m labels (nearest upsampling) -
  never stored with the inputs.
* Subsurface pixels (``datasets/subsurface/``): a pool drawn with the G06 sampler from the G06 store (+ optional DEM
  channels where covered), then per fit and per supervision VARIANT: ``g06`` (G06 labelling), ``improved``
  (system-balanced positives, distance-graded PU weights, hard negatives, full region balance), ``distance``
  (proximity regression target exp(-d / lambda)).
* Candidates (``datasets/candidates/``): candidate rows per unit (label-free features; evaluation-side relevance in a
  separate table) + synthetic training candidates (perturbed training truth / random negatives) flagged
  ``synthetic`` and usable only when their unit is a TRAINING unit.
* Meta (``datasets/meta/``): provenance of every honest held-out map per unit + sampled rank rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import SchemaError


class CombinedStore:
    """Routes feature names to the store that holds them (G06 store first, then DEM store)."""

    def __init__(self, stores: list, footprint_from: int = 0):
        from ..g06.persistence import stable_hash

        self.stores = [s for s in stores if s is not None]
        self.names, self._where = [], {}
        for i, s in enumerate(self.stores):
            for n in s.names:
                if n not in self._where:
                    self._where[n] = i
                    self.names.append(n)
        self.grid = self.stores[0].grid
        self.schema_hash = stable_hash("combined", [s.schema_hash for s in self.stores])
        self._fp = footprint_from
        regs = [s.registry[["feature_id", "family", "permitted_tracks", "usable"]] for s in self.stores]
        self.registry = pd.concat(regs, ignore_index=True).drop_duplicates("feature_id")

    def footprint(self) -> np.ndarray:
        return self.stores[self._fp].footprint()

    def usable(self, names=None) -> list[str]:
        ok = set(self.registry.loc[self.registry.usable.astype(bool), "feature_id"])
        return [n for n in (names or self.names) if n in ok]

    def by_family(self, families, track="A", usable_only=True) -> list[str]:
        R = self.registry
        out = list(R.loc[R.family.isin(list(families)) & R.permitted_tracks.astype(str).str.contains(track),
                         "feature_id"])
        return self.usable(out) if usable_only else out

    def _split(self, names):
        by = {}
        for j, n in enumerate(names):
            if n not in self._where:
                raise SchemaError(f"feature {n!r} is in no attached store")
            by.setdefault(self._where[n], []).append((j, n))
        return by

    def load(self, names, window=None) -> np.ndarray:
        names = list(names)
        h, w = self.grid.shape
        win = window or (0, h, 0, w)
        out = np.empty((len(names), win[1] - win[0], win[3] - win[2]), np.float32)
        for i, pairs in self._split(names).items():
            a = self.stores[i].load([n for _, n in pairs], win)
            for k, (j, _) in enumerate(pairs):
                out[j] = a[k]
        return out

    def sample(self, names, rows, cols) -> np.ndarray:
        names = list(names)
        out = np.empty((len(rows), len(names)), np.float32)
        for i, pairs in self._split(names).items():
            a = self.stores[i].sample([n for _, n in pairs], rows, cols)
            for k, (j, _) in enumerate(pairs):
                out[:, j] = a[:, k]
        return out


# =============================================================================== surface patches
def aligned_dem_reader(dem_path, grid, res_m: float):
    """WarpedVRT of a DEM aligned to the analysis grid at ``res_m`` (100 / res_m sub-cells per cell, average)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT

    f = int(round(grid.pixel_m / res_m))
    T = grid.affine()
    ds = rasterio.open(dem_path)
    vrt = WarpedVRT(ds, crs=grid.crs, transform=Affine(T.a / f, T.b, T.c, T.d, T.e / f, T.f),
                    width=grid.width * f, height=grid.height * f, resampling=Resampling.average,
                    src_nodata=ds.nodata if ds.nodata is not None else np.nan, nodata=np.nan)
    return ds, vrt, f


def read_aligned_patch(vrt, f: int, r0: int, c0: int, cells: int, halo_px: int) -> np.ndarray:
    """(P + 2 halo)^2 work-resolution elevation of analysis cells [r0, r0 + cells) x [c0, c0 + cells)."""
    from rasterio.windows import Window

    P = cells * f
    rr, cc = r0 * f - halo_px, c0 * f - halo_px
    n = P + 2 * halo_px
    out = np.full((n, n), np.nan, np.float32)                 # WarpedVRT: no boundless reads -> clip + NaN pad
    a0, a1 = max(rr, 0), min(rr + n, vrt.height)
    b0, b1 = max(cc, 0), min(cc + n, vrt.width)
    if a1 > a0 and b1 > b0:
        a = np.asarray(vrt.read(1, window=Window(b0, a0, b1 - b0, a1 - a0)), np.float32)
        out[a0 - rr:a1 - rr, b0 - cc:b1 - cc] = a
    out[~np.isfinite(out) | (np.abs(out) > 1e20)] = np.nan
    return out


def surface_channels(z: np.ndarray, res: float, cfg: dict, channels: list[str]) -> dict:
    """The requested fine-tier channels (``dem_features`` ids, unaggregated) of a padded aligned patch."""
    from .dem_features import dem_multiscale_features

    f = dem_multiscale_features(z, res, {**cfg, "scales_m": cfg.get("unet_scales_m", cfg["scales_m"])}, "fine")
    missing = [c for c in channels if c not in f]
    if missing:
        raise SchemaError(f"surface channels not produced at {res} m: {missing}")
    return {c: f[c] for c in channels}


def build_surface_patch_dataset(dem_path, grid, footprint: np.ndarray, coverage: np.ndarray, dem_cfg: dict,
                                unet_cfg: dict, out_dir, fold_map=None, eval_zones=None, progress=None) -> dict:
    """Patch index + float16 channel memmap of every analysis-aligned patch with DEM coverage >= threshold. Reused when
    its identity matches; missing DEM stays NaN (the U-Net's validity channel and loss mask depend on it)."""
    import shutil

    from ..g06.persistence import read_json, stable_hash, write_json, write_table

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = float(unet_cfg.get("res_m", 10.0))
    cells = int(unet_cfg.get("patch_cells", 24))
    stride = int(unet_cfg.get("patch_stride_cells", cells))
    chans = list(unet_cfg["channels"])
    min_cov = float(unet_cfg.get("min_patch_coverage", 0.8))
    h = stable_hash("patches.2", str(dem_path), res, cells, stride, chans, min_cov,
                    {k: dem_cfg.get(k) for k in ("unet_scales_m", "hillshade_azimuths", "relief_windows_m",
                                                 "openness_radius_m", "edge_guard_sigma", "integration_factor")}, n=10)
    old = read_json(out_dir / "surface_manifest.json")
    if isinstance(old, dict) and old.get("fingerprint") == h and old.get("complete") and (
            out_dir / "patch_index.parquet").exists() and (not old.get("memmap") or (out_dir / old["memmap"]).exists()):
        return old
    H, W = grid.shape
    rows = []
    cov = np.nan_to_num(coverage)
    for r0 in range(0, H - cells + 1, stride):
        for c0 in range(0, W - cells + 1, stride):
            sl = (slice(r0, r0 + cells), slice(c0, c0 + cells))
            share = float((cov[sl] >= 0.5).mean())
            if share < min_cov or not np.asarray(footprint)[sl].any():
                continue
            row = {"patch_id": f"s{r0:05d}_{c0:05d}", "r0": r0, "c0": c0, "cells": cells, "dem_share": share,
                   "footprint_share": float(np.asarray(footprint)[sl].mean())}
            if fold_map is not None:
                fm = np.asarray(fold_map)[sl]
                fm = fm[fm >= 0]
                row["fold"] = int(np.bincount(fm).argmax()) if fm.size else -1
            for d, m in (eval_zones or {}).items():
                row[f"eval_share_{d}"] = float(np.asarray(m)[sl].mean())
            rows.append(row)
    idx = pd.DataFrame(rows, columns=None if rows else ["patch_id", "r0", "c0", "cells", "dem_share", "footprint_share"])
    mm_path = out_dir / f"surface_{h}.npy"
    progress_path = out_dir / f"surface_{h}.progress.json"
    from .dem_features import required_halo_px

    halo = required_halo_px({**dem_cfg, "scales_m": dem_cfg.get("unet_scales_m", dem_cfg["scales_m"])}, res, "fine")
    if len(idx):
        P = cells * int(round(grid.pixel_m / res))
        need = len(idx) * len(chans) * P * P * 2
        free = shutil.disk_usage(out_dir).free
        done = int((read_json(progress_path) or {}).get("done", 0)) if mm_path.exists() else 0
        if not done and need > 0.9 * free:
            raise OSError(f"surface patches need {need / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free on the local disk: "
                          "raise surface.unet.patch_stride_cells / min_patch_coverage or use a larger runtime")
        mm = np.lib.format.open_memmap(mm_path, mode="r+" if done else "w+", dtype=np.float16,
                                       shape=(len(idx), len(chans), P, P))
        ds, vrt, f = aligned_dem_reader(dem_path, grid, res)
        try:
            for i, r in enumerate(idx.itertuples()):
                if i < done:
                    continue
                z = read_aligned_patch(vrt, f, r.r0, r.c0, cells, halo)
                ch = surface_channels(z, res, dem_cfg, chans)
                mm[i] = np.stack([np.clip(ch[c][halo:halo + P, halo:halo + P], -6e4, 6e4)    # NaN stays NaN
                                  for c in chans]).astype(np.float16)
                if (i + 1) % 200 == 0:
                    mm.flush()
                    write_json(progress_path, {"done": i + 1})
                if progress and i % 50 == 0:
                    progress(f"  surface patch {i + 1}/{len(idx)}")
        finally:
            vrt.close()
            ds.close()
        mm.flush()
        del mm
    write_table(out_dir / "patch_index.parquet", idx)
    meta = {"memmap": mm_path.name if len(idx) else None, "channels": chans, "res_m": res, "cells": cells,
            "stride_cells": stride, "halo_px": halo, "n_patches": len(idx), "dem_path": str(dem_path), "fingerprint": h,
            "complete": True, "target": "visible 100 m labels upsampled (nearest) per training visibility at train time"}
    write_json(out_dir / "surface_manifest.json", meta)
    progress_path.unlink(missing_ok=True)
    return meta


def validate_surface_patch_artifacts(patch_dir, unet_cfg: dict, grid) -> tuple[dict | None, pd.DataFrame, list[str]]:
    """(manifest, patch index, problems) of the patch dataset against the configuration and the analysis grid. An empty
    problem list means: the memmap loads, its shape equals (patches, channels, P, P), the channel ORDER and resolution
    match the configuration and every patch lies on the grid, so a 100 m target cell maps onto exactly f x f pixels."""
    from ..g06.persistence import read_json

    patch_dir = Path(patch_dir)
    man = read_json(patch_dir / "surface_manifest.json")
    if not isinstance(man, dict) or not man.get("complete"):
        return None, pd.DataFrame(), ["no complete surface patch manifest: run BUILD_DATASETS after DEM_FEATURES"]
    probs = []
    if not man.get("n_patches") or not man.get("memmap"):
        probs.append("the patch dataset is empty: no 2.4 km patch reaches surface.unet.min_patch_coverage of DEM")
        return man, pd.DataFrame(), probs
    if not Path(man.get("dem_path", "")).exists():
        probs.append(f"coarse DEM {man.get('dem_path')} is missing")
    if list(man.get("channels", [])) != list(unet_cfg["channels"]):
        probs.append("patch channels differ from surface.unet.channels (rebuild with BUILD_DATASETS)")
    if abs(float(man.get("res_m", -1)) - float(unet_cfg.get("res_m", 10.0))) > 1e-9:
        probs.append(f"patch resolution {man.get('res_m')} m != configured {unet_cfg.get('res_m', 10.0)} m")
    ratio = grid.pixel_m / float(man["res_m"])
    if abs(ratio - round(ratio)) > 1e-9:
        probs.append(f"{man['res_m']} m does not subdivide the {grid.pixel_m} m analysis cell")
    idx_path = patch_dir / "patch_index.parquet"
    idx = pd.read_parquet(idx_path) if idx_path.exists() else pd.DataFrame()
    if not len(idx):
        probs.append("patch_index.parquet is missing or empty")
        return man, idx, probs
    try:
        mm = np.load(patch_dir / man["memmap"], mmap_mode="r")
    except (OSError, ValueError) as e:
        probs.append(f"patch memmap {man['memmap']} does not load: {type(e).__name__}: {e}")
        return man, idx, probs
    P = int(man["cells"]) * int(round(ratio))
    if tuple(mm.shape) != (len(idx), len(man["channels"]), P, P):
        probs.append(f"patch memmap shape {tuple(mm.shape)} != ({len(idx)}, {len(man['channels'])}, {P}, {P})")
    H, W = grid.shape
    if (idx.r0 < 0).any() or (idx.c0 < 0).any() or (idx.r0 + idx.cells > H).any() or (idx.c0 + idx.cells > W).any():
        probs.append("a patch lies outside the analysis grid")
    return man, idx, probs


READINESS_COLUMNS = ("fit_id", "status", "reason", "n_train", "n_val", "n_train_pos", "n_val_pos", "n_train_neg",
                     "n_val_neg", "val_fold", "mean_dem_share", "input_channels", "res_m")


def surface_patch_readiness(fits: list[str], idx: pd.DataFrame, man: dict | None, problems: list[str], fit_inputs,
                            unet_cfg: dict) -> pd.DataFrame:
    """One row per fit: patch counts per role and READY / BLOCKED with the exact reason. ``fit_inputs(fit_id)`` ->
    (visibility, inner folds): training-visible faults only."""
    from .models_surface import select_surface_patches

    rows = []
    for fid in fits:
        row = dict.fromkeys(READINESS_COLUMNS)
        row.update(fit_id=fid, status="BLOCKED", input_channels=len((man or {}).get("channels", [])),
                   res_m=(man or {}).get("res_m"))
        if problems:
            row["reason"] = "; ".join(problems)
        else:
            vis, inner = fit_inputs(fid)
            I, facts = select_surface_patches(idx, vis, inner, unet_cfg)
            row.update({k: v for k, v in facts.items() if k in READINESS_COLUMNS})
            used = I[I.role.isin(["train", "val"])] if len(I) else I
            row["mean_dem_share"] = float(used.dem_share.mean()) if len(used) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=list(READINESS_COLUMNS))


# =============================================================================== subsurface pixels
def improved_weights(L: pd.DataFrame, cfg: dict) -> np.ndarray:
    """System-balanced positives (equal per system, capped), distance-graded PU weights, hard negatives, normalised
    so the background total equals ``neg_to_pos`` x the positive total."""
    w = np.zeros(len(L), np.float64)
    cls = L.cls.to_numpy()
    pos = cls == 1
    if pos.any():
        sid = L.system_id.to_numpy()[pos]
        n = pd.Series(sid).map(pd.Series(sid).value_counts()).to_numpy().astype(float)
        w[pos] = 1.0 / n
        w[pos] /= w[pos].sum()
    bg = np.isin(cls, (0, 2))
    if bg.any():
        d = L.dist_km.to_numpy()
        far = float(cfg.get("likely_background_km", 3.0))
        u0 = float(cfg.get("unlabeled_min_weight", 0.1))
        base = u0 + (1 - u0) * np.clip(d / far, 0, 1)
        hard = (L.stratum.to_numpy() == "evidence") & (d >= float(cfg.get("hard_negative_min_km", 1.0)))
        base = np.where(hard, base * float(cfg.get("hard_negative_weight", 2.0)), base)
        w[bg] = base[bg]
        w[bg] *= float(cfg.get("neg_to_pos", 1.0)) * max(w[pos].sum(), 1e-12) / max(w[bg].sum(), 1e-12)
    return w.astype(np.float32)


def build_subsurface_pixel_dataset(store, labels: np.ndarray, system_map: np.ndarray, sampling_cfg: dict,
                                   out_dir) -> dict:
    """Pool (meta + features) with the G06 sampler; labels are added per fit by ``subsurface_labels``."""
    from ..g06.datasets import build_pixel_pool
    from ..g06.persistence import write_json, write_table

    meta, X = build_pixel_pool(store, labels, system_map, sampling_cfg)
    out_dir = Path(out_dir)
    write_table(out_dir / "pool_meta.parquet", meta)
    write_table(out_dir / "pool_features.parquet", X)
    m = {"n_rows": len(meta), "n_features": X.shape[1] - 1, "schema_hash": store.schema_hash,
         "strata": meta.stratum.value_counts().to_dict()}
    write_json(out_dir / "pool_manifest.json", m)
    return m


def subsurface_labels(meta: pd.DataFrame, vis, valid, system_map, sup_cfg: dict, samp_cfg: dict, inner_cfg: dict,
                      improved_cfg: dict, seed: int, lambda_km: float) -> pd.DataFrame:
    """G06 label table of one fit + ``weight_improved`` and the ``proximity`` regression target."""
    from ..g06.datasets import label_pool_for_fit

    L = label_pool_for_fit(meta, vis, valid, system_map, sup_cfg, samp_cfg, inner_cfg, seed)
    L["weight_g06"] = L.weight
    L["weight_improved"] = improved_weights(L, {**sup_cfg, **improved_cfg})
    L["proximity"] = np.exp(-np.minimum(L.dist_km.to_numpy(), 5.0) / float(lambda_km)).astype(np.float32)
    return L


# =============================================================================== candidates / meta
def candidate_table_path(root, unit_key: str, kind: str = "") -> Path:
    from ..g06.datasets import unit_file

    return Path(root) / "datasets" / "candidates" / f"{unit_file(unit_key)}{kind}.parquet"


def build_candidate_dataset(engine, store, vu, lab: np.ndarray, pools: dict, evidence: dict, cfg: dict,
                            known_cfg: dict, sup_cfg: dict, truth: np.ndarray | None = None,
                            known: np.ndarray | None = None) -> dict:
    """Candidate rows of one unit: G06 segment features over the combined store (geology + DEM channels), honest
    evidence ranks, pool provenance / expert agreement; relevance table (evaluation side) separately. ``truth`` /
    ``known`` restrict relevance and context to what one training visibility may see (synthetic training candidates:
    ``training_labels.synthetic_training_geometry``); the default is the unit's own truth and permitted context."""
    from ..g06.datasets import _means, segment_features
    from ..g06.feature_context import permitted_fault_context
    from ..g06.supervision import build_segment_supervision, dominant_system

    u = vu.unit
    window = (u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
    truth = vu.eval_truth() if truth is None else truth
    kctx = permitted_fault_context(vu.known if known is None else known, engine.valid()[u.sl], known_cfg)
    seg_cfg = {**cfg, "geology_features": [f for f in cfg.get("geology_features", []) if f in store.names]}
    T = segment_features(lab, store, window, seg_cfg, kctx, evidence)
    if len(T):
        n = int(lab.max())
        rr, cc = np.nonzero(lab)
        lb = lab[rr, cc]
        agree = np.zeros(n)
        for k, m in pools.items():
            sh, _ = _means(lb, np.asarray(m)[rr, cc].astype(np.float64), n)
            T[f"on_{k}"] = sh
            agree += (np.nan_to_num(sh) >= 0.5)
        T["seg_n_experts"] = agree
        T.insert(0, "unit", vu.key)
        T.insert(1, "scenario", vu.scenario)
        T.insert(2, "zone", vu.zone)
        T["r_full"], T["c_full"] = T.row + u.sl[0].start, T.col + u.sl[1].start
    L = build_segment_supervision(lab, truth, sup_cfg)
    if len(L):
        L["system_id"] = dominant_system(lab, vu.sysmap, truth, float(sup_cfg["tol_px"]))
        L.insert(0, "unit", vu.key)
    return {"T": T, "L": L}


def build_meta_dataset(units: dict, maps: dict, provenance: dict, labels_full: np.ndarray, rows_cfg: dict,
                       seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(provenance table, sampled rank rows): every honest held-out map of every unit, ranked inside the unit."""
    from ..gems3.representation import rank01
    from ..p6.stack import sample_weights, training_masks

    prov, rows = [], []
    for key, vu in sorted(units.items()):
        R = {m: rank01(np.nan_to_num(d[key], nan=0.0), vu.unit.ev) for m, d in maps.items() if key in d}
        for m in R:
            prov.append({"unit": key, "map": m, **(provenance.get(m, {}).get(key) or {})})
        if not R:
            continue
        u = vu.unit
        pos, neg = training_masks(u.eval_truth(), u.scored_mask(), labels_full[u.sl], None,
                                  int(rows_cfg.get("neg_buffer_px", 5)))
        idx, y, w = sample_weights(pos, neg, np.zeros(pos.shape, bool), np.random.default_rng(seed + vu.zone),
                                   int(rows_cfg.get("neg_per_unit", 50000)), int(rows_cfg.get("pos_cap_per_unit", 20000)),
                                   1.0, 1.0)
        r, c = np.unravel_index(idx, pos.shape)
        rows.append(pd.DataFrame({**{m: a[r, c] for m, a in R.items()}, "y": y, "w": w, "unit": key,
                                  "r_full": r + u.sl[0].start, "c_full": c + u.sl[1].start}))
    return pd.DataFrame(prov), (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())
