"""Inventory of every data source and reusable artifact, and the adapter to the frozen GEMS engine.

* ``inspect_source_raster``: grid contract + EXACT band identity of the 19-band GeoDAWN file (fails closed later).
* ``discover_existing_artifacts``: configured artifact patterns on Drive / local (folds, splits, G01 / G02 / G05 /
  G05.1 / G05.2 outputs, M9 CNN maps, P6 W2 maps, template). A missing optional artifact stays UNAVAILABLE - it is
  never replaced by zeros; G01 / G02 rasters are historically incomplete in zone 2 and are reported, not backfilled.
* ``attach_engine``: the frozen B0 / B1 / W2 engine through the submission factory (the same path GeoMoE and G05.1
  use): data context (labels, footprint, template, folds, discovery splits), honest per-unit W2 / CNN maps and the
  V2 scenario configuration. Nothing is trained or recomputed by attaching.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import GridSpec


class EngineUnavailable(RuntimeError):
    """The frozen engine (folds, splits, W2 / CNN maps) could not be attached."""


# =============================================================================== sources
def resolve_source(cfg_fs: dict, key: str, roots) -> Path | None:
    """First existing candidate path of source ``key`` under the given roots (local first, then Drive)."""
    for r in roots:
        for c in (cfg_fs.get("sources") or {}).get(key, {}).get("candidates", []):
            p = Path(r) / c
            if p.exists():
                return p
    return None


def inspect_source_raster(path, expected_bands: dict, grid_cfg: dict) -> dict:
    """Grid + band table of the source raster and the list of contract problems (empty = conforming)."""
    import rasterio

    from ..g05_1.bands import resolution_table
    from .raster_io import band_descriptions, band_identifiers

    with rasterio.open(path) as ds:
        grid = GridSpec.from_dataset(ds)
        ids, desc = band_identifiers(ds), band_descriptions(ds)
        dtypes, nodata = list(ds.dtypes), ds.nodata
        block = ds.block_shapes[0] if ds.block_shapes else None
    probs = []
    if grid.shape != (int(grid_cfg["height"]), int(grid_cfg["width"])):
        probs.append(f"grid {grid.shape} != contract ({grid_cfg['height']}, {grid_cfg['width']})")
    if not str(grid.crs).upper().endswith(str(int(grid_cfg["epsg"]))):
        probs.append(f"CRS {grid.crs} != EPSG:{grid_cfg['epsg']}")
    if abs(grid.pixel_m - float(grid_cfg["pixel_m"])) > 1e-6:
        probs.append(f"pixel {grid.pixel_m} m != {grid_cfg['pixel_m']} m")
    if len(ids) != int(grid_cfg["n_bands"]):
        probs.append(f"{len(ids)} bands != {grid_cfg['n_bands']}")
    res = pd.DataFrame(resolution_table(list(expected_bands), ids))
    bad = res[res.status != "RESOLVED"]
    probs += [f"band {r.requested}: {r.status} (near misses: {r.substring_near_misses or '-'})" for r in
              bad.itertuples()]
    unexpected = [i for i in ids if i not in expected_bands]
    if unexpected:
        probs.append(f"identifiers in the file not declared in the config: {unexpected}")
    T = pd.DataFrame({"index": range(1, len(ids) + 1), "identifier": ids, "description": desc, "dtype": dtypes,
                      "nodata": [nodata] * len(ids),
                      "modality": [(expected_bands.get(i) or {}).get("modality", "UNDECLARED") for i in ids],
                      "unit_declared": [(expected_bands.get(i) or {}).get("unit", "") for i in ids],
                      "transform": [(expected_bands.get(i) or {}).get("transform", "") for i in ids]})
    return {"grid": grid, "band_table": T, "resolution": res, "problems": probs, "block_shape": block,
            "band_index": {r.requested: int(r.band_index) for r in res.itertuples() if r.status == "RESOLVED"}}


# =============================================================================== artifacts
def discover_existing_artifacts(drive_root, cfg_fs: dict, local_root=None) -> pd.DataFrame:
    """One row per configured artifact: FOUND / ABSENT, matched paths (first 5), role, required flag."""
    rows = []
    roots = [Path(x) for x in (local_root, drive_root) if x]
    for a in (cfg_fs.get("inventory") or {}).get("artifacts", []):
        hits = []
        for r in roots:
            hits += sorted(glob.glob(str(r / a["pattern"]), recursive=True))
            if hits:
                break
        rows.append({"artifact": a["id"], "role": a.get("role", ""), "required": bool(a.get("required", False)),
                     "pattern": a["pattern"], "status": "FOUND" if hits else "ABSENT", "n_matches": len(hits),
                     "paths": ";".join(hits[:5]), "note": a.get("note", ""),
                     "backfill": a.get("backfill", "") if not hits else ""})
    return pd.DataFrame(rows, columns=["artifact", "role", "required", "pattern", "status", "n_matches", "paths",
                                       "note", "backfill"])


def inspect_feature_coverage(artifacts: pd.DataFrame, footprint: np.ndarray | None = None,
                             zone_maps: dict | None = None) -> pd.DataFrame:
    """Per FOUND raster artifact: finite share inside the footprint, overall and per zone (e.g. G01 / G02 coverage
    gaps in zone 2). Non-raster artifacts are listed with ``kind``."""
    import rasterio

    rows = []
    for r in artifacts.itertuples():
        for p in (r.paths.split(";") if isinstance(r.paths, str) and r.paths else []):
            if not p.endswith((".tif", ".tiff")):
                rows.append({"artifact": r.artifact, "path": p, "kind": Path(p).suffix.lstrip("."),
                             "frac_finite": np.nan})
                continue
            try:
                with rasterio.open(p) as ds:
                    a = ds.read(1, masked=True)
                    fin = np.isfinite(np.ma.filled(a.astype(np.float32), np.nan))
                    row = {"artifact": r.artifact, "path": p, "kind": "tif", "bands": ds.count,
                           "shape": f"{ds.height}x{ds.width}"}
                if footprint is not None and fin.shape == footprint.shape:
                    row["frac_finite"] = float(fin[footprint].mean())
                    for z, m in (zone_maps or {}).items():
                        mm = m & footprint
                        row[f"frac_finite_{z}"] = float(fin[mm].mean()) if mm.any() else np.nan
                else:
                    row["frac_finite"] = float(fin.mean())
                rows.append(row)
            except Exception as e:
                rows.append({"artifact": r.artifact, "path": p, "kind": "tif", "error": f"{type(e).__name__}: {e}"})
    return pd.DataFrame(rows)


def validate_artifact_compatibility(artifacts: pd.DataFrame, grid_spec: GridSpec) -> pd.DataFrame:
    """Raster artifacts: same grid as the source (shape, transform, CRS)? Others: NOT_A_RASTER."""
    import rasterio

    from .contracts import ContractError, validate_grid

    rows = []
    for r in artifacts.itertuples():
        for p in (r.paths.split(";") if isinstance(r.paths, str) and r.paths else []):
            if not p.endswith((".tif", ".tiff")):
                rows.append({"artifact": r.artifact, "path": p, "status": "NOT_A_RASTER", "detail": ""})
                continue
            try:
                with rasterio.open(p) as ds:
                    validate_grid(GridSpec.from_dataset(ds), grid_spec, what=r.artifact)
                rows.append({"artifact": r.artifact, "path": p, "status": "COMPATIBLE", "detail": ""})
            except ContractError as e:
                rows.append({"artifact": r.artifact, "path": p, "status": "INCOMPATIBLE_GRID", "detail": str(e)})
            except Exception as e:
                rows.append({"artifact": r.artifact, "path": p, "status": "UNREADABLE",
                             "detail": f"{type(e).__name__}: {e}"})
    return pd.DataFrame(rows, columns=["artifact", "path", "status", "detail"])


def build_dependency_report(source: dict, artifacts: pd.DataFrame, compat: pd.DataFrame,
                            engine_status: dict) -> tuple[pd.DataFrame, dict]:
    """(dependency table, missing-dependencies json). Mandatory: source raster conforming, faults, engine
    (folds / splits / W2 maps for every stage after the feature store). Optional ones are recorded UNAVAILABLE."""
    rows = [{"dependency": "geodawn_19_band_raster", "required_for": "everything", "mandatory": True,
             "status": "OK" if not source.get("problems") else "FAIL",
             "detail": "; ".join(source.get("problems") or [])[:500]}]
    for r in artifacts.itertuples():
        rows.append({"dependency": r.artifact, "required_for": r.role, "mandatory": bool(r.required),
                     "status": "OK" if r.status == "FOUND" else ("FAIL" if r.required else "UNAVAILABLE"),
                     "detail": (r.paths if r.status == "FOUND" else r.backfill or r.note)[:500]})
    bad_grid = compat[compat.status == "INCOMPATIBLE_GRID"] if len(compat) else compat
    for r in bad_grid.itertuples():
        rows.append({"dependency": f"{r.artifact} grid", "required_for": "alignment", "mandatory": False,
                     "status": "INCOMPATIBLE", "detail": r.detail[:500]})
    rows.append({"dependency": "frozen_engine (folds, splits, W2 / CNN maps)", "required_for":
                 "splits, datasets, models, evaluation", "mandatory": True,
                 "status": "OK" if engine_status.get("attached") else "FAIL",
                 "detail": str(engine_status.get("detail", ""))[:500]})
    T = pd.DataFrame(rows)
    missing = {"mandatory_missing": list(T.loc[T.mandatory & (T.status != "OK"), "dependency"]),
               "optional_unavailable": list(T.loc[~T.mandatory & (T.status != "OK"), "dependency"]),
               "backfill_instructions": {r.artifact: r.backfill for r in artifacts.itertuples()
                                         if r.status != "FOUND" and r.backfill},
               "policy": "optional inputs stay UNAVAILABLE (never zero-filled); expensive G01 / G02 backfills are "
                         "never launched automatically"}
    return T, missing


# =============================================================================== frozen engine
@dataclass
class Engine:
    F: object
    fg: object
    bst: object
    p6: object
    ctx: object
    head: str
    v2_cfg: dict = field(default_factory=dict)
    _w2: dict = field(default_factory=dict, repr=False)

    @property
    def data(self):
        return self.ctx.data

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.ctx.data.shape)

    def labels(self) -> np.ndarray:
        return np.asarray(self.ctx.data.labels, bool)

    def valid(self) -> np.ndarray:
        return np.asarray(self.ctx.data.valid, bool)

    def template_valid(self) -> np.ndarray:
        return np.asarray(self.ctx.data.template_valid, bool)

    def w2_map(self, base) -> np.ndarray:
        """Honest held-out W2 map of a base unit (crop; P6 stack/maps/<task>_<zone>.npy)."""
        return np.asarray(self.bst.w2_map(base), np.float32)

    def w2_rank(self, base) -> np.ndarray:
        return np.asarray(self.bst.stack_rank(base, self.w2_map(base)), np.float32)

    def cnn(self, base) -> tuple[np.ndarray, np.ndarray]:
        return self.bst.cnn(base)

    def provenance(self, base) -> dict:
        tag = "oof" if base.task == "geo" else f"discovery_pred:{base.zone}"
        return {"cnn_b0": tag, "cnn_b1": tag, "w2": f"p6_stack_map:{base.task}_{base.zone}"}

    def release(self, base) -> None:
        for d in (getattr(self.bst, "_raw", {}), getattr(self.bst, "_cnn", {})):
            d.pop(base.key, None)
        base.release()

    def identity(self) -> dict:
        w2 = self.bst.w2_inputs() if self.bst is not None else {}
        return {"data": self.fg.data_fingerprint() if self.fg is not None else None,
                "w2_maps": w2.get("maps_sha256"), "w2_rows": w2.get("rows_sha256"), "head": self.head}


def attach_engine(nb: dict, head: str, factory=None) -> Engine:
    """Attach the frozen engine through the submission factory (read-only; CPU)."""
    from ..geomoe import run as GR

    try:
        F = factory
        if F is None:
            from ..leaderboard_focus.factory import Factory

            c = {**GR.NB_DEFAULTS, **{k: nb[k] for k in ("repo", "ref", "drive_root", "local_root", "reuse_cached",
                                                         "sync_to_drive")}}
            F = Factory(GR.factory_config(c), head)
        if F.selftests is None:
            F.selftest()
        if F.sess is None:
            F.preflight()
            if not F.attach():
                raise EngineUnavailable("factory attach failed: " + json.dumps(
                    {k: v for k, v in F.block.items() if v}, default=str)[:800])
        fg = F.sess.fg
        v2 = GR.load_config().get("v2") or {}
        return Engine(F, fg, fg.bst, fg.p6, fg.p6.ctx, head, v2)
    except EngineUnavailable:
        raise
    except Exception as e:
        raise EngineUnavailable(f"{type(e).__name__}: {e}") from e
