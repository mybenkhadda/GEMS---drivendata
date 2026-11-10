"""G11 orchestrator: resumable, idempotent stages behind the execution modes (``config.Mode``).

    P = G11Pipeline(load_config(overrides={"data_root": "/content/drive/MyDrive/GEMS_2"}))
    P.run("AUDIT")          # or INGEST / FEATURES / EDA / TRAIN / EVALUATE / INFER / SUBMIT / FULL

Every stage records ``manifests/<stage>.json`` (fingerprint, outputs + sha256, config hash, code commit, source
hashes, timestamp) and is skipped when its record is still valid. Integrity gates: the audit fails fast, the split
stage refuses to continue when a leakage check fails, test metrics are computed only after the validation-based
selection has been written, and the submission is exported only after template validation.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit as A
from . import fault_labels as FL
from . import spatial_splits as SS
from .config import MODE_STAGES, Mode
from .features import FeatureStore, write_geotiff
from .grid import load_master_grid, validate_alignment
from .registry import (RunContext, StageStore, atomic_write_json, atomic_write_parquet, atomic_write_text, read_json)


class StageGateError(RuntimeError):
    pass


class G11Pipeline:
    def __init__(self, cfg: dict, log=print, tracker=None):
        from .colab_runtime import StageTracker

        self.cfg, self.log = cfg, log
        self.ctx = RunContext(cfg)
        self.ctx.ensure_layout()
        self.stages = StageStore(self.ctx)
        self.tracker = tracker or StageTracker()
        self._grid = self._store = self._mask = self._split = self._full_grid = None
        self.reuse = bool(cfg.get("reuse_valid_cache", True))
        self.smoke = bool(cfg.get("smoke", {}).get("enabled", False))
        s = read_json(self.ctx.d("audit") / "audit_summary.json")
        if s and s.get("source_hashes"):
            self.ctx.source_hashes = s["source_hashes"]

    # ================================================================== helpers
    def run(self, mode) -> None:
        for st in MODE_STAGES[Mode(mode)]:
            self.log(f"========== G11 stage: {st}")
            with self.tracker.track(st, self.log):
                getattr(self, st)()

    def model_dir(self, exp: str) -> Path:
        from .experiments import EXPERIMENTS
        from .registry import MODEL_DIRS

        base = exp.split("__")[0]
        kind = EXPERIMENTS.get(base, {}).get("kind", "fusion" if base == "E5" else "pixel")
        return self.ctx.d("train", MODEL_DIRS[kind], exp)

    def pred_path(self, exp: str) -> Path:
        return self.ctx.d("predictions") / f"{exp}_probability.tif"

    def available_predictions(self) -> list[str]:
        """Experiment ids with a G11 full-grid probability map (E0..E5 and ablation variants E3__*, E4__*)."""
        return sorted(p.name[: -len("_probability.tif")] for p in self.ctx.d("predictions").glob("E*_probability.tif"))

    def _cached(self, stage: str, fp: str) -> bool:
        if not self.reuse:
            return False
        ok, why = self.stages.status(stage, fp)
        self.log(f"[{stage}] {'CACHED' if ok else 'COMPUTING'} ({why})")
        return ok

    @property
    def found(self) -> dict:
        s = read_json(self.ctx.d("audit") / "audit_summary.json")
        if s is None:
            raise StageGateError("run the AUDIT stage first")
        return s["found"]

    @property
    def full_grid(self):
        if self._full_grid is None:
            g = self.cfg["geophysical_grid"]
            self._full_grid = load_master_grid(self.found["geodawn"], g["valid_min_band_share"], g["nodata_abs"], g)
        return self._full_grid

    @property
    def smoke_window(self):
        if not self.smoke:
            return None
        rec = read_json(self.ctx.d("audit") / "smoke_window.json")
        if rec is None:
            raise StageGateError("SMOKE_TEST: no smoke window yet - run the audit stage first")
        return rec["window"]

    @property
    def grid(self):
        if self._grid is None:
            from .grid import crop_grid

            w = self.smoke_window
            self._grid = crop_grid(self.full_grid, w) if w else self.full_grid
        return self._grid

    @property
    def store(self) -> FeatureStore:
        if self._store is None:
            self._store = FeatureStore(self.ctx.d("features"), self.grid)
        return self._store

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            self._mask = FL.label_raster(self.found["fault_raster"], self.grid, self.cfg["faults"]["min_label_value"])
        return self._mask

    def catalog(self):
        p = self.ctx.d("ingest") / "dem_catalog.parquet"
        if not p.is_file():
            return None
        import geopandas as gpd
        from shapely import wkt

        t = pd.read_parquet(p)
        if not len(t):
            return None
        geom = t.pop("footprint_wkt").map(lambda s: wkt.loads(s) if isinstance(s, str) else None)
        return gpd.GeoDataFrame(t, geometry=geom, crs=self.grid.crs)

    def dem_reader(self):
        from .dem_reader import DEMReader

        cat = self.catalog()
        if cat is None:
            return None
        return DEMReader(cat, self.cfg, allow_unknown_vertical=bool(self.cfg["dem"].get("allow_unknown_vertical", False)))

    def segments(self):
        import geopandas as gpd

        p = self.ctx.d("ingest") / "fault_segments.gpkg"
        return gpd.read_file(p) if p.is_file() else None

    def split_result(self, check_manifest: bool = True) -> SS.SplitResult:
        """Deterministic recomputation of the splits; must equal the saved manifest (if one exists)."""
        if self._split is None:
            res = SS.make_splits(self.grid.valid, self.mask, self.grid, self.cfg)
            man = read_json(self.ctx.d("splits") / "split_manifest.json") if check_manifest else None
            if man is not None:
                got = {k: int(res.mask(k).sum()) for k in list(SS.SPLITS) + ["embargo"]}
                if got != man["pixels"]:
                    raise StageGateError(f"recomputed splits differ from the saved manifest: {got} vs {man['pixels']}")
            self._split = res
        return self._split

    def _get(self, name):
        try:
            return np.asarray(self.store.get(name), np.float32)
        except KeyError:
            return None

    # ================================================================== Phase 1: audit
    def audit(self) -> dict:
        cfg = self.cfg
        d = self.ctx.d("audit")
        reg, summ = A.build_artifact_registry(cfg)
        roots = [Path(cfg["data_root"]) / r for r in cfg["source_artifacts"]["search_roots"]]
        inv = A.discover_artifacts(roots, ["*"], cfg["source_artifacts"]["max_depth"])
        outs = [atomic_write_parquet(d / "artifact_registry.parquet", reg),
                atomic_write_json(d / "artifact_registry.json", reg.to_dict("records")),
                atomic_write_parquet(d / "artifact_inventory.parquet", inv),
                atomic_write_text(d / "audit_report.md", A.audit_report_md(reg, summ))]
        A.fail_fast(summ)
        grid = load_master_grid(summ["found"]["geodawn"], cfg["geophysical_grid"]["valid_min_band_share"],
                                cfg["geophysical_grid"]["nodata_abs"], cfg["geophysical_grid"])
        problems = []
        for key in ("fault_raster", "template"):
            m = A.inspect_raster(summ["found"][key])
            from .grid import GridSpec

            rep = validate_alignment(GridSpec(m.crs, m.transform, (m.height, m.width)), grid)
            if not rep.aligned:
                problems.append(f"{key} is not on the master grid: {rep.as_dict()}")
        summ["grid"] = grid.as_dict()
        hashes = {k: self.ctx.hashes.sha(summ["found"][k]) for k in ("geodawn", "fault_raster", "template")}
        summ["source_hashes"] = hashes
        self.ctx.source_hashes = hashes
        # --- DEM vertical units (authoritative metadata only) and duplicates across DEM folders
        from .dem_catalog import dem_files
        from .dem_units import audit_vertical_units

        dem_dirs = summ["found"].get("dem_dirs") or []
        files, dups = dem_files(dem_dirs)
        cats = [Path(x) / n for x in dem_dirs for n in ("usgs_1m_catalog.csv", "dem_1m_inventory.csv")]
        units = audit_vertical_units(files, cats)
        outs.append(atomic_write_text(d / "dem_vertical_units.csv", units.to_csv(index=False)))
        summ["dem_tiles"] = int(len(files))
        summ["dem_duplicates"] = len(dups)
        summ["dem_vertical_units"] = units["class"].value_counts().to_dict() if len(units) else {}
        if dups:
            outs.append(atomic_write_text(d / "dem_duplicate_tiles.csv", pd.DataFrame(dups).to_csv(index=False)))
        # --- existing model predictions: documented placement only, aligned to the master grid
        from .external_predictions import register_external

        fams = {k: v for k, v in summ["optional_paths"].items() if k in ("g07_arrays", "g07_prediction", "m9", "p6",
                                                                         "georegime", "g10")}
        fams = {k: [p for p in v if Path(p).suffix.lower() in (".npy", ".tif", ".tiff")] for k, v in fams.items()}
        comp = register_external(fams, grid, self.ctx.d("predictions", "external"))
        outs.append(atomic_write_text(d / "artifact_compatibility.csv", comp.to_csv(index=False)))
        summ["external_compatible"] = int(comp.status.str.startswith("COMPATIBLE").sum()) if len(comp) else 0
        if self.smoke:
            from .smoke import select_smoke_window

            sw = select_smoke_window(summ["found"]["fault_raster"], files, grid, self.cfg["smoke"]["window_km"],
                                     d / "smoke_window.json")
            summ["smoke_window"] = sw
            outs.append(d / "smoke_window.json")
        outs.append(atomic_write_json(d / "audit_summary.json", summ))
        outs.append(atomic_write_text(d / "dependency_graph.md", dependency_graph(summ)))
        outs.append(atomic_write_text(d / "audit_report.md", A.audit_report_md(reg, summ) + _audit_extra_md(units, comp, summ)))
        if problems:
            raise A.AuditError("G11 audit failed:\n  - " + "\n  - ".join(problems))
        self._full_grid = grid
        self._grid = None
        self.stages.record("audit", self.stages.fingerprint("audit"), outs, {"skipped": len(summ["skipped"]),
                                                                               "dem_tiles": len(files)})
        self.log(f"[audit] grid {grid.shape} EPSG:{grid.epsg}; {len(files)} DEM tiles "
                 f"{summ['dem_vertical_units']}; {summ['external_compatible']} compatible external prediction(s); "
                 f"{len(summ['skipped'])} optional source(s) skipped")
        return summ

    # ================================================================== Phase 2: ingestion
    def ingest(self) -> dict:
        cfg, grid = self.cfg, self.grid
        fp = self.stages.fingerprint("ingest")
        if self._cached("ingest", fp):
            return self.stages.load("ingest")["stats"]
        d = self.ctx.d("ingest")
        mask = self.mask
        if self.found.get("fault_vector"):
            segs = FL.repair_fault_geometries(FL.load_fault_geometries(self.found["fault_vector"], grid.crs))
            seg_src = "vector"
        else:
            segs = FL.segments_from_raster(mask, grid, cfg["faults"]["min_segment_length_m"], cfg["faults"]["simplify_m"])
            seg_src = "raster_skeleton (no vector source found)"
        outs = []
        if len(segs):
            p = d / "fault_segments.gpkg"
            tmp = d / "fault_segments.partial.gpkg"
            segs.to_file(tmp, driver="GPKG", layer="faults")
            tmp.replace(p)
            outs.append(p)
        targets = FL.make_targets(mask, grid.valid, grid, cfg, segs if seg_src == "vector" else None)
        for k, a in targets.items():
            outs.append(self.store.put(k, a, "target", None, "m" if k == "Y_DISTANCE" else "1", "fault labels",
                                       label_derived=True, fingerprint=fp))
        st = FL.label_states(mask, grid.valid, targets["Y_DISTANCE"], cfg)
        outs.append(write_geotiff(d / "label_states.tif", st, grid, ["label state (labels.states)"], nodata=None))
        outs.append(atomic_write_json(d / "master_grid.json", {**grid.as_dict(), "smoke_window": self.smoke_window,
                                                               "full_grid": self.full_grid.as_dict()}))
        # --- geophysical band registry (all bands validated by name, unit, finite fraction)
        from .features.geophysics import band_registry

        breg = band_registry(self.found["geodawn"])
        nb = cfg["geophysical_grid"].get("expected_band_count")
        breg_ok = nb is None or len(breg) == int(nb)
        outs.append(atomic_write_parquet(d / "geophysical_registry.parquet", breg))
        if not breg_ok and cfg.get("reporting", {}).get("strict_integrity", True):
            raise StageGateError(f"GeoDAWN has {len(breg)} bands, expected {nb} (geophysical_grid.expected_band_count)")
        # --- fault registry with fault-system grouping (8-connected components of the label raster)
        from scipy import ndimage

        from .grid import inside, world_to_pixel

        sysl, nsys = ndimage.label(mask, structure=np.ones((3, 3), bool))
        freg = pd.DataFrame(segs.drop(columns="geometry")) if len(segs) else pd.DataFrame()
        if len(segs):
            mid = segs.geometry.interpolate(0.5, normalized=True)
            r, c = world_to_pixel(mid.x.to_numpy(), mid.y.to_numpy(), grid)
            ok = inside(r, c, grid)
            sid = np.full(len(segs), -1)
            if nsys:
                near = ndimage.distance_transform_edt(sysl == 0, return_distances=False, return_indices=True)
                sid[ok] = sysl[near[0][r[ok], c[ok]], near[1][r[ok], c[ok]]]
                del near
            freg["fault_system_id"] = sid
            freg["fault_system_source"] = "8-connected label-raster component nearest to the segment midpoint"
        outs.append(atomic_write_parquet(d / "fault_registry.parquet", freg))
        stats = {"fault_pixels": int(mask.sum()), "segments": int(len(segs)), "segment_source": seg_src,
                 "segment_length_km": float(segs.length_m.sum() / 1000) if len(segs) else 0.0, "fault_systems": int(nsys),
                 "geodawn_bands": int(len(breg)), "band_count_ok": bool(breg_ok)}
        cov_rep = {"geophysics_valid_km2": float(grid.valid.sum() * grid.res[0] * grid.res[1] / 1e6),
                   "fault_pixels": int(mask.sum())}
        dem_dirs = self.found.get("dem_dirs") or []
        if dem_dirs:
            from .dem_catalog import build_dem_catalog, coverage_100m, save_catalog

            up = self.ctx.d("audit") / "dem_vertical_units.csv"
            units = pd.read_csv(up) if up.is_file() else None
            cat = build_dem_catalog(dem_dirs, grid.crs, cfg, units=units)
            outs += save_catalog(cat, d, grid)
            allc = coverage_100m(cat, grid)
            bad = cat.quality_flags.fillna("").str.contains("UNKNOWN_VERTICAL_UNIT|CONFLICTING_VERTICAL_UNIT|UNREADABLE")
            allow = bool(cfg["dem"].get("allow_unknown_vertical", False))
            usable_cat = cat if allow else cat[~bad]
            usable = coverage_100m(usable_cat, grid)["covered"] if len(usable_cat) else np.zeros(grid.shape, bool)
            fault_all, fault_use = int((mask & allc["covered"]).sum()), int((mask & usable).sum())
            cov_rep.update(dem_tiles=int(len(cat)), dem_tiles_usable=int(len(usable_cat)),
                           dem_covered_km2=float((allc["covered"] & grid.valid).sum() * grid.res[0] * grid.res[1] / 1e6),
                           dem_usable_km2=float((usable & grid.valid).sum() * grid.res[0] * grid.res[1] / 1e6),
                           fault_pixels_with_any_dem=fault_all, fault_pixels_with_usable_dem=fault_use,
                           fault_coverage_lost_to_unit_exclusion=(fault_all - fault_use) / max(fault_all, 1),
                           fault_fraction_with_usable_dem=fault_use / max(int(mask.sum()), 1),
                           flags=cat.quality_flags.value_counts().to_dict() if len(cat) else {},
                           duplicates_across_folders=len(cat.attrs.get("duplicates", [])))
            stats.update(dem_tiles=int(len(cat)), dem_tiles_usable=int(len(usable_cat)))
        else:
            allc, usable = None, np.zeros(grid.shape, bool)
            stats["dem_tiles"] = 0
        outs.append(atomic_write_json(d / "coverage_report.json", cov_rep))
        if cfg.get("reporting", {}).get("figures", True):
            outs += self._ingest_figures(d / "figures", mask, allc, usable)
        outs.append(self.store.flush())
        self.stages.record("ingest", fp, outs, stats, validation={"band_count_ok": breg_ok})
        self.log(f"[ingest] {stats}")
        return stats

    def _ingest_figures(self, fd: Path, mask, allc, usable) -> list:
        from .eda import figures as EG
        from .features.geophysics import read_all_bands
        from .features.terrain import compute_terrain_features

        grid, out = self.grid, []
        bands = read_all_bands(self.found["geodawn"], self.grid)
        cnt = np.sum([np.isfinite(a) for a in bands.values()], axis=0)
        out.append(EG.missing_data(grid, grid.valid, cnt, len(bands), fd / "missing_data.png"))
        for key in [k for k in bands if k in ("rtp", "iso_grav_anom", "det_elev")][:3] or list(bands)[:2]:
            out.append(EG.band_with_faults(grid, bands[key], mask, fd / f"band_{key}_faults.png", f"GeoDAWN {key}"))
        del bands
        if allc is not None:
            out.append(EG.dem_footprints(grid, allc["count"], usable, mask, fd / "dem_footprints_fault_coverage.png"))
            reader = self.dem_reader()
            segs = self.segments()
            if reader is not None and segs is not None and len(segs):
                k = 0
                for g in segs.sort_values("length_m", ascending=False).geometry:
                    if k >= 3:
                        break
                    c = g.centroid
                    b = (c.x - 1000, c.y - 1000, c.x + 1000, c.y + 1000)
                    try:
                        w = reader.read_dem_window(b, grid.crs, resolution=2.0, resampling="average")
                    except Exception:                                            # noqa: BLE001
                        continue
                    if w.valid.mean() < 0.3:
                        continue
                    tf = compute_terrain_features(w.elevation, w.valid, 2.0, [10], self.cfg,
                                                  families=("first_order", "hillshade"))
                    out.append(EG.native_window(tf["hillshade_315_10m"], tf["slope_10m"],
                                                (b[0] / 1e3, b[2] / 1e3, b[1] / 1e3, b[3] / 1e3),
                                                f"native DEM window @ E{c.x:.0f} N{c.y:.0f} | tiles {w.source_tiles}",
                                                fd / f"native_dem_window_{k}.png"))
                    k += 1
        return [p for p in out if p]

    # ================================================================== Phase 3: features
    def features(self) -> dict:
        from .features import geophysics as G
        from .features.multiscale import compute_multimodal_alignment, pct_rank, terrain_100m_summaries

        cfg, grid, store = self.cfg, self.grid, self.store
        fp = self.stages.fingerprint("features", self.stages.load("ingest") and self.stages.load("ingest")["fingerprint"])
        if self._cached("features", fp):
            return self.stages.load("features")["stats"]
        notes = []
        for name, a in G.read_all_bands(self.found["geodawn"], grid).items():
            store.put(f"raw_{name}", np.where(grid.valid, a, np.nan), "raw", 100.0, "as delivered", "geodawn", False, fp)
        bands = G.read_geodawn_bands(self.found["geodawn"], cfg["feature_engineering"]["geophysics"]["bands"], grid)
        notes += [f"band {k} absent from GeoDAWN" for k, v in bands.items() if v is None]
        G.compute_geophysical_to_store(bands, grid, cfg, store, fp)
        del bands
        opt = (read_json(self.ctx.d("audit") / "audit_summary.json") or {}).get("optional_paths", {})
        for reg in opt.get("g06_store", [])[:1]:
            try:
                notes.append(f"G06 store attached: {len(G.attach_g06_store(reg, store, grid))} layers")
            except Exception as e:                                        # noqa: BLE001
                notes.append(f"G06 store unusable ({type(e).__name__}: {e})")
        sc = [s for s in cfg["feature_engineering"]["geophysics"]["scales_m"] if s in (400, 800)] or \
            cfg["feature_engineering"]["geophysics"]["scales_m"][:1]
        gm = [self._get(f"{f}_gm_{int(sc[0])}m") for f in ("magnetic", "gravity")]
        gm = [x for x in gm if x is not None]
        if gm:
            store.put("geo_edge_rank", np.nanmax(np.stack([pct_rank(x, grid.valid) for x in gm]), 0), "evidence_rank",
                      float(sc[0]), "rank", "label-free", False, fp)
        reader = self.dem_reader()
        if reader is not None and len(reader.index.cat):
            from .dem_catalog import coverage_100m
            from .dem_reader import aggregate_dem_to_master

            cov = coverage_100m(reader.index.cat, grid)["covered"] & grid.valid
            self.log(f"[features] DEM aggregation over {int(cov.sum())} covered cells")
            aggregate_dem_to_master(reader, grid, cov, cfg, store, self.log)
            slope95 = self._get("dem100_slope_p95")
            if slope95 is not None:
                store.put("terrain_scarp_rank", pct_rank(slope95, grid.valid & np.isfinite(slope95)), "evidence_rank",
                          100.0, "rank", "label-free", False, fp)
            for s in sc:
                terr = terrain_100m_summaries(store, grid, int(s))
                if terr["coh"] is None:
                    continue
                geo = {}
                for f in ("magnetic", "gravity"):
                    for k in ("sc2", "ss2", "coh", "gm"):
                        a = self._get(f"{f}_{k}_{int(s)}m")
                        if a is not None:
                            geo[f"{f}_{k}_{int(s)}m"] = a
                mm = compute_multimodal_alignment(geo, terr, grid.valid, int(s))
                for r in mm.meta.itertuples():
                    store.put(r.name, mm[r.name], "multimodal", r.scale_m, r.unit, "geo x dem", False, fp)
            seg_out = self._known_fault_morphology(reader)
            notes += seg_out.pop("notes", [])
        else:
            seg_out = {}
            notes.append("no usable DEM tiles: DEM and multimodal features skipped (E2 / E4 unavailable)")
        out = store.flush()
        stats = {"n_features": int(len(store.reg)), "n_predictors": len(store.predictors()), "notes": notes,
                 "families": store.reg.family.value_counts().to_dict(), **seg_out}
        self.stages.record("features", fp, [out] + [Path(p) for p in stats.get("outputs", [])], stats)
        self.log(f"[features] {stats['n_predictors']} predictors; notes: {notes}")
        return stats

    def _known_fault_morphology(self, reader, max_segments: int = 2000, native_windows: int = 6) -> dict:
        """Cross-fault profiles of the KNOWN fault segments (03_features/segment_features.parquet) and a few native
        terrain feature windows (03_features/terrain_native/*.tif) for candidate-level analysis."""
        from .features.segment import segment_features
        from .features.terrain import compute_terrain_features
        from .grid import GridSpec

        segs = self.segments()
        d = self.ctx.d("features")
        if segs is None or not len(segs):
            return {"notes": ["no known fault segments: segment features skipped"]}
        cov = reader.index.cat.union_all() if hasattr(reader.index.cat, "union_all") else reader.index.cat.unary_union
        s = segs[segs.geometry.intersects(cov)].sort_values("length_m", ascending=False).head(max_segments)
        outs, notes = [], [f"{len(s)} of {len(segs)} known segments intersect the DEM coverage"]
        if len(s):
            rasters = {"geo_edge_rank": self._get("geo_edge_rank"), "terrain_scarp_rank": self._get("terrain_scarp_rank")}
            feats = segment_features(s.reset_index(drop=True), rasters, self.grid, self.cfg, reader)
            feats.insert(0, "segment_index", s.index.to_numpy())
            outs.append(atomic_write_parquet(d / "segment_features.parquet", feats))
        nd = self.ctx.d("features", "terrain_native")
        res = float(self.cfg["feature_engineering"]["cross_fault"]["resolution_m"])
        for k, g in enumerate(s.geometry[:native_windows]):
            c = g.centroid
            b = (c.x - 300, c.y - 300, c.x + 300, c.y + 300)                 # 600 m window (memory-bounded)
            try:
                w = reader.read_dem_window(b, self.grid.crs, resolution=res)
            except Exception as e:                                           # noqa: BLE001
                notes.append(f"native window {k}: {type(e).__name__}: {e}")
                continue
            if w.valid.mean() < 0.3:
                continue
            scales = [x for x in self.cfg["feature_engineering"]["terrain"]["scales_m"] if x <= 100][:3]
            tf = compute_terrain_features(w.elevation, w.valid, res, scales, self.cfg,
                                          families=("first_order", "second_order", "structural"))
            names = list(tf.arrays)
            wg = GridSpec(w.crs, w.transform, w.elevation.shape)
            outs.append(write_geotiff(nd / f"native_{k:02d}_E{c.x:.0f}_N{c.y:.0f}.tif",
                                      np.stack([tf[n] for n in names]), wg, names))
        return {"notes": notes, "outputs": [str(o) for o in outs], "known_segments_with_dem": int(len(s))}

    # ================================================================== Phase 4: EDA
    def eda(self) -> dict:
        from .eda import coverage as EC
        from .eda import distributions as ED
        from .eda import fault_contrast as EF
        from .eda import figures as EG
        from .eda import multiscale as EM
        from .eda import report as ER
        from .eda.spatial import block_sample, correlation_length, decorrelation_distance

        cfg, grid, store = self.cfg, self.grid, self.store
        e = cfg["eda"]
        fp = self.stages.fingerprint("eda", self.stages.load("features")["fingerprint"])
        if self._cached("eda", fp):
            return self.stages.load("eda")["stats"]
        d = self.ctx.d("eda")
        valid, mask = grid.valid, self.mask
        dist = np.asarray(store.get("Y_DISTANCE"), np.float32)
        reader = self.dem_reader()
        from .dem_catalog import coverage_100m

        dem_cov = coverage_100m(reader.index.cat, grid)["covered"] if reader is not None else np.zeros(grid.shape, bool)
        names = store.predictors()
        meta = store.reg[store.reg.name.isin(names)]
        region_px = max(int(40_000 / grid.res[0]), 1)
        nbc = -(-grid.shape[1] // region_px)
        ri, ci = np.indices(grid.shape, dtype=np.int32)
        regions = np.where(valid, (ri // region_px) * nbc + ci // region_px, -1)
        del ri, ci
        segs = self.segments()
        cov = EC.coverage_summary(valid, dem_cov, mask, segs, reader.index if reader else None, grid)
        groups = ED.group_masks(mask, dist, valid, dem_cov, cfg["labels"]["near_band_m"])
        dist_t = ED.distribution_table(store, names, groups, e["sample_block_px"], cfg["reproducibility"]["seed"],
                                       e["max_samples"])
        bpx = max(int(e["bootstrap_block_km"] * 1000 / grid.res[0]), 1)
        contrast = EF.contrast_table(store, names, mask, dist, valid, regions, cfg["labels"]["near_band_m"],
                                     block_px=bpx, n_boot=e["n_bootstrap"], seed=cfg["reproducibility"]["seed"],
                                     max_n=e["max_samples"])
        near = valid & (dist <= cfg["labels"]["near_band_m"])
        ctrl = valid & (dist > 1000) & (dist <= 3000)
        rp, cp, _ = block_sample(near, 2, cfg["reproducibility"]["seed"], e["max_samples"])
        rn, cn, _ = block_sample(ctrl, 2, cfg["reproducibility"]["seed"] + 1, e["max_samples"])
        util = EM.utility_by_scale(store, meta, (rp, cp), (rn, cn), regions)
        rs, cs, _ = block_sample(valid, e["sample_block_px"], cfg["reproducibility"]["seed"], e["max_samples"])
        redund = EM.redundancy(store, names, rs, cs, e["redundancy_rho"])
        corr = []
        for n in [x for x in names if x.endswith("_gm_400m") or x == "geo_edge_rank"][:4]:
            cl = correlation_length(np.asarray(store.get(n), np.float32), valid, grid.res[0])
            corr.append({"feature": n, "decorrelation_distance_m": decorrelation_distance(cl)})
        corr = pd.DataFrame(corr)
        failures, cand = self._eda_candidates(reader)
        from .eda import model_evidence as MEV

        native = EM.native_scale_contrast(reader, segs, grid, cfg) if reader is not None else pd.DataFrame()
        multi = pd.concat([util.assign(resolution_m=grid.res[0]), native], ignore_index=True)
        maps = MEV.load_compatible(str(self.ctx.d("audit") / "artifact_compatibility.csv"))
        if self.smoke and maps:                       # external maps are full-grid: crop to the smoke window
            r0, r1, c0, c1 = self.smoke_window
            maps = {k: {**v, "array": v["array"][r0:r1, c0:c1]} for k, v in maps.items()}
        compl = MEV.complementarity(maps, mask, dist, valid, self._get("terrain_scarp_rank")) if maps else pd.DataFrame()
        cov_json = read_json(self.ctx.d("ingest") / "coverage_report.json") or {}
        recs = ER.recommendation_table(contrast, util, native, compl, cov_json)
        outs = [atomic_write_parquet(d / "coverage_summary.parquet", cov),
                atomic_write_parquet(d / "feature_distributions.parquet", dist_t),
                atomic_write_text(d / "feature_contrast.csv", contrast.to_csv(index=False)),
                atomic_write_parquet(d / "feature_redundancy.parquet", redund),
                atomic_write_text(d / "multiscale_analysis.csv", multi.to_csv(index=False)),
                atomic_write_text(d / "model_complementarity.csv", compl.to_csv(index=False)),
                atomic_write_text(d / "modeling_recommendations.csv", recs.to_csv(index=False)),
                atomic_write_parquet(d / "spatial_correlation.parquet", corr),
                atomic_write_parquet(d / "failure_taxonomy.parquet", failures)]
        figs = [EG.coverage_map(grid, valid, dem_cov, mask, None, d / "fig_coverage.png"),
                EG.contrast_forest(contrast, d / "fig_contrast.png"), EG.utility_heatmap(util, d / "fig_utility.png")]
        disagree = MEV.disagreement_map(maps, valid) if maps else None
        figs += self._candidate_atlas(reader, cand, d / "candidate_atlas", maps=maps, disagree=disagree)
        figs = [f for f in figs if f]
        outs += figs
        summary = ER.decision_report(contrast, util, redund, failures)
        summary += "\n## Modeling recommendation table (evidence-based)\n\n" + (
            md_table(recs) if len(recs) else "no evidence available") + "\n"
        outs.append(atomic_write_text(d / "eda_summary.md", summary))
        outs.append(ER.html_report("G11 multiresolution EDA", [
            {"title": "Model evidence (existing predictions, provenance-labelled)", "tables": {"complementarity": compl}},
            {"title": "Native-resolution multiscale contrast", "tables": {"native": native}},
            {"title": "Modeling recommendations", "tables": {"recommendations": recs}},
            {"title": "Coverage", "text": "DEM vs geophysics vs mapped faults (km, km^2).", "tables": {"coverage": cov},
             "figures": [d / "fig_coverage.png"]},
            {"title": "Fault contrast", "text": "Near-fault vs local control, 90% spatial block-bootstrap CI.",
             "tables": {"contrast": contrast.sort_values("near_vs_local_control", key=abs, ascending=False)
                        if len(contrast) else contrast}, "figures": [d / "fig_contrast.png"]},
            {"title": "Multiscale utility", "tables": {"utility": util[util.region == "all"]},
             "figures": [d / "fig_utility.png"]},
            {"title": "Redundancy", "tables": {"redundancy": redund}},
            {"title": "Spatial autocorrelation", "tables": {"decorrelation": corr}},
            {"title": "Failure taxonomy (descriptive)", "tables": {"modes": failures.dominant_mode.value_counts()
                                                                    .rename_axis("mode").reset_index() if len(failures)
                                                                    else failures}},
            {"title": "Decision report", "text": summary}], d / "eda_report.html"))
        stats = {"n_features": len(names), "n_contrast_significant": int(((contrast.ci_low > 0) | (contrast.ci_high < 0)).sum())
                 if len(contrast) else 0, "n_candidates_eda": int(len(cand)) if cand is not None else 0,
                 "figures": len(figs), "atlas_sites": len(list((d / "candidate_atlas").glob("site_*.png"))),
                 "external_maps_analysed": len(maps), "native_features_tested": int(len(native))}
        self.stages.record("eda", fp, outs, stats)
        return stats

    def _eda_candidates(self, reader):
        """Label-free evidence candidates (geophysical edge rank) for the taxonomy and the atlas."""
        from .eda.failure_modes import failure_scores
        from .features.segment import segment_features
        from .inference.candidate_generation import generate_candidates

        ev = self._get("geo_edge_rank")
        if ev is None:
            return pd.DataFrame(), None
        cand = generate_candidates(ev, self.grid.valid, self.grid, self.cfg, "geo_edge_rank")
        if not len(cand):
            return pd.DataFrame(), cand
        cand = cand.assign(_len=cand.geometry.length).sort_values("_len", ascending=False).drop(columns="_len")
        cand = cand.reset_index(drop=True).iloc[:2000]
        rasters = {"geo_edge_rank": ev, "terrain_scarp_rank": self._get("terrain_scarp_rank"),
                   "slope_rank": self._get("terrain_scarp_rank"), "dem_seam": None}
        feats = segment_features(cand, rasters, self.grid, self.cfg, dem_reader=None)
        return failure_scores(feats), cand

    def _atlas_sites(self, cand, maps, disagree) -> list:
        """>= ``eda.atlas_sites`` geographically diverse review sites of several kinds (greedy farthest-point)."""
        from shapely.geometry import LineString, Point

        from .grid import inside, pixel_to_world, world_to_pixel

        segs = self.segments()
        dist = np.asarray(self.store.get("Y_DISTANCE"), np.float32)
        cov = self.catalog()
        dem_u = cov.union_all() if cov is not None and hasattr(cov, "union_all") else (cov.unary_union if cov is not None else None)
        pool = []
        if segs is not None and len(segs):
            s = segs if dem_u is None else segs[segs.geometry.intersects(dem_u)]
            pool += [("mapped fault with DEM" if dem_u is not None else "mapped fault", g) for g in
                     (s if len(s) else segs).sort_values("length_m", ascending=False).geometry[:60]]
        if cand is not None and len(cand):
            for g in cand.geometry[:400]:
                c = g.centroid
                r, cc = world_to_pixel(c.x, c.y, self.grid)
                if inside(r, cc, self.grid) and dist[r, cc] > 2000:
                    pool.append(("unmapped high-evidence lineament (UNVERIFIED)", g))
        if disagree is not None and np.isfinite(disagree).any():
            thr = np.nanquantile(disagree, 0.995)
            rr, cc = np.nonzero(np.nan_to_num(disagree) >= thr)
            for i in np.linspace(0, len(rr) - 1, min(len(rr), 40)).astype(int):
                x, y = pixel_to_world(rr[i], cc[i], self.grid)
                pool.append(("model disagreement region", LineString([(x - 200, y), (x + 200, y)])))
        if maps and segs is not None and len(segs):
            m0 = next(iter(maps.values()))["array"]
            med = float(np.nanmedian(m0)) if np.isfinite(m0).any() else np.nan
            for g in segs.geometry[:400]:
                c = g.centroid
                r, cc = world_to_pixel(c.x, c.y, self.grid)
                if inside(r, cc, self.grid) and np.isfinite(m0[r, cc]) and m0[r, cc] < med:
                    pool.append(("mapped fault with low existing-model score", g))
        ter = self._get("terrain_scarp_rank")
        if ter is not None and cand is not None and len(cand):
            for g in cand.geometry[:400]:
                c = g.centroid
                r, cc = world_to_pixel(c.x, c.y, self.grid)
                if inside(r, cc, self.grid) and ter[r, cc] >= 0.9 and dist[r, cc] > 1000:
                    pool.append(("terrain-supported unmapped lineament (UNVERIFIED)", g))
        n = int(self.cfg["eda"]["atlas_sites"])
        chosen, pts = [], []
        while pool and len(chosen) < n:
            best, bi = -1.0, None
            for i, (k, g) in enumerate(pool):
                c = g.centroid
                dmin = min((c.distance(Point(p)) for p in pts), default=np.inf)
                bonus = 1e6 if k not in {kk for kk, _ in chosen} else 0.0          # every kind first
                if dmin + bonus > best:
                    best, bi = dmin + bonus, i
            k, g = pool.pop(bi)
            chosen.append((k, g))
            pts.append((g.centroid.x, g.centroid.y))
        return chosen

    def _candidate_atlas(self, reader, cand, out_dir, maps=None, disagree=None) -> list:
        from .eda import figures as EG
        from .features.structural import sample_cross_fault_profiles
        from .features.terrain import compute_terrain_features

        segs = self.segments()
        sites = self._atlas_sites(cand, maps, disagree)
        out = []
        half = self.cfg["eda"]["atlas_half_km"] * 1000
        geo = self._get("magnetic_gm_400m")
        ev = self._get("geo_edge_rank")
        for i, (kind, g) in enumerate(sites):
            c = g.centroid
            b = (c.x - half, c.y - half, c.x + half, c.y + half)
            site = {"title": f"{kind} @ E{c.x:.0f} N{c.y:.0f}", "faults_xy": None}
            r0, c0 = int((self.grid.transform[5] - b[3]) / self.grid.res[1]), int((b[0] - self.grid.transform[2]) / self.grid.res[0])
            n = int(2 * half / self.grid.res[0])
            r0, c0 = max(r0, 0), max(c0, 0)
            ext = (b[0] / 1e3, b[2] / 1e3, b[1] / 1e3, b[3] / 1e3)
            if geo is not None:
                site["geophysics"] = (geo[r0:r0 + n, c0:c0 + n], ext)
            if ev is not None:
                site["score"] = (ev[r0:r0 + n, c0:c0 + n], ext)
            if segs is not None and len(segs):
                from shapely.geometry import box

                bb = box(*b)
                site["faults_xy"] = [tuple(np.asarray(s.intersection(bb).coords).T) for s in segs.geometry
                                     if s.intersects(bb) and s.intersection(bb).geom_type == "LineString"]
            if reader is not None:
                try:
                    win = reader.read_dem_window(b, self.grid.crs, resolution=2.0, resampling="average")
                    if win.valid.mean() > 0.05:
                        tf = compute_terrain_features(win.elevation, win.valid, 2.0, [10], self.cfg,
                                                      families=("first_order", "second_order", "hillshade"))
                        site["hillshade"] = (tf.arrays.get("hillshade_315_10m"), ext)
                        site["slope"] = (tf.arrays.get("slope_10m"), ext)
                        site["curvature"] = (tf.arrays.get("profile_curvature_10m"), ext)
                        cf = self.cfg["feature_engineering"]["cross_fault"]
                        prof = sample_cross_fault_profiles(g, reader, cf["offsets_m"], cf["spacing_m"], self.grid.crs, 2.0)
                        site["profiles"] = (prof.offsets_m, prof.elevations)
                        site["tiles"] = ",".join(win.source_tiles)
                except Exception as ex:                                     # noqa: BLE001
                    site["title"] += f" (DEM unavailable: {type(ex).__name__})"
            out.append(EG.atlas_site(site, Path(out_dir) / f"site_{i:02d}.png"))
        return out

    # ================================================================== Phase 5: splits
    def splits(self) -> dict:
        fp = self.stages.fingerprint("splits")
        if self._cached("splits", fp):
            self.split_result()
            return self.stages.load("splits")["stats"]
        self._split = None
        res = self.split_result(check_manifest=False)          # fresh protocol; the manifest is rewritten below
        d = self.ctx.d("splits")
        ld = self.store.reg[self.store.reg.label_derived.astype(bool)].name.tolist()
        checks = SS.run_leakage_checks(res, self.grid, self.store.predictors(), ld)
        outs = SS.save_splits(res, self.grid, d, self.ctx)
        outs.append(atomic_write_parquet(d / "leakage_checks.parquet", checks))
        outs.append(self.ctx.write_manifest(d / "leakage_report.json", {
            "passed": bool(checks.passed.all()), "checks": checks.to_dict("records"), "embargo_m": res.embargo_m,
            "embargo_components": res.embargo_parts, "training_stops_if_failed": True}))
        if self.cfg.get("reporting", {}).get("figures", True):
            from .eda.figures import split_map

            outs.append(split_map(self.grid, res.code, self.mask, d / "split_map.png"))
        if not checks.passed.all():
            raise SS.LeakageError(f"leakage checks failed:\n{checks[~checks.passed].to_string()}")
        stats = {"embargo_m": res.embargo_m, "pixels": {k: int(res.mask(k).sum()) for k in list(SS.SPLITS) + ["embargo"]}}
        self.stages.record("splits", fp, outs, stats)
        self.log(f"[splits] {stats}")
        return stats

    # ================================================================== Phase 6-7: training
    def train(self, experiments=None) -> pd.DataFrame:
        """Train the experiments sequentially (never concurrently); each one is its own resumable stage."""
        from .experiments import EXPERIMENTS

        cfg, grid, store = self.cfg, self.grid, self.store
        res = self.split_result()
        tgt_name = cfg["labels"].get("training_target", "Y_BUFFER_200M")
        target = np.asarray(store.get(tgt_name), np.float32)
        dist = np.asarray(store.get("Y_DISTANCE"), np.float32)
        hard_geo, hard_ter = self._get("geo_edge_rank"), self._get("terrain_scarp_rank")
        rows = []
        default = [e for e in ("E0", "E1", "E2", "E3", "E4") if e in EXPERIMENTS]
        exps = self.expand_experiments(default if experiments is None else list(experiments))
        for exp in exps:
            spec = EXPERIMENTS[exp.split("__")[0]]
            mdir = self.model_dir(exp)
            fp = self.stages.fingerprint(f"train_{exp}", self.stages.load("features")["fingerprint"], tgt_name)
            if spec.get("requires_dem") and not any(n.startswith("dem100_") for n in store.reg.name):
                rows.append({"experiment": exp, "status": "SKIPPED: no DEM features"})
                continue
            if self._cached(f"train_{exp}", fp) and self.pred_path(exp).is_file():
                rows.append({"experiment": exp, "status": "CACHED", **self.stages.load(f"train_{exp}")["stats"]})
                continue
            if spec["kind"] in ("patch", "multires"):
                try:
                    import torch  # noqa: F401
                except ImportError:
                    rows.append({"experiment": exp, "status": "SKIPPED: torch not installed"})
                    continue
            self.stages.begin(f"train_{exp}", fp)
            try:
                with self.tracker.track(f"train_{exp}", self.log) as trk:
                    pred, stats, outs = self._train_one(exp, spec, res, target, tgt_name, dist, hard_geo, hard_ter, mdir)
                    t1 = time.time()
                    outs.append(write_geotiff(self.pred_path(exp), pred, grid, [f"{exp} probability"]))
                    stats["inference_write_seconds"] = round(time.time() - t1, 1)
            except Exception as e:
                self.stages.fail(f"train_{exp}", fp, e)
                optional = "__" in exp and self.cfg.get("reporting", {}).get("failure_policy",
                                                                              "STOP_REQUIRED_SKIP_OPTIONAL") != "STOP_ON_ANY_FAILURE"
                if optional:                                   # ablation variants are optional: record and continue
                    rows.append({"experiment": exp, "status": f"SKIPPED (optional variant failed): {type(e).__name__}: {e}"})
                    continue
                raise
            stats.update(train_seconds=trk.get("seconds"), peak_ram_gb=trk.get("peak_ram_gb"),
                         peak_gpu_gb=trk.get("peak_gpu_gb"))
            outs.append(atomic_write_json(mdir / "metrics.json", stats))
            outs.append(self.ctx.write_manifest(mdir / "checkpoint_metadata.json", {
                "experiment": exp, "spec": spec, "target": tgt_name, "split_manifest": str(self.ctx.d("splits") /
                                                                                          "split_manifest.json"),
                "selection": "validation split only", **stats}))
            self.stages.record(f"train_{exp}", fp, outs, stats)
            rows.append({"experiment": exp, "status": "TRAINED", **stats})
        t = pd.DataFrame(rows)
        atomic_write_parquet(self.ctx.d("train") / "training_summary.parquet", t)
        return t

    def expand_experiments(self, exps) -> list[str]:
        """Add the configured ablation variants: E3__<loss> (training.loss_ablation), E3__finetune (compatible
        pretrained weights, models.segformer.pretrained_path), E4__<mode> (models.gated_fusion.mode_ablation)."""
        out = []
        t, m = self.cfg["training"], self.cfg["models"]
        for e in exps:
            out.append(e)
            if e == "E3":
                out += [f"E3__{x}" for x in t.get("loss_ablation", []) if x != t["loss"]]
                if m["segformer"].get("pretrained_path"):
                    out.append("E3__finetune")
            if e == "E4":
                out += [f"E4__{x}" for x in m["gated_fusion"].get("mode_ablation", []) if x != m["gated_fusion"]["mode"]]
        return list(dict.fromkeys(out))

    def _train_one(self, exp, spec, res, target, tgt_name, dist, hard_geo, hard_ter, mdir):
        from .datasets.pixel_dataset import GeoPixelDataset
        from .experiments import feature_selector
        from .models.xgb_baseline import predict_full_grid, train_pixel_model

        cfg, grid, store = self.cfg, self.grid, self.store
        t0 = time.time()
        if spec["kind"] == "pixel":
            feats = feature_selector(store, exp)
            tr = GeoPixelDataset.build(store, feats, target, tgt_name, res.code, "train", self.mask, dist, cfg,
                                       hard_geo, hard_ter)
            va = GeoPixelDataset.build(store, feats, target, tgt_name, res.code, "validation", self.mask, dist, cfg,
                                       hard_geo, hard_ter, n=max(len(tr.y) // 3, 1000))
            model = train_pixel_model(tr, va, cfg["models"]["xgb"], cfg["reproducibility"]["seed"])
            outs = model.save(mdir)
            imp = model.importance()
            outs.append(atomic_write_parquet(mdir / "feature_importance.parquet", imp))
            hist = model.history.get("validation", {})
            vap = hist.get("aucpr") or hist.get("average_precision") or []
            pred = predict_full_grid(model, store, grid.valid)
            stats = {"n_train": int(len(tr.y)), "n_val": int(len(va.y)), "prevalence_train": float(tr.y.mean()),
                     "n_features": len(feats), "backend": model.backend, "fit_seconds": round(time.time() - t0, 1),
                     "validation_aucpr_best": float(max(vap)) if vap else None}
            return pred, stats, outs
        return self._train_deep(exp, res, target, dist, mdir)

    def _train_deep(self, exp, res, target, dist, mdir):
        import torch

        from .datasets.patch_dataset import GeoPatchDataset, MultiresolutionPatchDataset, collate, patch_windows
        from .inference.blending import predict_tiled
        from .models.gated_fusion import GatedFusionNet
        from .models.segformer import SegFormer
        from .training.losses import build_loss
        from .training.sampling import category_masks, mixture_sample
        from .training.trainer import Trainer, pixel_eval_fn, seed_everything

        cfg, grid, store = self.cfg, self.grid, self.store
        base, _, variant = exp.partition("__")
        seed_everything(cfg["reproducibility"]["seed"], cfg["reproducibility"]["deterministic"])
        imp_p = self.model_dir("E1") / "feature_importance.parquet"
        k = int(cfg["models"]["segformer"].get("in_channels") or 16)
        channels = pd.read_parquet(imp_p).feature.tolist()[:k] if imp_p.is_file() else store.predictors(["geo_magnetic",
                                                                                                         "geo_gravity"])[:k]
        staging = {}
        try:                                                  # read patches from local scratch, not from Drive
            staging = store.localize(channels, Path(cfg["scratch_root"]) / "feature_cache")
        except OSError as e:
            self.log(f"[{exp}] local staging unavailable ({e}); reading features from Drive")
        P = int(cfg["sampling"]["patch_px"])
        tr_x = self._patch_X(channels)
        norm = SS.TrainOnlyNormalizer()
        rr, cc = np.nonzero(res.mask("train"))
        sel = np.random.default_rng(0).choice(rr.size, min(rr.size, 200_000), replace=False)
        norm.fit(np.stack([tr_x[i][rr[sel], cc[sel]] for i in range(len(channels))], 1),
                 np.full(sel.size, SS.CODES["train"]))
        norm.center, norm.scale = np.nan_to_num(norm.center), np.where(np.isfinite(norm.scale), norm.scale, 1.0)
        loaders = {}
        for split in ("train", "validation"):
            cats = category_masks(res.mask(split), self.mask, dist, cfg["labels"]["near_band_m"])
            n_p = int(cfg["sampling"].get(f"patches_{split}", 4000 if split == "train" else 1500))
            idx, _, _ = mixture_sample(cats, cfg["sampling"]["mix"], n_p, 100, cfg["reproducibility"]["seed"])
            r, c = np.unravel_index(idx, grid.shape)
            win = patch_windows(r, c, P, grid.shape, res.code, split)
            if not len(win):
                raise StageGateError(
                    f"{exp}: no {P}-px patch lies entirely inside the {split} split (patches never span splits). "
                    f"Use spatial_splits.block_km >= 2 x patch footprint ({2 * P * grid.res[0] / 1000:g} km) or a "
                    "smaller sampling.patch_px")
            if base == "E4":
                te = cfg["models"]["terrain_encoder"]
                ds = MultiresolutionPatchDataset(store, channels, target, win, res.code, split, norm, distance=dist,
                                                 dem_reader=self._dem_mosaic(float(te.get("dem_res_m", 25.0))), grid=grid,
                                                 dem_res_m=float(te.get("dem_res_m", 25.0)), dem_px=int(te.get("dem_px", 256)))
            else:
                ds = GeoPatchDataset(store, channels, target, win, res.code, split, norm, distance=dist)
            loaders[split] = torch.utils.data.DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                                                         shuffle=split == "train", collate_fn=collate, num_workers=0)
        loss_name, extra = cfg["training"]["loss"], {}
        if base == "E3":
            s = cfg["models"]["segformer"]
            model = SegFormer(len(channels), s["dims"], s["depths"], s["heads"], decoder_dim=s["decoder_dim"],
                              drop_path=s["drop_path"])
            if variant == "finetune":
                from .models.segformer import load_compatible_weights

                rep = load_compatible_weights(model, str(s["pretrained_path"]))
                extra["pretrained_coverage"] = rep["coverage"]
                if rep["coverage"] == 0:
                    raise StageGateError(f"{exp}: pretrained weights {s['pretrained_path']} share no tensor with the "
                                         "G11 SegFormer (incompatible architecture) - remove pretrained_path")
            elif variant:
                loss_name = variant
        else:
            gf = cfg["models"]["gated_fusion"]
            model = GatedFusionNet(len(channels), variant or gf["mode"], gf["geo_dim"], gf["dem_dim"], aux=False)
        tr = Trainer(model, build_loss(loss_name, cfg["training"]["pos_weight"]), cfg, mdir, evaluate_fn=pixel_eval_fn)
        hist = tr.fit(loaders["train"], loaders["validation"], meta={"channels": channels, "experiment": exp,
                                                                     "loss": loss_name})
        best = mdir / "best.pt"
        if best.is_file():
            model.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["model"])
        model.to(tr.device).eval()
        X = np.stack([np.nan_to_num(((tr_x[i] - norm.center[i]) / norm.scale[i]).astype(np.float32))
                      for i in range(len(channels))])

        @torch.no_grad()
        def fn(xb):
            t = torch.from_numpy(xb).to(tr.device)
            o = model(t) if base == "E3" else model(t)["logits"]
            return torch.sigmoid(o)[:, 0].float().cpu().numpy()

        pred = predict_tiled(fn, X, grid.valid, cfg["inference"]["tile_px"], cfg["inference"]["overlap_px"],
                             cfg["inference"]["batch_size"])
        outs = [mdir / "history.csv", best] + [atomic_write_json(mdir / "feature_schema.json",
                                                                {"channels": channels, "normalizer_fit": "train split"})]
        return pred, {"epochs": int(len(hist)), "channels": len(channels), "best_val": float(hist.val_metric.max())
                      if len(hist) else None, "dem_branch": base == "E4", "loss": loss_name,
                      "fusion_mode": (variant or cfg["models"]["gated_fusion"]["mode"]) if base == "E4" else None,
                      "locally_staged_channels": len(staging), "oom_batches": tr.oom, **extra}, \
            [o for o in outs if Path(o).is_file()]

    def _dem_mosaic(self, res_m: float):
        """Local (scratch) DEM mosaic at ``res_m`` for E4 patches, rebuilt only when the DEM catalog changes."""
        from .dem_catalog import coverage_100m
        from .dem_reader import MosaicDEMReader, build_dem_mosaic

        reader = self.dem_reader()
        if reader is None or not len(reader.index.cat):
            return None
        key = self.ctx.hashes.sha(self.ctx.d("ingest") / "dem_catalog.parquet")[:12]
        p = Path(self.cfg["scratch_root"]) / "dem_mosaic" / f"dem_{res_m:g}m_{key}.tif"
        if not p.is_file():
            cov = coverage_100m(reader.index.cat, self.grid)["covered"] & self.grid.valid
            build_dem_mosaic(reader, self.grid, cov, res_m, p, int(self.cfg["dem"]["aggregate_chunk_cells"]), self.log)
        return MosaicDEMReader(p)

    def _patch_X(self, channels):
        return [np.asarray(self.store.get(c), np.float32) for c in channels]

    # ================================================================== Phase 9: evaluation
    def fusion(self) -> pd.DataFrame:
        """Pixel-level fusion of the G11 experiment maps: fitted on STACK, selected on VALIDATION (E5)."""
        from .eda.spatial import block_sample
        from .features.multiscale import pct_rank
        from .fusion import compare

        exps = [e for e in self.available_predictions() if not e.startswith("E5")]
        if len(exps) < 2:
            self.log(f"[fusion] SKIPPED: {len(exps)} member map(s) (need >= 2)")
            return pd.DataFrame([{"method": "-", "status": f"SKIPPED: {len(exps)} member(s)"}])
        fp = self.stages.fingerprint("fusion", [(e, self.ctx.hashes.sha(self.pred_path(e))) for e in exps])
        if self._cached("fusion", fp) and self.pred_path("E5").is_file():
            return pd.read_csv(self.ctx.d("evaluate") / "fusion_comparison.csv")
        self.stages.begin("fusion", fp)
        res, grid = self.split_result(), self.grid
        members = {e: _read_tif(self.pred_path(e)) for e in exps}
        cond = {}
        dv = self._get("dem100_valid_fraction")
        if dv is not None:
            cond["dem_available"] = np.nan_to_num(dv, nan=0.0)
            rough = self._get("dem100_roughness_mean")
            if rough is not None:
                cond["terrain_roughness_rank"] = np.nan_to_num(pct_rank(rough, grid.valid), nan=0.5)
        seed = self.cfg["reproducibility"]["seed"]
        rs, cs, _ = block_sample(res.mask("stack"), 2, seed, 200_000)
        rv, cv, _ = block_sample(res.mask("validation"), 2, seed + 1, 200_000)
        si, vi = np.ravel_multi_index((rs, cs), grid.shape), np.ravel_multi_index((rv, cv), grid.shape)
        table, fit, maps = compare(members, self.mask.astype(int), grid.valid, si, vi, cond or None)
        ok = table[table.status == "OK"].sort_values("validation_pr_auc", ascending=False)
        if not len(ok):
            raise StageGateError("fusion: no method could be scored on validation")
        best = ok.iloc[0].method
        table["selected"] = table.method == best
        table["members"] = ",".join(exps)
        table["fit_split"], table["selection_split"] = "stack", "validation"
        outs = [write_geotiff(self.pred_path("E5"), maps[best], grid, [f"E5 fusion ({best}) of {','.join(exps)}"]),
                atomic_write_text(self.ctx.d("evaluate") / "fusion_comparison.csv", table.to_csv(index=False))]
        md = self.model_dir("E5")
        outs.append(atomic_write_json(md / "metrics.json", {"selected_method": best, "members": exps,
                                                            "stack_ap": fit.get("stack_ap"),
                                                            "best_single_on_stack": fit.get("best_single")}))
        self.stages.record("fusion", fp, outs, {"selected": best, "members": exps})
        self.log(f"[fusion] selected {best} on validation among {list(ok.method)}")
        return table

    def evaluate(self, final: bool = False) -> pd.DataFrame:
        """Consolidated leaderboard on IDENTICAL support (block-sampled validation pixels, same label definition)."""
        from .eda.spatial import block_sample
        from .evaluation.official_adapter import proxy_dti
        from .evaluation.pixel_metrics import evaluate_region
        from .evaluation.segment_metrics import candidate_metrics
        from .experiments import EXPERIMENTS
        from .inference.candidate_generation import generate_candidates
        from .inference.ranking import select_by_length_budget
        from .training.calibration import Calibrator, reliability

        cfg, grid = self.cfg, self.grid
        res = self.split_result()
        d = self.ctx.d("evaluate")
        rows = []
        truth = self.mask
        tgt = cfg["labels"].get("training_target", "Y_BUFFER_200M")
        bpx = max(int(cfg["evaluation"]["bootstrap_block_km"] * 1000 / grid.res[0]), 1)
        exps = self.available_predictions()
        if not exps:
            raise StageGateError("no trained experiment to evaluate")
        if final and not (d / "selection.json").is_file():
            raise StageGateError("test metrics require a prior validation-based selection (selection.json)")
        budgets = cfg["ranking"]["length_budgets_km"]
        mid_budget = float(sorted(budgets)[len(budgets) // 2])
        for exp in exps:
            p = _read_tif(self.pred_path(exp))
            base = exp.split("__")[0]
            spec = EXPERIMENTS.get(base, {"model": "fusion", "geophysics": "fusion of G11 maps", "dem": "-"})
            met = read_json(self.model_dir(exp) / "metrics.json") or {}
            for split in (["validation"] + (["test"] if final else [])):
                reg = res.mask(split)
                r, c, _ = block_sample(reg, 2, cfg["reproducibility"]["seed"], 200_000)
                m = evaluate_region(p, truth, reg, r, c, (r // bpx) * 10000 + c // bpx, cfg)
                dti = proxy_dti(p, truth, reg)
                cand = generate_candidates(np.where(reg, p, np.nan), reg, grid, cfg, exp)
                cmx = {}
                if len(cand):
                    from .features.segment import raster_attributes

                    share = reg.sum() / max(grid.valid.sum(), 1)
                    sc = np.array([raster_attributes(g, {"prob": p}, grid, grid.res[0] / 2).get("prob_mean", np.nan)
                                   for g in cand.geometry])
                    sel = select_by_length_budget(sc, cand.geometry.length.to_numpy(), mid_budget * share)
                    cmx = candidate_metrics(cand[sel], truth, reg, grid, cfg["evaluation"]["pixel_tolerance_m"])
                rows.append({"experiment_id": exp, "model": spec.get("model"), "features": spec.get("geophysics"),
                             "dem_inputs": spec.get("dem"), "target_definition": tgt, "train_region": "train",
                             "validation_region": "validation", "test_region": "test" if final else "not evaluated",
                             "split": split, "status": "EVALUATED", "PR_AUC": m.get("pr_auc"), "ROC_AUC": m.get("roc_auc"),
                             "PR_AUC_ci_low": m.get("pr_auc_ci_low"), "PR_AUC_ci_high": m.get("pr_auc_ci_high"),
                             "calibration_error": m.get("ece"),
                             **{k: v for k, v in m.items() if k.startswith(("recall_at", "precision_at"))},
                             "candidate_recall": cmx.get("length_recall"), "candidate_precision": cmx.get("length_precision"),
                             "predicted_length_km": cmx.get("predicted_km"), "fragmentation": cmx.get("fragmentation"),
                             "candidate_budget_km": mid_budget, "offline_DTI_proxy": dti["dti"],
                             "offline_DTI_label": dti["label"], "training_time_s": met.get("train_seconds"),
                             "inference_time_s": met.get("inference_write_seconds"),
                             "peak_GPU_memory_gb": met.get("peak_gpu_gb"), "peak_RAM_gb": met.get("peak_ram_gb"),
                             "n_samples": m.get("n_samples"), "prevalence": m.get("prevalence")})
        t = pd.DataFrame(rows)
        ev = t[(t.split == "validation") & (t.status == "EVALUATED")]
        if not final:
            best = ev.sort_values(["PR_AUC", "recall_at_0.01"], ascending=False).iloc[0]
            best = best.rename({"experiment_id": "experiment", "PR_AUC": "pr_auc"})
            p = _read_tif(self.pred_path(best.experiment))
            cal_rows = []
            for meth in self.cfg["calibration"]["methods"]:
                st = res.mask(self.cfg["calibration"]["region"])
                r, c, _ = block_sample(st, 2, 1, 100_000)
                cal = Calibrator(meth).fit(p[r, c], truth[r, c].astype(float), self.cfg["calibration"]["region"],
                                           self.cfg["calibration"]["region"])
                vr, vc, _ = block_sample(res.mask("validation"), 2, 2, 100_000)
                _, ece = reliability(cal.transform(p[vr, vc]), truth[vr, vc].astype(float))
                cal_rows.append({"method": meth, "validation_ece": ece})
            cal_t = pd.DataFrame(cal_rows)
            chosen = cal_t.sort_values("validation_ece").iloc[0].method
            self.ctx.write_manifest(d / "selection.json", {
                "selected_experiment": best.experiment, "criterion": "max validation PR-AUC, tie recall@1% area",
                "validation_pr_auc": best.pr_auc, "calibration": chosen, "calibration_table": cal_t.to_dict("records"),
                "selected_before_test": True})
        out = d / ("experiment_matrix_final.parquet" if final else "experiment_matrix.parquet")
        atomic_write_parquet(out, t)
        lb = atomic_write_text(d / ("leaderboard_final.csv" if final else "leaderboard.csv"), t.to_csv(index=False))
        self.stages.record("evaluate" + ("_final" if final else ""), self.stages.fingerprint("evaluate", final), [out, lb],
                           {"n_rows": len(t), "experiments": exps})
        return t

    def _selected_probability(self):
        from .eda.spatial import block_sample
        from .training.calibration import Calibrator

        sel = read_json(self.ctx.d("evaluate") / "selection.json")
        if sel is None:
            raise StageGateError("run EVALUATE first (no selection.json)")
        p = _read_tif(self.pred_path(sel["selected_experiment"]))
        res = self.split_result()
        reg = res.mask(self.cfg["calibration"]["region"])
        r, c, _ = block_sample(reg, 2, 1, 100_000)
        cal = Calibrator(sel["calibration"]).fit(p[r, c], self.mask[r, c].astype(float), self.cfg["calibration"]["region"],
                                                 self.cfg["calibration"]["region"])
        return np.where(self.grid.valid, cal.transform(p), np.nan).astype(np.float32), sel

    # ================================================================== Phase 8: candidate recovery
    def infer(self) -> dict:
        from .datasets.segment_dataset import FaultSegmentDataset
        from .evaluation.official_adapter import proxy_dti
        from .features.segment import segment_features
        from .inference.candidate_generation import generate_candidates
        from .inference.export import rasterize_segments
        from .inference.ranking import select_by_length_budget
        from .inference.topology import deduplicate, link_segments, stable_ids
        from .models.segment_ranker import train_ranker

        cfg, grid = self.cfg, self.grid
        cg = cfg["candidate_generation"]
        d = self.ctx.d("infer")
        p, sel = self._selected_probability()
        parts = [generate_candidates(p, grid.valid, grid, cfg, sel["selected_experiment"])]
        ev = self._get("geo_edge_rank")
        if ev is not None:                                   # generator 9: label-free geophysical lineaments
            parts.append(generate_candidates(ev, grid.valid, grid, cfg, "geo_edge_rank"))
        import geopandas as gpd

        parts = [x for x in parts if len(x)]
        if not parts:
            raise StageGateError("no candidate segment generated")
        cand = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=grid.crs)
        cand = deduplicate(link_segments(cand, cg["link_radius_m"], cg["link_max_angle_deg"]), cg["dedup_overlap"],
                           cg["dedup_angle_deg"])
        cand["candidate_id"] = stable_ids(cand)
        rasters = {"prob_selected": p, "geo_edge_rank": self._get("geo_edge_rank"),
                   "terrain_scarp_rank": self._get("terrain_scarp_rank"), "magrav_agree": self._get("magrav_orient_agree_400m")}
        for exp in [e for e in self.available_predictions() if "__" not in e][:6]:
            if exp != sel["selected_experiment"]:
                rasters[f"prob_{exp}"] = _read_tif(self.pred_path(exp))
        feats = segment_features(cand, rasters, grid, cfg, self.dem_reader())
        ov = FL.segment_overlap_labels(cand, self.mask, grid, cfg["ranking"]["label_overlap_m"])
        res = self.split_result()
        ds = FaultSegmentDataset.build(feats, cand, ov, res.code, res.groups, grid)
        tr, va = ds.split("train"), ds.split("validation")
        if tr.label.nunique() < 2 or va.label.nunique() < 2:
            raise StageGateError("ranker needs positive and unlabeled candidates in train and validation: "
                                 f"train {tr.label.value_counts().to_dict()}, validation {va.label.value_counts().to_dict()}"
                                 " - lower candidate_generation quantiles or check the split")
        rankers, val_rows = {}, []
        for obj in ("binary", "lambdarank"):
            rk = train_ranker(tr, va, ds.feature_names, obj, cfg["models"]["segment_ranker"], cfg["reproducibility"]["seed"])
            rankers[obj] = rk
            from .evaluation.pixel_metrics import pr_auc

            val_rows.append({"objective": obj, "validation_ap": pr_auc(va.label.to_numpy(), rk.score(va))})
        vt = pd.DataFrame(val_rows)
        best_obj = vt.sort_values("validation_ap", ascending=False).iloc[0].objective
        rk = rankers[best_obj]
        outs = rk.save(self.ctx.d("train", "segment_ranker"))
        score = rk.score(ds.table)
        s = (score - score.min()) / max(score.max() - score.min(), 1e-12) if best_obj != "binary" else score
        ds.table["score"] = s
        # budget choice on VALIDATION: budget scaled by the validation share of the eligible area
        vmask = res.mask("validation")
        share = vmask.sum() / max(grid.valid.sum(), 1)
        brows = []
        vi = ds.table.split == "validation"
        geoms_all = cand.iloc[ds.table.segment_id.to_numpy()]
        for b in cfg["ranking"]["length_budgets_km"]:
            selv = select_by_length_budget(s[vi.to_numpy()], ds.table[vi].length_m.to_numpy(), b * share)
            gsel = geoms_all[vi.to_numpy()][selv]
            r = rasterize_segments(gsel, s[vi.to_numpy()][selv], grid, vmask)
            dd = proxy_dti(np.nan_to_num(r), self.mask, vmask)
            brows.append({"budget_km": b, "validation_budget_km": b * share, "n_selected": int(selv.sum()),
                          "validation_proxy_dti": dd["dti"], "label": dd["label"]})
        bt = pd.DataFrame(brows)
        best_budget = float(bt.sort_values("validation_proxy_dti", ascending=False).iloc[0].budget_km)
        out_c = cand.iloc[ds.table.segment_id.to_numpy()].copy()
        out_c["score"], out_c["split"], out_c["label_overlap"] = s, ds.table.split.to_numpy(), ds.table.overlap_fraction.to_numpy()
        out_c["status"] = "UNVERIFIED candidate fault segment"
        gp = d / "candidates.gpkg"
        tmp = d / "candidates.partial.gpkg"
        out_c.to_file(tmp, driver="GPKG", layer="candidates")
        tmp.replace(gp)
        outs += [gp, atomic_write_parquet(d / "candidate_features.parquet", ds.table),
                 atomic_write_parquet(d / "ranker_validation.parquet", vt),
                 atomic_write_parquet(d / "budget_validation.parquet", bt)]
        man = {"selected_experiment": sel["selected_experiment"], "ranker_objective": best_obj, "budget_km": best_budget,
               "n_candidates": int(len(out_c)), "budget_selected_on": "validation proxy DTI (offline)"}
        outs.append(self.ctx.write_manifest(d / "inference_manifest.json", man))
        self.stages.record("infer", self.stages.fingerprint("infer", man), outs, man)
        return man

    # ================================================================== final report
    def report(self, plan=None, runtime_info=None) -> dict:
        from .colab_runtime import save_runtime_report
        from .reports import build_final_report

        save_runtime_report(self.ctx.d("runtime") / "runtime_report.json", runtime_info, None, self.tracker)
        return build_final_report(self, plan, runtime_info)

    # ================================================================== Phase 10: submission
    def submit(self) -> dict:
        from .evaluation.official_adapter import (OfficialEvaluatorUnavailable, export_submission, load_official_schema,
                                                  run_official_evaluator)
        from .inference.export import rasterize_segments
        from .inference.ranking import select_by_length_budget

        import geopandas as gpd

        man = read_json(self.ctx.d("infer") / "inference_manifest.json")
        if man is None:
            raise StageGateError("run INFER first")
        schema = load_official_schema(self.found["template"])
        cand = gpd.read_file(self.ctx.d("infer") / "candidates.gpkg")
        selm = select_by_length_budget(cand.score.to_numpy(), cand.geometry.length.to_numpy(), man["budget_km"])
        r = rasterize_segments(cand[selm], cand.score.to_numpy()[selm], self.grid, self.grid.valid)
        full = np.zeros(schema.shape, np.float32)
        if self.smoke:                                 # embed the subset into the full template (outside = 0)
            r0, r1, c0, c1 = self.smoke_window
            full[r0:r1, c0:c1] = np.nan_to_num(r)
        else:
            if r.shape != tuple(schema.shape):
                raise StageGateError(f"prediction grid {r.shape} != template {schema.shape}")
            full = np.nan_to_num(r)
        name = "g11_smoke_submission_NOT_FOR_UPLOAD.tif" if self.smoke else "g11_best_submission.tif"
        sd = self.ctx.d("submit")
        out = sd / name
        meta = {**man, "code_commit": self.ctx.commit, "config_hash": self.ctx.cfg_hash, "n_selected": int(selm.sum()),
                "selected_km": float(cand[selm].geometry.length.sum() / 1000), "smoke_test": self.smoke,
                "candidate_source": str(self.ctx.d("infer") / "candidates.gpkg"),
                "length_budget_km": man["budget_km"], "ranker_objective": man.get("ranker_objective"),
                "selected_model": man.get("selected_experiment")}
        exp = export_submission(np.where(schema.footprint, full, np.nan), schema, out, meta)
        try:
            official = run_official_evaluator(out, self.cfg)
        except OfficialEvaluatorUnavailable as e:
            official = {"official": False, "status": "OFFICIAL EVALUATOR UNAVAILABLE", "official_DTI": "unavailable",
                        "reason": str(e)}
        rec = {**exp, "official_evaluation": official, "official_DTI": official.get("official_DTI", official.get("dti")),
               "uploaded": False, "smoke_test": self.smoke}
        outs = [out, self.ctx.write_manifest(sd / "submission_manifest.json", {**meta, "path": str(out),
                                                                             "sha256": exp["sha256"]}),
                self.ctx.write_manifest(sd / "submission_validation.json", {
                    "path": str(out), "validation": exp["validation"], "schema": schema.as_dict(),
                    "official_schema_compatible": bool(exp["validation"]["passed"]), "official_evaluation": official}),
                atomic_write_text(sd / "submission_sha256.txt", f"{exp['sha256']}  {out.name}\n")]
        self.ctx.write_manifest(sd / "submission_record.json", rec)
        self.stages.record("submit", self.stages.fingerprint("submit", man), outs, {"official": official.get("status")},
                           validation=exp["validation"])
        return rec


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    """Plain Markdown table (no optional dependency)."""
    def cell(v):
        if isinstance(v, float):
            return floatfmt.format(v) if np.isfinite(v) else ""
        return str(v).replace("|", "/").replace("\n", " ")

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _audit_extra_md(units: pd.DataFrame, comp: pd.DataFrame, summ: dict) -> str:
    L = ["", "## DEM vertical units (authoritative metadata only)", ""]
    if len(units):
        L += [f"- {k}: {v} tile(s)" for k, v in units["class"].value_counts().items()]
        L.append("- UNKNOWN / CONFLICTING tiles are excluded from unit-sensitive features (see dem_vertical_units.csv)")
    else:
        L.append("- no DEM tiles found")
    L += ["", "## Existing model predictions", ""]
    if len(comp):
        L += [f"- {r.family} / {r.name}: {r.status} (metadata: {getattr(r, 'metadata_source', '?')}, "
              f"provenance {r.provenance})" for r in comp.itertuples()]
    else:
        L.append("- none discovered")
    if summ.get("smoke_window"):
        L += ["", "## SMOKE_TEST window", "", f"- {summ['smoke_window']}"]
    return "\n".join(L) + "\n"


def _read_tif(path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read(1, masked=True).astype(np.float32).filled(np.nan)


def dependency_graph(summary: dict) -> str:
    f = summary["found"]
    opt = summary.get("optional", {})

    def st(x):
        return "FOUND" if x else "ABSENT"

    return "\n".join([
        "# G11 dependency graph (generated by the audit)", "", "```mermaid", "flowchart LR",
        f"  GEO[GeoDAWN 100 m: {st(f.get('geodawn'))}] --> F[features]",
        f"  FR[fault raster: {st(f.get('fault_raster'))}] --> I[ingest]",
        f"  FV[fault vector: {st(f.get('fault_vector'))}] -.-> I",
        f"  DEM[1 m DEM tiles: {summary.get('dem_tiles', 0)} in {len(f.get('dem_dirs') or [])} folder(s), "
        f"units {' '.join(f'{k}={v}' for k, v in summary.get('dem_vertical_units', {}).items()) or 'n/a'}] -.-> I",
        f"  G06[G06 store: {opt.get('g06_store', 0)} file(s)] -.-> F",
        f"  G07[G07 arrays / rasters: {opt.get('g07_arrays', 0) + opt.get('g07_prediction', 0)} file(s), "
        f"{summary.get('external_compatible', 0)} compatible] -.-> E[EDA]",
        f"  TPL[template: {st(f.get('template'))}] --> S[submit]",
        "  I --> F --> E", "  F --> SP[splits] --> T[train] --> V[evaluate] --> N[infer] --> S", "```", "",
        "Dashed = optional (skipped and recorded when absent). Stage records: manifests/<stage>.json.", ""])
