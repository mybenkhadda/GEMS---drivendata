"""G11 orchestrator - recovery-first: features are processed once, saved, verified and reused by every experiment.

    P = G11Pipeline(load_config(overrides={"data_root": "/content/drive/MyDrive/GEMS_2"}))
    P.audit(); P.recover(); P.process_features(); P.finalize_features(); P.validate_features()
    P.eda(); P.splits(); P.train(["E0", "E1", "E2", "E3"]); P.predict_all(); P.fusion(); P.evaluate()
    P.infer(); P.evaluate(final=True); P.submit(); P.report()

Feature identity is a per-dependency-class specification (:mod:`feature_versions`), not the hash of the whole
config: split, loss, training, sampling and model settings never invalidate features. ``features.policy``:

* ``REUSE_VERIFIED``    (default) load the finalized version when it is complete and its spec matches; otherwise
                        recover the existing layers, compute ONLY the missing ones, finalize and validate;
* ``RESUME_MISSING``    full re-verification (every layer re-read) and repair of anything invalid, then finalize;
* ``FORCE_NEW_VERSION`` compute every layer again into a NEW version (refused if the version already exists).

Downstream artifacts (models, predictions, evaluation, candidates, submission, report) live in a run namespace
``<feature_version>__split_<split_version>`` so a new split or feature version never overwrites earlier results.
Integrity gates: audit fail-fast; split gates (no empty partition, positives, independent blocks, embargo, patch
feasibility, small-fault-system leakage, no target-derived predictor); test metrics only after a validation-based
selection; submission only after template validation; never uploaded.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit as A
from . import fault_labels as FL
from . import spatial_splits as SS
from .config import MODE_STAGES, ConfigError, Mode, stable_hash
from .features import FeatureStore, write_geotiff
from .grid import load_master_grid, validate_alignment
from .registry import (RunContext, StageStore, atomic_write_json, atomic_write_parquet, atomic_write_text, read_json)

FEATURE_POLICIES = ("REUSE_VERIFIED", "RESUME_MISSING", "FORCE_NEW_VERSION")
DOWNSTREAM = {"train", "predictions", "infer", "evaluate", "submit", "reports"}


class StageGateError(RuntimeError):
    pass


def _section(cfg: dict, dotted: str):
    v = cfg
    for k in dotted.split("."):
        if not isinstance(v, dict) or k not in v:
            return None
        v = v[k]
    return v


class G11Pipeline:
    def __init__(self, cfg: dict, log=print, tracker=None):
        from .colab_runtime import StageTracker

        self.cfg, self.log = cfg, log
        self.ctx = RunContext(cfg)
        self.ctx.ensure_layout()
        self.stages = StageStore(self.ctx)
        self.tracker = tracker or StageTracker()
        self._grid = self._mask = self._split = self._full_grid = None
        self._pool = self._dataset = self._cat = self._legacy_check = self._spec = self._expected = None
        self._cov = None
        self.reuse = bool(cfg.get("reuse_valid_cache", True))
        self.smoke = bool(cfg.get("smoke", {}).get("enabled", False))
        fe = cfg.get("features", {}) or {}
        self.feature_policy = str(fe.get("policy", "REUSE_VERIFIED"))
        if self.feature_policy not in FEATURE_POLICIES:
            raise ConfigError(f"features.policy {self.feature_policy!r} not in {FEATURE_POLICIES}")
        self.feature_version = str(fe.get("version", "g11_features_v1"))
        self.split_version = str(cfg["spatial_splits"].get("version", "v2"))
        s = read_json(self.ctx.d("audit") / "audit_summary.json")
        if s and s.get("source_hashes"):
            self.ctx.source_hashes = s["source_hashes"]

    # ================================================================== helpers
    def run(self, mode) -> None:
        for st in MODE_STAGES[Mode(mode)]:
            self.log(f"========== G11 stage: {st}")
            with self.tracker.track(st, self.log):
                getattr(self, st)()

    @property
    def ns(self) -> str:
        """Run namespace of every downstream artifact."""
        return f"{self.feature_version}__split_{self.split_version}"

    def d(self, stage: str, *parts) -> Path:
        return self.ctx.d(stage, self.ns, *parts) if stage in DOWNSTREAM else self.ctx.d(stage, *parts)

    def skey(self, stage: str) -> str:
        return f"{self.ns}/{stage}"

    def sfp(self, stage: str, sections, *inputs) -> str:
        """Fingerprint from the named config sections only (dotted paths), the source hashes and ``inputs``."""
        return stable_hash([stage, {s: _section(self.cfg, s) for s in sections}, self.ctx.source_hashes, *inputs])

    def model_dir(self, exp: str) -> Path:
        from .experiments import EXPERIMENTS
        from .registry import MODEL_DIRS

        base = exp.split("__")[0]
        kind = EXPERIMENTS.get(base, {}).get("kind", "fusion")
        return self.d("train", MODEL_DIRS[kind], exp)

    def pred_path(self, exp: str) -> Path:
        return self.d("predictions") / f"{exp}_probability.tif"

    def available_predictions(self) -> list[str]:
        """Experiment ids with a G11 full-grid probability map in this run namespace."""
        return sorted(p.name[: -len("_probability.tif")] for p in self.d("predictions").glob("*_probability.tif")
                      if p.name[0] in "EB" and ".partial" not in p.name)

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
    def mask(self) -> np.ndarray:
        if self._mask is None:
            self._mask = FL.label_raster(self.found["fault_raster"], self.grid, self.cfg["faults"]["min_label_value"])
        return self._mask

    def catalog(self):
        if self._cat is not None:
            return self._cat if len(self._cat) else None
        p = self.ctx.d("ingest") / "dem_catalog.parquet"
        if not p.is_file():
            return None
        import geopandas as gpd
        from shapely import wkt

        t = pd.read_parquet(p)
        if not len(t):
            self._cat = t
            return None
        geom = t.pop("footprint_wkt").map(lambda s: wkt.loads(s) if isinstance(s, str) else None)
        self._cat = gpd.GeoDataFrame(t, geometry=geom, crs=self.grid.crs)
        return self._cat

    def usable_catalog(self):
        cat = self.catalog()
        if cat is None:
            return None
        if bool(self.cfg["dem"].get("allow_unknown_vertical", False)):
            bad = cat.quality_flags.fillna("").str.contains("CONFLICTING_VERTICAL_UNIT|UNREADABLE")
        else:
            bad = cat.quality_flags.fillna("").str.contains("UNKNOWN_VERTICAL_UNIT|CONFLICTING_VERTICAL_UNIT|UNREADABLE")
        u = cat[~bad]
        return u if len(u) else None

    def dem_reader(self):
        from .dem_reader import DEMReader

        cat = self.catalog()
        if cat is None:
            return None
        return DEMReader(cat, self.cfg, allow_unknown_vertical=bool(self.cfg["dem"].get("allow_unknown_vertical", False)))

    def dem_coverage(self) -> dict:
        """100 m masks: ``any`` = some DEM tile, ``usable`` = usable tile (vertical unit verified), on valid cells."""
        if self._cov is None:
            from .dem_catalog import coverage_100m

            z = np.zeros(self.grid.shape, bool)
            cat, uc = self.catalog(), self.usable_catalog()
            anyc = coverage_100m(cat, self.grid)["covered"] & self.grid.valid if cat is not None else z
            use = coverage_100m(uc, self.grid)["covered"] & self.grid.valid if uc is not None else z
            self._cov = {"any": anyc, "usable": use}
        return self._cov

    def dem_availability(self) -> np.ndarray:
        c = self.dem_coverage()
        return np.where(c["usable"], 2, np.where(c["any"], 1, 0)).astype(np.uint8)

    def dem_catalog_signature(self) -> str | None:
        """Content signature of the usable DEM catalog (independent of when the parquet was rewritten)."""
        uc = self.usable_catalog()
        if uc is None:
            return None
        cols = [c for c in ("tile_id", "resolution_m", "width", "height", "vertical_scale", "quality_flags",
                            "priority_rank") if c in uc.columns]
        return stable_hash(pd.DataFrame(uc[cols]).sort_values("tile_id").astype(str).values.tolist())

    def segments(self):
        import geopandas as gpd

        p = self.ctx.d("ingest") / "fault_segments.gpkg"
        return gpd.read_file(p) if p.is_file() else None

    # ================================================================== feature identity
    @property
    def pool(self):
        from .feature_versions import LayerManifest

        if self._pool is None:
            self._pool = LayerManifest(self.ctx.d("features"))
        return self._pool

    def g06_registry(self):
        opt = (read_json(self.ctx.d("audit") / "audit_summary.json") or {}).get("optional_paths", {})
        regs = [p for p in opt.get("g06_store", []) if str(p).endswith(".parquet")]
        return regs[0] if regs else None

    def g06_sources(self) -> list:
        r = self.g06_registry()
        return [f"{Path(r).name}:{self.ctx.hashes.sha(r)}"] if r and Path(r).is_file() else []

    def g06_names(self) -> list[str]:
        """Names ``attach_g06_store`` would register (same grid only) - header reads, no data."""
        r = self.g06_registry()
        if not r or not Path(r).is_file():
            return []
        import rasterio

        reg = pd.read_parquet(r)
        col = next((c for c in ("path", "file", "output_path") if c in reg.columns), None)
        nc = next((c for c in ("name", "feature", "feature_id") if c in reg.columns), None)
        if col is None or nc is None:
            return []
        out = []
        for row in reg.itertuples():
            p = Path(getattr(row, col))
            p = p if p.is_absolute() else Path(r).parent / p
            if not p.is_file() or p.suffix.lower() not in (".tif", ".tiff"):
                continue
            try:
                with rasterio.open(p) as ds:
                    if tuple(ds.shape) == tuple(self.grid.shape) and np.allclose(tuple(ds.transform)[:6],
                                                                                 self.grid.transform[:6]):
                        out.append(f"g06_{getattr(row, nc)}")
            except Exception:                                            # noqa: BLE001
                continue
        return out

    @property
    def dem_res(self) -> float:
        return float(self.cfg["dem"]["stats_resolution_m"])

    @property
    def spec(self) -> dict:
        from .feature_versions import spec_hashes

        if self._spec is None:
            self._spec = spec_hashes(self.cfg, self.ctx.source_hashes, self.grid, self.dem_catalog_signature(),
                                     self.g06_sources())
        return self._spec

    def legacy_check(self) -> dict:
        from .recovery import legacy_config_check

        if self._legacy_check is None:
            m = read_json(self.ctx.root / "manifests" / "features.json") or \
                read_json(self.ctx.root / "manifests" / "ingest.json") or {}
            self._legacy_check = legacy_config_check(self.cfg, m.get("code_commit"))
        return self._legacy_check

    def expected(self, dem_res: float | None = None) -> pd.DataFrame:
        """Layers the requested feature version must contain (FORCE_NEW_VERSION: version-private layer ids)."""
        from .feature_versions import expected_layers

        res = float(dem_res if dem_res is not None else self.dem_res)
        if self._expected is not None and res == self.dem_res:
            return self._expected
        import rasterio

        from gems_bench.g06.raster_io import band_identifiers

        with rasterio.open(self.found["geodawn"]) as ds:
            raw = list(band_identifiers(ds))
        e = expected_layers(self.cfg, raw, self.g06_names(), self.usable_catalog() is not None, res)
        if self.feature_policy == "FORCE_NEW_VERSION":
            e["layer_id"] = e.layer_id + "#" + self.feature_version
        if res == self.dem_res:
            self._expected = e
        return e

    def dataset_id(self) -> dict:
        return {"version": self.feature_version, "spec": self.spec}

    # ================================================================== stores
    @property
    def store(self):
        """The finalized feature dataset when available, else a read-only view of the verified pool layers."""
        if self._dataset is None:
            self._dataset = self._open_dataset(strict=False)
        if self._dataset is not None:
            return self._dataset
        return self.pool_view()

    def pool_view(self):
        from .feature_dataset import PoolView

        e = self.expected()
        keep = []
        for r in e.itertuples():
            ok, _ = self.pool.reusable(r.layer_id, self.spec[r.dep_class])
            if ok:
                keep.append({**r._asdict(), "relpath": self.pool.row(r.layer_id)["relpath"],
                             "spec_hash": self.spec[r.dep_class],
                             "finite_fraction": self.pool.row(r.layer_id).get("finite_fraction")})
        cols = list(e.columns) + ["relpath", "spec_hash", "finite_fraction"]
        return PoolView(self.ctx.d("features"), pd.DataFrame(keep, columns=cols), self.grid)

    def _open_dataset(self, strict: bool = True):
        from .feature_dataset import FeatureDatasetError, FeatureDatasetIncomplete, dataset_dir, load_feature_dataset
        from .feature_versions import spec_diff

        d = dataset_dir(self.ctx.d("features"), self.feature_version)
        done = read_json(d / "COMPLETE.json")
        if done is None:
            return None
        try:
            diff = spec_diff(done.get("spec", {}), self.spec)
            if diff:
                msg = (f"feature dataset {self.feature_version} exists with another specification ({diff}). "
                       "Versions are never overwritten: set feature_version to a NEW name (with feature_policy "
                       "RESUME_MISSING every still-valid layer is reused, with FORCE_NEW_VERSION all are recomputed), "
                       "or restore the configuration that built it.")
                if strict:
                    raise StageGateError(msg)
                self.log(f"[features] WARNING {msg}")
                return None
            return load_feature_dataset(self.ctx.d("features"), self.feature_version, self.grid, verify=None)
        except (FeatureDatasetIncomplete, FeatureDatasetError) as e:
            if strict:
                raise StageGateError(str(e)) from e
            return None

    def _get(self, name):
        try:
            return np.asarray(self.store.get(name), np.float32)
        except KeyError:
            return None

    def split_result(self) -> SS.SplitResult:
        if self._split is not None:
            return self._split
        if self.split_version == "v1":
            res = SS.make_splits(self.grid.valid, self.mask, self.grid, self.cfg)
            man = read_json(self.ctx.d("splits") / "split_manifest.json")
            if man is not None:
                got = {k: int(res.mask(k).sum()) for k in list(SS.SPLITS) + ["embargo"]}
                if got != man["pixels"]:
                    raise StageGateError(f"recomputed splits differ from the saved manifest: {got} vs {man['pixels']}")
            self._split = res
            return res
        from .registry import sha256_file
        from .splits_v2 import load_split_code

        d = self.ctx.d("splits", self.split_version)
        man = read_json(d / "split_manifest.json")
        if read_json(d / "COMPLETE.json") is None or man is None:
            raise StageGateError(f"split {self.split_version} not complete - run the splits stage (05_splits/"
                                 f"{self.split_version}/COMPLETE.json is written only after every gate passed)")
        if sha256_file(d / "split_code.tif") != man.get("split_code_sha256"):
            raise StageGateError(f"split {self.split_version}: split_code.tif differs from its manifest")
        code = load_split_code(d / "split_code.tif")
        blocks = pd.read_parquet(d / "split_registry.parquet")
        groups = np.load(d / "groups.npy") if (d / "groups.npy").is_file() else np.full(code.shape, -1, np.int32)
        self._split = SS.SplitResult(code, blocks, float(man["embargo_m"]), man.get("embargo_components", {}), groups)
        return self._split

    def split_manifest(self) -> dict:
        if self.split_version == "v1":
            return read_json(self.ctx.d("splits") / "split_manifest.json") or {}
        return read_json(self.ctx.d("splits", self.split_version) / "split_manifest.json") or {}

    # ================================================================== Phase 0: audit
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
        self._grid = self._spec = self._expected = None
        self.stages.record("audit", self.stages.fingerprint("audit"), outs, {"skipped": len(summ["skipped"]),
                                                                               "dem_tiles": len(files)})
        self.log(f"[audit] grid {grid.shape} EPSG:{grid.epsg}; {len(files)} DEM tiles "
                 f"{summ['dem_vertical_units']}; {summ['external_compatible']} compatible external prediction(s); "
                 f"{len(summ['skipped'])} optional source(s) skipped")
        return summ

    # ================================================================== Phase 0b: ingestion (labels written later)
    INGEST_SECTIONS = ("geophysical_grid", "faults", "labels.buffers_m", "labels.near_band_m", "labels.states", "dem",
                       "source_artifacts.dem_dirs", "smoke", "reporting.figures")

    def ingest(self) -> dict:
        cfg, grid = self.cfg, self.grid
        fp = self.sfp("ingest", self.INGEST_SECTIONS)
        if self._cached("ingest", fp):
            return self.stages.load("ingest")["stats"]
        rec = self.stages.load("ingest")
        if self.reuse and rec and rec.get("source_hashes") == self.ctx.source_hashes:
            ok, why = self.stages.outputs_ok("ingest")
            lc = self.legacy_check()
            if ok and all(str(lc.get(k, "")).startswith("MATCH") for k in ("label", "dem")):
                self.log("[ingest] REUSED: earlier ingest record with identical sources, verified outputs and "
                         "unchanged ingest-relevant config (legacy record kept untouched)")
                return rec.get("stats", {})
        self.stages.begin("ingest", fp)
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
        dist = FL.make_targets(mask, grid.valid, grid, cfg, segs if seg_src == "vector" else None)["Y_DISTANCE"]
        st = FL.label_states(mask, grid.valid, dist, cfg)
        del dist
        outs.append(write_geotiff(d / "label_states.tif", st, grid, ["label state (labels.states)"], nodata=None))
        outs.append(atomic_write_json(d / "master_grid.json", {**grid.as_dict(), "smoke_window": self.smoke_window,
                                                               "full_grid": self.full_grid.as_dict()}))
        from .features.geophysics import band_registry

        breg = band_registry(self.found["geodawn"])
        nb = cfg["geophysical_grid"].get("expected_band_count")
        breg_ok = nb is None or len(breg) == int(nb)
        outs.append(atomic_write_parquet(d / "geophysical_registry.parquet", breg))
        if not breg_ok and cfg.get("reporting", {}).get("strict_integrity", True):
            raise StageGateError(f"GeoDAWN has {len(breg)} bands, expected {nb} (geophysical_grid.expected_band_count)")
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
        self._cat = self._cov = self._spec = self._expected = None
        if dem_dirs:
            from .dem_catalog import build_dem_catalog, coverage_100m, save_catalog

            up = self.ctx.d("audit") / "dem_vertical_units.csv"
            units = pd.read_csv(up) if up.is_file() else None
            cat = build_dem_catalog(dem_dirs, grid.crs, cfg, units=units)
            outs += save_catalog(cat, d, grid)
            self._cat = None
            allc = coverage_100m(cat, grid)
            cov = self.dem_coverage()
            usable = cov["usable"]
            uc = self.usable_catalog()
            fault_all, fault_use = int((mask & allc["covered"]).sum()), int((mask & usable).sum())
            cov_rep.update(dem_tiles=int(len(cat)), dem_tiles_usable=int(len(uc)) if uc is not None else 0,
                           dem_covered_km2=float((allc["covered"] & grid.valid).sum() * grid.res[0] * grid.res[1] / 1e6),
                           dem_usable_km2=float(usable.sum() * grid.res[0] * grid.res[1] / 1e6),
                           fault_pixels_with_any_dem=fault_all, fault_pixels_with_usable_dem=fault_use,
                           fault_coverage_lost_to_unit_exclusion=(fault_all - fault_use) / max(fault_all, 1),
                           fault_fraction_with_usable_dem=fault_use / max(int(mask.sum()), 1),
                           flags=cat.quality_flags.value_counts().to_dict() if len(cat) else {},
                           duplicates_across_folders=len(cat.attrs.get("duplicates", [])))
            stats.update(dem_tiles=int(len(cat)), dem_tiles_usable=cov_rep["dem_tiles_usable"])
        else:
            allc, usable = None, np.zeros(grid.shape, bool)
            stats["dem_tiles"] = 0
        outs.append(atomic_write_json(d / "coverage_report.json", cov_rep))
        if cfg.get("reporting", {}).get("figures", True):
            outs += self._ingest_figures(d / "figures", mask, allc, usable)
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

    # ================================================================== Phase 1: recovery
    def recover(self, verify: str | None = None):
        """Inspect / verify / classify the existing feature artifacts (never deletes). Returns a RecoveryReport."""
        from .recovery import recover_feature_stage

        v = verify or ("FULL" if self.feature_policy == "RESUME_MISSING" else
                       str((self.cfg.get("features", {}) or {}).get("recovery_verify", "AUTO")))
        cov = self.dem_coverage()["usable"] if self.usable_catalog() is not None else None
        rep = recover_feature_stage(self.ctx, self.cfg, grid=self.grid, source_hashes=self.ctx.source_hashes,
                                    expected=self.expected(), verify=v, dem_catalog_sha=self.dem_catalog_signature(),
                                    g06_sources=self.g06_sources(), dem_covered=cov, legacy_check=self.legacy_check(),
                                    log=self.log)
        self._pool = None                                   # reload the (possibly extended) completion manifest
        return rep

    # ================================================================== Phase 2: missing-layer processing
    def process_features(self, max_seconds: float | None = None) -> dict:
        """Compute ONLY the layers of the requested version that are not verified in the pool (local first,
        validated, committed to Drive with a completion manifest row). DEM statistics run chunk by chunk."""
        from .feature_versions import layer_relpath
        from .persist import commit_layer

        cfg, grid = self.cfg, self.grid
        if self.feature_policy == "FORCE_NEW_VERSION":
            from .feature_dataset import dataset_dir

            if (dataset_dir(self.ctx.d("features"), self.feature_version) / "COMPLETE.json").is_file():
                raise StageGateError(f"FORCE_NEW_VERSION: {self.feature_version} already exists - choose a new "
                                     "feature_version (versions are never overwritten)")
        e = self.expected()
        pool, spec = self.pool, self.spec
        need = e[[not pool.reusable(r.layer_id, spec[r.dep_class])[0] for r in e.itertuples()]]
        summary = {"expected": int(len(e)), "missing_at_start": int(len(need)), "computed": 0, "complete": False,
                   "by_family": need.family.value_counts().to_dict()}
        if need.empty:
            summary["complete"] = True
            self.log(f"[features] nothing to compute: all {len(e)} layer(s) of {self.feature_version} are verified")
            return summary
        self.log(f"[features] {len(need)} of {len(e)} layer(s) to compute for {self.feature_version} "
                 f"(DEM {self.dem_res:g} m): {summary['by_family']}")
        froot = self.ctx.d("features")
        local = Path(cfg["scratch_root"]) / "feature_commit"
        private = self.feature_version if self.feature_policy == "FORCE_NEW_VERSION" else None
        from .config import code_commit

        commit_id = code_commit()

        def commit(r, arr):
            row = {"layer_id": r.layer_id, "name": r.name, "family": r.family, "dep_class": r.dep_class,
                   "processing_resolution_m": r.processing_resolution_m if r.dep_class == "dem" else None,
                   "spec_hash": spec[r.dep_class], "provenance": "computed", "created_by": f"G11 {commit_id}",
                   "unit": r.unit, "scale_m": r.scale_m, "source": r.source, "label_derived": r.dep_class == "label"}
            rel = layer_relpath(r.name, r.family, r.dep_class, self.dem_res, private)
            commit_layer(arr, froot, rel, local, row, pool, grid.valid)
            summary["computed"] += 1

        names_needed = set(need.name)
        rows_by_name = {r.name: r for r in need.itertuples()}
        # ---- 1. labels
        lab = [n for n in names_needed if rows_by_name[n].dep_class == "label"]
        if lab:
            segs = self.segments() if self.found.get("fault_vector") else None
            tg = FL.make_targets(self.mask, grid.valid, grid, cfg, segs)
            for n in lab:
                commit(rows_by_name[n], tg[n])
            del tg
            pool.flush()
        # ---- 2. geophysics (raw, multiscale, cross) in a LOCAL staging store
        geo = [n for n in names_needed if rows_by_name[n].dep_class == "geophysical"
               and rows_by_name[n].family not in ("g06", "evidence_rank")]
        if geo:
            from .features import geophysics as G

            stg = FeatureStore(local / f"stage_geo_{self.feature_version}", grid)
            for nm, a in G.read_all_bands(self.found["geodawn"], grid).items():
                if f"raw_{nm}" in names_needed:
                    commit(rows_by_name[f"raw_{nm}"], np.where(grid.valid, a, np.nan))
            if any(not n.startswith("raw_") for n in geo):
                bands = G.read_geodawn_bands(self.found["geodawn"], cfg["feature_engineering"]["geophysics"]["bands"],
                                             grid)
                G.compute_geophysical_to_store(bands, grid, cfg, stg, spec["geophysical"])
                del bands
                for n in geo:
                    if not n.startswith("raw_"):
                        commit(rows_by_name[n], np.asarray(stg.get(n), np.float32))
            pool.flush()
        # ---- 3. G06 indices
        g6 = [n for n in names_needed if rows_by_name[n].family == "g06"]
        if g6:
            from .features import geophysics as G

            stg = FeatureStore(local / f"stage_g06_{self.feature_version}", grid)
            G.attach_g06_store(self.g06_registry(), stg, grid)
            for n in g6:
                commit(rows_by_name[n], np.asarray(stg.get(n), np.float32))
            pool.flush()
        self._dataset = None
        view = self.pool_view()
        from .features.multiscale import compute_multimodal_alignment, pct_rank, terrain_100m_summaries

        sc = [s for s in cfg["feature_engineering"]["geophysics"]["scales_m"] if s in (400, 800)] or \
            cfg["feature_engineering"]["geophysics"]["scales_m"][:1]
        if "geo_edge_rank" in names_needed:
            gm = [np.asarray(view.get(f"{f}_gm_{int(sc[0])}m"), np.float32) for f in ("magnetic", "gravity")
                  if view.has(f"{f}_gm_{int(sc[0])}m")]
            commit(rows_by_name["geo_edge_rank"], np.nanmax(np.stack([pct_rank(x, grid.valid) for x in gm]), 0))
            pool.flush()
        # ---- 4. DEM statistics (resumable chunks) -> dem100_*@r<res>
        dem100 = [n for n in names_needed if n.startswith("dem100_")]
        if dem100:
            done = self._process_dem(dem100, rows_by_name, commit, max_seconds)
            pool.flush()
            if not done:
                summary["complete"] = False
                summary["note"] = "DEM chunks remain - re-run process_features to resume (committed chunks are kept)"
                return summary
        view = self.pool_view()
        # ---- 5. DEM-derived: terrain_scarp_rank, geo x DEM multimodal
        if "terrain_scarp_rank" in names_needed:
            s95 = np.asarray(view.get("dem100_slope_p95"), np.float32)
            commit(rows_by_name["terrain_scarp_rank"], pct_rank(s95, grid.valid & np.isfinite(s95)))
        mm = [n for n in names_needed if rows_by_name[n].family == "multimodal"]
        if mm:
            for s in sc:
                terr = terrain_100m_summaries(view, grid, int(s))
                if terr["coh"] is None:
                    continue
                geo_d = {f"{f}_{k}_{int(s)}m": np.asarray(view.get(f"{f}_{k}_{int(s)}m"), np.float32)
                         for f in ("magnetic", "gravity") for k in ("sc2", "ss2", "coh", "gm")
                         if view.has(f"{f}_{k}_{int(s)}m")}
                cube = compute_multimodal_alignment(geo_d, terr, grid.valid, int(s))
                for r in cube.meta.itertuples():
                    if r.name in names_needed:
                        commit(rows_by_name[r.name], cube[r.name])
        pool.flush()
        # ---- 6. known-fault morphology + native windows (DEM 1 m; independent of the statistics resolution)
        if self.usable_catalog() is not None and not (froot / "segment_features.parquet").is_file():
            reader = self.dem_reader()
            summary["morphology"] = {k: v for k, v in self._known_fault_morphology(reader, self.pool_view()).items()
                                     if k != "outputs"}
        left = [r.layer_id for r in e.itertuples() if not self.pool.reusable(r.layer_id, spec[r.dep_class])[0]]
        summary["complete"] = not left
        summary["still_missing"] = left[:20]
        self.log(f"[features] computed {summary['computed']} layer(s); complete: {summary['complete']}")
        return summary

    def _process_dem(self, names, rows_by_name, commit, max_seconds) -> bool:
        from .dem_engine import TileCache, assemble_dem_layers, resume_feature_processing
        from .config import code_commit

        cfg, grid, res = self.cfg, self.grid, self.dem_res
        uc = self.usable_catalog()
        cov = self.dem_coverage()["usable"]
        scratch = Path(cfg["scratch_root"])

        def factory():
            tc = TileCache(uc, res, scratch / "dem_tile_cache", bool(cfg["dem"].get("engine_use_overviews", False)),
                           int(cfg.get("hardware", {}).get("workers", 2)), self.log)
            tab = tc.ensure()
            if "error" in tab and tab.error.notna().any():
                self.log(f"[dem-engine] {int(tab.error.notna().sum())} tile(s) unreadable (excluded): "
                         f"{tab[tab.error.notna()].tile_id.tolist()[:5]}")
            return tc.reader(cfg, bool(cfg["dem"].get("allow_unknown_vertical", False)))

        st = resume_feature_processing(factory, grid, cov, cfg, res, self.ctx.d("features"), scratch / "dem_chunks_local",
                                       self.spec["dem"], self.log, bool(cfg["dem"].get("save_every_chunk", True)),
                                       max_seconds, code_commit())
        if not st["complete"]:
            self.log(f"[dem-engine] {st['n_committed']}/{st['n_chunks']} chunk(s) committed - resume later")
            return False
        full = assemble_dem_layers(st["chunk_dir"], st["plan"], grid, cfg, res, cov, self.spec["dem"])
        tiles = full.pop("_source_tiles", [])
        for n in names:
            r = rows_by_name[n]
            r = r._replace(source=f"dem_engine {res:g} m from {len(tiles)} tile(s)")
            commit(r, full[n[len("dem100_"):]])
        return True

    def _known_fault_morphology(self, reader, view, max_segments: int = 2000, native_windows: int = 6) -> dict:
        """Cross-fault profiles of the KNOWN fault segments (03_features/segment_features.parquet) and a few native
        terrain feature windows (03_features/terrain_native/*.tif)."""
        from .features.segment import segment_features
        from .features.terrain import compute_terrain_features
        from .grid import GridSpec

        segs = self.segments()
        d = self.ctx.d("features")
        if segs is None or not len(segs) or reader is None:
            return {"notes": ["no known fault segments or no DEM: segment features skipped"]}
        cov = reader.index.cat.union_all() if hasattr(reader.index.cat, "union_all") else reader.index.cat.unary_union
        s = segs[segs.geometry.intersects(cov)].sort_values("length_m", ascending=False).head(max_segments)
        outs, notes = [], [f"{len(s)} of {len(segs)} known segments intersect the DEM coverage"]

        def g(n):
            return np.asarray(view.get(n), np.float32) if view.has(n) else None

        if len(s):
            rasters = {"geo_edge_rank": g("geo_edge_rank"), "terrain_scarp_rank": g("terrain_scarp_rank")}
            feats = segment_features(s.reset_index(drop=True), rasters, self.grid, self.cfg, reader)
            feats.insert(0, "segment_index", s.index.to_numpy())
            outs.append(atomic_write_parquet(d / "segment_features.parquet", feats))
        nd = self.ctx.d("features", "terrain_native")
        res = float(self.cfg["feature_engineering"]["cross_fault"]["resolution_m"])
        for k, geom in enumerate(s.geometry[:native_windows]):
            c = geom.centroid
            b = (c.x - 300, c.y - 300, c.x + 300, c.y + 300)
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

    # ================================================================== Phase 3: finalize + validate
    def _feature_config(self) -> dict:
        c = self.cfg
        return {"geophysical_grid": c["geophysical_grid"], "geophysics": c["feature_engineering"]["geophysics"],
                "dem": c["dem"], "faults": c["faults"], "labels": {k: c["labels"][k] for k in ("buffers_m", "near_band_m")},
                "cross_fault": c["feature_engineering"]["cross_fault"], "feature_policy": self.feature_policy}

    def finalize_features(self):
        from .feature_dataset import FeatureDataset, finalize_feature_dataset
        from .config import code_commit

        froot = self.ctx.d("features")
        ing = self.ctx.d("ingest")
        rec = read_json(self.ctx.root / "recovery" / "feature_recovery_report.json") or {}
        prov = {"source_hashes": self.ctx.source_hashes, "recovery": {k: rec.get(k) for k in ("counts", "legacy", "notes")},
                "dem_catalog_signature": self.dem_catalog_signature(), "g06_sources": self.g06_sources(),
                "known_fault_morphology": str(froot / "segment_features.parquet")
                if (froot / "segment_features.parquet").is_file() else None}
        man = finalize_feature_dataset(froot, self.feature_version, self.grid, self.pool, self.expected(), self.spec,
                                       dem_availability=self.dem_availability(), native_dir=froot / "terrain_native",
                                       fault_registry=ing / "fault_registry.parquet",
                                       fault_segments=ing / "fault_segments.gpkg", processing_config=self._feature_config(),
                                       provenance=prov, code_commit=code_commit(), log=self.log)
        self._finalize_legacy_version(prov)
        self._dataset = FeatureDataset(froot, self.feature_version, self.grid)
        self.stages.record(f"features_{self.feature_version}", stable_hash(self.dataset_id()),
                           [man.path / "COMPLETE.json", man.path / "dataset_manifest.json"], self.feature_stats())
        return man

    def _finalize_legacy_version(self, prov: dict) -> None:
        """Keep verified layers of another DEM resolution (e.g. the legacy 2 m run) as their own version."""
        from .feature_dataset import FeatureDatasetError, finalize_feature_dataset
        from .feature_versions import spec_hashes

        rc = self.cfg.get("recovery", {}) or {}
        lr = float(rc.get("legacy_dem_statistics_resolution_m", 2.0))
        if lr == self.dem_res or self.usable_catalog() is None or self.feature_policy == "FORCE_NEW_VERSION":
            return
        name = str((self.cfg.get("features", {}) or {}).get("legacy_version") or f"g11_features_legacy_r{lr:g}m")
        cfg_l = {**self.cfg, "dem": {**self.cfg["dem"], "stats_resolution_m": lr}}
        spec_l = spec_hashes(cfg_l, self.ctx.source_hashes, self.grid, self.dem_catalog_signature(), self.g06_sources())
        e = self.expected(lr)
        if not all(self.pool.reusable(r.layer_id, spec_l[r.dep_class])[0] for r in e.itertuples()):
            return
        try:
            finalize_feature_dataset(self.ctx.d("features"), name, self.grid, self.pool, e, spec_l,
                                     dem_availability=self.dem_availability(),
                                     native_dir=self.ctx.d("features") / "terrain_native",
                                     fault_registry=self.ctx.d("ingest") / "fault_registry.parquet",
                                     fault_segments=self.ctx.d("ingest") / "fault_segments.gpkg",
                                     processing_config={**self._feature_config(), "dem": cfg_l["dem"]},
                                     provenance={**prov, "note": f"verified legacy layers, DEM statistics {lr:g} m"},
                                     log=self.log)
        except FeatureDatasetError as ex:
            self.log(f"[features] legacy version {name} not finalized: {ex}")

    def validate_features(self, level: str | None = None):
        from .feature_dataset import validate_feature_dataset

        lv = level or ("FULL" if self.feature_policy == "RESUME_MISSING" else "QUICK")
        rep = validate_feature_dataset(self.ctx.d("features"), self.feature_version, self.grid, level=lv,
                                       seed=int(self.cfg["reproducibility"]["seed"]))
        from .feature_dataset import dataset_dir

        atomic_write_json(dataset_dir(self.ctx.d("features"), self.feature_version) / f"validation_{lv.lower()}.json",
                          rep.as_dict())
        if not rep.passed:
            raise StageGateError(f"feature dataset {self.feature_version} failed {lv} validation: {rep.problems}")
        self.log(f"[features] {self.feature_version}: {lv} validation passed ({len(rep.checks)} checks)")
        return rep

    def feature_stats(self) -> dict:
        ds = self.store
        return {"version": self.feature_version, "n_features": int(len(ds.reg)), "n_predictors": len(ds.predictors()),
                "families": ds.reg.family.value_counts().to_dict(),
                "dem_statistics_resolution_m": getattr(ds, "dem_resolution_m", None),
                "spec": getattr(ds, "spec", None)}

    def features(self) -> dict:
        """Recovery-first feature stage: reuse the finalized version, or recover + compute missing + finalize."""
        ds = self._open_dataset(strict=True)
        if ds is not None and self.feature_policy == "FORCE_NEW_VERSION":
            raise StageGateError(f"FORCE_NEW_VERSION: {self.feature_version} already exists - choose a new name")
        if ds is not None and self.feature_policy == "REUSE_VERIFIED":
            self._dataset = ds
            self.log(f"[features] REUSED finalized dataset {self.feature_version} (no feature computation)")
            return self.feature_stats()
        self.recover()
        st = self.process_features()
        if not st["complete"]:
            raise StageGateError(f"feature processing incomplete: {st.get('note') or st.get('still_missing')}")
        self.finalize_features()
        self.validate_features()
        return self.feature_stats()

    # ================================================================== Phase 4: EDA (saved features only)
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
        fp = self.sfp("eda", ["eda", "labels.near_band_m", "reproducibility"], self.dataset_id())
        if self._cached("eda", fp):
            return self.stages.load("eda")["stats"]
        self.stages.begin("eda", fp)
        d = self.ctx.d("eda")
        valid, mask = grid.valid, self.mask
        dist = np.asarray(store.get("Y_DISTANCE"), np.float32)
        reader = self.dem_reader()
        dem_cov = self.dem_coverage()["any"]
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
        if self.smoke and maps:
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
            {"title": f"Feature dataset {self.feature_version}", "text": f"Computed from the saved, verified dataset "
             f"(DEM statistics {getattr(store, 'dem_resolution_m', None)} m); no raw feature recomputation."},
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
                 "external_maps_analysed": len(maps), "native_features_tested": int(len(native)),
                 "feature_version": self.feature_version}
        self.stages.record("eda", fp, outs, stats)
        return stats

    def _eda_candidates(self, reader):
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
                bonus = 1e6 if k not in {kk for kk, _ in chosen} else 0.0
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

    # ================================================================== Phase 5: geographic splits (v2)
    SPLIT_SECTIONS = ("spatial_splits", "sampling.patch_px", "labels.near_band_m", "feature_engineering.geophysics.scales_m",
                      "dem.context_scales_m", "candidate_generation.link_radius_m", "faults")

    def splits(self) -> dict:
        if self.split_version == "v1":
            return self._splits_v1()
        from .splits_v2 import SplitGateError, make_splits_v2, save_split_version

        d = self.ctx.d("splits", self.split_version)
        fp = self.sfp(f"splits_{self.split_version}", self.SPLIT_SECTIONS, self.spec.get("label"),
                      self.dem_catalog_signature())
        done = read_json(d / "COMPLETE.json")
        if self.reuse and done and done.get("fingerprint") == fp:
            self._split = None
            self.split_result()
            self.log(f"[splits] {self.split_version} CACHED (gates passed {done.get('completed_utc')})")
            return done["stats"]
        if done and done.get("fingerprint") != fp:
            raise StageGateError(f"split {self.split_version} exists with other parameters; set "
                                 "spatial_splits.version to a new name (split versions are never overwritten)")
        self.stages.begin(f"splits_{self.split_version}", fp)
        st = self.store
        ld = st.reg[st.reg.label_derived.astype(bool)].name.tolist()
        g = (self.cfg["spatial_splits"].get("v2") or {}).get("gates") or {}
        patch = int(self.cfg["sampling"]["patch_px"]) if g.get("require_patch_windows", True) else None
        dem_u = self.dem_coverage()["usable"] if self.usable_catalog() is not None else None
        try:
            res = make_splits_v2(self.grid.valid, self.mask, self.grid, self.cfg, dem_u, patch, st.predictors(), ld)
        except SplitGateError as e:
            if e.gates is not None:
                atomic_write_parquet(d / "split_gates_FAILED.parquet", e.gates)
            self.stages.fail(f"splits_{self.split_version}", fp, e)
            raise StageGateError(f"split {self.split_version} rejected by its gates (nothing written as COMPLETE): {e}")
        outs = save_split_version(res, self.grid, d, self.ctx)
        from .registry import atomic_write_npy

        outs.append(atomic_write_npy(d / "groups.npy", res.groups))
        checks = SS.run_leakage_checks(res, self.grid, st.predictors(), ld)
        outs.append(atomic_write_parquet(d / "leakage_checks.parquet", checks))
        outs.append(self.ctx.write_manifest(d / "leakage_report.json", {
            "passed": bool(checks.passed.all() and res.gates.passed.all()), "checks": checks.to_dict("records"),
            "gates": res.gates.to_dict("records"), "embargo_m": res.embargo_m, "training_stops_if_failed": True}))
        if self.cfg.get("reporting", {}).get("figures", True):
            from .eda.figures import split_map

            outs.append(split_map(self.grid, res.code, self.mask, d / "split_map.png"))
        if not checks.passed.all():
            self.stages.fail(f"splits_{self.split_version}", fp, SS.LeakageError("leakage checks failed"))
            raise SS.LeakageError(f"leakage checks failed:\n{checks[~checks.passed].to_string()}")
        stats = {"version": self.split_version, "embargo_m": res.embargo_m, "generator": res.report["generator"],
                 "pixels": {k: int(res.mask(k).sum()) for k in list(SS.SPLITS) + ["embargo"]},
                 "fault_px": {k: v["fault_px"] for k, v in res.report["split_stats"].items()},
                 "regions": res.report["n_regions"], "gates_passed": True}
        atomic_write_json(d / "COMPLETE.json", {"fingerprint": fp, "stats": stats, "version": self.split_version,
                                                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self._invalidate_legacy_split()
        self.stages.record(f"splits_{self.split_version}", fp, outs, stats)
        self._split = None
        self.split_result()
        self.log(f"[splits] {stats}")
        return stats

    def _invalidate_legacy_split(self) -> None:
        """Mark the earlier root-level split (v1) INVALID - only after the new version passed every gate."""
        legacy = self.ctx.d("splits") / "split_manifest.json"
        marker = self.ctx.d("splits") / "SPLIT_V1_INVALID.json"
        if legacy.is_file() and not marker.is_file():
            m = read_json(legacy) or {}
            px = m.get("pixels", {})
            atomic_write_json(marker, {"status": "INVALID", "superseded_by": self.split_version,
                                       "reason": "required partitions empty or near-empty "
                                                 f"(validation {px.get('validation')}, stack {px.get('stack')}, test "
                                                 f"{px.get('test')} px); fault-system merging after a 3.2 km dilation "
                                                 "joined almost every block into one train group",
                                       "pixels": px, "files_kept": True, "marked_utc":
                                       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def _splits_v1(self) -> dict:
        fp = self.sfp("splits_v1", self.SPLIT_SECTIONS, self.spec.get("label"))
        if self._cached("splits", fp):
            self.split_result()
            return self.stages.load("splits")["stats"]
        self._split = None
        res = SS.make_splits(self.grid.valid, self.mask, self.grid, self.cfg)
        d = self.ctx.d("splits")
        ld = self.store.reg[self.store.reg.label_derived.astype(bool)].name.tolist()
        checks = SS.run_leakage_checks(res, self.grid, self.store.predictors(), ld)
        outs = SS.save_splits(res, self.grid, d, self.ctx)
        outs.append(atomic_write_parquet(d / "leakage_checks.parquet", checks))
        if not checks.passed.all():
            raise SS.LeakageError(f"leakage checks failed:\n{checks[~checks.passed].to_string()}")
        self._split = res
        stats = {"embargo_m": res.embargo_m, "pixels": {k: int(res.mask(k).sum()) for k in list(SS.SPLITS) + ["embargo"]}}
        self.stages.record("splits", fp, outs, stats)
        return stats

    # ================================================================== Phase 6: training data
    def training_data(self, exp: str, channels=None):
        from .experiments import EXPERIMENTS
        from .training_data import build_training_dataset

        spec = EXPERIMENTS[exp.split("__")[0]]
        return build_training_dataset(self.store, self.split_manifest(), {
            "experiment": exp.split("__")[0], "channels": channels, "families": spec.get("families"),
            "target": self.cfg["labels"].get("training_target", "Y_BUFFER_200M")},
            split_code=self.split_result().code, fault=self.mask, out_root=self.d("train"), cfg=self.cfg,
            hard_geo=self._get("geo_edge_rank"), hard_terrain=self._get("terrain_scarp_rank"), log=self.log)

    def deep_channels(self) -> list[str]:
        from .experiments import CHANNEL_SOURCE

        k = int(self.cfg["models"]["segformer"].get("in_channels") or 16)
        imp_p = self.model_dir(CHANNEL_SOURCE) / "feature_importance.parquet"
        if imp_p.is_file():
            return pd.read_parquet(imp_p).feature.tolist()[:k]
        return self.store.predictors(["geo_magnetic", "geo_gravity"])[:k]

    # ================================================================== Phase 7: training
    def train_fingerprint(self, exp: str) -> str:
        from .experiments import CHANNEL_SOURCE, EXPERIMENTS

        kind = EXPERIMENTS[exp.split("__")[0]]["kind"]
        secs = ["labels.training_target", "reproducibility", "sampling", "models.xgb"] if kind == "pixel" else \
            ["labels.training_target", "reproducibility", "sampling", "training", "inference", "models.segformer",
             "models.gated_fusion", "models.terrain_encoder"]
        extra = []
        if kind != "pixel":
            imp = self.model_dir(CHANNEL_SOURCE) / "feature_importance.parquet"
            extra.append(self.ctx.hashes.sha(imp) if imp.is_file() else None)
        return self.sfp(f"train_{exp}", secs, self.dataset_id(), self.split_manifest().get("split_code_sha256"), *extra)

    def train(self, experiments=None) -> pd.DataFrame:
        """Train experiments sequentially; each is its own resumable stage with its own directory."""
        from .experiments import EXPERIMENTS

        cfg, grid, store = self.cfg, self.grid, self.store
        res = self.split_result()
        tgt_name = cfg["labels"].get("training_target", "Y_BUFFER_200M")
        target = np.asarray(store.get(tgt_name), np.float32)
        dist = np.asarray(store.get("Y_DISTANCE"), np.float32)
        rows = []
        default = ["E0", "E1", "E2", "E3"]
        exps = self.expand_experiments(default if experiments is None else list(experiments))
        for exp in exps:
            spec = EXPERIMENTS[exp.split("__")[0]]
            if spec["kind"] in ("segment", "fusion"):
                rows.append({"experiment": exp, "status": "SKIPPED: trained in the INFER / FUSION stage"})
                continue
            mdir = self.model_dir(exp)
            key = self.skey(f"train_{exp}")
            fp = self.train_fingerprint(exp)
            if spec.get("requires_dem") and not any(n.startswith("dem100_") for n in store.reg.name):
                rows.append({"experiment": exp, "status": "SKIPPED: no DEM features"})
                continue
            if self._cached(key, fp) and self.pred_path(exp).is_file():
                rows.append({"experiment": exp, "status": "CACHED", **self.stages.load(key)["stats"]})
                continue
            if spec["kind"] in ("patch", "multires"):
                try:
                    import torch  # noqa: F401
                except ImportError:
                    rows.append({"experiment": exp, "status": "SKIPPED: torch not installed"})
                    continue
            self._prepare_model_dir(mdir, fp)
            self.stages.begin(key, fp)
            try:
                with self.tracker.track(f"train_{exp}", self.log) as trk:
                    pred, stats, outs = self._train_one(exp, spec, res, target, tgt_name, dist, mdir)
                    t1 = time.time()
                    outs.append(write_geotiff(self.pred_path(exp), pred, grid, [f"{exp} probability"]))
                    stats["inference_write_seconds"] = round(time.time() - t1, 1)
            except Exception as e:
                self.stages.fail(key, fp, e)
                optional = "__" in exp and self.cfg.get("reporting", {}).get("failure_policy",
                                                                              "STOP_REQUIRED_SKIP_OPTIONAL") != "STOP_ON_ANY_FAILURE"
                if optional:
                    rows.append({"experiment": exp, "status": f"SKIPPED (optional variant failed): {type(e).__name__}: {e}"})
                    continue
                raise
            stats.update(train_seconds=trk.get("seconds"), peak_ram_gb=trk.get("peak_ram_gb"),
                         peak_gpu_gb=trk.get("peak_gpu_gb"), fingerprint=fp)
            outs.append(atomic_write_json(mdir / "metrics.json", stats))
            outs.append(self.ctx.write_manifest(mdir / "checkpoint_metadata.json", {
                "experiment": exp, "spec": spec, "target": tgt_name, "fingerprint": fp,
                "feature_dataset": self.feature_version, "split_version": self.split_version,
                "split_code_sha256": self.split_manifest().get("split_code_sha256"),
                "selection": "validation split only", **stats}))
            self.stages.record(key, fp, outs, stats)
            rows.append({"experiment": exp, "status": "TRAINED", **stats})
        t = pd.DataFrame(rows)
        atomic_write_parquet(self.d("train") / "training_summary.parquet", t)
        return t

    def _prepare_model_dir(self, mdir: Path, fp: str) -> None:
        """A directory holding another configuration's checkpoints is archived (moved, never deleted), so a resumed
        run never continues from a stale ``last.pt`` and a failed run never erases a successful model."""
        meta = read_json(mdir / "checkpoint_metadata.json")
        last = read_json(mdir / "run_fingerprint.json")
        old = (meta or {}).get("fingerprint") or (last or {}).get("fingerprint")
        files = [p for p in mdir.iterdir() if p.is_file()] if mdir.is_dir() else []
        if files and old != fp:
            arch = mdir / "archive" / f"{old or 'unknown'}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
            arch.mkdir(parents=True, exist_ok=True)
            for p in files:
                p.replace(arch / p.name)
            self.log(f"[train] archived previous checkpoints of {mdir.name} -> {arch.relative_to(mdir)}")
        atomic_write_json(mdir / "run_fingerprint.json", {"fingerprint": fp})

    def expand_experiments(self, exps) -> list[str]:
        """Ablation variants: E2__<loss> (training.loss_ablation), E2__finetune (models.segformer.pretrained_path),
        E3__<mode> (models.gated_fusion.mode_ablation)."""
        out = []
        t, m = self.cfg["training"], self.cfg["models"]
        for e in exps:
            out.append(e)
            if e == "E2":
                out += [f"E2__{x}" for x in t.get("loss_ablation", []) if x != t["loss"]]
                if m["segformer"].get("pretrained_path"):
                    out.append("E2__finetune")
            if e == "E3":
                out += [f"E3__{x}" for x in m["gated_fusion"].get("mode_ablation", []) if x != m["gated_fusion"]["mode"]]
        return list(dict.fromkeys(out))

    def _train_one(self, exp, spec, res, target, tgt_name, dist, mdir):
        from .datasets.pixel_dataset import GeoPixelDataset
        from .models.xgb_baseline import predict_full_grid, train_pixel_model

        cfg, grid, store = self.cfg, self.grid, self.store
        t0 = time.time()
        if spec["kind"] == "pixel":
            td = self.training_data(exp)
            wu = float(cfg["models"]["xgb"]["pu_unlabeled_weight"])
            sets = {}
            for split in ("train", "validation"):
                X, y = td.load_xy(store, split, grid.shape)
                r, c = td.rows_cols(split, grid.shape)
                bpx = max(1, int(round(cfg["sampling"]["balance_block_km"] * 1000 / 100.0)))
                nbc = -(-grid.shape[1] // bpx)
                sets[split] = GeoPixelDataset(X, y.astype(np.float32), np.where(y > 0.5, 1.0, wu).astype(np.float32),
                                              r, c, (r // bpx) * nbc + c // bpx, split, td.categories[split],
                                              td.channels, tgt_name, {"training_data": str(td.dir)})
            tr, va = sets["train"], sets["validation"]
            model = train_pixel_model(tr, va, cfg["models"]["xgb"], cfg["reproducibility"]["seed"])
            outs = model.save(mdir)
            outs.append(atomic_write_parquet(mdir / "feature_importance.parquet", model.importance()))
            hist = model.history.get("validation", {})
            vap = [v for v in (hist.get("aucpr") or hist.get("average_precision") or []) if np.isfinite(v)]
            pred = predict_full_grid(model, store, grid.valid)
            stats = {"n_train": int(len(tr.y)), "n_val": int(len(va.y)), "prevalence_train": float(tr.y.mean()),
                     "val_positives": int((va.y > 0.5).sum()), "n_features": len(td.channels), "backend": model.backend,
                     "fit_seconds": round(time.time() - t0, 1), "training_data": td.dir.name,
                     "validation_aucpr_best": float(max(vap)) if vap else None}
            return pred, stats, outs
        return self._train_deep(exp, res, target, dist, mdir)

    def _build_deep(self, exp: str, n_channels: int):
        from .experiments import DEEP_PATCH
        from .models.gated_fusion import GatedFusionNet
        from .models.segformer import SegFormer

        base, _, variant = exp.partition("__")
        if base == DEEP_PATCH:
            s = self.cfg["models"]["segformer"]
            return SegFormer(n_channels, s["dims"], s["depths"], s["heads"], decoder_dim=s["decoder_dim"],
                             drop_path=s["drop_path"])
        gf = self.cfg["models"]["gated_fusion"]
        return GatedFusionNet(n_channels, variant or gf["mode"], gf["geo_dim"], gf["dem_dim"], aux=False)

    def _train_deep(self, exp, res, target, dist, mdir):
        import torch

        from .datasets.patch_dataset import GeoPatchDataset, MultiresolutionPatchDataset, collate, patch_windows
        from .experiments import DEEP_MULTIRES, DEEP_PATCH
        from .training.losses import build_loss
        from .training.sampling import category_masks, mixture_sample
        from .training.trainer import Trainer, pixel_eval_fn, seed_everything

        cfg, grid, store = self.cfg, self.grid, self.store
        base, _, variant = exp.partition("__")
        seed_everything(cfg["reproducibility"]["seed"], cfg["reproducibility"]["deterministic"])
        channels = self.deep_channels()
        staging = {}
        try:
            staging = store.localize(channels, Path(cfg["scratch_root"]) / "feature_cache")
        except OSError as e:
            self.log(f"[{exp}] local staging unavailable ({e}); reading features from Drive")
        P = int(cfg["sampling"]["patch_px"])
        tr_x = self._patch_X(channels)
        td = self.training_data(exp, channels)                  # scaler fitted on TRAIN samples only (persisted)
        norm = SS.TrainOnlyNormalizer()
        norm.center = np.asarray(td.scaler["center"], np.float64)
        norm.scale = np.asarray(td.scaler["scale"], np.float64)
        norm.fitted_on = "train"
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
                    f"Use larger split regions or a smaller sampling.patch_px")
            atomic_write_parquet(mdir / f"patch_windows_{split}.parquet", pd.DataFrame(win) if not isinstance(
                win, pd.DataFrame) else win)
            if base == DEEP_MULTIRES:
                te = cfg["models"]["terrain_encoder"]
                ds = MultiresolutionPatchDataset(store, channels, target, win, res.code, split, norm, distance=dist,
                                                 dem_reader=self._dem_mosaic(float(te.get("dem_res_m", 25.0))), grid=grid,
                                                 dem_res_m=float(te.get("dem_res_m", 25.0)), dem_px=int(te.get("dem_px", 256)))
            else:
                ds = GeoPatchDataset(store, channels, target, win, res.code, split, norm, distance=dist)
            loaders[split] = torch.utils.data.DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                                                         shuffle=split == "train", collate_fn=collate, num_workers=0)
        loss_name, extra = cfg["training"]["loss"], {}
        model = self._build_deep(exp, len(channels))
        if base == DEEP_PATCH:
            if variant == "finetune":
                from .models.segformer import load_compatible_weights

                s = cfg["models"]["segformer"]
                rep = load_compatible_weights(model, str(s["pretrained_path"]))
                extra["pretrained_coverage"] = rep["coverage"]
                if rep["coverage"] == 0:
                    raise StageGateError(f"{exp}: pretrained weights {s['pretrained_path']} share no tensor with the "
                                         "G11 SegFormer (incompatible architecture) - remove pretrained_path")
            elif variant:
                loss_name = variant
        tr = Trainer(model, build_loss(loss_name, cfg["training"]["pos_weight"]), cfg, mdir, evaluate_fn=pixel_eval_fn)
        hist = tr.fit(loaders["train"], loaders["validation"], meta={"channels": channels, "experiment": exp,
                                                                     "loss": loss_name})
        best = mdir / "best.pt"
        if not best.is_file() and (mdir / "last.pt").is_file():
            extra["checkpoint_used"] = "last.pt (validation metric undefined: no positive validation pixel)"
            best = mdir / "last.pt"
        if best.is_file():
            model.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["model"])
        schema = atomic_write_json(mdir / "feature_schema.json", {
            "channels": channels, "normalizer_fit": "train split", "center": norm.center.tolist(),
            "scale": norm.scale.tolist(), "experiment": exp, "checkpoint": best.name})
        pred = self._predict_deep(exp, model, channels, norm, tr.device, tr_x)
        outs = [mdir / "history.csv", best, schema]
        return pred, {"epochs": int(len(hist)), "channels": len(channels), "best_val": float(hist.val_metric.max())
                      if len(hist) and np.isfinite(hist.val_metric).any() else None, "dem_branch": base == DEEP_MULTIRES,
                      "loss": loss_name,
                      "fusion_mode": (variant or cfg["models"]["gated_fusion"]["mode"]) if base == DEEP_MULTIRES else None,
                      "locally_staged_channels": int(sum(c in staging for c in channels)), "oom_batches": tr.oom, "training_data": td.dir.name,
                      **extra}, [o for o in outs if Path(o).is_file()]

    def _predict_deep(self, exp, model, channels, norm, device, tr_x=None) -> np.ndarray:
        import torch

        from .experiments import DEEP_PATCH
        from .inference.blending import predict_tiled

        cfg, grid = self.cfg, self.grid
        tr_x = tr_x if tr_x is not None else self._patch_X(channels)
        model.to(device).eval()
        X = np.stack([np.nan_to_num(((tr_x[i] - norm.center[i]) / norm.scale[i]).astype(np.float32))
                      for i in range(len(channels))])
        is_patch = exp.split("__")[0] == DEEP_PATCH

        @torch.no_grad()
        def fn(xb):
            t = torch.from_numpy(xb).to(device)
            o = model(t) if is_patch else model(t)["logits"]
            return torch.sigmoid(o)[:, 0].float().cpu().numpy()

        return predict_tiled(fn, X, grid.valid, cfg["inference"]["tile_px"], cfg["inference"]["overlap_px"],
                             cfg["inference"]["batch_size"])

    def _dem_mosaic(self, res_m: float):
        """Local (scratch) DEM mosaic at ``res_m`` for E3 patches, rebuilt only when the DEM catalog changes."""
        from .dem_reader import MosaicDEMReader, build_dem_mosaic

        reader = self.dem_reader()
        if reader is None or not len(reader.index.cat):
            return None
        key = (self.dem_catalog_signature() or "none")[:12]
        p = Path(self.cfg["scratch_root"]) / "dem_mosaic" / f"dem_{res_m:g}m_{key}.tif"
        if not p.is_file():
            cov = self.dem_coverage()["any"]
            build_dem_mosaic(reader, self.grid, cov, res_m, p, int(self.cfg["dem"]["aggregate_chunk_cells"]), self.log)
        return MosaicDEMReader(p)

    def _patch_X(self, channels):
        return [np.asarray(self.store.get(c), np.float32) for c in channels]

    # ================================================================== Phase 8: full-grid inference
    def predict_all(self) -> pd.DataFrame:
        """Every trained model of this namespace has its full-grid probability map; a map missing after a disconnect
        is regenerated from the saved model (XGBoost model.json, deep best.pt + saved normaliser)."""
        from .experiments import EXPERIMENTS
        from .registry import sha256_file

        rows = []
        mdir_root = self.stages.dir / self.ns
        for m in sorted(mdir_root.glob("train_*.json")) if mdir_root.is_dir() else []:
            rec = read_json(m) or {}
            exp = m.stem[len("train_"):]
            p = self.pred_path(exp)
            if rec.get("status") != "COMPLETE":
                rows.append({"experiment": exp, "status": f"NOT TRAINED ({rec.get('status')})"})
                continue
            want = next((o for o in rec.get("outputs", []) if o["path"] == str(p)), None)
            if p.is_file() and (want is None or sha256_file(p) == want["sha256"]):
                rows.append({"experiment": exp, "status": "PRESENT", "path": str(p)})
                continue
            kind = EXPERIMENTS[exp.split("__")[0]]["kind"]
            if kind == "pixel":
                from .models.xgb_baseline import load_pixel_model, predict_full_grid

                pred = predict_full_grid(load_pixel_model(self.model_dir(exp)), self.store, self.grid.valid)
            else:
                import torch

                sch = read_json(self.model_dir(exp) / "feature_schema.json")
                model = self._build_deep(exp, len(sch["channels"]))
                ck = torch.load(self.model_dir(exp) / sch.get("checkpoint", "best.pt"), map_location="cpu",
                                weights_only=False)
                model.load_state_dict(ck["model"])
                norm = SS.TrainOnlyNormalizer()
                norm.center, norm.scale = np.asarray(sch["center"]), np.asarray(sch["scale"])
                pred = self._predict_deep(exp, model, sch["channels"], norm,
                                          torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            write_geotiff(p, pred, self.grid, [f"{exp} probability"])
            key = self.skey(f"train_{exp}")              # same model, same fingerprint: refresh the output hashes
            self.stages.record(key, rec["fingerprint"], [Path(o["path"]) for o in rec.get("outputs", [])],
                               {**rec.get("stats", {}), "prediction_regenerated_utc":
                                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, validation=rec.get("validation"))
            rows.append({"experiment": exp, "status": "REGENERATED", "path": str(p)})
        t = pd.DataFrame(rows, columns=["experiment", "status", "path"])
        atomic_write_parquet(self.d("predictions") / "prediction_inventory.parquet", t)
        return t

    # ================================================================== Phase 9: fusion + evaluation
    def fusion(self) -> pd.DataFrame:
        """Pixel-level fusion of the G11 experiment maps: fitted on STACK, selected on VALIDATION (E5)."""
        from .eda.spatial import block_sample
        from .features.multiscale import pct_rank
        from .fusion import compare

        exps = [e for e in self.available_predictions() if not e.startswith("E5")]
        if len(exps) < 2:
            self.log(f"[fusion] SKIPPED: {len(exps)} member map(s) (need >= 2)")
            return pd.DataFrame([{"method": "-", "status": f"SKIPPED: {len(exps)} member(s)"}])
        key = self.skey("fusion")
        fp = self.sfp("fusion", ["reproducibility"], [(e, self.ctx.hashes.sha(self.pred_path(e))) for e in exps])
        if self._cached(key, fp) and self.pred_path("E5").is_file():
            return pd.read_csv(self.d("evaluate") / "fusion_comparison.csv")
        self.stages.begin(key, fp)
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
                atomic_write_text(self.d("evaluate") / "fusion_comparison.csv", table.to_csv(index=False))]
        md = self.model_dir("E5")
        outs.append(atomic_write_json(md / "metrics.json", {"selected_method": best, "members": exps,
                                                            "stack_ap": fit.get("stack_ap"),
                                                            "best_single_on_stack": fit.get("best_single")}))
        self.stages.record(key, fp, outs, {"selected": best, "members": exps})
        self.log(f"[fusion] selected {best} on validation among {list(ok.method)}")
        return table

    def evaluate(self, final: bool = False) -> pd.DataFrame:
        """Leaderboard on IDENTICAL support (block-sampled pixels of the split, same label definition)."""
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
        d = self.d("evaluate")
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
                             "dem_inputs": spec.get("dem"), "target_definition": tgt, "feature_version": self.feature_version,
                             "split_version": self.split_version, "train_region": "train",
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
                "selected_before_test": True, "feature_version": self.feature_version,
                "split_version": self.split_version})
        out = d / ("experiment_matrix_final.parquet" if final else "experiment_matrix.parquet")
        atomic_write_parquet(out, t)
        lb = atomic_write_text(d / ("leaderboard_final.csv" if final else "leaderboard.csv"), t.to_csv(index=False))
        self.stages.record(self.skey("evaluate" + ("_final" if final else "")), self.stages.fingerprint("evaluate", final),
                           [out, lb], {"n_rows": len(t), "experiments": exps})
        return t

    def _selected_probability(self):
        from .eda.spatial import block_sample
        from .training.calibration import Calibrator

        sel = read_json(self.d("evaluate") / "selection.json")
        if sel is None:
            raise StageGateError("run EVALUATE first (no selection.json)")
        p = _read_tif(self.pred_path(sel["selected_experiment"]))
        res = self.split_result()
        reg = res.mask(self.cfg["calibration"]["region"])
        r, c, _ = block_sample(reg, 2, 1, 100_000)
        cal = Calibrator(sel["calibration"]).fit(p[r, c], self.mask[r, c].astype(float), self.cfg["calibration"]["region"],
                                                 self.cfg["calibration"]["region"])
        return np.where(self.grid.valid, cal.transform(p), np.nan).astype(np.float32), sel

    # ================================================================== Phase 8b: candidates + segment ranker (E4)
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
        d = self.d("infer")
        p, sel = self._selected_probability()
        write_geotiff(self.d("predictions") / "selected_calibrated_probability.tif", p, grid,
                      [f"{sel['selected_experiment']} calibrated ({sel['calibration']})"])
        parts = [generate_candidates(p, grid.valid, grid, cfg, sel["selected_experiment"])]
        ev = self._get("geo_edge_rank")
        if ev is not None:
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
        outs = rk.save(self.d("train", "segment_ranker", "E4"))
        score = rk.score(ds.table)
        s = (score - score.min()) / max(score.max() - score.min(), 1e-12) if best_obj != "binary" else score
        ds.table["score"] = s
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
        man = {"selected_experiment": sel["selected_experiment"], "ranker": "E4", "ranker_objective": best_obj,
               "budget_km": best_budget, "n_candidates": int(len(out_c)),
               "budget_selected_on": "validation proxy DTI (offline, NOT the official DTI)"}
        outs.append(self.ctx.write_manifest(d / "inference_manifest.json", man))
        self.stages.record(self.skey("infer"), self.stages.fingerprint("infer", man), outs, man)
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

        man = read_json(self.d("infer") / "inference_manifest.json")
        if man is None:
            raise StageGateError("run INFER first")
        schema = load_official_schema(self.found["template"])
        cand = gpd.read_file(self.d("infer") / "candidates.gpkg")
        selm = select_by_length_budget(cand.score.to_numpy(), cand.geometry.length.to_numpy(), man["budget_km"])
        r = rasterize_segments(cand[selm], cand.score.to_numpy()[selm], self.grid, self.grid.valid)
        full = np.zeros(schema.shape, np.float32)
        if self.smoke:
            r0, r1, c0, c1 = self.smoke_window
            full[r0:r1, c0:c1] = np.nan_to_num(r)
        else:
            if r.shape != tuple(schema.shape):
                raise StageGateError(f"prediction grid {r.shape} != template {schema.shape}")
            full = np.nan_to_num(r)
        name = "g11_smoke_submission_NOT_FOR_UPLOAD.tif" if self.smoke else "g11_best_submission.tif"
        sd = self.d("submit")
        out = sd / name
        meta = {**man, "code_commit": self.ctx.commit, "config_hash": self.ctx.cfg_hash, "n_selected": int(selm.sum()),
                "selected_km": float(cand[selm].geometry.length.sum() / 1000), "smoke_test": self.smoke,
                "candidate_source": str(self.d("infer") / "candidates.gpkg"), "feature_version": self.feature_version,
                "split_version": self.split_version, "length_budget_km": man["budget_km"],
                "ranker_objective": man.get("ranker_objective"), "selected_model": man.get("selected_experiment")}
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
        self.stages.record(self.skey("submit"), self.stages.fingerprint("submit", man), outs,
                           {"official": official.get("status")}, validation=exp["validation"])
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
        "  I --> R[recover] --> F --> E", "  F --> SP[splits] --> T[train] --> V[evaluate] --> N[infer] --> S", "```", "",
        "Dashed = optional (skipped and recorded when absent). Stage records: manifests/<stage>.json.", ""])
