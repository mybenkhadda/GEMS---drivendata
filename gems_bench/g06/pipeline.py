"""G06 orchestrator: one class, every stage, dependency-checked, cached by fingerprint, resumable.

    P = G06Pipeline(cfg, RUN, HEAD)          # cfg = config.load_config(overrides={"notebook": CONFIG, "run": RUN})
    P.run()                                  # or the per-section methods used by the notebook

Stages (RUN flags in brackets) and dependencies::

    tests            synthetic pytest of tests/test_g06_*.py (notebook ``run_synthetic_tests``); TESTS_FAILED blocks
                     every training stage
    preflight        [PREFLIGHT]           source grid / exact bands (fails closed), artifacts, engine attach
    raw_eda          [RAW_EDA]             footprint, band statistics, raw atlas         <- preflight
    features         [FEATURE_ENGINEERING | BUILD_FEATURE_STORE]   tiled + global groups <- raw_eda
    feature_atlas    [FEATURE_ATLAS]       feature figures + correlations + profiles     <- features
    splits           [BUILD_SPLITS]        units, visibilities, isolation audit          <- preflight
    pixel_dataset    [BUILD_PIXEL_DATASET] D1 pool + labels per fit, D2 patch index      <- features, splits
    segment_dataset  [BUILD_SEGMENT_DATASET] D3 + nested known + D4                      <- features, splits
    train_pixel      [TRAIN_RANDOM_FOREST / TRAIN_LIGHTGBM / TRAIN_XGBOOST]              <- pixel_dataset, tests
    train_segment    [TRAIN_SEGMENT_MODEL]                                               <- segment_dataset, tests
    train_patch      [TRAIN_PATCH_UNET]                                                  <- pixel_dataset, tests
    train_meta       [TRAIN_META_MODEL]                                                  <- train_pixel, tests
    evaluate         [EVALUATE]            every available held-out map vs W2            <- splits
    full_inference   [FULL_INFERENCE]      FULL-fit research GeoTIFF + parity            <- pixel_dataset, tests
    export           [EXPORT_SUBMISSION]   gated, READY_FOR_MANUAL_UPLOAD at best        <- evaluate, full_inference
    report           [REPORT]              reports + artifact inventory (always from persisted tables)

A stage that is not requested is ADOPTED (REUSED_VALID_CACHE) when its persisted outputs validate against the
current fingerprint, otherwise it is NOT_REQUESTED and dependants are BLOCKED_UPSTREAM. A FAILED / blocked stage
supports no conclusion (``reports/EXECUTION_STATUS.md``).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import G06_CODE_REV, G06_VERSION
from . import persistence as PS
from .config import REPO, RUN_DEFAULTS, compute_run_id, stage_config, validate_config

BLOCKED_UPSTREAM, COMPLETE, FAILED, NOT_REQUESTED = PS.BLOCKED_UPSTREAM, PS.COMPLETE, PS.FAILED, PS.NOT_REQUESTED
NOT_RUN, REUSED, SKIPPED, VALID, Workspace = PS.NOT_RUN, PS.REUSED, PS.SKIPPED, PS.VALID, PS.Workspace

TESTS_FAILED = "TESTS_FAILED"
STAGES = ("tests", "preflight", "raw_eda", "features", "feature_atlas", "splits", "pixel_dataset",
          "segment_dataset", "train_pixel", "train_segment", "train_patch", "train_meta", "evaluate",
          "full_inference", "export", "report")
FLAGS = {"preflight": ("PREFLIGHT",), "raw_eda": ("RAW_EDA",),
         "features": ("FEATURE_ENGINEERING", "BUILD_FEATURE_STORE"), "feature_atlas": ("FEATURE_ATLAS",),
         "splits": ("BUILD_SPLITS",), "pixel_dataset": ("BUILD_PIXEL_DATASET",),
         "segment_dataset": ("BUILD_SEGMENT_DATASET",),
         "train_pixel": ("TRAIN_RANDOM_FOREST", "TRAIN_LIGHTGBM", "TRAIN_XGBOOST"),
         "train_segment": ("TRAIN_SEGMENT_MODEL",), "train_patch": ("TRAIN_PATCH_UNET",),
         "train_meta": ("TRAIN_META_MODEL",), "evaluate": ("EVALUATE",), "full_inference": ("FULL_INFERENCE",),
         "export": ("EXPORT_SUBMISSION",), "report": ("REPORT",)}
DEPENDS = {"raw_eda": ("preflight",), "features": ("raw_eda",), "feature_atlas": ("features",),
           "splits": ("preflight",), "pixel_dataset": ("features", "splits"),
           "segment_dataset": ("features", "splits"), "train_pixel": ("pixel_dataset", "tests"),
           "train_segment": ("segment_dataset", "tests"), "train_patch": ("pixel_dataset", "tests"),
           "train_meta": ("train_pixel", "tests"), "evaluate": ("splits",),
           "full_inference": ("pixel_dataset", "tests"), "export": ("evaluate", "full_inference")}
MODULES = {"tests": [], "preflight": ["inventory", "raster_io", "contracts"],
           "raw_eda": ["feature_raw", "raster_io"],
           "features": ["feature_raw", "feature_physics", "feature_texture", "feature_structure", "feature_context",
                        "feature_store", "raster_io", "contracts"],
           "feature_atlas": ["visualization"], "splits": ["splits"],
           "pixel_dataset": ["datasets", "sampling", "supervision", "splits"],
           "segment_dataset": ["datasets", "supervision", "feature_context"],
           "train_pixel": ["models_pixel", "supervision"], "train_segment": ["models_segment"],
           "train_patch": ["models_patch"], "train_meta": ["models_meta"],
           "evaluate": ["evaluation"], "full_inference": ["inference", "models_pixel"], "export": ["inference"],
           "report": []}
OUTPUTS = {
    "tests": [("audit/tests.csv", ("test", "status"))],
    "preflight": [("inventory/band_inventory.csv", ("identifier",)), ("inventory/dependency_inventory.csv",
                                                                       ("artifact", "status")),
                  ("inventory/artifact_compatibility.csv", None), ("inventory/missing_dependencies.json", None),
                  ("inventory/dependencies.csv", ("dependency", "status")), ("manifests/dependencies.json", None)],
    "raw_eda": [("inventory/raw_bands.csv", ("band", "frac_finite")), ("inventory/entropy_edges.json", None),
                ("inventory/footprint.npz", None)],
    "features": [("features/feature_registry.parquet", ("feature_id", "group", "family", "usable")),
                 ("features/feature_schema.json", None), ("features/feature_coverage.csv", ("feature",)),
                 ("manifests/feature_schema.json", None)],
    "feature_atlas": [("figures/feature_distributions/feature_correlations.png", None)],
    "splits": [("splits/units.csv", ("unit", "scenario", "zone")), ("splits/fits.csv", ("fit_id", "predicts")),
               ("splits/isolation_audit.csv", ("fit_id", "check", "status")),
               ("manifests/validation_contract.json", None)],
    "pixel_dataset": [("datasets/pixels/pool_meta.parquet", ("row_id", "row", "col")),
                      ("datasets/pixels/pool_features.parquet", ("row_id",)),
                      ("datasets/pixels/label_summary.csv", ("fit_id",)),
                      ("datasets/dataset_manifests/pixel.json", None),
                      ("datasets/patches/patch_index.parquet", ("patch_id", "r0", "c0"))],
    "segment_dataset": [("datasets/segments/segment_summary.csv", ("unit",)),
                        ("datasets/dataset_manifests/segment.json", None),
                        ("datasets/dataset_manifests/connection.json", None)],
    "train_pixel": [("models/pixel_registry.csv", ("experiment", "fit_id", "status"))],
    "train_segment": [("models/segment_registry.csv", ("experiment", "unit", "status"))],
    "train_patch": [("models/patch_registry.csv", ("experiment", "fit_id", "status"))],
    "train_meta": [("models/meta_registry.json", None)],
    "evaluate": [("evaluation/metrics.csv", ("model", "scenario", "dti")), ("evaluation/per_zone.csv", None),
                 ("evaluation/system_recovery.csv", None), ("evaluation/ablations.csv", None),
                 ("evaluation/model_comparison.csv", None), ("evaluation/promotion_decisions.csv", None)],
    "full_inference": [("predictions/full_region/full_inference_manifest.json", None)],
    "export": [("submissions/submission_manifest.json", None)],
    "report": [("reports/EXECUTION_STATUS.md", None)],
}
MERGED = ("train_pixel", "train_segment", "train_patch")        # registries merged across runs (never archived)
# whole directories archived (moved, not copied) when a successful stage is recomputed with another fingerprint
ARCHIVE_EXTRA = {"features": ["features"], "pixel_dataset": ["datasets/pixels", "datasets/patches"],
                 "segment_dataset": ["datasets/segments", "datasets/connections"], "splits": ["splits"],
                 "evaluate": ["evaluation"]}


class G06Pipeline:
    def __init__(self, config: dict, run_flags: dict | None = None, repository_commit: str | None = None,
                 factory=None, progress=print):
        self.cfg = config
        if run_flags is not None:
            self.cfg["run"] = {**RUN_DEFAULTS, **run_flags}
        self.warnings = validate_config(self.cfg)
        self.nb, self.fs, self.bm = self.cfg["notebook"], self.cfg["feature_store"], self.cfg["benchmarks"]
        self.run_flags = self.cfg["run"]
        self.head = repository_commit or "unknown"
        self.say = progress or (lambda *_: None)
        self.local_root, self.drive_root = Path(self.nb["local_root"]), Path(self.nb["drive_root"])
        self._factory, self._engine, self._units, self._store, self._engine_error = factory, None, None, None, None
        self.src = self._resolve_sources()
        self.src_fp = {k: self._fingerprint_source(p) for k, p in self.src.items() if p is not None}
        self.run_id = compute_run_id(self.cfg, self.src_fp, self.head)
        self.ws = Workspace(self.local_root, self.drive_root, f"{self.nb['output_subdir']}/{self.run_id}",
                            bool(self.nb["sync_to_drive"]))
        self.root = self.ws.root
        self.ws.stage_in(["manifests", "inventory", "splits", "features", "datasets", "models", "predictions",
                          "evaluation", "reports", "audit", "figures", "submissions"])
        self.status = {s: NOT_RUN for s in STAGES}
        self.fps: dict = {}
        self.errors: list = []
        for w in self.warnings:
            self.say(f"[g06] config warning: {w}")
        self.save_run_manifest()

    # =========================================================================== plumbing
    def _resolve_sources(self) -> dict:
        from .inventory import resolve_source

        out = {}
        for key in (self.fs.get("sources") or {}):
            p = resolve_source(self.fs, key, [self.local_root, self.drive_root, REPO])
            if p is not None and str(p).startswith(str(self.drive_root)):
                rel = p.relative_to(self.drive_root)
                dst = self.local_root / rel
                if not dst.exists() or dst.stat().st_size != p.stat().st_size:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
                p = dst
            out[key] = p
        return out

    def _fingerprint_source(self, p: Path) -> dict:
        cache = self.local_root / ".g06_source_sha.json"
        c = PS.read_json(cache) or {}
        st = p.stat()
        k = f"{p}|{st.st_size}|{st.st_mtime_ns}"
        if k not in c:
            c[k] = PS.sha256_file(p)
            cache.parent.mkdir(parents=True, exist_ok=True)
            PS.write_json(cache, c)
        return {"bytes": int(st.st_size), "sha256": c[k], "path": str(p)}

    def requested(self, stage: str) -> bool:
        if stage == "tests":
            return bool(self.nb.get("run_synthetic_tests", True))
        return any(bool(self.run_flags.get(f)) for f in FLAGS.get(stage, ()))

    def fp_of(self, stage: str) -> str | None:
        if stage in self.fps:
            return self.fps[stage]
        m = PS.load_cached_stage(self.root, stage)
        return (m or {}).get("fingerprint")

    def _dep_ok(self, dep: str) -> bool:
        st = self.status.get(dep)
        if dep == "tests":
            return st not in (TESTS_FAILED, FAILED)
        if st in VALID:
            return True
        return False

    def _run(self, stage: str, parts: list, fn, *, force: bool = False) -> dict:
        """Dependency check -> (adopt | validated cache | archive + run) -> manifest -> Drive push."""
        bad = [d for d in DEPENDS.get(stage, ()) if not self._dep_ok(d)]
        fp = PS.stable_hash(G06_VERSION, stage, parts, PS.code_identity(MODULES.get(stage, [])))
        self.fps[stage] = fp
        outs = OUTPUTS.get(stage, [])
        reusable, probs = PS.validate_cached_stage(self.root, stage, fp, outs)
        if not self.requested(stage):
            if reusable:
                return self._set(stage, REUSED, fp, {"note": "not requested; valid cache adopted"})
            return self._set(stage, NOT_REQUESTED, fp, {"note": f"RUN flags {FLAGS.get(stage)} are False and no "
                                                                f"valid cache ({'; '.join(probs[:2])})"}, save=False)
        if bad:
            return self._set(stage, BLOCKED_UPSTREAM, fp, {"reason": f"upstream {[(d, self.status.get(d)) for d in bad]}"})
        if reusable and not force and self.nb.get("reuse_cached", True):
            return self._set(stage, REUSED, fp, {"note": f"fingerprint {fp[:10]}"})
        prev = PS.load_cached_stage(self.root, stage)
        if prev and prev.get("status") == COMPLETE and prev.get("fingerprint") != fp and stage not in MERGED:
            moved = PS.archive_outputs(self.root, stage, ARCHIVE_EXTRA.get(stage, []) + [o[0] for o in outs],
                                       prev.get("fingerprint"))
            if moved:
                self.say(f"[{stage}] earlier outputs archived to history/{stage}/{str(prev.get('fingerprint'))[:10]}")
        t0 = time.perf_counter()
        self.say(f"[{stage}] running ...")
        try:
            st, info = fn()
        except Exception as e:
            st = FAILED
            info = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-5000:]}
            self.errors.append({"stage": stage, **info, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if st == COMPLETE:
            _, probs = PS.check_outputs(self.root, outs)
            if probs:
                st, info = FAILED, {**info, "error": f"required outputs invalid: {probs[:4]}"}
        info["seconds"] = round(time.perf_counter() - t0, 1)
        return self._set(stage, st, fp, info)

    def _set(self, stage: str, st: str, fp, info: dict, save: bool = True) -> dict:
        self.status[stage] = st
        if save:
            PS.save_stage_manifest(self.root, stage, st, fp, OUTPUTS.get(stage, []), info, code_rev=G06_CODE_REV,
                                   version=G06_VERSION, commit=self.head, run_id=self.run_id)
        self.save_run_manifest()
        note = info.get("note") or info.get("reason") or info.get("error") or ""
        self.say(f"[{stage}] {st}" + (f" | {str(note)[:300]}" if note else ""))
        if st == FAILED and info.get("traceback"):
            print(info["traceback"][-1500:])
        if save:
            try:
                self.ws.push(["manifests"])
            except Exception as e:
                self.say(f"[{stage}] Drive push of manifests failed: {type(e).__name__}: {e}")
        return {"stage": stage, "status": st, **info}

    def push(self, rels) -> None:
        try:
            self.ws.push(rels)
        except Exception as e:
            self.say(f"[g06] Drive push failed ({type(e).__name__}: {e}); local outputs are intact")

    def save_run_manifest(self) -> None:
        PS.write_json(self.root / "manifests" / "run.json", {
            "run_id": self.run_id, "version": G06_VERSION, "code_rev": G06_CODE_REV, "commit": self.head,
            "sources": self.src_fp, "notebook_config": self.nb, "run_flags": self.run_flags,
            "stages": getattr(self, "status", {}), "fingerprints": getattr(self, "fps", {}),
            "n_errors": len(getattr(self, "errors", [])), "drive_dir": str(self.ws.drive_path()),
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")})
        PS.write_json(self.root / "manifests" / "runtime_errors.json", getattr(self, "errors", []))

    # =========================================================================== shared context
    def engine(self):
        from .inventory import EngineUnavailable, attach_engine

        if self._engine is None:
            if self._engine_error:
                raise EngineUnavailable(self._engine_error)
            try:
                self._engine = attach_engine(self.nb, self.head, self._factory)
            except EngineUnavailable as e:
                self._engine_error = str(e)
                raise
        return self._engine

    def units(self) -> dict:
        if self._units is None:
            from .splits import build_validation_units

            sc = tuple(self.bm["validation"]["scenarios"])
            self._units = {vu.key: vu for vu in build_validation_units(self.engine(), sc)}
        return self._units

    def store(self):
        from .feature_store import open_feature_store

        if self._store is None:
            self._store = open_feature_store(self.root)
        return self._store

    def grid(self):
        import rasterio

        from .contracts import GridSpec

        with rasterio.open(self.src["features"]) as ds:
            return GridSpec.from_dataset(ds)

    def band_index(self) -> dict:
        import rasterio

        from .raster_io import resolve_bands

        with rasterio.open(self.src["features"]) as ds:
            return resolve_bands(ds, list(self.fs["bands"]))

    def engine_identity(self) -> dict:
        return self.engine().identity()

    def fig_dir(self, sub: str) -> Path:
        return self.root / "figures" / sub

    def figures_on(self) -> bool:
        return bool(self.nb.get("save_figures", True))

    # =========================================================================== tests
    def run_tests(self) -> dict:
        def fn():
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfE", *sorted(
                str(p) for p in (REPO / "tests").glob("test_g06_*.py"))], cwd=str(REPO), capture_output=True,
                               text=True)
            out = r.stdout + "\n" + r.stderr
            self.ws.save_text(f"exit code {r.returncode}\n\n{out}", "audit/pytest_output.txt")
            import re

            fails = [m.group(1) for m in re.finditer(r"^(?:FAILED|ERROR) (\S+)", out, re.M)]
            T = pd.DataFrame([{"test": "pytest tests/test_g06_*.py", "status": "PASS" if r.returncode == 0 else
                               "FAIL", "detail": f"exit {r.returncode}"}] + [{"test": f, "status": "FAIL",
                                                                              "detail": ""} for f in fails])
            self.ws.save_table(T, "audit/tests.csv")
            if r.returncode != 0:
                return TESTS_FAILED, {"error": f"pytest exit code {r.returncode}: {fails[:5] or out[-400:]}"}
            return COMPLETE, {"note": "all G06 tests passed"}
        if not self.requested("tests"):
            return self._set("tests", SKIPPED, None, {"note": "run_synthetic_tests=False (conclusions carry "
                                                              "'tests not run')"})
        return self._run("tests", [time.strftime("%Y-%m-%dT%H:%M")], fn, force=True)

    # =========================================================================== preflight
    def preflight(self) -> dict:
        def fn():
            from . import inventory as INV

            if self.src.get("features") is None:
                return FAILED, {"error": "GeoDAWN feature raster not found under local / Drive roots: "
                                         f"{self.fs['sources']['features']['candidates']}"}
            S = INV.inspect_source_raster(self.src["features"], self.fs["bands"], self.fs["grid"])
            T = S["band_table"].merge(S["resolution"].rename(columns={"requested": "identifier"}), on="identifier",
                                      how="outer")
            self.ws.save_table(T, "inventory/band_inventory.csv")
            A = INV.discover_existing_artifacts(self.drive_root, self.fs, self.local_root)
            self.ws.save_table(A, "inventory/dependency_inventory.csv")
            C = INV.validate_artifact_compatibility(A, S["grid"])
            self.ws.save_table(C, "inventory/artifact_compatibility.csv")
            try:
                eng = self.engine()
                es = {"attached": True, "detail": json.dumps(eng.identity(), default=str)[:400]}
            except INV.EngineUnavailable as e:
                es = {"attached": False, "detail": str(e)}
            D, missing = INV.build_dependency_report(S, A, C, es)
            self.ws.save_table(D, "inventory/dependencies.csv")
            self.ws.save_json(missing, "inventory/missing_dependencies.json")
            self.ws.save_json({"grid": S["grid"].to_dict(), "band_index": S["band_index"], "sources": self.src_fp,
                               "engine": es, "problems": S["problems"], "missing": missing},
                              "manifests/dependencies.json")
            if S["problems"]:
                return FAILED, {"error": "source raster violates the contract (fail closed): " +
                                         "; ".join(S["problems"])[:800]}
            return COMPLETE, {"note": f"grid {S['grid'].shape}, {len(S['band_index'])} bands resolved exactly; "
                                      f"engine {'attached' if es['attached'] else 'NOT attached'}; "
                                      f"{(A.status == 'FOUND').sum()}/{len(A)} artifacts found"}
        return self._run("preflight", [self.src_fp, stage_config(self.cfg, "preflight")], fn, force=True)

    # =========================================================================== raw EDA
    def raw_eda(self) -> dict:
        def fn():
            from .feature_raw import band_stats, footprint_of
            from .raster_io import read_band_full

            bi = self.band_index()
            nd = self.fs.get("nodata") or {}
            fp = footprint_of(self.src["features"], len(bi), float(nd.get("footprint_min_share", 0.5)),
                              float(nd.get("abs_max", 1e30)))
            self.ws.save_npz("inventory/footprint.npz", footprint=np.packbits(fp), shape=np.asarray(fp.shape))
            stats, edges = band_stats(self.src["features"], bi, fp,
                                      int((self.fs["groups"].get("texture") or {}).get("entropy_levels", 16)))
            stats["modality"] = stats.band.map(lambda b: (self.fs["bands"].get(b) or {}).get("modality", ""))
            self.ws.save_table(stats, "inventory/raw_bands.csv")
            self.ws.save_json(edges, "inventory/entropy_edges.json")
            if self.figures_on():
                self._raw_figures(bi, fp, read_band_full)
            self.push(["inventory", "figures"])
            const = list(stats.band[stats.constant.astype(bool)])
            return COMPLETE, {"note": f"footprint {int(fp.sum())} px ({fp.mean():.3f}); constant bands {const}"}
        return self._run("raw_eda", [self.src_fp, stage_config(self.cfg, "raw_eda"), self.fp_of("preflight")], fn)

    def _raw_figures(self, bi: dict, fp: np.ndarray, read_band_full) -> None:
        from . import visualization as V

        f = int((self.fs.get("atlas") or {}).get("downsample", 8))
        small, units = {}, {}
        for b, i in bi.items():
            a = read_band_full(self.src["features"], i)
            small[b] = np.where(fp, a, np.nan)[::f, ::f]
            units[b] = (self.fs["bands"].get(b) or {}).get("unit", "")
            del a
        faults = self._faults_small(f)
        V.plot_feature_atlas(small, self.fig_dir("raw_atlas"), "raw_bands_atlas",
                             "GeoDAWN 19 bands (observed values; mapped faults outlined)", units, 1, 5, faults)
        pairs = [b for b in self.fs.get("atlas", {}).get("mag_grav_panels", []) if b in small]
        if pairs:
            V.plot_feature_atlas({b: small[b] for b in pairs}, self.fig_dir("raw_atlas"), "magnetic_vs_gravity",
                                 "Magnetic vs gravity anomalies and their provided gradients", units, 1, 2)
        V.plot_missingness(small, fp[::f, ::f], self.fig_dir("raw_atlas"), "missingness", 1)

    def _faults_small(self, f: int):
        import rasterio

        p = self.src.get("faults")
        if p is None:
            return None
        with rasterio.open(p) as ds:
            if (ds.height, ds.width) != tuple(self.fs["grid"][k] for k in ("height", "width")):
                return None
            a = ds.read(1, masked=True)
        lab = np.ma.filled(a.astype(np.float32), np.nan) >= 1
        return lab[::f, ::f]

    # =========================================================================== features
    def prepare_features(self) -> dict:
        """Raw EDA + feature engineering + feature-store construction + atlas (notebook sections 05-11)."""
        out = {"raw_eda": self.raw_eda(), "features": self.build_feature_store()}
        out["feature_atlas"] = self.feature_atlas()
        return out

    def build_feature_store(self) -> dict:
        def fn():
            from .contracts import GridSpec
            from .feature_store import build_feature_store, open_feature_store, validate_feature_store

            bi = self.band_index()
            with np.load(self.root / "inventory" / "footprint.npz") as z:
                shape = tuple(int(x) for x in z["shape"])
                fp = np.unpackbits(z["footprint"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
            stats = self.ws.load_table("inventory/raw_bands.csv")
            edges = self.ws.read_json("inventory/entropy_edges.json") or {}
            bt = self.ws.load_table("inventory/band_inventory.csv")
            grid = self.grid()
            summ = build_feature_store(self.src["features"], bi, grid, self.fs, self.root, fp, stats, edges, bt,
                                       PS.stable_hash(self.fps["features"]), G06_VERSION,
                                       reuse=bool(self.nb.get("reuse_cached", True)), progress=self.say)
            self._store = None
            st = open_feature_store(self.root)
            probs = validate_feature_store(st, GridSpec(**grid.to_dict()), int(self.fs["grid"]["epsg"]), bi)
            shutil.copy2(self.root / "features" / "feature_coverage.csv", self.root / "inventory" /
                         "feature_coverage.csv")
            shutil.copy2(self.root / "features" / "feature_schema.json", self.root / "manifests" /
                         "feature_schema.json")
            self.push(["features", "inventory", "manifests"])
            if probs:
                return FAILED, {"error": f"feature store invalid: {probs[:5]}"}
            return COMPLETE, {**summ, "note": f"{summ['n_features']} channels ({summ['n_usable']} usable), halo "
                                              f"{summ['halo_effective_px']} px, schema {summ['schema_hash']}",
                              "feature_schema_hash": summ["schema_hash"]}
        return self._run("features", [stage_config(self.cfg, "features"), self.src_fp, self.fp_of("raw_eda")], fn)

    def feature_atlas(self) -> dict:
        def fn():
            from . import visualization as V

            st = self.store()
            f = int((self.fs.get("atlas") or {}).get("downsample", 8))
            fp = st.footprint()
            R = st.registry
            groups = {"magnetic": "physical_features", "gravity": "physical_features",
                      "multiscale": "physical_features", "magnetic_spectral": "physical_features",
                      "gravity_spectral": "physical_features", "texture": "feature_distributions",
                      "structural": "structure", "context": "feature_distributions",
                      "context_global": "feature_distributions"}
            per = int((self.fs.get("atlas") or {}).get("per_group", 12))
            for g, sub in groups.items():
                ids = list(R.feature_id[(R.group == g) & R.usable.astype(bool)])
                if not ids:
                    continue
                pick = [ids[i] for i in np.linspace(0, len(ids) - 1, min(per, len(ids))).astype(int)]
                arrays = {n: st.load([n])[0][::f, ::f] for n in pick}
                units = dict(zip(R.feature_id, R.unit))
                V.plot_feature_atlas(arrays, self.fig_dir(sub), f"atlas_{g}", f"Feature group {g}", units, 1, 4)
            rng = np.random.default_rng(int(self.nb["seed"]))
            idx = np.flatnonzero(fp.ravel())
            pick = np.sort(rng.choice(idx, min(50000, idx.size), replace=False))
            r, c = np.unravel_index(pick, fp.shape)
            names = st.usable([n for n in st.names if not n.startswith("phys_")])
            X = pd.DataFrame(st.sample(names, r, c), columns=names)
            V.plot_feature_correlations(X, self.fig_dir("feature_distributions"), "feature_correlations",
                                        int((self.fs.get("atlas") or {}).get("max_correlation_features", 60)))
            prof = [n for n in (self.fs.get("atlas") or {}).get("profile_features", []) if n in st.names]
            lab = self._faults_full()
            if prof and lab is not None:
                from scipy import ndimage

                dist = (ndimage.distance_transform_edt(~lab) * 0.1).astype(np.float32)
                V.plot_fault_profiles(dist, {n: st.load([n])[0] for n in prof}, fp, self.fig_dir("structure"),
                                      "fault_profiles")
            self.push(["figures"])
            return COMPLETE, {"note": f"atlas of {len(groups)} groups, correlations of {len(names)} features"}
        return self._run("feature_atlas", [self.fp_of("features"), stage_config(self.cfg, "feature_atlas")], fn)

    def _faults_full(self):
        import rasterio

        p = self.src.get("faults")
        if p is None:
            return None
        with rasterio.open(p) as ds:
            if (ds.height, ds.width) != tuple(self.fs["grid"][k] for k in ("height", "width")):
                return None
            a = ds.read(1, masked=True)
        return np.ma.filled(a.astype(np.float32), np.nan) >= 1

    # =========================================================================== splits
    def build_splits(self) -> dict:
        def fn():
            from . import splits as SP

            eng = self.engine()
            U = self.units()
            val = self.bm["validation"]
            G, D = SP.load_geographic_folds(eng), SP.load_discovery_splits(eng)
            self.ws.save_table(SP.unit_table(list(U.values())), "splits/units.csv")
            fm = G["fold_map"]
            self.ws.save_table(pd.DataFrame([{"fold": k, "px": int((fm == k).sum()),
                                              "label_px": int((eng.labels() & (fm == k)).sum())}
                                             for k in range(G["n_folds"])]), "splits/geographic/fold_summary.csv")
            self.ws.save_table(pd.DataFrame([{"split": d, "withheld_px": int(D["splits"].withheld(d).sum()),
                                              "eval_zone_px": int(D["splits"].eval_zone(d).sum())}
                                             for d in range(D["n_splits"])]), "splits/discovery/split_summary.csv")
            nested = bool(val.get("nested_pixel_experiments"))
            fits = SP.fit_ids(D["n_splits"], G["n_folds"], val["scenarios"], nested,
                              tuple(val.get("nested_scenarios", ("V1", "V3"))))
            rows, audit, inner = [], [], []
            for fid in fits + ["FULL"]:
                vis = SP.build_training_visibility(eng, fid, val, tuple(val["scenarios"]))
                rows.append(SP.visibility_row(vis))
                outer = U.get(f"{vis.scenario}:{vis.outer}") if vis.inner is not None else None
                audit += SP.verify_fold_isolation(eng, vis, U, outer)
                ic = self.bm.get("inner_cv") or {}
                im = SP.make_inner_folds(vis.region, vis.visible, float(ic.get("block_km", 20)),
                                      int(ic.get("n_folds", 3)), int(self.nb["seed"]))
                for f in range(int(ic.get("n_folds", 3))):
                    inner.append({"fit_id": fid, "inner_fold": f, "px": int((im == f).sum()),
                                  "visible_px": int(((im == f) & vis.visible).sum())})
                del vis, im
            self.ws.save_table(pd.DataFrame(rows), "splits/fits.csv")
            A = pd.DataFrame(audit)
            self.ws.save_table(A, "splits/isolation_audit.csv")
            self.ws.save_table(pd.DataFrame(inner), "splits/inner_cv/inner_cv_summary.csv")
            self.ws.save_json({"primary": "V1 withheld complete systems (disc_official)",
                               "secondary": "V3 geographic folds", "diagnostic": "V2 network completion",
                               "fits": rows, "v1_training_region": val.get("v1_training_region"),
                               "rules": ["a fit predicts only the units in 'predicts'",
                                         "hidden labels are never positives, negatives or sampling strata",
                                         "pixel models never use known-fault context",
                                         "segment / meta models train on other units' truth only",
                                         "W2 / CNN-based learned models are SCREENING_ONLY"],
                               "n_fail": int((A.status != "PASS").sum()) if len(A) else 0},
                              "manifests/validation_contract.json")
            self.push(["splits", "manifests"])
            bad = A[A.status != "PASS"] if len(A) else A
            if len(bad):
                return FAILED, {"error": f"isolation checks failed: {bad.head(5).to_dict('records')}"}
            return COMPLETE, {"note": f"{len(U)} units, {len(rows)} fits, {len(A)} isolation checks PASS"}
        parts = [stage_config(self.cfg, "splits"), self.fp_of("preflight")]
        try:
            parts.append(self.engine_identity())
        except Exception as e:
            parts.append(f"engine unavailable: {e}")
        return self._run("splits", parts, fn)

    # =========================================================================== datasets
    def build_datasets(self) -> dict:
        return {"pixel": self.build_pixel_dataset(), "segment": self.build_segment_dataset()}

    def _fit_list(self) -> list[str]:
        F = self.ws.load_table("splits/fits.csv")
        return list(F.fit_id) if len(F) else []

    def build_pixel_dataset(self) -> dict:
        def fn():
            from .datasets import build_patch_index, build_pixel_pool, dataset_manifest, label_pool_for_fit, unit_file
            from .splits import build_training_visibility

            eng, st = self.engine(), self.store()
            sm = np.asarray(eng.ctx.splits.system_map)
            meta, X = build_pixel_pool(st, eng.labels(), sm, self.bm["sampling"])
            self.ws.save_table(meta, "datasets/pixels/pool_meta.parquet")
            self.ws.save_table(X, "datasets/pixels/pool_features.parquet")
            summ = []
            val = self.bm["validation"]
            for fid in self._fit_list():
                vis = build_training_visibility(eng, fid, val, tuple(val["scenarios"]))
                L = label_pool_for_fit(meta, vis, eng.valid(), sm, self.bm["supervision"], self.bm["sampling"],
                                       self.bm.get("inner_cv") or {}, int(self.nb["seed"]))
                self.ws.save_table(L, f"datasets/pixels/labels/{unit_file(fid)}.parquet")
                g = L.groupby("cls").agg(n=("row_id", "size"), weight=("weight", "sum"))
                summ.append({"fit_id": fid, **{f"n_cls{int(k)}": int(v) for k, v in g.n.items()},
                             **{f"w_cls{int(k)}": float(v) for k, v in g.weight.items()},
                             "n_systems": int(L.loc[L.cls == 1, "system_id"].nunique())})
                del vis
            S = pd.DataFrame(summ)
            self.ws.save_table(S, "datasets/pixels/label_summary.csv")
            feats = [c for c in X.columns if c != "row_id"]
            self.ws.save_json(dataset_manifest("pixel", "pixel", feats, ["target", "cls", "weight"], list(meta.columns),
                                               len(meta), self.fps["pixel_dataset"], st.schema_hash,
                                               "datasets/pixels", {"fits": list(S.fit_id), "strata": meta.stratum
                                                                   .value_counts().to_dict()}),
                              "datasets/dataset_manifests/pixel.json")
            pc = (self.bm.get("patch") or {}).get("index") or {}
            ez = {d: eng.ctx.splits.eval_zone(d) for d in range(eng.ctx.splits.n_splits)}
            PI = build_patch_index(st.footprint(), int(pc.get("patch_px", 128)), int(pc.get("stride_px", 128)),
                                   np.asarray(eng.ctx.folds.fold_map), ez, float(pc.get("min_footprint", 0.5)))
            self.ws.save_table(PI, "datasets/patches/patch_index.parquet")
            self.ws.save_json(dataset_manifest("patch", "patch", [], ["centreline"], list(PI.columns), len(PI),
                                               self.fps["pixel_dataset"], st.schema_hash, "datasets/patches"),
                              "datasets/dataset_manifests/patch.json")
            if self.figures_on():
                self._training_figures(meta, X)
            self.push(["datasets", "figures"])
            return COMPLETE, {"note": f"pool {len(meta)} rows x {len(feats)} features; {len(S)} fits labelled; "
                                      f"{len(PI)} patches", "feature_schema_hash": st.schema_hash}
        parts = [stage_config(self.cfg, "pixel_dataset"), self.fp_of("features"), self.fp_of("splits")]
        return self._run("pixel_dataset", parts, fn)

    def _training_figures(self, meta: pd.DataFrame, X: pd.DataFrame) -> None:
        from . import visualization as V
        from .datasets import unit_file

        fid = next((f for f in self._fit_list() if f.startswith("V1:") and "|" not in f), None)
        if fid is None:
            return
        L = self.ws.load_table(f"datasets/pixels/labels/{unit_file(fid)}.parquet")
        J = L.merge(X, on="row_id")
        feats = [n for n in (self.fs.get("atlas") or {}).get("profile_features", []) if n in J.columns][:12]
        if feats:
            V.plot_class_distributions(J[feats], J.cls, feats, self.fig_dir("training"), f"class_distributions_"
                                       f"{unit_file(fid)}")
        V.plot_sample_locations(meta, L, self.store().grid.shape, self.fig_dir("training"),
                                f"sample_locations_{unit_file(fid)}")

    def build_segment_dataset(self) -> dict:
        def fn():
            from . import datasets as DS
            from .supervision import build_connection_supervision

            unit_file, dataset_manifest = DS.unit_file, DS.dataset_manifest
            eng, st, U = self.engine(), self.store(), self.units()
            cfg = self.bm["segment_candidates"]
            sup = self.bm["supervision"]
            kcfg = self.fs.get("known_context") or {}
            summ, nested_rows = [], 0
            labs = {}
            for key in sorted(U):
                vu = U[key]
                out = DS.build_segment_dataset(eng, st, vu, cfg, sup["segment"], kcfg)
                f = unit_file(key)
                self.ws.save_table(out["T"], f"datasets/segments/units/{f}.parquet")
                self.ws.save_table(out["L"], f"datasets/segments/units/{f}_labels.parquet")
                self.ws.save_npz(f"datasets/segments/units/{f}_lab.npz", lab=out["lab"],
                                 window=np.asarray(out["provenance"]["window"]))
                self.ws.save_json(out["provenance"], f"datasets/segments/units/{f}_provenance.json")
                labs[key] = out["lab"]
                L = out["L"]
                ev = {"w2_rank": eng.w2_rank(vu.base)} if cfg.get("track_b_evidence", True) else {}
                P = DS.build_connection_dataset(out["lab"], out["T"], st, out["provenance"]["window"],
                                                sup["connection"], ev, key)
                if len(P):
                    gs = DS.gap_near_share(P, vu.eval_truth(), float(sup["segment"]["tol_px"]))
                    dom = L.system_id.to_numpy() if len(L) else np.zeros(0, int)
                    PL = build_connection_supervision(P, L, dom, gs, sup["connection"])
                    lab_cols = ["unit", "label_a", "label_b", "target", "cls", "label_conf", "same_system",
                                "gap_near_share"]
                    self.ws.save_table(P.drop(columns=["gap_r", "gap_c"]),
                                       f"datasets/connections/units/{f}.parquet")
                    self.ws.save_table(PL[lab_cols], f"datasets/connections/units/{f}_labels.parquet")
                summ.append({"unit": key, "n_candidates": len(out["T"]),
                             "n_pos": int((L.cls == "POSITIVE").sum()) if len(L) else 0,
                             "n_neg": int((L.cls == "NEGATIVE").sum()) if len(L) else 0,
                             "n_amb": int((L.cls == "AMBIGUOUS").sum()) if len(L) else 0, "n_pairs": len(P),
                             "pools": ",".join(out["pools"])})
                eng.release(vu.base)
                self.say(f"  segments {key}: {len(out['T'])} candidates, {len(P)} pairs")
            for key, vu in U.items():
                if vu.scenario not in ("V1", "V3"):
                    continue
                for other in U.values():
                    if other.scenario != vu.scenario or other.zone == vu.zone:
                        continue
                    K = DS.nested_known_table(eng, vu, other.zone, labs[key], kcfg)
                    self.ws.save_table(K, f"datasets/segments/nested_known/{vu.scenario}_{other.zone}__"
                                          f"{vu.zone}.parquet")
                    nested_rows += len(K)
            S = pd.DataFrame(summ)
            self.ws.save_table(S, "datasets/segments/segment_summary.csv")
            T0 = self.ws.load_table(f"datasets/segments/units/{unit_file(sorted(U)[0])}.parquet") if U else \
                pd.DataFrame()
            feat = [c for c in T0.columns if c not in ("unit", "scenario", "zone", "label", "row", "col", "r_full",
                                                         "c_full", "seg_id")]
            self.ws.save_json(dataset_manifest("segment", "segment", feat, ["target", "cls", "near_share"],
                                               ["unit", "label", "seg_id"], int(S.n_candidates.sum()),
                                               self.fps["segment_dataset"], st.schema_hash, "datasets/segments",
                                               {"pools": cfg.get("pools"), "nested_known_rows": nested_rows}),
                              "datasets/dataset_manifests/segment.json")
            self.ws.save_json(dataset_manifest("connection", "connection", [], ["target", "cls"],
                                               ["unit", "label_a", "label_b"], int(S.n_pairs.sum()),
                                               self.fps["segment_dataset"], st.schema_hash, "datasets/connections"),
                              "datasets/dataset_manifests/connection.json")
            self.push(["datasets"])
            return COMPLETE, {"note": f"{int(S.n_candidates.sum())} candidates in {len(S)} units, "
                                      f"{int(S.n_pairs.sum())} pairs", "feature_schema_hash": st.schema_hash}
        parts = [stage_config(self.cfg, "segment_dataset"), self.fp_of("features"), self.fp_of("splits")]
        return self._run("segment_dataset", parts, fn)

    # =========================================================================== pixel models
    def _pixel_experiments(self) -> dict:
        flag = {"random_forest": "TRAIN_RANDOM_FOREST", "lightgbm": "TRAIN_LIGHTGBM", "xgboost": "TRAIN_XGBOOST"}
        return {e: c for e, c in self.bm["pixel_experiments"].items() if self.run_flags.get(flag[c["model"]])}

    def pixel_registry(self) -> pd.DataFrame:
        return self.ws.load_table("models/pixel_registry.csv")

    def train_pixel_models(self) -> dict:
        def fn():
            from .models_pixel import crossfit_pixel_model

            st, U = self.store(), self.units()
            fits = [f for f in self._fit_list() if f != "FULL"]
            nested_exps = set(self.bm["validation"].get("nested_pixel_experiments") or [])
            prev = self.pixel_registry()
            rows = []
            for eid, ec in self._pixel_experiments().items():
                fl = [f for f in fits if "|" not in f or eid in nested_exps]
                params = self.bm["models"][ec["model"]]
                efp = PS.stable_hash(eid, ec, params, self.fp_of("pixel_dataset"), st.schema_hash,
                                     PS.code_identity(["models_pixel"]), int(self.nb["seed"]))
                rows += crossfit_pixel_model(eid, ec, params, st, self.root, fl, U, self.root, int(self.nb["seed"]),
                                             efp, True, self.say)
                self._importance_figure(eid, ec, efp)
                self.push(["models", "predictions"])
            R = pd.DataFrame(rows)
            keep = prev[~prev.experiment.isin(set(R.experiment))] if len(prev) and len(R) else prev
            ALL = pd.concat([keep, R], ignore_index=True) if len(keep) else R
            self.ws.save_table(ALL, "models/pixel_registry.csv")
            bad = R[~R.status.isin(["COMPLETE", "REUSED"])] if len(R) else R
            if not len(R):
                return FAILED, {"error": "no pixel experiment enabled by the RUN flags"}
            return COMPLETE, {"note": f"{len(R)} fits ({len(bad)} not complete) for {R.experiment.nunique()} "
                                      "experiments", "failed_fits": bad.to_dict("records")[:10]}
        flags = {k: self.run_flags.get(k) for k in FLAGS["train_pixel"]}
        parts = [stage_config(self.cfg, "train_pixel"), self.fp_of("pixel_dataset"), flags]
        return self._run("train_pixel", parts, fn)

    def _importance_figure(self, eid: str, ec: dict, efp: str) -> None:
        if not self.figures_on():
            return
        from . import visualization as V

        p = self.root / "models" / ec["model"] / eid / efp[:10]
        files = sorted(p.glob("*/feature_importance.csv"))
        if not files:
            return
        T = pd.concat([pd.read_csv(f).assign(fit=f.parent.name) for f in files])
        G = T.groupby("feature").agg(importance=("importance", "mean"), share=("share", "mean")).reset_index()
        V.plot_feature_importance(G, self.fig_dir("model_diagnostics"), f"importance_{eid}",
                                  title=f"{eid}: mean gain share across fits")

    # =========================================================================== segment models
    def _g06_fp(self, exp: str | None) -> str | None:
        from .models_pixel import current_fingerprints

        return current_fingerprints(self.pixel_registry()).get(exp) if exp else None

    def _g06_evidence(self, key: str, lab: np.ndarray, exp: str, fp10: str, outer) -> pd.DataFrame | None:
        from ..gems3.representation import rank01
        from .models_pixel import load_oof

        p = load_oof(self.root, exp, key, fp10, outer)
        if p is None:
            return None
        vu = self.units()[key]
        r = rank01(np.nan_to_num(p, nan=0.0), vu.unit.ev)
        rr, cc = np.nonzero(lab)
        lb = lab[rr, cc]
        n = int(lab.max())
        df = pd.DataFrame({"lb": lb, "v": r[rr, cc]}).groupby("lb").v.agg(["mean", "max"])
        out = pd.DataFrame({"label": np.arange(1, n + 1)})
        out["ev_g06_mean"] = df["mean"].reindex(out.label).to_numpy()
        out["ev_g06_max"] = df["max"].reindex(out.label).to_numpy()
        return out

    def train_segment_models(self) -> dict:
        def fn():
            from ..m7.ranking import _zone_mask
            from .datasets import load_dataset, unit_file
            from .models_segment import crossfit_segment_model, segment_to_raster

            U, eng = self.units(), self.engine()
            D = load_dataset(self.root, "segment")
            if not len(D["T"]):
                return FAILED, {"error": "segment dataset empty"}
            T = D["T"].merge(D["L"][["unit", "label", "target", "cls", "label_conf", "near_share"]],
                             on=["unit", "label"], how="left")
            buf = float(self.bm["validation"].get("segment_buffer_km", 5.0))
            zm = {k: _zone_mask(U[k].unit, eng.shape, buf) for k in T.unit.unique() if k in U}
            labs = {}
            for k in T.unit.unique():
                with np.load(self.root / "datasets" / "segments" / "units" / f"{unit_file(k)}_lab.npz") as z:
                    labs[k] = z["lab"]
            g06_exp = (self.bm.get("segment_map") or {}).get("g06_experiment")
            g06_fp = self._g06_fp(g06_exp)

            def unit_columns(key, outer):
                parts = []
                sc, z = key.split(":")
                if outer is not None:
                    p = self.root / "datasets" / "segments" / "nested_known" / f"{sc}_{outer}__{z}.parquet"
                    if not p.exists():
                        return None
                    parts.append(pd.read_parquet(p).drop(columns=["unit", "outer"]))
                if g06_fp:
                    E = self._g06_evidence(key, labs[key], g06_exp, g06_fp, outer)
                    if E is None:
                        return None
                    parts.append(E)
                if not parts:
                    return pd.DataFrame({"label": np.arange(1, int(labs[key].max()) + 1)})
                out = parts[0]
                for p in parts[1:]:
                    out = out.merge(p, on="label", how="outer")
                return out

            pools = [k for k, v in (self.bm["segment_candidates"].get("pools") or {}).items() if v]
            rows, infos = [], {}
            lam = float((self.bm.get("segment_map") or {}).get("lambda", 0.5))
            for eid, ec in self.bm["segment_experiments"].items():
                if ec.get("optional") and not self._lib_ok(ec["model"]):
                    rows.append({"experiment": eid, "unit": "*", "status": "SKIPPED",
                                 "detail": f"{ec['model']} not installed"})
                    continue
                params = self.bm["models"][ec.get("params", f"segment_{ec['model']}")]
                sub = T[T.scenario.isin(ec.get("scenarios", ["V1", "V3"]))]
                scores, info = crossfit_segment_model(eid, ec, sub, zm, params, int(self.nb["seed"]), unit_columns,
                                                      pools, self.say)
                efp = PS.stable_hash(eid, ec, params, self.fp_of("segment_dataset"), g06_fp, n=10)
                d = f"predictions/segments/{eid}/{efp}"
                self.ws.save_table(scores, f"{d}/scores.parquet")
                self.ws.save_json(info, f"{d}/info.json")
                for key, g in scores.groupby("unit"):
                    vu = U[key]
                    base = self._pixel_base(ec["track"], vu, g06_exp, g06_fp)
                    m = segment_to_raster(labs[key], g[["label", "score"]], base, vu.unit.ev, lam)
                    PS.atomic_write(self.root / d / f"{unit_file(key)}.npy", lambda t, m=m: np.save(t, m))
                infos[eid] = {"fingerprint": efp, "strictness": info["strictness"],
                              "reason": info["strictness_reason"], "dir": d}
                for key, v in info["units"].items():
                    rows.append({"experiment": eid, "unit": key, "status": v["status"], "fingerprint": efp,
                                 "strictness": info["strictness"], "n_train": v.get("n_train"),
                                 "n_pos": v.get("n_pos"), "n_test": v.get("n_test")})
                self.push(["predictions"])
            prev = self.ws.load_table("models/segment_registry.csv")
            R = pd.DataFrame(rows)
            keep = prev[~prev.experiment.isin(set(R.experiment))] if len(prev) else prev
            self.ws.save_table(pd.concat([keep, R], ignore_index=True) if len(keep) else R,
                               "models/segment_registry.csv")
            old = self.ws.read_json("models/segment_models.json") or {}
            self.ws.save_json({**old, **infos}, "models/segment_models.json")
            return COMPLETE, {"note": f"{len(infos)} segment experiments; strictness "
                                      f"{ {k: v['strictness'] for k, v in infos.items()} }"}
        parts = [stage_config(self.cfg, "train_segment"), self.fp_of("segment_dataset"),
                 self._g06_fp((self.bm.get("segment_map") or {}).get("g06_experiment"))]
        return self._run("train_segment", parts, fn)

    def _pixel_base(self, track: str, vu, g06_exp, g06_fp) -> np.ndarray:
        """Pixel score preserved by the segment-assisted map: W2 (S2), G06 pixel OOF or a label-free store feature
        (S1)."""
        from .models_pixel import load_oof

        if track == "S2":
            return self.engine().w2_rank(vu.base)
        if g06_fp:
            p = load_oof(self.root, g06_exp, vu.key, g06_fp)
            if p is not None:
                return p
        f = (self.bm.get("segment_map") or {}).get("fallback_feature")
        win = (vu.sl[0].start, vu.sl[0].stop, vu.sl[1].start, vu.sl[1].stop)
        return self.store().load([f], win)[0]

    @staticmethod
    def _lib_ok(kind: str) -> bool:
        import importlib.util

        mod = {"catboost": "catboost", "ranker": "lightgbm", "lightgbm": "lightgbm",
               "random_forest": "sklearn", "xgboost": "xgboost"}.get(kind, kind)
        return importlib.util.find_spec(mod) is not None

    # =========================================================================== patch model
    def train_patch_models(self) -> dict:
        def fn():
            from .datasets import unit_file
            from .models_patch import create_patch_dataset, predict_patch_model, save_patch_model, train_patch_model
            from .models_pixel import oof_path
            from .splits import build_training_visibility, make_inner_folds
            from .supervision import build_pixel_supervision

            pc = self.bm["patch"]
            st, eng, U = self.store(), self.engine(), self.units()
            chans = [c for c in pc["channels"] if c in st.usable()]
            if len(chans) < 2:
                return FAILED, {"error": f"patch channels unavailable: {pc['channels']}"}
            mm = create_patch_dataset(st, chans, self.root / "datasets" / "patches")
            index = self.ws.load_table("datasets/patches/patch_index.parquet")
            efp = PS.stable_hash(pc, chans, st.schema_hash, PS.code_identity(["models_patch"]), n=10)
            rows = []
            for fid in pc.get("fits", ["V1:0", "V1:1", "V1:2"]):
                vis = build_training_visibility(eng, fid, self.bm["validation"], tuple(self.bm["validation"]
                                                                                       ["scenarios"]))
                sup = build_pixel_supervision(vis.visible, vis.region, eng.valid(), self.bm["supervision"])
                ic = self.bm.get("inner_cv") or {}
                inner = make_inner_folds(vis.region, vis.visible, float(ic.get("block_km", 20)),
                                         int(ic.get("n_folds", 3)), int(self.nb["seed"]))
                mdir = self.root / "models" / "patch_unet" / pc.get("experiment", "G06-U1") / efp / unit_file(fid)
                model, scaler, info = train_patch_model(mm, index, vis, sup["classes"], inner, pc,
                                                        bool(self.nb.get("use_gpu")), int(self.nb["seed"]), self.say)
                save_patch_model(model, scaler, info, mdir)
                for key in vis.predicts:
                    vu = U.get(key)
                    if vu is None:
                        continue
                    win = (vu.sl[0].start, vu.sl[0].stop, vu.sl[1].start, vu.sl[1].stop)
                    p = predict_patch_model(model, mm, scaler, win, vu.unit.ev, int(pc["patch_px"]),
                                            float(pc.get("overlap", 0.5)), bool(self.nb.get("use_gpu")))
                    pf = oof_path(self.root, pc.get("experiment", "G06-U1"), fid, key, efp)
                    pf.parent.mkdir(parents=True, exist_ok=True)
                    PS.atomic_write(pf, lambda t, p=p: np.save(t, p))
                    PS.write_json(pf.with_suffix(".json"), {"experiment": pc.get("experiment", "G06-U1"),
                                                            "fit_id": fid, "unit": key, "window": list(win),
                                                            "channels": chans, "fingerprint": efp})
                rows.append({"experiment": pc.get("experiment", "G06-U1"), "fit_id": fid, "status": "COMPLETE",
                             "fingerprint": efp, "best_inner_val_loss": info["best_inner_val_loss"],
                             "n_train_patches": info["n_train_patches"], "model_dir": str(mdir.relative_to(
                                 self.root))})
                del vis, sup, inner
                self.push(["models", "predictions"])
            self.ws.save_table(pd.DataFrame(rows), "models/patch_registry.csv")
            return COMPLETE, {"note": f"{len(rows)} U-Net fits on {len(chans)} channels"}
        return self._run("train_patch", [stage_config(self.cfg, "train_patch"), self.fp_of("pixel_dataset")], fn)

    # =========================================================================== meta models
    def train_meta_models(self) -> dict:
        def fn():
            from .datasets import unit_file
            from .models_meta import build_oof_prediction_matrix, crossfit_meta, evaluate_model_complementarity
            from .models_pixel import load_oof

            mc = self.bm["meta"]
            eng, U = self.engine(), self.units()
            g06 = mc.get("g06_experiment")
            g06_fp = self._g06_fp(g06)
            if "g06_pixel" in mc.get("inputs", []) and not g06_fp:
                return FAILED, {"error": f"meta input g06_pixel needs a trained {g06} (TRAIN_LIGHTGBM first)"}
            st = self.store()
            dom_ok = "ctx_domain" in st.names

            def unit_inputs(vu, outer):
                maps = {}
                ins = mc.get("inputs", [])
                if "w2" in ins:
                    maps["w2"] = eng.w2_map(vu.base)
                if "cnn_b0" in ins or "cnn_b1" in ins:
                    b0, b1 = eng.cnn(vu.base)
                    maps.update(cnn_b0=b0, cnn_b1=b1)
                if "g06_pixel" in ins:
                    p = load_oof(self.root, g06, vu.key, g06_fp, outer)
                    if p is None:
                        return None
                    maps["g06_pixel"] = p
                dom = None
                if dom_ok and mc.get("use_domain", True):
                    win = (vu.sl[0].start, vu.sl[0].stop, vu.sl[1].start, vu.sl[1].stop)
                    dom = st.load(["ctx_domain"], win)[0]
                return build_oof_prediction_matrix(maps, vu.unit.ev, dom)

            units = [U[k] for k in sorted(U) if U[k].scenario in mc.get("scenarios", ["V1", "V3"])]
            maps, info = crossfit_meta(units, unit_inputs, eng.labels(), mc, int(self.nb["seed"]), self.say)
            efp = PS.stable_hash(mc, g06_fp, PS.code_identity(["models_meta"]), n=10)
            comp = []
            ev = self.bm["evaluation"]
            for kind, by_unit in maps.items():
                for key, m in by_unit.items():
                    PS.atomic_write(self.root / "predictions" / "meta" / kind / efp / f"{unit_file(key)}.npy",
                                    lambda t, m=m: np.save(t, m))
            for key in sorted({k for d in maps.values() for k in d}):
                comp.append(evaluate_model_complementarity({k: maps[k].get(key) for k in maps} |
                                                           {"g06_pixel": load_oof(self.root, g06, key, g06_fp)
                                                            if g06_fp else None}, U[key],
                                                           float(ev["primary"]["frac"]),
                                                           float(ev.get("system_recovered_support", 0.5)),
                                                           float(ev.get("system_recovered_frac", 0.5))))
            C = pd.concat(comp, ignore_index=True) if comp else pd.DataFrame()
            self.ws.save_table(C, "evaluation/complementarity.csv")
            self.ws.save_json({"fingerprint": efp, "strictness": info["strictness"], "units": info["units"],
                               "nested_ok": info["nested_ok"], "inputs": mc.get("inputs"), "g06_fp": g06_fp},
                              "models/meta_registry.json")
            self.push(["predictions", "models", "evaluation"])
            return COMPLETE, {"note": f"meta models {list(maps)}; strictness {info['strictness']}"}
        return self._run("train_meta", [stage_config(self.cfg, "train_meta"),
                                        self._g06_fp((self.bm.get("meta") or {}).get("g06_experiment"))], fn)

    # =========================================================================== evaluation
    def _score_maps(self) -> tuple[dict, dict]:
        """{model: {unit: map}} of every available held-out map + {model: strictness}."""
        from .datasets import unit_file

        U = self.units()
        maps, strict = {"W2": {}}, {"W2": "REFERENCE"}
        eng = self.engine()
        for k, vu in U.items():
            try:
                maps["W2"][k] = eng.w2_rank(vu.base)
            except (FileNotFoundError, ValueError):
                pass
        from .models_pixel import current_fingerprints, load_oof

        cur = current_fingerprints(self.pixel_registry())
        pr = self.ws.load_table("models/patch_registry.csv")
        if len(pr):
            cur.update({e: str(g.fingerprint.iloc[-1]) for e, g in pr.groupby("experiment")})
        for exp, fp10 in cur.items():
            maps[exp], strict[exp] = {}, "STRICT_ISOLATED"
            for k in U:
                p = load_oof(self.root, exp, k, fp10)
                if p is not None:
                    maps[exp][k] = p
        seg = self.ws.read_json("models/segment_models.json") or {}
        for exp, v in seg.items():
            maps[exp], strict[exp] = {}, v["strictness"]
            for k in U:
                p = self.root / v["dir"] / f"{unit_file(k)}.npy"
                if p.exists():
                    maps[exp][k] = np.load(p)
        meta = self.ws.read_json("models/meta_registry.json") or {}
        if meta:
            for kind, s in meta.get("strictness", {}).items():
                if kind == "w2":
                    continue
                name = f"G06-M1:{kind}"
                maps[name], strict[name] = {}, s
                for k in U:
                    p = self.root / "predictions" / "meta" / kind / meta["fingerprint"] / f"{unit_file(k)}.npy"
                    if p.exists():
                        maps[name][k] = np.load(p)
        return {m: d for m, d in maps.items() if d}, strict

    def evaluate_models(self) -> dict:
        def fn():
            from . import evaluation as EV

            U, eng = self.units(), self.engine()
            ev = self.bm["evaluation"]
            maps, strict = self._score_maps()
            rows, srows, pr, cal = [], [], [], []
            for key, vu in sorted(U.items()):
                proto = EV.unit_protocol(vu, eng.template_valid()[vu.sl])
                for m, d in maps.items():
                    if key not in d:
                        continue
                    r, s = EV.evaluate_prediction_grid(vu, d[key], ev["reps"], ev["fracs"], proto,
                                                       {"model": m, "strictness": strict.get(m, "")},
                                                       float(ev.get("system_recovered_support", 0.5)),
                                                       float(ev.get("system_recovered_frac", 0.5)))
                    rows += r
                    srows += s
                    pr.append({"model": m, "unit": key, "pr_auc": EV.pr_auc(d[key], vu, proto["elig"],
                                                                           int(ev.get("pr_auc_max_px", 2_000_000)))})
                    if m.startswith("G06-P") or m.startswith("G06-U"):
                        cal.append(EV.calibration_table(d[key], vu, proto["elig"],
                                                        int(ev.get("calibration_bins", 10))).assign(model=m))
                vu.unit.release()
                self.say(f"  evaluated {key}: {sum(key in d for d in maps.values())} models")
            Rw = pd.DataFrame(rows)
            S = pd.DataFrame(srows)
            self.ws.save_table(Rw, "evaluation/metrics_rows.csv")
            SUM = EV.aggregate_components(Rw)
            SUM["strictness"] = SUM.model.map(strict)
            self.ws.save_table(SUM, "evaluation/metrics.csv")
            self.ws.save_table(EV.per_zone(Rw), "evaluation/per_zone.csv")
            self.ws.save_table(S, "evaluation/system_rows.csv")
            SR = EV.system_recovery_report(S)
            self.ws.save_table(SR, "evaluation/system_recovery.csv")
            CMP = EV.compare_with_w2(SUM)
            self.ws.save_table(CMP, "evaluation/model_comparison.csv")
            self.ws.save_table(self._ablations(SUM), "evaluation/ablations.csv")
            sel = [EV.nested_policy_selection(Rw, m, sc) for m in maps for sc in sorted(set(Rw.scenario))]
            self.ws.save_table(pd.concat([s for s in sel if len(s)], ignore_index=True) if any(len(s) for s in sel)
                               else pd.DataFrame(), "evaluation/representation_selection.csv")
            PD = EV.promotion_decision(CMP, SR, strict, self.bm["promotion"], ev["primary"])
            self.ws.save_table(PD, "evaluation/promotion_decisions.csv")
            PRA = pd.DataFrame(pr)
            CAL = pd.concat([c for c in cal if len(c)], ignore_index=True) if any(len(c) for c in cal) else \
                pd.DataFrame()
            self.ws.save_table(PRA, "evaluation/pr_auc.csv")
            self.ws.save_table(CAL, "evaluation/calibration.csv")
            self.ws.save_table(pd.DataFrame([{"reference": k, "value": v, "note": "historical, other protocol - "
                                              "not comparable, never a tuning target"} for k, v in
                                             (self.bm.get("historical") or {}).items()]),
                               "evaluation/historical_references.csv")
            best = PD[PD.status == "PROMOTE_CANDIDATE"].sort_values("v1_gain", ascending=False) if len(PD) else PD
            self.ws.save_json({"primary": ev["primary"], "promote": list(best.model) if len(best) else [],
                               "strictness": strict, "statuses": dict(zip(PD.model, PD.status)) if len(PD) else {}},
                              "evaluation/decision.json")
            if self.figures_on():
                self._evaluation_figures(Rw, SUM, SR, PRA, CAL, maps)
            self.push(["evaluation", "figures"])
            return COMPLETE, {"note": f"{len(maps)} models x {len(U)} units; promote: "
                                      f"{list(best.model) if len(best) else 'none'}"}
        parts = [stage_config(self.cfg, "evaluate"), self.fp_of("splits"), self.fp_of("train_pixel"),
                 self.fp_of("train_segment"), self.fp_of("train_patch"), self.fp_of("train_meta")]
        return self._run("evaluate", parts, fn)

    def _ablations(self, SUM: pd.DataFrame) -> pd.DataFrame:
        """Feature-family ablation from the nested pixel experiments (P1 raw -> P5 full) at the primary cell."""
        ev = self.bm["evaluation"]["primary"]
        rows = []
        S = SUM[(SUM.rep == ev["rep"]) & (SUM.frac == float(ev["frac"]))] if len(SUM) else SUM
        for eid, ec in self.bm["pixel_experiments"].items():
            for sc in ("V1", "V3"):
                r = S[(S.model == eid) & (S.scenario == sc)] if len(S) else S
                w = S[(S.model == "W2") & (S.scenario == sc)] if len(S) else S
                if not len(r):
                    continue
                rows.append({"experiment": eid, "model": ec["model"], "families": ",".join(ec["families"]),
                             "scenario": sc, "dti": float(r.dti.iloc[0]), "worst_zone_dti": float(r.worst_zone_dti
                                                                                                 .iloc[0]),
                             "dti_w2": float(w.dti.iloc[0]) if len(w) else np.nan})
        T = pd.DataFrame(rows)
        if len(T):
            for sc in T.scenario.unique():
                b = T[(T.scenario == sc) & (T.families == "raw") & (T.model == "lightgbm")]
                if len(b):
                    T.loc[T.scenario == sc, "delta_vs_raw_lgbm"] = T.loc[T.scenario == sc, "dti"] - float(
                        b.dti.iloc[0])
        return T

    def _evaluation_figures(self, Rw, SUM, SR, PRA, CAL, maps) -> None:
        from . import evaluation as EV
        from . import visualization as V

        ev = self.bm["evaluation"]
        rep, frac = ev["primary"]["rep"], float(ev["primary"]["frac"])
        models = ["W2"] + [m for m in maps if m != "W2"]
        V.plot_model_comparison(SUM, self.fig_dir("model_diagnostics"), "model_comparison", rep, frac, models)
        for sc in sorted(set(Rw.scenario)):
            V.plot_dti_frontier(Rw, self.fig_dir("dti_frontiers"), f"dti_frontier_{sc}_{rep}", sc, rep, models)
        V.plot_geographic_robustness(EV.per_zone(Rw), self.fig_dir("model_diagnostics"), "geographic_robustness",
                                     rep, frac, models)
        if len(PRA):
            V.plot_pr_and_calibration(PRA, CAL, self.fig_dir("model_diagnostics"), "pr_auc_calibration")
        self._failure_atlas(SR, maps, rep, frac)

    def _failure_atlas(self, SR: pd.DataFrame, maps: dict, rep: str, frac: float) -> None:
        """Windows of systems newly recovered / lost vs W2 for the best G06 model (truth drawn post hoc only)."""
        from ..gems3 import representation as R
        from . import visualization as V

        if not len(SR):
            return
        cand = SR[(SR.model != "W2") & (SR.rep == rep) & (SR.frac == frac) & (SR.scenario == "V1")]
        if not len(cand):
            return
        best = cand.groupby("model").apply(lambda g: g.n_new.sum() - g.n_lost.sum()).idxmax()
        U, eng = self.units(), self.engine()
        for kind in ("new", "lost"):
            cases = []
            for r in cand[cand.model == best].itertuples():
                ids = [int(x) for x in str(getattr(r, f"{kind}_ids")).split(";") if x not in ("", "nan")]
                if not ids or r.unit not in maps.get(best, {}) or r.unit not in maps.get("W2", {}):
                    continue
                vu = U[r.unit]
                proto_el = vu.unit.scored_mask() & eng.template_valid()[vu.sl]
                src = maps["W2"][r.unit] if kind == "lost" else maps[best][r.unit]
                q = R.apply(ev_rep(self.bm, rep), np.nan_to_num(src, nan=-1.0), vu.unit.ev, frac, known=vu.known,
                            eligible=proto_el)
                for sid in ids[:12]:
                    m = vu.sysmap == sid
                    if not m.any():
                        continue
                    rr, cc = np.nonzero(m)
                    pad = 40
                    sl = (slice(max(rr.min() - pad, 0), rr.max() + pad), slice(max(cc.min() - pad, 0), cc.max() + pad))
                    cases.append({"label": f"{r.unit} system {sid} ({kind} vs W2, {best if kind == 'new' else 'W2'})",
                                  "unit": r.unit, "system_id": sid, "score": src[sl], "truth": m[sl],
                                  "selected": q[sl] > 0})
            V.plot_failure_cases(cases, self.fig_dir("fault_failures"), f"systems_{kind}_{best.replace(':', '_')}",
                                 f"Systems {kind} vs W2 - {best} (black: held-out truth, post hoc)")

    # =========================================================================== full region
    def reconstruct_full_region(self) -> dict:
        return {"full_inference": self.full_inference(), "export": self.export_submission()}

    def full_inference(self) -> dict:
        def fn():
            from .datasets import load_dataset
            from .inference import predict_full_raster, verify_inference_parity
            from .models_pixel import PixelModel, crossfit_pixel_model

            st = self.store()
            inf = self.bm.get("inference") or {}
            man = {"experiments": {}, "role": "RESEARCH_NOT_A_SUBMISSION"}
            for eid in inf.get("experiments", ["G06-P5"]):
                ec = self.bm["pixel_experiments"][eid]
                params = self.bm["models"][ec["model"]]
                efp = PS.stable_hash(eid, ec, params, self.fp_of("pixel_dataset"), st.schema_hash,
                                     PS.code_identity(["models_pixel"]), int(self.nb["seed"]), "FULL")
                rows = crossfit_pixel_model(eid, ec, params, st, self.root, ["FULL"], {}, self.root,
                                            int(self.nb["seed"]), efp, False, self.say)
                if not rows or rows[0]["status"] not in ("COMPLETE", "REUSED"):
                    man["experiments"][eid] = {"status": "FAILED", "rows": rows}
                    continue
                model = PixelModel.load(self.root / rows[0]["model_dir"])
                out = self.root / "predictions" / "research_geotiffs" / f"{eid}_{efp[:10]}_probability.tif"
                if not out.exists():
                    predict_full_raster(model, st, out, int(self.fs["tiling"]["tile_px"]),
                                        {"EXPERIMENT": eid, "FIT": "FULL", "FINGERPRINT": efp[:10]}, self.say)
                pool = load_dataset(self.root, "pixel", columns=model.feature_columns)
                par = verify_inference_parity(model, st, out, pool["X"], pool["meta"], seed=int(self.nb["seed"]))
                self.ws.save_json(par, f"predictions/full_region/parity_{eid}.json")
                man["experiments"][eid] = {"status": "COMPLETE", "raster": str(out.relative_to(self.root)),
                                           "model_dir": rows[0]["model_dir"], "parity": par, "fingerprint": efp[:10]}
            self.ws.save_json(man, "predictions/full_region/full_inference_manifest.json")
            self.push(["predictions", "models"])
            bad = [k for k, v in man["experiments"].items() if v["status"] != "COMPLETE" or
                   v.get("parity", {}).get("status") != "PASS"]
            if bad:
                return FAILED, {"error": f"full inference / parity failed for {bad}"}
            return COMPLETE, {"note": f"research rasters for {list(man['experiments'])} (parity PASS)"}
        return self._run("full_inference", [stage_config(self.cfg, "full_inference"), self.fp_of("pixel_dataset")],
                         fn)

    def export_submission(self) -> dict:
        def fn():
            from .inference import export_submission_candidate, read_full, submission_gate

            PD = self.ws.load_table("evaluation/promotion_decisions.csv")
            man = self.ws.read_json("predictions/full_region/full_inference_manifest.json") or {}
            exp = (self.bm.get("export") or {}).get("model") or (
                PD.loc[PD.status == "PROMOTE_CANDIDATE", "model"].iloc[0] if len(PD) and
                (PD.status == "PROMOTE_CANDIDATE").any() else None)
            info = (man.get("experiments") or {}).get(exp or "", {})
            ok, why = submission_gate(PD, exp or "", self.run_flags, info.get("parity"))
            if exp and exp not in (man.get("experiments") or {}):
                ok, why = False, why + [f"no full-region research raster for {exp} (only pixel experiments have one)"]
            record = {"model": exp, "eligible": ok, "reasons": why, "uploaded": False}
            if ok:
                eng = self.engine()
                ev = self.bm["evaluation"]["primary"]
                d = eng.data
                res = export_submission_candidate(read_full(self.root / info["raster"]), d.template_profile,
                                                  eng.template_valid(), self.store().footprint(), eng.labels(),
                                                  self.bm["evaluation"]["reps"][ev["rep"]], float(ev["frac"]),
                                                  self.root / "submissions" / f"g06_candidate_{exp}.tif",
                                                  {"MODEL": exp, "RUN_ID": self.run_id, "COMMIT": self.head})
                record.update(res)
            self.ws.save_json(record, "submissions/submission_manifest.json")
            self.push(["submissions"])
            return COMPLETE, {"note": ("READY_FOR_MANUAL_UPLOAD: " + record.get("path", "")) if ok else
                              f"no export: {why}"}
        return self._run("export", [stage_config(self.cfg, "export"), self.fp_of("evaluate"),
                                    self.fp_of("full_inference")], fn)

    # =========================================================================== reports
    def create_reports(self) -> dict:
        def fn():
            write_reports(self)
            PS.write_artifact_inventory(self.root)
            self.push(["reports", "manifests"])
            return COMPLETE, {"note": "reports written from persisted tables"}
        return self._run("report", [time.strftime("%Y-%m-%dT%H:%M:%S")], fn, force=True)

    # =========================================================================== all
    def run(self) -> dict:
        self.run_tests()
        self.preflight()
        self.prepare_features()
        self.build_splits()
        self.build_datasets()
        self.train_pixel_models()
        self.train_segment_models()
        self.train_patch_models()
        self.train_meta_models()
        self.evaluate_models()
        self.reconstruct_full_region()
        self.create_reports()
        self.final_message()
        return dict(self.status)

    def stage_table(self) -> pd.DataFrame:
        rows = []
        for s in STAGES:
            m = PS.load_cached_stage(self.root, s) or {}
            rows.append({"stage": s, "session_status": self.status.get(s), "manifest_status": m.get("status"),
                         "fingerprint": str(m.get("fingerprint") or "")[:10], "seconds": m.get("seconds"),
                         "note": str(m.get("note") or m.get("reason") or m.get("error") or "")[:200],
                         "conclusion_allowed": self.status.get(s) in VALID})
        return pd.DataFrame(rows)

    def final_message(self) -> None:
        print("\n" + "=" * 100)
        print(f"G06 {G06_CODE_REV} run {self.run_id} | commit {self.head}")
        for s in STAGES:
            print(f"  {s:<16} {self.status.get(s)}")
        print(f"local: {self.root}\nDrive: {self.ws.drive_path()}")
        print("Nothing was uploaded. Research GeoTIFFs are tagged ROLE=RESEARCH_NOT_A_SUBMISSION.")
        print("=" * 100)


def ev_rep(bm: dict, rep: str) -> dict:
    return bm["evaluation"]["reps"][rep]


# =============================================================================== reports (persisted tables only)
def _md(df: pd.DataFrame, n: int = 40, cols=None) -> str:
    if df is None or not len(df):
        return "_(no rows)_\n"
    d = df[[c for c in (cols or df.columns) if c in df.columns]].head(n)
    fmt = (lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    d = d.map(fmt) if hasattr(d, "map") else d.applymap(fmt)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def _next_experiment(P: "G06Pipeline", PD: pd.DataFrame, AB: pd.DataFrame, SUM: pd.DataFrame) -> list[str]:
    """Rule-based recommendation from the persisted results (labelled as such)."""
    out = []
    if P.status.get("features") not in VALID:
        return ["Build the feature store first (first-run configuration of docs/g06/COLAB_EXECUTION_GUIDE.md)."]
    if not len(SUM):
        return ["Run the second-run configuration: TRAIN_RANDOM_FOREST + TRAIN_LIGHTGBM + EVALUATE (features are "
                "reused from the cache)."]
    prim = P.bm["evaluation"]["primary"]
    s = SUM[(SUM.scenario == "V1") & (SUM.rep == prim["rep"]) & (SUM.frac == float(prim["frac"]))]
    w2 = s[s.model == "W2"]
    g = s[s.model.str.startswith("G06-P")].sort_values("dti", ascending=False)
    if len(g) and len(w2):
        gap = float(g.dti.iloc[0] - w2.dti.iloc[0])
        out.append(f"Best geology-only pixel model {g.model.iloc[0]}: V1 DTI {g.dti.iloc[0]:.4f} vs W2 "
                   f"{w2.dti.iloc[0]:.4f} (delta {gap:+.4f}) at {prim['rep']} @ {prim['frac']}.")
        if gap < -0.05:
            out.append("RULE: geology alone is far below W2 -> test complementarity, not replacement: "
                       "TRAIN_META_MODEL (G06-M1) and the S2 segment ranker, and read evaluation/complementarity.csv "
                       "(systems recovered by G06 but not by W2).")
        else:
            out.append("RULE: geology-only is competitive -> confirm with nested meta (strict) and V3; then "
                       "FULL_INFERENCE for the research raster.")
    if len(AB):
        a = AB[AB.scenario == "V1"].sort_values("dti", ascending=False)
        if len(a):
            out.append(f"Most useful feature families on V1: {a.families.iloc[0]} ({a.experiment.iloc[0]}).")
    if len(PD) and (PD.status == "PROMISING_SCREENING_ONLY").any():
        out.append("A screening-only model passes every gate except strict validation: the next step is the strict "
                   "path (nested inputs / label-free pools), not an export.")
    out.append("These lines are rule-based readings of the tables, not findings.")
    return out


def write_reports(P: "G06Pipeline") -> None:
    ws = P.ws
    reg = ws.load_table("features/feature_registry.parquet")
    cov = ws.load_table("features/feature_coverage.csv")
    schema = ws.read_json("features/feature_schema.json") or {}
    SUM = ws.load_table("evaluation/metrics.csv")
    CMP = ws.load_table("evaluation/model_comparison.csv")
    PD = ws.load_table("evaluation/promotion_decisions.csv")
    SR = ws.load_table("evaluation/system_recovery.csv")
    AB = ws.load_table("evaluation/ablations.csv")
    CO = ws.load_table("evaluation/complementarity.csv")
    hist = P.bm.get("historical") or {}
    hdr = f"Run `{P.run_id}` | code `{G06_CODE_REV}` | commit `{P.head}` | written {time.strftime('%Y-%m-%d %H:%M')}"
    # FEATURE_REPORT
    lines = ["# G06 feature report", "", hdr, ""]
    if len(reg):
        g = reg.groupby(["group", "family"]).agg(n=("feature_id", "size"), usable=("usable", "sum")).reset_index()
        lines += [f"Schema hash `{schema.get('schema_hash')}`, effective halo {schema.get('halo_effective_px')} px, "
                  f"{len(reg)} channels ({int(reg.usable.sum())} usable).", "", "## Groups", "", _md(g, 50),
                  "## Unusable channels (constant / empty; kept, never fed to a model)", "",
                  _md(reg[~reg.usable.astype(bool)], 60, ["feature_id", "group", "frac_finite", "std"]),
                  "## Lowest coverage", "", _md(cov.sort_values("frac_finite").head(20) if len(cov) else cov, 20)]
    else:
        lines += ["Feature store not built in this run directory."]
    lines += ["", "Label-free footprint statistics (fit_scope=full_footprint_unsupervised): entropy bin edges, strain "
              "asinh scales, k-means domains. No fault label enters any feature."]
    ws.save_text("\n".join(lines) + "\n", "reports/FEATURE_REPORT.md")
    # MODEL_BENCHMARK
    prim = P.bm["evaluation"]["primary"]
    lines = ["# G06 model benchmark", "", hdr, "",
             f"Primary metric: pooled official-proxy DTI (summed components) at the FIXED policy {prim['rep']} @ "
             f"{prim['frac']} of the eligible domain; worst zone with every pooled number. W2 = honest held-out W2 "
             "recomputed under exactly this protocol.", "",
             "Historical references (other representation rules - not comparable, never tuning targets): " +
             ", ".join(f"{k} = {v}" for k, v in hist.items()), ""]
    if len(SUM):
        s = SUM[(SUM.rep == prim["rep"]) & (SUM.frac == float(prim["frac"]))]
        lines += ["## Pooled DTI at the primary cell", "",
                  _md(s.sort_values(["scenario", "dti"], ascending=[True, False]), 60,
                      ["scenario", "model", "strictness", "dti", "worst_zone_dti", "worst_zone", "tp_w", "fp_w",
                       "fn_w", "area_frac_eligible", "zone_dtis"]),
                  "## Versus W2", "", _md(CMP[(CMP.rep == prim["rep"]) & (CMP.frac == float(prim["frac"]))]
                                          if len(CMP) else CMP, 60),
                  "## Feature-family ablation", "", _md(AB, 40),
                  "## Promotion decisions", "", _md(PD, 40),
                  "SCREENING_ONLY results are never reported as nested DTI. Nothing was uploaded.", ""]
    else:
        lines += ["No evaluation in this run directory (EVALUATE not run)."]
    ws.save_text("\n".join(lines) + "\n", "reports/MODEL_BENCHMARK.md")
    # FAILURE_ANALYSIS
    lines = ["# G06 failure analysis", "", hdr, ""]
    if len(SR):
        s = SR[(SR.rep == prim["rep"]) & (SR.frac == float(prim["frac"]))]
        lines += ["## Systems recovered / new / lost vs W2 (primary cell)", "",
                  _md(s, 80, ["model", "unit", "n_systems", "n_recovered", "n_recovered_w2", "n_new", "n_lost",
                              "new_ids", "lost_ids"]),
                  "## Complementarity with W2 (meta stage)", "", _md(CO, 60),
                  "Figures: `figures/fault_failures/` (truth outlines are post-hoc only)."]
    else:
        lines += ["No system recovery table (EVALUATE not run)."]
    ws.save_text("\n".join(lines) + "\n", "reports/FAILURE_ANALYSIS.md")
    # EXECUTION_STATUS
    st = P.stage_table()
    lines = ["# G06 execution status", "", hdr, "",
             "Only COMPLETE / REUSED_VALID_CACHE stages support conclusions. NOT_REQUESTED = RUN flag off and no valid "
             "cache; BLOCKED_UPSTREAM = a dependency is not valid; TESTS_FAILED blocks every training stage.", "",
             _md(st, 40), "", f"Runtime errors: {len(P.errors)} (manifests/runtime_errors.json).", "",
             "No script, test, model or notebook was executed while this code was authored; every number in this "
             "directory comes from the Colab run identified above."]
    ws.save_text("\n".join(lines) + "\n", "reports/EXECUTION_STATUS.md")
    # NEXT_EXPERIMENT
    ws.save_text("\n".join(["# G06 next experiment", "", hdr, ""] + [f"* {x}" for x in
                                                                     _next_experiment(P, PD, AB, SUM)]) + "\n",
                 "reports/NEXT_EXPERIMENT.md")
