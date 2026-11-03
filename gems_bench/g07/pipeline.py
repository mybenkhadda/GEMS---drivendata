"""G07 orchestrator (one class, every stage; dependency-checked, fingerprint-cached, resumable, Drive-synced).

    cfg = load_config(overrides={"notebook": CONFIG, "run": RUN})
    P = G07Pipeline(cfg, RUN, HEAD)
    P.run()

Stages [RUN flag] (dependencies) - see docs/g07/ARCHITECTURE.md::

    tests              [notebook run_synthetic_tests]   TESTS_FAILED blocks every training stage
    preflight          [PREFLIGHT]           sources, grid, frozen engine, G06 store, W2 artifacts
    dem_inventory      [DEM_INVENTORY]       catalog vs local DEM headers, licences, optional guarded download
    dem_coverage       [DEM_COVERAGE]        coverage of the ELIGIBLE region / V1 zones          (dem_inventory)
    dem_feasibility    [DEM_FEASIBILITY]     pilot DEM features, signal, recall, cases, GO / NO_GO (dem_coverage)
    dem_features       [DEM_FEATURES]        all local DEM -> 100 m store (BLOCKED_INPUT unless GO)  (dem_feasibility)
    g06_restore        [BUILD_DATASETS | TRAIN_SUBSURFACE | GENERATE_CANDIDATES | ...]  open + validate the G06 store
    datasets           [BUILD_DATASETS]      splits audit, subsurface pool + label variants, surface pool + patches
    train_subsurface   [TRAIN_SUBSURFACE]    G07-B0 / B1 / B2                                      (datasets)
    train_surface_classic [TRAIN_SURFACE_CLASSIC]  G07-A1 / A2 (DEM-covered cells)                 (datasets)
    train_surface_unet [TRAIN_SURFACE_UNET]  G07-C1                                                (datasets)
    generate_candidates [GENERATE_CANDIDATES] pools, inventory, recall (G07-C2), candidate tables  (datasets)
    train_candidate_ranker [TRAIN_CANDIDATE_RANKER]  G07-D1, calibration, G07-D2 portfolio, G07-E1 corridor
    evaluate           [EVALUATE]            A0 (W2 exact-k) and every held-out map, V1 / V3 / V2, promotion
    full_inference     [FULL_INFERENCE]      FULL fits, full-region construction, parity
    export             [EXPORT_SUBMISSION]   gated validated GeoTIFF (READY_FOR_MANUAL_UPLOAD at best)
    report             [REPORT]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import G07_CODE_REV, G07_VERSION
from . import confidence as CF
from . import persistence as PS
from .config import REPO, RUN_DEFAULTS, compute_run_id, stage_config, validate_config

COMPLETE, FAILED, REUSED, VALID = PS.COMPLETE, PS.FAILED, PS.REUSED, PS.VALID
BLOCKED_UPSTREAM, BLOCKED_INPUT, NOT_REQUESTED, SKIPPED = PS.BLOCKED_UPSTREAM, PS.BLOCKED_INPUT, PS.NOT_REQUESTED, \
    PS.SKIPPED
NOT_RUN, TESTS_FAILED = PS.NOT_RUN, "TESTS_FAILED"
STAGES = ("tests", "preflight", "dem_inventory", "dem_coverage", "dem_feasibility", "dem_features", "g06_restore",
          "datasets", "train_subsurface", "train_surface_classic", "train_surface_unet", "generate_candidates",
          "train_candidate_ranker", "evaluate", "full_inference", "export", "report")
FLAGS = {"preflight": ("PREFLIGHT",), "dem_inventory": ("DEM_INVENTORY",), "dem_coverage": ("DEM_COVERAGE",),
         "dem_feasibility": ("DEM_FEASIBILITY",), "dem_features": ("DEM_FEATURES",),
         "g06_restore": ("BUILD_DATASETS", "TRAIN_SUBSURFACE", "GENERATE_CANDIDATES", "TRAIN_CANDIDATE_RANKER",
                         "EVALUATE", "FULL_INFERENCE"),
         "datasets": ("BUILD_DATASETS",), "train_subsurface": ("TRAIN_SUBSURFACE",),
         "train_surface_classic": ("TRAIN_SURFACE_CLASSIC",), "train_surface_unet": ("TRAIN_SURFACE_UNET",),
         "generate_candidates": ("GENERATE_CANDIDATES",), "train_candidate_ranker": ("TRAIN_CANDIDATE_RANKER",),
         "evaluate": ("EVALUATE",), "full_inference": ("FULL_INFERENCE",), "export": ("EXPORT_SUBMISSION",),
         "report": ("REPORT",)}
DEPENDS = {"dem_coverage": ("dem_inventory",), "dem_feasibility": ("dem_coverage",),
           "dem_features": ("dem_feasibility",), "datasets": ("g06_restore",),
           "train_subsurface": ("datasets", "tests"), "train_surface_classic": ("datasets", "tests"),
           "train_surface_unet": ("datasets", "tests"), "generate_candidates": ("datasets", "tests"),
           "train_candidate_ranker": ("generate_candidates", "tests"), "evaluate": ("g06_restore",),
           "full_inference": ("evaluate", "tests"), "export": ("full_inference",)}
SOFT_DEPS = {"dem_coverage", "dem_feasibility", "dem_features"}   # BLOCKED_INPUT there does not block the geology path
MODULES = {"preflight": ["inventory"], "dem_inventory": ["inventory", "dem_io"], "dem_coverage": ["inventory"],
           "dem_feasibility": ["dem_io", "dem_features", "terrain_structure", "evaluation"],
           "dem_features": ["dem_io", "dem_features", "terrain_structure"], "g06_restore": ["models_subsurface"],
           "datasets": ["datasets", "training_labels"], "train_subsurface": ["crossfit", "models_subsurface"],
           "train_surface_classic": ["crossfit", "models_surface"], "train_surface_unet": ["models_surface"],
           "generate_candidates": ["candidate_generation", "datasets", "training_labels"],
           "train_candidate_ranker": ["models_candidates", "crossfit", "confidence", "representation"],
           "evaluate": ["evaluation", "representation"], "full_inference": ["inference", "representation"],
           "export": ["inference"], "report": [], "tests": []}
OUTPUTS = {
    "tests": [("audit/tests.csv", ("test", "status"))],
    "preflight": [("dependencies/existing_artifacts.csv", ("artifact", "status")),
                  ("dependencies/missing_dependencies.json", None), ("manifests/dependencies.json", None)],
    "dem_inventory": [("dem_catalog/dem_inventory.csv", ("tile_id", "status")),
                      ("dem_catalog/source_licenses.csv", ("source",))],
    "dem_coverage": [("tables/dem_coverage.csv", ("region", "local_dem_share")),
                     ("dem_catalog/dem_coverage.geojson", None)],
    "dem_feasibility": [("reports/DEM_FEASIBILITY.md", None), ("dem_catalog/feasibility_decision.json", None)],
    "dem_features": [("dem_features/dem_features_100m.tif", None), ("dem_features/dem_feature_registry.csv",
                                                                     ("feature_id",))],
    "g06_restore": [("dependencies/g06_store.json", None)],
    "datasets": [("splits/fits.csv", ("fit_id",)), ("datasets/subsurface/pool_meta.parquet", ("row_id",)),
                 ("datasets/dataset_summary.csv", ("dataset",))],
    "train_subsurface": [("models/subsurface_registry.csv", ("experiment", "fit_id", "status"))],
    "train_surface_classic": [("models/surface_registry.csv", ("experiment", "fit_id", "status"))],
    "train_surface_unet": [("models/surface_unet_registry.csv", ("experiment", "fit_id", "status"))],
    "generate_candidates": [("candidates/candidate_summary.csv", ("unit",)),
                            ("evaluation/recall/candidate_recall.csv", None)],
    "train_candidate_ranker": [("models/candidate_registry.json", None)],
    "evaluate": [("evaluation/metrics/metrics.csv", ("model", "scenario", "dti")),
                 ("evaluation/metrics/exact_budget.csv", None),
                 ("evaluation/promotion/promotion_decisions.csv", None)],
    "full_inference": [("predictions/full_region/full_region_manifest.json", None)],
    "export": [("evaluation/promotion/submission_eligibility.json", None)],
    "report": [("reports/DTI_EVALUATION.md", None)],
}
ARCHIVE_EXTRA = {"evaluate": ["evaluation/metrics", "evaluation/geographic", "evaluation/ablations"],
                 "generate_candidates": ["candidates", "datasets/candidates"]}
MERGED = ("train_subsurface", "train_surface_classic", "train_surface_unet")


class G07Pipeline:
    def __init__(self, config: dict, run_flags: dict | None = None, repository_commit: str | None = None,
                 factory=None, progress=print):
        self.cfg = config
        if run_flags is not None:
            self.cfg["run"] = {**RUN_DEFAULTS, **run_flags}
        self.warnings = validate_config(self.cfg)
        self.nb, self.run_flags = self.cfg["notebook"], self.cfg["run"]
        self.head = repository_commit or "unknown"
        self.say = progress or (lambda *_: None)
        self.local_root, self.drive_root = Path(self.nb["local_root"]), Path(self.nb["drive_root"])
        self.roots = [self.local_root, self.drive_root, REPO]
        self._factory, self._engine, self._units, self._g06, self._dem = factory, None, None, None, None
        self._engine_error = None
        self.src = self._resolve_sources()
        cache = self.local_root / ".g07_source_sha.json"
        self.src_fp = {k: PS.source_fingerprint(p, cache) for k, p in self.src.items() if p is not None}
        self.run_id = compute_run_id(self.cfg, self.src_fp, self.head)
        self.ws = PS.Workspace(self.local_root, self.drive_root, f"{self.nb['output_subdir']}/{self.run_id}",
                               bool(self.nb["sync_to_drive"]))
        self.root = self.ws.root
        self.ws.stage_in(["manifests", "dependencies", "dem_catalog", "tables", "dem_features", "splits", "datasets",
                          "models", "predictions", "candidates", "evaluation", "reports", "audit", "figures"])
        self.status = {s: NOT_RUN for s in STAGES}
        self.fps: dict = {}
        self.errors: list = []
        for w in self.warnings:
            self.say(f"[g07] config warning: {w}")
        self.save_run_manifest()

    # =========================================================================== plumbing
    def _resolve_sources(self) -> dict:
        import shutil

        from ..g06.inventory import resolve_source

        out = {}
        for key in (self.cfg["inputs"].get("sources") or {}):
            p = resolve_source({"sources": self.cfg["inputs"]["sources"]}, key, self.roots)
            if p is not None and str(p).startswith(str(self.drive_root)):
                dst = self.local_root / p.relative_to(self.drive_root)
                if not dst.exists() or dst.stat().st_size != p.stat().st_size:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
                p = dst
            out[key] = p
        return out

    def requested(self, stage: str) -> bool:
        if stage == "tests":
            return bool(self.nb.get("run_synthetic_tests", True))
        return any(bool(self.run_flags.get(f)) for f in FLAGS.get(stage, ()))

    def fp_of(self, stage: str):
        if stage in self.fps:
            return self.fps[stage]
        return (PS.load_cached_stage(self.root, stage) or {}).get("fingerprint")

    def _dep_ok(self, dep: str) -> bool:
        st = self.status.get(dep)
        if dep == "tests":
            return st not in (TESTS_FAILED, FAILED)
        if st in VALID or st == PS.NO_GO:
            return True
        return False

    def _run(self, stage: str, parts: list, fn, *, force: bool = False) -> dict:
        bad = [d for d in DEPENDS.get(stage, ()) if not self._dep_ok(d)]
        fp = PS.stable_hash(G07_VERSION, stage, parts, PS.code_identity([f"gems_bench.g07.{m}" for m in
                                                                         MODULES.get(stage, [])]))
        self.fps[stage] = fp
        outs = OUTPUTS.get(stage, [])
        reusable, probs = PS.validate_cache(self.root, stage, fp, outs)
        if not self.requested(stage):
            if reusable:
                return self._set(stage, REUSED, fp, {"note": "not requested; valid cache adopted"})
            return self._set(stage, NOT_REQUESTED, fp, {"note": f"RUN {FLAGS.get(stage)} off, no valid cache"},
                             save=False)
        if bad:
            st = BLOCKED_INPUT if all(self.status.get(d) == BLOCKED_INPUT for d in bad) else BLOCKED_UPSTREAM
            return self._set(stage, st, fp, {"reason": f"upstream {[(d, self.status.get(d)) for d in bad]}"})
        if reusable and not force and self.nb.get("reuse_cached", True):
            return self._set(stage, REUSED, fp, {"note": f"fingerprint {fp[:10]}"})
        prev = PS.load_cached_stage(self.root, stage)
        if prev and prev.get("status") == COMPLETE and prev.get("fingerprint") != fp and stage not in MERGED:
            moved = PS.archive_outputs(self.root, stage, ARCHIVE_EXTRA.get(stage, []) + [o[0] for o in outs],
                                       prev.get("fingerprint"))
            if moved:
                self.say(f"[{stage}] earlier outputs archived to history/{stage}/{str(prev['fingerprint'])[:10]}")
        t0 = time.perf_counter()
        self.say(f"[{stage}] running ...")
        try:
            st, info = fn()
        except Exception as e:
            st, info = FAILED, {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-5000:]}
            self.errors.append({"stage": stage, **info, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if st == COMPLETE:
            _, probs = PS.check_outputs(self.root, outs)
            if probs:
                st, info = FAILED, {**info, "error": f"required outputs invalid: {probs[:4]}"}
        info["seconds"] = round(time.perf_counter() - t0, 1)
        return self._set(stage, st, fp, info)

    def _set(self, stage, st, fp, info, save=True) -> dict:
        self.status[stage] = st
        if save:
            PS.save_manifest(self.root, stage, st, fp, OUTPUTS.get(stage, []), info, code_rev=G07_CODE_REV,
                             version=G07_VERSION, commit=self.head, run_id=self.run_id)
        self.save_run_manifest()
        note = info.get("note") or info.get("reason") or info.get("error") or ""
        self.say(f"[{stage}] {st}" + (f" | {str(note)[:300]}" if note else ""))
        if st == FAILED and info.get("traceback"):
            print(info["traceback"][-1500:])
        if save:
            PS.sync_artifacts(self.ws, ["manifests"])
        return {"stage": stage, "status": st, **info}

    def push(self, rels) -> None:
        PS.sync_artifacts(self.ws, rels)

    def save_run_manifest(self) -> None:
        PS.write_json(self.root / "manifests" / "run.json", {
            "run_id": self.run_id, "version": G07_VERSION, "code_rev": G07_CODE_REV, "commit": self.head,
            "sources": self.src_fp, "notebook_config": self.nb, "run_flags": self.run_flags,
            "stages": getattr(self, "status", {}), "fingerprints": getattr(self, "fps", {}),
            "drive_dir": str(self.ws.drive_path()), "written": time.strftime("%Y-%m-%dT%H:%M:%S")})
        PS.write_json(self.root / "manifests" / "runtime_errors.json", getattr(self, "errors", []))

    def figs(self, sub: str) -> Path:
        return self.root / "figures" / sub

    def figures_on(self) -> bool:
        return bool(self.nb.get("save_figures", True))

    # =========================================================================== shared context
    def engine(self):
        from ..g06.inventory import EngineUnavailable, attach_engine

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
            from ..g06.splits import build_validation_units

            sc = tuple(self.cfg["validation"]["scenarios"])
            self._units = {vu.key: vu for vu in build_validation_units(self.engine(), sc)}
        return self._units

    def grid(self):
        import rasterio

        from .contracts import GridSpec

        with rasterio.open(self.src["features"]) as ds:
            return GridSpec.from_dataset(ds)

    def g06_store(self):
        if self._g06 is None:
            from .inventory import inspect_existing_g06_store
            from .models_subsurface import load_g06_feature_store

            sel = inspect_existing_g06_store(self.cfg.get("g06") or {}, [self.drive_root, self.local_root])
            if sel["selected"] is None:
                raise FileNotFoundError(sel["detail"])
            self._g06 = load_g06_feature_store(sel["selected"]["run_dir"], self.local_root, self.drive_root,
                                               bool((self.cfg.get("g06") or {}).get("stage_local", True)))
        return self._g06

    def dem_store(self):
        from .dem_io import DemFeatureStore

        p = self.root / "dem_features" / "dem_features_100m.tif"
        if not p.exists():
            return None
        if self._dem is None or self._dem.path_ != p:
            self._dem = DemFeatureStore(p, float(self.cfg["dem"].get("min_coverage", 0.5)), self.fp_of("dem_features")
                                        or "")
        return self._dem

    def dem_sources(self):
        from .inventory import discover_dem_sources

        return discover_dem_sources(self.cfg["dem"], self.roots)

    def w2_maps(self) -> dict:
        out = {}
        eng = self.engine()
        for k, vu in self.units().items():
            try:
                out[k] = eng.w2_rank(vu.base)
            except (FileNotFoundError, ValueError):
                pass
        return out

    def seed(self) -> int:
        return int(self.nb["seed"])

    # =========================================================================== tests / preflight
    def run_tests(self) -> dict:
        def fn():
            import re

            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfE",
                                *sorted(str(p) for p in (REPO / "tests").glob("test_g07_*.py"))], cwd=str(REPO),
                               capture_output=True, text=True)
            out = r.stdout + "\n" + r.stderr
            self.ws.save_text(f"exit code {r.returncode}\n\n{out}", "audit/pytest_output.txt")
            fails = [m.group(1) for m in re.finditer(r"^(?:FAILED|ERROR) (\S+)", out, re.M)]
            self.ws.save_table(pd.DataFrame([{"test": "pytest tests/test_g07_*.py", "status": "PASS" if
                                              r.returncode == 0 else "FAIL", "detail": f"exit {r.returncode}"}] +
                                            [{"test": f, "status": "FAIL", "detail": ""} for f in fails]),
                               "audit/tests.csv")
            if r.returncode:
                return TESTS_FAILED, {"error": f"pytest exit {r.returncode}: {fails[:5] or out[-400:]}"}
            return COMPLETE, {"note": "all G07 tests passed"}
        if not self.requested("tests"):
            return self._set("tests", SKIPPED, None, {"note": "run_synthetic_tests=False"})
        return self._run("tests", [time.strftime("%Y-%m-%dT%H:%M")], fn, force=True)

    def preflight(self) -> dict:
        def fn():
            from ..g06 import config as G6C
            from ..g06.inventory import EngineUnavailable, inspect_source_raster
            from .inventory import build_dependency_report, inspect_existing_g06_store, inspect_w2_artifacts

            items = {}
            if self.src.get("features") is None:
                return FAILED, {"error": "GeoDAWN numerical raster not found"}
            fs6 = G6C._read_yaml(G6C.CONFIG_DIR / "feature_store.yaml")
            S = inspect_source_raster(self.src["features"], fs6["bands"], fs6["grid"])
            items["geodawn_raster"] = {"mandatory": True, "status": "OK" if not S["problems"] else "FAIL",
                                       "detail": "; ".join(S["problems"])[:400], "blocks": ["*"]}
            for k in ("faults", "submission"):
                items[k] = {"mandatory": True, "status": "OK" if self.src.get(k) else "FAIL",
                            "detail": str(self.src.get(k)), "blocks": ["datasets", "evaluate", "export"]}
            try:
                eng = self.engine()
                items["frozen_engine"] = {"mandatory": True, "status": "OK", "detail": json.dumps(
                    eng.identity(), default=str)[:300], "blocks": ["dem_feasibility", "datasets", "evaluate"]}
            except EngineUnavailable as e:
                items["frozen_engine"] = {"mandatory": True, "status": "FAIL", "detail": str(e)[:400],
                                          "blocks": ["dem_feasibility", "datasets", "evaluate"]}
            g6 = inspect_existing_g06_store(self.cfg.get("g06") or {}, [self.drive_root, self.local_root])
            items["g06_feature_store"] = {"mandatory": True, "status": g6["status"],
                                          "detail": (g6["selected"] or {}).get("run_dir") or g6["detail"],
                                          "blocks": ["datasets", "train_subsurface", "generate_candidates"]}
            w2 = inspect_w2_artifacts(self.cfg.get("w2") or {}, [self.local_root, self.drive_root])
            items["w2_heldout_maps"] = {"mandatory": True, "status": w2["status"], "detail": f"{len(w2['maps'])} "
                                        "unit maps", "blocks": ["evaluate", "generate_candidates"]}
            items["w2_full_region_score"] = {"mandatory": False, "status": "FOUND" if w2["full_score"] else "MISSING",
                                             "detail": w2["full_region_status"], "blocks": ["full_inference", "export"]}
            D, miss = build_dependency_report(items)
            A = D.rename(columns={"dependency": "artifact"})
            self.ws.save_table(A, "dependencies/existing_artifacts.csv")
            self.ws.save_json(miss, "dependencies/missing_dependencies.json")
            self.ws.save_json({"grid": S["grid"].to_dict(), "band_index": S["band_index"], "sources": self.src_fp,
                               "g06": g6, "w2": w2, "items": items}, "manifests/dependencies.json")
            self.push(["dependencies", "manifests"])
            if S["problems"]:
                return FAILED, {"error": "GeoDAWN raster contract violated: " + "; ".join(S["problems"])[:600]}
            return COMPLETE, {"note": f"mandatory missing: {miss['mandatory_missing'] or 'none'}; blocked stages: "
                                      f"{miss['blocked_stages'] or 'none'}"}
        return self._run("preflight", [self.src_fp, stage_config(self.cfg, "preflight")], fn, force=True)

    # =========================================================================== DEM inventory / coverage
    def inspect_dem(self) -> dict:
        return {"inventory": self.dem_inventory(), "coverage": self.dem_coverage()}

    def dem_inventory(self) -> dict:
        def fn():
            from .inventory import source_licenses

            dc = self.cfg["dem"]
            if (dc.get("download") or {}).get("enabled"):
                self._download_dem()
            T, sources = self.dem_sources()
            self.ws.save_table(T, "dem_catalog/dem_inventory.csv")
            self.ws.save_table(pd.DataFrame(T.attrs.get("existing_products", [])), "dem_catalog/existing_products.csv")
            self.ws.save_table(source_licenses(self.cfg, False), "dem_catalog/source_licenses.csv")
            if (dc.get("catalog") or {}).get("inspect_remote_headers") and len(T):
                from ..dem.inventory import inspect_headers

                cat = T[(T.status == "CATALOG_ONLY") & T.url.notna()]
                self.ws.save_table(inspect_headers(list(cat.url), int(dc["catalog"].get("max_remote_headers", 20))),
                                   "dem_catalog/remote_headers.csv")
            self.push(["dem_catalog"])
            n_local = int((T.status == "LOCAL").sum()) if len(T) else 0
            res = sorted(set(T.loc[T.status == "LOCAL", "res_m"].dropna().round(3))) if n_local else []
            return COMPLETE, {"note": f"{len(T)} DEM sources ({n_local} local and valid, resolutions {res} m); "
                                      f"{int((T.status == 'CATALOG_ONLY').sum()) if len(T) else 0} catalog-only",
                              "n_local": n_local}
        return self._run("dem_inventory", [stage_config(self.cfg, "dem_inventory"), self._dem_listing()], fn)

    def _dem_listing(self) -> list:
        from .dem_io import local_dem_files

        return [(p.name, p.stat().st_size) for p in local_dem_files(self.roots[:2], (self.cfg["dem"].get("catalog") or {})
                                                                    .get("local_dirs", ["data/raw/dem_1m"]))]

    def _download_dem(self) -> None:
        """Explicit, guarded download of catalog tiles intersecting the eligible region (dem.download.enabled)."""
        from ..artifacts.sync import DriveSync
        from ..dem.download import download_selected
        from ..dem.inventory import choose_per_tile, load_inventory, study_area_intersection

        dc = self.cfg["dem"]
        js = next((Path(r) / c for r in self.roots for c in (dc.get("catalog") or {}).get("url_lists", [])
                   if (Path(r) / c).exists()), None)
        if js is None:
            self.say("[dem] download requested but no URL list found")
            return
        inv = study_area_intersection(load_inventory(js), self.engine().ctx,
                                      float(dc["download"].get("min_valid_frac", 0.05)))
        sel = choose_per_tile(inv[inv.in_study_area], dc.get("overlap_policy", "latest"))
        sel = sel.sort_values("valid_px", ascending=False).head(int(dc["download"].get("max_tiles", 10)))
        R = download_selected(sel, self.local_root, max_tiles=int(dc["download"].get("max_tiles", 10)),
                              max_gib=float(dc["download"].get("max_gib", 6.0)),
                              sync=DriveSync(self.local_root, self.drive_root))
        self.ws.save_table(R, "dem_catalog/download_log.csv")

    def dem_coverage(self) -> dict:
        def fn():
            from .inventory import coverage_geojson, inspect_dem_coverage

            eng = self.engine()
            T, sources = self.dem_sources()
            ez = {d: eng.ctx.splits.eval_zone(d) for d in range(eng.ctx.splits.n_splits)}
            C, masks = inspect_dem_coverage(sources, self.grid(), eng.valid(), eng.template_valid(), eng.labels(), ez)
            self.ws.save_table(C, "tables/dem_coverage.csv")
            self.ws.save_json(coverage_geojson(sources), "dem_catalog/dem_coverage.geojson")
            if self.figures_on():
                from . import visualization as V

                V.plot_dem_coverage(masks["local"], masks["catalog"], masks["eligible"], self.grid().transform,
                                    self.figs("dem_atlas"))
            self.push(["tables", "dem_catalog", "figures"])
            e = C[C.region == "eligible"].iloc[0]
            return COMPLETE, {"note": f"local DEM covers {e.local_dem_share:.1%} of the eligible region "
                                      f"({e.local_dem_km2:.0f} km2); catalog {e.catalog_dem_share:.1%}"}
        return self._run("dem_coverage", [stage_config(self.cfg, "dem_coverage"), self.fp_of("dem_inventory")], fn)

    # =========================================================================== DEM features (pilot + full)
    def _dem_fp(self) -> str:
        """DEM-processing identity (configuration + code). Per-tile products are named by tile id and size, so the
        pilot and the full build share every finished tile."""
        dc = {k: v for k, v in self.cfg["dem"].items() if k not in ("download", "catalog", "research_use_ok",
                                                                    "competition_eligibility_verified")}
        return PS.stable_hash(dc, PS.code_identity(["gems_bench.g07.dem_features", "gems_bench.g07.terrain_structure",
                                                    "gems_bench.g07.dem_io"]), n=10)

    def _usable_sources(self, sources) -> list:
        """Local, valid sources; one per tile id (latest year) - overlapping projects are not double counted."""
        ok = [s for s in sources if s.status == "LOCAL"]
        best = {}
        for s in ok:
            if s.tile_id not in best or (s.year or 0) > (best[s.tile_id].year or 0):
                best[s.tile_id] = s
        return sorted(best.values(), key=lambda s: s.tile_id)

    def _build_dem_features(self, sources, tag: str, out_tif: Path) -> dict:
        """Fine tier per source + coarse tier over the merged coarse DEM -> mosaic ``out_tif`` (resumable)."""
        from .dem_features import aggregated_names, build_coarse_dem, feature_kinds, process_coarse, process_fine_source
        from .dem_io import mosaic_windows

        dc = self.cfg["dem"]
        grid = self.grid()
        fp = self._dem_fp()
        tdir = self.root / "dem_features" / "tiles" / fp
        tdir.mkdir(parents=True, exist_ok=True)
        files, names, max_bands = [], ["dem_coverage"], set()
        for i, s in enumerate(sources):
            p = tdir / f"fine_{s.tile_id}_{Path(s.path).stat().st_size}.tif"
            factor = max(int(round(float(dc.get("work_res_m", 1.0)) / float(s.res_m))), 1)
            if not (p.exists() and p.with_suffix(".json").exists()):
                self.say(f"  [{tag}] fine tier {i + 1}/{len(sources)} {s.tile_id} (factor {factor})")
                r = process_fine_source(s.path, s, dc, grid, p, factor, {"tile_id": s.tile_id, "project": s.project,
                                                                          "fingerprint": fp}, None)
                if r is None:
                    continue
                self.push([str((tdir / p.name).relative_to(self.root).as_posix())])
            files.append(p)
            k = feature_kinds(dc, float(s.res_m) * factor, "fine")
            names += [n for n in aggregated_names(k) if n not in names]
            max_bands |= {n for n, v in k.items() if v == "max"}
        if dc.get("coarse_scales_m") and sources:
            crs = sources[0].crs
            same = [s for s in sources if s.crs == crs]
            cfp = PS.stable_hash(fp, sorted(s.tile_id for s in same), n=10)          # depends on the source set
            cpath = self.root / "dem_features" / "coarse" / cfp / f"coarse_dem_{int(dc.get('coarse_res_m', 10))}m.tif"
            if not cpath.exists():
                self.say(f"  [{tag}] coarse DEM ({len(same)} sources at {dc.get('coarse_res_m', 10)} m)")
                build_coarse_dem(same, float(dc.get("coarse_res_m", 10.0)), cpath, crs)
            cdir = tdir / "coarse" / cfp
            cdir.mkdir(parents=True, exist_ok=True)
            cfiles = process_coarse(cpath, dc, grid, cdir, {"fingerprint": fp}, None)
            files += cfiles
            k = feature_kinds(dc, float(dc.get("coarse_res_m", 10.0)), "coarse")
            names += [n for n in aggregated_names(k) if n not in names]
            max_bands |= {n for n, v in k.items() if v == "max"}
        if not files:
            return {"status": "EMPTY"}
        mosaic_windows(files, grid, names, out_tif, tuple(max_bands))
        return {"status": "COMPLETE", "fingerprint": fp, "n_sources": len(sources), "n_products": len(files),
                "names": names, "coarse": str(cpath) if dc.get("coarse_scales_m") and sources else None}

    def assess_dem_feasibility(self) -> dict:
        def fn():
            from .evaluation import dem_signal_table, evaluate_candidate_recall, feasibility_decision
            from .representation import exact_k_mask

            dc, fc = self.cfg["dem"], self.cfg.get("feasibility") or {}
            T, sources = self.dem_sources()
            usable = self._usable_sources(sources)
            C = self.ws.load_table("tables/dem_coverage.csv")
            licence_ok = bool(dc.get("research_use_ok", True))
            facts, S, R = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            info = {}
            if usable and licence_ok:
                pilot = self._pilot_sources(usable, int(fc.get("max_tiles", 12)))
                out = self.root / "dem_features" / "pilot" / "dem_features_100m_pilot.tif"
                info = self._build_dem_features(pilot, "pilot", out)
                if info.get("status") == "COMPLETE":
                    from .dem_io import DemFeatureStore

                    st = DemFeatureStore(out, float(dc.get("min_coverage", 0.5)), info["fingerprint"])
                    w2 = self.w2_maps()
                    U = {k: v for k, v in self.units().items() if v.scenario in ("V1", "V3")}
                    S, facts = dem_signal_table(U, st, w2, self.cfg, float(dc.get("min_coverage", 0.5)),
                                                {"W2": w2})
                    R = self._dem_recall(U, st, w2, fc, exact_k_mask, evaluate_candidate_recall)
                    if self.figures_on():
                        self._feasibility_figures(S, U, st, w2, pilot)
            dec = feasibility_decision(C, S, facts, R, licence_ok, bool(usable), self.cfg)
            self.ws.save_table(S, "tables/dem_fault_signal.csv")
            self.ws.save_table(facts, "tables/dem_feasibility_units.csv")
            self.ws.save_table(R, "evaluation/recall/dem_candidate_recall.csv")
            self.ws.save_json({**dec, "pilot": {k: v for k, v in info.items() if k != "names"}},
                              "dem_catalog/feasibility_decision.json")
            self.ws.save_text(self._feasibility_report(dec, C, S, facts, R, T, usable), "reports/DEM_FEASIBILITY.md")
            self.push(["tables", "dem_catalog", "reports", "evaluation", "figures", "dem_features/pilot"])
            st = {"GO": COMPLETE, "NO_GO": PS.NO_GO, "INCONCLUSIVE": PS.NO_GO, "BLOCKED_INPUT": BLOCKED_INPUT}[
                dec["decision"]]
            if st == BLOCKED_INPUT:                     # outputs exist; record the blocked decision explicitly
                return st, {"note": f"DEM feasibility BLOCKED_INPUT: {dec['reasons']}", "decision": dec["decision"]}
            return (COMPLETE if st == COMPLETE else PS.NO_GO), {"note": f"DEM decision {dec['decision']} "
                                                                       f"{dec.get('reasons', [])}",
                                                               "decision": dec["decision"]}
        return self._run("dem_feasibility", [stage_config(self.cfg, "dem_feasibility"), self.fp_of("dem_coverage"),
                                             self._dem_listing()], fn)

    def _pilot_sources(self, usable: list, n: int) -> list:
        """Label-free pilot choice: the sources covering the largest eligible area inside V1 evaluation zones."""
        from .dem_io import analysis_window

        eng = self.engine()
        grid = self.grid()
        ez = np.zeros(grid.shape, bool)
        for d in range(eng.ctx.splits.n_splits):
            ez |= eng.ctx.splits.eval_zone(d)
        el = eng.valid() & eng.template_valid() & ~eng.labels() & ez
        score = []
        for s in usable:
            w = analysis_window(s.bounds, s.crs or "EPSG:26911", grid, pad=0) if s.bounds else None
            score.append(int(el[w[0]:w[1], w[2]:w[3]].sum()) if w else 0)
        order = np.argsort(-np.asarray(score), kind="stable")
        return [usable[i] for i in order[:n] if score[i] > 0] or usable[:n]

    def _dem_recall(self, U, st, w2, fc, exact_k_mask, evaluate_candidate_recall) -> pd.DataFrame:
        """Label-free DEM score (mean rank of configured features) vs W2 at the same budget, on covered cells."""
        from ..gems3.representation import rank01

        rows = []
        feats = [f for f in fc.get("dem_score_features", []) if f in st.names]
        if not feats:
            return pd.DataFrame()
        for key, vu in sorted(U.items()):
            u = vu.unit
            win = (u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
            cov = np.nan_to_num(st.coverage(win)) >= float(self.cfg["dem"].get("min_coverage", 0.5))
            el = u.scored_mask() & cov
            if not el.any() or key not in w2:
                continue
            A = st.load(feats, win)
            score = np.mean([rank01(np.nan_to_num(a, nan=0.0), el) for a in A], 0)
            for f in fc.get("recall_fracs", [0.01, 0.02]):
                pools = {"dem_score": exact_k_mask(score, el, f, offset=(u.sl[0].start, u.sl[1].start))[0],
                         "w2": exact_k_mask(w2[key], el, f, offset=(u.sl[0].start, u.sl[1].start))[0]}
                r = evaluate_candidate_recall(vu, np.zeros(el.shape, np.int32), pools)
                if len(r):
                    rows.append(r.assign(frac=f))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _feasibility_figures(self, S, U, st, w2, pilot) -> None:
        from . import visualization as V

        if len(S):
            V.plot_signal_auc(S, self.figs("dem_feasibility"))
        self._case_atlas(U, st, w2, pilot)

    def _case_atlas(self, U, st, w2, pilot) -> None:
        """20-30 representative cases inside the pilot DEM coverage: W2 misses, W2 detections, W2 false positives
        (selection uses held-out truth: post-hoc diagnostic only)."""
        from scipy import ndimage

        from . import visualization as V
        from .evaluation import evaluate_map, unit_protocol
        from .representation import exact_k_mask

        fc = self.cfg.get("feasibility") or {}
        quotas = fc.get("case_quotas", {"w2_missed": 10, "w2_recovered": 8, "w2_false_positive": 8})
        box = int(fc.get("case_box_cells", 40))
        eng = self.engine()
        prim = self.cfg["representation"]
        cases = {k: [] for k in quotas}
        for key, vu in sorted(U.items()):
            if vu.scenario != "V1" or key not in w2:
                continue
            u = vu.unit
            win = (u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
            cov = np.nan_to_num(st.coverage(win)) >= float(self.cfg["dem"].get("min_coverage", 0.5))
            proto = unit_protocol(vu, eng.template_valid()[u.sl])
            _, srows = evaluate_map(vu, w2[key], {"top": {"select": "top", "post": []}}, [float(prim["primary_frac"])],
                                    proto, {"model": "W2"}, self.cfg)
            for s in srows:
                m = vu.sysmap == s["system_id"]
                if not (m & cov).any():
                    continue
                rr, cc = np.nonzero(m)
                cases["w2_recovered" if s["recovered"] else "w2_missed"].append((key, int(rr.mean()), int(cc.mean()),
                                                                                 f"system {s['system_id']}"))
            q, _ = exact_k_mask(w2[key], u.scored_mask(), float(prim["primary_frac"]))
            far = q & cov & (ndimage.distance_transform_edt(~(vu.eval_truth() | vu.known)) * 0.1 > 2.0)
            lab, n = ndimage.label(far, np.ones((3, 3), bool))
            if n:
                sizes = np.bincount(lab.ravel())[1:]
                for lb in np.argsort(-sizes)[:3] + 1:
                    rr, cc = np.nonzero(lab == lb)
                    cases["w2_false_positive"].append((key, int(rr.mean()), int(cc.mean()), "W2 false positive"))
        rng = np.random.default_rng(self.seed())
        chosen = []
        for k, lst in cases.items():
            idx = rng.permutation(len(lst))[:int(quotas.get(k, 0))]
            chosen += [(k, *lst[i]) for i in idx]
        self.ws.save_table(pd.DataFrame(chosen, columns=["category", "unit", "row", "col", "what"]),
                           "tables/dem_feasibility_cases.csv")
        grid = self.grid()
        T = grid.transform
        for i, (cat, key, r, c, what) in enumerate(chosen[:int(fc.get("max_cases", 30))]):
            vu = U[key]
            r0, c0 = r + vu.sl[0].start - box // 2, c + vu.sl[1].start - box // 2
            r0, c0 = max(r0, 0), max(c0, 0)
            r1, c1 = min(r0 + box, grid.height), min(c0 + box, grid.width)
            tA = (T[0], 0, T[2] + c0 * T[0], 0, T[4], T[5] + r0 * T[4])
            panels = {"W2 rank": (w2[key][r0 - vu.sl[0].start:r1 - vu.sl[0].start, c0 - vu.sl[1].start:
                                          c1 - vu.sl[1].start], tA, None)}
            for f in (self.cfg.get("feasibility") or {}).get("case_panels", []):
                if f in st.names:
                    panels[f] = (st.load([f], (r0, r1, c0, c1))[0], tA, "gray" if "hs" in f else None)
            truth = np.zeros((r1 - r0, c1 - c0), bool)
            vt = vu.full(vu.eval_truth(), grid.shape)[r0:r1, c0:c1]
            truth |= vt
            xm, ym = T[2] + (c0 + box / 2) * T[0], T[5] + (r0 + box / 2) * T[4]
            panels.update(self._native_case_panels(pilot, xm, ym, box * grid.pixel_m))
            if i == 0 and "native hillshade" in panels:
                self._resolution_comparison(panels["native hillshade"], (r0, r1, c0, c1), tA)
            ext_x = (T[2] + c0 * T[0], T[2] + c1 * T[0])
            ext_y = (T[5] + r1 * T[4], T[5] + r0 * T[4])
            V.plot_case_atlas({"title": f"{cat}: {key} {what}", "panels": panels, "truth": (truth, tA),
                               "xlim": ext_x, "ylim": ext_y,
                               "attribution": "W2 recovered" if cat == "w2_recovered" else
                               ("W2 missed - does the DEM show it?" if cat == "w2_missed" else
                                "W2 false positive - does the DEM explain it?"),
                               "facts": {"category": cat, "unit": key, "row": r, "col": c}},
                              self.figs("dem_feasibility"), f"case_{i:02d}_{cat}")

    def _resolution_comparison(self, native: tuple, win: tuple, tA: tuple) -> None:
        """Native hillshade vs the GeoDAWN 100 m detrended elevation over the same extent (exact band lookup)."""
        import rasterio

        from ..g06.raster_io import read_window_bands, resolve_exact_band
        from . import visualization as V

        with rasterio.open(self.src["features"]) as ds:
            i = resolve_exact_band(ds, "det_elev")
        e = read_window_bands(self.src["features"], [i], win)[0]
        e[np.abs(e) > 1e20] = np.nan
        V.plot_resolution_comparison(native[0], native[1], e, tA, self.figs("dem_atlas"), "resolution_1m_vs_100m")

    def _native_case_panels(self, pilot: list, xm: float, ym: float, box_m: float) -> dict:
        """Native-resolution hillshade and scarp strength around (xm, ym) from the first pilot source covering it,
        at ``feasibility.case_res_m`` (memory-bounded; fine-tier operators on the two smallest scales)."""
        import rasterio
        from rasterio.transform import rowcol

        from ..g06.raster_io import TileWindow
        from .dem_features import dem_multiscale_features
        from .dem_io import read_dem_window, source_vscale

        fc = self.cfg.get("feasibility") or {}
        for s in pilot:
            w, so, e, n = s.bounds
            if not (w <= xm <= e and so <= ym <= n):
                continue
            with rasterio.open(s.path) as ds:
                res0 = abs(ds.res[0])
                f = max(int(round(max(float(fc.get("case_res_m", 2.0)), float(self.cfg["dem"].get("work_res_m", 1.0)))
                                  / res0)), 1)
                rr, cc = rowcol(ds.transform, xm, ym)
                h0 = int(box_m / (res0 * f) / 2)
                wwin = TileWindow("case", rr // f - h0, rr // f + h0, cc // f - h0, cc // f + h0, 64,
                                  ds.height // f, ds.width // f)
                z = read_dem_window(ds, wwin, f, source_vscale(s))
                sc = sorted(float(x) for x in self.cfg["dem"]["scales_m"])[:2]
                cfg = {**self.cfg["dem"], "scales_m": sc, "openness_radius_m": 0, "relief_windows_m": []}
                feats = dem_multiscale_features(z, res0 * f, cfg, "fine")
                rs, cs = wwin.core
                Tn = (ds.transform.a * f, 0, ds.transform.c + wwin.col0 * ds.transform.a * f, 0,
                      ds.transform.e * f, ds.transform.f + wwin.row0 * ds.transform.e * f)
            out = {}
            hs = [k for k in feats if k.startswith("dem_hs") and k != "dem_hs_range"]
            if hs:
                out["native hillshade"] = (feats[hs[0]][rs, cs], Tn, "gray")
            sk = [k for k in feats if k.startswith("dem_scarp_s") and "max" not in k]
            if sk:
                out["native scarp strength"] = (feats[sk[0]][rs, cs], Tn, None)
            return out
        return {}

    def _feasibility_report(self, dec, C, S, facts, R, T, usable) -> str:
        from ..g06.pipeline import _md

        lines = ["# DEM feasibility (G07 first milestone)", "",
                 f"Run `{self.run_id}` | code `{G07_CODE_REV}` | commit `{self.head}`", "",
                 f"## Decision: **{dec['decision']}**", ""]
        for k, v in (dec.get("criteria") or {}).items():
            lines.append(f"* {k}: {v['value']} -> {'PASS' if v['pass'] else 'FAIL'}")
        lines += ["", f"Reasons: {dec.get('reasons')}", "", f"_{dec.get('note', '')}_", "",
                  "## Sources", "", f"{len(T)} DEM sources in the catalog / on disk; {len(usable)} usable locally "
                  "(headers verified: resolution, CRS, vertical unit).", "",
                  _md(T, 30, ["tile_id", "status", "res_m", "crs", "vertical_unit", "vertical_unit_source", "project",
                              "problems"]),
                  "## Coverage of the eligible region", "", _md(C), "## Fault signal (held-out truth, diagnostic)", "",
                  _md(S, 30, ["feature", "auc", "auc_abs", "rho_with_coverage", "scale_stability",
                              "truth_orient_agreement", "mean_at_w2_fp", "mean_at_w2_tp", "is_reference"]),
                  "## Candidate recall at controlled area (DEM score vs W2, DEM-covered cells)", "",
                  _md(R, 30, ["unit", "frac", "source", "n_systems", "n_covered", "recall", "area_frac"]),
                  "## What to do next", ""]
        if dec["decision"] == "BLOCKED_INPUT":
            lines += ["* No usable high-resolution DEM (or research use not confirmed). Put the USGS TNM URL list at "
                      "`MyDrive/GEMS_2/data/raw/dem_1m/tnm_urls.json`, then either download pilot tiles with "
                      "`dem.download.enabled: true` (guarded: max tiles / GiB) or copy tiles to "
                      "`MyDrive/GEMS_2/data/raw/dem_1m/<project>/`. The 100 m `det_elev` band is NEVER used as a "
                      "substitute.", "* Continue with the geology path: BUILD_DATASETS + TRAIN_SUBSURFACE."]
        elif dec["decision"] == "GO":
            lines += ["* GO: DEM_FEATURES (all local tiles) -> BUILD_DATASETS -> TRAIN_SURFACE_CLASSIC."]
        else:
            lines += ["* NO_GO / INCONCLUSIVE: keep the DEM as an optional feature family; prioritise improved "
                      "subsurface supervision (TRAIN_SUBSURFACE) and candidate ranking. Re-assess only with more "
                      "coverage (never by lowering the predeclared thresholds)."]
        return "\n".join(lines) + "\n"

    def prepare_dem_features(self) -> dict:
        def fn():
            from .contracts import DEMFeatureSpec
            from .dem_features import feature_kinds

            dec = (self.ws.read_json("dem_catalog/feasibility_decision.json") or {}).get("decision")
            if self.cfg["dem"].get("require_go", True) and dec != "GO":
                return BLOCKED_INPUT, {"note": f"DEM feasibility decision {dec}: DEM features not built "
                                               "(dem.require_go)"}
            _, sources = self.dem_sources()
            usable = self._usable_sources(sources)
            if not usable:
                return BLOCKED_INPUT, {"note": "no usable local DEM"}
            info = self._build_dem_features(usable, "full", self.root / "dem_features" / "dem_features_100m.tif")
            if info.get("status") != "COMPLETE":
                return BLOCKED_INPUT, {"note": "no DEM data produced"}
            dc = self.cfg["dem"]
            rows = []
            for tier, res in (("fine", float(dc.get("work_res_m", 1.0))), ("coarse", float(dc.get("coarse_res_m",
                                                                                                10.0)))):
                for k, v in feature_kinds(dc, res, tier).items():
                    rows.append(DEMFeatureSpec(k, f"{tier} tier {k}", "", None, k.split("_")[1], v, res).to_row())
            self.ws.save_table(pd.DataFrame(rows), "dem_features/dem_feature_registry.csv")
            self._agreement_raster()
            self._dem = None
            self.push(["dem_features"])
            return COMPLETE, {"note": f"{info['n_sources']} sources -> {len(info['names'])} bands",
                              "dem_fingerprint": info["fingerprint"]}
        return self._run("dem_features", [stage_config(self.cfg, "dem_features"), self._dem_listing(),
                                          self.fp_of("dem_feasibility")], fn)

    def _agreement_raster(self) -> None:
        """Terrain-vs-geophysics orientation agreement (100 m) when the G06 store has the structure orientations."""
        from ..g06.raster_io import write_feature_raster
        from .dem_io import DemFeatureStore
        from .terrain_structure import orientation_agreement

        try:
            g6 = self.g06_store()
        except Exception as e:
            self.say(f"  agreement raster skipped ({e})")
            return
        d = DemFeatureStore(self.root / "dem_features" / "dem_features_100m.tif")
        sc = int((self.cfg.get("dem") or {}).get("agreement_scale_m", 300))
        dem_o = next((n[:-5] for n in d.names if n.endswith("_cos2") and n.startswith("dem_orient_s")), None)
        out = {}
        for fld in ("rtp", "iso_grav_anom"):
            oc, os_ = f"struct_{fld}_ocos_s{sc}", f"struct_{fld}_osin_s{sc}"
            if dem_o and oc in g6.names and os_ in g6.names:
                out[f"x_dem_{fld}_agree"] = orientation_agreement(d.load([dem_o + "_cos2"])[0], d.load(
                    [dem_o + "_sin2"])[0], g6.load([oc])[0], g6.load([os_])[0]).astype(np.float32)
        if out:
            cov = d.coverage()
            out["dem_coverage"] = cov
            write_feature_raster(self.root / "dem_features" / "dem_geo_agreement.tif", out, self.grid(),
                                 {"G07": "terrain-geophysics agreement"})

    # =========================================================================== G06 restore / datasets
    def restore_g06(self) -> dict:
        def fn():
            st = self.g06_store()
            from ..g06.feature_store import validate_feature_store

            probs = validate_feature_store(st, self.grid(), 32611)
            self.ws.save_json({"root": str(st.root), "schema_hash": st.schema_hash, "n_features": len(st.names),
                               "problems": probs}, "dependencies/g06_store.json")
            if probs:
                return FAILED, {"error": f"G06 store invalid: {probs[:4]}"}
            return COMPLETE, {"note": f"G06 store {st.schema_hash} ({len(st.names)} channels) restored, not rebuilt"}
        return self._run("g06_restore", [stage_config(self.cfg, "g06_restore")], fn)

    def combined_store(self, with_dem: bool = True):
        from .datasets import CombinedStore
        from .dem_io import DemFeatureStore

        stores = [self.g06_store()]
        if with_dem and self.dem_store() is not None:
            stores.append(self.dem_store())
            ag = self.root / "dem_features" / "dem_geo_agreement.tif"
            if ag.exists():
                stores.append(DemFeatureStore(ag, float(self.cfg["dem"].get("min_coverage", 0.5))))
        return CombinedStore(stores)

    def surface_store(self):
        from .datasets import CombinedStore
        from .dem_io import DemFeatureStore

        d = self.dem_store()
        if d is None:
            return None
        ag = self.root / "dem_features" / "dem_geo_agreement.tif"
        return CombinedStore([d] + ([DemFeatureStore(ag, float(self.cfg["dem"].get("min_coverage", 0.5)))] if
                                    ag.exists() else []))

    def prepare_training_data(self) -> dict:
        def fn():
            from ..g06 import splits as SPL
            from ..g06.datasets import unit_file
            from .datasets import build_subsurface_pixel_dataset, build_surface_patch_dataset, subsurface_labels

            build_training_visibility, fit_ids = SPL.build_training_visibility, SPL.fit_ids
            verify_fold_isolation, visibility_row = SPL.verify_fold_isolation, SPL.visibility_row

            eng, U = self.engine(), self.units()
            val = self.cfg["validation"]
            S = eng.ctx.splits
            fits = fit_ids(S.n_splits, int(eng.ctx.folds.n_folds), val["scenarios"], bool(val.get("nested")),
                           tuple(val.get("nested_scenarios", ("V1", "V3")))) + ["FULL"]
            rows, audit = [], []
            for fid in fits:
                vis = build_training_visibility(eng, fid, val, tuple(val["scenarios"]))
                rows.append(visibility_row(vis))
                outer = U.get(f"{vis.scenario}:{vis.outer}") if vis.inner is not None else None
                audit += verify_fold_isolation(eng, vis, U, outer)
            A = pd.DataFrame(audit)
            self.ws.save_table(pd.DataFrame(rows), "splits/fits.csv")
            self.ws.save_table(A, "splits/isolation_audit.csv")
            if len(A) and (A.status != "PASS").any():
                return FAILED, {"error": f"isolation failed: {A[A.status != 'PASS'].head(3).to_dict('records')}"}
            sm = np.asarray(S.system_map)
            lab = self.cfg["labels"]
            summary = []
            g6 = self.g06_store()
            m = build_subsurface_pixel_dataset(g6, eng.labels(), sm, {**self.cfg["datasets"]["subsurface_sampling"],
                                                                     "seed": self.seed()},
                                               self.root / "datasets" / "subsurface")
            summary.append({"dataset": "subsurface_pool", **{k: v for k, v in m.items() if k != "strata"}})
            meta = pd.read_parquet(self.root / "datasets" / "subsurface" / "pool_meta.parquet")
            for fid in fits:
                vis = build_training_visibility(eng, fid, val, tuple(val["scenarios"]))
                L = subsurface_labels(meta, vis, eng.valid(), sm, lab["supervision"], self.cfg["datasets"][
                    "subsurface_sampling"], self.cfg.get("inner_cv") or {}, lab.get("improved", {}), self.seed(),
                                      float(lab.get("proximity_lambda_km", 0.3)))
                self.ws.save_table(L, f"datasets/subsurface/labels/{unit_file(fid)}.parquet")
                summary.append({"dataset": f"subsurface_labels:{fid}", "n_rows": len(L),
                                "n_pos": int((L.cls == 1).sum()), "n_systems": int(L.loc[L.cls == 1, "system_id"]
                                                                                  .nunique())})
            ds = self.dem_store()
            if ds is not None:
                from ..g06.datasets import build_pixel_pool

                smeta, sX = build_pixel_pool(ds, eng.labels(), sm, {**self.cfg["datasets"]["surface_sampling"],
                                                                   "seed": self.seed()})
                ag = self.surface_store()
                if ag is not None and len(ag.names) > len(ds.names):
                    extra = [n for n in ag.names if n not in ds.names]
                    sX = pd.concat([sX, pd.DataFrame(ag.sample(extra, smeta.row.to_numpy(), smeta.col.to_numpy()),
                                                     columns=extra)], axis=1)
                self.ws.save_table(smeta, "datasets/surface/pool_meta.parquet")
                self.ws.save_table(sX, "datasets/surface/pool_features.parquet")
                for fid in fits:
                    vis = build_training_visibility(eng, fid, val, tuple(val["scenarios"]))
                    L = subsurface_labels(smeta, vis, eng.valid(), sm, lab["supervision"],
                                          self.cfg["datasets"]["surface_sampling"], self.cfg.get("inner_cv") or {},
                                          lab.get("improved", {}), self.seed(), float(lab.get("proximity_lambda_km", 0.3)))
                    self.ws.save_table(L, f"datasets/surface/labels/{unit_file(fid)}.parquet")
                summary.append({"dataset": "surface_pool", "n_rows": len(smeta), "n_features": sX.shape[1] - 1})
                cands = sorted((self.root / "dem_features" / "coarse").glob("*/coarse_dem_*m.tif"),
                               key=lambda p: p.stat().st_mtime)
                cpath = cands[-1] if cands else None                     # the most recent (full) coarse DEM
                if self.cfg["datasets"].get("surface_patches", True) and cpath is not None:
                    ez = {d: S.eval_zone(d) for d in range(S.n_splits)}
                    pm = build_surface_patch_dataset(cpath, self.grid(), eng.valid(), ds.coverage(), self.cfg["dem"],
                                                     self.cfg["surface"]["unet"], self.root / "datasets" / "surface"
                                                     / "patches", np.asarray(eng.ctx.folds.fold_map), ez, self.say)
                    summary.append({"dataset": "surface_patches", "n_rows": pm["n_patches"]})
            else:
                summary.append({"dataset": "surface", "n_rows": 0, "note": "BLOCKED_INPUT: no DEM feature store"})
            self.ws.save_table(pd.DataFrame(summary), "datasets/dataset_summary.csv")
            self.push(["splits", "datasets"])
            return COMPLETE, {"note": f"{len(fits)} fits; {len(summary)} dataset parts",
                              "feature_schema_hash": g6.schema_hash}
        parts = [stage_config(self.cfg, "datasets"), self.fp_of("g06_restore"), self.fp_of("dem_features")]
        return self._run("datasets", parts, fn)

    # =========================================================================== model benchmarks
    def _fit_list(self, nested_for: str | None = None, exps_nested=()) -> list:
        F = self.ws.load_table("splits/fits.csv")
        fits = [f for f in (list(F.fit_id) if len(F) else []) if f != "FULL"]
        return [f for f in fits if "|" not in f or nested_for in exps_nested]

    def _registry(self, rel, rows) -> pd.DataFrame:
        prev = self.ws.load_table(rel)
        R = pd.DataFrame(rows)
        keep = prev[~prev.experiment.isin(set(R.experiment))] if len(prev) and len(R) else prev
        ALL = pd.concat([keep, R], ignore_index=True) if len(keep) else R
        self.ws.save_table(ALL, rel)
        return R

    def benchmark_subsurface_models(self) -> dict:
        def fn():
            from .crossfit import crossfit_subsurface_model
            from .models_subsurface import select_subsurface_features

            st = self.g06_store()
            sc = self.cfg["subsurface"]
            d = self.root / "datasets" / "subsurface"
            meta = pd.read_parquet(d / "pool_meta.parquet")
            rows = []
            for eid, ec in sc["experiments"].items():
                cols = select_subsurface_features(st, ec["families"])
                X = pd.read_parquet(d / "pool_features.parquet", columns=["row_id"] + cols)
                params = self.cfg["models"][ec.get("params", "lightgbm")]
                efp = PS.stable_hash(eid, ec, params, self.fp_of("datasets"), st.schema_hash, self.seed(), n=16)
                fl = self._fit_list(eid, sc.get("nested_experiments", []))
                rows += crossfit_subsurface_model(eid, "lightgbm", cols, st, X, meta, d / "labels", fl, self.units(),
                                                  self.root, params, self.seed(), efp, ec.get("weight", "weight_g06"),
                                                  ec.get("target", "classification"), progress=self.say)
                self.push(["models", "predictions"])
            R = self._registry("models/subsurface_registry.csv", rows)
            return COMPLETE, {"note": f"{len(R)} fits of {R.experiment.nunique() if len(R) else 0} experiments"}
        return self._run("train_subsurface", [stage_config(self.cfg, "train_subsurface"), self.fp_of("datasets")],
                         fn)

    def benchmark_surface_models(self) -> dict:
        def fn():
            from .crossfit import crossfit_surface_model

            st = self.surface_store()
            if st is None:
                return BLOCKED_INPUT, {"note": "no DEM feature store (DEM_FEATURES not GO / not built)"}
            sc = self.cfg["surface"]
            d = self.root / "datasets" / "surface"
            if not (d / "pool_meta.parquet").exists():
                return BLOCKED_INPUT, {"note": "no surface pool (BUILD_DATASETS after DEM_FEATURES)"}
            meta = pd.read_parquet(d / "pool_meta.parquet")
            cov = st.footprint()
            rows = []
            for eid, ec in sc["experiments"].items():
                cols = [n for n in st.usable() if n.startswith(tuple(ec["prefixes"]))]
                if not cols:
                    rows.append({"experiment": eid, "fit_id": "*", "status": "BLOCKED_INPUT", "detail": "no feature"})
                    continue
                X = pd.read_parquet(d / "pool_features.parquet", columns=["row_id"] + cols)
                params = self.cfg["models"][ec.get("params", ec["model"])]
                efp = PS.stable_hash(eid, ec, params, self.fp_of("datasets"), st.schema_hash, self.seed(), n=16)
                fl = self._fit_list(eid, sc.get("nested_experiments", []))
                rows += crossfit_surface_model(eid, ec["model"], cols, st, X, meta, d / "labels", fl, self.units(),
                                               self.root, params, self.seed(), efp, ec.get("weight", "weight_improved"),
                                               "classification", cov, progress=self.say)
                self.push(["models", "predictions"])
            R = self._registry("models/surface_registry.csv", rows)
            return COMPLETE, {"note": f"{len(R)} surface fits"}
        return self._run("train_surface_classic", [stage_config(self.cfg, "train_surface_classic"),
                                                   self.fp_of("datasets")], fn)

    def train_surface_unet_stage(self) -> dict:
        def fn():
            from ..g06.datasets import unit_file
            from ..g06.models_patch import save_patch_model
            from ..g06.models_pixel import oof_path
            from ..g06.splits import build_training_visibility
            from ..g06.supervision import build_orientation_targets, build_pixel_supervision
            from .crossfit import build_inner_folds
            from .models_surface import predict_surface_tiles, train_surface_unet

            pdir = self.root / "datasets" / "surface" / "patches"
            man = self.ws.read_json("datasets/surface/patches/surface_manifest.json")
            if not man or not man.get("memmap"):
                return BLOCKED_INPUT, {"note": "no surface patch dataset (needs DEM GO + BUILD_DATASETS)"}
            uc = self.cfg["surface"]["unet"]
            eng, U = self.engine(), self.units()
            idx = pd.read_parquet(pdir / "patch_index.parquet")
            f = int(round(100.0 / float(man["res_m"])))
            efp = PS.stable_hash(uc, man["fingerprint"], self.seed(), n=10)
            rows = []
            for fid in uc.get("fits", ["V1:0", "V1:1", "V1:2"]):
                vis = build_training_visibility(eng, fid, self.cfg["validation"], tuple(self.cfg["validation"]
                                                                                       ["scenarios"]))
                sup = build_pixel_supervision(vis.visible, vis.region, eng.valid(), self.cfg["labels"]["supervision"])
                orient = build_orientation_targets(vis.visible, 2.0, 0.5)
                ic = self.cfg.get("inner_cv") or {}
                inner = build_inner_folds(vis.region, vis.visible, float(ic.get("block_km", 20)),
                                          int(ic.get("n_folds", 3)), self.seed())
                model, scaler, info = train_surface_unet(pdir / man["memmap"], idx, vis, sup["classes"], orient,
                                                         inner, uc, f, bool(self.nb.get("use_gpu")), self.seed(),
                                                         self.say)
                mdir = self.root / "models" / "surface_unet" / efp / unit_file(fid)
                save_patch_model(model, scaler, info, mdir)
                full = predict_surface_tiles(model, scaler, pdir / man["memmap"], idx, self.grid().shape, f,
                                             bool(self.nb.get("use_gpu")))
                for key in vis.predicts:
                    vu = U.get(key)
                    if vu is None:
                        continue
                    pf = oof_path(self.root, "G07-C1", fid, key, efp)
                    pf.parent.mkdir(parents=True, exist_ok=True)
                    crop = np.where(vu.unit.ev, full[vu.sl], np.nan).astype(np.float32)
                    PS.atomic_save(pf, lambda t, a=crop: np.save(t, a))
                rows.append({"experiment": "G07-C1", "fit_id": fid, "status": "COMPLETE", "fingerprint": efp,
                             "best_inner_val_loss": info["best_inner_val_loss"], "n_train": info["n_train"]})
                del vis, sup, inner
                self.push(["models", "predictions"])
            self._registry("models/surface_unet_registry.csv", rows)
            return COMPLETE, {"note": f"{len(rows)} U-Net fits"}
        return self._run("train_surface_unet", [stage_config(self.cfg, "train_surface_unet"),
                                                self.fp_of("datasets")], fn)

    # =========================================================================== candidates
    def _current_fp(self, rel: str) -> dict:
        from ..g06.models_pixel import current_fingerprints

        return current_fingerprints(self.ws.load_table(rel))

    def _model_map(self, which: str, key: str, outer=None):
        """Honest held-out (or nested inner, ``outer``) map of the configured surface / subsurface experiment."""
        from ..g06.models_pixel import load_oof

        exp = (self.cfg["candidates"].get("model_maps") or {}).get(which)
        reg = {"surface": "models/surface_registry.csv", "subsurface": "models/subsurface_registry.csv"}[which]
        fp = self._current_fp(reg).get(exp) if exp else None
        if not fp:
            return None
        return load_oof(self.root, exp, key, fp, outer)

    def generate_candidates(self) -> dict:
        def fn():
            from ..g06.datasets import nested_known_table, unit_file
            from .candidate_generation import generate_unit_candidates
            from .datasets import build_candidate_dataset
            from .evaluation import evaluate_candidate_recall
            from .training_labels import perturbed_candidates, random_negative_candidates

            eng, U = self.engine(), self.units()
            store = self.combined_store()
            cc = self.cfg["candidates"]
            W = self.grid().width
            rec, summ = [], []
            labs = {}
            for key in sorted(U):
                vu = U[key]
                mm = {k: self._model_map(k, key) for k in ("surface", "subsurface")}
                out = generate_unit_candidates(eng, store, vu, cc, mm, self.seed(), W)
                lab = out["lab"]
                labs[key] = lab
                ev = {f"{k}_rank": v for k, v in out["scores"].items() if k in ("w2", "cnn", "surface", "subsurface")}
                D = build_candidate_dataset(eng, store, vu, lab, out["pools"], ev, cc, self.cfg.get("known_context",
                                                                                                     {}),
                                            self.cfg["labels"]["candidate"])
                f = unit_file(key)
                self.ws.save_table(D["T"].assign(synthetic=False), f"datasets/candidates/{f}.parquet")
                self.ws.save_table(D["L"], f"datasets/candidates/{f}_labels.parquet")
                self.ws.save_npz(f"datasets/candidates/{f}_lab.npz", lab=lab)
                self.ws.save_table(out["dedup"], f"candidates/{f}_provenance.csv")
                aug = cc.get("augmentation") or {}
                if aug.get("enabled", True):
                    rng = np.random.default_rng(self.seed() + 31 * vu.zone)
                    slab, ST = perturbed_candidates(vu.eval_truth(), vu.sysmap, rng, int(aug.get("per_system", 4)),
                                                    float(aug.get("max_shift_px", 4)), float(aug.get("max_rot_deg", 15)))
                    nlab = random_negative_candidates(vu.unit.scored_mask(), vu.eval_truth() | vu.known, rng,
                                                      int(aug.get("n_negatives", 200)))
                    for tag, off, L_ in (("syn_pos", 1_000_000, slab), ("syn_neg", 2_000_000, nlab)):
                        if L_.max() > 0:
                            Ds = build_candidate_dataset(eng, store, vu, L_, out["pools"], ev, cc,
                                                         self.cfg.get("known_context", {}),
                                                         self.cfg["labels"]["candidate"])
                            # synthetic label ids never collide with real ones (nested overrides merge on label;
                            # synthetic rows therefore get NaN nested context instead of non-nested context)
                            Ts, Ls = Ds["T"].copy(), Ds["L"].copy()
                            if len(Ts):
                                Ts["label"] += off
                            if len(Ls):
                                Ls["label"] += off
                            self.ws.save_table(Ts.assign(synthetic=True, synthetic_kind=tag),
                                               f"datasets/candidates/{f}_{tag}.parquet")
                            self.ws.save_table(Ls, f"datasets/candidates/{f}_{tag}_labels.parquet")
                            self.ws.save_npz(f"datasets/candidates/{f}_{tag}_lab.npz", lab=L_)
                r = evaluate_candidate_recall(vu, lab, out["pools"])
                if len(r):
                    rec.append(r)
                summ.append({"unit": key, "n_candidates": int(lab.max()), "pools": ",".join(out["pools"]),
                             "provenance": json.dumps(out["provenance"])})
                eng.release(vu.base)
                self.say(f"  candidates {key}: {int(lab.max())} from {list(out['pools'])}")
            for key, vu in U.items():
                if vu.scenario not in ("V1", "V3"):
                    continue
                for other in U.values():
                    if other.scenario == vu.scenario and other.zone != vu.zone:
                        K = nested_known_table(eng, vu, other.zone, labs[key], self.cfg.get("known_context", {}))
                        self.ws.save_table(K, f"datasets/candidates/nested_known/{vu.scenario}_{other.zone}__"
                                              f"{vu.zone}.parquet")
            R = pd.concat(rec, ignore_index=True) if rec else pd.DataFrame()
            self.ws.save_table(R, "evaluation/recall/candidate_recall.csv")
            self.ws.save_table(pd.DataFrame(summ), "candidates/candidate_summary.csv")
            if self.figures_on() and len(R):
                from . import visualization as V

                V.plot_candidate_recall(R, self.figs("model_comparison"))
            self.push(["candidates", "datasets/candidates", "evaluation", "figures"])
            return COMPLETE, {"note": f"{sum(s['n_candidates'] for s in summ)} candidates in {len(summ)} units"}
        parts = [stage_config(self.cfg, "generate_candidates"), self.fp_of("datasets"),
                 self._current_fp("models/surface_registry.csv"), self._current_fp("models/subsurface_registry.csv")]
        return self._run("generate_candidates", parts, fn)

    def _load_candidates(self) -> pd.DataFrame:
        d = self.root / "datasets" / "candidates"
        parts = []
        for p in sorted(d.glob("*.parquet")):
            if p.stem.endswith("_labels") or p.parent.name == "nested_known":
                continue
            lp = p.with_name(p.stem + "_labels.parquet")
            T = pd.read_parquet(p)
            if not len(T) or not lp.exists():
                continue
            L = pd.read_parquet(lp)
            parts.append(T.merge(L[["unit", "label", "target", "cls", "label_conf", "near_share", "system_id"]],
                                 on=["unit", "label"], how="left").assign(source_file=p.stem))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def train_candidate_ranker(self) -> dict:
        def fn():
            from ..g06.datasets import unit_file
            from ..g06.models_segment import segment_to_raster
            from ..m7.ranking import _zone_mask
            from .crossfit import crossfit_candidate_ranker
            from .models_candidates import candidate_feature_columns, estimate_candidate_utility
            from .representation import candidate_portfolio_selection, exact_k_mask

            D = self._load_candidates()
            if not len(D):
                return FAILED, {"error": "no candidate dataset (GENERATE_CANDIDATES first)"}
            eng, U = self.engine(), self.units()
            rc, rep = self.cfg["ranker"], self.cfg["representation"]
            zm = {k: _zone_mask(U[k].unit, eng.shape, float(self.cfg["validation"].get("segment_buffer_km", 5.0)))
                  for k in D.unit.unique() if k in U}
            labs = {}
            for k in D.unit.unique():
                p = self.root / "datasets" / "candidates" / f"{unit_file(k)}_lab.npz"
                if p.exists():
                    with np.load(p) as z:
                        labs[k] = z["lab"]

            def unit_columns(key, outer):
                if outer is None:
                    return None
                sc, z = key.split(":")
                p = self.root / "datasets" / "candidates" / "nested_known" / f"{sc}_{outer}__{z}.parquet"
                if not p.exists():
                    return None
                K = pd.read_parquet(p).drop(columns=["unit", "outer"])
                for which in ("surface", "subsurface"):
                    if f"ev_{which}_rank_mean" not in D:
                        continue
                    m = self._model_map(which, key, outer)
                    if m is None:
                        return None
                    from ..gems3.representation import rank01

                    r = rank01(np.nan_to_num(m, nan=0.0), U[key].unit.ev)
                    lab = labs[key]
                    rr, cc = np.nonzero(lab)
                    g = pd.DataFrame({"label": lab[rr, cc], "v": r[rr, cc]}).groupby("label").v.agg(["mean", "max"])
                    K = K.merge(g.rename(columns={"mean": f"ev_{which}_rank_mean", "max": f"ev_{which}_rank_max"})
                                .reset_index(), on="label", how="left")
                return K

            reg, maps_written = {}, {}
            for eid, ec in rc["experiments"].items():
                w2a = bool(ec.get("w2_aware", False))
                params = self.cfg["models"][ec.get("params", "candidate_lightgbm")]
                inputs = {"w2": w2a, "cnn": w2a and any(c.startswith("ev_cnn") for c in D.columns),
                          "surface": "ev_surface_rank_mean" in D, "subsurface": "ev_subsurface_rank_mean" in D}
                scores, info = crossfit_candidate_ranker(eid, D, lambda F, w=w2a: candidate_feature_columns(F, w),
                                                         ec.get("model", "lightgbm"), params, zm, self.seed(),
                                                         unit_columns if ec.get("nested", True) else None, inputs,
                                                         bool(rc.get("require_strict", False)), self.say)
                efp = PS.stable_hash(eid, ec, params, self.fp_of("generate_candidates"), n=10)
                d = f"predictions/candidates/{eid}/{efp}"
                self.ws.save_table(scores, f"{d}/scores.parquet")
                real = D[~D.synthetic.astype(bool)]
                cal_rows = []
                for key, g in scores.groupby("unit"):
                    sc_ = key.split(":")[0]
                    other = scores[(scores.unit.str.startswith(sc_ + ":")) & (scores.unit != key)].merge(
                        real[["unit", "label", "target"]], on=["unit", "label"])
                    p, cinfo = CF.calibrate_candidate_confidence(other.score.to_numpy(), other.target.to_numpy(),
                                                              g.score.to_numpy())
                    G = g.assign(p=p)
                    cal_rows.append(G.assign(calibration=cinfo["method"]))
                    vu = U[key]
                    lab = labs.get(key)
                    if lab is None:
                        continue
                    base_px = eng.w2_rank(vu.base) if w2a else self._model_map("subsurface", key)
                    if base_px is None:
                        base_px = eng.w2_rank(vu.base)
                    smap = segment_to_raster(lab, G[["label", "score"]], base_px, vu.unit.ev,
                                             float(rc.get("lambda", 0.5)))
                    el = vu.unit.scored_mask() & eng.template_valid()[vu.sl]
                    off = (vu.sl[0].start, vu.sl[1].start)
                    w2r = eng.w2_rank(vu.base)
                    base, _ = exact_k_mask(w2r, el, float(rep["primary_frac"]) * float(rep["portfolio"][
                        "w2_base_share"]), offset=off, width_full=self.grid().width, seed=self.seed())
                    Ut = estimate_candidate_utility(lab, G, base, el, float(rep["portfolio"].get("kappa", 0.25)))
                    q, prep = candidate_portfolio_selection(w2r, el, float(rep["primary_frac"]), lab, Ut,
                                                            float(rep["portfolio"]["w2_base_share"]),
                                                            float(rep["portfolio"].get("min_utility", 0.0)), off,
                                                            self.grid().width, self.seed())
                    chosen_labels = np.unique(lab[q & (lab > 0)])                 # candidates the portfolio kept
                    width = CF.estimate_localization_uncertainty(smap, lab)
                    soft = CF.build_confidence_corridor(lab, chosen_labels, G, width, vu.unit.ev,
                                                     float(rep["corridor"].get("halo_value", 0.5)),
                                                     float(rep["corridor"].get("v_min", 0.4)))
                    soft = np.maximum(soft, np.where(q & (lab == 0), 1.0, 0.0)).astype(np.float32)
                    for name, arr in ((f"{eid}:map", smap), (f"G07-D2:{eid}", q.astype(np.float32)),
                                      (f"G07-E1:{eid}", soft)):
                        p_ = self.root / d / name.split(":")[0] / f"{unit_file(key)}.npy"
                        p_.parent.mkdir(parents=True, exist_ok=True)
                        PS.atomic_save(p_, lambda t, a=arr: np.save(t, a))
                        maps_written.setdefault(name, {})[key] = str(p_.relative_to(self.root))
                    self.ws.save_json(prep, f"{d}/portfolio/{unit_file(key)}.json")
                self.ws.save_table(pd.concat(cal_rows, ignore_index=True) if cal_rows else pd.DataFrame(),
                                   f"{d}/calibrated_scores.parquet")
                reg[eid] = {"fingerprint": efp, "strictness": info["strictness"], "reason": info["strictness_reason"],
                            "w2_aware": w2a, "dir": d, "units": {k: {kk: vv for kk, vv in v.items() if kk != "importance"}
                                                                for k, v in info["units"].items()},
                            "maps": {k: v for k, v in maps_written.items() if eid in k}}
                imp = pd.DataFrame([{"unit": k, "feature": f, "importance": v} for k, u in info["units"].items()
                                    for f, v in (u.get("importance") or {}).items()])
                self.ws.save_table(imp, f"{d}/importance.csv")
                self.push(["predictions", "models"])
            self.ws.save_json(reg, "models/candidate_registry.json")
            strict = {k: v["strictness"] for k, v in reg.items()}
            return COMPLETE, {"note": f"{len(reg)} candidate models; strictness {strict}"}
        parts = [stage_config(self.cfg, "train_candidate_ranker"), self.fp_of("generate_candidates")]
        return self._run("train_candidate_ranker", parts, fn)

    # =========================================================================== evaluation
    def _score_maps(self) -> tuple[dict, dict, dict]:
        """({model: {unit: map}}, {model: strictness}, {model: policy kind}); kind 'score' -> exact-k policies,
        'selection' -> evaluated as produced (raw)."""
        from ..g06.models_pixel import load_oof

        U = self.units()
        maps, strict, kind = {"W2": self.w2_maps()}, {"W2": "REFERENCE"}, {"W2": "score"}
        for rel in ("models/subsurface_registry.csv", "models/surface_registry.csv",
                    "models/surface_unet_registry.csv"):
            for exp, fp in self._current_fp(rel).items():
                maps[exp] = {k: m for k in U if (m := load_oof(self.root, exp, k, fp)) is not None}
                strict[exp], kind[exp] = "STRICT_ISOLATED", "score"
        reg = self.ws.read_json("models/candidate_registry.json") or {}
        for eid, v in reg.items():
            for name, by in (v.get("maps") or {}).items():
                mname = eid if name == f"{eid}:map" else name
                maps[mname] = {k: np.load(self.root / p) for k, p in by.items()}
                strict[mname] = v["strictness"]
                kind[mname] = "score" if name == f"{eid}:map" else "selection"
        return {m: d for m, d in maps.items() if d}, strict, kind

    def evaluate(self) -> dict:
        def fn():
            from . import evaluation as EV

            eng, U = self.engine(), self.units()
            ec, rep = self.cfg["evaluation"], self.cfg["representation"]
            maps, strict, kind = self._score_maps()
            templates = {k: eng.template_valid()[vu.sl] for k, vu in U.items()}
            score_maps = {m: d for m, d in maps.items() if kind[m] == "score"}
            sel_maps = {m: d for m, d in maps.items() if kind[m] == "selection"}
            pol = rep["policies"]
            W = self.grid().width
            allr, alls = [], []
            for sc in self.cfg["validation"]["scenarios"]:
                R, S = EV.evaluate_scenario(U, score_maps, sc, pol, ec["fracs"], templates, self.cfg, strict,
                                            secondary=None, width_full=W, progress=self.say)
                R2, S2 = EV.evaluate_scenario(U, sel_maps, sc, {"as_selected": {"select": "raw", "post": []}},
                                              [float(rep["primary_frac"])], templates, self.cfg, strict,
                                              width_full=W)
                allr += [R, R2]
                alls += [S, S2]
            Rw = pd.concat([r for r in allr if len(r)], ignore_index=True)
            Sw = pd.concat([s for s in alls if len(s)], ignore_index=True)
            self.ws.save_table(Rw, "evaluation/metrics/metrics_rows.csv")
            SUM = EV.aggregate_components(Rw)
            SUM["strictness"] = SUM.model.map(strict)
            self.ws.save_table(SUM, "evaluation/metrics/metrics.csv")
            self.ws.save_table(EV.per_zone(Rw), "evaluation/metrics/per_zone.csv")
            self.ws.save_table(SUM[SUM.scenario == "V3"], "evaluation/geographic/v3_summary.csv")
            EB = EV.exact_budget_ok(Rw)
            self.ws.save_table(EB, "evaluation/metrics/exact_budget.csv")
            SR = EV.system_recovery_report(Sw)
            self.ws.save_table(SR, "evaluation/metrics/system_recovery.csv")
            prim = {"rep": "top", "frac": float(rep["primary_frac"])}
            # selection maps are evaluated 'as_selected' at the primary fraction: align them with the W2 top row
            SUMc = SUM.copy()
            SUMc.loc[SUMc.rep == "as_selected", "rep"] = "top"
            CMP = EV.compare_to_w2(SUMc)
            self.ws.save_table(CMP, "evaluation/metrics/model_comparison.csv")
            SRc = SR.copy()
            if len(SRc):
                SRc.loc[SRc.rep == "as_selected", "rep"] = "top"
            leak = self.ws.load_table("splits/isolation_audit.csv")
            leak_ok = bool(len(leak)) and bool((leak.status == "PASS").all())
            PD = EV.evaluate_promotion(CMP, SRc, strict, self.cfg["promotion"], prim, EB, leak_ok)
            self.ws.save_table(PD, "evaluation/promotion/promotion_decisions.csv")
            self.ws.save_table(self._ablations(SUM), "evaluation/ablations/ablations.csv")
            self.ws.save_table(self._complementarity(Sw), "evaluation/metrics/complementarity.csv")
            sel = [EV.nested_policy_selection(Rw[Rw.rep != "as_selected"], m, s) for m in score_maps
                   for s in sorted(set(Rw.scenario))]
            self.ws.save_table(pd.concat([s for s in sel if len(s)], ignore_index=True) if any(len(s) for s in sel)
                               else pd.DataFrame(), "evaluation/metrics/representation_selection.csv")
            from .datasets import build_meta_dataset

            prov = {m: {k: {"strictness": strict.get(m)} for k in d} for m, d in maps.items()}
            MP, MR = build_meta_dataset(U, score_maps, prov, eng.labels(), self.cfg.get("meta_rows", {}), self.seed())
            self.ws.save_table(MP, "datasets/meta/provenance.csv")
            self.ws.save_table(MR, "datasets/meta/rank_rows.parquet")
            best = PD[PD.status == "PROMOTE_CANDIDATE"].sort_values("v1_gain", ascending=False) if len(PD) else PD
            self.ws.save_json({"primary": prim, "promote": list(best.model) if len(best) else [], "strictness": strict,
                               "statuses": dict(zip(PD.model, PD.status)) if len(PD) else {},
                               "exact_budget": EB.to_dict("records")}, "evaluation/promotion/decision.json")
            if self.figures_on():
                self._evaluation_figures(Rw, SUM, SR)
            self.push(["evaluation", "figures", "datasets/meta"])
            return COMPLETE, {"note": f"{len(maps)} models; promote {list(best.model) if len(best) else 'none'}"}
        parts = [stage_config(self.cfg, "evaluate"), self.fp_of("train_subsurface"), self.fp_of("train_surface_classic"),
                 self.fp_of("train_surface_unet"), self.fp_of("train_candidate_ranker"), self.fp_of("g06_restore")]
        return self._run("evaluate", parts, fn)

    def _ablations(self, SUM) -> pd.DataFrame:
        if not len(SUM):
            return pd.DataFrame()
        f = float(self.cfg["representation"]["primary_frac"])
        S = SUM[(SUM.frac == f) & SUM.rep.isin(["top", "as_selected"])]
        groups = {"surface: A1 -> A2": ["G07-A1", "G07-A2"], "subsurface: B0 -> B1 -> B2": ["G07-B0", "G07-B1",
                                                                                             "G07-B2"]}
        rows = []
        for name, ms in groups.items():
            for m in ms:
                for sc in ("V1", "V3"):
                    r = S[(S.model == m) & (S.scenario == sc)]
                    w = S[(S.model == "W2") & (S.scenario == sc)]
                    if len(r):
                        rows.append({"ablation": name, "model": m, "scenario": sc, "dti": float(r.dti.iloc[0]),
                                     "worst_zone_dti": float(r.worst_zone_dti.iloc[0]),
                                     "dti_w2": float(w.dti.iloc[0]) if len(w) else np.nan})
        return pd.DataFrame(rows)

    def _complementarity(self, Sw) -> pd.DataFrame:
        """Systems recovered by each model but not by W2 at the primary exact budget (and the union)."""
        if not len(Sw):
            return pd.DataFrame()
        f = float(self.cfg["representation"]["primary_frac"])
        S = Sw[(Sw.frac == f) & Sw.rep.isin(["top", "as_selected"]) & Sw.recovered]
        w2 = {(r.unit, r.system_id) for r in S[S.model == "W2"].itertuples()}
        rows = []
        allnew = set()
        for m, g in S.groupby("model"):
            if m == "W2":
                continue
            rec = {(r.unit, r.system_id) for r in g.itertuples()}
            allnew |= rec - w2
            rows.append({"model": m, "n_recovered": len(rec), "n_new_vs_w2": len(rec - w2), "n_lost_vs_w2": len(w2 - rec),
                         "new": ";".join(f"{u}#{s}" for u, s in sorted(rec - w2))})
        rows.append({"model": "UNION(all non-W2)", "n_recovered": np.nan, "n_new_vs_w2": len(allnew)})
        return pd.DataFrame(rows)

    def _evaluation_figures(self, Rw, SUM, SR) -> None:
        from . import visualization as V

        f = float(self.cfg["representation"]["primary_frac"])
        models = ["W2"] + [m for m in sorted(SUM.model.unique()) if m != "W2"]
        S2 = SUM.copy()
        S2.loc[S2.rep == "as_selected", "rep"] = "top"
        V.plot_model_comparison(S2, self.figs("model_comparison"), "model_comparison", "top", f, models[:9])
        for sc in sorted(set(Rw.scenario)):
            V.plot_dti_frontier(Rw[Rw.rep == "top"], self.figs("dti_frontiers"), f"dti_frontier_{sc}", sc, "top",
                                models[:9])
        from ..g06.evaluation import per_zone

        V.plot_geographic_robustness(per_zone(Rw[Rw.rep == "top"]), self.figs("model_comparison"),
                                     "geographic_robustness", "top", f, models[:9])
        V.plot_exact_budget(Rw, self.figs("model_comparison"))
        self._recovery_atlas(SR, f)

    def _recovery_atlas(self, SR: pd.DataFrame, f: float, max_cases: int = 12) -> None:
        """Systems newly recovered / lost vs W2 by the best non-W2 model (V1, primary budget; truth post hoc)."""
        from ..g06.visualization import plot_failure_cases
        from .representation import exact_k_mask

        if not len(SR):
            return
        C = SR[(SR.model != "W2") & (SR.frac == f) & SR.rep.isin(["top", "as_selected"]) &
               SR.unit.str.startswith("V1")]
        if not len(C):
            return
        best = C.groupby("model").apply(lambda g: g.n_new.sum() - g.n_lost.sum()).idxmax()
        maps, _, kind = self._score_maps()
        U, eng = self.units(), self.engine()
        for which in ("new", "lost"):
            cases = []
            for r in C[C.model == best].itertuples():
                ids = [int(x) for x in str(getattr(r, f"{which}_ids")).split(";") if x not in ("", "nan")]
                vu = U.get(r.unit)
                if vu is None or r.unit not in maps.get(best, {}) or r.unit not in maps["W2"]:
                    continue
                src = maps[best][r.unit] if which == "new" else maps["W2"][r.unit]
                el = vu.unit.scored_mask() & eng.template_valid()[vu.sl]
                sel = src > 0 if kind.get(best) == "selection" and which == "new" else \
                    exact_k_mask(src, el, f, offset=(vu.sl[0].start, vu.sl[1].start))[0]
                for sid in ids[:max_cases]:
                    m = vu.sysmap == sid
                    if not m.any():
                        continue
                    rr, cc = np.nonzero(m)
                    sl = (slice(max(rr.min() - 30, 0), rr.max() + 30), slice(max(cc.min() - 30, 0), cc.max() + 30))
                    cases.append({"label": f"{r.unit} system {sid}", "unit": r.unit, "system_id": sid,
                                  "score": np.asarray(src, float)[sl], "truth": m[sl], "selected": sel[sl]})
            plot_failure_cases(cases[:max_cases], self.figs("recovery_atlas"),
                               f"systems_{which}_{best.replace(':', '_')}",
                               f"Systems {which} vs W2 - {best if which == 'new' else 'W2'} (black: truth, post hoc)")

    # =========================================================================== full region / export
    def reconstruct_full_region(self) -> dict:
        return {"full_inference": self.full_inference(), "export": self.export_submission()}

    def full_inference(self) -> dict:
        def fn():
            from ..g06.raster_io import write_feature_raster
            from ..gems3.submission import eligible_mask
            from .inference import construct_final_prediction, read_w2_full, verify_export_parity
            from .inventory import inspect_w2_artifacts

            w2a = inspect_w2_artifacts(self.cfg.get("w2") or {}, [self.local_root, self.drive_root])
            if not w2a["full_score"]:
                return BLOCKED_INPUT, {"note": w2a["full_region_status"]}
            eng = self.engine()
            data = eng.data
            el = eligible_mask(data, 0, "eligible")
            w2 = read_w2_full(w2a["full_score"], eng.valid())
            dec = self.ws.read_json("evaluation/promotion/decision.json") or {}
            chosen = (dec.get("promote") or [None])[0]
            policy = "portfolio" if chosen and chosen.startswith("G07-D2") else "w2_exact"
            frac = float(self.cfg["representation"]["primary_frac"])
            lab = util = None
            man = {"policy": policy, "chosen_model": chosen, "w2_full_score": w2a["full_score"]}
            if policy == "portfolio":
                lab, util, extra = self._full_candidates(chosen, w2, el)
                man.update(extra)
            q, rep = construct_final_prediction(w2, el, frac, policy, lab, util,
                                                float(self.cfg["representation"]["portfolio"]["w2_base_share"]),
                                                self.seed())
            out = self.root / "predictions" / "full_region" / f"final_{policy}.tif"
            write_feature_raster(out, {"prediction": q}, self.grid(), {"ROLE": "RESEARCH_NOT_A_SUBMISSION",
                                                                      "POLICY": policy, "MODEL": str(chosen)})
            par = verify_export_parity(q, el, frac, eng.labels(), model_parity=man.get("model_parity"))
            man.update({"raster": str(out.relative_to(self.root)), "selection": rep, "parity": par})
            self.ws.save_json(man, "predictions/full_region/full_region_manifest.json")
            self.push(["predictions"])
            return COMPLETE, {"note": f"{policy}: parity {par['status']}"}
        return self._run("full_inference", [stage_config(self.cfg, "full_inference"), self.fp_of("evaluate")], fn)

    def _full_candidates(self, chosen: str, w2: np.ndarray, el: np.ndarray):
        """Full-region candidate inventory scored by a candidate model refitted on every unit (FULL)."""
        from .candidate_generation import unit_scores  # noqa: F401  (documented reuse; full scores built below)
        from .confidence import calibrate_candidate_confidence
        from .crossfit import system_balanced_weights
        from .inference import predict_candidate_inventory_full
        from .models_candidates import candidate_feature_columns, estimate_candidate_utility, train_candidate_classifier
        from .representation import exact_k_mask

        eid = chosen.split(":", 1)[1]
        ec = self.cfg["ranker"]["experiments"][eid]
        D = self._load_candidates()
        D = D[D.target.notna()]
        cols = candidate_feature_columns(D, bool(ec.get("w2_aware", False)))
        y = D.target.to_numpy(int)
        est = train_candidate_classifier(ec.get("model", "lightgbm"), D[cols].to_numpy(np.float32), y,
                                         system_balanced_weights(D, y),
                                         self.cfg["models"][ec.get("params", "candidate_lightgbm")], self.seed())
        reg = self.ws.read_json("models/candidate_registry.json") or {}
        cal = self.ws.load_table(f"{reg[eid]['dir']}/calibrated_scores.parquet")
        real = D[~D.synthetic.astype(bool)][["unit", "label", "target"]]
        C = cal.merge(real, on=["unit", "label"])

        def calibrator(s):
            return calibrate_candidate_confidence(C.score.to_numpy(), C.target.to_numpy(), s)[0]

        st = self.combined_store()
        eng = self.engine()
        scores = {"w2": w2}
        lab, T = predict_candidate_inventory_full(st, scores, el, eng.labels(), eng.valid(), self.cfg["candidates"],
                                                  self.cfg.get("known_context", {}), est, cols, self.seed(), calibrator)
        base, _ = exact_k_mask(w2, el, float(self.cfg["representation"]["primary_frac"]) *
                               float(self.cfg["representation"]["portfolio"]["w2_base_share"]), seed=self.seed())
        U = estimate_candidate_utility(lab, T, base, el, float(self.cfg["representation"]["portfolio"].get("kappa",
                                                                                                            0.25)))
        self.ws.save_table(T, "predictions/full_region/full_candidates.parquet")
        return lab, U, {"n_full_candidates": int(lab.max()), "candidate_model": eid,
                        "note": "full-region candidate pools: W2 only (surface / subsurface FULL rasters are added "
                                "when their research rasters exist)"}

    def export_submission(self) -> dict:
        def fn():
            from .inference import write_submission_candidate

            dec = self.ws.read_json("evaluation/promotion/decision.json") or {}
            man = self.ws.read_json("predictions/full_region/full_region_manifest.json") or {}
            chosen = man.get("chosen_model")
            uses_dem = bool(chosen) and any(x in str(chosen) for x in ("A1", "A2", "C1")) or \
                bool(self.cfg["candidates"].get("model_maps", {}).get("surface") and chosen)
            gates = {"EXPORT_SUBMISSION": bool(self.run_flags.get("EXPORT_SUBMISSION")),
                     "auto_submit_disabled": not bool(self.nb.get("auto_submit")),
                     "promoted_strict_model": bool(chosen) and dec.get("statuses", {}).get(chosen) == "PROMOTE_CANDIDATE",
                     "full_region_parity": (man.get("parity") or {}).get("status") == "PASS",
                     "dem_competition_eligibility": (not uses_dem) or bool(self.cfg["dem"].get(
                         "competition_eligibility_verified", False)),
                     "provenance_complete": bool(man.get("raster")) and bool(self.head and self.head != "unknown")}
            res = {"gates": gates, "chosen_model": chosen, "uploaded": False,
                   "note": "W2 is retained when no model passes every gate"}
            if all(gates.values()):
                from ..g06.raster_io import read_band_full

                q = read_band_full(self.root / man["raster"], 1)
                res.update(write_submission_candidate(np.nan_to_num(q), self.engine().data, self.root / "submissions" /
                                                      f"g07_{str(chosen).replace(':', '_')}.tif",
                                                      {"run_id": self.run_id, "commit": self.head, "model": chosen},
                                                      gates))
            self.ws.save_json(res, "evaluation/promotion/submission_eligibility.json")
            self.push(["evaluation", "submissions"])
            return COMPLETE, {"note": res.get("status", "NOT_ELIGIBLE: " + ",".join(k for k, v in gates.items()
                                                                                     if not v))}
        return self._run("export", [stage_config(self.cfg, "export"), self.fp_of("full_inference")], fn, force=True)

    # =========================================================================== reports
    def build_report(self) -> dict:
        def fn():
            write_reports(self)
            PS.write_artifact_inventory(self.root)
            self.push(["reports", "manifests"])
            return COMPLETE, {"note": "reports written from persisted tables"}
        return self._run("report", [time.strftime("%Y-%m-%dT%H:%M:%S")], fn, force=True)

    def run(self) -> dict:
        self.run_tests()
        self.preflight()
        self.inspect_dem()
        self.assess_dem_feasibility()
        self.prepare_dem_features()
        self.restore_g06()
        self.prepare_training_data()
        self.benchmark_subsurface_models()
        self.benchmark_surface_models()
        self.train_surface_unet_stage()
        self.generate_candidates()
        self.train_candidate_ranker()
        self.evaluate()
        self.reconstruct_full_region()
        self.build_report()
        self.final_message()
        return dict(self.status)

    def stage_table(self) -> pd.DataFrame:
        rows = []
        for s in STAGES:
            m = PS.load_cached_stage(self.root, s) or {}
            rows.append({"stage": s, "session_status": self.status.get(s), "manifest_status": m.get("status"),
                         "fingerprint": str(m.get("fingerprint") or "")[:10], "seconds": m.get("seconds"),
                         "note": str(m.get("note") or m.get("reason") or m.get("error") or "")[:200]})
        return pd.DataFrame(rows)

    def final_message(self) -> None:
        print("\n" + "=" * 100)
        print(f"G07 {G07_CODE_REV} run {self.run_id} | commit {self.head}")
        for s in STAGES:
            print(f"  {s:<24} {self.status.get(s)}")
        print(f"local: {self.root}\nDrive: {self.ws.drive_path()}")
        print("Nothing was uploaded. Offline proxy DTI is not a leaderboard score; the 0.40 target is not claimed.")
        print("=" * 100)


# =============================================================================== reports
def write_reports(P: G07Pipeline) -> None:
    from ..g06.pipeline import _md

    ws = P.ws
    hdr = f"Run `{P.run_id}` | code `{G07_CODE_REV}` | commit `{P.head}` | {time.strftime('%Y-%m-%d %H:%M')}"
    hist = P.cfg.get("historical") or {}
    SUM = ws.load_table("evaluation/metrics/metrics.csv")
    CMP = ws.load_table("evaluation/metrics/model_comparison.csv")
    PD = ws.load_table("evaluation/promotion/promotion_decisions.csv")
    EB = ws.load_table("evaluation/metrics/exact_budget.csv")
    CO = ws.load_table("evaluation/metrics/complementarity.csv")
    AB = ws.load_table("evaluation/ablations/ablations.csv")
    REC = ws.load_table("evaluation/recall/candidate_recall.csv")
    f = float(P.cfg["representation"]["primary_frac"])
    regs = {k: ws.load_table(f"models/{k}_registry.csv") for k in ("subsurface", "surface", "surface_unet")}
    lines = ["# G07 training results", "", hdr, ""]
    for k, R in regs.items():
        lines += [f"## {k}", "", _md(R, 40, ["experiment", "fit_id", "status", "kind", "target", "n_train",
                                              "best_iteration", "seconds"])]
    cr = ws.read_json("models/candidate_registry.json") or {}
    lines += ["## Candidate models", ""] + [f"* {k}: {v.get('strictness')} - {v.get('reason')}" for k, v in cr.items()]
    ws.save_text("\n".join(lines) + "\n", "reports/TRAINING_RESULTS.md")
    lines = ["# G07 complementarity", "", hdr, "", "Systems recovered at the primary exact budget by each model but "
             "not by W2 (V1 / V3 / V2 units; held-out truth, diagnostic).", "", _md(CO, 40),
             "## Candidate recall (geometric, before ranking)", "", _md(REC, 40, ["unit", "source", "n_systems",
                                                                                   "n_covered", "recall", "area_frac"])]
    ws.save_text("\n".join(lines) + "\n", "reports/COMPLEMENTARITY.md")
    lines = ["# G07 DTI evaluation", "", hdr, "",
             f"Matched policy: exact-k top {f:.3%} of the eligible domain for every score map (deterministic label-free "
             "ties); portfolio / corridor selections evaluated as produced at the same budget. Pooled DTI from summed "
             "components; worst zone with every pooled number.", "",
             "Historical references (other protocols - not comparable): " + ", ".join(f"{k} = {v}" for k, v in
                                                                                      hist.items()), ""]
    if len(SUM):
        s = SUM[(SUM.frac == f) & SUM.rep.isin(["top", "as_selected"])]
        lines += ["## Pooled DTI", "", _md(s.sort_values(["scenario", "dti"], ascending=[True, False]), 60,
                                           ["scenario", "model", "rep", "strictness", "dti", "worst_zone_dti",
                                            "tp_w", "fp_w", "fn_w", "area_frac_eligible"]),
                  "## vs W2", "", _md(CMP[CMP.frac == f] if len(CMP) else CMP, 40),
                  "## Exact budget audit", "", _md(EB), "## Ablations", "", _md(AB, 40),
                  "## Promotion", "", _md(PD, 30)]
    else:
        lines += ["EVALUATE has not run in this run directory."]
    lines += ["", "**The 0.40 leaderboard target is never inferred from these offline proxy numbers.**"]
    ws.save_text("\n".join(lines) + "\n", "reports/DTI_EVALUATION.md")
    el = ws.read_json("evaluation/promotion/submission_eligibility.json") or {}
    lines = ["# Submission eligibility", "", hdr, "", f"Gates: `{json.dumps(el.get('gates', {}))}`", "",
             f"Status: **{el.get('status', 'NOT EVALUATED / NOT ELIGIBLE')}** (uploaded: False)", "",
             "A file is written only when every gate passes; it is READY_FOR_MANUAL_UPLOAD and never uploaded."]
    ws.save_text("\n".join(lines) + "\n", "reports/SUBMISSION_ELIGIBILITY.md")
    dec = (ws.read_json("dem_catalog/feasibility_decision.json") or {}).get("decision")
    nxt = ["# Next experiment", "", hdr, "", f"* DEM feasibility: **{dec}**."]
    if dec in (None, "BLOCKED_INPUT"):
        nxt.append("* Provide DEM tiles (docs/g07/DEM_DATA_CONTRACT.md) or continue on the geology path "
                   "(BUILD_DATASETS + TRAIN_SUBSURFACE + EVALUATE).")
    elif dec == "GO":
        nxt.append("* GO: DEM_FEATURES -> BUILD_DATASETS -> TRAIN_SURFACE_CLASSIC (A1 / A2) -> EVALUATE; U-Net only "
                   "if A1 / A2 add systems vs W2.")
    else:
        nxt.append("* NO_GO: keep DEM optional; run B0 / B1 / B2, then candidates + ranker (D1 / D2).")
    if len(PD):
        best = PD.sort_values("v1_gain", ascending=False).iloc[0]
        nxt.append(f"* Best V1 gain vs W2: {best.model} {best.v1_gain:+.4f} ({best.status}; {best.reasons}).")
    nxt.append("* Rule-based readings of persisted tables - not findings.")
    ws.save_text("\n".join(nxt) + "\n", "reports/NEXT_EXPERIMENT.md")
    if not ws.exists("reports/DEM_FEASIBILITY.md"):
        ws.save_text("# DEM feasibility\n\nNot run in this run directory.\n", "reports/DEM_FEASIBILITY.md")
