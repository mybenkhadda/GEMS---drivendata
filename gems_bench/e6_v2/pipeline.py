"""E6V2Pipeline - the one orchestrator. Every method is restartable: finished artifacts with a matching
fingerprint are reused, stale ones are recomputed (logged), missing upstream artifacts raise ``MissingArtifact``
with their recovery procedure. Nothing outside ``<drive_root>/<namespace>/`` is written; nothing outside it except
the raw inputs is read.
"""
from __future__ import annotations

import logging
import sys

import numpy as np

from . import artifacts, figures
from .cache import (MissingArtifact, atomic_write_geotiff, atomic_write_json, load_npy, manifest_status, read_json,
                    write_manifest)
from .config import CODE_VERSION, module_hash, stable_hash
from .contracts import schema_hash
from .paths import E6V2Paths
from .spatial_splits import DISC_UNITS, GEO_UNITS

log = logging.getLogger("e6v2")
RUNTIME_ONLY = ("infer_batch", "infer_overlap", "max_stage_minutes")


def setup_logging(paths: E6V2Paths):
    root = logging.getLogger("e6v2")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    paths.reports.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(paths.reports / "pipeline.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)


class E6V2Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.paths = E6V2Paths.from_config(cfg)
        self._data = None
        self._trainers = {}
        if self.paths.drive_root.exists():
            self.paths.ensure()
            setup_logging(self.paths)
            artifacts.write_graph(self.paths, cfg)

    # ------------------------------------------------------------------ 02 / 03
    def inventory(self) -> dict:
        from .inventory import build_inventory

        inv = build_inventory(self.paths, self.cfg, with_dem=True)
        return {"raw_fingerprint": inv["fingerprint"], "issues": inv["issues"], "rows": len(inv["rows"]),
                "dem_tiles": int(len(inv["dem_tiles"]))}

    def preflight(self, raise_on_block: bool = True) -> dict:
        from .preflight import run_preflight

        return run_preflight(self.paths, self.cfg, raise_on_block=raise_on_block)

    def _inv(self) -> dict:
        from .inventory import load_inventory

        ok, why = artifacts.v_preflight(self.paths)
        if not ok:
            raise MissingArtifact(f"preflight is {why} - run section 03 and clear every blocker first")
        return load_inventory(self.paths)

    # ------------------------------------------------------------------ 04
    def prepare_geophysics(self, force: bool = False) -> dict:
        from .geophysics import prepare_base, prepare_potential, prepare_terrain

        inv = self._inv()
        out = {"base": prepare_base(self.paths, self.cfg, inv, force)}
        out["terrain"] = prepare_terrain(self.paths, self.cfg, inv, force)
        out["potential_fields"] = prepare_potential(self.paths, self.cfg, inv, force)
        return {k: v.get("fingerprint") for k, v in out.items()}

    # ------------------------------------------------------------------ 05 / 06
    def prepare_dem(self, force: bool = False, assemble: bool = True) -> dict:
        """Section 05: ``assemble=False`` (mosaic + 2 m features per window); section 06: ``assemble=True``."""
        from .dem_aggregation import prepare_dem
        from .geophysics import load_base

        artifacts.require(self.paths, self.cfg, "base_masks", "dem_source_inventory")
        res = prepare_dem(self.paths, self.cfg, self._inv(), load_base(self.paths), force, assemble)
        if res.get("status") in ("incomplete", "windows_complete"):
            return res
        return {"status": "complete", **res.get("stats", {})}

    # ------------------------------------------------------------------ 07 / 08
    def prepare_datasets(self, include_splits: bool = True, force: bool = False) -> dict:
        from .datasets import prepare_datasets

        artifacts.require(self.paths, self.cfg, "terrain_features", "potential_features", "dem_features_100m")
        man = prepare_datasets(self.paths, self.cfg, force)
        out = {"datasets": man["fingerprint"], "schema_hash": man["schema_hash"],
               "dem_cells_forced_uncovered": man.get("dem_cells_forced_uncovered")}
        if include_splits:
            out["splits"] = self.prepare_splits(force)
        return out

    def prepare_splits(self, force: bool = False) -> dict:
        from .geophysics import load_base
        from .spatial_splits import build_splits

        artifacts.require(self.paths, self.cfg, "base_masks")
        d = self.paths.splits
        man = d / "manifest.json"
        fp = stable_hash([read_json(self.paths.base / "manifest.json")["fingerprint"], self.cfg["splits"],
                          module_hash("spatial_splits")])
        if not force and manifest_status(man, fp, self.paths.root)[0] == "complete":
            return read_json(man)["identity"]
        base = load_base(self.paths)
        res = build_splits(base["valid"], base["labels"], base["grid"].pixel_m, self.cfg, d)
        write_manifest(man, stage="splits", fingerprint=fp, outputs=res["outputs"], root=self.paths.root,
                       extra={"identity": res["identity"], "withheld_km": res["withheld_km"],
                              "total_trace_km": res["total_trace_km"],
                              "inner_cluster_discovery": res["inner_cluster_discovery"]})
        return res["identity"]

    def load_splits(self) -> dict:
        artifacts.require(self.paths, self.cfg, "splits")
        d = self.paths.splits
        with np.load(d / "geo_folds.npz") as g, np.load(d / "discovery_splits.npz") as s:
            out = {"geo_roles": g["roles"], "disc_roles": s["roles"], "disc_inner": s["inner"]}
        out["identity"] = read_json(d / "manifest.json")["identity"]
        return out

    # ------------------------------------------------------------------ 09 - 11
    def device(self):
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def data(self):
        if self._data is None:
            from .datasets import E6Data

            artifacts.require(self.paths, self.cfg, "datasets")
            self._data = E6Data(self.paths, self.cfg)
        return self._data

    def identity_base(self) -> dict:
        data = self.data()
        inv = read_json(self.paths.manifests / "raw_data_inventory.json")
        tr = {k: v for k, v in self.cfg["training"].items() if k not in RUNTIME_ONLY}
        return {"code_version": CODE_VERSION, "schema_hash": schema_hash(self.cfg),
                "raw_fingerprint": inv["raw_fingerprint"], "dataset_fingerprint": data.fingerprint,
                "normalization_hash": data.schema["normalization_hash"], "training_config": stable_hash(tr),
                "model_config": stable_hash(self.cfg["model"]),
                "code_hash": module_hash("model", "encoders", "fusion", "context", "decoder", "heads", "losses",
                                         "trainer", "datasets", "augmentation")}

    def trainer(self, unit: str):
        if unit not in self._trainers:
            from .geophysics import load_base
            from .trainer import UnitTrainer

            self._trainers[unit] = UnitTrainer(unit, self.cfg, self.paths, self.data(), self.load_splits(),
                                               load_base(self.paths), self.identity_base(), self.device())
        return self._trainers[unit]

    def _need_gpu(self):
        if self.device().type != "cuda":
            raise RuntimeError("training needs a CUDA GPU (Runtime > Change runtime type)")

    def train_reference(self, units=GEO_UNITS) -> dict:
        """Stages A + B (section 09)."""
        self._need_gpu()
        return {u: self.trainer(u).run_reference().get("sha256") for u in units}

    def train_e6(self, units=GEO_UNITS) -> dict:
        """Stages C + D + E (section 10)."""
        self._need_gpu()
        return {u: self.trainer(u).run_e6() for u in units}

    def train(self, units=GEO_UNITS + DISC_UNITS) -> dict:
        """All stages for ``units`` (section 11 runs it for the discovery units)."""
        self._need_gpu()
        out = {}
        for u in units:
            tr = self.trainer(u)
            tr.run_reference()
            out[u] = tr.run_e6()
        return out

    # ------------------------------------------------------------------ 12
    def evaluate(self) -> dict:
        from .evaluator import evaluate

        artifacts.require(self.paths, self.cfg, *[f"e6_{u}" for u in GEO_UNITS + DISC_UNITS])
        res = evaluate(self.paths, self.cfg, {u: self.trainer(u) for u in GEO_UNITS + DISC_UNITS})
        self.figures()
        return res

    def figures(self) -> list:
        import pandas as pd

        out, made = self.paths.figures, []
        h = self.paths.metrics / "training_history.csv"
        if h.exists() and h.stat().st_size > 10:
            made += figures.loss_curves(pd.read_csv(h), out)
        cov = self.paths.dem_dir("coverage_100m") / "dem_coverage_100m.npy"
        if cov.exists():
            c = load_npy(cov).astype(np.float32)
            v = load_npy(self.paths.base / "valid.npy")
            made += figures.raster_map(np.where(v, c, np.nan), out, "dem_coverage", "Measured 1 m DEM coverage",
                                       "share of usable 2 m pixels per 100 m cell", "Blues")
        g = self.paths.metrics / "gate_weights.csv"
        if g.exists() and g.stat().st_size > 10:
            made += figures.gate_weights(pd.read_csv(g), out)
        p = self.paths.predictions / "e6_full_probability.tif"
        if p.exists():
            import rasterio

            with rasterio.open(p) as s:
                made += figures.raster_map(s.read(1), out, "prediction_map", "E6 fold-mean fault probability",
                                           "probability", "magma")
        return made

    # ------------------------------------------------------------------ 13
    def infer_full_region(self, force: bool = False) -> dict:
        from .geophysics import load_base
        from .inference import predict, system_fn

        artifacts.require(self.paths, self.cfg, *[f"e6_{u}" for u in GEO_UNITS])
        base = load_base(self.paths)
        prof = base["grid"].profile()
        H, W = base["valid"].shape
        maps, side = [], []
        for k, u in enumerate(GEO_UNITS):
            tr = self.trainer(u)
            sysm, sha = tr.load_system()
            p = self.paths.predictions / f"e6_fold{k}_full.tif"
            js_p = p.with_suffix(".json")
            js = read_json(js_p)
            want = {"unit": u, "checkpoint_sha256": sha, "dataset_fingerprint": tr.data.fingerprint}
            if not force and js and all(js.get(a) == b for a, b in want.items()) and p.exists():
                import rasterio

                with rasterio.open(p) as s:
                    prob = s.read(1)
                log.info("%s: full-region map reused (checkpoint unchanged)", u)
            else:
                prob = predict(system_fn(sysm), tr.data, (0, H, 0, W), self.cfg, tr.device, tr.amp).astype(np.float32)
                prob[~base["valid"]] = np.nan
                atomic_write_geotiff(p, prob, prof, [f"e6_fold{k}_probability"])
                atomic_write_json(js_p, {**want, "shape": [H, W]})
            maps.append(prob)
            side.append(want)
            del sysm
        mean = np.mean(np.stack(maps), axis=0).astype(np.float32)
        mean[~base["valid"]] = np.nan
        out = self.paths.predictions / "e6_full_probability.tif"
        atomic_write_geotiff(out, mean, prof, ["e6_fold_mean_probability"])
        atomic_write_json(out.with_suffix(".json"), {"combination": self.cfg["export"]["fold_combination"],
                                                     "folds": side, "fingerprint": stable_hash(side)})
        self.figures()
        fin = mean[base["valid"]]
        return {"path": str(out), "mean_probability": float(np.nanmean(fin)), "p99": float(np.nanpercentile(fin, 99))}

    # ------------------------------------------------------------------ 14 / 15
    def export_submission(self) -> dict:
        import rasterio

        from .geophysics import load_base
        from .inventory import load_inventory, raw_path
        from .submission import export

        artifacts.require(self.paths, self.cfg, "full_probability", "raw_template")
        base = load_base(self.paths)
        inv = load_inventory(self.paths)
        tpath = raw_path(inv, "geodawn_template")
        p = self.paths.predictions / "e6_full_probability.tif"
        with rasterio.open(p) as s:
            prob = s.read(1)
        from .raw_data import read_template

        tvalid, tprof = read_template(tpath, base["grid"])
        prov = {"full_probability": read_json(p.with_suffix(".json")), "raw_fingerprint": inv["raw_fingerprint"],
                "template_sha256": next(r["checksum"] for r in inv["rows"] if r["source_id"] == "geodawn_template"),
                "code_version": CODE_VERSION}
        rep = export(self.paths.submissions, tpath, tprof, prob, tvalid, base["valid"], base["labels"], self.cfg, prov)
        if not rep["ok"]:
            bad = {k: [c for c, v in f["checks"].items() if not v] for k, f in rep["files"].items() if not f["ok"]}
            raise RuntimeError(f"submission validation failed: {bad} (see submissions/submission_validation.json)")
        return {"final": str(self.paths.submissions / "submission_e6_final.tif"), "policy": rep["final_policy"],
                "area_fraction": rep["area_fraction"], "sha256": rep["files"]["final"]["sha256"]}

    def validate_submission(self) -> dict:
        from .inventory import load_inventory, raw_path
        from .submission import read_validation, validate

        rep = read_validation(self.paths.submissions)
        if not rep:
            raise MissingArtifact("submissions/submission_validation.json missing - run section 14")
        final = self.paths.submissions / "submission_e6_final.tif"
        import rasterio

        with rasterio.open(final) as s:
            arr = s.read(1)
        again = validate(final, raw_path(load_inventory(self.paths), "geodawn_template"), arr, rep["final_policy"])
        again["matches_export_sha256"] = again["sha256"] == rep["files"]["final"]["sha256"]
        again["ok"] = again["ok"] and again["matches_export_sha256"]
        atomic_write_json(self.paths.submissions / "submission_revalidation.json", again)
        return again

    def status(self):
        return artifacts.status(self.paths, self.cfg)
