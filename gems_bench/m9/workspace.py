"""outputs/m9_geofault_production writer: the M7 workspace (atomic files, per-stage fingerprints, PNG + SVG figures
with their data, Parquet next to CSV, GeoPackage vectors, immutable records) with the M9 folder layout."""
from __future__ import annotations

import json
import time

from ..artifacts.atomic import atomic_write_json
from ..m7.workspace import M7Workspace

M9_VERSION = "m9.1"


class M9Workspace(M7Workspace):
    SUBDIRS = ("manifests", "manifests/units", "manifests/sections",
               "baseline", "features", "features/figures", "spectral", "spectral/figures",
               "datasets", "datasets/cv", "corruption", "corruption/samples", "corruption/benchmark",
               "models", "models/checkpoints", "training", "training/logs",
               "predictions", "predictions/oof", "predictions/discovery", "predictions/full",
               "validation", "validation/maps", "validation/metrics", "validation/figures",
               "dense_analysis", "tracing", "tracing/units", "tracing/oracle", "monte_carlo", "rl",
               "candidates", "candidates/units", "gbm", "gbm/models", "fusion", "inference", "inference/tiles",
               "reports", "reports/model_registry", "submissions", "gems3_engine")

    def finish(self, name: str, fp: str, info: dict | None = None, seconds: float | None = None) -> None:
        atomic_write_json(self.root / "manifests" / "units" / f"{name}.json",
                          {"unit": name, "fingerprint": fp, "status": "done", "m9_version": M9_VERSION,
                           "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": seconds,
                           "outputs": sorted(set(self._written)), "info": info or {}})
        self._unit = None

    @staticmethod
    def experiment_id(section: str, fp: str) -> str:
        return f"m9_{section}_{fp[:10]}"

    def section_manifest(self, section: str, payload: dict):
        """manifests/sections/<section>.json: what ran, with which fingerprint / profile, which files, where."""
        drive = self.cfg.get("_drive_root")
        body = {"section": section, "m9_version": M9_VERSION, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "profile": (self.cfg.get("_profile") or {}).get("name"), "local_root": str(self.root),
                "files": sorted(set(self._written)), **payload}
        if drive:
            body["drive_root"] = str(drive)
        return atomic_write_json(self.root / "manifests" / "sections" / f"{section}.json",
                                 json.loads(json.dumps(body, default=str)))
