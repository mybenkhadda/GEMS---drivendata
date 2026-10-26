"""outputs/m8_physics_probabilistic writer: the M7 workspace (atomic files, per-stage fingerprints, PNG + SVG figures
with their data, Parquet next to CSV, GeoPackage vectors, immutable records) with the M8 folder layout and version."""
from __future__ import annotations

import time

from ..artifacts.atomic import atomic_write_json
from ..m7.workspace import M7Workspace

M8_VERSION = "m8.1"


class M8Workspace(M7Workspace):
    SUBDIRS = ("manifests", "manifests/units", "manifests/sections",
               "baseline", "baseline/recovered_m7", "baseline/heldout",
               "signal_recovery", "signal_recovery/case_studies",
               "geophysical_features", "geophysical_features/magnetic", "geophysical_features/gravity",
               "geophysical_features/terrain", "geophysical_features/structural_worms",
               "geophysical_features/multimodal", "geophysical_features/context",
               "fault_normal_profiles", "fault_normal_profiles/gallery",
               "candidate_generation", "candidate_generation/units", "candidate_generation/oracle",
               "monte_carlo", "monte_carlo/realizations", "monte_carlo/stability", "monte_carlo/uncertainty_maps",
               "monte_carlo/extras",
               "fault_networks", "fault_networks/units", "fault_networks/benchmark",
               "dem", "dem/inventory", "dem/pilot", "dem/features", "dem/candidates",
               "feature_tables", "feature_tables/units", "ranking", "graphs", "rasterization",
               "ensemble", "validation", "validation/maps", "validation/metrics", "validation/figures",
               "reports", "submissions")

    def finish(self, name: str, fp: str, info: dict | None = None, seconds: float | None = None) -> None:
        atomic_write_json(self.root / "manifests" / "units" / f"{name}.json",
                          {"unit": name, "fingerprint": fp, "status": "done", "m8_version": M8_VERSION,
                           "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": seconds,
                           "outputs": sorted(set(self._written)), "info": info or {}})
        self._unit = None

    @staticmethod
    def experiment_id(section: str, fp: str) -> str:
        return f"m8_{section}_{fp[:10]}"

    def section_manifest(self, section: str, payload: dict):
        """manifests/sections/<section>.json: what ran, with which fingerprint / profile, which files, where."""
        import json

        drive = self.cfg.get("_drive_root")
        body = {"section": section, "m8_version": M8_VERSION, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "profile": (self.cfg.get("_profile") or {}).get("name"), "local_root": str(self.root),
                "files": sorted(set(self._written)), **payload}
        if drive:
            body["drive_root"] = str(drive)
        return atomic_write_json(self.root / "manifests" / "sections" / f"{section}.json",
                                 json.loads(json.dumps(body, default=str)))
