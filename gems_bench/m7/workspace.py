"""outputs/m7_discovery_first writer: the GEMS3 workspace (atomic files, per-stage fingerprints, PNG + SVG figures
with their data, Parquet next to CSV, GeoPackage vectors) with the M7 folder layout, experiment identifiers and
immutable records that are never overwritten."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts.atomic import atomic_save, atomic_write_json
from ..gems3.workspace import G3Workspace

M7_VERSION = "m7.1"


class M7Workspace(G3Workspace):
    SUBDIRS = ("manifests", "manifests/units", "manifests/sections",
               "recovered_baselines", "recovered_baselines/heldout",
               "failure_analysis/figures", "failure_analysis/tables", "failure_analysis/case_studies",
               "candidate_generation/rasters", "candidate_generation/vectors", "candidate_generation/vectors/units",
               "candidate_generation/tables", "candidate_generation/oracle",
               "dem/inventory", "dem/pilot", "dem/features", "dem/candidates",
               "map_completion/candidates", "map_completion/corruptions", "map_completion/metrics",
               "candidate_ranking/features", "candidate_ranking/models", "candidate_ranking/predictions",
               "validation/metrics", "validation/figures", "validation/maps",
               "reports", "submissions", "gems3_engine")

    def finish(self, name: str, fp: str, info: dict | None = None, seconds: float | None = None) -> None:
        atomic_write_json(self.root / "manifests" / "units" / f"{name}.json",
                          {"unit": name, "fingerprint": fp, "status": "done", "m7_version": M7_VERSION,
                           "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": seconds,
                           "outputs": sorted(set(self._written)), "info": info or {}})
        self._unit = None

    # -- identifiers -------------------------------------------------------------
    @staticmethod
    def experiment_id(section: str, fp: str) -> str:
        """Unique, readable id of one computation: section + fingerprint of its inputs and configuration."""
        return f"m7_{section}_{fp[:10]}"

    # -- immutable records ---------------------------------------------------------
    def save_immutable_table(self, df: pd.DataFrame, name: str, sub: str, fp: str) -> Path:
        """``<sub>/<name>__<fp>.csv`` (+ Parquet): written once. An existing file with the same fingerprint is
        kept (same inputs, same protocol); a different protocol gets its own file. ``<name>__latest.json`` points at
        the newest one. Nothing is ever replaced silently."""
        p = self.root / sub / f"{name}__{fp}.csv"
        if not p.exists():
            atomic_save(p, lambda t: df.to_csv(t, index=False))
            if self.cfg.get("save_parquet") and len(df):
                from ..pipeline.m6 import _parquet_safe

                try:
                    atomic_save(p.with_suffix(".parquet"), lambda t: _parquet_safe(df).to_parquet(t, index=False))
                except Exception as e:
                    print(f"[m7] parquet skipped for {name}: {type(e).__name__}: {e}")
        self._track(p)
        if p.with_suffix(".parquet").exists():
            self._track(p.with_suffix(".parquet"))
        self.save_json({"file": p.name, "fingerprint": fp, "pointed": time.strftime("%Y-%m-%dT%H:%M:%S")},
                       f"{sub}/{name}__latest.json")
        return p

    def load_immutable_table(self, name: str, sub: str) -> pd.DataFrame:
        ptr = self.root / sub / f"{name}__latest.json"
        if not ptr.exists():
            return pd.DataFrame()
        f = json.loads(ptr.read_text(encoding="utf-8"))["file"]
        p = self.root / sub / f
        for q, reader in ((p.with_suffix(".parquet"), pd.read_parquet), (p, pd.read_csv)):
            if q.exists():
                try:
                    return reader(q)
                except Exception:
                    continue
        return pd.DataFrame()

    def section_manifest(self, section: str, payload: dict) -> Path:
        """manifests/sections/<section>.json: what ran, with which fingerprint, which files, and where they are."""
        drive = self.cfg.get("_drive_root")
        files = sorted(set(self._written))
        body = {"section": section, "m7_version": M7_VERSION, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "local_root": str(self.root), "files": files, **payload}
        if drive:
            body["drive_root"] = str(drive)
        return atomic_write_json(self.root / "manifests" / "sections" / f"{section}.json",
                                 json.loads(json.dumps(body, default=str)))

    def path(self, rel: str) -> Path:
        return self.root / rel


def mask_hash(a: np.ndarray | None) -> str:
    """Short content hash of a boolean mask (shape included): protocol keys compare masks, not their names."""
    if a is None:
        return "none"
    b = np.ascontiguousarray(np.asarray(a, bool))
    h = hashlib.sha1(np.packbits(b.ravel()).tobytes())
    h.update(str(b.shape).encode())
    return h.hexdigest()[:12]
