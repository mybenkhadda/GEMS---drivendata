"""Typed, validated configuration of the Geological Data Atlas.

The user-facing dict (notebook section 00) is merged over ``DEFAULTS`` and turned into an ``AtlasConfig``.
Scientific parameters live here too, so every fingerprint can be derived from one object.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

ATLAS_VERSION = "atlas-v1.0"
DEM_MODES = ("FAST_10M", "DETAILED_2M")

DEFAULTS: dict = {
    "drive_root": "/content/drive/MyDrive/GEMS_2",
    "atlas_root": "GEMS_GEOLOGICAL_DATA_ATLAS_V1",
    "scratch_root": "/content/atlas_scratch",
    "snapshot_raw": True,
    "verify_checksums": True,
    "enable_new_dem_downloads": False,
    "dem_mode": "FAST_10M",
    "run_ingestion": True,
    "run_feature_engineering": True,
    "run_eda": True,
    "reuse_valid_cache": True,
    "save_figures": True,
    "export_csv": True,
    "export_parquet": True,
    "export_geotiff": True,
    "seed": 2026,
    # ---- raw-source discovery (relative to drive_root; ONLY these original raw folders) ----
    "raw": {
        "geodawn_file": "gems-geodawn-numerical-features.tif",
        "faults_file": "existing_faults.tif",
        "template_file": "example_submission.tif",
        "geodawn_dirs": ["data/raw/geodawn", "data/raw", "data/raw/faults", ""],
        "dem_dir": "data/raw/dem_1m",
        "catalog_files": ["data/raw/dem_1m/tnm_urls.json", "data/raw/dem_1m/tnm_items.json",
                          "data/raw/tnm_items.json", "data/raw/tnm_urls.json"],
        "provenance_globs": ["data/raw/**/*LICENSE*", "data/raw/**/*license*", "data/raw/**/*PROVENANCE*",
                             "data/raw/**/*provenance*", "data/raw/**/*README*", "data/raw/**/*readme*"],
        "nodata_abs": 1.0e30,
        "valid_min_band_share": 0.5,
        "label_min_value": 1,
    },
    "grid_expected": {"epsg": 32611, "pixel_m": 100.0, "height": 3730, "width": 3292},
    # ---- snapshot ----
    "snapshot": {"chunk_mib": 16, "retries": 3, "retry_wait_s": 5.0, "allow_partial_dem": False,
                 "capacity_margin": 1.05},
    # ---- optional acquisition ----
    "acquisition": {"max_new_tiles": 10, "max_gib": 4.0, "min_new_cells": 200, "license": "USGS 3DEP 1 m DEM, "
                    "public domain (U.S. Government work); credit U.S. Geological Survey",
                    "allowed_host": "prd-tnm.s3.amazonaws.com", "allowed_prefix": "/StagedProducts/Elevation/1m/",
                    "timeout_s": 120},
    # ---- potential fields ----
    "potential": {"magnetic_band": "rtp", "gravity_band": "iso_grav_anom", "magnetic_unit": "nT",
                  "gravity_unit": "mGal", "continuation_m": [500, 1000, 2000, 5000], "pad_frac": 0.25,
                  "taper_px": 32, "edge_guard_m": 1000.0, "continuation_guard_cap_m": 2000.0,
                  "structure_scales_m": [300, 1000, 3000], "duplicate_rho": 0.98},
    # ---- terrain 100 m ----
    "terrain": {"elevation_band": "det_elev", "slope_band": "det_elev_slope", "scales_m": [300, 1000, 2000, 3000],
                "edge_guard_sigma": 2.0},
    # ---- DEM ----
    "dem": {
        "native_res_min_m": 0.5, "native_res_max_m": 3.0, "vertical_unit_override": None,
        "window_cells": 50, "seam_guard_m": 60.0, "edge_guard_sigma": 4.0, "min_cell_support": 0.25,
        "edge_slope_deg": 20.0, "edge_coherence": 0.5, "workers": 0, "budget_minutes": 0,
        "FAST_10M": {"work_res_m": 10.0, "halo_m": 1200.0, "gradient_sigma_m": 10.0, "curvature_sigma_m": 30.0,
                     "structure_scales_m": [50, 200], "tpi_radii_m": [100, 500], "relief_window_m": 200.0,
                     "scarp_offset_m": 30.0, "roughness_sigma_m": 30.0},
        "DETAILED_2M": {"work_res_m": 2.0, "halo_m": 420.0, "gradient_sigma_m": 2.0, "curvature_sigma_m": 10.0,
                        "structure_scales_m": [5, 20], "tpi_radii_m": [50, 200], "relief_window_m": 50.0,
                        "scarp_offset_m": 10.0, "roughness_sigma_m": 10.0},
        "detailed_example_windows": 3,
    },
    # ---- cross-modal / lineaments / EDA ----
    "evidence": {"strong_rank": 0.9, "line_rank": 0.8, "min_coherence": 0.3, "continuity_half_length_m": 1000.0},
    "lineaments": {"evidence_quantile": 0.92, "min_length_km": 3.0, "min_linearity": 0.85, "smooth_sigma_px": 1.0,
                   "simplify_m": 100.0},
    "proximity": {"bands_m": [0, 300, 1000, 3000], "block_px": 10, "max_blocks_per_band": 20000,
                  "profile_half_length_m": 5000.0, "profile_examples": 6, "profile_min_trace_km": 5.0},
    "analytics": {"sample_block_px": 10, "max_samples": 60000, "redundant_rho": 0.9, "spatially_limited": 0.3,
                  "noisy_lag1": 0.3, "artifact_ratio": 1.5},
    "figures": {"dpi": 200, "max_display_px": 1400},
}


def deep_merge(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def stable_hash(obj, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()[:n]


def module_hash(*names: str) -> str:
    here = Path(__file__).resolve().parent
    h = hashlib.sha256(ATLAS_VERSION.encode())
    for n in sorted(names):
        h.update((here / f"{n}.py").read_bytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class AtlasConfig:
    data: dict = field(default_factory=dict)

    @classmethod
    def from_user(cls, user: dict | None = None) -> "AtlasConfig":
        cfg = cls(deep_merge(DEFAULTS, user or {}))
        cfg.validate()
        return cfg

    def __getitem__(self, k):
        return self.data[k]

    def get(self, k, default=None):
        return self.data.get(k, default)

    @property
    def dem_params(self) -> dict:
        return {**{k: v for k, v in self.data["dem"].items() if k not in DEM_MODES}, **self.data["dem"][self.dem_mode]}

    @property
    def dem_mode(self) -> str:
        return self.data["dem_mode"]

    def section_hash(self, *keys: str) -> str:
        return stable_hash({k: self.data.get(k) for k in keys})

    def validate(self) -> None:
        d = self.data
        bools = ("snapshot_raw", "verify_checksums", "enable_new_dem_downloads", "run_ingestion",
                 "run_feature_engineering", "run_eda", "reuse_valid_cache", "save_figures", "export_csv",
                 "export_parquet", "export_geotiff")
        for b in bools:
            if not isinstance(d[b], bool):
                raise ValueError(f"config {b} must be True/False, got {d[b]!r}")
        if d["dem_mode"] not in DEM_MODES:
            raise ValueError(f"dem_mode must be one of {DEM_MODES}")
        if not isinstance(d["seed"], int):
            raise ValueError("seed must be an int")
        dp = self.dem_params
        res = dp["work_res_m"]
        if abs(100.0 / res - round(100.0 / res)) > 1e-9:
            raise ValueError("DEM work_res_m must divide 100 m")
        support = max(2 * max(dp["tpi_radii_m"]), 2 * max(dp["structure_scales_m"]) + 4 * dp["gradient_sigma_m"]
                      + dp["scarp_offset_m"], 4 * dp["curvature_sigma_m"], 8 * dp["roughness_sigma_m"],
                      dp["relief_window_m"] / 2)
        if dp["halo_m"] < support:
            raise ValueError(f"DEM halo {dp['halo_m']} m < largest operator support {support} m")
        a = d["acquisition"]
        if d["enable_new_dem_downloads"] and not (1 <= a["max_new_tiles"] <= 100 and 0 < a["max_gib"] <= 100):
            raise ValueError("acquisition budgets must be explicit: 1..100 tiles and 0..100 GiB")
        if sorted(d["proximity"]["bands_m"]) != d["proximity"]["bands_m"]:
            raise ValueError("proximity.bands_m must be ascending")

    def as_dict(self) -> dict:
        return copy.deepcopy(self.data)
