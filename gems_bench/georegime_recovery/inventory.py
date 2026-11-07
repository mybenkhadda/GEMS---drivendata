"""Expected V1 / Atlas artifacts (EXPECTED paths, verified at runtime - never assumed to exist)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

V1_FAMILIES = ("magnetic", "gravity", "terrain", "raw", "cross_modal")       # gems_bench.georegime.data.FEATURES
HASH_KINDS = {"config", "checkpoint", "normalizer", "prediction", "router", "prediction_meta", "table", "template",
              "mask", "atlas_schema", "lineaments", "registry"}


def expected_sources(paths, cfg: dict, schema: dict | None, bank_key: str | None) -> list[dict]:
    u = cfg["units"]
    units = u["geographic"] + u["discovery"] + [u["full"]]
    S = []

    def add(role, key, path, required, kind, note=""):
        S.append({"role": role, "key": key, "path": Path(path), "required": bool(required), "kind": kind, "note": note})

    add("v1_config", "config.json", paths.src("config.json"), True, "config")
    for un in units:
        add("checkpoint", un, paths.checkpoint(un), True, "checkpoint")
        add("normalizer", un, paths.normalizer(un), False, "normalizer", "cross-check; the checkpoint carries its normalizer")
        add("v1_prediction", un, paths.v1_prediction(un), un == u["full"], "prediction",
            "full model score raster (sections 03/04/16)" if un == u["full"] else "original ROI prediction (section 06)")
        add("v1_prediction_meta", un, paths.v1_prediction(un).with_suffix(".json"), False, "prediction_meta")
        add("v1_router", un, paths.v1_router(un), False, "router")
    for t in ("geographic_dti.json", "discovery_dti.json", "validation_components.csv", "prediction_mass_diagnostic.csv",
              "candidate_model_review.csv"):
        add("v1_table", t, paths.src("tables", t), False, "table")
    for h in sorted(paths.src("tables").glob("*_history.csv")) if paths.src("tables").is_dir() else []:
        add("v1_table", h.name, h, False, "table")
    a = paths.atlas
    add("atlas_schema", "feature_stack_schema.json", a / "06_ml_ready/schemas/feature_stack_schema.json", True, "atlas_schema")
    for fam in V1_FAMILIES:
        f = (schema or {}).get("families", {}).get(fam, {}).get("file", f"06_ml_ready/feature_stack/{fam}_stack.tif")
        add("atlas_stack", fam, a / f, True, "stack", "fingerprinted by the V1 Bank key (size, mtime_ns)")
    for m in ("valid_numerical", "fault_labels", "template_valid"):
        add("atlas_mask", m, a / f"02_geospatial/masks/{m}.tif", True, "mask")
    add("template", "example_submission.tif", a / "00_raw_snapshot/official_template/example_submission.tif", True, "template")
    add("atlas_registry", "feature_registry.parquet", a / "03_features/feature_registry/feature_registry.parquet", False,
        "registry", "needed for geological evidence (sections 11, 13, 14 policy 4)")
    add("atlas_lineaments", "candidate_lineaments.parquet", a / "04_eda/candidate_lineaments/candidate_lineaments.parquet",
        False, "lineaments", "section 13")
    add("atlas_grid", "analysis_grid.json", a / "02_geospatial/grids/analysis_grid.json", False, "grid")
    if bank_key:
        add("v1_bank_cache", "bank", paths.v1_cache / f"bank_{bank_key}.npy", False, "bank",
            "read-only reuse; rebuilt in local scratch from the Atlas if absent")
        add("v1_bank_cache", "context", paths.v1_cache / f"context_{bank_key}.npy", False, "bank")
    return S


def inventory(sources: list[dict], hashes) -> pd.DataFrame:
    rows = []
    for s in sources:
        p: Path = s["path"]
        row = {"role": s["role"], "key": s["key"], "path": str(p), "required": s["required"], "exists": False,
               "size": None, "mtime_ns": None, "sha256": "", "status": "", "detail": s["note"]}
        try:
            if p.is_file():
                st = p.stat()
                row.update(exists=True, size=int(st.st_size), mtime_ns=int(st.st_mtime_ns), status="OK")
                if s["kind"] in HASH_KINDS:
                    row["sha256"] = hashes.sha(p)
            else:
                row["status"] = "MISSING" if s["required"] else "OPTIONAL_MISSING"
        except OSError as e:                       # inaccessible is NOT missing (Drive disconnects: Errno 107)
            row["status"] = "INACCESSIBLE"
            row["detail"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return pd.DataFrame(rows)
