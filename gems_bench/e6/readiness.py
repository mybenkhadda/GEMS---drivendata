"""Fast E6 readiness inventory; does not train, test, or copy large data."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json


def report_e6_readiness(drive_root: str | Path, *, output_folder="outputs/e6_submission_milestone") -> dict:
    drive = Path(drive_root)
    e1 = drive / "experiments/e1_gsn_parity__b8b8c3db"
    checkpoints = e1 / "checkpoints"
    manifest_path = e1 / "manifest.json"
    model_files = [checkpoints / f"cv_fold{k}_model.pt" for k in range(3)]
    available = [p.name for p in model_files if p.is_file()]
    missing = [p.name for p in model_files if not p.is_file()]

    result = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "base_architecture": "GeoStructureNet E1 gated two-branch",
        "checkpoint_dir": str(checkpoints),
        "available_cv_checkpoints": available,
        "missing_cv_checkpoints": missing,
        "complete_e1_cv_warmstart": not missing,
        "e6_model_source": "gems_bench/models/geostructurenet_e6.py",
        "ready_to_train_full_e6_cv": False,
        "ready_for_e6_submission": False,
        "default_submission": "W2 frozen champion; not the untrained E6 model",
        "no_tests": True,
        "no_mirror_data": True,
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result["e1_stage_status"] = {
            name: info.get("status", "?") for name, info in
            (manifest.get("stages") or {}).items()
        }
        result["e1_units"] = manifest.get("units", {})

    # Optional DEM caches; no filesystem walk of 1m original files.
    root = drive / "outputs/g07_multiscale_fault_discovery"
    results = []
    if root.is_dir():
        for run_dir in root.glob("g07-*/dem_features/fast_coarse/manifest.json"):
            try:
                m = json.loads(run_dir.read_text(encoding="utf-8"))
                product = run_dir.parent / Path(m["output"]).name
                if not product.exists():
                    product = root / run_dir.parts[-4] / m["output"]
                results.append({
                    "manifest": str(run_dir), "output": m.get("output"),
                    "tile_count": m.get("n_complete", 0), "exists": product.is_file(),
                })
            except (ValueError, KeyError, OSError):
                continue
    result["fast_dem_candidates"] = results
    result["ready_to_train_full_e6_cv"] = not missing and any(x["exists"] for x in results)
    out = drive / output_folder
    out.mkdir(parents=True, exist_ok=True)
    (out / "e6_readiness.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
