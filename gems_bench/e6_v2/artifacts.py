"""Artifact dependency graph of the E6 V2 pipeline.

Every artifact has: artifact_id, producer, consumer, input_dependencies, expected_path (relative to the
namespace, or to drive_root for raw inputs), file_type, schema, fingerprint, completion_marker,
validation_function, recovery_procedure. ``write_graph`` saves it to ``manifests/artifact_graph.json``;
``status`` runs every validation function and writes ``manifests/artifact_status.csv``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cache import atomic_write_csv, atomic_write_json, read_json
from .paths import E6V2Paths
from .spatial_splits import DISC_UNITS, GEO_UNITS

UNITS = GEO_UNITS + DISC_UNITS


def _a(aid, producer, consumer, deps, path, ftype, schema, fp, marker, vfn, recovery):
    return {"artifact_id": aid, "producer": producer, "consumer": consumer, "input_dependencies": deps,
            "expected_path": path, "file_type": ftype, "schema": schema, "fingerprint": fp,
            "completion_marker": marker, "validation_function": vfn, "recovery_procedure": recovery}


def graph(cfg: dict) -> list[dict]:
    r = cfg["raw"]
    G = [
        _a("raw_geodawn_features", "external (GEMS data release)", ["raw_data_inventory", "base_masks", "terrain_features",
           "potential_features"], [], f"<drive_root>/<search_dirs>/{r['features_file']}", "GeoTIFF",
           "19 float bands, EPSG:32611, 100 m, band_name tags", "sha256 (inventory)", "inventory row status ok",
           "v_inventory_row:geodawn_features", "copy the official file into one of raw.search_dirs; never edit it"),
        _a("raw_faults", "external (GEMS data release)", ["raw_data_inventory", "base_masks"], [],
           f"<drive_root>/<search_dirs>/{r['faults_file']}", "GeoTIFF", "fault raster, value >= 1 = fault",
           "sha256 (inventory)", "inventory row status ok", "v_inventory_row:geodawn_faults", "as above"),
        _a("raw_template", "external (GEMS data release)", ["raw_data_inventory", "base_masks", "submission_final"], [],
           f"<drive_root>/<search_dirs>/{r['template_file']}", "GeoTIFF", "official submission template",
           "sha256 (inventory)", "inventory row status ok", "v_inventory_row:geodawn_template", "as above"),
        _a("raw_dem_tiles", "external (USGS 3DEP 1 m)", ["raw_data_inventory", "dem_source_inventory", "dem_windows"],
           [], f"<drive_root>/{r['dem_dirs'][0]}/**/*.tif (+ FGDC .xml)", "GeoTIFF", "1 m DEM, metre or foot vertical",
           "sha256(size, first MiB, last MiB) per tile", "inventory rows status ok", "v_inventory_dem",
           "download the 1 m tiles (and their metadata XML) into raw.dem_dirs; set dem.vertical_unit_override only "
           "after reading the metadata"),
        _a("raw_data_inventory", "E6V2Pipeline.inventory / preflight", ["preflight_report", "all cache stages"],
           ["raw_geodawn_features", "raw_faults", "raw_template", "raw_dem_tiles"], "manifests/raw_data_inventory.json",
           "JSON + CSV", "rows: source_id,type,path,size,checksum,checksum_kind,crs,resolution,shape,bands,nodata,bounds,"
           "vertical_unit,validation_status; raw_fingerprint", "raw_fingerprint = hash(source_id, checksum, status)",
           "file exists", "v_exists:manifests/raw_data_inventory.json", "re-run section 02"),
        _a("preflight_report", "E6V2Pipeline.preflight", ["user"], ["raw_data_inventory"],
           "manifests/preflight_report.json", "JSON + MD", "status READY|BLOCKED, blockers, warnings, facts",
           "none (always rewritten)", "status == READY", "v_preflight", "fix every blocker listed, re-run section 03"),
        _a("base_masks", "geophysics.prepare_base", ["terrain_features", "potential_features", "dem_windows",
           "datasets", "splits"], ["raw_data_inventory"], "cache/base/{valid,labels,template_valid}.npy, grid.json",
           "npy (bool) + JSON", "(H, W) bool; grid dict", "hash(GeoDAWN checksums, raw config, code)",
           "cache/base/manifest.json", "v_manifest:cache/base/manifest.json", "re-run section 04 (recomputes)"),
        _a("terrain_features", "geophysics.prepare_terrain", ["datasets"], ["base_masks"],
           "cache/terrain/{feature_stack.npy,feature_registry.csv,normalization.json}", "npy (Ct,H,W) f32 + CSV + JSON",
           "channels = contracts.terrain_channels", "hash(base fp, terrain + normalization config, code)",
           "cache/terrain/manifest.json", "v_manifest:cache/terrain/manifest.json", "re-run section 04"),
        _a("potential_features", "geophysics.prepare_potential", ["datasets"], ["base_masks"],
           "cache/potential_fields/{magnetic,gravity,subsurface}_features.npy, feature_registry.csv, normalization.json",
           "npy (C,H,W) f32 + CSV + JSON", "channels = contracts.field_channels(mag|grav) + subsurface",
           "hash(base fp, potential + normalization config, code)", "cache/potential_fields/manifest.json",
           "v_manifest:cache/potential_fields/manifest.json", "re-run section 04"),
        _a("dem_source_inventory", "inventory.build_inventory", ["dem_windows"], ["raw_dem_tiles"],
           "cache/dem/source_inventory/dem_sources.csv", "CSV", "one row per tile: header facts, vertical unit, status",
           "part of raw_fingerprint", "file exists", "v_csv:cache/dem/source_inventory/dem_sources.csv:validation_status",
           "re-run sections 02-03"),
        _a("dem_window_plan", "dem_aggregation.prepare_dem", ["dem_windows"], ["dem_source_inventory", "base_masks"],
           "cache/dem/mosaic/{window_plan,overlap_order}.csv", "CSV", "window_id,row0,col0,n_rows,n_cols,tile_priorities",
           "rewritten each run (deterministic)", "file exists", "v_csv:cache/dem/mosaic/window_plan.csv:window_id",
           "re-run section 05"),
        _a("dem_windows", "dem_aggregation.process_window", ["dem_features_100m", "dem_coverage_100m"],
           ["dem_window_plan", "raw_dem_tiles"], "cache/dem/windows/<window_id>.{npz,json}",
           "npz (features (22,r,c) f32, coverage (r,c) f32) + JSON",
           "features in contracts.DEM_CHANNELS order", "hash(dem config, code, grid, window, tile checksums)",
           "<window_id>.json written after the npz", "v_dem_windows",
           "re-run section 05: finished windows are kept, missing / stale ones are recomputed"),
        _a("dem_features_100m", "dem_aggregation.prepare_dem", ["datasets"], ["dem_windows"],
           "cache/dem/features_100m/{dem_features_100m.npy,normalization.json}", "npy (22,H,W) f32 + JSON",
           "contracts.DEM_CHANNELS", "hash(code fp, tile checksums, base fp)", "cache/dem/manifests/dem_manifest.json",
           "v_manifest:cache/dem/manifests/dem_manifest.json", "re-run section 06 (assembly only if windows complete)"),
        _a("dem_coverage_100m", "dem_aggregation.prepare_dem", ["datasets", "gates", "evaluation slices"],
           ["dem_windows"], "cache/dem/coverage_100m/dem_coverage_100m.npy", "npy (H,W) f32 in [0,1]",
           "measured share of usable 2 m pixels", "as dem_features_100m", "cache/dem/manifests/dem_manifest.json",
           "v_manifest:cache/dem/manifests/dem_manifest.json", "re-run section 06"),
        _a("datasets", "datasets.prepare_datasets", ["training", "inference"], ["terrain_features", "potential_features",
           "dem_features_100m", "dem_coverage_100m", "base_masks"],
           "datasets/{terrain,field,dem,dem_coverage,context,labels,valid,template_valid}.npy, dataset_schema.json",
           "npy (padded grid) + JSON", "dataset_schema.json (groups, axial pairs, normalisation, hashes)",
           "hash(upstream fingerprints, schema hash, code)", "datasets/manifest.json",
           "v_manifest:datasets/manifest.json", "re-run section 07"),
        _a("splits", "spatial_splits.build_splits", ["training", "evaluation"], ["base_masks"],
           "datasets/splits/{geo_folds.npz,discovery_splits.npz,blocks.csv,fault_systems.csv,split_summary.csv}",
           "npz + CSV", "roles (K,H,W) int8; discovery roles / inner (D,H,W) int8",
           "hash(base fp, splits config, code); identity = sha256 of the role arrays", "datasets/splits/manifest.json",
           "v_manifest:datasets/splits/manifest.json", "re-run section 08 (changes every model identity)"),
    ]
    for u in UNITS:
        sec = "09" if u.startswith("geo") else "11"
        G.append(_a(f"reference_{u}", "trainer.UnitTrainer.run_reference (stages A, B)", [f"e6_{u}"],
                    ["datasets", "splits"], f"checkpoints/{u}/reference.pt + reference.json", "torch state + JSON",
                    "ReferenceNet state dict", "identity (schema, raw, dataset, normalisation, split, spec, training, code)",
                    f"checkpoints/{u}/reference.json (sha256)", f"v_reference:{u}",
                    f"immutable. If the identity differs, delete checkpoints/{u}/ and re-run section {sec}"))
        sec = "10" if u.startswith("geo") else "11"
        G.append(_a(f"e6_{u}", "trainer.UnitTrainer.run_e6 (stages C, D, E)",
                    ["evaluation"] + (["fold_full_predictions"] if u.startswith("geo") else []),
                    [f"reference_{u}"], f"checkpoints/{u}/{{C_parity.json,D_final.pt,E_final.pt,E_parity.json}}",
                    "torch state + JSON", "E6System state dict (e6.* and reference.*)", "same identity as the reference",
                    f"checkpoints/{u}/E_final.json + E_parity.json", f"v_e6:{u}",
                    f"re-run section {sec}: resumes from <stage>_last.pt"))
    G += [
        _a("unit_eval_predictions", "evaluator.unit_prediction", ["metrics"], [f"e6_{u}" for u in UNITS],
           "predictions/units/<unit>_eval.{npy,json}", "npy f32 + JSON", "probabilities on the bbox of the scored region",
           "stage-E checkpoint sha256", "<unit>_eval.json", "v_exists:predictions/units",
           "re-run section 12 (recomputed when the checkpoint changed)"),
        _a("metrics", "evaluator.evaluate", ["user", "figures"], ["unit_eval_predictions"],
           "metrics/{training_history,geographic_dti,discovery_dti,dti_components,coverage_slices,short_fault_recall,"
           "gate_weights}.csv + evaluation_summary.json", "CSV + JSON",
           "dti_components: protocol,unit,policy,slice,eval_px,dti,tp_w,fp_w,fn_w,n_truth,pred_mass,...",
           "rewritten by each evaluation", "metrics/evaluation_summary.json", "v_csv:metrics/dti_components.csv:dti",
           "re-run section 12"),
        _a("figures", "pipeline.figures", ["user"], ["metrics", "dem_coverage_100m", "full_probability"],
           "figures/{loss_curves,dem_coverage,gate_weights,prediction_map}.{png,svg}", "PNG + SVG", "plots",
           "rewritten", "files exist", "v_exists:figures", "re-run the figure cell (section 12 / 13)"),
        _a("fold_full_predictions", "pipeline.infer_full_region", ["full_probability"], [f"e6_{u}" for u in GEO_UNITS],
           "predictions/e6_fold{0,1,2}_full.tif + .json", "GeoTIFF f32 (grid), NaN outside footprint",
           "fold-k E6System probability", "stage-E checkpoint sha256 + dataset fingerprint (sidecar)",
           "e6_fold<k>_full.json", "v_geotiff:predictions/e6_fold0_full.tif", "re-run section 13"),
        _a("full_probability", "pipeline.infer_full_region", ["submission_continuous", "submission_binary"],
           ["fold_full_predictions"], "predictions/e6_full_probability.tif + .json", "GeoTIFF f32",
           "mean of the three fold maps", "hash of the three sidecars", "e6_full_probability.json",
           "v_geotiff:predictions/e6_full_probability.tif", "re-run section 13"),
        _a("submission_continuous", "submission.export", ["submission_final"], ["full_probability", "raw_template"],
           "submissions/e6_continuous.tif", "GeoTIFF f32, template grid", "finalise(prob)", "sha256 in validation",
           "submissions/submission_validation.json", "v_submission:continuous", "re-run section 14"),
        _a("submission_binary", "submission.export", ["submission_final"], ["full_probability", "raw_template",
           "base_masks"], "submissions/e6_area_budget_binary.tif", "GeoTIFF f32 {0,1}, template grid",
           "finalise(top-n selection)", "sha256 in validation", "submissions/submission_validation.json",
           "v_submission:area_budget_binary", "re-run section 14"),
        _a("submission_final", "submission.export", ["user (manual upload)"], ["submission_continuous",
           "submission_binary"], "submissions/submission_e6_final.tif", "GeoTIFF f32, template grid",
           "the export.final policy", "sha256", "submissions/submission_sha256.txt", "v_submission:final",
           "re-run sections 14-15; upload is manual"),
    ]
    return G


# ---------------------------------------------------------------------------
# validation functions (named in the graph)
# ---------------------------------------------------------------------------
def v_exists(paths: E6V2Paths, rel: str) -> tuple[bool, str]:
    p = paths.root / rel
    return p.exists(), "ok" if p.exists() else f"missing {rel}"


def v_manifest(paths: E6V2Paths, rel: str) -> tuple[bool, str]:
    from .cache import manifest_status

    m = read_json(paths.root / rel)
    if m is None:
        return False, f"missing {rel}"
    st, why = manifest_status(paths.root / rel, m["fingerprint"], paths.root)
    return st == "complete", why


def v_csv(paths: E6V2Paths, rel: str, column: str) -> tuple[bool, str]:
    p = paths.root / rel
    if not p.exists():
        return False, f"missing {rel}"
    df = pd.read_csv(p, nrows=5)
    return column in df.columns, "ok" if column in df.columns else f"column {column} missing"


def v_geotiff(paths: E6V2Paths, rel: str) -> tuple[bool, str]:
    p = paths.root / rel
    if not p.exists():
        return False, f"missing {rel}"
    import rasterio

    g = read_json(paths.base / "grid.json") or {}
    with rasterio.open(p) as s:
        ok = (s.height, s.width) == (g.get("height"), g.get("width"))
    return ok, "ok" if ok else "grid differs from cache/base/grid.json"


def v_inventory_row(paths: E6V2Paths, source_id: str) -> tuple[bool, str]:
    inv = read_json(paths.manifests / "raw_data_inventory.json")
    if inv is None:
        return False, "no inventory"
    for r in inv["rows"]:
        if r["source_id"] == source_id:
            return str(r["validation_status"]).startswith("ok"), str(r["validation_status"])
    return False, "not inventoried"


def v_inventory_dem(paths: E6V2Paths) -> tuple[bool, str]:
    inv = read_json(paths.manifests / "raw_data_inventory.json")
    if inv is None:
        return False, "no inventory"
    n = inv.get("dem_tiles_ok", 0)
    return n > 0, f"{n} tiles ok"


def v_preflight(paths: E6V2Paths) -> tuple[bool, str]:
    rep = read_json(paths.manifests / "preflight_report.json")
    if rep is None:
        return False, "not run"
    return rep["status"] == "READY", rep["status"]


def v_dem_windows(paths: E6V2Paths) -> tuple[bool, str]:
    plan = paths.dem_dir("mosaic") / "window_plan.csv"
    if not plan.exists():
        return False, "no window plan"
    ids = pd.read_csv(plan).window_id
    d = paths.dem_dir("windows")
    done = sum((d / f"{w}.json").exists() and (d / f"{w}.npz").exists() for w in ids)
    return done == len(ids), f"{done}/{len(ids)} windows"


def v_reference(paths: E6V2Paths, unit: str) -> tuple[bool, str]:
    js = read_json(paths.unit_dir(unit) / "reference.json")
    return js is not None, "ok" if js else "not trained"


def v_e6(paths: E6V2Paths, unit: str) -> tuple[bool, str]:
    d = paths.unit_dir(unit)
    ok = (d / "E_final.json").exists() and (d / "E_parity.json").exists()
    par = read_json(d / "E_parity.json") or {}
    return ok and par.get("uncovered_exact", False), "ok" if ok else "not trained"


def v_submission(paths: E6V2Paths, key: str) -> tuple[bool, str]:
    rep = read_json(paths.submissions / "submission_validation.json")
    if rep is None:
        return False, "not exported"
    f = rep["files"].get(key)
    return bool(f and f["ok"]), "ok" if f and f["ok"] else "validation failed"


def _run(paths: E6V2Paths, spec: str) -> tuple[bool, str]:
    name, *args = spec.split(":")
    fn = globals()[name]
    return fn(paths, *args)


def write_graph(paths: E6V2Paths, cfg: dict) -> Path:
    return atomic_write_json(paths.manifests / "artifact_graph.json", {"artifacts": graph(cfg)})


def status(paths: E6V2Paths, cfg: dict) -> pd.DataFrame:
    rows = []
    for a in graph(cfg):
        try:
            ok, why = _run(paths, a["validation_function"])
        except Exception as exc:
            ok, why = False, f"{type(exc).__name__}: {exc}"
        rows.append({"artifact_id": a["artifact_id"], "ok": bool(ok), "detail": why, "producer": a["producer"],
                     "recovery_procedure": a["recovery_procedure"]})
    df = pd.DataFrame(rows)
    atomic_write_csv(paths.manifests / "artifact_status.csv", df)
    return df


def require(paths: E6V2Paths, cfg: dict, *artifact_ids: str):
    """Raise ``MissingArtifact`` naming the recovery procedure of every missing upstream artifact."""
    from .cache import MissingArtifact

    g = {a["artifact_id"]: a for a in graph(cfg)}
    bad = []
    for aid in artifact_ids:
        ok, why = _run(paths, g[aid]["validation_function"])
        if not ok:
            bad.append(f"{aid}: {why} -> {g[aid]['recovery_procedure']}")
    if bad:
        raise MissingArtifact("missing upstream artifacts:\n  " + "\n  ".join(bad))
