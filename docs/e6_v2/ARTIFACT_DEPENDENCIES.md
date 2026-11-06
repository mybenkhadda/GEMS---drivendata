# Artifact dependencies

Source of truth: `gems_bench/e6_v2/artifacts.py::graph`, written to `manifests/artifact_graph.json` whenever the
pipeline is constructed. `PIPE.status()` runs every validation function and writes `manifests/artifact_status.csv`.
Every stage calls `artifacts.require(...)` on its inputs and raises `MissingArtifact` with the recovery procedure.
Paths are relative to `GEMS_2/e6_v2_from_raw/` unless marked `<drive_root>`.

| artifact_id | producer | consumer | inputs | expected_path | completion_marker | validation_function | recovery |
|---|---|---|---|---|---|---|---|
| raw_geodawn_features | external | inventory, base, terrain, potential | - | `<drive_root>/<search_dir>/gems-geodawn-numerical-features.tif` | inventory row ok | v_inventory_row | copy the official file |
| raw_faults | external | inventory, base | - | `<drive_root>/<search_dir>/existing_faults.tif` | inventory row ok | v_inventory_row | copy |
| raw_template | external | inventory, base, submission | - | `<drive_root>/<search_dir>/example_submission.tif` | inventory row ok | v_inventory_row | copy |
| raw_dem_tiles | external (USGS 3DEP) | inventory, DEM | - | `<drive_root>/data/raw/dem_1m/**/*.tif` | inventory rows ok | v_inventory_dem | download tiles + XML |
| raw_data_inventory | inventory / preflight | all | raw_* | `manifests/raw_data_inventory.json/.csv` | file | v_exists | section 02 |
| preflight_report | preflight | user | inventory | `manifests/preflight_report.json`, `reports/preflight_report.md` | status READY | v_preflight | fix blockers, section 03 |
| base_masks | geophysics.prepare_base | terrain, potential, DEM, datasets, splits | inventory | `cache/base/*.npy, grid.json` | `cache/base/manifest.json` | v_manifest | section 04 |
| terrain_features | geophysics.prepare_terrain | datasets | base | `cache/terrain/*` | `cache/terrain/manifest.json` | v_manifest | section 04 |
| potential_features | geophysics.prepare_potential | datasets | base | `cache/potential_fields/*` | `.../manifest.json` | v_manifest | section 04 |
| dem_source_inventory | inventory | DEM | raw_dem_tiles | `cache/dem/source_inventory/dem_sources.csv` | file | v_csv | sections 02-03 |
| dem_window_plan | prepare_dem | windows | dem sources, base | `cache/dem/mosaic/{window_plan,overlap_order}.csv` | file | v_csv | section 05 |
| dem_windows | process_window | DEM 100 m | plan, tiles | `cache/dem/windows/<id>.{npz,json}` | `<id>.json` after `<id>.npz` | v_dem_windows | re-run section 05 |
| dem_features_100m | prepare_dem (assemble) | datasets | windows | `cache/dem/features_100m/*` | `cache/dem/manifests/dem_manifest.json` | v_manifest | section 06 |
| dem_coverage_100m | prepare_dem (assemble) | datasets, gates, slices | windows | `cache/dem/coverage_100m/dem_coverage_100m.npy` | same | v_manifest | section 06 |
| datasets | prepare_datasets | training, inference | terrain, potential, DEM, base | `datasets/*.npy, dataset_schema.json` | `datasets/manifest.json` | v_manifest | section 07 |
| splits | build_splits | training, evaluation | base | `datasets/splits/*` | `datasets/splits/manifest.json` | v_manifest | section 08 |
| reference_<unit> (6) | run_reference (A, B) | e6_<unit> | datasets, splits | `checkpoints/<unit>/reference.pt` | `reference.json` (sha256) | v_reference | immutable; delete the unit folder if the identity changed |
| e6_<unit> (6) | run_e6 (C, D, E) | evaluation, full inference (geo) | reference_<unit> | `checkpoints/<unit>/{C_parity.json,D_final.pt,E_final.pt,E_parity.json}` | `E_final.json` + `E_parity.json` | v_e6 | sections 10 / 11 resume |
| unit_eval_predictions | evaluator | metrics | e6_* | `predictions/units/<unit>_eval.{npy,json}` | json (checkpoint sha) | v_exists | section 12 |
| metrics | evaluator | user, figures | unit_eval_predictions | `metrics/*.csv, evaluation_summary.json` | summary json | v_csv | section 12 |
| figures | pipeline.figures | user | metrics, coverage, probability | `figures/*.{png,svg}` | files | v_exists | sections 06 / 12 / 13 |
| fold_full_predictions | infer_full_region | full_probability | e6_geo* | `predictions/e6_fold{0,1,2}_full.tif + .json` | sidecar json | v_geotiff | section 13 |
| full_probability | infer_full_region | submission | fold maps | `predictions/e6_full_probability.tif + .json` | json | v_geotiff | section 13 |
| submission_continuous | submission.export | final | probability, template | `submissions/e6_continuous.tif` | `submission_validation.json` | v_submission | section 14 |
| submission_binary | submission.export | final | probability, template, base | `submissions/e6_area_budget_binary.tif` | same | v_submission | section 14 |
| submission_final | submission.export | user (manual upload) | both maps | `submissions/submission_e6_final.tif` | `submission_sha256.txt` | v_submission | sections 14-15 |

Fingerprints (the `fingerprint` field of the graph):
- base = hash(GeoDAWN checksums, `raw` config, code of geophysics/raw_data)
- terrain / potential = hash(base, their config sections + normalisation, code)
- DEM window = hash(dem config + DEM code, grid, window id, names + checksums of its tiles); DEM stage = hash(DEM
  code, all tile checksums, base)
- datasets = hash(terrain, potential, DEM, base fingerprints, schema hash, code)
- splits = hash(base, splits config, code); identity = sha256 of the role arrays
- checkpoints: identity = code version, schema hash, raw fingerprint, dataset fingerprint, normalisation hash,
  training config (minus runtime-only keys), model config, model/training code hash, unit, split identity, model
  spec, prior
- fold maps: stage-E checkpoint sha256 + dataset fingerprint
