# G06 artifact registry

Root: `<local_root or drive_root>/outputs/g06_geological_feature_store/<run_id>/`,
`run_id = g06-<sha10(G06 version, feature_store config, seed, source sha256)>` (`run_id_mode='commit'` adds the
commit). Local `/content/gems_work/...` is mirrored to `MyDrive/GEMS_2/...` after every stage (tile caches excluded).

| path | stage | content |
|---|---|---|
| `manifests/run.json`, `runtime_errors.json` | all | run identity, sources, flags, stage statuses, fingerprints |
| `manifests/stages/<stage>.json` | all | status, fingerprint, required outputs, code rev, commit, notes |
| `manifests/dependencies.json` | preflight | grid, exact band index, sources, engine status, missing deps |
| `manifests/feature_schema.json` | features | copy of the store schema |
| `manifests/validation_contract.json` | splits | units, fits, rules |
| `manifests/artifact_inventory.csv` | report | every file (history / tiles excluded) |
| `audit/tests.csv`, `audit/pytest_output.txt` | tests | pytest of `tests/test_g06_*.py` |
| `inventory/band_inventory.csv`, `dependency_inventory.csv`, `artifact_compatibility.csv`, `dependencies.csv`, `missing_dependencies.json` | preflight | inventory |
| `inventory/raw_bands.csv`, `entropy_edges.json`, `footprint.npz` | raw_eda | band statistics, footprint |
| `inventory/feature_coverage.csv` | features | copy of the coverage |
| `features/<dir>/<group>.tif`, `<group>.json` | features | channel groups (float32, deflate, NaN nodata) + group manifests |
| `features/feature_registry.parquet`, `feature_schema.json`, `feature_coverage.csv`, `footprint.npz` | features | registry / schema / coverage |
| `features/context/domain_model.joblib` | features | k-means domain model |
| `tiles/<fp10>/<group>/<tile>.npz/.json` | features | resumable tile cache (deleted after assembly unless `keep_tile_cache`) |
| `splits/units.csv`, `fits.csv`, `isolation_audit.csv`, `geographic/fold_summary.csv`, `discovery/split_summary.csv`, `inner_cv/inner_cv_summary.csv` | splits | validation design |
| `datasets/pixels/*`, `datasets/patches/*`, `datasets/segments/*`, `datasets/connections/*`, `datasets/dataset_manifests/*` | datasets | D1-D4 (see DATASET_CONTRACTS.md) |
| `models/<kind>/<exp>/<fp10>/<fit>/model.joblib, model.json, feature_importance.csv` | train_pixel / full_inference | pixel models |
| `models/pixel_registry.csv`, `segment_registry.csv`, `segment_models.json`, `patch_registry.csv`, `meta_registry.json` | train_* | registries (merged across runs) |
| `models/patch_unet/<exp>/<fp10>/<fit>/unet.pt, unet.json, history.csv` | train_patch | U-Net |
| `predictions/oof/<exp>/<fp10>/[nested/]<fit>__<unit>.npy (+ .json)` | train_pixel / train_patch | held-out crops |
| `predictions/segments/<exp>/<fp10>/scores.parquet, info.json, <unit>.npy` | train_segment | segment scores and maps |
| `predictions/meta/<kind>/<fp10>/<unit>.npy` | train_meta | meta maps |
| `predictions/research_geotiffs/<exp>_<fp10>_probability.tif` | full_inference | ROLE=RESEARCH_NOT_A_SUBMISSION |
| `predictions/full_region/full_inference_manifest.json`, `parity_<exp>.json` | full_inference | parity |
| `evaluation/metrics.csv`, `metrics_rows.csv`, `per_zone.csv`, `system_rows.csv`, `system_recovery.csv`, `model_comparison.csv`, `ablations.csv`, `representation_selection.csv`, `promotion_decisions.csv`, `decision.json`, `pr_auc.csv`, `calibration.csv`, `complementarity.csv`, `historical_references.csv` | evaluate / train_meta | results |
| `figures/raw_atlas`, `physical_features`, `structure`, `feature_distributions`, `training`, `model_diagnostics`, `dti_frontiers`, `fault_failures`, `previews` | several | PNG + SVG + source CSV / Parquet |
| `submissions/submission_manifest.json` (+ `g06_candidate_<model>.tif` only if every gate passes) | export | READY_FOR_MANUAL_UPLOAD at best |
| `reports/FEATURE_REPORT.md`, `MODEL_BENCHMARK.md`, `FAILURE_ANALYSIS.md`, `EXECUTION_STATUS.md`, `NEXT_EXPERIMENT.md` | report | reports from persisted tables |
| `history/<stage>/<fp10>/...`, `history/index.csv` | any | earlier successful outputs replaced by a new fingerprint |
