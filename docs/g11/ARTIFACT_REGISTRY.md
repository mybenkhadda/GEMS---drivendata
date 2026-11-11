# Artifact registry (`outputs/G11_MULTISCALE_FAULT_DISCOVERY/`; SMOKE_TEST: `..._SMOKE/`)

## Stage records

Every stage writes `manifests/<stage>.json`. Downstream stages are namespaced as `manifests/<ns>/<stage>.json`, where `ns = <feature_version>__split_<split_version>`. Each record holds:
- the stage name and fingerprint;
- config hash, source hashes and code commit;
- `started_utc` and `completed_utc`;
- status (RUNNING / COMPLETE / FAILED);
- outputs (path, size, sha256);
- validation results.

**Fingerprints are section-scoped:** each stage hashes only the config sections it uses (see [RECOVERY_FIRST.md](RECOVERY_FIRST.md)).

**Never modified by the recovery-first code:** the legacy records `manifests/features.json`, `splits.json`, `train_E0.json` and `train_E1.json`, and the legacy files under `03_features/` and `05_splits/`. They are read and verified only.

## Files by folder

| folder | files | producer |
|---|---|---|
| `00_runtime/` | `runtime_report.json` | §03, report |
| `01_audit/` | `artifact_registry.*`, `artifact_inventory.parquet`, `artifact_compatibility.csv`, `dem_vertical_units.csv`, `audit_report.md`, `audit_summary.json`, `dependency_graph.md`, `smoke_window.json` | audit |
| `02_ingestion/` | `master_grid.json`, `geophysical_registry.parquet`, `dem_catalog.parquet`, `dem_coverage.*`, `fault_segments.gpkg`, `fault_registry.parquet`, `label_states.tif`, `coverage_report.json`, `figures/` | ingest (labels are no longer written here) |
| `recovery/` | `feature_recovery_registry.parquet` (one row per layer: status, reason, header, sha256, provenance, reusable_for_request), `feature_recovery_report.json`, `feature_coverage_map.png`, `missing_chunks.parquet` (layers + DEM chunks still to compute) | §04 recover |
| `03_features/layer_manifest.parquet` | completion manifest of the pool: one row per VERIFIED_COMPLETE layer (layer_id, relpath, size, mtime_ns, sha256, finite fraction, spec hash, provenance, processing resolution) | recover, process_features |
| `03_features/{geophysical,targets,terrain_100m}/*.npy` | legacy layers, kept in place; adopted as `name` or `name@r2m` | legacy run |
| `03_features/terrain_100m/r10m/*.npy` | new DEM-class layers (`dem100_*`, multimodal, `terrain_scarp_rank`) at 10 m | process_features |
| `03_features/dem_chunks/r10m/<chunk>.npz` + `<chunk>.json` | committed DEM statistic chunks + completion sidecars (spec, source tiles, size, sha256) | dem_engine |
| `03_features/datasets/<version>/` | `dataset_manifest.json`, `feature_schema.parquet`, `label_schema.parquet`, `valid_mask.npy`, `dem_availability.npy`, `feature_availability.parquet`, `native_window_index.parquet`, `fault_geometry_registry.parquet`, `processing_config.json`, `source_provenance.json`, `validation_report.json`, `validation_quick.json` / `validation_full.json`, **`COMPLETE.json`** (last) | finalize / validate |
| `03_features/versions/<version>/...` | layers of a FORCE_NEW_VERSION run (version-private) | process_features |
| `03_features/terrain_native/*.tif`, `segment_features.parquet` | native 1 m windows, known-fault cross-fault morphology | legacy run / process_features |
| `04_eda/` | `eda_report.html`, `eda_summary.md`, `feature_contrast.csv`, `multiscale_analysis.csv`, `model_complementarity.csv`, `modeling_recommendations.csv`, …, `candidate_atlas/` | eda |
| `05_splits/<version>/` | `split_code.tif`, `{train,validation,stack,test,embargo}_mask.tif`, `groups.npy`, `split_registry.parquet`, `regions.parquet`, `split_gates.parquet` (or `split_gates_FAILED.parquet`), `split_report.json`, `split_manifest.json`, `leakage_checks.parquet`, `leakage_report.json`, `split_map.png`, **`COMPLETE.json`** | splits |
| `05_splits/SPLIT_V1_INVALID.json` | marks the legacy root-level split INVALID (written only after v2 passed); legacy files kept | splits |
| `06_models/<ns>/training_data/<exp>_<key>/` | `indices_<split>.npy`, `categories_<split>.npy`, `scaler.json`, `class_weights.json`, `sampling_config.json`, `manifest.json` | training_data |
| `06_models/<ns>/{xgboost,segformer,gated_fusion,fusion,segment_ranker}/<exp>/` | model / checkpoints (`model.json`, `best.pt`, `last.pt`), `feature_schema.json` (channels + normaliser), `metrics.json`, `checkpoint_metadata.json`, `run_fingerprint.json`, `archive/<fp>_<time>/` (checkpoints of another configuration, moved, never deleted) | train / infer |
| `07_predictions/<ns>/` | `<exp>_probability.tif`, `selected_calibrated_probability.tif`, `prediction_inventory.parquet` | train, predict_all, infer |
| `07_predictions/external/` | aligned G07 / M9 / P6 maps (EDA only) | audit |
| `08_candidates/<ns>/` | `candidates.gpkg` (UNVERIFIED), `candidate_features.parquet`, `ranker_validation.parquet`, `budget_validation.parquet`, `inference_manifest.json` | infer (E4) |
| `09_evaluation/<ns>/` | `leaderboard.csv`, `leaderboard_final.csv`, `experiment_matrix*.parquet`, `selection.json` (before any test metric), `fusion_comparison.csv` | fusion, evaluate |
| `10_submissions/<ns>/` | `g11_best_submission.tif` + `.json` / `.sha256`, `submission_manifest.json`, `submission_validation.json`, `submission_record.json` (never uploaded) | submit |
| `11_reports/<ns>/` | `G11_FINAL_EXECUTION_REPORT.md/.html`, `G11_EXPERIMENT_LEADERBOARD.csv` | report |
| local `/content/g11_work/` | `dem_tile_cache/r10m/` (each tile read once), `dem_chunks_local/`, `feature_commit/` (staging before commit), `feature_cache/` (deep-model channels), `dem_mosaic/` | engine / train |
