# Artifact registry (`outputs/G11_MULTISCALE_FAULT_DISCOVERY/`; SMOKE_TEST: `..._SMOKE/`)

Every stage writes `manifests/<stage>.json`. Each record holds: stage, fingerprint, config hash, source hashes, code commit, `started_utc`, `completed_utc`, status (RUNNING / COMPLETE / FAILED), outputs (path, size, sha256) and validation results.

| folder | files | producer |
|---|---|---|
| `00_runtime/` | `runtime_report.json` (hardware, adjustments, per-stage seconds, peak RAM and GPU) | notebook §03, `report` |
| `01_audit/` | `artifact_registry.parquet/.json`, `artifact_inventory.parquet`, `artifact_compatibility.csv` (external predictions: metadata source, nesting, provenance, status), `dem_vertical_units.csv` (VERIFIED_METRES / VERIFIED_FEET / UNKNOWN / CONFLICTING plus the statements), `dem_duplicate_tiles.csv`, `audit_report.md`, `audit_summary.json`, `dependency_graph.md`, `smoke_window.json` | `audit` |
| `02_ingestion/` | `master_grid.json`, `geophysical_registry.parquet` (every band: name, unit, dtype, nodata, valid fraction, statistics), `dem_catalog.parquet`, `dem_coverage.gpkg`, `dem_coverage_100m.tif`, `fault_segments.gpkg`, `fault_registry.parquet` (with `fault_system_id`), `label_states.tif`, `coverage_report.json` (incl. fault coverage lost to unit exclusions), `figures/*.png` | `ingest` |
| `03_features/` | `geophysical/*.npy`, `terrain_100m/*.npy`, `targets/*.npy` (label-derived; never predictors), `terrain_native/*.tif` (native windows), `segment_features.parquet` (known-fault cross-fault morphology), `feature_registry.parquet` | `ingest`, `features` |
| `04_eda/` | `eda_report.html`, `eda_summary.md`, `feature_contrast.csv`, `multiscale_analysis.csv` (100 m + native), `model_complementarity.csv`, `modeling_recommendations.csv`, `feature_distributions.parquet`, `feature_redundancy.parquet`, `failure_taxonomy.parquet`, `spatial_correlation.parquet`, `coverage_summary.parquet`, `fig_*.png`, `candidate_atlas/site_*.png` | `eda` |
| `05_splits/` | `split_registry.parquet`, `{train,validation,stack,test,embargo}_mask.tif`, `split_manifest.json`, `split_map.png`, `leakage_report.json`, `leakage_checks.parquet` | `splits` |
| `06_models/xgboost/E0..E2/` | `model.json`, `feature_schema.json`, `training_history.json`, `feature_importance.parquet`, `metrics.json`, `checkpoint_metadata.json` | `train` |
| `06_models/segformer/E3[__<loss>]/`, `06_models/gated_fusion/E4[__<mode>]/` | `best.pt`, `last.pt`, `history.csv`, `feature_schema.json`, `metrics.json`, `checkpoint_metadata.json` | `train` |
| `06_models/fusion/E5/`, `06_models/segment_ranker/` | `metrics.json`; `ranker.txt`, `feature_schema.json`, `training_history.json` | `fusion`, `infer` |
| `07_predictions/` | `<experiment>_probability.tif` (E0–E5 and variants); `external/<family>__<name>.tif` (aligned G07 / M9 / P6 maps) | `train`, `fusion`, `audit` |
| `08_candidates/` | `candidates.gpkg` (UNVERIFIED), `candidate_features.parquet`, `ranker_validation.parquet`, `budget_validation.parquet`, `inference_manifest.json` | `infer` |
| `09_evaluation/` | `leaderboard.csv`, `leaderboard_final.csv`, `experiment_matrix*.parquet`, `selection.json` (written before any test metric), `fusion_comparison.csv` | `fusion`, `evaluate` |
| `10_submissions/` | `g11_best_submission.tif` (SMOKE: `g11_smoke_submission_NOT_FOR_UPLOAD.tif`), `.json`, `.sha256`, `submission_manifest.json`, `submission_validation.json`, `submission_sha256.txt`, `submission_record.json` | `submit` |
| `11_reports/` | `G11_FINAL_EXECUTION_REPORT.md/.html`, `G11_EXPERIMENT_LEADERBOARD.csv` | `report` |
| `manifests/`, `cache/sha256.jsonl` | stage records, hash cache | all |
| local `/content/g11_work/` | `feature_cache/` (sha256-verified staged channels + `cache_manifest.json`), `dem_mosaic/dem_25m_<catalog-hash>.tif` | `train` (E3 / E4) |
