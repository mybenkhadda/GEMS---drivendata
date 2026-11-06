# E6 artifact manifest

Paths relative to `GEMS_2/` on Drive (mirrored under `/content/gems_work/` in Colab). `<id>` =
`e6_masked_threeway_full__<config hash8>` (printed by the notebook; the yaml change of this audit gives a new id).
`<fp>` = DEM cache fingerprint (cached parts + extractor code). Outcome classes of the preflight:
`AVAILABLE_VALID` / `DERIVABLE_MISSING` / `BLOCKED_MISSING`.

## Inputs

| artifact | path | if missing | produced by |
|---|---|---|---|
| GeoDAWN numerical raster | `data/raw/geodawn/gems-geodawn-numerical-features.tif` | BLOCKED | competition |
| fault labels | `data/raw/faults/existing_faults.tif` (or `data/raw/geodawn/`) | BLOCKED | competition |
| submission template | `data/raw/geodawn/example_submission.tif` | BLOCKED | competition |
| geographic folds | `outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026/fold_map.tif` | BLOCKED | master pipeline `spatial_folds` |
| withheld-system splits | `outputs/benchmark/folds/discovery_d3_link1km_s2026/discovery_map.tif` | BLOCKED | master pipeline `discovery_splits` |
| fault segments | `outputs/benchmark/datasets/segment_attributes.parquet` or `outputs/eda/datasets/known_fault_segments.parquet` | BLOCKED | EDA / master pipeline |
| base channel schema | `configs/baselines/legacy_specs/P4_B_gated.json` (repo) | BLOCKED | repo |
| terrain / potential-field caches (incl. derivatives, FFT products) | `outputs/benchmark/cache/features/*.npy`, `outputs/phase4/cache/features/*.npy` | DERIVABLE (rebuilt from the raster) | phase-2 / phase-4 stores |
| G07 cached 10 m DEM parts | `outputs/g07_multiscale_fault_discovery/g07-*/dem_features/coarse_parts/<fp>/<tile>_<bytes>.tif` | BLOCKED | G07 DEM features stage |
| DEM catalog / native 1 m tiles | `data/raw/dem_1m/` | not used by E6 | - |
| DEM 2 m intermediates | none in E6 | - | - |
| E1 warm-start models | `experiments/e1_gsn_parity__b8b8c3db/checkpoints/<unit>_model.pt` | DERIVABLE (unit trains its baseline from scratch) | E1 experiment |

## Produced (in order)

| artifact | path | stale protection |
|---|---|---|
| DEM 100 m tiles | `outputs/e6_pipeline/cache/<fp>/tiles/<tile>.tif` (+ `.json`) | folder keyed by `<fp>` |
| DEM 100 m bank incl. coverage | `outputs/e6_pipeline/cache/<fp>/dem_features_100m.tif`, `manifest.json`, `tile_log.csv` | `<fp>` |
| DEM channel caches | `outputs/e6_pipeline/features/<fp>/dem_*.npy` | `<fp>` |
| normalisation + coverage rule | `experiments/<id>/checkpoints/normalisation.json` | rewritten at every stack build |
| input record | `experiments/<id>/reports/e6_input_cache.json`, `e6_architecture.json`, `spec.json` | - |
| resume checkpoints | `experiments/<id>/checkpoints/<unit>_ep<NNN>.pt`, `<unit>_latest.json` | spec hash (incl. DEM `<fp>`) |
| final models (6 units) | `experiments/<id>/checkpoints/{cv_fold0,1,2,disc0,1,2}_model.pt` | spec hash; carries `spec`, `e6_warm_start` |
| fold / split predictions | `experiments/<id>/predictions/{oof_fold<k>,oof,disc<d>}.tif` | unit skipped only with a valid final model |
| training tables | `experiments/<id>/metrics/{training_history,fold_metrics,discovery_runs,runtime_profile,gpu_memory_profile}.csv` | - |
| E6 DTI | `experiments/<id>/metrics/e6_dti.csv` (`continuous` + `exact_k`, pooled) | - |
| full-region fold maps | `experiments/<id>/predictions/e6_full_fold<k>.tif` + `.json` (checkpoint sha256, spec hash) | sidecar |
| final probability | `experiments/<id>/predictions/e6_full_probability.tif` | recomputed at every export |
| **submission** | `experiments/<id>/submissions/submission_e6_geostructurenet.tif` (+ `.json`); a re-export adds `__<timestamp>` | readback + sha256 |
| submission manifest | `experiments/<id>/reports/e6_submission_manifest.json`: sha256, Drive path, gates, readback, selection report, fold checkpoint sha256s, spec hash, DEM `<fp>`, git commit | - |

A submission is certified only when every gate is True, `readback.status == PASS` and the Drive copy has the same
sha256. Missing checkpoints, incomplete coverage of the valid footprint or a failed format check stop the export.
