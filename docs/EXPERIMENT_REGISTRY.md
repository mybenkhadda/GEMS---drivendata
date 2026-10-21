# Experiment registry

All runs live in `outputs/benchmark/experiments/<exp_id>/` and append a row to
`outputs/benchmark/experiment_registry.csv` when finished. No run has been executed yet (2026-10-20).

## Planned runs (notebook 02)
| label in nb | exp_id | family | inputs | section |
|---|---|---|---|---|
| B0_lineament | `b0_lineament_fused_s2026` | baseline | Phase-1 `fused_score` (resampled) | §3 |
| B0_terrain_ridge | `b0_terrain_ridge_s2026` | baseline | `terrain:ridge@1km` | §3 |
| lgbm_G1 | `lgbm_G1_s2026` | gbm | terrain only | §4 P0 |
| lgbm_G2 | `lgbm_G2_s2026` | gbm | 19 raw bands | §4 P0 |
| lgbm_G3 | `lgbm_G3_s2026` | gbm | raw + engineered | §4 P1 |
| lgbm_G4 | `lgbm_G4_s2026` | gbm | G3 + context | §4 P1 |
| cnn_A_bcedice | `cnn_A_bcedice_p256_s2026` | cnn | raw, BCE+Dice | §5 |
| cnn_A_tversky | `cnn_A_tversky_p256_s2026` | cnn | raw, Tversky 0.2/0.8 | §5 |
| cnn_B_bcedice | `cnn_B_bcedice_p256_s2026` | cnn | raw + 10 enhancements | §5 |
| (discovery) | `disc{d}_<cv exp_id>` | gbm/cnn | as CV run, withheld systems ignored | §7 |

## Files per run
`config.json` (model/data config, seed, feature list, fold dir) · `run.json` (status, timings, RSS, environment, artifact index) ·
`metrics_fold.csv` (DTI components + diagnostics per fold + pooled) · `history.csv` · `importance.csv` / `shap_summary.csv` (GBM) ·
`sampling_summary.csv` · `transform_grid.csv`, `transform_crossfit.csv`, `output_transform.json` · `oof.tif` (CV) or `pred.tif` (single fit) ·
`models/fold{k}.txt|.pt`, `models/normalisation.json` (CNN).

## Shared artifacts
| artifact | path |
|---|---|
| feature cache + registry | `outputs/benchmark/cache/features/*.npy`, `feature_registry.csv`, `feature_build_log.csv` |
| geographic folds | `outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026/` (`fold_map.tif`, `blocks.parquet`, `fold_summary.csv`, `fold_params.json`) |
| discovery splits | `outputs/benchmark/folds/discovery_d3_link1km_s2026/` (`discovery_map.tif`, `fault_systems.csv`, `discovery_summary.csv`) |
| native segment ids | `outputs/benchmark/datasets/native_segment_ids.tif`, `segment_attributes.parquet` |
| ensembles | `outputs/benchmark/ensembles/ens_*.tif|json`, `ensemble_summary.csv` |
| submissions | `outputs/benchmark/submissions/<candidate>__<policy>.tif|json`, `full_maps/` |
| reports | `outputs/benchmark/benchmark_summary.csv`, `model_comparison.png`, `fault_recovery_by_archetype.csv`, `pseudo_discovery_results.csv`, `model_disagreement.csv`, `BENCHMARK_FINDINGS.md` |

## Reload snippets
```python
from gems_bench.config import BenchPaths
from gems_bench.data import load_native
from gems_bench.folds import GeoFolds
from gems_bench.persist import ExperimentRun, load_registry
from gems_bench.ensemble import read_map

paths = BenchPaths.discover(); data = load_native(paths)
folds = GeoFolds.load(paths.folds / 'geo_k3_b20km_buf13km_s2026', data.grid)
run = ExperimentRun(paths, 'lgbm_G1_s2026', 'gbm')      # reads run.json
oof = read_map(run.oof_path); metrics = run.table('metrics_fold')
load_registry(paths)                                     # all finished runs
```
