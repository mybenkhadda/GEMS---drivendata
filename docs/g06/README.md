# G06 - geological feature store & multi-task ML benchmark

**Question:** can models trained on engineered geological observations discover fault evidence that W2 has not
learned? G06 builds a reusable, fold-safe feature store from the 19-band GeoDAWN raster once, then benchmarks pixel,
segment, patch and meta models against the frozen W2 under the canonical official-proxy protocol.

Code rev `g06.1-r1`. **Status: authored, never executed** - see `SESSION_HANDOFF.md`.

| what | where |
|---|---|
| notebook | `notebooks/colab/GEMS_G06_GEOLOGICAL_FEATURE_STORE_AND_ML_BENCHMARK.ipynb` |
| package | `gems_bench/g06/` (orchestrator `pipeline.G06Pipeline`) |
| configuration | `configs/g06/feature_store.yaml`, `configs/g06/benchmarks.yaml` |
| tests (synthetic) | `tests/test_g06_*.py` (10 files; run in Colab by the `tests` stage) |
| outputs | `MyDrive/GEMS_2/outputs/g06_geological_feature_store/<run_id>/` |

## Documents

* `ARCHITECTURE.md` - modules, stages, dependencies, caching.
* `FEATURE_DICTIONARY.md` - geophysical meaning of every feature family.
* `DATASET_CONTRACTS.md` - D1-D4 schemas and label scope.
* `TRAINING_PROTOCOL.md` - how each model consumes each dataset.
* `VALIDATION_PROTOCOL.md` - units, visibilities, incomplete labels, strictness, comparison with W2.
* `ARTIFACT_REGISTRY.md` - every persisted file.
* `COLAB_EXECUTION_GUIDE.md` - exact CONFIG / RUN for each run.
* `SESSION_HANDOFF.md` - state, decisions, blockers.

## Experiments

| id | features | model | track |
|---|---|---|---|
| G06-P0 | raw | Random Forest | A (geology) |
| G06-P1 | raw | LightGBM | A |
| G06-P2 | raw + physics | LightGBM | A |
| G06-P3 | raw + physics + texture | LightGBM | A |
| G06-P4 | raw + physics + structure | LightGBM | A |
| G06-P5 | full bank (+ regional context) | LightGBM | A |
| G06-P5X | full bank | XGBoost | A |
| G06-S1 | segment geology (+ G06 pixel evidence) | LightGBM (CatBoost / ranker optional) | A |
| G06-S2 | segment geology + honest W2 / CNN | LightGBM | B (W2-aware) |
| G06-U1 | selected channels | compact U-Net | A |
| G06-M1 | honest W2 / CNN + G06 | rank average / residual LightGBM / domain MoE | B |

The primary metric is pooled official-proxy DTI (summed components) under one fixed policy
(`evaluation.primary`, default top 1 % of the eligible domain), V1 first, V3 and V2 next to it, worst zone always
shown. Historical G05 numbers (0.282538 / 0.242156 / 0.130868) and the public score (0.0942) are references from other
protocols, never comparable or tuning targets. Nothing is uploaded automatically.
