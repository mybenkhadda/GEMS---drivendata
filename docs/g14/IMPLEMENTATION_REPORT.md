# G14 implementation report

Branch `g14-geofaultformer`, based on `main` at `da9d17a`. Version `g14.0.1-untested`.

**NO CODE, TESTS, NOTEBOOKS, TRAINING, INFERENCE OR EXPERIMENTS WERE EXECUTED.** No data was downloaded and no
Drive was touched. No results exist. The only command-line operations were git (status / grep / diff / branch /
commit / push), plus one accidental `python3` invocation with an empty script, which ran no repository code.

## Inventory

| path | purpose |
|---|---|
| `gems_bench/g14/config.py` | typed config, run modes, safety flags, validation, semantic hash |
| `gems_bench/g14/experiments.py` | ladder T0–T7 + ablation registry |
| `gems_bench/g14/contract.py` | channel schema, composites, known context, float16 pyramid stack, train-only normaliser, contract checks |
| `gems_bench/g14/splits.py` | `UnitSystem` (g13 / phase5 backends), units, fault-system grouping, leakage audit |
| `gems_bench/g14/multiscale.py` | targets, per-unit inputs, aligned three-view dataset, D4 / axial / vector augmentation, sensor dropout |
| `gems_bench/g14/model.py` | GeoFaultFormer: modality fusion, local CNN encoders, Swin / MiT, global tokens, gated context, FiLM, heads |
| `gems_bench/g14/losses.py` | masked BCE, Tversky, distance-centreline, clDice, sparsity, orientation, connectivity, DTI surrogate |
| `gems_bench/g14/train.py` | per-unit training, AMP, checkpoints with hash check, early stopping, time budget |
| `gems_bench/g14/inference.py` | tiled full-grid prediction, Hann blending, TTA, MC dropout, held-out maps |
| `gems_bench/g14/evaluation.py` | official-unit scoring, cross-fitted operating points, diagnostics, bootstrap, locked TEST |
| `gems_bench/g14/graph.py` | evidence-gated segment linking (rules + optional classifier on visible labels) |
| `gems_bench/g14/ensemble.py` | rank blends, cross-fitted weights, OOF stacking, spatial MoE, regime gate |
| `gems_bench/g14/submission.py` | guarded GeoTIFF export via G13 `export_candidate` |
| `gems_bench/g14/pipeline.py` | orchestration used by the notebook |
| `gems_bench/g14/viz.py` | plots |
| `notebooks/colab/GEMS_G14_GEOFAULTFORMER_MASTER.ipynb` | master notebook, safe defaults |
| `tests/test_g14_core.py`, `tests/test_g14_model.py` | unexecuted tests |
| `docs/g14/*.md` | audit, leakage register, architecture, contract, validation, losses, matrix, runbook, risks |

## G13 integration

* **Reused:** input discovery and loading, grid checks, feature store (optional), splits and locked TEST,
  `crossfit` / `area_table` / bootstrap, representations / render, topology, `export_candidate` / validate.
* **Reused from P6 / Phase 5:** `known_features`, `load_context5`.
* **Replaced:**
  * the tabular model with a multiscale network;
  * global feature scaling as the model input with per-unit train-only normalisation;
  * the 171-representation `dti_opt` with two cheap representations and a wider area grid;
  * G13's "no known faults" choice with permitted known-fault context.
* **Not modified:** any existing file.

## Static checks performed

* Every cross-package import was located with `git grep`.
* Config keys used in the docs were checked against `config.py`.
* The diff was confirmed to touch only new paths.
* `git grep` was run for tokens, keys and passwords.

None of these is a substitute for running the tests.
