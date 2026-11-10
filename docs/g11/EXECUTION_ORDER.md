# Execution order

**Entry point:** `notebooks/colab/GEMS_G11_MASTER_COLAB_EXECUTION.ipynb` (see [COLAB_MASTER.md](COLAB_MASTER.md)). The five `GEMS_G11_{0..4}` notebooks run single phases.

| § | section | call | gate |
|---|---|---|---|
| 00 | configuration | `CONFIG` dict | Validated in §03 by `notebook_config.build_plan`: unknown keys and auto-upload are refused. |
| 01 | Drive + runtime | mount, `nvidia-smi`, RAM, disk | Drive root must exist. |
| 02 | checkout + deps | `git ls-remote` / fetch / detached checkout; `pip install -e .[colab]` | Stops if `ref` is not on GitHub, lacks `gems_bench/g11`, or the checkout has local modifications. |
| 03 | LEVEL 1 tests + plan | pytest (synthetic), `build_plan`, `detect_runtime`, `configure_runtime`, `G11Pipeline`, `StepRunner` | Tests must pass. |
| 04 | discovery | `audit.build_artifact_registry` | read-only |
| 05 | real-data audit | `P.audit()` | fail-fast (CRS, grid, template) |
| 06–08 | geophysics, DEM catalog, faults | `P.ingest()` | band count (strict), faults on the grid |
| 09 | features | `P.features()` | no label-derived predictor |
| 10 | EDA | `P.eda()` | optional |
| 11 | splits | `P.splits()` | **training stops on any leakage failure** |
| 12 | XGBoost E0–E2 | `P.train([...])` | required |
| 13 | SegFormer E3 + variants | `P.train(["E3"])` | optional |
| 14 | gated fusion E4 + variants | `P.train(["E4"])` | optional |
| 15 | model fusion, validation selection, candidates | `P.fusion()`, `P.evaluate()`, `P.infer()` | Selection is written before any test metric. |
| 16 | segment ranking | (tables from `P.infer`) | |
| 17 | model comparison | validation leaderboard | |
| 18 | held-out test (once) | `P.evaluate(final=True)` | Refused without `selection.json`. |
| 19 | submission | `P.submit()` | template validation; no upload |
| 20 | report | `P.report(PLAN, INFO)` | |

Failure policy (`StepRunner`):
- **Required** steps (audit, ingest, features, splits, XGBoost, evaluate, submit) stop the notebook on failure.
- **Optional** steps (EDA, deep models, ablation variants, fusion) are recorded as FAILED and skipped.
- `failure_policy="STOP_ON_ANY_FAILURE"` stops on everything.
