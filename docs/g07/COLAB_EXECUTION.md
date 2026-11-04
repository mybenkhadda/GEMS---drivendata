# G07 Colab execution

Code rev `g07.1-r2`. Nothing here has been run yet: the times below are design budgets, not measurements.

## First GPU run

1. Commit and push the G07 code, then open `notebooks/colab/GEMS_G07_MULTISCALE_FAULT_DISCOVERY.ipynb` from GitHub.
2. Runtime -> Change runtime type -> **GPU**. Prefer high RAM (DEM features and evaluation load large arrays).
3. Runtime -> Run all, with the notebook's default `CONFIG`, `RUN`, `OVERRIDES`. Authorize Drive when asked.

The notebook stops at the first mandatory stage that does not succeed, with the reason (`STRICT_NOTEBOOK = True`).

| section | stage | device | stops the notebook when |
|---|---|---|---|
| 03 | tests, preflight, GPU readiness | CPU | a test fails, the raster contract fails, `use_gpu` without CUDA |
| 04-05 | DEM inventory, coverage | CPU | the stage fails (excluded tiles are listed, not fatal here) |
| 07 | DEM feasibility + **training gate display** | CPU | the stage fails (every decision is displayed) |
| 08 | DEM features | CPU | the gate blocks: NO_GO in strict mode, or no usable DEM |
| 10, 12 | G06 restore, datasets, patch readiness | CPU | G06 store invalid, isolation audit fails, patches missing |
| 14 | **U-Net G07-C1** | **GPU** | any fit missing or invalid (PARTIAL raises too) |
| 15, 16 | subsurface B0-B2, surface A1 / A2 / A1RF | CPU | no fit valid (PARTIAL is shown and allowed) |
| 21 | evaluation | CPU | W2 maps incomplete, stage failed |
| 29-30 | reports, execution summary | CPU | - |

`STRICT_NOTEBOOK = False` prints the same reasons and continues; `P.run()` never raises (batch use).

## Prerequisites on Drive (`/content/drive/MyDrive/GEMS_2`)

| what | where | if missing |
|---|---|---|
| GeoDAWN raster, faults, template | `data/raw/...` (`inputs.sources` of `discovery.yaml`) | preflight FAILED |
| DEM tiles | `data/raw/dem_1m/**/*.tif` | feasibility BLOCKED_INPUT |
| G06 store | `outputs/g06_geological_feature_store/g06-cda34ef1f1` | g06_restore FAILED |
| P6 W2 maps, folds, splits | `outputs/p6_fast_track/...` (frozen engine) | preflight reports it, evaluation BLOCKED_INPUT |

Existing DEM files are used as they are. `dem.download.enabled` stays `False` and is never switched on automatically.

## Vertical units

If section 04 shows `UNKNOWN_VERTICAL_UNIT`, no explicit metadata states the elevation unit (band units, tags, the vertical part of a compound CRS, the `<altunits>` of a sidecar XML). Check one tile's USGS metadata, then in section 01:

```python
OVERRIDES["dem"]["vertical_unit_override"] = "metre"
```

It applies only to tiles whose headers are silent, is recorded per tile (`vertical_unit_source = override`) and keeps the same run directory. Finished stages before DEM features stay valid.

## Strict versus exploratory training

```python
OVERRIDES["dem"]["require_go"] = True    # STRICT_RESEARCH (default): DEM-based training needs a feasibility GO
OVERRIDES["dem"]["require_go"] = False   # EXPLORATORY_TRAINING: also trains after NO_GO / INCONCLUSIVE
# or, by name (wins over require_go):
OVERRIDES["dem"]["training_mode"] = "EXPLORATORY_TRAINING"
```

| | STRICT_RESEARCH | EXPLORATORY_TRAINING |
|---|---|---|
| feasibility GO | trains, strict-eligible | trains, SCREENING_ONLY |
| NO_GO / INCONCLUSIVE | blocked at section 08 | trains, SCREENING_ONLY |
| BLOCKED_INPUT, no usable tile, invalid CRS, unknown unit, leakage | blocked | **blocked** |
| promotion / submission | possible through every gate | **never** |

The decision is saved with its reason, requirements and corrective action in `dem_catalog/training_gate.json` and displayed in section 07. Thresholds (`feasibility.go_if`) are predeclared: never lower them to get a GO. Changing the mode reuses DEM tiles and datasets; only the DEM-based models retrain, under a new fingerprint.

## Reading the results

* `manifests/execution_summary.csv`: the status table of the session.
* `models/<name>_registry.csv`: every expected fit with status, device, strictness; `<name>_validation.csv`: why a fit is invalid; `<name>_experiment_status.csv`: COMPLETE / PARTIAL / BLOCKED / FAILED.
* `evaluation/metrics/evaluation_status.csv`: FULLY / PARTIALLY / NOT_EVALUATED and STRICTLY_VALIDATED / SCREENING_ONLY / BLOCKED. Only fully evaluated (model, scenario) pairs have a pooled DTI; other rows are in `partial_unit_rows.csv`.
* `reports/DTI_EVALUATION.md`, `reports/NEXT_EXPERIMENT.md`. Offline DTI is a proxy, not a leaderboard score.

## Reusing completed stages

Rerun from the top with the same `CONFIG`. Validated stages show `REUSED_VALID_CACHE`. Resume points:

| product | resumes at | persisted |
|---|---|---|
| coarse DEM parts, fine-tier tiles, coarse windows | next unfinished tile / window | Drive, when each finishes |
| subsurface pool, labels | reused; per-fit labels resume | Drive |
| surface patches | same runtime: last 200-patch mark; new runtime: rebuilt if the memmap exceeded `surface.unet.max_push_gib` | Drive up to that size |
| U-Net | finished fits reused; interrupted fit from its epoch checkpoint | Drive, every epoch |
| LightGBM / Random Forest | per fit | Drive, per fit |

A code change in `dem_features`, `terrain_structure` or `dem_io` rebuilds DEM products only (new `tiles/<fp>/`); the G06 store and the subsurface pool are untouched. Time budgets: `feasibility.pilot_budget_hours`, `dem.fine_budget_hours`, `dem.coarse_budget_hours`. When a budget ends the stage is `PARTIAL`: rerun to continue.

## GPU out-of-memory recovery

The registry row of the fit says what to do. In order: lower `OVERRIDES["surface"]["unet"]["batch"]` (4 -> 2), then `base_channels` (16 -> 12), then restart the runtime and rerun. A changed setting is a new fingerprint: earlier fits stay on Drive but are not reused.

## Debugging a failed stage

| symptom | look at | usual fix |
|---|---|---|
| `TESTS_FAILED` | `audit/pytest_output.txt` | a real defect: report it. These tests have never been run: `run_synthetic_tests=False` only after reading the failure |
| `gpu_preflight FAILED` | `audit/gpu_runtime.json` | select a GPU runtime |
| all tiles excluded | `dem_catalog/dem_invalid_sources.csv`, `dem_source_validation.json` | the codes state the remedy |
| `dem_features BLOCKED_INPUT` | `dem_catalog/training_gate.json` | corrective action in the file |
| `datasets PARTIAL` | `datasets/surface/patch_readiness.csv` | reason per fit (no DEM, disk space, no positive patch) |
| U-Net fit `FAILED` | `models/surface_unet_registry.csv`, column `detail` | OOM guidance or non-finite loss with epoch and step |
| a figure is missing | `manifests/auxiliary_errors.json` | figures never fail a stage |
| any exception | `manifests/runtime_errors.json`, `manifests/stages/<stage>.json` | full traceback |

## Next run: candidate ranking

Only after reading the first results. The ranker's synthetic candidates are now generated per fit from training-visible geometry.

```python
RUN.update(GENERATE_CANDIDATES=True, TRAIN_CANDIDATE_RANKER=True, EVALUATE=True)
```

This also trains the nested inner fits (B1, A2) the strict ranker needs: CPU work, best on a CPU high-RAM runtime with `TRAIN_SURFACE_UNET=False`. Full region and export stay separate:

```python
RUN.update(FULL_INFERENCE=True)       # needs outputs/p6_fast_track/stack/maps/full_score.tif
RUN.update(EXPORT_SUBMISSION=True)    # only after reading evaluation/promotion/promotion_decisions.csv; never uploads
```
