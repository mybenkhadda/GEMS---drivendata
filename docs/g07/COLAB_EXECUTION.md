# G07 Colab execution

Pin `CONFIG['ref']` to the pushed commit; keep the same `CONFIG` between runs (same run id: validated stages are
reused). Only changed `RUN` keys are listed; everything else stays at the notebook defaults.

## Run 1 - DEM feasibility (CPU, high-RAM)

Prerequisites on Drive: `data/raw/geodawn/*` (raster, faults, template), the P6 workspace (`outputs/p6_fast_track`),
the folds / splits, the DEM URL list and/or tiles under `data/raw/dem_1m/` (`DEM_DATA_CONTRACT.md`).
```python
RUN.update(PREFLIGHT=True, DEM_INVENTORY=True, DEM_COVERAGE=True, DEM_FEASIBILITY=True, REPORT=True)
# optional pilot download (guarded): OVERRIDES = {'dem': {'download': {'enabled': True, 'max_tiles': 10, 'max_gib': 6}}}
```
Sections 03-07, 29-30. Read `reports/DEM_FEASIBILITY.md`.

## Run 2a - after GO: DEM features + datasets + surface classical

```python
RUN.update(DEM_FEATURES=True, BUILD_DATASETS=True, TRAIN_SURFACE_CLASSIC=True, TRAIN_SUBSURFACE=True, EVALUATE=True)
```
## Run 2b - after NO_GO / BLOCKED_INPUT: geology path

```python
RUN.update(BUILD_DATASETS=True, TRAIN_SUBSURFACE=True, EVALUATE=True)
```
Requires a COMPLETE G06 feature store (set `CONFIG['g06_run_dir']` if several exist).

## Run 3 - candidates + ranker + portfolio

```python
RUN.update(GENERATE_CANDIDATES=True, TRAIN_CANDIDATE_RANKER=True, EVALUATE=True)
```
## Run 4 (conditional) - U-Net (GPU runtime)

```python
CONFIG.update(use_gpu=True); RUN.update(TRAIN_SURFACE_UNET=True, EVALUATE=True)
```
## Run 5 - full region (+ export only if promoted)

```python
RUN.update(FULL_INFERENCE=True)                    # needs outputs/p6_fast_track/stack/maps/full_score.tif
RUN.update(EXPORT_SUBMISSION=True)                 # only after reading evaluation/promotion/promotion_decisions.csv
```
Export writes `submissions/g07_<model>.tif` only when every gate passes (`reports/SUBMISSION_ELIGIBILITY.md`). It is
never uploaded; upload manually after review.

## Troubleshooting

* `dem_feasibility BLOCKED_INPUT`: no usable local DEM / unknown vertical unit / licence flag - see the report.
* `datasets FAILED: G06 store`: run the G06 notebook first-run configuration (feature store) or set `g06_run_dir`.
* `TESTS_FAILED`: read `audit/pytest_output.txt`; training stages stay blocked.
* Memory in DEM processing: lower `dem_tile_px` or raise `dem.work_res_m` (changes the run id; tiles are recomputed).
