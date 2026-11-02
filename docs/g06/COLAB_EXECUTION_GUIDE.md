# G06 Colab execution guide

Open `notebooks/colab/GEMS_G06_GEOLOGICAL_FEATURE_STORE_AND_ML_BENCHMARK.ipynb`, set `CONFIG['ref']` to the pushed
commit SHA, run sections 01-02 and the init cell, then the sections of the run. Keep the same `CONFIG` (same run id)
between runs: every validated stage is reused (`REUSED_VALID_CACHE`). Only the keys that change are shown; all other
`RUN` flags as in the notebook defaults.

## 1. Feature preparation (first run, CPU, high-RAM runtime recommended)

```python
CONFIG.update(use_gpu=False, reuse_cached=True)
RUN.update(PREFLIGHT=True, RAW_EDA=True, FEATURE_ENGINEERING=True, FEATURE_ATLAS=True, BUILD_FEATURE_STORE=True,
           BUILD_SPLITS=True, BUILD_PIXEL_DATASET=True, BUILD_SEGMENT_DATASET=True, REPORT=True)
# every TRAIN_* / EVALUATE / FULL_INFERENCE / EXPORT_SUBMISSION = False
```
Sections 03-15, 25, 26. The feature store is the long step; after a disconnect re-run from the top: finished tiles
and groups are reused. Read `inventory/band_inventory.csv` (exact bands, declared units) and
`reports/FEATURE_REPORT.md` before training.

## 2. Classical model training (second run, CPU)

```python
RUN.update(TRAIN_RANDOM_FOREST=True, TRAIN_LIGHTGBM=True, EVALUATE=True, REPORT=True)
# data flags may stay True (validated cache adopted) or be False (cache adopted when valid)
```
Sections 03, 16-18, 22, 23, 25. Runs G06-P0 (RF) and G06-P1..P5 (LightGBM) incl. the nested fits of G06-P5.
Optional: `TRAIN_XGBOOST=True` adds G06-P5X.

## 3. Segment training

```python
RUN.update(TRAIN_SEGMENT_MODEL=True, EVALUATE=True)
```
Section 19 then 22. Uses the G06-P5 predictions (nested for training units) when present. For a strict S1 set
`OVERRIDES['benchmarks'] = {'segment_candidates': {'pools': {'w2': None, 'cnn': None}}}` (a null pool is disabled;
the physics pool stays) - this rebuilds the segment dataset (new segment_dataset fingerprint), not the feature store.

## 4. Optional patch segmentation (GPU runtime)

```python
CONFIG.update(use_gpu=True)
RUN.update(TRAIN_PATCH_UNET=True, EVALUATE=True)
```
Section 20 then 22. Writes the channel memmap once (`datasets/patches/stack_<h>.npy`).

## 5. Meta-model training

```python
RUN.update(TRAIN_META_MODEL=True, EVALUATE=True)   # needs G06-P5 (TRAIN_LIGHTGBM run first)
```
Section 21 then 22. Read `evaluation/complementarity.csv` (systems recovered by G06 but not W2).

## 6. Full-region research inference

```python
RUN.update(FULL_INFERENCE=True)          # EXPORT_SUBMISSION stays False unless a strict PROMOTE_CANDIDATE exists
```
Section 24. Writes `predictions/research_geotiffs/G06-P5_<fp10>_probability.tif` (ROLE=RESEARCH_NOT_A_SUBMISSION)
and the parity check. An export is written only when the promotion decision, parity and `EXPORT_SUBMISSION` all
allow it, and is never uploaded.

## Troubleshooting

* `preflight FAILED: band ... MISSING / DUPLICATE`: fix the identifiers in `configs/g06/feature_store.yaml`
  (exact match; near misses are listed but never used).
* `splits BLOCKED_UPSTREAM` / engine not attached: the submission-factory preflight failed
  (`outputs/leaderboard_focus/submission_factory/reports/PREFLIGHT.md`); P6 W2 maps and the folds must exist on Drive.
* `TESTS_FAILED`: read `audit/pytest_output.txt`; training stages are blocked until fixed.
* Memory: lower `tiling.tile_px` (feature store) or disable groups in `feature_store.yaml` (new run id).
