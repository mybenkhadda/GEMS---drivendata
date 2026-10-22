# Next-session handoff (2026-10-22)

## What this session added

A single resumable Colab entry point that no longer depends on hand-uploaded Phase 1–4 outputs.

* `notebooks/colab/GEMS_MASTER_PIPELINE.ipynb`: the recommended start (23 sections, control panel, mode-aware cells).
* `gems_bench/pipeline/`: `config` (validated schema), `registry`, `dependencies`, `manifests`, `executor`,
  `context`, `stages`, `baselines`, `eda_rebuild`, `experiments`, `status`.
* `gems_bench/data/acquisition.py`: resumable, verified, URL-free-on-disk acquisition of the raw files and the DEM list.
* `gems_bench/artifacts/validation.py`: content-level validators (GeoTIFF, grid alignment, Parquet, JSON, npy,
  fold and split folders, EDA groups, legacy runs).
* `configs/pipeline/master.yaml`; `configs/baselines/legacy_specs/*.json`, which are training-config snapshots of the
  legacy baselines (recipes only, no results).
* Compatibility fixes in the library (all backward compatible):
  * `p3.context.P3Context.oof` raises `FileNotFoundError` for a missing Phase 2 map, so `has_oof` returns False
    instead of crashing.
  * `load_context` / `load_context4` / `load_context5` accept `fold_dir` / `disc_dir`, for versioned validation designs.
  * `training.spec.base_spec_dict` falls back to the snapshot. `cnn5` experiments (B, C, E) no longer need
    `outputs/phase4/experiments/p4_P4_B_gated_p256_s2026/config.json`.
  * `ZooRunner._spec` uses `base_spec_dict`.
  * `gems pipeline plan|run|status` CLI.
* Docs: `MASTER_COLAB_GUIDE.md`, `PIPELINE_DEPENDENCIES.md`, `ARTIFACT_RECOVERY.md`; README and the Drive layout updated.
* Tests (authored, **not run**): `tests/test_pipeline_engine.py`, `tests/test_pipeline_artifacts.py`.

## Nothing was executed

No Python, notebook, test, download or training ran. Expected first-run work:

1. Run `pytest tests/test_pipeline_engine.py tests/test_pipeline_artifacts.py` locally, then the full suite.
2. On Colab, run the master notebook with `run_mode='SETUP_ONLY'`, then `PREPARE_DATA`.

Fix first-run errors in place, keeping config hashes stable where possible.

## Ready to execute (in order)

1. `SETUP_ONLY`: verifies Drive, GPU, and the raw files. The observed Drive has `data/raw/example_submission.tif` and
   `data/raw/tnm_items.json` at non-canonical paths; both are accepted.
2. `PREPARE_DATA`: minimal EDA rebuild (`outputs/eda` is missing on Drive), segment rasters, reuse of the
   regenerated folds and splits, feature store (raw + Phase 2 + `eda:*` + RS features used by baselines),
   DEM inventory.
3. `BASELINES` with `train_baselines=['B0_lineament', 'G1', 'G3']` (CPU). Then add `RS05_G3_noSS_RS` (CPU), and on a
   GPU `CNN_A_dti`, `DL00_CNN_B_dti`, `P4_B_gated`. Alternatively upload the legacy run folders
   (`docs/ARTIFACT_RECOVERY.md`) after checking that the fold files are identical.
4. `ADVANCED_RESEARCH` with `run_advanced_models=True` on a GPU: b0 → b1, b2 → c1 → e1.

## Unverified integration points (check on first run)

* `gems_eda` rebuild through a redirected `EDAPaths` subclass (`eda_rebuild._rebuild_paths`): `Manifest` relative
  paths, and `s3_profiles` / `s5_crossmodal` memory use on the 2048 px grid.
* `segments.segment_id_raster` needs geopandas (`[colab]` extra) and the rebuilt GeoPackage CRS.
* Retraining `CNN_A_dti` through `replace(ctx, paths=ctx.p3paths, ...)` on a `P5Context`.
* `competition_scoreboard` / `recovery_matrix` when the reference model `G1` is not available (scoreboard of a
  partial zoo).
* DEM footprint GeoPackage writing several layers (one per UTM zone) with pyogrio.
* `SubmissionRunner` supports `cnn5` and LightGBM champions only. Legacy `cnn3` / `cnn4` zoo champions raise
  `NotImplementedError` (pre-existing limitation).

## Compatibility risks

* If the regenerated folds on Drive differ from the historical local folds (sklearn KMeans version), legacy OOF
  maps and new runs are not comparable. Compare `fold_map.tif` sha256 values before uploading legacy runs.
* Experiments are opened with `resume_policy="config"`, so a newer commit does not block a resume (it is recorded
  as a warning).

## Exact next milestone

Bootstrap a fresh Drive to a valid `feature_store` with `PREPARE_DATA`. Then obtain verified baselines (G1, G3,
B0) and the first unified scoreboard. After that, run the A1 gated-encoder pseudo-discovery and the B / C
comparisons against B0 on a GPU.
