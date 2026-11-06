# E6 V2 session handoff

Update this file after each phase (date, what ran, evidence, next step).

## 2026-11-06 - phase 0: code authored, nothing executed
- Written: `gems_bench/e6_v2/` (33 modules + `__init__.py`), `configs/e6_v2/e6_v2.yaml`,
  `notebooks/colab/GEMS_E6_V2_FULL_PIPELINE.ipynb` (sections 00-16), `docs/e6_v2/` (9 docs).
- Executed: **nothing** - no local Python, no tests, no notebook, no Colab. No artifact in
  `GEMS_2/e6_v2_from_raw/` exists yet. No DTI, runtime or memory number exists.
- Not committed: the files are in the working tree of `main`. The brief asked for branch
  `e6-geostructurenet-from-raw`; git was off limits for this session, so create it yourself:
  `git switch -c e6-geostructurenet-from-raw && git add gems_bench/e6_v2 configs/e6_v2 docs/e6_v2
  notebooks/colab/GEMS_E6_V2_FULL_PIPELINE.ipynb && git commit && git push -u origin e6-geostructurenet-from-raw`,
  then set `REF = "e6-geostructurenet-from-raw"` in notebook cell 00 (it clones `main` by default).

## Decisions taken while authoring (check them)
1. TMI-family magnetic bands are excluded (orientation-dependent; see RAW_DATA_CONTRACT.md).
2. DEM 2 m rasters are not kept for the whole footprint (estimated tens of GB); only `dem.qa_windows` = 3 windows
   keep 2 m mosaic + feature GeoTIFFs for inspection. `cache/dem/mosaic/` holds the window plan and overlap order.
3. DEM tile checksums are partial (size + first / last MiB).
4. Normalisation statistics use all footprint inputs (label-free), identical for every fold.
5. nnPU prior = training-visible fault share x 1.5 (assumption, config `prior_inflation`).
6. DEM dropout only in stage E (in D it would carry no gradient).
7. Export default `area_budget_binary`; the official continuous-vs-binary rule was not verified.

## Next step
Push the code, run sections 00-03 and read `reports/preflight_report.md`. Expected first blockers: DEM tiles
without determinable vertical unit (if the XML metadata was not downloaded), missing raw files.

## Open risks (first execution of all code)
- Every module is unexecuted; expect first-run errors (typos, API details of rasterio `WarpedVRT`, shapes).
- DEM stage runtime unknown (about 2000 windows estimated from 37.9 % coverage; seconds per window unmeasured).
- GPU memory for batch 8 x 256 px with three encoders unmeasured; lower `training.batch_size` and raise
  `grad_accum` if it does not fit (both change the training identity - do it before the first training run).
