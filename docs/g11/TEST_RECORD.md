# Test record

## 2026-11-11: recovery-first pipeline (local, synthetic fixtures)
**Environment:** scratchpad venv, Python 3.14, torch CPU.

**Last complete full-suite run** (before the final fix): every test passed except one.
- The failure was `test_checkpoint_restoration_regenerates_missing_prediction`. A regenerated probability map left a stale sha256 in its training stage record, so the model was retrained instead of reused.
- The bug was fixed in `pipeline.predict_all`: the stage record is now refreshed after regeneration.
- The targeted re-run of that test after the fix did not print its pass/fail line.
- **The final full-suite re-run was stopped at the user's request ("skip tests").** The suite is therefore not confirmed green on the final code.
- `ruff --select F,E9`: clean on the last checked revision.

**Covered by the new `test_recovery.py`:**
- fresh versioned dataset finalization;
- changes to split, loss and hyperparameters do NOT invalidate features (files untouched);
- the same version with another spec is refused;
- a resolution change creates a new version that computes only the DEM-class layers, with the 2 m layers kept separate;
- a truncated layer or `.partial` file is INCOMPLETE, and only that layer is recomputed;
- legacy layers without a completion manifest are reconstructed (RECOVERABLE), with 0 recomputed;
- duplicates and orphans are classified;
- changed content, and an unverifiable legacy config, are rejected;
- DEM engine: an interrupted chunk run resumes; assembly matches the direct aggregation; a corrupted chunk is detected; invalid chunks are rejected;
- split v2 partitions are all non-empty, and an empty test partition is gated;
- a split change reuses the features and namespaces the models;
- checkpoint restoration: a lost map is regenerated, and a config change archives the old checkpoints;
- a submission on the wrong grid is refused;
- the notebook adapter's modes and keys.

**Full-scale split simulation** (synthetic faults on the real 3730 × 3292 grid, 20 km blocks, 6.4 km embargo; script `split_scale_sim.py` in the scratchpad):
- v1 reproduced the real failure: train 7,482,723, validation 30,017, stack 22,159, test 35,730 px.
- v2 passed every gate in 17 s: train 2.69 M, validation 1.22 M, stack 1.02 M, test 1.23 M px, each with ≥ 1,723 fault px.
- The embargo took about 28 % of the area.

**Not executed:** anything on the real data.

## 2026-11-10 (b): Colab master execution system (local, synthetic fixtures)
**Environment:** scratchpad venv, Python 3.14.0, numpy 2.5.4, rasterio 1.5.2, geopandas 1.2.0, xgboost 3.4.1, lightgbm 4.7.0, torch 2.15.0+cpu.

**Command:** `PYTHONPATH=. python -m pytest gems_bench/g11/tests -q`

**Result: 102 tests, 97 passed, 5 skipped.** The skipped tests are `test_real_data.py`, which needs `G11_REAL_DATA_ROOT`; Drive is not mounted locally. `ruff --select F,E9`: clean.

| new / changed test | covers |
|---|---|
| test_colab_infra | runtime detection and configuration (scientific parameters unchanged, adjustments recorded); memory estimates labelled ESTIMATE; stage tracker (COMPLETE / FAILED, peak RAM); runtime report; notebook CONFIG adapter (unknown key, auto-upload, bad buffer / resolution / model / contradictory flags rejected; SMOKE plan; model filters); vertical units (band units, tag, **compound EPSG:26911+5703 read via GTIFF_REPORT_COMPD_CS**, US-survey-foot scale, CONFLICTING, UNKNOWN); duplicate tiles across folders; G07-style `.npy` placed via an adjacent `manifest.json`, exact 2×2 replication, support preserved, fold-masked vs full-domain provenance, UNLOCATED and non-nesting arrays excluded; SMOKE window and cropped reads; fusion (all 7 methods, fit on stack, selected on validation); local staging with checksum |
| test_dem | + chunked == single-chunk aggregation; local 25 m mosaic equals direct reads, nodata stays nodata |
| test_pipeline | full stage chain with the **new layout and outputs**: E0/E1/E2 + E5 fusion + leaderboard columns + candidates + `g11_best_submission.tif` / `submission_manifest.json` / `submission_validation.json` + `official_DTI = unavailable` + final report; **SMOKE_TEST plan end to end** on a cropped grid (audit window, ingest, features incl. DEM, EDA, splits, E1, **E4 gated fusion through the local DEM mosaic**); E3 SegFormer + `bce_boundary` ablation + cached resume |
| test_real_data | LEVEL 2 on the real Drive data, opt-in: discovery and alignment, DEM window, units table, features, splits, E1, E3 + restoration, evaluation, candidates, submission validation |

**Bugs found and fixed during this round:**
- the `bce_boundary` loss had no distance target in the patch datasets;
- training crashed when validation patches had no positive pixel (PR-AUC is now NaN, and the last checkpoint is used);
- empty external-prediction tables crashed the EDA;
- `train([])` trained every experiment;
- the vertical CRS of USGS tiles was invisible without `GTIFF_REPORT_COMPD_CS`;
- E4 re-read 1 m Drive tiles for every patch (now served from a local 25 m mosaic).

**Not executed:** anything on the real data. The real-data tests are written but have not run.

## 2026-11-10 (a): initial G11 package
86 tests passed, as recorded in the first G11 session.
