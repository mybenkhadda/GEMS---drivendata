# G06 session handoff

## State (2026-11-02)

* Code rev `g06.1-r1` authored. **Nothing executed**: no Python, test, lint, notebook or data step was run locally;
  no metric in this repository comes from G06. Static checks only (VS Code diagnostics: ruff import-format / line
  length fixed in the files it analysed; remaining items are informational bandit notes).
* Uncommitted on `main`: `gems_bench/g06/` (24 modules), `configs/g06/{feature_store,benchmarks}.yaml`,
  `tests/test_g06_*.py` (10), `notebooks/colab/GEMS_G06_GEOLOGICAL_FEATURE_STORE_AND_ML_BENCHMARK.ipynb`,
  `docs/g06/` (9).

## Next actions (in order)

1. Commit on a branch, push, pin `CONFIG['ref']`.
2. Colab first run (COLAB_EXECUTION_GUIDE.md section 1). Expect the `tests` stage to surface bugs of this unexecuted
   code first (`audit/pytest_output.txt`).
3. Check `inventory/band_inventory.csv`: 19 identifiers RESOLVED; correct the declared units / transforms if the
   descriptions disagree (changes the run id - intended).
4. Feature store: check `features/feature_coverage.csv`, the effective halo (expected ~124 px with the default
   1500 m structural scale), unusable channels, the seam diagnostic (section 10), memory / time per tile.
5. Second run: RF + LightGBM + EVALUATE. Read `reports/MODEL_BENCHMARK.md` and `evaluation/ablations.csv`.
6. Then segment, meta (and optionally U-Net, full inference) per `reports/NEXT_EXPERIMENT.md`.

## Decisions (implementation)

* Run directory keyed by the data-engineering identity (feature-store config + source sha + version + seed); every
  stage cached by its own fingerprint incl. the source hash of its modules; replaced outputs archived to `history/`.
* One tiled pass computes all tiled groups with NaN-padded halo windows and normalised convolution -> tile == whole
  raster (tested synthetically). FFT features are whole-raster (one band in memory).
* One label-independent pixel pool, relabelled per visibility; label-stratum rows on hidden faults are dropped.
* Pixel models never get known-fault context (it would encode the target). Segment / meta models use per-unit
  permitted context and nested context for training units.
* Prediction / model paths are fingerprint-keyed; registries are merged across runs.
* Notebook sections 06-09 preview the feature modules on one real central window; section 10 builds the store.

## Scientific assumptions to verify

* Band units / transforms in `feature_store.yaml` are declarations from names, not verified metadata.
* V1 training region = whole footprint (hidden faults unlabelled at their natural rate) is the deployment analogue;
  `outside_eval_zone` is the stricter alternative if V1 looks optimistic relative to V3.
* Footprint-level label-free statistics (entropy bins, strain scale, domains) are transductive.
* Default segment pools include W2 / CNN -> S1 is SCREENING_ONLY unless pools = {physics, g06}.
* Truncation / curvature features are proxies, not displacement.

## Known risks / limitations

* Runtime / memory of the feature store (~300 channels, several GB compressed) and of the 19 pixel fits per
  nested experiment; disable groups or the nested list if needed.
* `geomoe.scenarios.build` always builds V2 (filtered afterwards); V2 needs `configs/geomoe` `v2` section.
* Full-region segment inference (`inference.predict_full_segments`) is implemented but not wired into a stage.
* Export works only for pixel experiments (they are the only ones with a full-region raster); meta / segment models
  can reach PROMOTE_CANDIDATE but need a full-region map before any export.
* G01 / G02 / G05 outputs are inventoried only (not used as features).

## Disclosure

No shell command, script, test or notebook was executed while authoring G06 (read-only file inspection only).
