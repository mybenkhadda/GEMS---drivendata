# GeoRegime V2 recovery: session handoff

## 2026-11-07, phase 0: code authored, nothing executed
**Written:**
* `gems_bench/georegime_recovery/`: 22 files (21 modules + `__init__.py`).
* `notebooks/colab/GEMS_GEOREGIME_RECOVERY_V2.ipynb`: the skeleton's 37 cells in the original order. Only cell 01 (bootstrap) is filled in.
* `docs/georegime_recovery/`: 11 docs.

The skeleton file `GEMS_GEOREGIME_RECOVERY_V2_SKELETON.ipynb` is unchanged.

**Executed: nothing.** No local Python, tests, notebook or Colab run. No recovery folder exists on Drive; no metric, figure or decision exists yet. Not committed.

**Sources read:**
* GeoRegime V1 (`gems_bench/georegime/{model,loss,data,run,__init__}.py`, `docs/georegime/README.md`, the V1 notebook) from GitHub `main` at f897b84 via raw.githubusercontent. The local checkout at 50effba does not contain `gems_bench/georegime/`, so `git pull` before editing.
* `gems_bench/metric.py` (unchanged since faec7ba) and the Atlas modules from the local checkout, identical to `main`.

## Key facts for a fresh session
* **Checkpoint contract:** `train_mask_sha = sha256(train.tobytes())[:12]`, `features = bank.key`, `config_hash = sha256(json.dumps(cfg, sort_keys=True))[:12]`, `epoch + 1 == epochs`. Section 05 compares against the first; preflight checks the others.
* **V1 `predict` defect:** the cache key has no ROI. It is never called; `inference.py` replicates its tiling exactly with ROI = `bank.valid`.
* **Primary full-area exclusion** = training-visible labels (labels & train); the sensitivity variant excludes all non-held labels.
* **Fixed in config before results:** policies, gates, thresholds.

## Decisions taken while authoring (check them)
1. The full model is re-inferred too (unit `full`). This gives its heads for section 10 and a reproduction check against `full_prediction.tif`. The export uses the original V1 raster, not the re-inference.
2. `full_prediction.tif` is mandatory at preflight. Fold V1 rasters are optional; they are needed for section 06 and the reproduction gate.
3. Normalizer JSONs are optional. The checkpoint's embedded normalizer is authoritative, and a differing JSON fails preflight.
4. Gate thresholds: +3% relative pooled gain, at most 5% loss on any fold, unanimous choice, at most 5% geographic loss, bootstrap 90% CI lower bound > 0, label-free, provenance.
5. `geo_support` S = robust rescale of the mean of `xm_evidence_mean` and `xm_multiscale_persistence`. Both are label-free, but `xm_multiscale_persistence` is also a V1 input channel.
6. Expert verdict thresholds are in `config["experts"]`.

## Next step
Push to `main`, then run sections 00–05 on a GPU runtime. Read `00_inventory/preflight.json` and `01_split_audit/checkpoint_split_validation.csv`. If all `train_mask_sha` match, continue with section 07.

## Open risks (first execution of all code)
* Runtime errors are expected on the first run. The likeliest sources:
  * rasterio window / rasterize details (candidate review);
  * pandas groupby / apply shapes in figures;
  * matplotlib layout.
* The bank key depends on the Atlas stack mtimes. If the Atlas was re-run after V1 training, preflight fails with "bank fingerprint differs". That is correct behaviour, not a bug.
* Inference time and memory are unmeasured. Accumulators are about 350 MB in RAM, and the snapshots are compressed npz files of similar uncompressed size.
* AMP nondeterminism could exceed `reproduction_tol = 0.01`. In that case inspect `p999_abs_diff` before changing the tolerance.
* `torch.load(weights_only=False)` is required by the V1 checkpoints (numpy RNG state); only the user's own files are loaded.
* The style linter reports long lines and import order; these are not functional issues.
