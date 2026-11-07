# GeoRegime V2 recovery: architecture and inputs

**Notebook:** [GEMS_GEOREGIME_RECOVERY_V2.ipynb](../../notebooks/colab/GEMS_GEOREGIME_RECOVERY_V2.ipynb)
**Package:** `gems_bench/georegime_recovery/`
**Status:** authored 2026-11-07; nothing executed.

This milestone is inference-only. It loads the existing GeoRegime V1 checkpoints (`gems_bench/georegime`, commit
f897b84 on `main`), re-infers them, re-scores them, explains the errors and decides whether a predeclared
label-free score transform may be promoted. It does not train, fine-tune, or add an architecture.

## V1 interfaces reused (read from `origin/main` f897b84)

| V1 object | Used for |
|---|---|
| `georegime.data.Bank(atlas_root, cache, scratch)` | Feature schema, grid, the `valid = valid_numerical & template_valid` and `labels = fault_labels & valid` masks, `key` (bank fingerprint), `crop()` (normalisation + clip ±5), `_make_bank` / `_make_context` (rebuild in local scratch only if the V1 cache lacks the bank), `regime_prior` formula, `group_indices` |
| `georegime.data.geo_masks`, `discovery_masks`, `stage_file` | Exact split reconstruction; read-only staging |
| `georegime.model.GeoRegimeNet(nmag, ngrav, nterrain, width)` | Strict `load_state_dict`; `forward` returns `prob`, `global_logits`, `experts`, `router_logits`, `routes`, `specialist_weight`, `proximity`, `orientation` |
| `georegime.run._tile_positions` | Identical tiling |
| `georegime.run._write_submission_masked` | The fixed V1 submission writer, used for the guarded export |
| Checkpoint dict | `epoch`, `model`, `normalizer`, `contract{features, unit, train_mask_sha, config_hash}`, `feature_schema`, `model_type` |
| `runs/<key>/config.json` | V1 `config` (patch 256, stride 192, width 24, epochs 8, buffers 15 / 8 px, AMP) and `bank_fingerprint` |
| `gems_bench.metric` | `dti_components`, `MaskedScorer`, `local_weighted_max`, `clean_prediction`, `pool`, `OFFICIAL_EXCLUDE_SUPPORTS_TP` |
| `geological_atlas.atlas.GeologicalAtlas` | `load_feature`, `load_coverage`, `load_lineaments` (label-free evidence) |

Not called anywhere: `GeoRegimeExperiment.train_unit`, `predict` (its cache key ignores the ROI) and `export`.

## Inputs on Drive (expected; verified by section 02)

```text
GEMS_2/GEMS_GEOREGIME_V1/runs/8dfa3b352417/
  config.json                                   mandatory
  checkpoints/{geo0..2,disc0..2,full}_latest.pt mandatory (all 7)
  checkpoints/*_normalizer.json                 optional (cross-check; the checkpoint carries its normalizer)
  predictions/full_prediction.tif               mandatory (sections 03/04/13/16)
  predictions/{geo*,disc*}_prediction.tif       optional (section 06 reproduction, inference reproduction check)
  predictions/*_prediction.json, *_router_magnetic.tif   optional
  tables/{geographic_dti,discovery_dti}.json, validation_components.csv,
         prediction_mass_diagnostic.csv, *_history.csv   optional (comparison only)
GEMS_2/GEMS_GEOREGIME_V1/cache/{bank,context}_<bank_key>.npy   optional (read-only reuse)
GEMS_2/GEMS_GEOLOGICAL_DATA_ATLAS_V1/
  06_ml_ready/schemas/feature_stack_schema.json + the 5 family stacks   mandatory
  02_geospatial/masks/{valid_numerical,fault_labels,template_valid}.tif mandatory
  00_raw_snapshot/official_template/example_submission.tif              mandatory
  03_features/feature_registry/feature_registry.parquet (+ single-band features)  optional (sections 11, 13, policy 4)
  04_eda/candidate_lineaments/candidate_lineaments.parquet              optional (section 13)
```

## Module map

| module | role |
|---|---|
| `config.py` | Defaults, all thresholds, policies and gates, `resolve` / `validate` (`train_models` must be False) |
| `paths.py` | Output namespace; `check_writable` refuses any path inside the V1 tree or the Atlas |
| `cache.py` | Atomic writes, SHA-256 hash cache, fingerprinted stage records |
| `contracts.py` | `StageResult`, table schemas, error types |
| `inventory.py`, `preflight.py` | Section 02 |
| `split_audit.py` | Section 05 |
| `metrics.py` | Section 06/08 scoring and `ComponentMaps` |
| `inference.py` | Section 07 |
| `prediction_diagnostics.py`, `fault_proximity.py` | Sections 03, 04 |
| `expert_analysis.py` | Section 10 |
| `geological_evidence.py` | Sections 09 (regions), 11 |
| `structural_analysis.py` | Section 12 |
| `candidate_review.py` | Section 13 |
| `transforms.py`, `crossfit.py` | Sections 14, 15 |
| `submission.py` | Section 16 |
| `figures.py`, `reports.py` | Figure groups, interpretations, final report |
| `run.py` | `RecoveryRun`: every notebook call |
