# E6 static audit

Branch `e6-submission-fasttrack` (base commit `326f563`) plus the uncommitted fixes of this audit (2026-11-06).
**Method: source reading only. Nothing was executed** (no Python, tests, notebook, Git, data access). The three
documents the audit brief named as inputs (`E6_ARTIFACT_CONTRACT.md`, `E6_DEPENDENCY_GRAPH.md`,
`E6_IMPLEMENTATION_INVENTORY.md`) do not exist on the branch; the only E6 document was `SUBMISSION_MILESTONE.md`. The
graph below was reconstructed from source.

## A. Source-level trace

| # | caller -> callee | arguments / config | artifact in -> out | failure behaviour | upstream produced? |
|---|---|---|---|---|---|
| 1 | notebook cell 03 -> `git clone --branch e6-submission-fasttrack`, `pip install -e .[colab]`, `runtime.colab.configure`, `DriveSync.stage_in(minimum)` | `REF`, `LOCAL`, `DRIVE`, `CODE` | Drive raw / folds / datasets / eda -> `/content/gems_work` | missing Drive folders skipped (printed) | user's Drive |
| 2 | notebook 01b -> `e6.preflight.run_preflight(config, LOCAL, DRIVE, CODE)` **(new)** | config path | headers only, read-only | raises with every blocker | - |
| 3 | `Experiment.from_yaml` -> `ExperimentConfig.from_yaml` -> `runners.get_runner("e6")` -> `E6Runner` | `kind: e6` | `experiments/<name>__<hash8>/` (staged from Drive if present) | unknown kind -> ValueError | yes |
| 4 | `Experiment.prepare` -> `E6Runner.prepare` -> `Cnn5Runner.prepare` -> `E6Runner.spec()` **(new override)** -> `Cnn5Runner.spec` -> `spec_from_experiment(ctx, model, name, seed)` | `model.base: P4_B_gated`, `add_channels`, `overrides` | `configs/baselines/legacy_specs/P4_B_gated.json` (or the legacy run's `config.json`) | FileNotFoundError if neither | snapshot in repo |
| 5 | `E6Runner.ctx` -> `Runner.ctx` -> `Experiment.ctx` -> `load_context5` -> `load_context4` -> `p3.context.load_context` | folds `outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026`, splits `discovery_d3_link1km_s2026` | GeoDAWN raster, faults, template, folds, splits, segments -> `P5Context` | FileNotFoundError (folds / splits / segments) | Drive (runtime) |
| 6 | `E6Runner.ctx` -> `store.drive_mirror = (root, drive)` | - | feature caches copied on first read | none if absent (store builds) | - |
| 7 | `E6Runner.ctx` -> `e6.features.stage_cached_dem(exp, ctx, max_tiles)` -> `find_parts` **(new)** -> `g07.fast_coarse.extract_fast_features` -> `rasterio.warp.reproject` -> `g07.dem_io.write_window` -> `mosaic_windows(coverage_paths=...)` | `e6.max_cached_dem_tiles` | `outputs/g07_multiscale_fault_discovery/g07-*/dem_features/coarse_parts/*/*.tif` -> `outputs/e6_pipeline/cache/<fp>/{tiles/*.tif,dem_features_100m.tif,manifest.json,tile_log.csv}` | no part: FileNotFoundError; a failed part: RuntimeError **(new)** | G07 run on Drive (runtime) |
| 8 | `E6Runner.ctx` -> `register_e6_dem(ctx, folder, fp, root)` -> `phase5_store.dem_specs` | - | mosaic band names -> `dem:*` specs; `store.p5_dir = outputs/e6_pipeline/features/<fp>` | RuntimeError without `dem_coverage` | yes (#7) |
| 9 | `Cnn5Runner.prepare` -> `training.cv.ensure_channels` -> `Phase5FeatureStore.build` -> `_dem_band` | channel names | mosaic -> `outputs/e6_pipeline/features/<fp>/dem_*.npy`; base channels from phase-2 / phase-4 caches | KeyError for unknown channels; ValueError if mosaic off-grid | yes |
| 10 | `Cnn5Runner.train` -> `cv.prepare_stack5` -> `cnn3.prepare_stack` -> `cnn.normalise_stack` (+ E6 coverage rule **(fixed)**) -> `tensor_stack5` | `norm: robust` | (C,H,W) fp16 stack, `checkpoints/normalisation.json` | - | yes |
| 11 | `run_cv5` / `run_discovery5` -> `losses.compose.build_targets5` -> `train5` -> `models.factory.build_model5` -> `GeoStructureNetE6` -> `warm_start` (+ schema check **(new)**) | `e6_warm_dir`, `e6_finetune_baseline` **(new)** | E1 `<unit>_model.pt` -> E6 `checkpoints/<unit>_{ep*,latest,model}.pt`, `predictions/{oof_fold<k>,disc<d>,oof}.tif` | spec mismatch on resume: RuntimeError; warm/cold mismatch: RuntimeError **(new)** | E1 run on Drive (runtime) |
| 12 | `train5` -> `phase4_training.make_batch4` / `_forward` -> `loss5` (nnPU + Tversky + E6 soft Tversky + proximity + orientation) | `pu: nnpu`, `pu_prior`, `tversky_*`, `aux_*` | - | OOM back-off, collapse guard | - |
| 13 | `CheckpointManager.save / save_final / latest / final_ok` | `ckpt_every`, `ckpt_sync_min` | atomic `.pt` + pointer json + sha256 | corrupt / foreign-spec files refused | yes |
| 14 | `E6Runner.evaluate` **(rewritten)** -> `metric.dti_components`, `g07.representation.exact_k_mask`, `metric.pool` | `e6.submission_fraction` | OOF / disc rasters -> `metrics/e6_dti.csv` | FileNotFoundError per missing unit | yes (#11) |
| 15 | `E6Runner.export` -> `CheckpointManager.final_ok(spec_hash)` **(new)** -> `load_unit_model` -> `inference.full_raster.predict_heads_mask` | `infer_batch` | `cv_fold<k>_model.pt` -> `predictions/e6_full_fold<k>.tif` + `.json` provenance **(new)** -> `e6_full_probability.tif` | incomplete CV / stale models: raise | yes |
| 16 | `export` -> `gems3.submission.eligible_mask` -> `g07.inference.construct_final_prediction("w2_exact" = exact-k of the E6 score)` -> `verify_export_parity` | `submission_fraction` | selection | parity FAIL raises | yes |
| 17 | `export` -> `g07.inference.write_submission_candidate` -> `gems3.submission.finalise` -> `submission.write_submission` -> `gems3.submission.validate` -> readback `verify_export_parity(written=...)` **(new)** -> `Experiment.persist` -> Drive sha256 check **(new)** | gates | `submissions/submission_e6_geostructurenet.tif` + `.json`, `reports/e6_submission_manifest.json` | any gate / validation / readback / Drive check fails: raise | yes |
| 18 | notebook 06 -> checksum + readback check **(new)** -> `files.download(OUTPUT["path"])` | - | the exported E6 file | raises | yes |

Signature checks done by reading both sides: every call in rows 1-18 exists with the arguments passed
(`write_window`, `mosaic_windows(coverage_paths=)`, `analysis_window(pad=)`, `construct_final_prediction(seed=)`,
`verify_export_parity(written=, data=)`, `write_submission_candidate`, `predict_heads_mask`, `load_unit_model`,
`CheckpointManager(spec_hash=)`, `dti_components(exclude_mask=, exclude_supports_tp=)`, `exact_k_mask(width_full=, seed=)`,
`ExperimentConfig.get(dotted)`, `RunDir.save_json/save_table/save_raster/read_raster/raster_path`).

## B. Audit matrix

| Contract | Source file | Required artifact | Status | Defect | Resolution |
|---|---|---|---|---|---|
| Runner dispatch `kind: e6` | `experiment/runners/__init__.py` | - | STATIC_VERIFIED | - | - |
| Spec resolution E6 groups (terrain, subsurface, dem, coverage) | `training/spec.py`, `models/factory.py` | `legacy_specs/P4_B_gated.json` | STATIC_VERIFIED | - | - |
| DEM cache identity covers sources + code | `e6/features.py` | coarse parts | FIXED | identity used absolute paths (local vs Drive changed it) and no extractor code: an edited extractor reused old tiles | tile + bytes + `VERSION` + hash of `fast_coarse.py` / `features.py` |
| Partial / temporary DEM parts never used | `e6/features.py` | coarse parts | FIXED | `*.tif` glob also matched `.name.tmp-…tif` / `.partial` | `_is_temporary` filter; rejected list recorded |
| Failed DEM tile surfaced | `e6/features.py` | - | FIXED | a failing part was logged and the mosaic cached under an identity claiming it | RuntimeError listing the failures |
| Catalog-only DEM never treated as data | `e6/features.py` | - | STATIC_VERIFIED | only readable local / Drive rasters are listed | - |
| Checkpoints bound to the DEM bank | `training/spec.py`, `runners/e6.py` | - | FIXED | spec hash ignored the DEM content: a rebuilt bank reused models trained on the old one | `e6_dem_fingerprint` (semantic) set by `E6Runner.spec` |
| DEM tile seams | `g07/fast_coarse.py` | - | FIXED | guard 3 sigma (one-sided bias ~1 % of slope remains), coherence 4 sigma for an 8 sigma operator | guards = full support (4 sigma, 8 sigma for coherence) |
| Missing DEM never a measurement | `training/cv.py` | - | FIXED | `normalise_stack` sets NaN to 0 = robust median inside covered cells | coverage set to 0 where any DEM feature is not finite |
| Coverage only an availability constraint | `models/geostructurenet_e6.py` | - | STATIC_VERIFIED | not in any encoder; enters as `log m` in the gate and the final `torch.where` | - |
| Zero coverage -> DEM gate weight 0 | `MaskedThreeWayGate` | - | STATIC_VERIFIED | `masked_fill(m<=0, -inf)` before softmax at each of the 5 levels | - |
| Zero coverage -> E1 reference | `geostructurenet_e6.py`, `training/trainer.py` | E1 checkpoints | FIXED | baseline unfrozen after 2 epochs and fine-tuned: uncovered pixels drifted from E1 | `e6_finetune_baseline: false` (default): warm baseline frozen for the whole unit, BN in eval |
| Cold units (no E1 file) | `trainer.py` | - | STATIC_VERIFIED (by design) | no frozen reference exists | parity is with the unit's own 2-way path; recorded as `e6_warm_start` in the final model |
| E1 warm start loads the right encoders | `geostructurenet_e6.py` | E1 `<unit>_model.pt` | FIXED | only key / shape checks: equal-sized groups in another order or a swapped channel loaded silently | `_check_e1_schema` compares group lists, order, widths, fusion, context |
| Resume consistency warm / cold | `trainer.py` | resume files | FIXED | resuming after the E1 file appeared / vanished mixed freezing and learning-rate regimes | `e6_warm` stored in resume state; mismatch raises |
| No withheld labels in DEM preprocessing | `e6/features.py`, `fast_coarse.py` | - | STATIC_VERIFIED | label-free | - |
| Channel order training == inference | `cv.prepare_stack5`, `runners/e6.py` | - | FIXED | export used the runner spec without comparing the checkpoint's | export compares `spec_k.channels` and DEM fingerprint |
| DEM units / scales | `fast_coarse.py` | - | STATIC_VERIFIED | gradients per metre at the part resolution; sigma = 30 / 60 / 150 m | - |
| Potential-field derivative units / FFT scales | phase-2 / phase-4 feature code | caches | RUNTIME_CHECK_REQUIRED | not changed by E6; inherited from E1 | - |
| Axial orientation + augmentation | `phase4_training.make_batch4`, `fast_coarse.py` | - | STATIC_VERIFIED | only 180 deg rotation (axial targets invariant); `hs_range` uses 4 symmetric azimuths | - |
| nnPU non-negative risk | `losses/pu.py` | - | STATIC_VERIFIED | `clamp_min(r_neg, 0)` and `clamp_min(fp_u - prior...)`; positive term unscaled by prior (a scaled variant, non-negative) | - |
| Tversky 300 m, 0.2 / 0.8 | `p3/cnn3.py`, `losses/compose.py` | - | STATIC_VERIFIED | kernel radius 300 m; nnPU surrogate uses config 0.2 / 0.8; E6 term hard-codes 0.2 / 0.8 | - |
| Aux heads only in permitted regions | `losses/compose.py` | - | STATIC_VERIFIED | proximity weighted by `w` (training zone minus ignore band), orientation by `ovalid * w`; targets from visible labels | - |
| Checkpoint reload with architecture + schema | `trainer.load_unit_model` | final models | STATIC_VERIFIED | spec stored in every final model; strict `load_state_dict`; idx buffers rebuilt from spec | - |
| Discovery DTI | `runners/e6.py` | disc rasters | FIXED | truth = all labels in the eval zone: credit for retained training faults | truth = withheld systems; retained faults excluded pixel-exactly |
| Reported DTI describes the exported file | `runners/e6.py` | - | FIXED | evaluation scored continuous maps, export writes a top-1 % binary map | `exact_k` rows with the submission fraction next to `continuous` |
| Full-region fold maps not stale | `runners/e6.py` | `e6_full_fold<k>.tif` | FIXED | reused whenever the file existed, also after retraining | sidecar json with checkpoint sha256 + spec hash |
| Full eligible region incl. no-DEM areas | `runners/e6.py` | - | STATIC_VERIFIED | prediction over `data.valid`; gate `full_region_prediction` requires all finite | - |
| No W2 / B0 map exported | `runners/e6.py` | - | STATIC_VERIFIED | only `e6_full_fold*` maps; `"w2_exact"` is the exact-k policy name, applied to the E6 score | - |
| Continuous values | `runners/e6.py` | - | STATIC_VERIFIED (documented) | export is binary exact-k, predeclared fraction, no search: recorded as `representation` | - |
| Template grid + nodata | `gems3/submission.py` | `example_submission.tif` | STATIC_VERIFIED | `finalise`: NaN where template NaN, 0 inside template outside data | - |
| Written raster reopened | `runners/e6.py` | submission | FIXED | only `validate` (format); values never compared | readback must equal `finalise(selection)` |
| Drive copy + checksum | `runners/e6.py`, notebook | submission | FIXED | not verified | Drive sha256 compared; notebook re-checks local sha256 before download |
| Notebook after a skipped prepare | notebook | - | FIXED | `EXP.runner._e6_inputs_ready` AttributeError | `getattr(..., False)` |
| Preflight | `e6/preflight.py` **(new)**, notebook 01b | all | FIXED | none existed | read-only report, stops on blockers |
| GeoDAWN raster / faults / template | `p3.context` | Drive | RUNTIME_CHECK_REQUIRED | - | preflight |
| Folds / splits / segments | `p3.context` | Drive | RUNTIME_CHECK_REQUIRED | - | preflight |
| Base feature caches | `features` | Drive | RUNTIME_CHECK_REQUIRED | absent ones are rebuilt from the raster (CPU time) | preflight reports DERIVABLE |
| G07 10 m parts | `e6/features.py` | Drive | RUNTIME_CHECK_REQUIRED | - | preflight |
| E1 checkpoints | `trainer.py` | Drive | RUNTIME_CHECK_REQUIRED | fold 2 absent per milestone doc | preflight (schema + order) |
| Export gate `licensed_external_usgs_dem` | yaml | - | RUNTIME_CHECK_REQUIRED | value is the user's statement, not verified by code | user decision |
| `freeze_encoder_epochs`, `e6_freeze_epochs` | yaml | - | FIXED | ignored / meaning changed | annotated in the yaml |
| `register_e6_dem` moves `store.p5_dir` | `e6/features.py` | - | STATIC_VERIFIED (risk noted) | every `p5:` channel would follow the new folder; the E6 base channels contain none | do not add `p5:` channels to E6 without changing this |

Failure classes the brief listed, and where they are now caught: FileNotFoundError (preflight + explicit messages in
#5, #7, #14, #15), EmptyDataError (no empty registry is read: `fold_metrics` is only read after CV wrote it),
AttributeError (#18 notebook fix; every called method exists), KeyError (`ensure_channels` names unknown channels),
TypeError (signatures checked in A), interrupted Drive (atomic writes, ledger, Drive sha256 at export), incomplete folds
(export refuses), channel order (#15), checkpoint keys (strict load + E1 schema check), stale fingerprints (DEM identity
+ spec fingerprint + fold-map provenance), valid-geometry-but-wrong submissions (readback + exact-k scoring).

## Final acceptance statement

```text
STATIC INTEGRATION:
  verified (after the fixes listed above; source reading only)

ARTIFACT CONTRACT:
  fully traced (runtime existence of the Drive inputs is checked by the preflight, not by this audit)

ARCHITECTURE PARITY:
  established statically for warm-started units (frozen E1 baseline, exact torch.where fallback);
  requires runtime evidence for numerical equality and for cold units, which have no E1 reference

TRAINING:
  implemented

INFERENCE:
  implemented

SUBMISSION:
  implemented

RUNTIME VALIDATION:
  NOT EXECUTED

COLAB READY:
  READY FOR RUNTIME PREFLIGHT

REMAINING BLOCKERS:
  see E6_BLOCKERS_AND_FIXES.md: R1 dem_cached_10m_parts, R2 e1_cv_fold2 / e1_disc*, R3 geodawn_*, folds,
  segments, R4 base_feature_caches, R5 licensed_external_usgs_dem (user statement), R6 unexecuted code
```
