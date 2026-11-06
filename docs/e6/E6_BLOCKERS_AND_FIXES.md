# E6 blockers and fixes

Nothing below was executed. "Fixed" = the source was corrected and re-read; it has not run.

## Fixed in source (2026-11-06)

| ID | file | defect | effect if left | fix |
|---|---|---|---|---|
| F1 | `gems_bench/training/cv.py` | `normalise_stack` turns NaN DEM features into 0 (the robust median) inside covered cells | edge-guarded / partial cells looked like ordinary terrain to the DEM encoder | `dem:dem_coverage` = 0 wherever any DEM feature is not finite; counts saved in `checkpoints/normalisation.json` |
| F2 | `gems_bench/experiment/runners/e6.py` | discovery truth = every label in the evaluation zone | discovery DTI credited the model for reproducing retained training faults | truth = withheld systems; retained faults excluded pixel-exactly (`exclude_supports_tp=False`) |
| F3 | `runners/e6.py` | evaluation scored probabilities, export writes the top-1 % binary map | reported DTI did not describe the submitted file | `e6_dti.csv` has `continuous` and `exact_k` (submission policy) rows, pooled per protocol |
| F4 | `runners/e6.py` | `e6_full_fold<k>.tif` reused whenever present | after a retrain, the submission averaged stale maps | sidecar json (checkpoint sha256 + spec hash); mismatch recomputes |
| F5 | `runners/e6.py` | export never compared the checkpoint's channel list to the inference stack | silent channel misalignment | `spec_k.channels` and DEM fingerprint must equal the runner spec |
| F6 | `runners/e6.py` | final models not checked against the current spec | stale / foreign models exportable | `CheckpointManager(..., spec_hash).final_ok()` for every fold |
| F7 | `runners/e6.py` | written TIFF only format-validated; Drive copy unverified | a corrupted or partial file could be offered for upload | readback equality with `finalise(selection)`; Drive file sha256 compared; `drive_path` recorded |
| F8 | `gems_bench/training/spec.py`, `runners/e6.py` | DEM content outside the spec hash | a rebuilt DEM bank reused old checkpoints | `e6_dem_fingerprint` set from the staged cache in `E6Runner.spec()` |
| F9 | `gems_bench/e6/features.py` | DEM cache identity from absolute paths, no code identity | local vs Drive changed the id; an edited extractor reused stale tiles | tile + bytes + version + code hash |
| F10 | `e6/features.py` | glob picked up `.tmp-` / `.partial` files | an interrupted write used as a DEM part | filtered and listed as rejected |
| F11 | `e6/features.py` | a failing part was only logged | mosaic cached under an identity that includes the missing tile | RuntimeError with the failures (finished tiles kept) |
| F12 | `gems_bench/g07/fast_coarse.py` | edge guard 3 sigma (4 sigma for an 8 sigma operator) | residual one-sided smoothing bias along 10 km part edges | guard = full operator support |
| F13 | `gems_bench/models/geostructurenet_e6.py` | warm start checked only keys / shapes | E1 encoders loaded onto the wrong group if sizes match | `_check_e1_schema`: group lists and order, widths, fusion, long range, no context |
| F14 | `training/trainer.py`, `spec.py`, yaml | baseline fine-tuned after 2 frozen epochs | no-DEM pixels no longer equalled E1 | `e6_finetune_baseline: false`: warm baseline frozen for the whole unit |
| F15 | `training/trainer.py` | resume after the E1 file appeared or vanished | mixed freezing / learning rates in one unit | `e6_warm` in resume state; mismatch raises with the remedy |
| F16 | notebook | `EXP.runner._e6_inputs_ready` after a skipped prepare | AttributeError in a resumed session | `getattr` |
| F17 | notebook, `gems_bench/e6/preflight.py` | no readiness check | blockers found hours into a run | read-only preflight cell 01b, stops on blockers |
| F18 | notebook cell 06 | download without integrity check | wrong / partial file downloaded | readback status + local sha256 required |
| F19 | yaml | `freeze_encoder_epochs`, `e6_freeze_epochs` silently ignored | misleading config | annotated |

## Runtime-only blockers (checked by the preflight; cannot be settled from source)

| ID | artifact | needed by | if missing | action |
|---|---|---|---|---|
| R1 | `dem_cached_10m_parts`: `GEMS_2/outputs/g07_multiscale_fault_discovery/g07-*/dem_features/coarse_parts/*/*.tif` | prepare | **BLOCKED** (E6 never starts native 1 m work) | run the G07 notebook's DEM features stage once (it writes the parts) |
| R2 | `e1_cv_fold0/1/2`, `e1_disc0/1/2`: `GEMS_2/experiments/e1_gsn_parity__b8b8c3db/checkpoints/<unit>_model.pt` | train | DERIVABLE: that unit trains its baseline from scratch (no E1 parity). Fold 2 is absent per the milestone doc | none required; a present file with a wrong schema is BLOCKED |
| R3 | `geodawn_features`, `geodawn_faults`, `geodawn_submission`, `geographic_folds`, `discovery_splits`, `fault_segments` | prepare | BLOCKED | restore from the master pipeline outputs |
| R4 | `base_feature_caches`: `outputs/benchmark/cache/features`, `outputs/phase4/cache/features` | prepare | DERIVABLE (rebuilt from the raster on first use, CPU time unmeasured) | stage from Drive if available |
| R5 | `licensed_external_usgs_dem` export gate (`e6.usgs_dem_license_verified: true`) | export | the value is your statement | confirm the competition's external-data rules before uploading |
| R6 | the corrected code itself | all | unexecuted | the first Colab run is its first execution; read the first stage outputs before leaving it unattended |

Numbers that do not exist yet: runtimes, GPU memory, every DTI. The leaderboard effect of E6 is unmeasured.
