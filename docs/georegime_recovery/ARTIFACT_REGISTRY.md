# Artifact registry

Root: `/content/drive/MyDrive/GEMS_2/GEMS_GEOREGIME_RECOVERY_V2/`. Every file is written atomically (`.partial` → `os.replace`). Tables are written as CSV and Parquet according to `save_csv` / `save_parquet`.

Required columns per table are listed in `contracts.TABLE_SCHEMAS` and checked at write time; extra columns are allowed. Stage records live in `manifests/<stage>.json` and hold the fingerprint, outputs (path, size, sha256), stats and input tables. Notebook results are saved as `manifests/<stage>_result.json`.

| path | producer | key columns / content |
|---|---|---|
| `00_inventory/source_inventory.csv` | preflight | role, key, path, required, exists, size, sha256, status (OK / MISSING / OPTIONAL_MISSING / INACCESSIBLE; checks OK / FAILED / WARN), detail |
| `00_inventory/source_fingerprints.json` | preflight | per-file size / mtime / sha256, bank key, checkpoint sha, train_mask_sha, normalizer hash, code commit / hash |
| `00_inventory/preflight.json` | preflight | status, failures, limitations, CUDA, bank summary, V1 prediction metadata |
| `01_split_audit/checkpoint_split_validation.csv` | audit | unit, protocol, train_mask_sha (reconstructed / checkpoint), match, pixel counts, adjacency counts |
| `01_split_audit/split_overlap.csv`, `holdout_fragmentation.csv`, `original_masks_manifest.json`, `masks/<unit>_masks.tif` | audit | see SPLIT_AUDIT.md |
| `02_full_inference/<unit>_full_prediction.tif`, `experts/<unit>_{global,expert0,expert1}.tif`, `router/<unit>_{route0,specialist_weight}.tif` | inference | float32, NaN outside the footprint, deflate |
| `02_full_inference/inference_manifests/<unit>_full_area.json`, `inference_artifacts.csv` | inference | identity, key, outputs + sha256, reproduction check |
| `03_validation/original_metrics.csv`, `original_pooled.csv`, `scorer_settings.json` | section 06 | unit, protocol, source, dti, tp_w, fp_w, fn_w, n_truth, pred_mass, recall_w, precision_w, roi_status, dti_v1_table, abs_diff_* |
| `03_validation/full_area_metrics.csv`, `pooled_metrics.csv` | section 08 | + exclusion, eval_px, excluded_px, fp_share_of_denominator; pooled: n_units |
| `03_validation/regional_metrics.csv`, `block_metrics.csv`, `error_regions.csv`, `fp_by_visible_distance.csv`, `fault_normal_profiles.csv`, `tp_fp_fn_spatial/*.tif` | section 09 | see GEOLOGICAL_ERROR_ATLAS.md |
| `03_validation/fault_length_recall.csv`, `length_class_recall.csv`, `linearity_recall.csv`, `strike_agreement.csv`, `strike_stats.csv`, `regional_recall.csv`, `strike_samples.csv` | section 12 | |
| `04_expert_analysis/expert_{similarity,diversity,performance,environment,summary}.csv`, `routing_statistics.csv` | section 10 | see EXPERT_DIAGNOSTICS.md |
| `05_geological_evidence/geological_disagreement.csv`, `flag_frequency.csv`, `evidence_class_profiles.csv`, `confidence_vs_evidence.csv`, `evidence_availability.csv` | section 11 | |
| `05_geological_evidence/candidate_model_agreement.csv` | section 13 | candidate_id, length_km, label_free_support, model_mean_p, model_pct_rank, agreement_class, ... |
| `06_postprocessing/policy_registry.json`, `geo_support_S.{tif,json}`, `transform_{components,blocks,regions,response_curves}.csv` | section 14 | policy, unit, protocol, DTI components |
| `06_postprocessing/crossfit_results.csv`, `promotion_gates.csv`, `crossfit_regional.csv`, `crossfit_bootstrap.csv`, `recovery_decision.json` | section 15 | fold, chosen_policy, dti_other_folds, dti_heldout_chosen, dti_heldout_identity, delta_dti |
| `07_figures/<folder>/<NN>_<name>.{png,svg}`, `07_figures/figure_manifest.json` | plot_group | figures, skipped figures with reasons, interpretation |
| `08_tables/baseline_*.csv`, `proximity_*.csv`, `high_confidence_far_components.csv`, `all_diagnostics.parquet` (long format: stage, table, row, column, value_str, value_num), `regional_summaries.csv` | sections 03, 04, 17 | |
| `09_reports/executive_summary.md`, `scientific_findings.md`, `limitations.md`, `artifact_manifest.json`, `artifact_inventory.csv` | section 17 | |
| `10_submission/submission_georegime_recovery_v2.tif`, `submission_validation.json`, `submission_sha256.txt` | section 16 (guarded) | the GeoTIFF only when every check passes |
| `cache/hash/sha256.jsonl`, `cache/inference/<identity>/snapshot.npz` (temporary), `cache/bank/` | infrastructure | |

## Figures (file stems)

| folder | figures |
|---|---|
| `prediction_distribution` | 01 confidence map, 02 distribution (histogram, exceedance, Lorenz), 03 mapped-fault proximity (mass vs area, enrichment, cumulative, band means and endpoints), 04 high-confidence classes and block concentration |
| `validation` | 05 geographic folds, 06 discovery holdouts, 07 split leakage audit, 08 narrow vs full masks, 09 original components, 10 full-area predictions, 11 reproduction check, 12 DTI breakdown and folds |
| `spatial_errors` | 13 TP / FP / FN maps, 14 FP density and proximity, 15 zoom panels, 16 fault-normal profiles |
| `experts` | 17 global vs specialists, 18 router / strength / mixing, 19 joint head distributions, 20 expert disagreement, 21 expert performance, 22 router vs persistence prior |
| `geology` | 23 confidence vs evidence, 24 evidence by error class, 25 gravity context vs predictions, 26 evidence zooms (FP, missed), 27 recall by length and linearity, 28 strike agreement, 29 regional recall, 30 lineaments vs confidence, 31 support vs model, 32 candidate zooms |
| `recovery` | 33 transform response, 34 policy trade-offs (TP gain vs FP, held-out DTI), 35 identity vs candidates, 36 cross-fit uncertainty (fold scores, bootstrap, regional), 37 recommendation gates |

That is 37 figure files, some of them multi-panel. A figure whose evidence is missing is skipped with its reason recorded in `figure_manifest.json`, never drawn empty.
