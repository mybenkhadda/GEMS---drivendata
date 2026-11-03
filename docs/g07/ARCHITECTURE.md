# G07 architecture

## Modules (`gems_bench/g07/`)

| module | purpose | public API (main) |
|---|---|---|
| `config` | YAML + notebook merge, validation, identity | `load_config`, `validate_config`, `canonical_config`, `compute_run_id`, `stage_config` |
| `contracts` | DEM / candidate / model / prediction schemas; unit + grid checks | `GridSpec`, `DEMSource`, `DEMTile`, `DEMFeatureSpec`, `TrainingVisibility`, `FaultCandidate`, `CandidateGeometry`, `ModelArtifact`, `PredictionArtifact`, `EvaluationResult`, `vertical_scale`, `validate_dem_source`, `assert_not_100m_substitute` |
| `inventory` | DEM sources, eligible-region coverage, G06 store / W2 artifacts, licences | `discover_dem_sources`, `inspect_dem_coverage`, `coverage_geojson`, `inspect_existing_g06_store`, `inspect_w2_artifacts`, `source_licenses`, `build_dependency_report` |
| `dem_io` | catalog, halo windows, block means, exact 100 m aggregation, mosaic, DEM store | `open_dem_catalog`, `iter_dem_tiles`, `read_dem_window`, `align_dem_to_analysis_grid`, `aggregate_dem_feature`, `validate_dem_tile`, `build_vrt`, `mosaic_windows`, `DemFeatureStore` |
| `dem_features` | native geomorphometry (two tiers) + processing drivers | `dem_slope`, `dem_hillshade`, `dem_curvature`, `dem_local_relief`, `dem_roughness`, `dem_openness`, `dem_terrain_position`, `dem_multiscale_features`, `feature_kinds`, `process_fine_source`, `build_coarse_dem`, `process_coarse`, `required_halo_px` |
| `terrain_structure` | scarps, lineaments, truncation, drainage, persistence, orientation agreement | `terrain_lineaments`, `scarp_candidates`, `directional_coherence_dem`, `terrain_truncation`, `drainage_alignment`, `multiscale_scarp_persistence`, `orientation_agreement` |
| `candidate_generation` | label-free pools, merge, provenance | `generate_surface_candidates`, `generate_subsurface_candidates`, `generate_w2_candidates`, `merge_candidate_inventory`, `deduplicate_candidates`, `generate_unit_candidates` |
| `training_labels` | visible-only supervision variants, augmentation | `visible_fault_labels`, `fault_distance_targets`, `fault_orientation_targets`, `positive_unlabeled_targets`, `make_hard_negative_pool`, `build_fault_system_groups`, `perturbed_candidates`, `random_negative_candidates` |
| `datasets` | combined store, surface patches, subsurface pool + labels, candidates, meta | `CombinedStore`, `build_surface_patch_dataset`, `build_subsurface_pixel_dataset`, `subsurface_labels`, `build_candidate_dataset`, `build_meta_dataset` |
| `models_surface` | A1 / A2 classical, C1 U-Net (orientation-consistent augmentation) | `train_surface_random_forest`, `train_surface_lightgbm`, `build_surface_unet`, `train_surface_unet`, `predict_surface_tiles`, `augment` |
| `models_subsurface` | B0 / B1 / B2 on the reused G06 store | `load_g06_feature_store`, `select_subsurface_features`, `train_subsurface_classifier`, `train_fault_distance_model`, `predict_subsurface_raster` |
| `models_candidates` | relevance classifier / ranker, utility | `train_candidate_classifier`, `train_candidate_ranker`, `estimate_candidate_utility`, `score_candidate_inventory`, `candidate_feature_columns` |
| `crossfit` | isolation + strictness | `build_outer_folds`, `build_inner_folds`, `crossfit_surface_model`, `crossfit_subsurface_model`, `crossfit_candidate_ranker`, `validate_upstream_honesty`, `system_balanced_weights` |
| `confidence` | calibration, localisation, disagreement, corridors | `calibrate_candidate_confidence`, `estimate_localization_uncertainty`, `estimate_model_disagreement`, `build_confidence_corridor` |
| `representation` | policy `g07.rep.1` | `exact_k_mask`, `apply_policy`, `soft_corridor`, `variable_width_representation`, `candidate_portfolio_selection`, `compare_representation_policies` |
| `evaluation` | canonical scoring, exact-budget audit, recall, feasibility, promotion | `evaluate_discovery`, `evaluate_geographic`, `evaluate_completion`, `compare_to_w2`, `evaluate_candidate_recall`, `exact_budget_ok`, `dem_signal_table`, `feasibility_decision`, `evaluate_promotion` |
| `inference` | full region, parity, gated writer | `predict_surface_full_region`, `predict_subsurface_full_region`, `predict_candidate_inventory_full`, `construct_final_prediction`, `verify_export_parity`, `write_submission_candidate` |
| `visualization` | DEM / feasibility / recall / budget / change / case atlases | `plot_dem_native_panels`, `plot_resolution_comparison`, `plot_dem_coverage`, `plot_signal_auc`, `plot_candidate_recall`, `plot_exact_budget`, `plot_tp_fp_fn_change`, `plot_case_atlas` (+ G06 plots) |
| `persistence` | manifests, caches, Drive | `artifact_fingerprint`, `save_manifest`, `validate_cache`, `atomic_save`, `sync_artifacts`, `write_artifact_inventory`, `source_fingerprint` |
| `pipeline` | `G07Pipeline` | `preflight`, `inspect_dem`, `assess_dem_feasibility`, `prepare_dem_features`, `prepare_training_data`, `benchmark_surface_models`, `benchmark_subsurface_models`, `generate_candidates`, `train_candidate_ranker`, `evaluate`, `reconstruct_full_region`, `build_report`, `run` |

Interface adjustments (documented; the notebook uses these): `G07Pipeline(config, run_flags, repository_commit,
factory=None, progress=print)`; extra stage methods `run_tests`, `dem_inventory`, `dem_coverage`, `restore_g06`,
`train_surface_unet_stage`, `full_inference`, `export_submission`; `inspect_dem` = inventory + coverage;
`reconstruct_full_region` = full inference + export.

## Reused GEMS components

`gems_bench.dem.inventory` (TNM URL list parsing, overlap policy, study-area intersection, remote headers),
`gems_bench.dem.download` (guarded resumable download), G06: feature store (`open_feature_store`, `FeatureStore`),
masked operators (`feature_physics.masked_smooth`), structure tools (`along_cross_features`, `directional_coherence`,
`axial_agreement`), splits / visibilities / isolation, supervision, sampler (`build_pixel_pool`, `label_pool_for_fit`),
`PixelModel` / `fit_pixel_model` / `predict_pixel_raster` / `oof_path` / `load_oof`, segment features / nested known
context / segment classifiers, U-Net builder / `FoldScaler`, evaluation helpers, persistence and Drive workspace,
raster I/O; engine attach (submission factory: folds, splits, honest W2 / CNN maps); `gems3.official` (units,
FastScorer), `m7.protocol` (eligibility, protocol keys, `score_row`), `gems3.representation` (post operations),
`gems3.submission` (`eligible_mask`, `finalise`, `validate`), `gems_bench.submission.write_submission`,
`geomoe.scenarios.system_recovery`, `p6.segment_recovery`, `p6.stack` (sampling of meta rows), `m7.ranking._zone_mask`.
Nothing of G06 is rebuilt; one G06 fix: `models_segment.class_weights` now balances classes exactly.

## Stages

See the `pipeline.py` docstring. Caching: stage fingerprint = version + stage + config section + upstream
fingerprints + source hash of the stage's modules; DEM per-tile products are keyed by the DEM-processing fingerprint
and tile id / size (pilot and full builds share them); models / predictions are fingerprint-keyed; successful outputs
replaced by another fingerprint move to `history/`. `BLOCKED_INPUT` (missing DEM, licence, no GO) never blocks the
geology path.
