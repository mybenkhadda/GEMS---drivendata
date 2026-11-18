# G14.1 tests: written, never executed

Command (Colab, after cell 03):

```
cd /content/GEMS && python -m pytest -q tests/test_g14_1_core.py tests/test_g14_1_torch.py tests/test_g14_1_regressions.py
```

| area (Phase V) | test |
|---|---|
| artifact discovery, Drive fallback, local preference, grid check | `test_discovery_prefers_local_and_ledger_resumes` |
| raster alignment, flips, transposition, shifts | `test_phase_correlation_detects_integer_and_subpixel_shift`, `test_d4_match_detects_flip` |
| transform-derived, resolution-aware metric kernel | `test_kernel_radius_from_transform`, `test_kernel_radius_is_resolution_aware` |
| known-fault exclusion, masking order, area accounting | `test_area_accounting_and_known_zero`, `test_mask_after_lets_known_compete_for_budget`, `test_sparse_soft_keeps_probabilities` |
| target / context disjointness | `test_conditional_draws_are_disjoint_and_change`, `test_supervision_masks_context_carries_no_loss`, `test_original_g14_units_are_flagged_as_shortcut`, `test_geophysics_only_removes_context` |
| train / validation target separation | `test_separated_targets_never_use_inner_labels` |
| protocol consistency, discovery scorecard availability | `test_protocol_vs_primary_task`, `test_protocol_check_messages`, `test_operating_point_preference` |
| full-region aggregation | `test_ratio_of_sums_differs_from_mean_of_ratios`, `test_region_components_match_metric` |
| failure classes need evidence | `test_failure_classes_need_evidence` |
| loss numerical stability (empty positives), gradient diagnostics | `test_every_objective_is_finite_and_differentiable`, `test_gradient_diagnostics_report_norms_and_conflicts` |
| missing-modality / architecture output contract | `test_architectures_share_the_output_contract` |
| checkpoint / experiment provenance | `test_experiment_hash_check`, `test_compatibility_rules` |
| ensemble leakage protection | `test_blend_weight_for_a_zone_is_chosen_on_other_zones` |
| submission configuration transitions | `test_incompatible_stage_and_mode_is_an_error_not_a_skip`, `test_submission_refuses_wrong_mode`, `test_g14_submit_with_train_flag_is_rejected_and_g141_modes_are_consistent` |
| missing final manifest / map / scorecard | `test_missing_final_map_and_scorecard_give_actionable_errors`, `test_g14_predict_final_message_names_the_fix` |
| notebook stage dependencies | `test_stage_plan_dependencies_and_profiles` |
| spec registry is one-factor | `test_spec_registry_changes_one_factor` |
| regression `p, t, w = a, b` | `test_no_tuple_unpacking_arity_mismatch` (AST, all g14 / g14_1 code), `test_soft_cldice_runs` |
| regression `for name, (f, _, _, S) in zip(LEVEL_NAMES, self.views)` | `test_views_are_unpacked_as_pairs` (AST), `test_sensor_dropout_path_runs` |

## Not covered

The following need real data and are measured by the stages, not by unit tests:

* submission round-trip parity on the real template (R01 `round_trip`);
* CRS equality on real files (R01 grid table);
* the spatial integrity of real splits (G14 `assert_no_leakage`, R07 audit).
