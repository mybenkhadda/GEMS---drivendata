# G14.1 leakage and shortcut register

Status: risks identified by reading the code (G14 at `b8a580e`); the measurements are G14.1 stages, all UNTESTED.

| # | risk | where (G14) | severity | static status | G14.1 measurement | G14.1 correction |
|---|---|---|---|---|---|---|
| L1 | training targets are context inputs (distance 0, density, strike) | `splits.geo_unit/disc_unit/final_unit` permitted ⊇ labels; `multiscale.UnitInputs` | Critical | confirmed | `leakage.target_context_overlap`, `copy_baseline`, `known_only_separability` (R02); counterfactuals (R06) | `discovery.ConditionalDiscoveryUnit` (M2), `geophysics_only` (M1) |
| L2 | inner-validation faults are in the context | same permitted map | High | confirmed | R06 `target_removed` intervention on the inner region | conditional eval context hides inner faults |
| L3 | inner labels shape training targets (soft target, distance, ignore ring, reliable negatives, orientation, endpoints) | `train.fit_unit` L138 | Medium | confirmed | `discovery.target_leak_check` (R07) | `discovery.separate_targets` + label-free boundary embargo |
| L4 | inner validation inside the training geography (autocorrelation 12.8 km) | `splits._block_select`, G13 roles | Medium (optimism) | confirmed | inner region DTI vs held-out DTI, rank correlation (F5) | selection on full-region inner DTI; discovery held-out units for calibration |
| L5 | geo-calibrated operating point used for discovery submission | notebook cell 20 fallback | High (calibration) | confirmed (user report) | area optimum by task (R03 / R05) | `submission` labels PROVISIONAL_GEO; discovery calibration first |
| L6 | dropout never removes known context | `multiscale._radiometric` | High (amplifies L1) | confirmed | D1 vs M0 | `TrainSpec.known_dropout` |
| L7 | final model context drops the TEST-fold faults | `splits.final_unit` | Medium | confirmed | full-map placement by region | `discovery.corrected_final_context` = every mapped fault |
| L8 | ensemble / blend weights chosen on the scored zone | n/a in G14.1 | High if present | guarded | `test_blend_weight_for_a_zone_is_chosen_on_other_zones` | `champion.crossfit_blend`, `conditional_blend` |
| L9 | calibration fitted on the scored zone | n/a | High if present | guarded | `calibration.crossfit_calibration` records `fit_on_zones`, flags `in_sample` | fitted on other zones only |
| L10 | champion residual weights from a champion that saw the held-out zone | M6 | High if present | guarded | `TrainSpec.residual_champion` requires a nested champion map | pipeline skips M6 units without `champion_nested/<unit>.npy` |
| L11 | combining maps across grids or protocols | any comparison | High | guarded | `artifacts.compatible`, `g14.evaluation.load_*` refusals | label-free comparison only for full maps |
| L12 | locked TEST reused | `g14.evaluation.evaluate_locked_test` | High | guarded (single use, log) | log file | unchanged |
| L13 | leaderboard used for selection | process | High | policy | provenance records the status of each candidate | no automatic upload, no LB-driven sweep |
