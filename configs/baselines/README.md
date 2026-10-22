# Baseline training-configuration snapshots

`legacy_specs/<name>.json` are verbatim copies of the `config.json` files that the Phase 2–4 runs wrote
(`outputs/<phase>/experiments/<exp_id>/config.json`): feature or channel lists, hyperparameters, sampling, and
architecture settings. They are **recipes, not results**. They hold no metric, prediction, or weight.

They are used only when the legacy run folder is absent, which is the case on a fresh Google Drive:

* `gems_bench.training.spec.base_spec_dict` falls back to them, so every `cnn5` experiment (B, C, E) can resolve
  its base model (`P4_B_gated`) without the Phase 4 upload.
* `gems_bench.pipeline.baselines.train_baseline` retrains a baseline from them when it is listed in
  `stages.train_baselines`. The result is written to the canonical legacy folder only if no finished run is
  there. It is labelled `origin: regenerated_by_master_pipeline` and `historical_equivalence: false` in
  `PROVENANCE.json`, because library versions and GPU kernels differ from the original runs.

| file | legacy folder |
|---|---|
| `B0_lineament.json` | `outputs/benchmark/experiments/b0_lineament_fused_s2026` |
| `G1.json` / `G3.json` | `outputs/benchmark/experiments/lgbm_G{1,3}_s2026` |
| `RS05_G3_noSS_RS.json` | `outputs/phase4/experiments/p4_RS05_G3_noSS_RS_s2026` |
| `CNN_A_dti.json` | `outputs/phase3/experiments/p3_CNN_A_dti_p256_s2026` |
| `DL00_CNN_B_dti.json`, `DL02_CNN_TMGB_dti.json` | `outputs/phase4/experiments/p4_DL0{0,2}_*_p256_s2026` |
| `P4_B_gated.json` | `outputs/phase4/experiments/p4_P4_B_gated_p256_s2026` |

`n_params` inside `P4_B_gated.json` is the legacy model's parameter count. It is architecture metadata,
not a score.
