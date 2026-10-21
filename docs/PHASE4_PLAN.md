# Phase 4 — design plan, experiments, safeguards, reload paths

Authored 2026-10-21 (code only, nothing executed). Notebook: `notebooks/04_remote_sensing_and_geological_encoders.ipynb`.
Configs: `configs/phase4/rs_features.yaml` (feature bank), `configs/phase4/experiments.yaml` (experiments, reuse
aliases, channel groups, architectures, SSL, promotion thresholds). Outputs: `outputs/phase4/`.

## Design in one paragraph
Every Phase-4 model is scored by the Phase-3 machinery (per-fold binary top-area masks, cross-fitted area budget,
exact DTI; discovery3 for pseudo-discovery under `excl` and `retained_fp`), through one scorecard
(`phase4_eval.scorecard`) that adds thin-structure recall (exact TP support of the binary mask per label pixel,
stratified by Phase-1 segment length / sinuosity / junction and skeleton endpoint / branch proximity). Every
candidate has a named control that differs in one factor; promotion follows section 12 (`phase4_eval.promotion`).
Phase-3 runs are reused, never retrained (`reuse` aliases).

## Experiments (P = priority)
| P | id | what changes vs control | control | module |
|---|---|---|---|---|
| P0 | DL00_CNN_B_dti | CNN-B inputs, loss bce_dice → dti (spec reloaded from p3 config.json) | CNN_B_bcedice | cnn3 unchanged |
| P0 | DL01 | discovery models for CNN_A_dti (epochs = median best = 10); same splits / policies / budgets | CNN_A_bcedice (p3 disc) | cnn3 unchanged |
| P0 | DL02_CNN_TMGB_dti | raw terrain+mag+grav+basement only (no seis/strain/cond) | CNN_A_dti | cnn3 unchanged |
| P0 | RS01-RS05(c) | G1 / G3 ± RS_T / RS_P / RS_X, ± seismicity/strain; variant D2_conservative | RS01 / G3_D2 / RS05c | lgbm3 unchanged |
| P0 | `<best>_drop_<family>` | one RS family removed | best RS set | lgbm3 |
| P1 | P4_B_concat / P4_B_gated | terrain + subsurface encoders, per-level fusion | P4_B_ctrl (param-matched single) | geological_encoders, train4 |
| P1 | P4_C_context | + 1/4-res context branch (4x extent, strict masking) | P4_B_ctrl | train4 |
| P1 | P4_D_aux | + proximity and cos/sin 2θ heads (training only) | P4_B_ctrl | train4 |
| P1 | P4_A_repro (off) | Phase-4 trainer on CNN_A_dti inputs: trainer sanity check | CNN_A_dti | train4 |
| P2 | SSL01_strict / SSL01_transductive | masked-AE encoder init | P4_B_ctrl (random init) | geophysical_ssl |
| P2 | SSL02_strict (off) | cross-modal reconstruction init | P4_B_ctrl | geophysical_ssl |
| P2 | EO01 | AOI selection + 1 m DEM statistics on the 100 m grid (local files only) | 100 m features in AOIs | external_eo |
| P3 | EO02 / EO03, transformers, GANs, diffusion, GNNs | deferred | — | interfaces only |

## RS feature bank (67 specs; `rs:<modality>:<transform>[_<output>][_<tag>]@<scale>km`)
- RS_T (terrain, basis = detrended_elevation): TPI 0.3/1 km, TRI 0.1/0.3, local range 1/2, profile + tangential
  curvature 0.5/1, LoG 0.5, hillshade ⟂ each strike family + multi-azimuth, slope break 0.5/1, simplified
  geomorphon (convexity, ridge/valley) 0.5/1.5, coherence 0.5, ridge scale-persistence 0.5-4 km.
- RS_P (magnetic RTP, isostatic gravity, basement): analytic signal, TDX, THDR ridges, Hessian ridges, DoG,
  log-Gabor bank (max, orientation entropy, strike-family energy share), coherence, THDR-ridge persistence.
- RS_X: terrain-magnetic / gravity-magnetic axial orientation agreement, edge agreement (min = both, max =
  either: agreement never forced), joint orientation confidence (axial resultant), local/regional edge contrast,
  fault-normal vs fault-parallel step (orientation from the same or an independent modality), lineament
  persistence count across modalities.
- Redundancy audit (|Spearman| ≥ 0.95 vs any existing feature) removes duplicates of provided derivative bands
  (`tmi_hg`, `tmi_vg`, `iso_grav_anom_hg/_vg`) before experiments.

## Spatial-leakage safeguards
- Folds, discovery splits, sampler, seed: Phase-2 artifacts, reloaded (no re-splitting).
- LightGBM: lgbm3 sampler (fold training zone only, PU ignore band, withheld systems + 1 km ring weight 0 in discovery).
- CNN: patches fully inside the fold's inner-training zone (integral-image check); targets from visible labels
  only; checkpoints chosen on inner-val (training side); orientation / proximity targets from visible labels.
- Context branch: training context built from the stack with everything outside `folds.train(k)` zeroed.
- SSL strict: per-fold pretraining on `folds.train(k)` only; transductive reported in separate rows.
- Features are label-free; support beyond the 13 km buffer = validation *inputs* only (`feature_registry.csv`
  column `exceeds_buffer`). Strike families are a documented hyper-parameter from Phase-1 morphology.
- Budgets / ensemble weights are cross-fitted (chosen on the other folds/splits).

## GPU memory (RTX 3070 Laptop 8 GB)
fp16 stack on the GPU (~25 MB per channel; ~30 channels ≈ 0.75 GB), bf16 autocast, channels_last, batch 16 × 256 px,
`grad_accum` for larger effective batches, `infer_batch` 16 tiles, cuDNN autotune, `empty_cache()` between
folds; host-resident fallback when the stack does not fit. Phase-3 U-Nets peaked at 2.3-2.7 GB.

## Reload snippets
```python
from gems_bench.phase4_context import load_context4
ctx = load_context4()                         # Phase-2 data/folds/splits + Phase-4 paths, RS store, aliases
ctx.oof('CNN_A_dti'); ctx.oof('RS03_G3_potential')   # p3 alias / p4 run (after add_map or via reuse)
ctx.discovery_pred('CNN_A_dti', 0)            # p4 first, then p3
import pandas as pd; pd.read_csv(ctx.paths.out / 'phase4_model_comparison.csv')
from gems_bench.phase4_training import load_cnn4_fold_models
load_cnn4_fold_models(ctx.paths.experiments / 'p4_P4_B_gated_p256_s2026')
```
Checkpoints: `experiments/<exp>/models/fold<k>.pt` (CV), `disc<d>_<key>/models/model.pt` (discovery),
`ssl_<key>_<protocol>[_fold<k>]/models/encoder.pt` (SSL). RS features: `outputs/phase4/cache/features/*.npy`.

## Expected outputs (`outputs/phase4/`)
Headline: feature_registry, rs_feature_ablation, cnn_architecture_comparison, dual_encoder_results,
ssl_pretraining_history, ssl_finetuning_results, pseudo_discovery_results, fault_recovery_by_length,
model_complementarity, phase4_model_comparison, phase4_promotion (CSV) and PHASE4_FINDINGS.md.
Tables: rs_redundancy_audit, rs_lgbm_scorecard, rs_group_importance, cnn_controls_scorecard, dl01_pseudo_discovery_curves,
architecture_parameters, architecture_deltas, unique_discovery_systems(_list), error_atlas_windows_*, eo01_aoi_candidates.
Figures (PNG + SVG): feature_transformation_atlas, directional_structure_comparison, terrain_subsurface_evidence_maps,
encoder_architecture_diagram, cnn_training_curves, p4_training_curves, rs_geo_dti_comparison, geo_dti_comparison,
pseudo_discovery_comparison_{excl,retained_fp}, error_atlas_*, disagreement_*, candidate_lineament_gallery_*, ssl_pretraining_curves.
Datasets: candidate_lineaments_<best>.gpkg/.parquet, dem1m_*.tif. Registry / manifests: experiment_registry.csv,
experiment_manifest.csv, manifest.json.
