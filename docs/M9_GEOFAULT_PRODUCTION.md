# M9: GeoFault production (guide)

Notebook: `notebooks/colab/GEMS_M9_GEOFAULT_PRODUCTION.ipynb` · code: `gems_bench/m9/` · config:
`configs/m9/geofault_production.yaml` · outputs: `MyDrive/GEMS_2/outputs/m9_geofault_production/` · cache:
`cache/m9/` (local, rebuilt if lost) · tests: `tests/test_m9.py`.

**Status: code authored only. Nothing has been executed** (no cell, test, training, inference or raster job). No M9
number exists. The values below are earlier offline records, not leaderboard results. No M7 / M8 output exists in the
local checkout; section 00 copies M8's saved gate tables from Drive verbatim (or records them as absent), so nothing
about M8 is assumed.

## Why

Recorded offline values (official-style, earlier sessions): B0 / B1 geographic DTI ~0.165 / ~0.168, fusion ~0.178,
fusion + crest ~0.183; discovery B0 / B1 raw ~0.104 / ~0.092, fusion + crest ~0.162, + 300 m ring ~0.198 (the ring may
exploit the pseudo-discovery protocol). The M7 candidate generator covered 5 / 120 withheld systems and 0 / 93 missed
by both CNNs. Four weaknesses: incomplete, non-random labels; segmentation that does not represent thin continuous
networks; extraction that discards weak evidence before ranking; validation that only approximates expert discovery.

Hypothesis (evaluated, never assumed): a multiscale, physics-aware network trained to recognise and *reconstruct*
incomplete fault systems, plus structured tracing and candidate verification, recovers geometry that historical-label
segmentation misses.

## Architecture

```text
cached features (02) + spectral (03, gated) -> cube (04)
   -> GeoFaultNet (06): modality stems | gated fusion | U-Net | low-res context branch
        heads: dense evidence, (cos 2t, sin 2t), endpoint, junction
   trained with corruption (05) + staged losses (07) per geographic fold / discovery split (08)
   -> held-out maps (09) -> tracing (11) [MC 12, RL 13 optional] -> candidate features (14) -> GBMs (15)
   -> dense + structured (+ legacy) cross-fitted fusion (16, Gate 6) -> full-area inference (17) -> GeoTIFFs (18)
```

The dense channel is a complete submission route by itself: tracing, ranking, Monte Carlo and RL are additive and
gated, and the B0 / B1 champion is always exported as the fallback.

## Reuse matrix (also `manifests/reuse_matrix.csv`)

| capability | existing module | reuse / change | M9 consumer |
|---|---|---|---|
| native grid, bands, labels, template | `data/__init__.py` | unchanged | features, dataset, submission |
| 139 cached features + registry | `features/`, `phase4_context.py`, `features/phase5_store.py` | read-only memmaps | features, dataset |
| potential-field operators, upward continuation | `m8/potential_fields.py` | unchanged | spectral, features |
| folds, discovery splits | `folds.py` | unchanged (CV-A, CV-B) | dataset, validation |
| official metric, units | `metric.py`, `gems3/official.py` | unchanged | every score |
| protocol keys, eligible budget, system hits, block bootstrap | `m7/protocol.py` (M7 engine) | unchanged | everywhere |
| baseline recovery + champion maps | `m8/baseline.py`, `m7/baseline.py` | unchanged (M9 serves M8's interface) | s01 |
| nested selection, representations | `gems3/metric_opt.py`, `gems3/representation.py` | unchanged | validation, gbm, fusion |
| held-out maps, comparison, verdicts | `m7/rasterize.py`, `m7/comparison.py`, `m7/decision.py` | unchanged (view restricted to trained zones) | validation, fusion |
| candidate sets, skeleton tracing | `gems3/candidates.py`, `m7/generator.py`, `p3/lineaments.py` | unchanged | tracing, candidates |
| candidate descriptors, PU labels, LOZO rows | `m7/cand_features.py`, `gems3/descriptors.py`, `m7/ranking.py` | reused + M9 features | candidates, gbm |
| corruption classes (benchmark) | `m7/corruption.py`, `gems3/map_completion.py` | reused for CV-C; new training operators | corruption |
| orientation algebra / targets, skeleton topology | `geology/` | unchanged | dataset, losses, tracing |
| orientation loss, clDice, EMA teacher, PU weights | `losses/` | unchanged | losses, train |
| patch sampler, Hann tiling | `cnn.py` | sampler unchanged; tiling extended (multi-head, context, resumable tiles) | dataset, infer |
| checkpoints, GPU config | `training/checkpoint.py`, `training/gpu.py` | unchanged (`best_and_latest`) | train |
| submission writer / validator / ledger | `gems3/submission.py`, `m7/export.py` | unchanged + M9 checks | submission |
| workspace | `m7/workspace.py` | subclassed | workspace |

Changes to existing code: `Pipeline.m9()` in `gems_bench/pipeline/__init__.py`; optional extras `m9` (xgboost,
catboost) and `rl` (gymnasium, stable-baselines3) in `pyproject.toml`. No baseline experiment, prediction or
checkpoint is modified.

## Stages and sections

| stage | section | module | main outputs (under `outputs/m9_geofault_production/`) | gate |
|---|---|---|---|---|
| — | 00 setup | `run.py`, `inventory.py` | `manifests/{setup.json, reuse_matrix, band_availability, protocol_keys, m9_config.json}`, `baseline/m8_saved_outputs` | CRS / template |
| PREPARE | 01 baseline | `m8/baseline.py` (reused) | `baseline/*`, `validation/maps/s01_*`, `submissions/m9_fallback_*.tif`, `baseline/gate0` | **G0** |
| PREPARE | 02 features | `features.py` | `features/{inventory, selected_registry, redundancy_pairs, crossmodal_new, family_counts, selected_spearman}` | — |
| PREPARE | 03 spectral | `spectral.py`, `features.family_screen` | `spectral/{spectral_registry, spectral_vs_existing, screen_*, gate1}`, figures | **G1** |
| PREPARE | 04 dataset | `dataset.py` | `datasets/{training_units, normalization_manifest.json, channels_*.json}`, `datasets/cv/*` | — |
| PREPARE | 05 corruption | `corruption.py` | `corruption/{samples/*, operation_counts, corruption_manifest.json, benchmark/*}` | leakage audit |
| TRAIN | 06 architecture | `model.py` | `models/{architecture, net_spec_*.json}` | — |
| TRAIN | 07 losses | `losses.py` | `training/loss_schedule` | — |
| TRAIN | 08 training | `train.py` | `models/checkpoints/<variant>/<unit>_model.pt`, `training/logs/*`, `training/training_units`, `reports/model_registry/*` | — |
| VALIDATE | 09 validation | `validation.py`, `fusion.compare` | `predictions/*`, `validation/maps/m9_dense_*`, `validation/metrics/*`, `validation/gate2`, `gate3` | **G2, G3** |
| VALIDATE | 10 dense analysis | `run.py` | `dense_analysis/{orientation_error, endpoint_junction, reliability}` | — |
| TRACE | 11 tracing | `tracing.py` | `tracing/units/*`, `tracing/candidates.gpkg`, `tracing/oracle/{oracle_curve, pooled_oracle, gate4}` | **G4** |
| TRACE | 12 Monte Carlo | `montecarlo.py` | `monte_carlo/{stability, convergence, occupancy_*, mc_only_coverage, assumptions.json}` | G5 |
| TRACE | 13 RL | `rl.py` | `rl/tracer_comparison` | G5 |
| RANK | 14 candidates | `candidates.py` | `candidates/{candidate_table, units/*, feature_groups, feature_provenance, label_balance, candidates.gpkg}` | needs G4 |
| RANK | 15 GBMs | `gbm.py` | `gbm/{summary, per_zone, candidate_metrics, system_recall, stability, choice, feature_importance, models/*}` | — |
| RANK | 16 fusion | `fusion.py` | `fusion/{grid, summary, per_zone, ring, gate6, method_parameters.json}`, final `validation/metrics/*` | **G6** |
| INFER | 17 inference | `submission.full_dense`, `infer.full_region` | `predictions/full/core_heads.tif`, `inference/tiles/*`, `tracing/full_region.gpkg` | — |
| SUBMIT | 18 submission | `submission.py` | `submissions/*.tif` + `.json`, `submission_index.csv` | never uploads |
| SUBMIT | 19 reports | `gates.py`, `report.py` | `reports/{gates, M9_GEOFAULT_PRODUCTION_REPORT.md, model_registry/*}` | — |

`m9.run(stage=...)` runs one stage; `m9.run()` everything enabled. Profiles: `PILOT` (default: geo fold 0 + split 0,
12 epochs x 80 steps, variant `core`; comparisons paired on those zones; nothing can be promoted with < 3 zones),
`FULL` (all folds / splits, 40 x 150, variants `core, recon_off, stage_c, no_spectral`). Profiles may only carry
scope keys; variants may only change `features, spectral, data, corruption, model, loss, training`.

## Geological features (02-03)

Families (first matching rule): potential gradients (THDR / analytic signal), tilt / normalised derivatives, curvature,
ridge / crest, directional (log-Gabor, Gabor bank, directional derivatives, DoG), coherence, cross-modal (Phase-4
`rs:xmod:*`, EDA rasters), terrain morphology, persistence; regional context (smooth / 10-20 km) goes to the
low-resolution branch only, with strain and seismicity (regional descriptors; low seismicity never rejects a fault).
Pruning is label-free: greedy |Spearman| <= 0.92 per family on a fixed sample, at most 6 per family, B0's channels
first. New layers (only what did not exist): tolerant magnetic-gravity edge alignment (300 / 800 m), tolerant
terrain-magnetic alignment (surface / subsurface), tolerant doubled-angle orientation agreement, signed magnetic /
gravity strength ratio, local contrast normalisation of magnetic and gravity edges.

Spectral (03): windowed FFT (64 px Hann windows, stride 16, plane removal, nodata-aware) -> anisotropy, dominant
strike (trace = wavevector + 90 deg, stored as doubled angle), directional entropy, short / long ratio, radial slope
(shape only, never a depth), band energy; steerable one-sided log-Gabor wavelets (0.4-3.2 km, 6 orientations) ->
energy by scale, orientation, coherence, persistence, fine / coarse ratio; upward continuation bands (no downward
continuation); optional STFT ablation. Gate 1 evidence = nested pixel LightGBM screen (training geography of each
trained zone only, held-out official DTI): `raw`, `base`, `base_no_new`, `base_spectral`; INCLUDE a component at
pooled gain >= 0.003 or >= 2 unique systems. Scaling: per-unit median / IQR fitted on the unit's permitted pixels.

## Network and losses (06-08)

GeoFaultNet: stems for raw / structural / spectral / optional DEM / map context (damaged map + availability flag),
per-pixel gated fusion, residual U-Net (widths 32-64-128-192), low-resolution context branch (4x pooled, 4x extent)
cropped at the bottleneck + FiLM, heads seg / orientation / endpoint / junction (~ a few million parameters).
Losses: L1 focal BCE with weak unlabeled negatives (0.5 x label-free background weight; context pixels 0.1); L2 kernel
Tversky surrogate (alpha 0.2 / beta 0.8, 300 m); L3 skeleton recall or clDice; L4 doubled-angle cosine on reliable
targets; L5 topology heatmaps inside a boundary-aware mask; L6 reconstruction of removed geometry; L7 optional EMA
consistency. Curriculum A (L1 + L6) -> B (+ L4, L5) -> C (+ Stage-C terms, enabled only via the `stage_c` variant).
Checkpoint selection: inner-validation official-style DTI with NO map context (reconstruction must not buy
discovery). Discovery units: fixed epochs = median best epoch of the geo units.

## Validation and reconstruction protocol (05, 09)

CV-A geographic folds (inner-train -> inner-val -> val), CV-B whole-system discovery (withheld systems + ring weight 0,
never negatives, never shown as context), CV-C reconstruction (M7 corruption classes on systems assigned to folds
BEFORE damage, evaluated inside the fold's validation zone only, damaged-map context vs none), CV-D recall by system
length, CV-E DTI by label-free geological domain (k-means of 20 km block descriptors), CV-F recall / FP mass by
distance to known faults. Discovery units are predicted twice: no context (independent discovery) and with the
retained known map (the test-time situation, `validation.production_context`). Nested (rep, frac) per zone; every
held-out map re-scored by M7's comparison (pooled + worst-zone DTI, system / short-fault recall, density, regions,
proximity, paired deltas, spatial block bootstrap — never pixel bootstrap). Gate 2: PROMOTE or COMPLEMENT (>= 3
unique systems or fusion gain >= 0.005); Gate 3: reconstruction recall +0.02 without >= 0.005 discovery / geo loss.

## Tracing, Monte Carlo, RL (11-13)

Deterministic baseline (crest skeleton + minimum-cost bridges), batched beam search over every seed in both
directions (heading turns +-7.5 / 15 deg, orientation alignment, curvature penalty, gap tolerance, branch re-traces),
optional Gumbel-stochastic beams; dedup by overlap; oracle recall vs density and systems added over the dense map at
the same pixel count (Gate 4: recall >= 0.30 and >= 3 additional systems at 40 km / 100 km^2). MC-A perturbs the
dense evidence (correlated noise, one registration shift, smoothing / orientation / seed jitter: assumptions, not a
noise model); MC-B samples stochastic paths (occupancy, branch frequency); MC-C = seeded variants averaged into
`m9_dense_core_ens`. RL: Gymnasium-compatible environment, behavioural cloning, optional PPO / DQN, learned edge
cost; trained on other zones' truth only; promoted only with >= 2 more held-out systems than beam search.

## Ranking and fusion (14-16)

Candidate features: M7 descriptors + tracing statistics, fault-normal contrasts (200 / 400 / 800 m), spectral /
cross-modal means, M9 / B0 / B1 support and disagreement, context percentiles, Monte Carlo stability (`has_mc`).
XGBoost / LightGBM / CatBoost with identical tables, folds, weights, seeds and budgets; rasterised nested DTI decides;
within 0.003 the steadier library (worst zone) wins. Fusion channels: dense, structured (percentile among candidate
pixels, 0 elsewhere), legacy (champion rank fusion); simplex weights / representation / frac nested; ring evaluated
separately. Gate 6 promotes only a reproducible M9 method passing every M7 promotion rule (>= 3 zones).

## Production inference and submission (17-18)

Fold-average of the `core` CV checkpoints (or the `full` unit), 1024 px tiles, 64 px halo with linear seam ramps,
Hann-blended patches, resumable per tile, CPU / GPU fallback, production context = the known map. Files: always
`m9_fallback_fusion_thin.tif` (B0 / B1 champion, ring 0); `m9_champion_<method>.tif` only on Gate 6 PROMOTE.
Template authority, NaN only where the template is NaN, float32 in [0, 1], eligible budget, area ledger, validation
that raises, never overwrite; M9 checks: positive fraction in [1e-4, 0.25], no known-fault positives, checkpoint
sha256 = registry. Nothing is uploaded.

## Colab execution sequence

1. Setup cells (configuration, install with `[colab,m9]`, context), section 00.
2. `PREPARE` on CPU: 01 (Gate 0: read `baseline/gate0.csv`), 02, 03 (Gate 1), 04, 05.
3. `TRAIN` on a GPU runtime (T4 is enough): 06, 07, 08. Set `training.max_runtime_min` on free sessions; re-run 08
   after a disconnect (resumes).
4. `VALIDATE`: 09 (Gates 2 / 3), 10.
5. `TRACE`: 11 (Gate 4). 12 / 13 only when testing P2 extensions.
6. `RANK`: 14, 15, 16 (Gate 6) — 14 / 15 refuse after a Gate-4 STOP.
7. Switch to `FULL` and repeat 03 -> 16 (variants `recon_off`, `stage_c`, `no_spectral` give Gates 1 / 3 their
   network-level evidence).
8. `INFER` (GPU) + `SUBMIT`: 17, 18, 19. Upload at most one controlled probe; log its score by hand.

## Computational requirements (estimates, not measured)

* Cube: ~60 channels x 12.3 M px x 2 B ~ 1.5 GB on disk per channel set (+ pooled ~0.1 GB); RAM copy when it fits.
* Training (T4, fp16): ~3-5 M parameters, batch 16 x 256^2; PILOT 2 units x 12 epochs x 80 steps; FULL 6 units x 4
  variants x 40 x 150 steps (several GPU-hours; resumable). Inner validation predicts the inner zone each epoch.
* Spectral: one FFT per field + 24 inverse FFTs (~4.7 k x 4.1 k padded) for wavelets; windowed FFT on ~48 k windows.
* Tracing: vectorised over 2048-seed chunks (memory ~ T x chunk x beam x 2 floats).
* Full inference: ~150 tiles x (fold models) on a T4; resumable.

## Known limitations and assumptions

* Nothing has run: expect first-run errors (shapes, dtypes, missing cache names, library versions) and fix in place.
* PILOT trains one fold + one split: no nested choice is possible there (the champion's choice is borrowed and
  labelled), no promotion is possible, gates are PROVISIONAL.
* Pseudo-discovery under-represents new faults next to known traces; known-fault proximity is reported (CV-F) and the
  ring stays out of the champion by default.
* Zones were inspected in earlier sessions: not an untouched test set; 3 folds / 3 splits = descriptive paired evidence.
* The hidden new-fault density is unknown: the submission area fraction (geo all-zone choice) is the largest
  unverified assumption.
* Monte Carlo distributions are assumptions; occupancy is not a probability. RL rewards use mapped (incomplete) traces.
* `geology.skeleton.orientation_targets` on unit crops (section 10) assumes the structure tensor depends only on the
  pixel size, not the grid shape (unverified).
* XGBoost `min_child_weight` is not equivalent to LightGBM / CatBoost leaf counts (documented, left at default).

## First experiments to run

1. PILOT `PREPARE` (CPU): confirm Gate 0 PASS, the compatibility table, `features/selected_registry.csv` (channel
   count per family) and the Gate-1 screen (`spectral/screen_pooled.csv`).
2. PILOT `TRAIN` + `VALIDATE` on a T4: one `core` geo fold + one split. Read `training/logs/core__geo_f0.csv`
   (loss decomposition, `inner_dti`, `loss_unlabeled_mean_p`), then `validation/metrics/method_pooled.csv`:
   `m9_dense_core` vs `s01_fusion_thin` on geo zone 0 and disc split 0, and `m9_dense_core` vs `_noctx` on the split
   (does the known-map context help discovery?) and `reconstruction_pooled.csv` (damaged vs none).
3. PILOT `TRACE`: `tracing/oracle/gate4.csv` and `pooled_oracle.csv` (additional systems over the dense map).
4. Only if 2-3 look promising: FULL training of `core` and `recon_off` (Gate 3), then `stage_c` / `no_spectral`.
5. FULL `RANK` -> Gate 6. Export the fallback regardless; the M9 champion only on PROMOTE.
