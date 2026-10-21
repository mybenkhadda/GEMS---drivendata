# GEMS — Phase 2 benchmark findings

_Generated 2026-10-20 20:54 by `notebooks/02_fault_benchmark.ipynb` from saved artifacts. All scores are out-of-fold on geographic folds (not leaderboard scores)._

## Results

| model            |   dti_pooled_raw |   dti_pooled_crossfit_tf |   dti_fold_mean |   dti_fold_std |   dti_fold_min |   recall@100m |   precision@100m |   pseudo_discovery_recall_w |
|:-----------------|-----------------:|-------------------------:|----------------:|---------------:|---------------:|--------------:|-----------------:|----------------------------:|
| B0_lineament     |           0.0607 |                   0.0688 |          0.0614 |         0.0065 |         0.0547 |        0.5918 |           0.0404 |                      0.2948 |
| B0_terrain_ridge |           0.0606 |                   0.0717 |          0.0613 |         0.0078 |         0.0513 |        0.6665 |           0.0405 |                    nan      |
| lgbm_G1          |           0.0584 |                   0.0644 |          0.0589 |         0.007  |         0.0502 |        0.6615 |           0.0548 |                      0.7657 |
| lgbm_G2          |           0.0585 |                   0.0671 |          0.0586 |         0.0067 |         0.0528 |        0.5202 |           0.0517 |                      0.7203 |
| lgbm_G3          |           0.0612 |                   0.0815 |          0.0617 |         0.0087 |         0.0536 |        0.5216 |           0.0613 |                      0.4081 |
| lgbm_G4          |           0.0597 |                   0.0769 |          0.0602 |         0.008  |         0.0514 |        0.5007 |           0.0607 |                    nan      |

## Decisions

**1. Primary baseline:** `lgbm_G3` (pooled cross-fitted DTI 0.0815, worst fold 0.0536). Candidates within one fold-std of the leader: lgbm_G3, lgbm_G4; the most fold-robust of them was chosen.

**2. Do non-terrain modalities add value (G2 raw-all vs G1 terrain-only)?** verdict = **inconclusive** (mean ΔDTI -0.0002, per fold {0: 0.0026, 1: 0.0007, 2: -0.0041}).
   - gain → keep all modalities; inconclusive → keep them but test modality-drop ablations on G3; loss → terrain-only is the stronger prior, drop low-SHAP modalities before Phase 3.

**3. Is multi-scale feature engineering worth keeping (G3 vs G2)?** verdict = **inconclusive** (mean ΔDTI 0.0030). Regional context (G4 vs G3): **inconclusive** (mean ΔDTI -0.0015).
   - keep engineered layers only on a 'gain' verdict; on 'inconclusive' keep the top-SHAP subset only.

**4. U-Net vs LightGBM:** not evaluated (one family missing).

**5. Poorly recovered fault types (primary model, strata with ≥ 20 segments):**

| stratum       | value          |   n_segments |   trace_km |   weighted_recall |
|:--------------|:---------------|-------------:|-----------:|------------------:|
| region_name   | NE             |          302 |    695.489 |             0.307 |
| fold          | 2              |          767 |   1969.53  |             0.318 |
| region_name   | CW             |          236 |    483.113 |             0.329 |
| length_bin    | <1             |          681 |    493.074 |             0.334 |
| strike_sector | (120.0, 150.0] |          209 |    576.618 |             0.338 |

**6. Next research phase:** **ensembling** — the best OOF ensemble adds 0.031 DTI over the primary baseline.
   Rule order: PU first if pseudo-discovery recall < 0.7 × geographic recall; else architecture if CNN − GBM > 0.02; else ensembling if ensemble gain > 0.01; else targets / post-processing.

## Pseudo-discovery (withheld fault systems)

| model        |   dti_excl_retained |   dti_no_exclusion |   weighted_recall |   recall@100m |   recall@300m |   extra_confident_km2 |
|:-------------|--------------------:|-------------------:|------------------:|--------------:|--------------:|----------------------:|
| B0_lineament |              0.0326 |             0.0728 |            0.2948 |        0.2915 |        0.3285 |               2148.8  |
| lgbm_G1      |              0.0264 |             0.0786 |            0.7657 |        0.9468 |        0.9593 |              10640.9  |
| lgbm_G2      |              0.0259 |             0.0822 |            0.7203 |        0.9332 |        0.9576 |               9826.77 |
| lgbm_G3      |              0.036  |             0.118  |            0.4081 |        0.428  |        0.5059 |               1625.7  |

## Ensembles (OOF, cross-fitted)

| ensemble                             |   dti_crossfit_tf |   fold_min |   fold_max | path                                                                        |
|:-------------------------------------|------------------:|-----------:|-----------:|:----------------------------------------------------------------------------|
| mean                                 |            0.0757 |     0.0668 |     0.0882 | C:\Users\myben\Downloads\GEMS\outputs\benchmark\ensembles\ens_mean.tif      |
| rank_mean                            |            0.1125 |     0.0537 |     0.1281 | C:\Users\myben\Downloads\GEMS\outputs\benchmark\ensembles\ens_rank_mean.tif |
| weighted                             |            0.0815 |     0.0681 |     0.0951 | C:\Users\myben\Downloads\GEMS\outputs\benchmark\ensembles\ens_weighted.tif  |
| weighted(weights only, no transform) |            0.0612 |   nan      |   nan      | nan                                                                         |

## Caveats

- Geographic OOF DTI scores mapped USGS faults in held-out regions; the hidden test scores expert-mapped *new* faults.
- Phase-1 transform/scale choices used all labels (mild selection effect on G3/G4).
- Filters wider than the fold buffer see validation-area inputs (never labels): see `tables/leakage_audit.csv`.
- Known-fault policy for submissions (`model` vs `model_max_known`) is unresolved until leaderboard feedback.