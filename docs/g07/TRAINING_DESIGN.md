# G07 training design

All parameters live in `configs/g07/discovery.yaml` (compact, regularised; no open-ended search).

## Supervision (training-visible faults only)

* Classes (G06): POSITIVE (visible fault), IGNORE (outside region, <= 300 m of a visible fault), UNLABELED
  (PU), LIKELY_BACKGROUND (>= 3 km). Hidden faults are unlabelled pixels at their natural rate.
* Variants per fit (`datasets/subsurface|surface/labels/<fit>.parquet`): `weight_g06` (G06 design), `weight_improved`
  (equal total weight per visible system, unlabelled weight graded by distance 0.1 -> 1 over 3 km, hard negatives x2,
  background total = positive total), `proximity` = exp(-d / 0.3 km) (distance-aware regression).
* Candidate relevance (D1): near-truth share of TRAINING units' candidates (POSITIVE >= 0.5, NEGATIVE < 0.1, AMBIGUOUS
  excluded) + synthetic candidates: controlled perturbations of training truth (lateral shift <= 4 px, rotation <= 15
  deg, partial length) relabelled by their own near-truth share, and random segments >= 10 px from any fault
  (negatives). Synthetic rows train other units only and are never scored. Weights: one unit of weight per truth
  system, classes balanced exactly.

## Models

| task | experiments | data | model |
|---|---|---|---|
| A surface | G07-A1 (geomorphometry), G07-A2 (+ structure + agreement), G07-A1RF | surface pool (DEM-covered cells) | LightGBM / Random Forest, inner-fold early stopping |
| C1 | compact U-Net, 3 heads (centreline, cos 2t, sin 2t) | aligned 10 m patches, 2.4 km | weighted BCE + 0.5 Dice + 0.2 orientation MSE, rot90 / mirror with consistent orientation targets, inner early stopping |
| B subsurface | G07-B0 / B1 / B2 | subsurface pool, G06-P4 families | LightGBM classifier (B0, B1) / regressor (B2) |
| C candidate ranking | G07-D1 (geology / DEM only), G07-D1W (W2-aware) | candidate tables | LightGBM (CatBoost / lambdarank available) |
| D selection | G07-D2 portfolio, G07-E1 corridor | held-out maps | no fitted parameter beyond calibration |

Order: A0, A1 / A2, B0 / B1 / B2 first; U-Net only if A1 / A2 add systems vs W2; ranker / portfolio after candidate
recall shows complementary systems. Full-region models are FULL fits (all mapped faults visible), research only.
