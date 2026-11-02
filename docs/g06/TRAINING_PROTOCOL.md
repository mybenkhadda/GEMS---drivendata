# G06 training protocol

All hyper-parameters are fixed in `configs/g06/benchmarks.yaml` (compact, regularised; no search). Seeds come from
`CONFIG['seed']`.

## Pixel models (Track A: geology only) - `models_pixel.crossfit_pixel_model`

* Input: `pool_features.parquet` columns of the experiment's families (`store.by_family(families, 'A')`, usable
  channels only, no `phys_*`, no known-fault context) joined with `labels/<fit>.parquet` (weight > 0).
* Target: POSITIVE = 1; UNLABELED / LIKELY_BACKGROUND = 0 with PU weights (0.3 / 1.0, hard negatives 1.5); positives
  reweighted per visible system (sqrt length); background total = positive total; partial macro-region balance.
* Fits: `V1:d` (visible = labels - withheld(d), region = footprint), `V3:k` (labels outside fold k, region = fold-k
  train role), nested `V1:j|k`, `V3:j|k` for `validation.nested_pixel_experiments`, `FULL` (full-region research only).
* Early stopping (LightGBM / XGBoost): inner spatial fold 0 of the training region (buffer rows excluded), then a refit
  on every training row with the selected size. Random Forest: training medians impute NaN (stored with the model).
* Prediction: each fit predicts only its `predicts` units, tile by tile from the store, in the model's exact column
  order (schema hash checked) -> `predictions/oof/<exp>/<fp10>/[nested/]<fit>__<unit>.npy`.

## Segment models - `models_segment.crossfit_segment_model`

* Input: D3 rows merged with their labels; columns by track (S1: geometry, geology, orientation, permitted known
  context, uncertainty, G06 geology-model evidence; S2: + honest W2 / CNN evidence).
* Held-out unit (scenario s, zone z): trained on the other zones of s, candidates whose centroid lies in z's
  evaluation zone + 5 km dropped; AMBIGUOUS rows excluded; weights = length x label confidence, classes balanced.
* Training units get NESTED context: `kn_*` from `nested_known/<s>_<z>__<k>`, `ev_g06_*` from the nested pixel fit
  `s:z|k`; the held-out unit uses its own permitted context and its primary G06 prediction.
* Output: segment scores + a segment-assisted map that keeps the pixel ranking: (1 - lambda) pixel rank + lambda
  segment rank (S2 pixel base = W2; S1 = G06 pixel OOF, else a label-free physics feature).

## Patch U-Net (opt-in) - `models_patch`

Channels memmap; per fit: robust scaling fitted on training-region pixels, centreline target (visible faults dilated
1 px), PU loss weights from the pixel classes (IGNORE = 0), patches inside the training region, 50 % of every batch
from patches with visible positives, rotations / flips, AdamW, early stopping on inner validation patches, Hann-blended
overlapping inference over the predicted unit.

## Meta models - `models_meta.crossfit_meta`

Inputs ranked inside each unit: W2, CNN B0 / B1, G06 pixel (nested for training units), domain, |W2 - G06|.
`rank_average` = predeclared weights, no fit. `residual_lgbm` = LightGBM correction with init score logit(W2 rank).
`moe` = per-domain L2 logistic regressions blended with a global one by n_d / (n_d + tau). Rows: each training unit's
scored truth (positives) and scored pixels far from every mapped fault (negatives), rows within 5 km of the held-out
zone dropped.

## Full-region research model

`FULL` fit of `inference.experiments` (default G06-P5), predicted over the footprint, parity-checked against the
training rows. Never a submission by itself.
