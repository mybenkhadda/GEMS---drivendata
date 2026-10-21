# Validation protocol

## Metric

Distance-weighted Tversky index, α = 0.2, β = 0.8, R = 300 m, k(d) = max(0, 1 − d/R) (`gems_bench.metric`, from the
problem statement). ε, NaN handling, clipping, the scored region and the known-fault convention are **not** fixed by
the organisers: A2 (`a2_parity.yaml`) checks the local scorer against an independent KD-tree implementation and,
when a local copy of the official scorer is provided, against it. Until that passes, local DTI is only "the
problem-statement DTI", not "the leaderboard DTI".

DTI is maximised by a binary map, so models are compared by their **ranking at an area budget**, chosen by
cross-fitting (on the other folds / splits): "area-selected DTI". The raw-probability DTI is reported separately.

## Score families (never mixed, never substituted)

| column | meaning |
|---|---|
| `geo_dti_area` | geographic 3-fold (20 km blocks, 13 km buffer), cross-fitted binary area |
| `geo_dti_raw` | same folds, raw probabilities |
| `disc_dti_excl` | pseudo-discovery: withheld fault systems; retained faults ±300 m excluded |
| `disc_dti_retained_fp` | pseudo-discovery: retained faults are not truth, predictions on them are FP |
| `short_recall_w`, `recall_w_len_*` | weighted recall by trace length at the matched budget (5 %) |
| `unique_systems_vs_G1_*` | withheld systems recovered (≥ 0.5) that G1 misses by ≥ 0.3 |
| `public_lb` | only when that exact model was submitted and logged |

Every table carries `protocol_hash` (folds, splits, budgets, policies, metric constants); results with different
hashes are not compared (`evaluation.protocol.assert_same_protocol`).

## Leakage rules

* CNN patches lie fully inside the training zone; targets only from labels visible in the training zone; withheld
  systems + 1 km ring have weight 0 (never negatives).
* Area budgets, ensemble weights, stacking, thresholds: chosen on training-side folds / splits only (nested).
  Ensemble discovery weights for split d exclude split d's withheld systems (`geo_masked`) or use the other splits.
* Label-free inputs may be precomputed on the whole map (transductive); buffer ≥ feature support is audited.
* Hard subregions are reporting strata only.

## A3 designs

`system_k5` (every system withheld once), `short_lt2km_k3` (only < 2 km systems withheld), `region_loro`
(leave-one-region-out, 13 km buffer). Use them to confirm a promising result, not to search for one.

## B: positive-unlabeled assumptions

The PU prior π is not identifiable and is swept. SCAR is violated (mapped faults are the long, well-expressed ones),
so the PU correction can over-forgive false positives; safeguards: non-negative clamp, collapse guard on the mean
probability of unlabeled pixels, confidence mass > 1 km from faults, matched-budget evaluation, comparison with B0
under an equal training budget.

## Promotion (section 17)

A candidate is promoted only if, against its control under one protocol: (1) primary pseudo-discovery (`excl`)
gains ≥ 0.005; (2) `retained_fp` does not drop by > 0.002; (3) geographic DTI does not drop by > 0.005; (4) ≥ 1
uniquely recovered withheld system; (5) far-from-fault share of the matched-budget mask rises ≤ 0.05; (6) worst
fold / split does not drop by > 0.005 and ≥ 2 of 3 splits improve; (7) all artifacts (OOF + every discovery map)
are saved. Training loss, visual appeal and novelty are never evidence.
