# G06 validation protocol

## Units (the GEMS units, `geomoe.scenarios.build` on the frozen engine)

| scenario | task | truth | role |
|---|---|---|---|
| V1 | `disc_official` | withheld complete fault systems of discovery split d; retained faults masked pixel-exactly, masked predictions ignored | PRIMARY |
| V3 | `geo` | all mapped faults of geographic fold k | secondary (geographic generalisation) |
| V2 | `v2_net` | artificially hidden network elements inside fold k | diagnostic (completion), predicted by the V3:k fits |

## Training visibilities (fold isolation)

* A fit sees only its visible labels; hidden labels are never positives, never negatives and never a sampling stratum
  (label-stratum rows on hidden faults are dropped; hidden pixels in the background stratum are ordinary unlabelled
  pixels at their natural rate - the PU setting of the real task).
* V1 default region = the whole footprint (deployment analogue: every visible fault everywhere);
  `outside_eval_zone` removes the evaluation zone + buffer (stricter, less data).
* V3 region = the fold's TRAIN role (fold and its 13 km buffer removed).
* Nested fits `j|k` hide the truth of both j and k; they produce the inner predictions strict meta / segment models of
  outer j train on.
* `splits.verify_fold_isolation` tests, per fit: visible / hidden disjoint; no scored truth pixel of a predicted unit
  visible; geographic regions do not touch the evaluation zone; the unit's permitted known context excludes its truth;
  for nested fits the outer unit's truth is invisible. A single FAIL fails the splits stage.
* Inner folds (early stopping): 20 km spatial blocks of the training region balanced on visible positives, 2 km
  buffer removed from inner validation. Outer labels never fit preprocessing, feature selection or model choice.

## Incomplete labels

Unmapped != negative. Classes POSITIVE / IGNORE (300 m halo, outside region) / UNLABELED (PU weight 0.3) /
LIKELY_BACKGROUND (>= 3 km). Segment and connection targets are evaluation-side and used only to train on OTHER units.

## Strictness labels

| label | meaning | promotable |
|---|---|---|
| `STRICT_ISOLATED` | pixel / patch model fitted on its fit's visible labels only, no upstream supervised input | yes |
| `STRICT_LABEL_FREE` | no fitted parameter (G06-M1 rank average with predeclared weights) | yes |
| `STRICT_NESTED` | learned on other units with nested inputs everywhere (S1 with label-free pools, G06-only meta) | yes |
| `SCREENING_ONLY` | uses W2 / CNN inputs whose training saw other units' truth (S2, residual / MoE meta), or incomplete nested inputs | no - at best `PROMISING_SCREENING_ONLY` |
| `REFERENCE` | W2 | - |

With the default candidate pools (W2 + CNN + physics) S1 is SCREENING_ONLY because the candidate generator uses
W2 / CNN maps; set `segment_candidates.pools` to `{physics, g06}` for a strict S1.

## Comparison with W2

* W2 = the honest held-out P6 map of each unit, evaluated by G06 with exactly the same scorer, budget domain, reps and
  fracs. The historical 0.282538 / 0.242156 / 0.130868 came from other representation rules; the public 0.0942 from
  the hidden test - none is compared directly.
* Fixed policy: `evaluation.primary` (top 1 % of the eligible domain) for every model. Representation selection is
  reported separately and nested (leave one zone out: the cell is chosen on the other zones).
* Pooled DTI = summed components (never a mean of zone DTIs); worst zone and per-zone always reported; systems
  recovered = support >= 0.5 on >= 50 % of the system's scored pixels; new / lost vs W2 per unit.

## Promotion (all required, at the primary cell)

V1 gain >= +0.005 (recommended +0.010); no zone drop > 0.01; V3 delta >= -0.005; net extra V1 systems >= 1; strict
validation; export parity (checked at export on the written raster). Export additionally needs
`RUN['EXPORT_SUBMISSION']`, a full-region raster of the model and parity PASS; the file is
`READY_FOR_MANUAL_UPLOAD` and is never uploaded.
