# Known-context shortcut experiments (D1–D5)

The G14 input channels `known:dist_km`, `known:density`, `known:tip_*` and `known:strike_*` come from the unit's
permitted known map, and that map contains every training positive (S01). A network can reach a high inner score by
reproducing the input. None of the following treats a high segmentation score as proof of geophysical learning.

## Static measurements: CPU only, R02

| measurement | meaning | shortcut signature |
|---|---|---|
| `train_pos_in_context` | share of training positives that are context pixels | ≈ 1 for every original G14 unit |
| copy baseline | region DTI of "the context raster itself" (and its 300 m kernel) on the inner region | close to G14's inner 0.85 |
| separability | AUC of −distance-to-context for training targets (training pixels) vs held-out truth (held-out pixels) | train ≈ 1, held-out much lower |

## D1: known-context-only model (R08)

Spec `D1_known_only`: G14 supervision, loss and augmentation, with the known channels as the only input. If its inner
score approaches G14's, the inner metric measures copying.

## D2: geophysics-only model (R09, M1)

Spec `M1_geophysics_only`:

* channels: raw, G13 and G14 composites; no `known:*` channel at all;
* targets: separated;
* objective: weighted BCE;
* augmentation: conservative.

It is the reference that every one-factor ablation is compared against.

## D3: counterfactuals on the existing checkpoint (R06, inference only)

`leakage.counterfactuals` reloads `T1_local_cnn__da3f44d05d84` unit models with their own normaliser and the rebuilt
stack. The schema hash is verified, so a mismatched stack refuses to load. The interventions are:

| intervention | what changes |
|---|---|
| original | nothing |
| disabled | availability mask of the known modality = 0 (the modality encoder zeroes its features) |
| neutral | known channels = training median (normalised value 0) |
| shifted | context translated 20 px (2 km) |
| subset | random 50 % of context components |
| target_removed | inner-validation faults removed from the context |

Every intervention is scored on the inner region (truth = inner faults; also with context-excluded scoring) and the
held-out region, with official region components, plus probability-vs-distance curves.

`target_removed` is the decisive test: it is the only intervention that makes the inner faults look like official
test targets. The other interventions create distribution shift, so a drop under them is evidence of dependence, not
proof of leakage.

## D4: dependence measurements (R02, R04, R06)

* Probability by distance to known faults (`probability.distance_curve`).
* Discovery recall by distance of the held-out truth to the nearest permitted fault
  (`metric_audit.recall_by_known_distance`).
* Mass near mapped traces and mass removed by the pixel-exact mask (`probability.summary`).
* Changes under interventions (R06).
* Performance without target-derived inputs (M1).
* Performance across isolated fault systems (atlas category `isolated_structure`).

## D5: conditional discovery supervision (R10, M2)

`discovery.ConditionalDiscoveryUnit` implements Y_target = Y_visible \ Y_context.

**Systems:**

* The unit's visible faults inside its training geography are grouped by `g14.splits.fault_system_groups`: linked
  only if they touch, or if strike differs by ≤ 30° and the gap is ≤ 1 km and collinear.
* Each epoch, the systems are shuffled within strata of 4 consecutive sizes. `context_fraction` (default 0.5) of each
  stratum goes to the context, and at least one system goes to each side.

**Inputs:**

* All `known:*` channels are recomputed from the context map only (`leakage.set_context`).
* Target systems never enter any channel.
* Faults outside the training geography are context only.
* `known_dropout = 0.2` sometimes removes the context entirely.

**Supervision** (`discovery.supervision_masks`, all distances in metres):

* positives = target systems in the embargoed training region;
* context fault pixels carry no loss, because the official scorer masks them;
* pixels around context faults are ordinary negatives, so painting the shoulders of known faults costs FP;
* no negative within 300 m of a target system (the positive-unlabelled ring);
* reliable negatives lie more than 1 km from every visible fault.

**Evaluation:**

* Inner: context = every training-geography fault; inner faults hidden and scored.
* Held-out: the unit's permitted map (never its held-out truth).
* Final: every mapped fault.

**Comparison:** M2 vs M1 on identical units and folds, plus M3 (combined targets) and M0 (original) as controls.
