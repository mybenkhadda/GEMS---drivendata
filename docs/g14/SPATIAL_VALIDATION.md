# G14 spatial validation

Implemented in `gems_bench/g14/splits.py` and `gems_bench/g14/evaluation.py`. **UNTESTED.**

## 1. Two backends (never mixed)

| backend | units | locked TEST | use |
|---|---|---|---|
| `g13` (default) | G13 `Splits`: 4 folds of 20 km blocks with 13 km buffer; fold 0 = locked TEST; 3 CV folds; 3 discovery splits | yes | G14 research ladder, final claims |
| `phase5` | `phase5_context.load_context5`: same geo folds / disc splits as P6 W2 and P7 | no | apples-to-apples comparison with W2 (T0) |

`UnitSystem` refuses a `phase5` context whose grid or labels differ from the loaded data. G13 model maps cannot be
loaded under `phase5`.

## 2. Units

Each `TrainUnit` holds:

* `train` (supervised geography);
* `inner_val` (1 in 5 blocks of 200 px, inside the training geography, early stopping only);
* `labels` (visible);
* `ignore`;
* `permitted_known`;
* `heldout`.

| protocol | held out | permitted known faults |
|---|---|---|
| `spatial_cv_embargo` (geo) | CV fold k | faults outside fold k, its buffer and TEST |
| `fault_system_holdout` (disc) | withheld systems of split d | retained systems only |
| `regime_holdout` | one label-free regime (if stable) | faults outside it |
| final | nothing (TEST excluded unless final export) | all faults (test-time input) |

## 3. Fault systems

`fault_system_groups` links components only if they touch, or if their strikes differ by ≤ 30° and the gap is
collinear within `link_m`. Distance alone never merges two faults. `system_crossing.csv` lists systems split across
folds.

## 4. Audit (`fold_audit.csv`, `assert_no_leakage`)

Per unit:

* overlap of train with held-out, test and inner validation (must be 0);
* held-out truth inside the loss (0);
* held-out truth inside the permitted known map (0);
* minimum train-to-held-out distance;
* train pixels whose regional window reaches the held-out zone (reported).

## 5. Scoring

* Official DTI through `gems3.official` units: radius 300 m, α 0.2, β 0.8, `disc_official` with known pixels
  masked, geo per fold.
* Representations: `top` (top-a area) and `crest_w1` (one-pixel crest).
* Area grid 0.005–0.3. Choices at the grid edge are flagged.
* **Cross-fitting:** the operating point for zone z is chosen on the other zones of the same task.
  `optimism = in-sample − cross-fitted` is reported.
* Block bootstrap (`g13.evaluation.bootstrap`) gives CIs and paired deltas against the champion.
* **Locked TEST:** `evaluate_locked_test` runs once and appends to `locked_test_log.jsonl`. A second look needs
  `force_second_look=True` and is labelled `SECOND_LOOK`.

## 6. What is NOT comparable

Public LB (G05 0.0942), W2 nested 0.242 (Phase-5) and G13 proxies (G13 units) come from different protocols. G14
compares models only within one backend.
