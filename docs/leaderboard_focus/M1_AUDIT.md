# M1 repository audit (read-only, 2026-10-28, commit `300d99e`)

Written before any M1 code. Sources: the code under `gems_bench/p6/`, the configs under `configs/p6/`, and the Colab
outputs saved inside `notebooks/colab/GEMS_P6_BOOST_FULL_GPU.ipynb` (run of 2026-10-27/28). Nothing was executed
locally.

## 1. What exists and is reusable

| need | existing implementation |
|---|---|
| B0 / B1 CNN maps | experiments `b0_reference__d54e938a` (`P5_B0_gated`), `b1_bg_weight__7755635b` (`P5_B1_bgw`), profile FULL, `predictions/{oof,disc0,disc1,disc2}.tif`. These IDs were seen in the saved run log; M1 rediscovers them by pattern at run time. |
| W2 stacker | `outputs/p6_fast_track/stack/`: `candidate.json` (gate PROMOTE), `features/rows.parquet`, `maps/{geo,disc_official}_{0,1,2}.npy` (cross-fitted held-out maps), `grid.csv`, `maps/full_score.tif` (historical refit) |
| saved benchmark grid | `outputs/p6_fast_track/boost/benchmark/grid.csv`: official components per (candidate, blend, rep, frac, unit), protocol keys |
| P6.1 recovery records | `outputs/p6_fast_track/recovery/export_parity/{decision.json, policy_comparison.csv, per_zone_metrics.csv, train_inference_parity.csv, unit_overlap.csv}` |
| one-interface nested evaluation | `p6.export_parity.evaluate_policy`, `p6.full_benchmark.{normalize_grid, protocol_check, evaluate_candidates, choose}` |
| zone-stitched full-region map, recipe, GeoTIFF, validation | `p6.final_submission.{build_map, recipe, write_file, validate, readiness, export_one, inspect_spec}`, `p6.export_parity.{build_recipe, plan_audit, audit_export}`, `gems3.submission.{validate, finalise, eligible_mask}` |
| official scorer | `gems3.official.FastScorer` via `m7.protocol.score_row` (`p6.boost.Boost._eval_q`) |
| leaderboard log helpers | `p6.leaderboard` (P6-specific manifest schema) |
| Drive staging | `artifacts.sync.DriveSync` (`stage_in` / `push`, skip unchanged, ledger) |

## 2. Run evidence in the saved notebook outputs

* The start-up check passed (commit `300d99e…`, T4 GPU).
* The artifact audit passed: 18 REUSE, 4 RECOMPUTE (optional boost B–D, G02), 2 INCOMPATIBLE, no blocking item. The two
  incompatible artifacts:
  * `W2 full-region score (footprint-ranked refit)`, note "recovery parity: w2_score BLOCKED on geo units";
  * `historical W2 submission (mixed rep / frac)`.
* The GPU stage (W1 extra seeds) was **interrupted**. The last log line is `b1_bg_weight_s2027` on discovery split 0,
  epoch 5. The cell has no execution count.
* **The benchmark, selection and submission cells never ran.** No p6g submission exists, and no public score exists in
  the repository.

## 3. The W2 parity blocker: cause, from code and the recorded table

The historical export (`p6.stack.full_region`) has three independent defects. Each is visible in the code; the first
is also confirmed by the recovery table.

1. **Feature-transformation mismatch (rank domain).** In training and evaluation every ranked feature (CNN ranks,
   potential-field ranks) is ranked inside each unit's evaluation mask. In `full_region` they are ranked inside the
   whole footprint (`rank_in(maps[0], valid)`). The recovery table `train_inference_parity.csv` records `w2_score`
   and `cnn_disagree` as BLOCKED on geo units.
2. **In-sample score.** `full_region` refits LightGBM on *every* primary-task row
   (`T[T.task == prim]`), so the full-region score at a zone was produced by a model that saw that zone's rows. The
   held-out maps used for validation are cross-fitted. This is an OOF / in-sample mismatch.
3. **Mixed policy.** The uploaded historical file took rep and blend from the discovery choice but frac from the
   geographic choice (`area_frac_source: geo`, skeleton_1px @ 0.05). No nested procedure selected that cell.
   `stack.py` now refuses it unless `allow_mixed_policy`.

**Consequence.** `stack/maps/full_score.tif` and the historical submission cannot be the M1 champion. They are kept
only for the parity diagnostic.

**The validated construction already exists and is unexecuted.** `p6.final_submission` (p6g) builds each geographic
fold zone from the cross-fitted held-out maps evaluated on that zone, ranks inside the zone exactly as in
evaluation, and applies ONE all-zone (rep, frac, blend) cell. It avoids all three defects without new inference.

## 4. Candidate inventory (recorded values, not recomputed)

Most values come from `configs/p6/recovery.yaml` → `references`. The disc and geo values there come from **separate**
nested selections, so they are not one deployable policy.

| candidate | space | recorded disc | worst | recorded geo | note |
|---|---|---|---|---|---|
| W2 original (`stack/candidate.json`) | blend × rep × frac (stack grid) | 0.224640 | 0.153414 | 0.180843 | separate selections |
| W2 crest | boost A, crest_1px | 0.238882 | 0.192575 | 0.185609 | separate selections |
| W2 skeleton | boost A, skeleton_1px | 0.242156 | 0.198579 | 0.168841 | separate selections |
| W2 hybrid crest | boost A | 0.231509 | 0.156216 | 0.182415 | |
| historical uploaded cell | fixed skeleton @ 0.05, blend 1 | 0.178905 | – | 0.168841 | not a nested choice |
| W2 skeleton, ONE policy, objective disc | P6.1 recovery run | 0.242156 | 0.198579 | 0.105687 | memory record of that run |

P6.1 recovery selected nothing. The geo floor of 0.1758 failed under the single-policy objective, and parity was
BLOCKED.

The p6g run later replaced the absolute floor with a relative geo rule (`configs/p6/full_gpu.yaml` →
`selection.geo_rule.mode: relative`). That change was accepted in the previous session. M1 keeps the p6g rule by
default and reports the 0.1758 floor as a separate flag: whether the floor blocks the champion is a user decision
(`configs/leaderboard_focus/m1.yaml` → `selection.geo_floor_blocking`).

## 5. M1 design decision

M1 is a thin orchestration layer over p6g in `mode: recovery` (CPU, no training):

1. **W2-family selection.** The p6g rule `full_benchmark.choose` runs on the saved boost-A grid, restricted to the W2
   family plus the B0+B1 fusion fallback.
2. **Validation.** Before any export, M1 independently recomputes the chosen cell's per-unit official components and
   checks that the recorded values reproduce.
3. **Export.** `final_submission.export_one` writes the zone-stitched export under
   `outputs/leaderboard_focus/m1_champion_recovery/`, with stricter M1 gates.

No model is retrained or re-run, and the boost grid is not recomputed.
