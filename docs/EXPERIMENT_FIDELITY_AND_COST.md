# Experiment fidelity and cost: SCREEN → VALIDATE → FULL

GEMS research experiments (`configs/experiments/*.yaml`, kind `cnn5`) used to cost the same however weak the idea
was. One FULL experiment is three geographic CV trainings plus three pseudo-discovery trainings. Each runs up to
40 epochs of 150 steps, followed by full-resolution inference, the section-17 evaluation and Drive uploads. The first
advanced run on a Tesla T4 took ~39–42 s per epoch and roughly 35–60 GPU minutes per experiment (B0: 57 min of
training, B2: 36 min, C1: 55 min; read from the saved notebook log).

Fidelity profiles spend GPU time in proportion to the evidence an idea has earned:

```
SCREEN (triage) ──shortlist──▶ VALIDATE (intermediate) ──shortlist──▶ FULL (section-17 protocol, promotion gate)
```

Implementation: `gems_bench/pipeline/fidelity.py` (profiles, identity, masks), `pipeline/experiments.py`
(reuse-first planning), `pipeline/screening.py` (proxy evaluation, shortlist, reports),
`pipeline/inventory.py` (what exists), `pipeline/resource_budget.py` (memory / time estimates).

## Execution mode vs fidelity

| key | question it answers | values |
|---|---|---|
| `run_mode` | *which stages run* | `PREPARE_DATA`, `BASELINES`, `ADVANCED_RESEARCH`, `FULL_PIPELINE`, `RESUME`, … |
| `experiment_profile` (`experiments.profile`) | *how much evidence each GPU research run produces* | `SCREEN`, `VALIDATE`, `FULL` |

The two keys are independent. Neither changes the meaning of the other. The YAML default is `FULL`, which keeps
the historical behaviour. The master notebook selects `SCREEN`.

## The three profiles

| | SCREEN | VALIDATE | FULL |
|---|---|---|---|
| geographic folds | `[0]` (one real fold, its full buffer) | `[0, 1, 2]` | `[0, 1, 2]` |
| pseudo-discovery splits | none | `[0]` | `[0, 1, 2]` |
| max epochs | 5 | 10 | YAML (40, patience 6) |
| max steps / epoch | 80 | 150 | YAML (150) |
| training fraction | 1.0 | 1.0 | 1.0 |
| validation fraction (scored) | 0.25 of the held-out fold | 1.0 | 1.0 |
| inference | only the scored subset | full held-out regions | full |
| metrics | `proxy_*` (matched-area DTI, recall@300 m, collapse) | `proxy_*` per fold + discovery split 0 | section-17 scoreboard |
| leaderboard / registry / promotion gate | **never** | **never** | yes |
| typical GPU minutes (estimate) | ~3–5 | ~25–35 | ~35–60 |

GPU minutes are **static estimates** (`resource_budget.profile_cost`, from the epoch times recorded in the first
run). They are not benchmarks. Measured times are written to each run's `metrics/runtime_profile.csv` and to
`reports/experiment_screening/resource_usage.csv`.

Override any knob per profile (FULL accepts none, by design):

```python
CONFIG['fidelity_overrides'] = {'SCREEN': {'max_epochs': 4, 'validation_fraction': 0.5},
                                'VALIDATE': {'discovery_splits': [1]}}
```

Knobs: `geographic_folds`, `discovery_splits`, `max_epochs`, `max_train_steps_per_epoch`, `training_fraction`,
`validation_fraction`, `full_resolution_inference`, `train_block_px`, `eval_block_px`, `subsample_seed`,
`checkpoint_policy`.

### Why `training_fraction` defaults to 1.0

GPU cost is epochs × steps × batch. The size of the training zone does not enter it. Sampling patches from a
quarter of the training area saves no compute: it removes data diversity and makes "no admissible patch" more
likely in fragmented zones. The knob exists, with the leakage guarantees below, but the profiles control cost
through epochs, steps, folds, splits and the scored subset.

## Scientific rules

* **Immutable designs.** Every profile uses the session's fold and split files (`outputs/benchmark/folds/...`).
  Fidelity only *selects* folds and splits. It never regenerates them.
* **No leakage.** Training subsampling keeps whole `train_block_px` blocks of `folds.inner_train(k)`. The
  result is always a subset of the eligible training zone, so held-out regions and buffers are never touched.
  The patch sampler rejects any patch that touches a non-allowed valid pixel. Evaluation subsampling keeps
  whole `eval_block_px` blocks of `folds.val(k)`, `inner_val(k)` or `eval_zone(d)`.
* **Determinism.** Subsets depend only on (mask, fraction, `subsample_seed`, block size, unit salt), so every
  candidate and its control are scored on identical pixels. Each saved metric row carries the scored mask's
  sha256 (`mask_digest`). A paired delta is computed only when the digests and the fidelity `protocol_hash` match.
  Otherwise the result is `INCOMPARABLE`.
* **Matched control.** A SCREEN or VALIDATE candidate is compared with `b0_reference` trained under the *same*
  profile. The planner adds that control automatically, and only when a candidate needs it.
* **Effective budget recorded.** `metrics/fidelity_coverage.csv` holds eligible / used pixels per unit and zone.
  `reports/runtime_config.json` holds the effective epochs, steps, batch and placement.
  `reports/fidelity_protocol.json` holds the protocol record.
* **Proxy ≠ section 17.** SCREEN and VALIDATE metrics are per-fold DTI at the matched area budget
  (`evaluation.match_frac`, 5 %). The section-17 score is cross-fitted over folds. Proxy numbers are labelled
  `proxy_*`. They never enter `scoreboard.csv`, the model zoo, ensembles, the registry or the promotion gate.
  Maps of non-FULL runs are recorded under `manifest["fidelity_maps"]`, never `manifest["maps"]`.

## Identity and reuse

* A FULL config is the committed YAML unchanged, so its id is the historical one (`b0_reference__d54e938a`,
  `b2_nnpu_p010__e0173a9a`, `c1_geometry_heads__9a77180d`, `e1_gsn_parity__b8b8c3db`).
* SCREEN and VALIDATE add a semantic `fidelity:` block. This changes the config hash, and the folder becomes
  `<name>__<profile>__<hash8>` (for example `b1_bg_weight__screen__1a2b3c4d`). A cheap run can never overwrite or
  resume a FULL run. Two different screening budgets get two different folders.
* Before anything trains, `plan_experiments` reads the inventory and decides:

| action | meaning |
|---|---|
| `REUSE_FULL` | a compatible FULL run is complete: never retrained because a cheaper profile is selected |
| `REUSE_HIGHER` | a compatible VALIDATE run is complete and SCREEN was requested |
| `DONE` | the requested profile is complete |
| `RESUME` | incomplete: finished units are skipped, the current unit resumes from its checkpoint |
| `RUN` | nothing usable: train at the requested profile |
| `NOT_NEEDED` | a control at SCREEN / VALIDATE that no candidate needs |

## Shortlisting (triage, not acceptance)

`pipeline.screening.shortlist` records one decision per SCREEN or VALIDATE candidate, with its reason and the
rules used:

| decision | rule (defaults, `experiments.screening` to change) |
|---|---|
| `PROMOTE_TO_VALIDATE` / `PROMOTE_TO_FULL` | mean proxy DTI gain ≥ +0.005 vs the matched control, recall@300 m drop ≤ 0.02; VALIDATE: beats the control in ≥ 2 folds |
| `REVIEW` | in between, or numerically identical to the control (the change may be inactive) |
| `REJECT` | gain ≤ −0.010, or prediction collapse (> 50 % of scored pixels with p > 0.99, or mean p < 0.001) |
| `PARITY_CONFIRMED` | a parity check (E1) identical to its control within 1e-4 |
| `INCOMPARABLE` | no control at the same protocol, or the scored masks differ |
| `NOT_EVALUATED` | no proxy metrics saved |

These thresholds are triage heuristics chosen to be conservative relative to the section-17 rules
(`min_disc_gain` 0.005). They are not scientific acceptance criteria. A decision only recommends the next
fidelity level. Moving up is an explicit step: put the names from `pipe.shortlist('SCREEN')` into
`advanced_experiments` and set `experiment_profile='VALIDATE'`. Promoting a model for **submission** remains
the manual registry transition (`gems registry promote`).

## Checkpoints

`experiments.checkpoint` (non-semantic: never changes an experiment id):

```yaml
checkpoint: {policy: best_and_latest, save_every_epochs: 5, sync_every_minutes: 15}
```

* `best_and_latest` keeps `<unit>_latest.pt` (full resume state: model, optimizer, scheduler, scaler, EMA
  teacher, NumPy / torch / CUDA RNG, history, best-so-far), `<unit>_best.pt` (model-only best inner-validation
  state) and `<unit>_model.pt` (final, for inference). Files are replaced in place, so Drive holds one resume file
  per unit instead of one per checkpoint epoch.
* The previous resume file (`<unit>_latest.prev.pt`) is kept until the new one is written and, when syncing,
  verified on Drive (sha256, `.partial` + rename). The pointer `<unit>_latest.json` records the sha256. A
  mismatched or unreadable file falls back to the previous state. A file written by another spec is refused.
* Selection: CV units keep the best *finite* inner-validation score; NaN is never an improvement. Discovery
  units have no inner-validation zone by design (fixed epochs = median best CV epoch), so their final model is
  the last epoch and no `_best.pt` is written.
* Older runs with `_ep<NNN>.pt` files resume unchanged under the new policy. `rotating` keeps the old behaviour.

## Caching and Drive I/O

* **Feature caches:** `feature_stage='lazy'` (notebook default) stages only the registries before the feature
  stage. Each `.npy` is copied from Drive the first time a run reads it (`FeatureStore.drive_mirror`;
  validation reads the Drive headers). A CNN run touches its ~29 channels instead of all 139 (~7 GB). Arrays
  stay on local disk for the session and are memory-mapped. `'all'` restores up-front staging.
* **Ledger:** unchanged files are skipped by (size, mtime_ns). Large files are hashed once, at copy time, never
  re-hashed per session. The Drive copy of the ledger is written at most once a minute and at every stage
  boundary (`flush`). A stale Drive ledger after a crash only causes re-copies.
* **Stage scopes:** after each training unit only that unit's checkpoints and raster, the metric tables and the
  log are pushed (`DriveSync.push_stage` / `STAGE_SCOPES` for training / evaluation / reporting).
* **GPU memory:** `resource_budget.placement_overrides` estimates stack + aux + context + weights / optimizer +
  activations. The input stack stays on the host (patches moved per batch) when the estimate exceeds 85 % of
  VRAM. The stack is converted once per stage and shared by every fold and split. `data_on_gpu` is non-semantic
  (excluded from the spec hash).

## Launching from the master notebook

```python
CONFIG.update({'run_mode': 'ADVANCED_RESEARCH', 'run_advanced_models': True,
               'experiment_profile': 'SCREEN',
               'advanced_experiments': ['b0_reference', 'b1_bg_weight', 'b2_nnpu_p010', 'c1_geometry_heads']})
```

Run the configuration and pipeline cells, then section 14–15:

1. `pipe.inventory()` and `pipe.compare_results()`: read-only, no GPU.
2. `pipe.plan_experiments()`: B2 and C1 → `REUSE_FULL`, B1 → `RUN` at SCREEN, B0 → `RUN` at SCREEN (matched
   control).
3. `pipe.step('advanced_models')`: trains only what the plan says, then writes `reports/experiment_screening/`.
4. `pipe.research_dashboard()` and `pipe.shortlist('SCREEN')`.
5. VALIDATE: `CONFIG.update({'experiment_profile': 'VALIDATE', 'advanced_experiments': pipe.shortlist('SCREEN')})`.
   Re-run.
6. FULL: `CONFIG.update({'experiment_profile': 'FULL', 'advanced_experiments': pipe.shortlist('VALIDATE')})`.
   Re-run.

Resume after a disconnect: re-run the same cells with the same CONFIG. Finished units are skipped and the
current unit resumes from `<unit>_latest.pt`.

## Limitations

* Proxy metrics on a 25 % subset of one fold are noisy. Treat a SCREEN `REVIEW` as "inconclusive", not
  "negative".
* With 5 epochs, ideas that act late in training (PU priors with ramp-ups, consistency ramp-up 5 epochs) can
  look neutral at SCREEN. Raise `max_epochs` for those via `fidelity_overrides`.
* Fidelity applies to `cnn5` experiments. CPU kinds (LightGBM D1, DEM F4-G1) always run FULL.
* Nothing in this document is measured: every speed-up claim is an estimate until a run records it.
