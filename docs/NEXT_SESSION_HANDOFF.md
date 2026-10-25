# Next-session handoff (2026-10-25: experiment fidelity, reuse, cheaper Colab runs)

Code authored only. **Nothing was executed**: no notebook, script, test, training or download. Every runtime
claim below comes from the saved Colab notebook (commits fae15c4 / 8bccce0 / 88c97ed, the run at code commit
8bccce0 on 2026-10-22) or is marked as unverified.

## State found (saved notebook, 88c97ed)

| experiment | id | evidence in the saved log |
|---|---|---|
| `b0_reference` | `b0_reference__d54e938a` | 3 CV folds (17 epochs) + 3 discovery splits (11) trained, evaluated, exported; train 57 min. Evaluated with **1 map**: its control `P4_B_gated` is not on Drive, so its delta is `INCOMPARABLE` |
| `b1_bg_weight` | `b1_bg_weight__7755635b` | failed: `B0_lineament` map not available |
| `b2_nnpu_p010` | `b2_nnpu_p010__e0173a9a` | complete (2 maps: candidate + `P5_B0_gated`); train 36 min |
| `c1_geometry_heads` | `c1_geometry_heads__9a77180d` | complete (2 maps); train 55 min |
| `e1_gsn_parity` | `e1_gsn_parity__b8b8c3db` | fold 0 at epoch 14 when the log ends; completion not established |

Baseline status in that run: every legacy baseline (B0_lineament, G1, G3, RS05, CNN_A, DL00, DL02, P4_B_gated)
= `NOT_AVAILABLE` ("no run folder"), and `train_baselines` was empty. Feature store: 139 caches (raw 19,
phase2 48, phase2_eda 5, phase4_rs 67).

## Findings

* **B1 root cause.** The `B0_lineament` recipe exists (label-free: rank-normalised `eda:fused_score`, CPU), but
  nothing built it: it was not in `train_baselines`, and `check_prerequisites` only raised. Fix:
  `experiments.ensure_label_free_map`. It looks for the map in the session context, then on Drive (staged +
  `baseline_status` verification: run finished, native grid, fold provenance). Otherwise it builds the map with the
  original recipe (`baselines.train_baseline`), pushes it, and re-verifies. Only `losses.pu.LABEL_FREE_MAPS` are
  accepted. An `INCOMPATIBLE` copy is never overwritten. The earlier failure stopped in the prerequisite check,
  before `Experiment` opened, so no B1 folder should exist. A FULL B1 run would use `b1_bg_weight__7755635b` (its
  config hash is unchanged); B1 at SCREEN gets its own `__screen__` folder.
* **E1 audit.** `e1_gsn_parity` *is* routed to `GeoStructureNet`: `type: geostructurenet`, gated fusion, no
  long-range block, no context. Widths 24/32 equal P4_B_gated's legacy spec (`dual`, enc 24, dec 32, gated,
  context false). `GeoStructureNet.__init__` registers the same blocks (`Encoder` ×2 → `GatedFuse` ×5 →
  `Decoder` → head) in the same order as `GeoUNet` with the same seed, so the initial weights and the forward pass
  are identical. The identical fold-0 losses and inner DTI (14 epochs, 4 decimals) are therefore the **expected
  parity result**, not a routing bug. No model was changed. E1 is now catalogued as `role="parity_check"`, removed
  from the notebook's default candidate list, and given a `PARITY_CONFIRMED` shortlist decision. A full E1 run
  costs about 60 GPU min and adds no evidence beyond parity. Test:
  `tests/test_fidelity.py::test_e1_parity_model_is_numerically_identical_at_init`.

## Changes in this session

New modules:

* `gems_bench/pipeline/fidelity.py`: SCREEN / VALIDATE / FULL profiles, validation, `apply_fidelity`
  (semantic block, FULL = YAML unchanged), deterministic leakage-free block subsampling, mask digests, protocol
  hash, eligibility.
* `gems_bench/pipeline/inventory.py`: read-only `experiment_inventory` (profile, compatibility against the
  current config, units cross-checked with final model files, metrics, reusable, next action),
  `remaining_units`, `last_checkpoint`.
* `gems_bench/pipeline/screening.py`: `evaluate_fidelity_run` (proxy metrics, no registry),
  `metrics_comparison`, `paired_deltas` (identical protocols only; `NOT_EVALUATED` for missing values),
  `resource_usage`, `shortlist` (configurable triage heuristics, recorded reasons), `export_screening_report`
  (→ `reports/experiment_screening/`), `research_dashboard`.
* `gems_bench/pipeline/resource_budget.py`: static GPU-memory estimate → `data_on_gpu` placement; GPU-minute
  estimates per profile.

Modified:

* `experiment/config.py`: `profile`; non-FULL ids `<name>__<profile>__<hash8>`.
* `experiment/runners/cnn5.py`: budget caps, selected folds / splits, fidelity masks, checkpoint policy from
  `runtime.checkpoint`, one shared tensor stack per stage, non-FULL maps only in `manifest["fidelity_maps"]`,
  proxy evaluation below FULL.
* `experiment/core.py`, `artifacts/store.py`: contract stubs say `not_evaluated` below FULL; ledger flush on
  persist.
* `training/cv.py`: `fold_ids` / `split_ids` / `fidelity` / `ckpt_policy`; integrity-checked unit skip
  (`final_ok`); `fidelity_coverage.csv`; stack placement honours `data_on_gpu`; unit-scoped Drive pushes.
* `training/checkpoint.py`: `best_and_latest` policy (latest + prev + best, pointer sha256, fallback, legacy
  `_ep` resume). `rotating` is kept as the class default for older callers.
* `training/trainer.py`: `improved()` (NaN / inf never best).
* `artifacts/sync.py`: include / exclude filters, `STAGE_SCOPES`, `push_stage`, throttled Drive ledger +
  `flush()`.
* `features/__init__.py`: `FeatureStore.drive_mirror` lazy staging (copy on first read).
* `pipeline/config.py`, `configs/pipeline/master.yaml`: `experiments.{profile, fidelity_overrides, screening,
  auto_provision_maps, checkpoint}`, `features.stage`; flat keys `experiment_profile`, `fidelity_overrides`,
  `screening_rules`, `checkpoint`, `auto_provision_maps`, `feature_stage`.
* `pipeline/experiments.py`: roles, `control_stem`, `session_fidelity`, profile-aware `experiment_config`,
  `ensure_label_free_map`, `plan_experiments` (reuse-first; adds matched controls only when needed).
* `pipeline/stages.py`: `advanced_models` = `always_run` (plans, trains only RUN / RESUME, writes the
  screening report); profile-aware validate / fingerprint (FULL fingerprint unchanged); lazy feature
  validation / stage-in.
* `pipeline/__init__.py`: `pipe.inventory()`, `plan_experiments()`, `compare_results()`, `shortlist()`,
  `research_dashboard()`, `ensure_map()`.
* `pipeline/context.py`: lazy feature mirror on `ctx5`; ledger flush after stage pushes.
* `runtime/colab.py`: `stage_features(names)`.
* Notebook `GEMS_MASTER_PIPELINE.ipynb`: fidelity controls (SCREEN default, E1 out of the candidate list,
  `feature_stage='lazy'`), section 14–15 rewritten (inventory → compare → plan → step), a new dashboard /
  promotion cell. Saved outputs were cleared; the run with outputs is commit 88c97ed.
* Tests (authored, **not run**): `tests/test_fidelity.py`.
* Docs: `EXPERIMENT_FIDELITY_AND_COST.md` (new), `MASTER_COLAB_GUIDE.md`, `PIPELINE_DEPENDENCIES.md`,
  `ARTIFACT_RECOVERY.md`, this file.

## Architectural decisions

1. FULL ≡ the committed YAML: no fidelity block, the same config hash, the same ids, so old runs stay reusable
   and resumable. Fidelity applies to `cnn5` only.
2. Fidelity is semantic (it changes the id); the checkpoint policy and cadence are runtime (they never change
   the id).
3. Proxy metrics never share a table, map registry or gate with section-17 results.
4. `training_fraction` defaults to 1.0 in every profile: area does not drive GPU cost (documented).
5. Shortlists recommend; moving up a level and promoting for submission are explicit human steps.
6. `advanced_models` is always run when enabled: the planner, not the executor fingerprint, decides reuse.

## Run next

1. Locally: `pytest tests/test_fidelity.py tests/test_pipeline_engine.py tests/test_pipeline_recovery.py
   tests/test_phase5_infrastructure.py tests/test_phase5_science.py`.
2. Colab (GPU): open the master notebook from `main`, keep `experiment_profile='SCREEN'`, run all. Expected plan:
   `b2_nnpu_p010` / `c1_geometry_heads` → `REUSE_FULL`; `b1_bg_weight` → `RUN` (B0_lineament built first, CPU);
   `b0_reference` → `RUN` at SCREEN (matched control).
3. Read `reports/experiment_screening/SCREENING_REPORT.md`, then promote shortlisted names to VALIDATE
   explicitly.

## Unverified at runtime

* The whole fidelity path (caps, masks, proxy evaluation, `fidelity_maps`) on Colab.
* `BinaryCurve` on a block-subsampled mask (bbox padding near block edges) and on `withheld & labels` truth.
* `best_and_latest` resume on Drive FUSE (os.replace of a 136 MB file, pointer sha256 timing).
* Lazy feature staging: `has()` true for Drive-only files during `ensure_channels` / `build` skips; the first-read
  copy time on FUSE.
* `placement_overrides` thresholds (estimates; the T4 run measured peaks in `gpu_memory_profile.csv`).
* `ensure_label_free_map` → `train_baseline('B0_lineament')` on the rebuilt EDA (`eda:fused_score` present?).
* Whether `phase4_context` resolves `B0_lineament` from the regenerated canonical folder (expected:
  `b0_lineament_fused_s2026/oof.tif`).
* Dashboard Drive-usage walk: rglob over `experiments/` on FUSE may take tens of seconds.

## Known limitations / risks

* SCREEN (5 epochs, 25 % of one fold) is noisy. A `REVIEW` is inconclusive, not negative. PU / consistency ideas
  with ramp-ups may need a larger `max_epochs`.
* `b0_reference`'s own FULL control (`P4_B_gated`) is absent from Drive, so its trainer-variance delta stays
  `INCOMPARABLE` until the legacy Phase-4 folder is uploaded or P4_B_gated is retrained (GPU).
* Drive still holds the old `_ep005.pt` resume files of the first run (never deleted automatically, ~136 MB
  each).
* Earlier risks remain: signed URLs in history (expired), rebuilt EDA ≠ historical EDA, regenerated folds vs
  local folds, `SubmissionRunner` supports only `cnn5` / LightGBM champions.
