# P6 full GPU benchmark, selection and DrivenData submission (p6g.1)

Code: `gems_bench/p6/{gpu_runtime,full_benchmark,final_submission,leaderboard,full_figures,full_gpu}.py` ·
config `configs/p6/full_gpu.yaml` · notebook `notebooks/colab/GEMS_P6_BOOST_FULL_GPU.ipynb` · tests
`tests/test_p6_full_gpu.py` · audit `docs/p6/P6_RECOVERY_TO_BOOST_AUDIT.md` · outputs `MyDrive/GEMS_2/outputs/p6_boost/`.

**Nothing in p6g.1 has been executed.** No test, notebook or data step ran in the authoring session. No metric exists
until the Colab run writes it.

## 1. What one run does

```text
startup (GPU check) -> stage inputs -> audit -> GPU: W1 seeds + TTA -> [boost B/C/D] -> benchmark + selection
  -> GENERATE AND VALIDATE FINAL DRIVENDATA SUBMISSION -> figures -> report -> (manual upload) -> leaderboard record
```

## 2. Design decisions (and why)

* **One evaluation interface.**
  * Every candidate is a policy space over saved grid rows evaluated by `export_parity.evaluate_policy`: per outer
    zone, ONE cell is chosen on the other zones and applied to that zone in both tasks.
  * Grids are pooled only when, unit by unit, their protocol keys and truth mass equal those of boost A
    (`protocol_check`).
  * A candidate missing a zone is `INCOMPLETE`, never pooled on fewer zones. Unexecuted candidates carry a status and
    the evidence for it, never a number.
* **Selection objective: `disc`.** The P6.1 run showed the overlap exclusion is infeasible: any geo-aware selection
  would see the held-out geography's labels. `disc_geo_floor` / `joint` are reported as sensitivity only.
* **Selection rule (predeclared, `configs/p6/full_gpu.yaml` → `selection`).**
  * The primary criterion is nested disc_official DTI.
  * Eligibility conditions:
    * worst zone ≥ 0.153414;
    * geo DTI ≥ reference geo − 0.010;
    * the candidate is one deployable policy with a full-region builder;
    * ensembles must beat their best member by 0.005 without a lower worst zone.
  * Replacing the strongest reference needs a gain of ≥ 0.005 and being no worse on ≥ 2 of 3 zones. A leaderboard
    slot needs ≥ 0.010.
  * **This changes the P6.1 default**: the absolute geo floor 0.1758 is no longer a gate. That floor failed every W2
    policy under the honest objective (geo about 0.10–0.11). Geo DTI scores rediscovery of *mapped* faults, which is
    not the official target. `selection.geo_rule.mode: gate` restores the P6.1 rule.
* **Zone-stitched submission (`final_submission.py`).**
  * Each geographic fold zone (`valid & fold_map == k`, disjoint) is predicted from the maps evaluated on that zone:
    * B0 / B1 OOF;
    * the cross-fitted held-out stacker map (W2 or a boost variant);
    * or the W1 seed families.
  * All maps are ranked inside the zone, exactly as in evaluation. The policy cell is applied per zone on the
    full-region eligible domain.
  * This replaces the historical construction, which parity BLOCKED in the recovery run.
  * Parity is re-checked per zone: Spearman of the stitched score crop vs the evaluation construction ≥ 0.999.
    `BLOCKED` fails validation.
* **GPU work that is validatable.** W1 extra seeds are trained by the same committed experiments and produce OOF and
  discovery maps, so seed ensembles are benchmarked like everything else. rot180 TTA uses the same checkpoint that
  made each honest map. Fold-average full maps are opt-in diagnostics only (each fold model saw the other zones).
* **Not buildable full-region.** Boost D hybrids and X3 H1 / H2 are post-hoc map combinations with no validated
  full-region builder. They are benchmarked but never selected (`deploy: none`).

## 3. Exact Colab instructions

Once, locally (by you):

```bash
git switch -c p6-full-gpu
pytest -q tests/test_p6_full_gpu.py tests/test_p6_recovery.py tests/test_p6_boost.py
ruff check gems_bench/p6 tests/test_p6_full_gpu.py
git add gems_bench/p6 configs/p6/full_gpu.yaml notebooks/colab/GEMS_P6_BOOST_FULL_GPU.ipynb \
        tests/test_p6_full_gpu.py docs/p6/P6_FULL_GPU*.md docs/p6/P6_RECOVERY_TO_BOOST_AUDIT.md
git commit -m "P6 full GPU: one-interface benchmark, selection, zone-stitched validated submission"
git push -u origin p6-full-gpu && git rev-parse HEAD   # put the SHA in CONFIG['ref'] and EXPECT['commit']
```

Colab:

1. Open `notebooks/colab/GEMS_P6_BOOST_FULL_GPU.ipynb`. Choose Runtime → Change runtime type → **GPU** (L4 / A100 if
   available) and **High-RAM**. The recovery preflight measured 11.5 GB available against 24 GB wanted.
2. Section 0: set `CONFIG['ref']` and `EXPECT['commit']`, and keep `mode='full'` and `M9_CONFIG['profile']='PILOT'`.
3. Run sections 0 → 11 in order. Section 2 raises `GPURequired` without a GPU in mode `full`; switch to
   `mode='recovery'` to benchmark and export without training.
4. Optional, for the geological / wavelet priors:
   1. set `g02_backfill=True`;
   2. re-run section 4 (the audit must say `READY_FOR_BACKFILL`);
   3. set `boost_geology=True`;
   4. re-run sections 7 and 9.

| section | expect | estimate (unmeasured) |
|---|---|---|
| 3 attach | ~4.5 GB staged (M9 layers, G01 / G02, P6 workspace) | 8–12 min (recovery run: 376 s + 86 s + 43 s) |
| 5 GPU | `seeds/train_log.csv`, `seeds/maps/*rot180.npy`, `seeds/grid.csv` | bounded by the W1 cap of 4 GPU-h |
| 7 benchmark | `benchmark/model_selection.json` | minutes (tables only) |
| 9 submission | `submissions/submission_p6_boost.tif` + manifest / validation | 5–20 min CPU (crest maps per zone) |

## 4. Outputs (`outputs/p6_boost/`)

* `startup/runtime.json`: device, CUDA, bf16, seeds, package versions, memory.
* `audit/artifact_audit.csv`, `audit/decision.json`.
* `gpu/gpu_log.csv`, `gpu/decision.json`, and `gpu/maps/*fold_average.tif` (optional).
* `benchmark/`:
  * `candidate_metrics.csv` / `.parquet`, `zone_metrics.csv`, `candidate_rankings.csv`, `candidate_ablations.csv`;
  * `objective_sensitivity.csv`, `protocol_consistency.csv`;
  * `experiment_manifest.json`: per candidate, the grid sha256, producing-stage fingerprint and commit, checkpoints,
    config fingerprint, input-data fingerprint and code commit;
  * `model_selection.json`: rule, winner, baseline, eligibility reasons, META.
* `submissions/`:
  * `submission_p6_boost.tif` + `.json` sidecar;
  * `submission_p6_baseline.tif` (strongest existing reference, same construction);
  * `<stem>_{manifest,validation,plan}.json`, `<stem>_diagnostics.csv`, `<stem>_sha256.txt`, `<stem>_preview.png`;
  * canonical `submission_manifest.json`, `submission_validation.json`, `submission_preview.png`,
    `submission_diagnostics.csv`, `submission_sha256.txt`, `official_spec.json`, `submission_index.csv`;
  * `rasters/<stem>_maps.tif` (band 1 zone-stitched score, 2 binary map, 3 zone id).
* `figures/`: `input_layers`, `prediction_maps` (winner, baseline, difference), `heldout_errors` (hit / miss / FP on
  the held-out discovery units), `zone_dti`, `performance_vs_runtime`, `continuity`. Each is PNG + SVG +
  `_data.csv`, plus `winner_segments.parquet`.
* `leaderboard/`: `leaderboard_history.csv`, `leaderboard_comparison.csv`, `leaderboard_report.md`.
* `reports/`: `full_gpu_report.md`, `experiment_status.csv`.

## 5. Manual upload and leaderboard record

1. Upload only a file whose `submissions/submission_manifest.json` → `winner_status` is `READY_FOR_UPLOAD`.
2. Download `MyDrive/GEMS_2/outputs/p6_boost/submissions/submission_p6_boost.tif` and check that its sha256 equals
   `submission_sha256.txt`.
3. On the DrivenData competition page, submit the file under Submissions. The limit is 3 scored submissions per
   rolling 7 days.
4. Optionally upload `submission_p6_baseline.tif` in another slot, for a paired comparison.
5. In section 0, set `LEADERBOARD_ENTRY = {'submission': 'submission_p6_boost.tif', 'public_score': <score>,
   'date': 'YYYY-MM-DD', 'notes': '...'}`, then run sections 0–3 and 10.
   * The row is refused if the file changed (sha mismatch), already has a score, or the score is not in [0, 1].
   * `leaderboard_report.md` compares local and public results and lists possible causes of a gap.
   * The public board is a spatial subset: never tune on it, and never claim a private-leaderboard result.

## 6. What to inspect after the first real run

1. `audit/artifact_audit.csv`: nothing in `needed_by = every candidate` may be anything but REUSE.
2. `benchmark/protocol_consistency.csv`: every `ok` must be True; otherwise that grid is INCOMPATIBLE.
3. `benchmark/candidate_rankings.csv` and `model_selection.json`: winner, decision, reasons, and META compared with
   the winner.
4. `submissions/<winner>_validation.json`: all groups PASS. Check the `zone_stitched` parity rows and
   `uncovered_frac`.
5. `submissions/<winner>_diagnostics.csv`: per-zone area fraction ≈ `frac`, and 0 positive pixels on mapped faults.
6. `figures/heldout_errors.png` and `prediction_maps.png`.

## 7. Risks (first execution)

* API mismatches on real objects. The most likely spots:
  * boost-variant grid candidate names (`S1`, `C1`, …) and the `features` column of `boost/<stage>/ablation.csv`;
  * `M9Source` staging of `outputs/p6_boost` on the first run (the directory may not exist on Drive yet);
  * `seeds.family_maps` on disc units when TTA maps are missing;
  * `data.paths.raw('labels')` (recorded as None if unknown).
* Zone coverage: valid pixels outside every geo fold are written 0. Validation fails if they exceed 0.1 % of the
  footprint; inspect `coverage` in the manifest.
* Three zones: every nested choice sees two zones. Differences under about 0.005 are noise, and META is the honest
  number for "we picked the best candidate".
* The proxy (withheld mapped systems) is not the hidden truth: see `leaderboard.gap_causes`.
