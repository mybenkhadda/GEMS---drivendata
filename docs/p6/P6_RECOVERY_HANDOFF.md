# P6.1 recovery — handoff (r1, 2026-10-27)

Code `gems_bench/p6/{export_parity,candidate_recall,segment_recovery,boost_diagnostics}.py` · config
`configs/p6/recovery.yaml` · notebook `notebooks/colab/GEMS_P6_RECOVERY.ipynb` · tests `tests/test_p6_recovery.py` ·
outputs `MyDrive/GEMS_2/outputs/p6_fast_track/recovery/`.

**Nothing in P6.1 was executed.** No Python, test, Ruff, notebook cell or shell command ran in the authoring session
(code-authoring-only brief). Every number below is either a recorded offline value quoted from the brief or a plan.
No stage is COMPLETE until its `recovery/<stage>/status.json` says so on Drive.

## 1. Starting point (recorded offline, nested; not leaderboard)

| candidate | disc DTI | worst disc zone | geo DTI | note |
|---|---:|---:|---:|---|
| fallback fusion crest | 0.162482 | 0.126160 | 0.182832 | boost A |
| W2 (recorded) | 0.224640 | 0.153414 | 0.180843 | W2 nested, own grid |
| W2 crest | 0.238882 | 0.192575 | 0.185609 | boost A |
| W2 skeleton | 0.242156 | 0.198579 | 0.168841 | boost A: geo below the 0.1758 floor |
| hybrid crest | 0.231509 | 0.156216 | 0.182415 | boost A |
| uploaded W2 file (skeleton @ 5 %, blend 1) | 0.178905 | — | 0.168841 | fixed cell, not nested |

In boost A, disc and geo were nested **separately**: one row's disc and geo numbers can come from different cells.
P6.1 never combines them into one deployable score without re-evaluating the policy as one decision per zone.

## 2. What changed

### New modules

* `export_parity.py` (X1)
  * Policies are evaluated from the saved boost grid. Per outer zone, ONE cell (candidate, blend, rep, frac) is
    chosen on the selection zones and applied to that zone in both tasks.
  * Objectives:
    * `disc_geo_floor` (deployment, predeclared): the best discovery DTI among cells whose geo DTI on the same zones
      is ≥ 0.1758.
    * `disc` and `joint`: sensitivity only.
    * `separate`: reproduces boost and is flagged not deployable when the per-task cells differ.
  * Selection zones exclude every unit whose evaluation mask overlaps the held-out zone (`unit_overlap`). If that
    exclusion is infeasible, the code falls back to same-index exclusion and switches the objective to `disc`, and
    records the switch. This stops geo labels of the held-out region from entering selection.
  * `meta_nested`: an honest estimate of choosing the policy itself.
  * `deploy_choice`: the predeclared deployment rule.
  * `policy_tables`: every table, plus the reproduction check against boost `nested_selection.csv`.
  * Parity (Colab):
    * W2 held-out map vs the `full_score.tif` crop.
    * Each W2 feature: unit construction vs full-region construction, as consumed by the model.
    * `rows.parquet` values vs the inference value at the same pixels.
    * The binary map at the deployment cell built from each side.
    * Fusion weight per zone vs all-zone.
    * Statistics: Spearman, Pearson, top-1 % / top-5 % Jaccard, quantiles, KS. Known-fault features are
      `expected_by_design` (WARNING, never BLOCKED).
  * Export contract:
    * `build_recipe` is the only way to describe an export. A `rep_source` / `frac_source` / `blend_source` that
      differs from the `selection_id` raises `PolicyMismatch`, which is exactly the historical defect.
    * `plan_audit` and `save_plan` write the plan before any file exists.
    * `write_validated` is the single writer (`m7.export._write`). Files are never overwritten, and a rerun for the
      same recipe fingerprint reuses its file.
    * `audit_export` checks CRS, transform, shape, the NaN pattern vs the template, the [0, 1] range, eligibility,
      known-fault pixels, the area budget, sidecar = recipe, and sha256.
  * `w2_policy_map` rebuilds the W2 export from the frozen `full_score.tif`, with no refit, using rep, frac and blend
    from one selection.
* `candidate_recall.py` (X2)
  * Inventory: B0, B1, fallback fusion, W2, W2 ∪ fallback (labelled: area up to 2×), and G02-C on the units that
    have it.
  * Budgets: 0.5–5 %, both reps.
  * Per system: support, hit share and recovered, through `system_hits` (official matching), plus attributes:
    * length, orientation, curvature, fragments;
    * distance to permitted known faults, eligible share;
    * mag / grav gradient rank, CNN agreement, mg orientation agreement, G02-C;
    * local prediction density.
  * The oracle (`oracle_select`) picks pool segments with held-out labels within the same eligible area (constrained)
    or without any area limit (unconstrained). Its rows carry `role = oracle_*`, and `export_parity.guard_oracle`
    refuses them in every selection.
  * The taxonomy has nine evidence-based categories. All flags are saved, and a sensitivity table varies the support
    threshold (×0.5 / 1 / 2) and the displacement factor (1.5 / 2 / 3).
  * Decision: ABSENT / RANKING / GEOMETRY / MASKING_COVERAGE / MIXED / INCONCLUSIVE (< 10 missed systems).
* `segment_recovery.py` (X3)
  * `skeleton_graph`: endpoints, junction clusters and branches; short or isolated pieces are kept.
  * `dedup_inventory`: parallel copies are removed; crossing lines are kept.
  * `segment_features`.
  * `strategy_map` H0 / H1 / H2 is the single implementation, used by both unit evaluation and the full-region
    export. It asserts area ≤ H0.
  * Nested evaluation uses the X1 objective. `budget_violations` is applied, and the stop condition is enforced.
* `boost_diagnostics.py`
  * `Recovery` runner: manifest, status and decision per stage; the fingerprint covers config and input hashes; a
    stage reloads only if every required output exists (`OUTPUTS`), and a missing output turns COMPLETE into FAILED.
  * Preflight.
  * X4: `geology_audit` (dependency graph + classification + repair plan), `g02_backfill` (explicit), `geology_s1`
    (S1 first; S2/S3 only after S1 ACCEPT or a declared reason).
  * X5 `robustness`: weak zone per policy, plus optional boost C1/C2.
  * `select`, using `recovery_gate`.
  * `export`: requires the selected candidate fingerprint.
  * `report`.
  * `startup_check`.

### Modified

* `gems_bench/p6/boost.py` (`BOOST_CODE_REV` → `p6b.1-r3`):
  * `system_set` parses `"3.0"` (CSV float reload).
  * `_eval_q` keeps `_last_hits`.
  * New `binary()` and `unit_sources()`.
  * `Boost.export` now goes through `build_recipe` → `save_plan` → `write_validated`. The audit decides whether the
    file is logged (r2 relied on `S.validate` returning errors, but it raises).
* `gems_bench/p6/boost_audit.py`:
  * `g02_backfill_plan` compares the run's recorded channels against `M9Source.has()`. It reports
    `registry_only_channels` instead of the false `missing_channels`.
  * `parity_metadata` reads the historical sidecar's `frac_source`.
* `gems_bench/p6/stack.py`: `full_region` refuses `area_frac_source: geo` unless `allow_mixed_policy: true`.
* `configs/p6/fast_track.yaml`: `stack.full_region` = `{area_frac_source: primary, allow_mixed_policy: false}`. The
  historical file is untouched.
* `configs/p6/boost.yaml`: comment on the r3 export path.
* `gems_bench/p6/run.py`: new `P6FastTrack.recovery(config, boost_config)`.

The old notebooks (`GEMS_P6_BOOST_STANDALONE.ipynb`, `GEMS_P6_FAST_TRACK.ipynb`) were not modified. To use them on
r3 code, set their `EXPECT['code_rev']` to `p6b.1-r3`: their start-up check refuses r3 otherwise, by design.

## 3. Evaluation / export inconsistencies addressed

| # | Inconsistency | Fix |
|---|---|---|
| 1 | W2 file = rep / blend from the disc choice + frac from the geo choice (`skeleton_1px @ 5 %`, never selected) | `build_recipe` raises `PolicyMismatch`; `stack.full_region` refuses `geo`; config → `primary` |
| 2 | Disc and geo numbers of one "candidate" came from separately nested selections | `evaluate_policy`: one cell per outer zone for both tasks; `separate` kept only as a flagged reproduction |
| 3 | Policy choice itself was in-sample | `meta_nested` (choose the policy on inner zones) |
| 4 | Geo rows of the held-out geography could enter selection | overlap-based exclusion, with a recorded fallback to `disc` |
| 5 | Exported files were validated by an assertion that aborts before the sidecar is written | `write_validated` catches it; `audit_export` is the PASS / FAILED gate |
| 6 | No pre-write plan of the full recipe | `export_policy.json` + `plans/<fp8>/export_plan.json` before any file |
| 7 | `systems` reloaded as floats broke `system_set` | parsed via `float` |
| 8 | `g01:edge` / `g01:surface` reported missing | they are virtual channels (THDR of raw bands), so the class is `REGISTRY_MISMATCH`, not a missing input |

## 4. G01 / G02 dependency chain (from code; not yet confirmed on Drive)

* G02 run `g02_23686a63cb` depends on:
  * M9 layer cache `.npy` layers (`input_fingerprints`);
  * the virtual channels `g01:edge` (THDR of the raw magnetic band) and `g01:surface` (THDR of detrended elevation);
  * raw bands (`band:mag|grav|terrain`);
  * optionally the M9 prediction.
* G02 recomputes every G01 map it needs. A persisted G01 run is read only by the G01 review, so it is **not** a raster
  dependency.
* `backfill plan: missing g01:edge / g01:surface` was a loader-registry false positive:
  * `M9Source.names()` lists only cached `.npy` stems.
  * `input_drift()` already fingerprints these channels and reported no change.
* Real blocker: the unit rasters `geo_2` and `disc_official_2` are missing (a missing saved artifact).
  * The repair status becomes `READY_FOR_BACKFILL` when the attachment, drift and sources are OK.
  * The backfill reuses the run config and per-unit training-geography references, writes `backfill: True` metadata,
    resumes per tile and never rewrites existing rasters.
* Full region: `stack.g02_full` is label-free (whole footprint as reference). It is computed only if a G02 candidate
  is promoted.
* Confirm on Drive with `rc.geology_audit()`:
  * `geology/compatibility_report.csv` (per channel: class, recorded vs current fingerprint);
  * `dependency_graph.json`;
  * `feature_provenance.json` (G01 runs on disk, wavelengths, drift);
  * `repair_plan.json`.

## 5. Colab execution order

Local, once (by you; nothing was run here):

```bash
git switch -c p6-recovery            # from p6-boost (7a48ee4)
pytest -q tests/test_p6_recovery.py tests/test_p6_boost.py tests/test_p6.py
ruff check gems_bench/p6 tests/test_p6_recovery.py
git add gems_bench/p6 configs/p6 notebooks/colab/GEMS_P6_RECOVERY.ipynb tests/test_p6_recovery.py docs/p6/P6_RECOVERY_HANDOFF.md
git commit -m "P6.1 recovery: deployable policies, export contract, candidate recall, segment recovery, G02 audit"
git push -u origin p6-recovery && git rev-parse HEAD     # pin as CONFIG['ref'] / EXPECT['commit']
```

Colab: CPU, high-RAM. Stop at the first failure. All estimates are unmeasured; the wb cap is 4 CPU-h shared with
boost, so check `p6.budget.table()` first.

| step | section / switch | expect | est. CPU | RAM |
|---|---|---|---|---|
| 0–2 | config, clone, start-up check, attach, `preflight` | `preflight/decision.json` `ready.*`; boost A must be `done` | 5–15 min (G02 attach stages the layer cache) | context |
| 3 | `export_parity` | `decision.json` verdict, chosen policy, META, parity rows | tables < 1 min; numeric parity 10–30 min | full-region features ≈ 12–16 full-grid float32 arrays + unit crops (same order as a boost export) |
| 4 | `candidate_recall` | verdict + sensitivity, oracle bound, examples | 0.3–0.6 h (≈ 360 scored maps + oracle) | one unit's raw features + maps at a time |
| 5 | `segment_recovery` | H1 / H2 ACCEPT or REJECT, 0 budget violations | 0.3–0.8 h (180 scored maps + inventories) | unit crops |
| 6 | `g02_repair_audit` → `g02_backfill` → `geology` | `READY_FOR_BACKFILL` → 2 rasters → S1 decision | audit min; backfill ≤ 2 × 1698 s upper bound; S1 cross-fit 0.2–0.5 h | one G02 unit |
| 7 | `robustness` | weak zone per policy | min (+ boost C ≈ 0.4 h if enabled) | — |
| 8 | `selection` (export OFF) | gate per candidate, `candidate_fp` | min | — |
| 9 | `export` + `RECOVERY_EXPORT['candidate_fp']` | validated GeoTIFF + audit PASS | 5–15 min (W2 policy: no refit) | full grid |
| 10 | `report` | `reports/recovery_report.md` | min | — |

## 6. Expected outputs (`outputs/p6_fast_track/recovery/`)

* `preflight/`: `preflight.json`, `dependency_check.csv`, `decision.json`, `manifest.json`, `status.json`.
* `export_parity/`:
  * `policy_comparison.csv`, `nested_policy_selection.csv`, `per_zone_metrics.csv`, `unit_overlap.csv`;
  * `train_inference_parity.csv`, `feature_distribution_shift.csv`;
  * `export_policy.json`, `export_audit.json`, `plans/<fp8>/export_plan.json`;
  * `policy_frontier.png` + `policy_frontier_data.csv`;
  * `decision.json`.
* `candidate_recall/`:
  * `coverage_by_zone.csv`, `coverage_by_budget.csv`, `system_inventory.parquet`;
  * `candidate_source_attribution.csv`, `candidate_redundancy.csv`;
  * `missed_system_taxonomy.csv`, `attribute_contrast.csv`;
  * `oracle_upper_bound.csv`, `source_availability.csv`;
  * `coverage_curves.png` + `_data.csv`, `missed_system_examples.png` + `_data.csv`;
  * `decision.json`.
* `segments/`:
  * `segments.parquet`, `segment_features.parquet`;
  * `segment_selection.csv`, `graph_diagnostics.csv`, `grid.csv`, `budget_violations.csv`;
  * `hybrid_comparison.csv`, `system_recovery.csv`, `per_zone_metrics.csv`;
  * `recovery_examples.png` + `_data.csv`;
  * `decision.json`.
* `geology/`:
  * `dependency_graph.json`, `compatibility_report.csv`, `unit_rasters.csv`, `feature_provenance.json`;
  * `repair_plan.json`, `decision.json`;
  * after S1: `s1_ablation.csv`, `s1_decision.json`. Its cross-fitted maps share `boost/geology/maps/`: same
    fingerprint parts, new files only.
* `robustness/`: `weak_zone_identity.csv`, `recall_by_zone.csv`, `boost_c_ablation.csv` (if run), `decision.json`.
* `selection/`: `candidates.csv`, `decision.json`.
* `reports/`: `recovery_report.md`, `artifact_registry.csv`, `experiment_status.csv`.
* After an export: `p6_fast_track/submissions/p6r_<candidate>_<fp8>.tif` + `.json` sidecar (recipe fingerprint) and a
  leaderboard-log row.

## 7. Gates and stop conditions

* **X1 PASS**: the chosen deployable policy has disc > W2 reference, geo ≥ 0.1758, worst ≥ 0.153414, no BLOCKED parity
  row, and a PASS plan audit.
  * Watch for: skeleton fails the geo floor in boost A (0.168841), so under `disc_geo_floor` crest is the expected
    contender. That is to be confirmed by the run, not assumed.
* **X2**: the dominant group decides the direction (section 11 of the brief), provided ≥ 10 missed systems, a gap of
  more than 0.10 between the top two groups, and a sensitivity table that does not flip. Otherwise MIXED or
  INCONCLUSIVE.
* **X3**: ACCEPT needs a nested disc gain > 0, worst ≥ H0, geo ≥ floor, FP weight ≤ +10 % and no budget violation.
  More systems without these is a REJECT.
* **X4**: S1 ACCEPT needs Δdisc ≥ +0.005, worst ≥ S0 − 0.005 and geo ≥ floor. S2/S3 run only after S1 ACCEPT or a
  non-empty `geology.s2_s3_diagnostic_reason`.
* **Promotion** (`recovery_gate`):
  * disc > max(W2 reference, matched W2 benchmark);
  * worst ≥ 0.153414, geo ≥ 0.1758;
  * a single deployable policy;
  * provenance clean, parity clean with numeric parity run, recipe planned.
  * A leaderboard slot also needs +0.010.
  * Oracle, the historical cell and reference rows are never promotable.
  * Nothing is uploaded automatically.

## 8. Outstanding risks

* None of the code has been executed. The first Colab run may hit API mismatches. The most likely spots:
  * `split_branches` output on real masks;
  * `Unit.scorer.g_rc` on the official units;
  * column names of the saved boost grid;
  * the `probes._row` fields.
* Unit-overlap exclusion: if disc units cover the whole footprint, the objective falls back to `disc`, so the geo
  floor is enforced only by the gate, not inside the selection.
* With three zones, every inner META selection sees one zone: high variance, and differences under 0.005 are noise.
* The W2 policy export reuses `stack/maps/full_score.tif`. If that file is absent or changed (sha mismatch), the export
  is refused.
* X2 / X3 use held-out CNN maps (training construction). The full-region maps differ, and that difference is measured
  in X1 parity.
* H1 / H2 parameters are predeclared, not tuned. A REJECT there does not rule out tracing work in general.
* The `geology_s1` maps land in `boost/geology/maps` (shared cache). Existing files are never rewritten.
* G02 full-region maps (`g02_full`) are unmeasured and heavy. They are needed only if a G02 candidate is promoted.

## 9. Paste-ready summary for the next session

> P6.1 recovery is authored and **unexecuted** on local branch `p6-boost`, on top of `7a48ee4` and not yet committed.
> New modules: `gems_bench/p6/{export_parity,candidate_recall,segment_recovery,boost_diagnostics}.py`. Also new:
> `configs/p6/recovery.yaml`, notebook `notebooks/colab/GEMS_P6_RECOVERY.ipynb` and `tests/test_p6_recovery.py` (not
> run).
>
> Modified:
>
> * `boost.py` (r3: export through `export_parity.build_recipe` / `write_validated`, robust `system_set`,
>   `unit_sources` / `binary`).
> * `boost_audit.py` (`g01:edge` / `g01:surface` are virtual channels: registry mismatch, not missing).
> * `stack.full_region` (refuses the mixed rep / frac `geo` source) and `fast_track.yaml` (`area_frac_source:
>   primary`).
> * `run.py` (`p6.recovery()`).
>
> Next steps:
>
> 1. Commit to `p6-recovery`, run pytest / ruff locally, push, and pin the SHA.
> 2. In Colab, run sections 0–4: `preflight`, `export_parity`, `candidate_recall`, `report`.
> 3. Read `recovery/export_parity/decision.json` (chosen deployable policy, META, parity) and
>    `recovery/candidate_recall/decision.json` (ABSENT / RANKING / GEOMETRY + sensitivity) before claiming any number.
>
> Recorded references, offline: W2 0.224640 / worst 0.153414 / geo 0.180843; boost W2 crest 0.238882 / 0.192575 /
> 0.185609; W2 skeleton 0.242156 / 0.198579 / 0.168841 (geo below the floor). The uploaded W2 file `skeleton_1px @ 5 %`
> scores 0.178905 / 0.168841 as a fixed cell. G02 zone 2 still needs the explicit backfill (`rc.geology_audit()` →
> `READY_FOR_BACKFILL` → `rc.g02_backfill()` → `rc.geology_s1()`).
