# GEMS discovery-first GeoMoE (dgm.2-r1)

Repairs the dgm.1-r1 Colab run (commit 7dab78f: stage 04 FAILED, everything after it failed or was misreported) and
makes the comparison scientifically fair. **Authored, never executed: no number below is a result.**

* Notebook: `notebooks/colab/GEMS_DISCOVERY_GEOMOE.ipynb` (same file; CPU, High-RAM)
* Code: `gems_bench/geomoe/` (`run.DiscoveryGeoMoE`); config `configs/geomoe/discovery_geomoe.yaml`
* Tests: `tests/test_discovery_geomoe.py` + runtime `gems_bench/geomoe/selftest.py` (both run in stage [00])
* Outputs: `MyDrive/GEMS_2/outputs/discovery_geomoe/`

## 1. What dgm.1-r1 got wrong, and the fix

| defect (Colab evidence) | fix |
|---|---|
| `candidates.unique_coverage`: `F[others].any(1)` TypeError (pandas positional axis) | explicit `axis=1` on numpy arrays; same for `sum(1)` / `mean(1)`; empty-input path; regression test `test_regression_unique_coverage_colab_typeerror` + self-test `unique_coverage_regression` |
| pytest failed but the notebook continued | stage [00] runs self-tests + pytest; nonzero exit or failed critical self-test = `TESTS_FAILED`, failing ids printed, `audit/pytest_output.txt`; [04]+ blocked |
| downstream stages ran on missing inputs (`KeyError: 'scenario'`, missing npz) | `decide.DEPENDS` checked before every stage: `BLOCKED_UPSTREAM` (compute) / `NOT_EVALUATED` (decision); required outputs validated (exists, readable, columns) after a run and before any reuse |
| [06] reported `INCONCLUSIVE, geology_unique=false` without coverage | per-scenario verdicts `GENERATION_LIMITED / RANKING_LIMITED / MIXED / NOT_DOMINANT / NOT_EVALUATED`; no majority vote (`MIXED` + `disagreement`); `geology_unique = None` unless an independent family was evaluated; [06] FAILS when nothing was measured |
| "W2 retained" after a failed comparison | `select_model` returns `NO_MEASURED_COMPARISON` unless the reference and a fused model were evaluated on every scenario; export then writes no file and says "not evidence for W2" |
| one failure lost a whole unit | a unit saves its inventory (labels, pools, segments, provenance) before coverage; each part validated (unit fingerprint, array shapes, columns) and only invalid parts recomputed |
| scenario-optimised W2 shown as the baseline | `A_submitted_W2`: the factory's evaluated champion cell (rep, frac, blend, grid candidate, deploy kind) frozen for V1 / V2 / V3; scenario-selected W2 kept only as a diagnostic |

## 2. Methodology changes

* **Reference**: `A_submitted_W2` is built with the factory's evaluation construction (`final_submission.zone_score`,
  per-zone champion fusion) at the frozen cell; `audit/baseline_policy_parity.csv` compares each map with
  `Boost.binary` (Jaccard) and the frozen boost-A grid components, and records the W2 map sha256 and recipe id.
* **Fixed policy** (model selection): every model uses ONE rule for every scenario; the outer unit's (rep, frac) is
  chosen on development units whose evaluation zone overlaps it by <= 5 % (`selection.policy_max_overlap`; V2 / V3 of
  the same fold never qualify), by the mean over scenarios of the pooled DTI. When no unit qualifies the within-scenario
  nested rule is used and flagged `DEV_OVERLAP_FALLBACK` (`validation/policy_development_units.csv`). The deployable
  cell = the same rule on all units (`gating/deployable_policies.json`). Per-scenario nested numbers =
  `gating/model_summary_scenario_selected.csv` (diagnostic only).
* **V2 fairness**: `network_w2ctx` = network evidence from W2's own permitted-known context; `B_ctxW2` scores E4 with it
  and drops tips-only candidates, so `B - B_ctxW2` isolates the benefit of the extra retained context. W2 cannot be
  re-run with the V2 context (no persisted model): stated, not hidden.
* **V2 subtypes**: `validation/v2_by_subtype.csv` (events, truth length, inventory / family coverage, recovered share,
  gain vs the reference, official TP / FP / FN and DTI with the other withheld classes masked) and
  `validation/v2_coverage_by_subtype.csv` (geometric, available after [05]).
* **Leakage audit**: explicit tests only (`PASS` / `FAIL` / `UNVERIFIED`): known context vs truth pixels, real-pixel V2
  truth, CNN provenance tags of the loaded maps, known-distance > 0 on truth, G02 raster identity, training zones of every
  normalisation / fit, no truth column in gate features. CNN training-label exclusion is `UNVERIFIED` (frozen
  experiments).
* **G02**: `backfill_missing_g02=True` computes only the missing unit rasters with the attached run's frozen config
  (`Boost.backfill`: refused on input drift, existing rasters never rewritten), then pushes them. Otherwise missing.
  Consistent availability: E3 is dropped for a whole scenario when a zone lacks it; `B_E3_avail` (E3 where available)
  is a non-promotable ablation.
* **Coverage**: geometric (systems, length, family, unique beyond E1 / W2, subtype, distance band) and
  **kernel-weighted oracle** (`candidates/kernel_weighted_oracle.csv`: official components under the evaluation area
  fractions, diagnostic only). 300 m geometric coverage is not official TP support.
* **Reconciliation**: `audit/candidate_bottleneck_reconciliation.csv` (label-free vs learned vs unified pool: system
  and length coverage, ranking- vs generation-limited share of W2's missed length).
* **Rules**: R6 magnetic / gravity agreement added; each rule ablated. Selection unchanged in spirit: B must beat the
  frozen W2 on >= 2 scenarios without a drop; C / D must beat B by a paired margin; a collapsed gate is never promoted.
  `gating/effect_decomposition.csv` separates pool, fusion, routing, learning, network (V2), geography (V3).

## 3. Stage order and dependencies

`[00] tests -> [01] attach -> [02] audit (+backfill) -> [03] scenarios -> [04] candidates -> [05] oracle ->
[06] decision` (= FAST_DIAGNOSTIC), then `[07] experts -> [08] static -> [09] rules -> [10] learned -> [10b] compare
-> [11] deployment -> [12] export`, always `[14] integrity -> [13] reports`.

## 4. Colab run order

1. Commit and push; open the notebook from GitHub; CPU, High-RAM.
2. CONFIG: `mode='FAST_DIAGNOSTIC'`; pin `ref` / `EXPECT['commit']`. Run all cells.
3. Read the printed coverage verdict, `reports/EXECUTION_STATUS.md`, `audit/stage_integrity.csv`.
4. If [00]-[06] are COMPLETE: set `mode='FULL'`, run all cells again (finished stages print `REUSED_VALID_CACHE`).
5. Optional: `backfill_missing_g02=True` once (CPU time per unit was measured at 0.15-0.5 h in P6), then rerun.

## 5. Outputs to inspect

`audit/` stage_integrity, pytest_output.txt, selftests, label_leakage_audit, baseline_policy_parity,
candidate_bottleneck_reconciliation, metric_equivalence, g02_backfill.json · `validation/` frozen_w2_baseline,
frozen_w2_summary, v2_events, v2_by_subtype, v2_coverage_by_subtype, per_zone_metrics, near_known_metrics,
per_fault_type_metrics, policy_development_units · `candidates/` candidate_inventory.parquet, source_provenance,
oracle_coverage, oracle_curves, kernel_weighted_oracle, coverage_by_distance, unique_systems, missed_systems,
coverage_by_fault_type, coverage_decision.json · `experts/` expert_features.parquet, expert_scores.parquet,
source_availability, source_normalization.json · `gating/` model_summary, model_summary_scenario_selected,
static_fusion_results, rule_gate_results, learned_gate_results, paired_comparison, effect_decomposition,
feature_importance, gating_weights.parquet, selection.json, deployable_policies.json, deployment_* ·
`predictions/heldout|full_region/` · `submissions/` · `figures/` baseline_comparison, oracle_vs_actual,
unique_geology_recovery, v2_fault_subtypes, expert_complementarity, gating_comparison, geographic_robustness (+ data
CSVs, MISSING.txt) · `reports/` EXECUTION_STATUS, SCIENTIFIC_FINDINGS, MODEL_COMPARISON, LEADERBOARD_RECOMMENDATION,
NEXT_ACTION · `manifests/` run.json, runtime_errors.json, stages/*.json.

Reload examples (Colab, after cells 1-2):

```python
pd.read_csv(OUT / 'audit/stage_integrity.csv')
json.loads((OUT / 'candidates/coverage_decision.json').read_text())
pd.read_parquet(OUT / 'candidates/candidate_inventory.parquet')
G.s14_integrity(); G.s13_reports()          # rebuild integrity + reports from persisted tables
```

## 6. Runtime-unverified risks

* Nothing ran locally: every test, self-test and engine call is unverified until Colab.
* `Boost.binary` vs `EV.policy_q` parity is expected exact for V1 / V3; a FAIL blocks selection.
* Discovery-split evaluation zones may overlap every geo fold and each other; then V1 uses the flagged fallback rule.
* G02 backfill cost and its refusal on input drift; DEM stays absent unless the cached mosaic is on Drive.
* Memory of stage 04 on full crops (one unit at a time, pools stored as packed bits).
* The frozen policy is read from the factory's `F.C` table; if the factory attach changes that table the configured
  fallback (`frozen_w2`) is used and recorded.

## 7. Fresh-session handoff

```
GEMS discovery GeoMoE dgm.2-r1 (repair of dgm.1-r1 Colab run 7dab78f): authored 2026-10-29, NOTHING executed.
Before claiming anything read on Drive outputs/discovery_geomoe/: audit/stage_integrity.csv, reports/EXECUTION_STATUS.md,
audit/pytest_output.txt, candidates/coverage_decision.json, validation/frozen_w2_baseline.csv,
audit/baseline_policy_parity.csv, gating/selection.json (status + measured), submissions/submission_manifest.json.
NOT_EVALUATED / BLOCKED_UPSTREAM / NO_MEASURED_COMPARISON are not results. Reference = A_submitted_W2 (public 0.0942,
external only). No neural training, no M9.
```
