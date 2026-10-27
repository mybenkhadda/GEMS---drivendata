# P6 Recovery → Boost: repository and artifact audit (2026-10-28)

**How this was produced.** The audit was written by reading code and the outputs saved in the notebooks on `main`
(`acbc694`). The run records are `notebooks/colab/GEMS_P6_RECOVERY.ipynb` (13 executed cells) and
`GEMS_P6_BOOST_STANDALONE.ipynb`. Nothing was executed while writing it.

**What the numbers are.** Every number below is quoted from those saved outputs, and the Drive files themselves were
not inspected. They are nested, held-out offline estimates; none is a leaderboard score. The runtime version of this
audit is `outputs/p6_boost/audit/artifact_audit.csv`, written by `fg.audit()`.

## 1. What P6 Recovery implements

`gems_bench/p6/boost_diagnostics.py` defines `Recovery` and is driven by the notebook `GEMS_P6_RECOVERY.ipynb`. Its
stages:

* **X1, `export_parity.py`: deployable policies.**
  * Evaluates the boost-A grid as single deployable policies. For each outer zone it chooses ONE cell on the other zones
    and applies that cell to the held-out zone in both tasks.
  * Covers W2 policies A–E and the M9 fallback reference.
  * Also produces META (the policy choice nested), the train/inference parity tables, and the export contract
    (`build_recipe`, `plan_audit`, `audit_export`).
* **X2, `candidate_recall.py`.** Recall ceiling of the candidate pool, an oracle used for diagnosis only, and a
  failure taxonomy of missed systems.
* **X3, `segment_recovery.py`.** Segment recovery strategies H0 / H1 / H2, compared at matched area.
* **X4: G01 / G02 dependency audit.**
  * Classifies each blocker, then supports an explicit backfill and S1.
  * The backfill and S1 were not run.
* **X5.** Identifies the weak zone of each policy. The optional boost C was not run.
* **Selection, export, report.**
  * The gate was applied to every candidate.
  * Export is opt-in and was not run.

## 2. What actually ran on real data

**Recovery run** (`acbc694`, 2026-10-27 21:55–22:45, Colab CPU, profile PILOT):

* Stages that reached COMPLETE: `preflight`, `export_parity`, `candidate_recall`, `segments`, `geology` (audit only),
  `robustness`, `selection`. The wb ledger spent rose from 0.097 h to 1.035 h.
* `export`, `g02_backfill` and `geology_s1` were switched off and did not run.
* Preflight: 2 checks failed.
  * RAM available was 11.5 GB, against 24 GB required.
  * G02 rasters exist only for `disc_official_0/1` and `geo_0/1`.

**Boost run:** `preflight`, A `benchmark` and `report`. Stages B (geology), C (robustness), D (hybrid) and E
(selection) were all switched off.

**Earlier P6 work** (memory notes plus the budget ledger):

* W2 ran with the gate PROMOTE. It produced the historical file `submissions/p6_stack_fusion.tif`.
* W1 shows 0.000 GPU-h spent: no extra seed and no TTA exist.
* W3 failed earlier (missing cube / xmod layer).

### Recorded X1 results (single deployable policy, nested, 3 zones)

The selection objective fell back to `disc` because the overlap exclusion was infeasible. Under `disc_geo_floor`, geo
labels of the held-out geography could enter the selection, which is why the fallback applies.

| policy | objective | disc DTI | worst zone | geo DTI |
|---|---|---:|---:|---:|
| A_w2_original | disc | 0.224640 | 0.153414 | 0.102664 |
| B_w2_crest | disc | 0.238882 | 0.192575 | 0.114259 |
| **C_w2_skeleton** | disc | **0.242156** | **0.198579** | 0.105687 |
| R_fallback_fusion (M9) | disc | 0.162482 | 0.126160 | 0.158416 |
| A_w2_original | disc_geo_floor | 0.187338 | 0.163983 | 0.186103 |
| C_w2_skeleton | disc_geo_floor | 0.206651 | 0.184673 | 0.132302 |
| R_fallback_fusion | disc_geo_floor | 0.154004 | 0.129112 | 0.183273 |
| E_w2_uploaded_cell (fixed skeleton @ 5 %, blend 1) | — | 0.178905 | 0.155761 | 0.168841 |

* **META.** Choosing the policy itself, nested, gives disc 0.224640 and worst 0.153414. Zones 0 and 1 picked C and zone 2
  picked A.
* **Separate selection.** When disc and geo are selected independently, boost A's numbers are reproduced (e.g. W2
  skeleton 0.242156 / geo 0.168841). That combination is not one deployable file.
* **The P6.1 gate failed for every candidate.**
  * Geo fell below the absolute floor of 0.1758.
  * Parity was BLOCKED.
  * D, H1 and H2 do not beat the reference.

### Other recorded results

| stage | result |
|---|---|
| X2 | Verdict `RANKING`: 82.9 % of the missed length is ranking-limited, 17.1 % geometry-limited. |
| X3 | Chosen strategy `H0` (pure W2). H1 0.223435 and H2 0.239703 were not accepted; 0 budget violations. |
| X4 | `READY_FOR_BACKFILL`. The real blocker is missing unit rasters `geo:2` and `disc_official:2`. `g01:edge` / `g01:surface` are `REGISTRY_MISMATCH` (virtual channels), not missing inputs. All 32 cached spectral layers have unchanged fingerprints. |

### Train / inference parity (the export blocker)

| check | PASS | WARNING | BLOCKED |
|---|---:|---:|---:|
| feature | 30 | 25 | 11 |
| w2_score | – | 3 | 3 |
| map_agreement | – | 6 | – |
| fusion_weight | 6 | – | – |

The BLOCKED rows fall into two groups:

* **Construction mismatches** (a real defect of the historical export):
  * `w2_score` on `geo:0/1/2` and `cnn_disagree` on `geo:0/2`.
  * Cause: training ranks every feature inside the unit, while `stack.full_region` ranks it inside the whole footprint
    and applies a refit model.
  * `|rank_b0 − rank_b1|` is not invariant to the ranking domain, so `cnn_disagree` differs between the two
    constructions.
* **Expected by design** (not a defect):
  * `cnn_b0`, `cnn_b1` and `cnn_disagree` on `disc_official:0/1/2`.
  * Training uses the honest discovery models (withheld systems unseen); inference uses the stitched out-of-fold maps.

## 3. Reusable artifacts

| artifact | class | reason |
|---|---|---|
| `stack/candidate.json`, `stack/features/rows.parquet` | REUSE | W2 PROMOTE; read-only provenance |
| `stack/maps/{geo,disc_official}_{0,1,2}.npy` (W2 cross-fitted held-out maps) | **REUSE** | the exact objects that were evaluated, and the basis of the zone-stitched submission |
| `stack/maps/full_score.tif` (refit, footprint-ranked) | **INCOMPATIBLE** | parity BLOCKED (`w2_score` on geo units); kept read-only, never exported again |
| `stack/models/stack_full.joblib` | INCOMPATIBLE (for export) | produced `full_score.tif`; same construction mismatch |
| `boost/benchmark/grid.csv` + `nested_selection.csv` | REUSE | stage A done; protocol keys recorded |
| `boost/{geology,robustness,hybrid}/grid.csv` | RECOMPUTE (opt-in) | never run |
| `recovery/*` tables and decisions | REUSE | COMPLETE on `acbc694` |
| `recovery/export_parity/export_policy.json` (recipe of the chosen policy) | INCOMPATIBLE | footprint-ranked `full_score.tif` source |
| M9 B0 / B1 OOF + `disc0-2` maps (`P5_B0_gated`, `P5_B1_bgw`, FULL) | REUSE | honest maps of every unit |
| B0 / B1 CV checkpoints | UNKNOWN | needed only for TTA / fold-average; checked at run time |
| `seeds/grid.csv` (W1) | RECOMPUTE (GPU) | W1 never trained |
| G02 run `g02_23686a63cb` rasters zones 0 / 1 | REUSE | fingerprints unchanged |
| G02 rasters zone 2 | RECOMPUTE | explicit backfill |
| `submissions/p6_stack_fusion.tif` (historical W2 file) | INCOMPATIBLE | rep from the disc choice, frac from the geo choice (a cell no nested choice selected) |

## 4. Compatibility of P6 Recovery and P6 Boost

* **Feature schema.** Both use the frozen W2 feature list and the same `rows.parquet`. Boost B / C variants add or alter
  columns, and their cross-fitted maps are cached in `boost/<stage>/maps/<variant>_<fp8>/`.
* **Coordinates.** One grid (`ctx.data.grid`, EPSG:32611, 100 m) is used throughout. Units are crops `u.sl` of that
  grid. The template must match the feature grid, and the export asserts it.
* **Masks.** One eligibility rule, `m9.elig(u, 0)`, applies to every grid row. The full-region eligibility is valid &
  template finite & not a mapped fault.
* **Scoring.** One scorer (`m7.protocol.score_row` → `FastScorer`, `disc_official` pixel-exact mask) applies
  everywhere. `full_benchmark.protocol_check` refuses to pool grids whose protocol keys or truth mass differ.
* **Incompatible.** Only the historical full-region construction (`full_score.tif`) and the historical mixed-policy
  file.

## 5. How the inputs enter inference

* **Mapped fault traces.**
  * Training truth (geo task: all mapped faults; disc task: withheld systems).
  * Known-fault features (`known_dist_km`, `known_density`, `tip_dist_km`, `tip_along_cos`): permitted known faults in
    training, every mapped fault at inference. That difference is by design.
  * The official pixel-exact mask: mapped faults never receive budget.
* **Raster features.**
  * B0 / B1 CNN probability maps (Phase-5 cnn5 on the raster stack): OOF on geo units, discovery models on disc units.
  * Potential-field derivatives (`pf_full`: magnetic `rtp`, gravity `iso_grav_anom`).
* **Geological priors (G02-C, G02-A) and G01 wavelet persistence.** Used as extra W2 features in boost B only (S1–S3).
  They are not in the current champion.
* **Postprocessing.** Rank fusion, then blend, then a thin representation (`crest_1px` / `skeleton_1px`) at area
  fraction `frac` of the eligible domain. Ring 0.

## 6. Validation leakage risks

1. **Geo labels of the held-out geography in selection.** Unit masks overlap, so `disc_geo_floor` / `joint` would see
   the withheld systems through geo rows. Mitigation: the objective falls back to `disc` (recorded); geo is applied
   only as a gate or report.
2. **Choosing the policy is itself a choice.** Report META next to the winner. With three zones it is noisy, and
   differences below about 0.005 are noise.
3. **Oracle rows (X2).** Refused by `guard_oracle` in every selection.
4. **Fold-average full maps.** Each fold model saw the other zones' labels, so these maps cannot be validated on
   held-out zones. They are a diagnostic only.
5. **Parity.** A deployed map must be the evaluated construction. Answered by the zone-stitched submission
   (`final_submission.py`).
6. **Leaderboard feedback.** Never an input to selection (`leaderboard.py` only records and compares).
7. **Per-zone fusion weight.** Evaluation uses the champion's per-zone nested `w`; deployment uses the all-zone `w`.
   This is irrelevant at blend 1.0, and it is checked as a WARNING-level parity row.

## 7. Strongest verified score

* **Public leaderboard.** No public score is recorded in the repository (`reports/p6/leaderboard_log.csv` holds only a
  header). If the historical W2 file was scored, record that score with `lb.record` or in the history CSV; it is not
  known here.
* **Strongest nested offline single deployable policy:** `C_w2_skeleton` under `disc`. Disc 0.242156, worst 0.198579,
  geo 0.105687.
  * Its recorded export recipe used the incompatible `full_score.tif`.
  * The zone-stitched builder gives it a construction identical to its evaluation.
* **Honest estimate of the selection process (META):** 0.224640.
