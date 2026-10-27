# P6: fast track to the leaderboard (guide and handoff)

Notebook `notebooks/colab/GEMS_P6_FAST_TRACK.ipynb` · code `gems_bench/p6/` · config `configs/p6/fast_track.yaml` ·
outputs `MyDrive/GEMS_2/outputs/p6_fast_track/` · log `reports/p6/leaderboard_log.csv` · tests `tests/test_p6.py`.

**Status (2026-10-27): code authored; nothing has run on real data.** Only the synthetic tests (37 passed) and Ruff
ran locally, in an isolated virtual environment. There is no P6 number and no leaderboard score. Every value below
is a recorded offline value from earlier sessions or a plan.

## 1. Objective and reference

Goal: the strongest valid submission before 2026-12-03 (freeze 11-25, final upload 11-30), within 12 T4 GPU-hours and
20 CPU-hours of new compute. The target is a public DTI of 0.40-0.49. It is ambitious and not guaranteed.

Reference champion: `m9_fallback_fusion_thin.tif` (B0+B1 rank fusion, 1-px crest, ring 0, area fraction from the geo
choice). Recorded offline values: geographic 0.183 (worst fold 0.170), `disc_official` 0.162 (worst 0.126). These
are earlier records. Gates use the values recomputed by M9 section 01 (`baseline_selection.json`) when they exist
(`select.fallback_reference`).

## 2. Plan, cost and gates

| Workstream | Existing modules reused | Files | Est. GPU / CPU | Exit gate |
|---|---|---|---|---|
| W0 probes | `m7.export.export_baseline` (`k_px`, ring), `gems3.submission.{eligible_mask, area_ledger, validate}`, `submission.write_submission`, `artifacts.atomic.sha256_file` | `p6/probes.py` | 0 / 0.3 h | verdict: broad discovery, near-known completion, different budget |
| W1 seeds + TTA | `pipeline.experiments.run_experiment` (seed override), `gems3.official.unit_score`, `inference.full_raster.predict_heads_mask`, `metric_opt.nested_select` | `p6/seeds.py` | ≤ 4 / 1 h | disc ≥ ref + 0.005, geo and worst zone no regression |
| W2 stacker | `unit_score`, G02 `rasters/units/*.npz` + `G02Experiment.tile_maps`, `m8.potential_fields`, `m7.ranking._zone_mask`, `R.apply`, `m7.protocol.{score_row, system_hits}`, `m7.baseline.heldout_map` | `p6/stack.py` | 0 / ≤ 6 h | disc ≥ max(0.167, ref + 0.005), worst ≥ 0.126, geo ≥ ref − 0.005 |
| W3 M9 throughput | `m9.dataset`, `m9.model`, `m9.losses`, `m9.train.inner_score` | `p6/throughput.py` | ≤ 2 / 0.2 h | ≤ 90 s / epoch, else stop M9 |
| W4 DEM pilot | `m7.dem_morph` (mosaic, sections 05 / 06), `m7.protocol.system_hits` | `p6/dem_pilot.py` | 0 / ≤ 6 h | ≥ 30 % trace recall at 40 km / 100 km², ≥ 3 new systems, shift ≤ 1 cell |
| W5 completion | `gems3.map_completion.{corrupt, endpoint_extension}`, `m7.export._write` | `p6/completion.py` | 0 / ≤ 1 h | blocked unless W0 = near-known completion |
| W6 selection | `p3.lineaments`, `gems3.workspace.save_lines` | `p6/select.py` | 0 / 0.5 h | ordered criteria |

Caps are in `budget.workstreams`. `run.Budget` records the wall time of every step in `manifests/compute_ledger.csv`,
and `BudgetExceeded` stops a workstream once its cap is spent. Total caps: 6 GPU-h, 16 CPU-h.

## 3. Workstreams

**W0 (`probes.py`).** A is the existing fallback (read-only; if it is missing it is rebuilt with the identical rule
and labelled as such). B keeps A's rule and A's number of selected pixels and adds a 3 px **suppression** ring. C uses
A's rule with the nested `disc_official` area fraction; it is skipped when that fraction equals A's. D is a
diagnostic: 1 on every template-finite pixel that is not a known fault.

* Interpretation: B > A means predictions near known faults cost more than they earn, so the evidence favours broad
  discovery. B < A means new public truth lies next to known traces, so W5 is activated. The brief stated this
  mapping the other way round; in this repository the ring removes predictions.
* D gives a bracket for the public new-fault density, ρ ∈ [0.2/(1/DTI − 1 + 0.2 c_max), 0.2/(1/DTI − 0.8)], with
  c_max = Σk ≈ 9.38. It is valid only under the audited scorer contract. It is never a point estimate and never a
  candidate.
* Log: `submission_file, sha256, model_source, area_frac_eligible, representation, ring_px, valid_px, positive_px,
  positive_frac_valid, date_utc, public_score (manual), offline_disc_dti, offline_geo_dti, notes`.

**W1 (`seeds.py`).** Uses at most 3 members per family. Extra seeds are allocated round-robin within the remaining
GPU budget. Family maps are rank-averaged. 180° TTA is re-predicted with the same held-out checkpoint
(`cv_fold<k>` / `disc<d>`). The evaluation uses the champion's own grid with nested selection. Sanity check: with one
member per family and no TTA, the `seeds` variant must reproduce the fallback's numbers.

**W2 (`stack.py`).** The feature set has 15 features or fewer, in priority order:
* CNN ranks: B0, B1 and their disagreement.
* G02-C, G02-A, G02-F and G01 persistence, when present in every unit.
* Magnetic and gravity horizontal gradient, magnetic tilt, and magnetic–gravity strike agreement.
* Geometry of the *permitted* known map: distance, density, distance to a trace tip, and alignment with the tip.

The model is LightGBM, fitted leave-one-zone-out with a 2 km buffer and fixed hyper-parameters. Labels are positives
from the unit's truth and sampled negatives that never touch a mapped fault, the discovery ignore ring or a 3 px
buffer. Label-free ambiguous negatives (top 5 % CNN / G02-C evidence) get weight 0.3. Cross-modal store layers and
DEM features were left out to stay in scope; DEM is to be added only after W4 passes.

Leakage guards (`LeakageError`):
* CNN provenance per unit: OOF on geo folds, the split's own discovery map on discovery splits.
* G02 raster metadata must match the unit (task, zone, window, no labels stored).
* Known-fault features use only the faults visible to the unit.
* Features are ranked inside each unit's evaluation mask.

**W3 (`throughput.py`).** Times each part of an epoch on the real M9 code path, for a memory-mapped or RAM cube and
pageable or pinned batches: patch read, scale, context, corruption, batch build, host-to-device copy, forward and
backward pass, inner validation. Epoch time ≈ steps × accumulation × max(batch, host-to-device + forward/backward) +
inner validation / every. Optimisations go to the dominant component only, after measurement.

**W4 (`dem_pilot.py`).** Reuses the cached M7 mosaic. If it is missing, runs M7 sections 05 / 06 with the pilot capped
at 30 tiles; downloads still need M7's explicit approval flag. Evidence is the mean rank of scarp density, scarp
length, persistence and micro-relief inside covered cells. Coverage is only a mask.

**W5 (`completion.py`).** Tip extension (≤ 1 km), gap bridging between facing tips (≤ 0.6 km, ≤ 15°) and a
DEM-snapping constraint check. It is benchmarked on M7 corruption classes. The submission keeps the fallback's pixel
count.

**W6 (`select.py`).** Ranks the fallback and every candidate that passed its gate by discovery DTI, then worst zone,
geographic DTI, unique systems and public score. Within the 0.003 tolerance the next criterion decides; the fallback
wins a full tie. Writes the GPKG of the chosen lines with B0 / B1 / fused (/ stack) ranks. Segment confidence is a
heuristic rank, not a probability.

## 4. Colab execution order

1. Section 0 (GPU runtime only when W1 trains or runs TTA, or for W3).
2. Section 1 (W0) → upload A, B, C manually (3 scored submissions per rolling 7 days) → type the scores in 1b → read
   the verdict. D only with a spare slot.
3. Section 3 (W2, CPU). This is the main experiment, independent of W0.
4. Section 2 (W1): plan first; `TRAIN_SEEDS = True` only if the plan fits.
5. Section 4 (W3, GPU), short.
6. Section 5 (W4) only when the M7 mosaic exists or downloads are approved.
7. Section 6 (W5) runs only after a W0 near-known verdict.
8. Section 7 (W6) by 2026-11-25; upload the chosen file by 2026-11-30. At most one controlled probe per promoted
   candidate.

Every section reloads from persisted files. After a disconnect, run section 0 and then any section.

## 5. Outputs (`outputs/p6_fast_track/`)

`manifests/{compute_ledger.csv, sections/*.json}` (configuration fingerprint, code provenance, seconds) ·
`probes/{leaderboard_log.csv, probe_plan.csv, probe_meta.json, interpretation.json}` ·
`seeds/{train_plan, grid, summary, per_zone_*, tta_log}.csv`, `seeds/maps/*.npy`, `seeds/candidate.json` ·
`stack/{provenance, row_counts, crossfit_audit, grid, per_zone, unique_systems, feature_importance}.csv`,
`stack/features/{rows.parquet, pf_full.npz, feature_set.json, g02_full/}`, `stack/maps/*`,
`stack/models/stack_full.joblib`, `stack/candidate.json` · `throughput/{timings.csv, decision.json}` ·
`dem/{gate.json, systems.csv, dem_evidence_100m.tif}` · `completion/*` ·
`selection/{fallback_reference.json, candidates.csv, elimination_trace.csv, decision.json}` ·
`vectors/final_lineaments.gpkg` · `submissions/*.tif + .json` (never overwritten; timestamped on collision).

## 6. What remains unverified

* Every P6 number: nothing has run on real data.
* Whether `ctx.zoo5` keys a new seed run by its overridden `name` (W1). If not, `seeds.members` will not see it; check
  `seeds/plan.json` after training.
* G02 run availability and key names (`G02-C`, `G02-A`; `F:` / `G01-` prefixes for G02-F and persistence). Absent
  families are dropped consistently and listed in `stack/features/feature_set.json`.
* Full-region G02 cost (G02-A is the heaviest map). It is needed only after a W2 PROMOTE.
* W2 memory: about 15 features × the crop of each of 6 units held in RAM. Use a high-RAM runtime if it fails.
* M7's DEM pilot quotas sum to 9 tiles. Whether raising `n_pilot_tiles` to 30 fills the quotas is untested.
* TTA assumes `prepare_stack5` gives the same stack as the run's original inference.
* Training–inference mismatch in W2: discovery-split CNN maps are used for training, stitched OOF maps for the full
  region. Ranks reduce, but do not remove, the shift.
* Probe interpretation is public-subset evidence only (organizer: public / private chunking not disclosed).

## 7. Next decisive metric

The W0 public scores of A and B (and C), and `stack/candidate.json` → `disc_dti`, `disc_worst` and `unique_systems`
against `selection/fallback_reference.json`.
