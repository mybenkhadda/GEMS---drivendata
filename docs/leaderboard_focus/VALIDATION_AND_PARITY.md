# Validation and parity

## Two evaluation tasks

* **disc_official (primary).** Pseudo-discovery: withheld fault systems are the truth, retained faults are masked
  pixel-exactly, and predictions on them are ignored. This is the closest analogue of the hidden test, which scores
  new faults only. It is a proxy, not a leaderboard estimate.
* **geo.** Geographic folds: every mapped fault of the held-out fold is the truth. It measures rediscovery of mapped
  faults.

## Official DTI

TP / FP / FN with the 300 m linear kernel, Tversky α = 0.2 (FP), β = 0.8 (FN):

DTI = TP / (TP + 0.2 FP + 0.8 FN + ε)

The scores are pooled over zones. The implementation is `gems3.official.FastScorer`; M1 never re-implements it.

## Out-of-fold requirement

Every score map used for a zone must come from a model that did not see that zone's labels:

* CNN: OOF on geo folds; the split's own discovery map on discovery units.
* W2: cross-fitted held-out maps (`stack/maps`).

## Deployable policy consistency

A deployable policy is ONE (rep, frac, blend) cell from ONE selection. It is applied unchanged to every zone of the
exported map.

The nested estimate:
1. For each outer zone, choose the cell on the other zones.
2. Apply it to that zone in both tasks.

The deploy cell is the cell chosen on all zones. `export_parity.build_recipe` refuses rep, frac or blend coming from
different selections.

## Which numbers are comparable

Three evaluation protocols appear in the records. Only values under the same protocol, or the same selection
criterion, can be compared with the champion (`validation/protocol_comparison.csv` lists every value with its label).

| protocol | how the cell is chosen | one deployable file? | recorded examples |
|---|---|---|---|
| `single_policy` (the champion) | per outer zone, on the **discovery** labels of the other zones; applied to both tasks | yes | P6.1 X1: W2 original 0.224640 / geo 0.102664; crest 0.238882 / 0.114259; skeleton 0.242156 / 0.105687; M9 fusion 0.162482 / 0.158416 |
| `separate` (boost A, W2's own grid) | discovery cell on discovery labels, geo cell on **geo** labels, independently | no (two different cells) | W2 original geo 0.180843; crest geo 0.185609; skeleton geo 0.168841 |
| `fixed` | one historical cell | n/a | uploaded cell: disc 0.178905, geo 0.168841 |

* **Discovery values are comparable** across `single_policy` and `separate`: both pick the discovery cell on
  discovery labels (the records agree: 0.242156 in both).
* **Separate-protocol geographic values are not comparable** with a single-policy geographic DTI. A separate geo
  value comes from a cell chosen to maximise geo; the deployable policy applies the discovery-chosen cell. The gap
  (0.168841 → 0.105687 for skeleton) is the cost of deploying one policy, documented by the P6.1 X1 table — it is not
  evidence of a software defect, and M1 does not treat it as one.
* **The 0.1758 floor** (`configs/p6/{boost,recovery}.yaml`) sits between those regimes: no W2 single-policy record
  reaches it, and neither does the M9 fusion fallback (0.158416). Its derivation is not recorded in the configs.

Recorded values are still recomputed from the frozen grid (an integrity check): `REPRODUCED`, `NOT_REPRODUCED` or
`NOT_AVAILABLE`. A non-reproduced value is a WARNING (default) or blocking (`validation.recorded_mismatch:
blocking`); the binding checks remain the per-zone component recomputation and the sha256 chain.

## Promotion gates (`selection/promotion_gates.csv`)

A COMPLETE selection is not a promotion. READY_FOR_EXPORT requires every mandatory gate:

| gate | PASS when |
|---|---|
| `SELECTION_COMPLETE` | selection COMPLETE with one champion and one recipe id across the selection files |
| `DISCOVERY_ELIGIBLE` | eligible under the predeclared p6g rule, worst discovery zone ≥ 0.153414 |
| `GEOGRAPHIC_GATE_PASS` | single-policy geo DTI ≥ 0.1758. `promotion.geographic_gate: mandatory` (default) → BLOCKED below; `advisory` → WARNING, only with a written `override_note` |
| `VALIDATION_PASS` | validation COMPLETE (WARNING passes with its warnings listed), produced for the current selection |
| `PARITY_PASS` | parity COMPLETE, produced for the current validation |

The gates never change the selection: `selection.geo_floor_blocking` (default false) is the separate, older knob
that makes the floor part of the selection rule itself.

### Two decisions, never conflated (M1.3)

`gates.decisions` writes both to `selection/promotion_gates.json` and the promotion manifest:

| decision | gates | question |
|---|---|---|
| `technical_validity` | SELECTION_COMPLETE, VALIDATION_PASS, PARITY_PASS | can the recipe be written as a file that meets the official contract and reproduces the evaluated prediction? |
| `internal_promotion` | DISCOVERY_ELIGIBLE, GEOGRAPHIC_GATE_PASS | does the model meet our own quality rules? |

A candidate can be technically valid and still fail internal promotion. The floor is never lowered automatically.
When the geographic gate is the **only** failure (`exploratory_eligible`), the opt-in exploratory export (stage 5b,
`exploratory.enabled` + a written `reason`, default OFF) writes `exploratory_m1_<family>_<recipe_id>.tif` with the same
checks as a champion file, status `EXPLORATORY_READY` (never `READY_FOR_UPLOAD`), its own manifest / sha256 / README. The
champion submission stays BLOCKED, and the file is never described as the validated champion. Upload is manual.

## Per-zone area budget (M1.3 contract)

`frac` is the share of the zone's **eligible** domain the selector may take: pool = ceil(frac × eligible_px)
(`representation._k`). Thin representations keep only part of that pool: `skeleton_1px` skeletonises the top-pool
mask; `crest_1px` takes the top crest pixels, all of them when there are fewer. The benchmark DTI was measured on
exactly that thinned area. A zone therefore passes when:

* area / eligible ≤ frac × (1 + `budget_tolerance`) (0.02, `configs/p6/full_gpu.yaml`) — the same rule as
  `export_parity.audit_export` on the written file and `boost_audit.budget_violations` on the grid;
* no positive pixel lies on a mapped fault (the official pixel-exact mask);
* the map is not empty.

`validation/zone_budget.csv` records `pool_px`, `cap_px`, `pool_yield` (= positive / pool), `rel_error` (final area vs
frac, informational), the evaluated cell's area (`eval_positive_px`, `eval_area_frac_eligible`), the old two-sided
verdict `m1_1_two_sided_ok`, and a `cause`.

**Why it changed (OBSERVED on Drive, M1.2 run 2026-10-28).** Champion cell skeleton_1px @ 0.01:

| zone | eligible px | positive px | area / eligible | pool yield | evaluated cell px (area) | px on mapped faults |
|---|---|---|---|---|---|---|
| 0 | 1,868,064 | 9,232 | 0.494 % | 0.494 | 7,648 (0.405 %) | 0 |
| 1 | 1,526,003 | 7,001 | 0.459 % | 0.459 | 6,009 (0.388 %) | 0 |
| 2 | 1,709,351 | 8,089 | 0.473 % | 0.473 | 6,806 (0.394 %) | 0 |

M1.1–M1.2 required |area / frac − 1| ≤ 0.02. No thin cell can satisfy that, including the evaluated grid cells
themselves, so every zone failed (rel_error −0.51 to −0.54). This was a `CHECK_DEFINITION_MISMATCH`, not a builder
defect: no over-allocation, no mapped-fault leak. The repair changes the check only. The prediction, the recipe id
`0a8692b2ab` and the validated array are unchanged.

**Evaluated vs deployed map.** The deployed zones have 17–21 % more pixels than the evaluated cells (binary Jaccard
0.53–0.55, score Spearman 1.0; `parity/zone_stitched_parity.csv`, OBSERVED). This is by design. The geo-unit
evaluation keeps the zone's mapped faults as truth, so part of its pool lands on them. The deployed eligible domain
excludes mapped faults (they are masked by the scorer), so the whole pool lands elsewhere and survives
skeletonisation. `diagnostics/representation_comparison.csv` splits the disagreement per zone into
*evaluated-only on mapped faults* (official mask) and *re-ranked pool* (`map_comparison`). Measured in Colab only.
Residual caveat: the geographic DTI is measured on the evaluated map, not on the deployed one. On the leaderboard the
mapped faults are masked, so the deployed map's quality on new faults is estimated, not measured.

## Known parity problems (historical W2 export)

See `M1_AUDIT.md` §3. The historical export had three defects:

1. **Rank domain.** Features were ranked inside the whole footprint at inference but inside each unit during
   training. Recorded as BLOCKED for `w2_score` and `cnn_disagree` on the geo units.
2. **In-sample refit.** `full_score.tif` came from a refit on every row.
3. **Mixed policy.** The uploaded cell took rep from one selection and frac from another.

M1 does not repair that export. The champion is built zone by zone from the evaluated held-out maps
(`final_submission.build_map`), which removes all three causes.

## M1 diagnostics (`parity/parity_report.csv`, scope / source labelled)

| scope | check | source |
|---|---|---|
| historical | recorded parity rows by cause | P6.1 table |
| historical | W2 held-out map vs `full_score.tif` crop (Spearman, top-k) | recomputed |
| historical | OOF vs in-sample | the code of `stack.full_region` |
| historical | historical recipe (frac source) | historical sidecar |
| champion | fold assignment, duplicated / missing pixels | recomputed |
| champion | zone score vs evaluation construction; binary map agreement | dry build |
| champion | one all-zone selection (recipe id) | notebook 01 |

Only champion rows can block. A WARNING on "binary map vs evaluation map" is expected by design: the full-region
eligible domain masks mapped faults, while geo evaluation counts them as truth.

## Failure diagnostics

| symptom | likely cause | action |
|---|---|---|
| recomputed ≠ grid | a frozen input changed after boost A | compare the provenance hashes; restore the input |
| recorded `NOT_REPRODUCED` (WARNING) | a different grid on Drive, or a different selection tie-break | compare `grid_provenance.csv` sha256 with the boost-A manifest; never relax the tolerance |
| zone parity WARNING | the blend < 1 champion uses a per-zone vs all-zone fusion w | explicit decision (`strict_parity`) |
| budget fail (M1.3 rule) | over-allocation, mapped-fault leak or empty map | read `cause` in `zone_budget.csv` and `diagnostics/root_cause_summary.json` |

## What a valid champion is

All of the following must hold:

* every required artifact is OK (stage 1);
* the champion is eligible under the predeclared p6g rule with one deploy cell (stage 2);
* the components are recomputed exactly; recorded values are reproduced or their difference is explained (stage 3);
* the zone parity is PASS, the per-zone budget and the known mask hold, and the recipe is consistent (stages 3–4);
* every mandatory promotion gate passes, including the geographic floor unless you made it advisory (stage 4);
* the written band is the validated array and every official and audit check passes (stage 5).

An exploratory file (stage 5b) meets every bullet except the geographic floor. It is a leaderboard baseline, not a
valid champion.
