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
| budget fail | eligible-domain or known-mask problem | inspect `zone_budget.csv` |

## What a valid champion is

All of the following must hold:

* every required artifact is OK (stage 1);
* the champion is eligible under the predeclared p6g rule with one deploy cell (stage 2);
* the components are recomputed exactly; recorded values are reproduced or their difference is explained (stage 3);
* the zone parity is PASS, the per-zone budget and the known mask hold, and the recipe is consistent (stages 3–4);
* every mandatory promotion gate passes, including the geographic floor unless you made it advisory (stage 4);
* the written band is the validated array and every official and audit check passes (stage 5).
