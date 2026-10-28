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
| recorded not reproduced | a different grid on Drive | restore it; never relax the tolerance |
| zone parity WARNING | the blend < 1 champion uses a per-zone vs all-zone fusion w | explicit decision (`strict_parity`) |
| budget fail | eligible-domain or known-mask problem | inspect `zone_budget.csv` |

## What a valid champion is

All of the following must hold:

* every required artifact is OK (notebook 00);
* the champion is eligible under the predeclared p6g rule with one deploy cell (notebook 01);
* the components are recomputed exactly and the recorded values reproduce (notebook 02);
* the zone parity is PASS, the per-zone budget and the known mask hold, and the recipe is consistent (notebook 02);
* the written band is the validated array and every official and audit check passes (notebook 03).

The geo floor (0.1758) is reported; it blocks only when `selection.geo_floor_blocking` is set.
