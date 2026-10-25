# M6 research plan: model selection, ensemble optimisation and geological error analysis

**Goal.** Get more value out of the completed FULL experiments before any new training. Find the most reliable
model or ensemble for discovering unmapped faults, under strict spatial validation, or record that no candidate
passes.

**Central question.** Can B0's short-fault recall and retained-FP robustness be combined with B1's geographic
generalisation and discovery DTI into a more reliable fault-discovery system?

**Constraint.** M6 does no neural training (`mode: ANALYZE_EXISTING`; `train_models: true` is refused). It reads
the saved FULL predictions. It never writes into experiment folders, the registry or submissions.

## Starting point (reference observations, not results)

The previous notebook review reported the FULL scoreboard below. M6 treats these numbers as **observations to
check**. `tables/reference_check.csv` compares them with the scores each experiment persisted and with M6's own
recomputation on one protocol. Differences are reported, never reconciled.

| model | geo DTI | worst fold | discovery DTI (excl) | retained-FP DTI | < 2 km recall |
|---|---:|---:|---:|---:|---:|
| B0 `P5_B0_gated` | 0.16456 | 0.15341 | 0.20304 | 0.09635 | 0.19240 |
| B1 `P5_B1_bgw` | 0.16788 | 0.16309 | 0.20860 | 0.09128 | 0.17954 |
| B2 `P5_B2_nnpu_p010` | 0.13618 | 0.12436 | 0.20808 | 0.09003 | 0.15444 |
| C1 `P5_C1_heads` | 0.14246 | 0.11691 | 0.19573 | 0.09512 | 0.14651 |

## Workstreams and hypotheses

| id | question | method (module) | decision it feeds |
|---|---|---|---|
| M6.1 | What does each model actually have on Drive, and is it comparable? | read-only artifact inventory, provenance, pairwise compatibility (`evaluation.comparative`) | which models enter the analysis |
| M6.2 | Is B1's geographic gain consistent across regions? | one consolidated section-17 scorecard; per-fold / per-split paired deltas; paired spatial-block bootstrap | gate criteria 2, 3, 7 |
| M6.3 | Are B0 / B1 poorly calibrated, and does it matter for DTI? | `p3.transforms` families cross-fitted; raw-vs-rank invariance; absolute-threshold vs area-fraction transfer (`evaluation.calibration`) | area-budget policy, gate criterion 6 |
| M6.4 | Does a B0+B1 blend beat both? Do B2 / C1 add anything? | nested weight + budget selection (`discovery.nested_selection`) | the candidates |
| M6.5 | Do B0 and B1 make different geological errors? | equal-area overlap, recovery classes by length / fold / strike / contrast / junction / edge, discovery systems, FP distance profile, case-study panels (`evaluation.complementarity`, `evaluation.geological_errors`) | interpretation, which hypotheses to test next |
| M6.6 | Can lightweight topology postprocessing help without losing short faults? | per-operation nested ablation (`postprocess.topology`) | optional export rule |
| M6.7 | Which candidate (if any) is justified? | section-17 gate extended, against both B0 and B1 (`evaluation.selection_gate`) | recommendation |

## Leakage rules (enforced in code, audited in tables)

1. **Geographic.** For held-out fold *k*, every label-dependent choice (area fraction, calibration parameter,
   ensemble weight, postprocessing parameter) maximises the pooled DTI of the folds ≠ *k* (`crossfit_select`,
   `_select_for_zone`). Fold *k* is scored once.
2. **Refinement.** A finer weight grid around fold *k*'s coarse choice is used only when fold *k*'s own dev choice
   is interior. Fold *k*'s coarse choice never sees weights that another fold's refinement added, because those
   were centred using fold *k*'s labels.
3. **Pseudo-discovery.** Weights for split *d* use either the geographic OOF maps with split *d*'s withheld
   systems and ignore ring removed from truth and from the scored area (`geo_masked`, primary), or the other
   splits (`loso`). The discovery area budget is cross-fitted over splits by `discovery3`.
4. **Normalisation** (per-fold rank, robust quantile) is label-free.
5. **Audit.** `ensemble_index_audit.csv`, `ensemble_selection_discovery.csv`, `calibration_selection_audit.csv` and
   `postprocessing_selection_audit.csv` list the selection zones and the held-out zone of every choice.
   `index_audit` raises `LeakageError` on any overlap, and the gate rejects a candidate whose audit shows one.

## Statistics

There are three geographic folds and three pseudo-discovery splits. M6 therefore reports per-unit differences,
directional consistency, the worst unit, and the effect size *d_z* (descriptive). It does not report p-values.
The spatial uncertainty unit is the folds' own 20 km block lattice. Blocks are resampled with replacement inside
each held-out zone, with the same draw for both models, at the cross-fitted operating point. The block
decomposition is exact: TP and FN are attributed to the truth pixel's block, FP to the predicted pixel's block,
and the sums equal `BinaryCurve.at`. The interval describes how much a difference depends on which blocks were
scored, conditional on the protocol. Pixels are never treated as independent.

## Order of experiments

1. Audit → consolidated scorecard → paired evidence.
2. Calibration. Expected: per-fold binary masks are invariant to monotone recalibration, which the invariance
   table checks. The open question is budget transfer across regions.
3. B0+B1: mean, rank mean, nested weighted rank, nested weighted robust (geo-masked discovery weights), and
   weighted rank with LOSO discovery weights. The **prespecified primary candidate** is the nested weighted rank
   blend with geo-masked discovery weights.
4. Extended sets (B0+B1+B2, B0+B1+C1, all four) only afterwards. An extra member is "justified" only if it adds at
   least `min_member_gain` geographic DTI **and** does not reduce primary discovery DTI, relative to the best B0+B1
   candidate.
5. Complementarity and error analysis.
6. Postprocessing: each operation alone, then a combination of only those that helped alone without costing more
   than `max_short_recall_drop` of < 2 km recall.
7. Gate and report.

## Gate (M6.7)

The section-17 criteria come from `configs/validation/phase5_protocol.yaml` (unchanged thresholds). M6 adds:
short-fault recall drop ≤ 0.01, cross-fitted area ≤ 1.5× the control's, block-bootstrap P(geo Δ > 0) ≥ 0.5 when
geography is claimed, leakage-free audit, COMPLETE FULL members only, and inference cost ≤ 4× a single model.
Each candidate is assessed against **both** B0 and B1, and the overall verdict is the most cautious one. A
candidate is recommended only with `PROMOTE` against both. Among those, the rule picks the highest primary
discovery DTI, the protocol's designated objective; no weighted average of metrics is used. Registry promotion,
full inference and submission stay manual.

## Cost

M6 is CPU-only. Prediction rasters (OOF + 3 discovery maps per model) are staged once, converted to local `.npy`
and read memory-mapped. Feature stacks and checkpoints are never staged. The dominant costs, all estimates and
unverified, are: the consolidated scorecard (minutes), the ensemble weight curves (5 weights × 2 normalisations
× 14 budgets × 3 folds), and the postprocessing ablation (≈ 14 operations × 3 budgets × 3 folds per model).
Every unit is fingerprinted and reused on re-run.
