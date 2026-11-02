# G05.1 validation contract

## What is honest by construction

* **Candidates** are label-free (GeoMoE pools from honest W2 / CNN maps and label-free layers).
* **Features** (registry, displacement, graph) read no truth; `registry.assert_label_free` raises on any `y_*`,
  `*truth*`, `system_id`, `recovered` column. `y_near_truth` is a target only.
* **Cross-fitting**: a held-out unit (scenario s, zone j) is scored by models fitted on the other zones of s only,
  minus candidates whose centroid lies in zone j dilated by the training buffer; `assert_disjoint` is executed.
  `tests/test_g05_1_crossfit.py` checks that permuting the held-out truth leaves held-out scores unchanged.
* **Selection**: fixed policy on development units; oracle rows are refused (`guard_oracle`).
* **Atlas truth** is drawn post-hoc only; the atlas feeds nothing.

## What is NOT strictly nested (and is labelled so)

The W2 / CNN maps of training zone k were fitted with zone j's withheld systems as ordinary labels, and
`p6.stack.permitted_known(k)` keeps zone-j truth pixels as known faults. A learned ranker therefore sees features
shaped by zone-j truth. Consequences:

| model | strictness | may be promoted |
|---|---|---|
| G0 (segment W2) | STRICT_LABEL_FREE | reference only |
| S1 (fixed blend, predeclared weight) | STRICT_LABEL_FREE | yes, if every gate passes |
| G1–G5 | SCREENING_ONLY | no — at best `PROMISING_SCREENING_ONLY` |

Strict learned nesting = `STRICT_NESTING_BLOCKED`: it needs W2 and CNN refits per (outer j, inner k) pair with
withheld(j) ∪ withheld(k) removed, and recomputed known features (`nesting/retraining_plan.csv`). CNN refits are GPU
work and are refused in this notebook.

**A screening score is never reported as nested DTI.**

## Reporting rules

* Pooled DTI = summed components; worst-zone DTI reported with every pooled number.
* 5 / 10 % budgets are recall ceilings (`role = diagnostic`), never deployable.
* G05 numbers and the public score are references of other protocols (`configs/g05_1/recovery.yaml: historical`).
* A stage that is BLOCKED_UPSTREAM / NOT_EVALUATED / NOT_REQUESTED supports no conclusion
  (`audit/stage_integrity.csv`).

## Export

`research_score_*.tif` is never a submission. A validated candidate requires Stage I `PROMOTE_CANDIDATE`,
`RUN['export']`, and every technical + read-back check (budget, official mask, coverage, recipe audit, array SHA
parity). Status `READY_FOR_MANUAL_UPLOAD` at best; no code path uploads.
