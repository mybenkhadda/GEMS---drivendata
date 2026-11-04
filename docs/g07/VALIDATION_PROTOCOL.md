# G07 validation protocol

* **Views:** V1 complete withheld systems (primary, pooled discovery DTI), V3 geographic folds, V2 network completion
  (diagnostic). GEMS units, FastScorer, M7 protocol keys - identical to W2.
* **Matched comparison:** every score map is evaluated with the same policies and fractions, top-area selections with
  EXACTLY k eligible pixels (`representation.exact_k_mask`: score, then label-free secondary, then a seeded hash of the
  full-grid pixel index). The legacy `score >= threshold` rule is reported next to it (`evaluation/metrics/
  exact_budget.csv`); a model whose rows are not exact cannot be promoted. Portfolio / corridor maps are evaluated as
  produced at the primary budget. No per-zone or per-model favourable representation; representation selection is
  reported separately and nested (leave one zone out).
* **Isolation:** visibilities `V1:d`, `V3:k`, nested `j|k`, `FULL` (G06 splits); every fit predicts only its units;
  `splits/isolation_audit.csv` must be all PASS. Hidden truth never creates inputs, candidates or training rows for
  its holdout; synthetic positives come from TRAINING units' truth only.
* **Strictness:** `STRICT_ISOLATED` (pixel models on visible labels), `STRICT_NESTED` (cross-unit models with nested
  inputs only), `STRICT_LABEL_FREE`, `SCREENING_ONLY` (W2 / CNN inputs or incomplete nesting), `BLOCKED_UPSTREAM`
  (`ranker.require_strict` and a nested input missing). Screening results are never promoted.
* **Candidate training geometry:** synthetic candidates are generated per fit. For held-out unit `s:z`, geometry = the
  other units' truth AND `visible` of fit `s:z`, asserted disjoint from its hidden faults and from unit `s:z`'s truth
  (`LeakageError`). Every row records source scenario, outer and inner fold, training visibility id, generation method,
  mapped-geometry use, training systems and upstream provenance. `crossfit_candidate_ranker` trains outer z only on the
  rows of outer z and re-asserts it; rows without provenance are dropped and the model is `SCREENING_ONLY`. Inference
  candidates stay label-free.
* **U-Net isolation:** targets come from `visible` only (visible / hidden overlap raises). Scaling is fitted on training
  patches. Validation patches lie entirely in one inner fold, training patches touch none of its cells, straddling
  patches are dropped. Evaluation uses only the fit that hides the unit.
* **DEM feasibility** uses held-out truth as a diagnostic of the data source. Thresholds are predeclared and may not be
  lowered after seeing results. **V1 gates; V3 is reported separately and never pooled.** Recovered systems are keyed by
  (scenario, unit, system id) at `feasibility.primary_recall_frac`: identical ids in different units are different
  structures. The coverage-artefact screen uses |rho|: strong negative correlation is an artefact too. DEM-model claims
  must be confirmed on V3.
* **Training gate (`evaluation.training_gate`):** `STRICT_RESEARCH` needs a GO. `EXPLORATORY_TRAINING` also trains after
  NO_GO / INCONCLUSIVE, always `SCREENING_ONLY`: registries, evaluation and ranker inputs carry that strictness, so
  promotion stops at `PROMISING_SCREENING_ONLY` and export refuses. Neither mode bypasses missing files, invalid CRS,
  unknown vertical units, invalid transforms, missing training data or leakage checks.
* **Registry completeness:** expected fits are declared before training. A fit counts only when its registry row is
  finished, its model loads and matches the schema, every owed prediction exists with the unit's shape and finite values
  in [0, 1], and its strictness is recorded (`models/<name>_validation.csv`). Stage: COMPLETE = all valid, PARTIAL = some.
* **Evaluation status:** per model and scenario, FULLY / PARTIALLY / NOT_EVALUATED and STRICTLY_VALIDATED / SCREENING_ONLY /
  BLOCKED (`evaluation/metrics/evaluation_status.csv`). Only fully evaluated pairs enter a pooled DTI; per-unit rows of
  partial models stay in `partial_unit_rows.csv`.
* **Promotion (all required, primary cell):** exact budget; V1 gain >= +0.005 vs matched W2 (prefer >= +0.010); no
  zone drop > 0.01; V3 delta >= -0.005; net extra V1 systems >= 1; leakage audit PASS; strict validation; then at
  export: full-region parity PASS, provenance complete, DEM competition eligibility verified when DEM features are
  used, `EXPORT_SUBMISSION` set, `auto_submit` False. Otherwise W2 is retained.
* **Never:** the 0.40 target presented as achieved; offline proxy DTI called a leaderboard score; an automatic upload.
