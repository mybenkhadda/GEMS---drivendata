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
* **DEM feasibility** uses held-out V1 / V3 truth as a diagnostic of the data source; its thresholds are predeclared
  and may not be lowered after seeing results; DEM-model claims must be confirmed on V3.
* **Promotion (all required, primary cell):** exact budget; V1 gain >= +0.005 vs matched W2 (prefer >= +0.010); no
  zone drop > 0.01; V3 delta >= -0.005; net extra V1 systems >= 1; leakage audit PASS; strict validation; then at
  export: full-region parity PASS, provenance complete, DEM competition eligibility verified when DEM features are
  used, `EXPORT_SUBMISSION` set, `auto_submit` False. Otherwise W2 is retained.
* **Never:** the 0.40 target presented as achieved; offline proxy DTI called a leaderboard score; an automatic upload.
