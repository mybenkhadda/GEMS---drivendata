# G14.1: critical failure analysis and discovery recovery

**Status: UNTESTED.** Nothing in this branch has been executed: no notebook, test, Python code, training, inference or
experiment. Every number quoted below was supplied by the user from the G14 Colab run, or comes from a repository
document.

* Notebook: `notebooks/colab/GEMS_G14_1_CRITICAL_RECOVERY_MASTER.ipynb`
* Package: `gems_bench/g14_1/`
* Tests (unexecuted): `tests/test_g14_1_core.py`, `tests/test_g14_1_torch.py`, `tests/test_g14_1_regressions.py`

## Documents

| document | content |
|---|---|
| [STATIC_CODE_AUDIT.md](STATIC_CODE_AUDIT.md) | 21 findings with file, mechanism, severity, verification and fix; S01 known-context shortcut and S05 protocol mismatch are the leading root-cause candidates |
| [LEAKAGE_RISKS.md](LEAKAGE_RISKS.md) | leakage and shortcut register |
| [GEOMETRY_AND_METRIC.md](GEOMETRY_AND_METRIC.md) | raster parity, metric kernel derivation, hard-coded assumptions |
| [KNOWN_CONTEXT_EXPERIMENTS.md](KNOWN_CONTEXT_EXPERIMENTS.md) | D1–D5: how the shortcut is measured and removed |
| [CORRECTED_EVALUATION.md](CORRECTED_EVALUATION.md) | metric vocabulary, protocols, inner validation, calibration, locked TEST |
| [ARTIFACT_REUSE.md](ARTIFACT_REUSE.md) | what is found where, compatibility rules, resumability |
| [EXPERIMENT_LADDER.md](EXPERIMENT_LADDER.md) | R00–R17, profiles, the training-spec registry |
| [SUBMISSION.md](SUBMISSION.md) | one-cell submission generation |
| [TESTS.md](TESTS.md) | tests written but never run |
| [UNTESTED_RISKS.md](UNTESTED_RISKS.md) | blockers and uncertainties |

## FAST mode: execution order (no GPU, no training)

1. Cell 01: keep `profile='FAST'`, `mode='AUDIT_ONLY'`. Set `G14_OVERRIDES` to the semantic overrides of the
   original run (`{}` if defaults were used).
2. Cells 02–05: setup, checkout, Drive, **R00** inventory and preflight. Read `preflight.csv`.
3. Cell 06: **R01** geometry and submission parity. The P0 question: is the exported file what we think it is?
4. Cells 07–08: G14 map inventory, experiment-hash check, feature provenance, then **R02** raw probability
   diagnostics and the static leakage audit (overlap, copy baseline, separability).
5. Cell 11: **R03** representation / area / masking-order sweep, then **R04** decomposition and failure classes.
6. Cell 12: sweep tables (requested vs actual area, crest dilation factor) and figures 5–7.
7. Cells 14–15: **R05** champion comparison and **R12** complementarity (cross-fitted blend weights).
8. Cell 27: decision report.

Then STANDARD in `mode='COUNTERFACTUAL'` with `stages.checkpoint_counterfactuals=True`. This runs cell 09 / R06:
inference-only interventions on the existing checkpoints, plus the inner-validation audit. It also adds the atlas
(R05b), calibration and discovery splits (R07).

## Corrected training: execution order (`mode='TRAIN'`, `profile='FULL'`, `stages.train_models=True`)

1. **R07** discovery splits: protocol check, conditional draws, separated-target checks.
2. **R08** D1 known-only and M0 G14 control. These measure the shortcut ceiling.
3. **R09** M1 geophysics-only.
4. **R10** M2 conditional discovery and M3 combined-target control.
5. **R11** objectives, negatives, augmentation and feature ablations. Each is a one-factor change against M1.
6. **R13** N1 / N2 CNN context.
7. **R14** N3 Swin and N5 directional, but only if R13 shows that context helps.
8. **R15** N4 full GeoFaultFormer.
9. **R16** ensemble with the champion, cross-fitted.
10. `pipe.train_final('<best spec>')`, then mode SUBMIT, cell 25 with `SUBMIT_MODEL='g14_1:<spec>'`.

`training.experiments` filters which specs run (default: M1 and M2 only).

## Submission

See [SUBMISSION.md](SUBMISSION.md). In short: set `mode='SUBMIT'`, rerun cells 01 and 05, then run cell 25. It
writes files only, and you upload them by hand.
