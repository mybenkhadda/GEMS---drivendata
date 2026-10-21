# Phase 3 plan, experiments and reload paths

## Priorities
| P | section | experiment ids (outputs/phase3/experiments/…) | training? |
|---|---|---|---|
| P0 | §1 3A audit + 10 transform families on G3, G1, G4, B0_lineament, ens_rank_mean | — (tables only) | no |
| P0 | §2 scoring tests + metric correspondence | — | no |
| P0 | §3 3B recall-area-DTI (9 Phase-2 maps), corrected pseudo-discovery (G1, G2, G3, B0) | — | no |
| P1 | §4 3C LightGBM grid D0-D6, F1-F3 on G3 | `p3_lgbm_<cfg>_G3_s2026`, `disc<d>_<cfg>` (5 cfgs) | CPU |
| P1 | §5 3D ablation A0-A5 | `p3_abl_<A>_<base>_s2026`, `disc<d>_abl_<A>` | CPU |
| P1 | §6 3E CNN-A bce_dice / dti, CNN-B bce_dice | `p3_<CNN>_p256_s2026`, `disc<d>_CNN_A_bcedice` | GPU |
| P2 | §7 3F lineaments from best GBM OOF | — | no |
| P2 | §8 3G 0.1125 decomposition, nested ensembles, discovery ensemble | — | no |
| P2 | §9-10 final comparison, error atlas, findings, leaderboard probes | — | no |

## LightGBM grid (gems_bench/p3/lgbm3.py::LGBM_GRID)
C1 = Phase-2 (63 leaves, min_child 200, ff 0.8, l2 1). C2 = 15 leaves, min_child 2000, ff 0.5, bagging 0.5, l2 10, max_bin 63.
S1 = Phase-2 sampler; S2 = 20 % positives, 40 % uniform negatives, no ring. ES "any" = Phase-2 behaviour; "auc" = first_metric_only on AUC.
D0 (.05,C1,S1,any) · D1 (.05,C1,S1,auc) · D2 (.05,C2,S1,auc) · D3 (.01,C2,S1,auc) · D4 (.01,C1,S1,auc) · D5 (.05,C2,S2,auc) ·
D6 (.01,C2,S2,auc) · F1 100 rounds C2 · F2 400 rounds C2 · F3 400 rounds C1. Seed 2026, same folds.

## Leakage safeguards
- Transform parameters, area budgets, ensemble weights, fusion fits: chosen on the other geographic folds (`curves.crossfit_select`).
- Per-fold quantiles are label-free; the same rule is applied to full maps at submission time.
- A5 top features: fold-k subset from the fold-k model; discovery split-d subset from the split-d A0 discovery model.
- Pseudo-discovery: withheld systems + 1 km ring weight 0 (GBM sampler `ignore_mask`, CNN loss weight), never positive patch centres.
- CNN targets from visible labels only; patches fully inside the fold training zone; checkpoint on inner-val (training side).

## Reload snippets
```python
from gems_bench.p3 import load_context
ctx = load_context()                                  # data, folds, splits, segments, model zoo
oof = ctx.oof('G3')                                   # Phase-2 OOF;  ctx.oof('p3_lgbm_D2_conservative_G3_s2026') for Phase 3
pred = ctx.discovery_pred('G3', 0)                    # Phase-2 discovery prediction, split 0
import pandas as pd; pd.read_csv(ctx.paths.out / 'matched_budget_comparison.csv')
```
CNN checkpoints: `experiments/p3_<CNN>_p256_s2026/models/fold<k>.pt` (`cnn3.load_cnn3_fold_models`).
Normalisation: `models/normalisation.json`. LightGBM models: `models/fold<k>.txt`.

## Deferred / not implemented
Region-adaptive fusion (only if per-fold weights agree), local/regional two-branch CNN, SegFormer / foundation models,
CNN test-time augmentation, probability-map thinning in submissions (only if `thinned_vs_topmask_dti.csv` validates it).
