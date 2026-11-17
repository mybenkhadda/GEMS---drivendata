# G14 losses

Implemented in `gems_bench/g14/losses.py`; combined by `CompositeLoss`. **UNTESTED.** Every term is masked by
`loss_mask`, so pixels outside the unit's training geography contribute nothing.

| term | default weight | definition | why |
|---|---:|---|---|
| masked_bce | 0.35 | BCE with `pos_weight`, negatives weighted by `neg_weight` (near-trace unlabeled down-weighted) | calibrated base; PU-aware |
| soft_tversky | 0.25 | 1 − TP / (TP + α FP + β FN), α 0.3, β 0.7 | extreme class imbalance (~1 % positives) |
| distance_centerline | 0.20 | BCE against the soft (Gaussian-distance) target | rewards near misses like DTI's 300 m kernel |
| soft_cldice | 0.15 | soft skeleton (iterated min/max-pool) precision / recall; skipped when a batch has no positives | thin, connected traces |
| background_sparsity | 0.05 | mean probability on reliable negatives | DTI penalises diffuse FP mass |
| orientation (aux) | 0.10 | 1 − cosine between predicted and target axial vector, weighted near traces | strike structure |
| connectivity (aux) | 0 (T5: > 0) | BCE on endpoint / junction maps | topology |
| dti_surrogate (aux) | 0 | differentiable mirror of the official DTI construction (distance kernel 1 − d/300 m, α, β) | align objective and metric; labelled a surrogate |
| deep_supervision (aux) | 0.30 | masked BCE on decoder aux logits | optimisation |

## Stability rules

* Empty positives: clDice is skipped (validity flag); Tversky uses ε; the total stays finite (`test_losses_finite_*`).
* Empty loss mask: every term returns 0 with a gradient path.
* Heteroscedastic mode: logits are sampled as `logit + exp(0.5·logvar)·ε` (T samples) before BCE.
* Logged per term (`manifests/{unit}_train.json`) so ablations can attribute changes.

## Planned ablations

* `A_loss_bce_only`;
* `A_loss_no_cldice`;
* `A_loss_no_tversky`;
* `A_loss_dti_surrogate`;
* `A_loss_no_pu_weighting`.
