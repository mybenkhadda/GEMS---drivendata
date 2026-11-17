# G14 experiment matrix

The registry is in `gems_bench/g14/experiments.py` (`REGISTRY`, `resolve(name)`, `matrix()`). Ablations keep split
and evaluation settings fixed. Only the `validation` group may change them. **No experiment has been run; every
result cell is empty.**

## Ladder

| id | change vs previous | hypothesis | accept if |
|---|---|---|---|
| T0_champion | references: P6 W2 maps (phase5) or top-3 G13 models (g13) | — | baseline |
| T1_local_cnn | U-Net on the local view only, all channels + known context | dense spatial model beats tabular G13 | disc and geo CI above best G13 / at W2 |
| T2_regional | + Swin regional encoder, gated injection | km-scale continuity helps | paired Δ > 0 with CI excluding 0 on ≥ 1 task |
| T3_global | + global tokens (cross-attention + FiLM) | regime context helps | as above |
| T4_fusion | concat → gated mask-aware fusion | uneven coverage handled better | as above; modality weights sensible |
| T5_topology | + clDice / connectivity weights | fewer fragments, better recall of long faults | DTI not worse; fragmentation down |
| T6_graph | + evidence-gated gap linking | recovers broken traces | DTI ≥ T5 cross-fitted |
| T7_ensemble | cross-fitted rank blend / OOF stack of T-members | complementary errors | DTI ≥ best member |

## Ablations

| group | entries |
|---|---|
| architecture | local encoder unet / resnet / convnext; regional swin / mit / none |
| fusion | concat / gated / cross_attention |
| loss | bce_only, no_cldice, no_tversky, dti_surrogate, no_pu_weighting |
| features | raw_only, no_g13, no_g14_composites, no_known_context, per modality-drop |
| context | regional view 400 m, no context gate, strict context embargo |
| recovery | graph rules only vs classifier |
| validation | phase5 backend, regime holdout, fault-system holdout |

## Reporting (for each experiment, `results/<exp>/`)

* `scorecard.json`: cross-fitted disc and geo DTI, chosen (rep, area) per zone, optimism, edge flags;
* `bootstrap.csv`: CI and paired Δ against T0;
* diagnostics: fragmentation, centreline offset, within-300 m fraction, length-stratified recall;
* training curves and modality weights.

Decision rule: never promote on a point estimate alone. Use the paired block-bootstrap CI and look at both tasks.
