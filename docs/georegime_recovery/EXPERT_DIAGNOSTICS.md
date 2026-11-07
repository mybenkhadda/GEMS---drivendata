# Expert diagnostics (section 10)

V1 mixes three heads:

`prob = (1 - s) · σ(global) + s · (r0 · σ(e0) + r1 · σ(e1))`

with `s = σ(specialist_strength)`, whose bias is initialised at -1.4 (s ≈ 0.2 at initialisation), and `r = softmax(router)`. The router was trained toward the label-free prior `0.5 + 1.5 (mag_edge_persistence - grav_edge_persistence)`.

## Units analysed
`disc0..2` and the full model, all from cached full-area heads. Correlations use one random pixel per 1 km block, because neighbouring pixels are not independent.

## Measures (`04_expert_analysis/`)

| file | content |
|---|---|
| `expert_diversity.csv` | Per head: mean, median, q99, q99.9, std, mass, share ≥ 0.5 (eligible, outside mapped faults). |
| `expert_similarity.csv` | Per pair (e0/e1, global/e0, global/e1, global/prob, e/prob): Spearman, Pearson, top-1% Jaccard, median log ratio (scale at equal rank), mean absolute difference. |
| `routing_statistics.csv` | r0 mean / std / share > 0.5; route entropy (bits); s mean / median / q90; mean \|prob - global\|; share with a material change (> 10% relative); top-1% membership change; Spearman(r0, label-free prior); r0 and s by geographic zone and by structural environment. |
| `expert_performance.csv` | Full-area discovery DTI components of each head on the withheld labels (disc folds). |
| `expert_environment.csv` | Weighted recall of held truth and FP-mass share per environment. Environments are prior < 0.35 (gravity-dominant edges), 0.35–0.65, and > 0.65 (magnetic-dominant edges). |
| `expert_summary.csv` | Verdict per unit. |

## Verdict rules
The thresholds are fixed in `config["experts"]` and the labels are descriptive.

| verdict | rule |
|---|---|
| `REDUNDANT_OR_SCALE_ONLY` | Spearman(e0, e1) ≥ 0.95 and top-1% Jaccard ≥ 0.7: the same structures, possibly on different confidence scales (see the median log ratio). |
| `WEAK_MIXING` | Median s < 0.1 and top-1% membership change < 0.1: the specialists barely alter the global head. |
| `DIVERGENT_HEADS` | Spearman < 0.8 and Jaccard < 0.5. This is necessary but not sufficient for complementarity; check `expert_performance` and `expert_environment`. |
| `PARTIALLY_OVERLAPPING` | Otherwise. |

A visually textured router map is never counted as evidence of specialisation. If r0 only reproduces its training prior (high Spearman with the prior), the router has learned no new partition. The V1-era label-derived fault clusters are not used as routing targets.
