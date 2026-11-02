# G05.1 experiment design

All thresholds live in `configs/g05_1/recovery.yaml` and are predeclared. Nothing is tuned on an outer held-out
zone, outer truth or a leaderboard score.

## Units and scorer (reused from GeoMoE)

* V1 = official discovery split (withheld systems of zones 0–2); V2 = network completion inside geo folds; V3 =
  geographic folds. One unit per (scenario, zone).
* Candidates = GeoMoE stage-04 inventory (W2 / CNN / fusion / potential-field / G01-G02 / network pools,
  deduplicated, split to ≤ max segment length).
* Score: official FastScorer components (300 m linear kernel, pixel-exact known-fault mask); DTI =
  tp / (tp + 0.2 fp + 0.8 fn). System recovered = support ≥ 0.5 on ≥ 50 % of its scored pixels.
* Reference = the frozen submitted W2 policy (factory construction, frozen cell, no selection).

## Stages

| stage | what | key outputs |
|---|---|---|
| A | dependency inventory, grid contract, exact bands, G05 run by manifest | `preflight/` |
| B | 8 targets + 2 matched recovered controls each; 5 panels; metrics; observation vs HYPOTHESIS | `atlas/` |
| C | experts and unions at 0.5/1/2 % (final) and 5/10 % (ceiling), max-rank and quota unions, leave-one-out | `recall/` |
| D | registry D1–D7 per candidate; dictionary; provenance; geometry | `registry/` |
| E | shifted-profile correspondence on `rtp`, `iso_grav_anom` vs same-side control | `displacement/` |
| F | continuity / branch graph, components, support, regional agreement, tip distance | `graph/` |
| G | G0–G5 + S1, cross-fitted | `rankers/` |
| H | nesting audit, strictness, retraining plan | `nesting/` |
| I | reps × budgets, fixed policy, geographic guardrail, promotion | `selection/` |
| J | family ablations at the chosen cell, TreeSHAP | `ablation/` |
| K | research score raster; validated candidate if promoted | `reconstruction/`, `submissions/` |

## Failure categories (Stage B)

`NOT_FOUND_IN_UNIT` (id not among scored truth), `NOT_IN_INVENTORY` (generation-limited), `IN_INVENTORY_NOT_SELECTED`
(ranking-limited), `PARTIAL_SELECTION` (localisation / continuity), `RECOVERED` (controls). The hypothesis line is a
rule-based reading and is labelled as such.

## Recall constructions (Stage C)

Expert score = unit rank of the mean of its member segment means. Union max-rank = max of member unit ranks, one
budget. Union quota = each of n members gets budget / n, OR-ed. Anchors: W2 pixel map at the same representation /
budget, and the submitted W2 policy at its frozen cell. The oracle ceiling (truth-ordered) is copied from GeoMoE and
labelled diagnostic.

## Rankers (Stage G)

Score = logit(W2 unit rank) + λ · correction. Labels: positive (near-truth share ≥ 0.5), ambiguous (0.1–0.5,
excluded), hard negative (< 0.1 and W2 rank ≥ 0.9), unlabelled (PU weight 0.3); weights × length. λ ∈ {0, .25, .5, 1}
chosen on inner (training) zones by weighted AP; ties → weaker correction. G1 residual LightGBM (init score = W2
logit); G2 lambdarank in 20 km blocks; G3 G1 gated to W2 rank ≤ 0.7 and evidence rank ≥ 0.7; G4 + spatial domain,
doubled regularisation; G5 + graph features; S1 = 0.75 · W2 rank + 0.25 · evidence rank (no fit).

## Selection and promotion (Stage I)

Score map = segment score + 1e-9 · pixel W2 rank (partial segments). Fixed policy = the (rep, budget) cell maximising
the mean-over-scenarios pooled DTI on development units only. Gates (all required): disc gain ≥ +0.005 vs frozen W2
(recommend ≥ +0.010); no zone drop > 0.01; V3 DTI under the exact frozen cell ≥ −0.005 vs frozen W2; net extra V1
systems ≥ 1; provenance and upstream honesty verified; export parity verified on the written file (Stage K).
SCREENING_ONLY models stop at `PROMISING_SCREENING_ONLY`.
