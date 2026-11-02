# GEMS G05.1 — fault-system recovery & intelligent geological ranking

Code rev `g051.1-r1`. **Authored, not executed**: no number in this folder comes from a run. Every result is produced
in Colab by `notebooks/colab/GEMS_G05_1_SYSTEM_RECOVERY_AND_RANKING.ipynb`.

## Question

G05 found that a larger union of experts covers 46 withheld systems vs 38 for W2 (120 withheld), and that eight
specific systems (zone 0: 471, 492, 531; zone 1: 243, 559; zone 2: 356, 544, 560) were missed by W2. G05.1 asks:

1. **Why** were they missed (generation-, ranking-, or localisation-limited)? → Stage B failure atlas.
2. Does an **equal-budget** union recover them, or did G05's larger pool do the work? → Stage C.
3. Can a **residual ranker on top of frozen W2** turn that into official discovery DTI without losing geographic DTI,
   under an honest protocol? → Stages D–I.

## Layout

| path | role |
|---|---|
| `configs/g05_1/recovery.yaml` | every predeclared threshold, budget, target, gate |
| `gems_bench/g05_1/runner.py` | `G051(DiscoveryGeoMoE)`: stages, caching, archiving, per-stage manifests |
| `gems_bench/g05_1/{bands,geometry,graph,displacement,recall,registry,rankers,crossfit,selection,explain,export,atlas,inventory,report}.py` | pure science + Colab helpers |
| `gems_bench/g05_1/selftest.py` | synthetic self-tests (stage 00) and fixtures of the pytest files |
| `tests/test_g05_1_{features,splits,geometry,crossfit,evaluation,export}.py` | synthetic pytest suite |
| `docs/g05_1/` | this README, experiment design, validation contract, artifact registry, session handoff |

## Running (Colab)

1. Push the branch, pin `CONFIG['ref']` to the commit, open the notebook.
2. First run with the default `RUN` (preflight, atlas, recall, registry, graph).
3. Inspect `atlas/ATLAS_SUMMARY.md`, `recall/recall_summary.csv`, `recall/target_system_recovery.csv`.
4. Only then: `RUN['displacement']`, `RUN['training']`, `RUN['strict']` (+ selection), then `RUN['ablations']`,
   `RUN['reconstruction']`, and `RUN['export']` only if Stage I says `PROMOTE_CANDIDATE`.

Outputs: `MyDrive/GEMS_2/outputs/g05_1_system_recovery/<run_id>/`. See `ARTIFACT_REGISTRY.md`.

## Non-negotiables

* Truth of a held-out unit never reaches a candidate, a feature or a fit (post-hoc overlays only).
* Learned rankers are `SCREENING_ONLY`; strict learned nesting is `STRICT_NESTING_BLOCKED`. Only the label-free S1
  blend can be `PROMOTE_CANDIDATE`.
* Pooled DTI = summed components, never averaged DTIs.
* No automatic upload; the public score (0.0942) and historical numbers are references, never targets.
* An earlier successful stage output is archived to `history/` before it is replaced.
