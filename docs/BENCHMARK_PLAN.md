# Phase 2 benchmark plan

## Goal
Reliable, geographically honest baselines under the exact competition metric, answering RQ1–RQ5 (see notebook header),
then a first set of submission candidates. No advanced architectures before the baselines are in.

## Priorities
| P | item | where |
|---|---|---|
| P0 | native-grid data + feature registry | `data.py`, `features.py`, nb §1 |
| P0 | geographic folds (+ buffer, inner val) | `folds.py`, nb §2 |
| P0 | exact DTI scorer + tests | `metric.py`, `tests/test_metric.py`, nb §3 |
| P0 | zero-training floors (lineament / terrain ridge) | `gbm.run_map_baseline`, nb §3 |
| P0 | LightGBM G1 terrain-only, G2 all raw | `gbm.run_gbm_cv`, nb §4 |
| P0 | OOF persistence + minimal comparison | `persist.py`, nb §4 |
| P1 | LightGBM G3 engineered, G4 context | nb §4 |
| P1 | U-Net CNN-A (BCE+Dice, Tversky), CNN-B | `cnn.py`, nb §5 |
| P1 | error analysis + atlas | `segments.py`, `geo_eval.py`, nb §6 |
| P1 | pseudo-discovery | `folds.make_discovery_splits`, `discovery.py`, nb §7 |
| P2 | complementarity + ensembles | `ensemble.py`, nb §8 |
| P2 | distance-aware CNN target (`CNNSpec(target="kernel")`) | `cnn.py` (implemented, not scheduled in nb) |
| — | submissions (3 candidates × 2 known-fault policies) | `submission.py`, nb §9 |
| — | report + decisions | `report.py`, nb §10 |

## Leakage controls
- Labels in val + buffer never influence training: not sampled, not used for negative distances, CNN patches forbidden there.
- Early stopping / checkpoint choice on `inner_val` (a training cluster), never on the scored fold.
- Output transform and ensemble weights cross-fitted across folds.
- Known, accepted effects: label-free filters wider than the buffer see val-area *inputs*; Phase-1 transform/scale choices used all labels;
  robust standardisation statistics are computed on the full footprint (input-only).

## Expected cost (estimates, unverified)
- Feature cache: ~70 features × 49 MB ≈ 3.5 GB disk; log-Gabor/tilt are the slow ones (FFT on ~4.7k × 4.1k).
- LightGBM per fold: 600 k samples, ≤ 3000 rounds with early stopping; full-fold prediction 128-row chunks.
- Output-transform grid: 100 transforms × 3 folds of local-max scoring per run (a few minutes).
- CNN: 30 epochs × 200 steps × batch 16 at 256 px per fold; Colab T4-class GPU recommended; CPU impractical.

## Decision rules (implemented in `report.write_findings`)
- Gain = all folds positive and mean ΔDTI > 0.005; loss = mirror; else inconclusive.
- Primary baseline = best cross-fitted pooled DTI; ties within the leader's fold std → highest worst-fold DTI.
- Next phase: PU if pseudo-discovery recall@300m < 0.7 × geographic recall@300m; else architecture if CNN − GBM > 0.02;
  else ensembling if ensemble − primary > 0.01; else targets / post-processing.
