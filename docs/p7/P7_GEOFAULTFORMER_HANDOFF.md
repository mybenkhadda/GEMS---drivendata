# P7 GeoFaultFormer — design, Colab procedure and handoff

Code `gems_bench/p7/` · configs `configs/p7/{geofaultformer,experiments,colab_gpu}.yaml` · notebook
`notebooks/colab/GEMS_P7_GEOFAULTFORMER.ipynb` · tests `tests/test_p7_geofaultformer.py` · outputs
`MyDrive/GEMS_2/outputs/p7_geofaultformer/`.

**Status (2026-10-28): code authored, nothing executed.** The brief forbade every local execution: no Python, no
pytest, no Ruff, no notebook cell, no data access. The tests were written but never run. No P7 number exists. Every
value below is either a recorded reference from earlier sessions (labelled as such) or a planning estimate
(labelled "unmeasured").

## 1. System architecture

```text
M9 cube (registry channels, footprint)  [+ optional label-free extra layers: G02 full-grid, B0_lineament]
  -> FoldScaler per unit (permitted pixels only) -> P7Batch (PatchSampler, visible-label targets, augmentation)
  -> GeoFaultFormer
       early fusion stem (or grouped modality stems + gate)
       residual GroupNorm encoder 1/2, 1/4, 1/8, 1/16
       shifted-window attention after stages 3 and 4 (window 8, 4 heads, 2 blocks, rel. position bias, masks)
       U-Net decoder [ConvGRU refiner at 1/4, 3 shared steps, residual, zero-init]
       heads: fault | center | orient (cos 2t, sin 2t) | distance | endpoint / junction
  -> train.py (AMP, accumulation, clipping, OneCycle, explicit inner status, CheckpointManager)
  -> infer.py (Hann-blended patches, halo tiles, resumable, rot180 TTA, MC dropout, seam-free crests)
  -> evaluate.py: rank in unit -> Boost.eval_score (official components) -> export_parity.evaluate_policy (nested)
  -> tracer.py (T4) | w2.py (T5) -> selection -> export_parity.write_validated (zone-stitched GeoTIFF)
```

Module map (`gems_bench/p7/`): `config` (merge, ladder, semantic view, presets) · `attention` · `convgru` ·
`model` · `losses` · `targets` · `augment` · `data` (registry, contract, units, batches, audits, shift) · `train` ·
`infer` · `evaluate` · `tracer` · `w2` · `figures` · `run` (`P7GeoFaultFormer`).

## 2. Existing components reused (not duplicated)

| need | reused |
|---|---|
| units, folds, splits | `m9.dataset.TrainUnit` definitions, `ctx.folds.{inner_train, inner_val, val}`, `ctx.splits.{ignore, withheld, eval_zone}` |
| inputs | `m9.cube_for` / `build_cube` / `open_cube`, `m9.features.channel_groups` (registry) |
| scaling | `m9.dataset.FoldScaler` (robust, permitted pixels of the unit) |
| targets | `m9.dataset.build_targets` (seg, kernel, orientation, endpoints, junctions, skeleton, PU weights) |
| sampling | `cnn.PatchSampler`, `m9.train.Prefetcher`, `m9.train._grad_scaler` |
| checkpoints | `training.checkpoint.CheckpointManager` (best_and_latest, spec-hash refusal), `artifacts.atomic` |
| inference blending | `m9.infer._hann`, `_ramp`, `bbox`; `gems3.representation.crest_map` |
| official scoring | `p6.boost.Boost.{eval_score, eval_variant, _eval_q, unit_sources, fusion, binary, elig}` -> `m7.protocol.score_row` / `system_hits` -> `gems3.official.FastScorer` |
| nested evaluation | `p6.export_parity.evaluate_policy` (the p6g one-interface benchmark) |
| W2 | `p6.stack.{unit_raw, unit_rows, train_index, make_model, predict_unit, rank_in, pf_full, g02_full}` |
| export | `p6.export_parity.{build_recipe, save_plan, write_validated}` (template checks + `audit_export`) |
| GPU facts | `p6.gpu_runtime.{detect, require_gpu, seed_everything, memory_report, dependency_record}` |

## 3. New files (no existing file was modified)

`gems_bench/p7/{__init__,config,attention,convgru,model,losses,targets,augment,data,train,infer,evaluate,tracer,w2,figures,run}.py`,
`configs/p7/{geofaultformer,experiments,colab_gpu}.yaml`, `notebooks/colab/GEMS_P7_GEOFAULTFORMER.ipynb`,
`tests/test_p7_geofaultformer.py`, `docs/p7/P7_GEOFAULTFORMER_HANDOFF.md`.

## 4. Configurations

* `geofaultformer.yaml`: base (scope, inputs, data, model, loss, training, inference, evaluation, promotion, tracer,
  w2). Recorded references live under `recorded`; gates use recomputed values.
* `experiments.yaml`: 8 predeclared rungs, `max_experiments: 8` enforced (`P7ConfigError`). Overrides are deep-merged.
  A loss on a disabled head, a width not divisible by the head count, or an `inputs` override is refused.
* `colab_gpu.yaml`: presets T4 / L4 / A100 / CPU split the fixed `training.effective_batch` (16) into micro-batch ×
  accumulation. They are excluded from the spec hash, so a run can resume across GPUs.

## 5. Hypotheses and rungs

| rung | kind | hypothesis | change | control |
|---|---|---|---|---|
| T0 | reference | baseline | B0, B1, fusion, W2 on the same units / protocol | — |
| T1_cnn | train | H1 control | GeoFaultFormer, attention bypassed | — |
| T1 | train | H1 context | CNN + window attention (1/8, 1/16), fault head, loss L0 | T1_cnn |
| T2 | train | H2 structure | + centerline (0.5) + axial orientation (0.2), loss L2 | T1 |
| T2c | train | H2 ablation | + centerline only, loss L1 | T1 |
| T3 | train | H3 recurrence | T2 + ConvGRU (64 hidden, 3 steps, 1/4) | T2 |
| T4 | tracer | H4 tracing | FaultTraceGRU on T2 evidence | geometric linking, matched-area base |
| T5 | stacker | H5 complementarity | W2 features + P7 features (`w2_p7.v1`) | `w2_ctrl` on the same rows |

L3 (topology / distance) is configured (`loss.weights.topology`, `distance`) but is not a predeclared rung. It
would be added only after L2 is analysed. Verdicts (`evaluate.hypothesis_verdict`) use nested disc DTI:

* SUPPORTED: Δ ≥ 0.005, worst-zone drop ≤ 0.01 and ≥ 2 of 3 zones won.
* INCONCLUSIVE: |Δ| < 0.005.
* NOT_SUPPORTED: otherwise.
* INCOMPLETE: a side is missing.

Parameter counts (unmeasured estimates; exact counts in `experiment.json` and `memory_profile.csv`):

| rung | estimated parameters | note |
|---|---|---|
| T1 | ≈ 3.4 M | |
| T1_cnn | ≈ 2.6 M | attention adds ≈ 0.85 M; not width-matched, recorded |
| T3 | ≈ 3.7 M | |

## 6. Colab execution procedure

Once, locally (by you; the authoring session ran none of this):

```bash
git switch -c p7-geofaultformer
pytest -q tests/test_p7_geofaultformer.py          # never run by the authoring session
ruff check gems_bench/p7 tests/test_p7_geofaultformer.py
git add gems_bench/p7 configs/p7 notebooks/colab/GEMS_P7_GEOFAULTFORMER.ipynb tests/test_p7_geofaultformer.py docs/p7
git commit -m "P7 GeoFaultFormer: hybrid CNN / window attention / ConvGRU ladder, tracer, W2 schema, notebook"
git push -u origin p7-geofaultformer && git rev-parse HEAD   # -> CONFIG['ref'], EXPECT['commit']
```

Colab:

1. Open the notebook. Choose a GPU runtime (L4 or A100) with High-RAM, and set `CONFIG['ref']` / `EXPECT['commit']`.
2. Run sections 0–7 with the defaults (T1 only). Read `environment/preflight.json`, `folds/leakage_overlap_audit.csv`,
   `folds/target_audit.csv`, then `comparisons/summary.csv` and `comparisons/hypotheses.csv`.
3. For H1, set `train_t1_cnn=True` and re-run sections 6–8.
4. Next rungs, one at a time (§12). T4: `train_t4=True` after T2 is evaluated. T5: `w2_integration=True`.
5. `full_inference` → `selection` → `export` only once a rung is COMPLETE on all zones. Upload by hand, only a file
   whose `selection/final_decision.json` says `PROMOTE` and whose audit status is PASS.

## 7. Feature requirements

* The M9 section-02 registry `features/selected_registry` (+ section-03 spectral, Gate-1 table) on Drive, under
  `outputs/m9_geofault_production`. The cube is built once into `cache/m9` (memory-mapped).
* The frozen W2 outputs, `outputs/p6_fast_track/stack/{candidate.json, maps/*.npy}`, for the W2 reference and T5.
* Honest B0 / B1 maps (OOF + discovery) for the references, through `unit_score`.
* Optional extra inputs are off by default. G02: `stack.g02_full` (heavy, cost unmeasured). `B0_lineament` (label-free).
  OOF CNN maps are refused as inputs: their models saw the P7 held-out fold. DEM only through the M9 dem group.

## 8. GPU and RAM estimates (unmeasured planning figures)

* Peak GPU memory (256 px patches, about 30 channels, bf16 autocast, micro-batch 16): T1 ≈ 4–10 GB, T3 ≈ 6–15 GB.
  The preflight measures every rung on synthetic tensors.
* Step time on L4 ≈ 0.15–0.3 s. Patch building on the single prefetch thread may dominate: read `samples_per_s`.
* Epoch (100 steps + inner validation) ≈ 0.5–1.5 min, so a unit takes ≈ 10–30 min.
* T1 total (6 units) ≈ 1–3 GPU-h. T3 ≈ 1.3× T1. The tracer takes minutes; T5 is CPU (W2 cost, ≤ 6 h cap).
* Host RAM: the cube is about C × H × W × 2 bytes. It goes into RAM only if under 40 % of available memory, otherwise
  it stays memory-mapped. Per-unit full-grid targets come to about 13 arrays. Use High-RAM.

## 9. Expected artifact paths (`outputs/p7_geofaultformer/`)

* `environment/`: `preflight.json`, `reference_availability.csv`, `memory_profile.csv`.
* `folds/`: `leakage_overlap_audit.csv`, `fold_provenance.csv`, `target_audit.csv`.
* `features/`: `channel_registry.csv`, `feature_contract.json`, `geographic_shift.csv`, `extra_<key>.npy`.
* `experiments/registry.csv`.
* `experiments/<id>/`:
  * `checkpoints/`: `<unit>_{model,last,best,latest}.pt`, `<unit>_selection.json`, `tracer_*.pt`;
  * `manifests/`: `experiment.json`, `units.csv`, `<unit>_scaler.json`;
  * `metrics/`: `history_<unit>.csv`, `grid.csv`, `summary.csv`, `per_zone.csv`, `prediction_mass.csv`,
    `full_region_parity.csv`;
  * `predictions/`: `units/<task>_<zone>.npz`, `oof_<head>.tif`, `disc_<head>.tif`, `full_region_<head>.tif`,
    `full_region_manifest.json`;
  * `tracer/`: `tracer_grid.csv`, `tracer_history.csv`, `decision.json`;
  * `figures/`: `*.png`, `*.svg`, `*_data.csv`, `figure_index.json`.
* `comparisons/`: `reference_grid.csv`, `reference_summary.csv`, `summary.csv`, `per_zone.csv`,
  `paired_zone_deltas.csv`, `matched_cell_deltas.csv`, `hypotheses.csv`.
* `candidate_recall/`: `recall_<id>.csv`, `h5_complementarity.csv`.
* `stacker/w2_p7.v1/<id>/`: `feature_schema.json`, `provenance.csv`, `rows.parquet`, `maps/`, `grid.csv`,
  `summary.csv`, `per_zone.csv`, `crossfit_audit.csv`, `feature_importance.csv`.
* `selection/`: `candidates.csv`, `decision.json`, `final_decision.json`.
* `submissions/`: the files `write_validated` produces.
* `manifests/status/<stage>.json`.
* `reports/p7_report.md`.

## 10. Evaluation and promotion rules

* Headline evaluation: nested disc_official DTI from `evaluate_policy` (objective `disc`, as in p6g). The output
  also reports geo DTI, worst zone, per-zone values, TP / FP / FN, system recall, coverage, fragmentation and mass.
* The following quantities are reported separately and never compared with each other:
  * inner checkpoint DTI;
  * nested outer DTI;
  * matched-cell deltas (diagnostic);
  * the public leaderboard.
* Promotion (`promotion`), all required:
  * disc ≥ recomputed W2 + 0.005;
  * **geo ≥ 0.1758**;
  * worst zone ≥ W2 worst − 0.010;
  * no leakage (both audits);
  * parity;
  * a deployable single policy;
  * a validated export.
* If promotion fails, a rung that recovers ≥ 1 system W2 misses at 3 % is `AUXILIARY` (keep it for T5).
  Otherwise it is REJECTED. W2 stays the champion.
* Reference values (recorded, not evidence):
  * W2 crossfit disc ≈ 0.2246;
  * best P6.1 W2 policy disc 0.242 / worst 0.199 / geo 0.106.
* The milestone is disc > 0.30; 0.40 is aspirational and never a gate.

## 11. Known blockers and risks

* **Geo gate conflict.** The brief's absolute geo floor of 0.1758 failed every W2 policy under the honest objective
  (geo ≈ 0.10–0.11, P6.1 run). p6g replaced it with a relative rule. Under this gate, PROMOTE may be unreachable even
  for a good discovery model. AUXILIARY covers that case. Changing `promotion.geo_min` is your decision.
* **API mismatches on real objects (first run).** The most likely spots:
  * `Boost.unit_sources` / `w2_inputs` / `fusion` when boost selection files are missing;
  * `m9.cube_for` before the M9 registry is staged;
  * channel names not matching `modality_patterns`: check `features/channel_registry.csv`, where every unmatched
    channel is `other`;
  * `spearmanr(...).statistic` (needs scipy ≥ 1.9).
* **Patch admissibility.** Patches must lie fully inside `inner_train(k)`. Fragmented folds can raise "no admissible
  patch"; then lower `data.patch_px`, e.g. to 192.
* **Throughput.** The single-thread patch builder may be CPU-bound. That has not been measured for P7.
* **Tracer.** Branch proposals are predicted but not followed. Evidence channels are per-unit ranks. Training uses
  other units' withheld systems (cross-fitted, buffered), so the trajectory count may be small on 3 splits.
* **Pseudo-discovery** under-represents near-known corrections, so disc_official is a proxy, not a leaderboard
  estimate.
* **Parity check in `full_inference`.** The zone-stitched map reuses the evaluated unit crops, so parity holds by
  construction. The Spearman check guards against assembly errors only.
* **Seed variance.** Three zones and a single seed per rung: nested differences below 0.005 are treated as noise.

## 12. Recommended next experiment (and what justifies the following ones)

First run: **T1** (and ideally **T1_cnn**) on all 3 geo folds and all 3 discovery splits, with the defaults:
256 px, conservative augmentation, 20 × 100 steps, effective batch 16.

* **T2** is justified if T1 is SUPPORTED or INCONCLUSIVE vs T1_cnn with no worst-zone loss, or if T1 is
  COMPLEMENTARY (≥ 1 new system vs W2 at 3 %).
* **T3** is justified if T2 is SUPPORTED vs T1, or if its centerline candidate (`p7_T2_center`) raises system recall
  at ≤ 1.5 % budgets without a disc loss.
* **T4** is justified if T2's missed systems are mostly fragmented / displaced rather than absent, judged by the P6.1
  candidate-recall taxonomy or by the P7 recall tables and error maps.
* **T5** is justified if any rung is COMPLEMENTARY, or if its held-out maps exist on every unit and it is within
  0.01 of W2.
* **Stop P7** if no rung is SUPPORTED or COMPLEMENTARY after T2.

## 13. Paste-ready summary for the next Claude Code session

```text
P7 GeoFaultFormer was AUTHORED 2026-10-28 (code-only; nothing executed, tests never run). Code gems_bench/p7/
(config, attention, convgru, model, losses, targets, augment, data, train, infer, evaluate, tracer, w2, figures, run),
configs/p7/{geofaultformer,experiments,colab_gpu}.yaml, notebook notebooks/colab/GEMS_P7_GEOFAULTFORMER.ipynb,
tests tests/test_p7_geofaultformer.py, doc docs/p7/P7_GEOFAULTFORMER_HANDOFF.md. No existing file modified.
Entry: P7GeoFaultFormer(P6FastTrack(m9), P7_CONFIG).{preflight, prepare_features, train('T1'), evaluate, candidate_recall,
tracer('T4'), w2_integration, full_inference, selection, export, figures, report}. Outputs outputs/p7_geofaultformer.
Scoring = Boost.eval_score + export_parity.evaluate_policy (objective disc) on the same units as B0/B1/fusion/W2.
Ladder: T1_cnn (H1 control), T1, T2, T2c, T3, T4 tracer, T5 W2+P7 (schema w2_p7.v1, matched w2_ctrl).
Code rev p7.1-r1 (checked by the notebook). Before claiming anything: read manifests/status/*.json,
comparisons/summary.csv, comparisons/hypotheses.csv on Drive. Known conflict: promotion.geo_min 0.1758 (brief) vs p6g
relative geo rule; W2's honest geo was ~0.106. Next: run T1 (+T1_cnn) in Colab; decide T2 by handoff section 12.
```
