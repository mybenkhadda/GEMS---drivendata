# M1 architecture

## B0 / B1 → W2 → candidate maps

```mermaid
flowchart LR
  F[geophysical feature store] --> B0[B0 CNN P5_B0_gated]
  F --> B1[B1 CNN P5_B1_bgw]
  B0 -->|OOF geo folds / discovery maps| R0[rank inside unit]
  B1 -->|OOF geo folds / discovery maps| R1[rank inside unit]
  R0 --> FU[B0+B1 rank fusion, per-zone w]
  R1 --> FU
  R0 --> W2[W2 LightGBM, cross-fitted per zone]
  R1 --> W2
  PF[potential-field gradients] --> W2
  KN[permitted known-fault geometry] --> W2
  W2 -->|held-out maps stack/maps/*.npy| S[score = 1-l fusion + l rank W2]
  FU --> S
  S --> REP[crest / skeleton 1 px at area frac]
  REP --> GRID[boost-A grid: official components per cell and unit]
```

## Champion recovery workflow

```mermaid
flowchart TD
  N0[00 inventory + staging] -->|COMPLETE| N1[01 champion: p6g rule on frozen grid, W2 family + fusion fallback]
  N1 -->|recipe_id| N2a[02 validation: recompute cells, reproduce records, dry zone-stitched build]
  N2a --> N2b[02 parity: historical blocker diagnosis + champion parity]
  N2b -->|COMPLETE + validated array sha| N3[03 export: final_submission.export_one + M1 gates]
  N3 -->|READY_FOR_UPLOAD| UP[manual DrivenData upload]
  UP --> N4[04 leaderboard log]
```

## Artifact dependencies

```mermaid
flowchart LR
  inv[artifact_inventory.json] --> sel[selection/*]
  grid[boost/benchmark/grid.csv] --> sel
  maps[stack/maps geo_k.npy] --> sel
  oof[B0/B1 oof.tif] --> sel
  sel --> val[validation/*]
  sgrid[stack/grid.csv] --> val
  val --> par[parity/*]
  rec[recovery/export_parity/train_inference_parity.csv] --> par
  full[stack/maps/full_score.tif historical] --> par
  par --> sub[submissions/*.tif]
  tpl[example_submission.tif] --> sub
  sub --> lb[leaderboard/*]
```

## Gates

```mermaid
flowchart TD
  A{inventory: required artifacts OK?} -- no --> X1[BLOCKED]
  A -- yes --> B{champion eligible + one deploy cell?}
  B -- no --> X2[BLOCKED]
  B -- yes --> C{recomputed == grid, recorded reproduced, template OK?}
  C -- no --> X3[BLOCKED]
  C -- yes --> D{zone parity PASS, budget, coverage, recipe consistent?}
  D -- no --> X4[BLOCKED]
  D -- yes --> E{written band == validated array, official checks, audit, size?}
  E -- no --> X5[BLOCKED]
  E -- yes --> R[READY_FOR_UPLOAD]
```

## Code map (thin adapters; no P6 function is duplicated)

| module | role | reuses |
|---|---|---|
| `leaderboard_focus/config.py` | config, statuses | – |
| `artifacts.py` | inventory, staging, notebook-00 stage | `artifacts.sync.DriveSync`, `artifacts.atomic` |
| `session.py` | one-call attach, stage manifests, fingerprint reload | `Pipeline`, `P6FastTrack`, `p6.full_gpu` (mode recovery) |
| `champion.py` | candidate set, artifact gates, selection, recipe | `p6.full_benchmark.{load_grids, resolve_specs, protocol_check, benchmark_tables, meta_estimate}`, `export_parity.unit_overlap` |
| `validation.py` | recomputation, reproduction, dry build | `Boost.{fusion, stack_rank, cnn, _apply, _eval_q}`, `gems3.metric_opt.nested_select`, `final_submission.{inspect_spec, build_map}` |
| `parity.py` | blocker diagnosis | `export_parity.{agreement, classify_parity, read_band}`, `final_submission.zone_partition` |
| `submission.py` | export gate | `final_submission.export_one` (recipe, writer, `validate`, `readiness`) |
| `reporting.py` | reports, figures, leaderboard log | – |
