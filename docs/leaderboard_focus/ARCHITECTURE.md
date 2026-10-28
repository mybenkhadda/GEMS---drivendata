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

## One notebook, one runner

```mermaid
flowchart TD
  C[Stage 0: CONFIG + strict clone + Runner] --> S1[1 artifacts: inventory + selective staging]
  S1 -->|COMPLETE| ENG[attach P6 engine once: mode recovery, root = M1 root]
  ENG --> S2[2 selection: load_grids sources=boost_a required=boost_a, p6g rule, recipe]
  S2 -->|COMPLETE| S3[3 validation: recompute cells, reproduce records by protocol, dry build]
  S3 -->|COMPLETE or WARNING| S4a[4 parity: champion export parity, historical diagnosis]
  S4a --> S4b[4 promotion gates]
  S2 --> S4b
  S3 --> S4b
  S4b -->|READY_FOR_EXPORT| S5[5 export: final_submission.export_one + M1 checks]
  S4b -->|BLOCKED| NOFILE[no GeoTIFF written]
  S5 -->|READY_FOR_UPLOAD| UP[manual DrivenData upload]
  UP --> S6[6 leaderboard log]
  S6 --> S7[7 final report]
  NOFILE --> S7
```

Each stage function records `manifests/<stage>.json` (+ a copy in `manifests/history/`). A stage whose upstream is not
usable records INCOMPLETE and does no work; an exception is recorded FAILED with its traceback, the failure report is
written, and the exception is re-raised.

## Promotion gates

```mermaid
flowchart TD
  A{SELECTION_COMPLETE: selection COMPLETE, one recipe id} -- no --> X[BLOCKED]
  A -- yes --> B{DISCOVERY_ELIGIBLE: p6g eligible, worst >= 0.153414}
  B -- no --> X
  B -- yes --> G{GEOGRAPHIC_GATE_PASS: geo >= 0.1758}
  G -- no, mandatory --> X
  G -- no, advisory + note --> W[WARNING]
  G -- yes --> V{VALIDATION_PASS and PARITY_PASS, fresh for this selection}
  W --> V
  V -- no --> X
  V -- pending / stale --> P[INCOMPLETE]
  V -- yes --> R[READY_FOR_EXPORT]
```

## Artifact dependencies

```mermaid
flowchart LR
  inv[artifact_inventory.json + staging_manifest.json] --> sel[selection/*]
  grid[boost/benchmark/grid.csv] --> sel
  maps[stack/maps geo_k.npy] --> sel
  oof[B0/B1 oof.tif, disc0-2.tif] --> sel
  sel --> val[validation/*]
  sgrid[stack/grid.csv] --> val
  val --> par[parity/*]
  rec[recovery/export_parity/*] --> par
  full[stack/maps/full_score.tif historical] --> par
  sel --> prom[selection/promotion_gates.*]
  val --> prom
  par --> prom
  prom --> sub[submissions/*.tif]
  tpl[example_submission.tif] --> sub
  sub --> lb[leaderboard/*]
```

## Code map (thin adapters; no P6 function is duplicated)

| module | role | reuses |
|---|---|---|
| `p6/full_benchmark.py` | **shared** grid loader: `load_grids(fg, sources, required)`, `read_csv_checked`, `read_grid_csv`, `GridUnavailable`, `resolve_specs(fg, prov, candidates)` | – (P6 callers keep `load_grids(fg)`) |
| `leaderboard_focus/config.py` | config, status vocabulary, notebook CONFIG checks | – |
| `artifacts.py` | inventory, required-file checks, staging, stage 1 | `artifacts.sync.DriveSync`, `artifacts.atomic` |
| `session.py` | attach, targeted staging, stage manifests, implementation fingerprints, cache | `Pipeline`, `P6FastTrack`, `p6.full_gpu` (mode recovery) |
| `workflow.py` | `Runner`: modes, stage order, engine-once, failure report | every stage module |
| `champion.py` | candidate set, artifact gates, selection, recipe | `full_benchmark.{load_grids, resolve_specs, protocol_check, benchmark_tables, meta_estimate}` |
| `validation.py` | recomputation, protocol-labelled reproduction, dry build | `Boost.{fusion, stack_rank, cnn, _apply, _eval_q}`, `gems3.metric_opt.nested_select`, `final_submission.{inspect_spec, build_map}` |
| `parity.py` | export parity, historical blocker diagnosis | `export_parity.{agreement, classify_parity, read_band}`, `final_submission.zone_partition` |
| `gates.py` | promotion gates | – |
| `submission.py` | export precheck + gate | `final_submission.export_one` |
| `reporting.py` | reports, figures, leaderboard log, executive summary, failure catalogue | – |
