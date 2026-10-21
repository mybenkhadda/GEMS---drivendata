# Experiment guide

## Anatomy of a config

```yaml
extends: [_base_cnn5.yaml]          # deep merge; child wins; lists replaced
name: b2_nnpu_p010                  # experiment id = <name>__<config hash 8>
model_name: P5_B2_nnpu_p010         # name of the produced maps in the zoo / registry
model:
  base: P4_B_gated                  # start from this run's exact config.json
  type: geounet | geostructurenet
  fusion / long_range               # GeoStructureNet axes
  add_channels: {terrain: [...], subsurface: [...], context: [...]}
  overrides: {pu: nnpu, pu_prior: 0.01, aux_orient: 0.1, ...}   # any P5Spec field
evaluation:
  control: P5_B0_gated              # promotion decision is always candidate vs this model
  models: [G1, P4_B_gated]          # also reported
  promotion: {...}                  # section-17 thresholds
  restrict: dem_coverage            # F4 only
validation:                          # optional: another A3 design
  discovery_dir: data/processed/splits/short_lt2km_k3
runtime: {max_stage_minutes: 0, gpu: {}}   # NOT in the hash
```

Hashing: everything except `runtime`, `drive`, `notes`, `description`. Changing a semantic field creates a new
experiment; resuming with a different commit is refused under `resume_policy="strict"` (use `"config"` to accept
code changes that do not affect results, e.g. documentation).

## Stages

| stage | cnn5 | lgbm5 | zoo | ensemble5 | dem_pilot | submission |
|---|---|---|---|---|---|---|
| prepare | spec, GPU config, build channels, param counts | feature list, build, buffer audit | completion plan | member check | inventory, pilot | resolve champion |
| train | 3 CV + 3 discovery units | CV + discovery (+ drop-family) | missing Phase-4 discovery | fusion maps | download (approved) + tiles + mosaic | full-region map |
| evaluate | scoreboard, decision, geometry, PU | scoreboard, decision, family ablation | scoreboard | scoreboard, decisions, region stability | coverage, figures | – |
| export | figures, contract | contract | baselines → registry | contract | – | GeoTIFFs + audit |

## Scientific decision rules (ask before adding a run)

What geological hypothesis is tested? Which earlier finding motivates it? Which control isolates it (same inputs,
splits, budget, capacity: compare `architecture.csv`)? Could the gain come from a larger prediction footprint
(check `geo_far_share_at_match`, `pu_confidence_mass`)? Does it recover systems others miss (`unique_withheld_systems`,
`fault_recovery_atlas`)? Is the GPU time justified by the information gained?

## The programme (in order)

1. **A** – parity, gated-encoder discovery, unified scoreboard, designs (`a1`–`a3`).
2. **B** – `b0_reference` (control) → `b1_bg_weight`, `b2_nnpu_p005/010/020` → `b3_consistency` only if B1/B2 pass.
3. **C** – `c1_geometry_heads` → `c2_thin_continuity`, `c3_cldice` (controls: C1).
4. **D** – `d1_profiles_lgbm` (with family ablation) → `d2_profiles_cnn` with the families that helped.
5. **E** – `e1_gsn_parity` (≈ B0) → one axis at a time: `e2`, `e3`, `e4` (dilated; Fourier as a separate copy), `e5`.
6. **F** – `f1_dem_pilot` → approve download → `f4_g1_dem`, `f4_dual_dem` on identical covered pixels. A
   hierarchical 1 m / 100 m encoder only if F4 shows a compelling, fold-stable gain.
7. **G** – `g1_ensemble` with every promoted candidate added to `members`.
8. **5D** – promote, `z_submission`, log leaderboard results, archive.

Combining winners (e.g. PU + geometry heads + profiles) is a new experiment whose control is the best single
component, not B0.
