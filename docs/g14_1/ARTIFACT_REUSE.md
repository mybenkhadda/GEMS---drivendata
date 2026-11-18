# Artifact reuse strategy (Phase B)

`gems_bench/g14_1/artifacts.py`. Everything is found automatically, with local copies first and Drive copies used in
place. Nothing is copied by hand, and nothing is regenerated just because it is missing locally.

| artifact | path pattern (relative to root) | roots (local → Drive) | used by |
|---|---|---|---|
| G14 full prediction | `maps/full.npy`, `maps/full_mc_std.npy` | `<local>/experiments/<exp>/`, `<drive>/outputs/G14_GEOFAULTFORMER/experiments/<exp>/` | R01 R02 R03 R05 R17 |
| G14 held-out maps | `maps/geo_*.npy`, `maps/disc_official_*.npy` | same | R02–R05 |
| G14 checkpoints | `checkpoints/*_best.pt` | same | R06 |
| training manifests (+ normaliser, channels) | `manifests/*_train.json` | same | R00 R06 |
| scorecards / choice tables | `evaluation/*__scorecard.json`, `*__geo_choice.csv` | same | R03 R17 |
| exported submissions | `submissions/*.tif` (+ `.json`, `_g14.json`) | same | R01 |
| channel stack | `stack/manifest.json`, `stack/stack_l*.npy` | `<local>/shared/`, `<drive>/.../shared/` | R06 R08+ |
| G13 splits / regimes / scoreboard / features | `splits/params.json`, `regimes/regimes.npy`, `tables/scoreboard.csv`, `features/registry.csv` | `/content/gems_g13_work`, `<drive>/outputs/G13_GEODISCOVERY` | R03 R05 R07 |
| G13 maps | `predictions/<model>/{oof,disc<d>,full}.npy` | same | R05 |
| P6 W2 | `stack/maps/{task}_{zone}.npy` (Phase-5 unit crops), `stack/maps/full_score.tif` | `<drive>/outputs/p6_fast_track` | R05 |
| P6 / boost submissions | `**/submissions/*.tif` | `<drive>/outputs/p6_fast_track`, `p6_boost` | R01 R05 |
| P7 maps | `**/{oof,disc,full_region}_*.tif` | `<drive>/outputs/p7_geofaultformer` | R05 (label-free if full grid) |
| G07 maps | `**/*{oof,disc,full}*` | `<drive>/outputs/g07_multiscale_fault_discovery` | R05 (label-free if full grid) |
| official template / known faults / features | content-verified by `g13.contracts.discover_inputs` | `<drive>` | all |

## Rules

* **Inventory:** `discover()` records, for every file:
  * location, size, mtime;
  * header shape and dtype (npy header or GeoTIFF profile, payload not read);
  * grid check against the template;
  * duplicate local / Drive status;
  * the stages that need it.
* **Best copy:** `best_copy()` takes the local copy when it exists and is not smaller than the Drive copy; otherwise
  the Drive path is read in place. `restore()` copies to local only when a consumer needs a local path (for example,
  checkpoints restored into the G14 experiment directory for R06), and only if the copy is missing or differs in
  size.
* **Experiment hash:** the directory suffix (`da3f44d05d84`) must equal every manifest's `cfg_hash`. The G14 config
  is rebuilt with `leakage.g14_config_for`, which raises with instructions if the semantic overrides differ.
* **Compatibility:** `compatible()` refuses different grids, different protocol families (g13 vs phase5 units) and
  full maps mixed with unit maps. G13 maps are scored only on the g13 backend; P6 / P7 unit crops only on Phase-5
  units; full maps are compared label-free.
* **Resumability:** `StageLedger` (`manifests/stages.json`) stores, per stage:
  * status;
  * input digest (config hash, source experiment, inputs);
  * outputs;
  * seconds;
  * code revision.

  A completed stage with the same digest and all its outputs present is reused (`resume=True`, `force_rebuild=False`).
* **Preflight:** `preflight()` gives READY / PARTIAL / BLOCKED per stage, with the missing required and optional
  artifacts.
* **Channel stack:** `stack_l*.npy` was not synced by G14. If the local copy is absent, R06 rebuilds it with the G14
  code (label-free, deterministic), and the checkpoint loader refuses a different schema hash. If the rebuild differs
  (for example, the G13 feature store is missing), set `data.g13_features_dir` as in the original run.
