# Architecture

## Layering

```
notebooks/colab/0*.ipynb        thin orchestration: bootstrap, Experiment(...).run_all(), display
        │
gems_bench.experiment           Experiment (stages, resume guard, Drive push) → runners/<kind>.py
        │
science modules                 losses · models · training · features.profiles · dem · discovery · evaluation
        │
Phase 1-4 code (unchanged)      phase4_training / phase4_eval / p3.* / gbm / cnn / metric / folds / features
        │
infrastructure                  paths · artifacts (atomic, store, sync, registry) · runtime.colab · _version
```

Phase-5 code **reuses** the Phase-3/4 functions rather than re-implementing them: scoring (`p3.curves`, `p3.budget`,
`p3.discovery3`, `phase4_eval`), LightGBM (`p3.lgbm3`), batch builder / tiled inference / context pooling
(`phase4_training`), U-Net blocks (`geological_encoders`). This keeps every Phase-5 number comparable with the saved
Phase-3/4 tables under the same protocol.

## Contexts and the model zoo

`load_context5()` = `load_context4()` (data, folds, discovery splits, segments, Phase-2 + Phase-4 feature stores) +
`Phase5FeatureStore` (profile `p5:*`, DEM `dem:*`) + Phase-5 zoo. `ctx.oof(name)` / `ctx.discovery_pred(name, d)`
resolve, in order: Phase-5 experiment manifests (`maps`), Phase-4 / Phase-3 folders through the alias table
(`models.zoo.register_aliases`), Phase-2 runs. `ctx.paths` remains the Phase-4 layout so that completing a Phase-4
protocol (A1) writes exactly where Phase-4 results live.

## Experiment kinds

| kind | runner | experiments |
|---|---|---|
| `zoo` | `runners/zoo.py` | A1 (complete gated-encoder discovery), A4 scoreboard |
| `parity` | `runners/audit.py` | A2 metric parity, convention audit, submission audit |
| `splits` | `runners/audit.py` | A3 validation designs |
| `cnn5` | `runners/cnn5.py` | B0–B3, C1–C3, D2, E1–E5, F4 (CNN) |
| `lgbm5` | `runners/lgbm5.py` | D1, F4 (LightGBM) |
| `ensemble5` | `runners/ensemble5.py` | G |
| `dem_pilot` | `runners/dem.py` | F1–F3 |
| `submission` | `runners/submission.py` | 5D |

## Training (cnn5)

`P5Spec` extends the Phase-4 `P4Spec`; defaults reproduce Phase 4. `training.cv.run_cv5` / `run_discovery5` are the
Phase-4 protocol with resumable units; `training.trainer.train5` is `train4` plus PU / geometry losses, mean teacher,
full-state checkpoints, runtime budget, OOM back-off and collapse guard. Each unit uses its own seeded RNG
(`seed + 1000·fold`, `seed + split`) so units resume independently. The consequence is that P5 runs are not
bit-identical to Phase-4 runs, which is why **B0 / E1 re-run the gated encoder** as the control.

## Persistence

* every write is atomic (`artifacts.atomic`); rasters via temporary `.tif` + `os.replace`;
* `artifacts.sync.DriveSync`: subtree mirror local ↔ Drive with a ledger, `.partial` + verification, never deletes on Drive;
* checkpoints: rotated resume files + a final `<unit>_model.pt`; resume files removed only after the final model is on Drive.

## Deferred: `src/` layout

The target layout puts the package under `src/`. It is deferred deliberately: the executed Phase 1–4 notebooks
import `gems_bench` / `gems_eda` via `sys.path.insert(ROOT)`, and a live kernel would break mid-run. Migration when
convenient: `git mv gems_bench gems_eda src/`, set `[tool.setuptools.packages.find] where = ["src"]`, and use
`pip install -e .` in the local environment (the legacy notebooks then import the installed package). `paths.code_root()`
already looks for `configs/` relative to the checkout, so nothing else changes.

Three modules became packages so the target names exist (`data`, `features`, `discovery`): their contents are the
original modules unchanged (relative imports deepened by one level), and every old import path still works.
