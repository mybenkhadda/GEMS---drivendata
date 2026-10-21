# GEMS 2.0: geological fault-discovery research platform

`gems-fault-discovery` (import `gems_bench`) is a research library for the DrivenData **GEMS** challenge: discovering
previously unmapped faults in Western Nevada from the 19-band GeoDAWN raster (100 m) and, optionally, USGS 1 m DEMs.
It contains Phases 1–4 (EDA, benchmark, calibration and discovery, remote-sensing features and geology-aware encoders)
and the Phase-5 platform: installable package, Colab GPU + Google Drive runtime, resumable experiments, positive-unlabeled
and geometry-aware learning, GeoStructureNet, 1 m DEM pilot, discovery-aware ensembling and a formal model registry.

> Local validation DTI and the public leaderboard are **different quantities**: local scores use spatial folds and
> pseudo-discovery splits of the released fault map. They are never substituted for each other (see
> `docs/VALIDATION_PROTOCOL.md`).

## 1. Install

```bash
# local (editable) - from the project root
pip install -e ".[dev]"                    # + torch matching your CUDA: https://pytorch.org/get-started
# Colab / any machine, from GitHub (pin a commit for reproducibility)
pip install "git+https://github.com/OWNER/REPOSITORY.git@<commit>"
pip install -e ".[colab]"                  # inside a clone (Colab already provides torch)
```

Python ≥ 3.10. Core dependencies: numpy, scipy, pandas, pyarrow, rasterio, scikit-image, scikit-learn, lightgbm, pyyaml.
The `gems` command is installed with the package.

## 2. Configure data

Data never goes into Git (`data/README.md`). Two layouts are recognised automatically:

* **local / original**: the three GeoTIFFs at the project root next to `outputs/` (Phases 1–4 run unchanged);
* **workspace**: `data/raw/geodawn/…`, `data/raw/faults/…` under `$GEMS_ROOT` (Colab: `/content/gems_work`).

`$GEMS_ROOT` (data workspace), `$GEMS_DRIVE_ROOT` (Drive mirror) and `$GEMS_CODE_ROOT` (checkout with `configs/`) override
discovery. See `docs/GOOGLE_DRIVE_LAYOUT.md` for the one-time upload of the Phase 1–4 artifacts.

## 3. Colab + Drive

Open `notebooks/colab/00_setup_and_drive.ipynb`, set `REPO` / `COMMIT`, run all. It mounts Drive, clones and installs
the package, checks the GPU, creates `MyDrive/GEMS_2/…` folders, stages core data to `/content`, and lists existing
experiments and the registry. Details: `docs/COLAB_QUICKSTART.md`.

## 4. Run a baseline evaluation

```python
from gems_bench.experiment import Experiment
exp = Experiment.from_yaml("configs/experiments/a1_zoo_discovery.yaml")   # A1 + A4
exp.run_all()                    # prepare -> train (missing discovery runs) -> evaluate -> export
exp.rd.load_table("scoreboard")
```

```bash
gems run --config configs/experiments/a2_parity.yaml      # metric parity, CPU
```

## 5. Train a challenger

```python
b0 = Experiment.from_yaml("configs/experiments/b0_reference.yaml"); b0.run_all()    # control
b2 = Experiment.from_yaml("configs/experiments/b2_nnpu_p010.yaml"); b2.run_all()    # PU challenger
b2.rd.load_table("promotion_decisions")
```

Each experiment is one YAML (`extends` shared bases) → one folder `experiments/<name>__<hash>/` with `config.yaml`,
`manifest.json`, `checkpoints/`, `metrics/`, `predictions/`, `figures/`, `logs/`, `reports/`.

## 6. Resume after a disconnect

Re-run the same cell. Completed stages and units (CV folds, discovery splits, DEM tiles) are skipped; the
interrupted unit continues from its last checkpoint (model, optimizer, scheduler, scaler, RNG states). The config
hash and code commit are checked first.

```python
exp = Experiment.resume("b2_nnpu_p010__1a2b3c4d")     # or Experiment.from_yaml(same yaml)
exp.status(); exp.units()
```

```bash
gems status --experiment b2_nnpu_p010__1a2b3c4d
```

## 7. Find figures and results

`MyDrive/GEMS_2/experiments/<id>/figures/*.png|svg` and `metrics/*.csv` (scorecard, scoreboard, geographic_dti,
pseudo_discovery_dti, subgroup_recall, predicted_area_curve, model_complementarity, runtime / GPU profiles),
`reports/experiment_summary.md`. Every figure is regenerable from those tables (`gems_bench.visualization.figures`).

## 8. Load results

```python
import pandas as pd
board = pd.read_csv("/content/drive/MyDrive/GEMS_2/experiments/<id>/metrics/scoreboard.csv")
from gems_bench.phase5_context import load_context5
ctx = load_context5(); p = ctx.oof("P5_B2_nnpu_p010"); d0 = ctx.discovery_pred("P5_B2_nnpu_p010", 0)
```

## 9. Generate a submission

```bash
gems registry promote --model P5_B2_nnpu_p010 --experiment b2_nnpu_p010__1a2b3c4d   # needs a passing decision
gems run --config configs/experiments/z_submission.yaml
```

Writes `submissions/<model>__binary_area__<known_policy>.tif` (single band, float32, template grid, [0, 1]) plus a
provenance JSON. Record uploads and public scores in `reports/leaderboard_log.csv`.

## Documentation

`docs/ARCHITECTURE.md` · `docs/COLAB_QUICKSTART.md` · `docs/GOOGLE_DRIVE_LAYOUT.md` · `docs/DATA_PROVENANCE.md` ·
`docs/EXPERIMENT_GUIDE.md` · `docs/VALIDATION_PROTOCOL.md` · `docs/MODEL_REGISTRY.md` · `docs/PROJECT_STATE.md` ·
`docs/NEXT_SESSION_HANDOFF.md`. Phase 1–4 plans and findings: `docs/PHASE*_PLAN.md`, `outputs/*/…FINDINGS.md`.

License: MIT (code only; data licenses in `docs/DATA_PROVENANCE.md`).
