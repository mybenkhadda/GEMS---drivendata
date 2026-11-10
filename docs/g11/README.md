# G11 — Multiresolution geological fault discovery

**Run it:** `notebooks/colab/GEMS_G11_MASTER_COLAB_EXECUTION.ipynb` (see [COLAB_MASTER.md](COLAB_MASTER.md))
· **Package:** `gems_bench/g11/` · **Config:** `configs/g11_multiscale.yaml`
· **Tests:** `gems_bench/g11/tests/` (102: 97 synthetic tests pass; 5 real-data tests run only with `G11_REAL_DATA_ROOT`)

G11 combines the 100 m GeoDAWN geophysics, native-resolution (about 1 m) USGS DEM windows, mapped faults and ML models:
- XGBoost;
- SegFormer;
- gated geophysics × DEM fusion;
- pixel fusion;
- a segment ranker.

It generates fault-candidate segments, links, deduplicates and ranks them, and selects a subset under a length budget. The output is a submission in the official raster schema.

**Status (2026-11-10).** The code is written and tested on synthetic fixtures. No real-data stage has run yet, so no metric, DTI or submission exists. The DTI ≥ 0.40 target is a goal, not a measurement.

| doc | content |
|---|---|
| [COLAB_MASTER.md](COLAB_MASTER.md) | Launch, editable values, resource estimates, longest stages, limitations |
| [EXECUTION_ORDER.md](EXECUTION_ORDER.md) | Notebook sections, stage calls, gates, failure policy |
| [DATA_PATHS.md](DATA_PATHS.md) | Real Drive layout (verified from metadata) and the config keys |
| [ARTIFACT_REGISTRY.md](ARTIFACT_REGISTRY.md) | Every output path |
| [AUDIT_STATIC.md](AUDIT_STATIC.md) | Repository audit; where the brief and the repository disagree |
| [API.md](API.md) | Public functions |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Hashes, seeds, manifests, leakage protocol |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Failure messages and fixes |
| [INSTALLATION.md](INSTALLATION.md) | Colab and local setup |
| [TEST_RECORD.md](TEST_RECORD.md) | What was executed and the results |
| [SESSION_HANDOFF.md](SESSION_HANDOFF.md) | State, decisions, next steps |
