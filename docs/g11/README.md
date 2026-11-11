# G11 — Multiresolution geological fault discovery (recovery-first)

**Run it:** `notebooks/colab/GEMS_G11_MASTER_COLAB_EXECUTION.ipynb`. See [RECOVERY_FIRST.md](RECOVERY_FIRST.md) and [COLAB_MASTER.md](COLAB_MASTER.md).

- **Package:** `gems_bench/g11/`.
- **Config:** `configs/g11_multiscale.yaml`.
- **Tests:** `gems_bench/g11/tests/`. The synthetic tests run anywhere; the real-data tests run only when `G11_REAL_DATA_ROOT` is set.

Expensive geophysical and DEM features are processed once and verified. They are saved on Drive as an immutable, versioned dataset and reused by every split, loss, model and hyperparameter experiment. The models are:

- E0: XGBoost on geophysics;
- E1: E0 plus the DEM 100 m summaries;
- E2: SegFormer;
- E3: gated multiresolution fusion;
- E4: segment ranker;
- E5: pixel ensemble.

The output is the official single-band GeoTIFF on the template grid. It is never uploaded.

**Status (2026-11-11).** Code written and tested on synthetic fixtures. No real-data stage of the recovery-first pipeline has run yet. The interrupted 2026-11-10 run's artifacts were inspected through their Drive metadata only; [RECOVERY_FIRST.md §1](RECOVERY_FIRST.md) has the findings. No metric, DTI or submission exists.

| doc | content |
|---|---|
| [RECOVERY_FIRST.md](RECOVERY_FIRST.md) | What the interrupted run left on Drive, the feature identity rules, policies, persistence protocol, DEM engine, datasets, split v2 and the exact Colab instructions |
| [COLAB_MASTER.md](COLAB_MASTER.md) | Launch, editable values, resource estimates, longest stages, limitations |
| [EXECUTION_ORDER.md](EXECUTION_ORDER.md) | Sections 00–20, calls, gates, run modes |
| [ARTIFACT_REGISTRY.md](ARTIFACT_REGISTRY.md) | Every output path |
| [DATA_PATHS.md](DATA_PATHS.md) | Real Drive layout (verified from metadata) |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Failure messages and fixes |
| [TEST_RECORD.md](TEST_RECORD.md) | What was executed and the results |
| [SESSION_HANDOFF.md](SESSION_HANDOFF.md) | State, decisions, next steps |
| [AUDIT_STATIC.md](AUDIT_STATIC.md), [API.md](API.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [INSTALLATION.md](INSTALLATION.md) | Background docs from the first G11 session |
