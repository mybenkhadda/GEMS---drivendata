# GEMS — project state (read this first)

_Last updated 2026-10-21. Code-authoring-only workflow: Claude writes code, the user executes._

## Status
| phase | artifact | state |
|---|---|---|
| 1 EDA | `notebooks/01_geological_eda.ipynb`, `gems_eda/`, `outputs/eda/` | executed |
| 2 benchmark | `notebooks/02_fault_benchmark.ipynb`, `outputs/benchmark/` | executed |
| 3 calibration + discovery | `notebooks/03_...ipynb`, `gems_bench/p3/`, `outputs/phase3/PHASE3_FINDINGS.md` | executed |
| 4 RS features + encoders | `notebooks/04_...ipynb`, `configs/phase4/`, `outputs/phase4/` | **executed** (RS LightGBM + drop-group, DL00/DL02 + discovery, P4_B concat / gated / ctrl CV). Not done: P4_B_gated discovery, P4_D_aux, P4_C_context incomplete (no area_crossfit), SSL |
| 5 platform + science | package `gems-fault-discovery`, `configs/{data,validation,features,models,experiments,colab}`, `notebooks/colab/00-07` | **authored 2026-10-21, NOT run** (no code, test or notebook executed) |

## Best numbers so far (local, saved CSVs — not leaderboard; one protocol: geo_k3 folds, discovery_d3 splits)
| model | geo DTI (area) | disc excl | disc retained_fp | <2 km recall |
|---|---:|---:|---:|---:|
| P4_B_gated (gated dual encoder) | 0.158 reported (held-out folds 0.135 / 0.177 / 0.167) | **missing** | **missing** | – |
| DL00_CNN_B_dti | 0.1496 | 0.1675 | 0.0807 | 0.190 |
| RS05_G3_noSS_RS | 0.1429 | 0.1186 | 0.0945 | 0.163 |
| G1 | 0.1327 | 0.1001 | 0.0849 | 0.138 |
| CNN_A_dti | 0.1302 | **0.1788** | 0.0918 | 0.159 |
| DL02_CNN_TMGB_dti | 0.1215 | 0.1709 | **0.0998** | 0.149 |
Sources: `outputs/phase4/cnn_controls_scorecard.csv`, `rs_lgbm_scorecard.csv`, `experiments/p4_P4_B_*/area_crossfit.csv`.
Phase-3 nested ensemble 0.1504 geo, failed the discovery gate. Leaderboard reference ~0.4x is NOT comparable.

## Confirmed findings (Phases 1-4)
1. DTI is maximised by a binary map → compare rankings at matched / cross-fitted area budgets.
2. Terrain dominates; RS terrain features help (RS02 vs RS01 +0.011 geo); seismicity/strain hurt tabular models.
3. DTI-surrogate loss lifts CNNs (CNN-B 0.111 → 0.150). Gated dual encoder > concat > parameter-matched single (0.158 / ~0.146 / ~0.109).
4. Geographic and pseudo-discovery rankings disagree; CNNs lead `excl`, reduced-modality CNN leads `retained_fp`.
5. Short (<2 km), curved and weak faults remain the main miss.

## Phase 5 — where things are
- Package (flat layout, `pip install -e .`): `gems_bench/{paths,_version,cli,phase5_context}.py`, subpackages
  `artifacts experiment geology features(profiles, phase5_store) losses models training evaluation discovery(ensemble5, atlas) dem inference calibration visualization runtime`.
  `data.py` / `features.py` / `discovery.py` became packages (same exports). Raw files now also resolve under `data/raw/...` and `$GEMS_ROOT`.
- Experiments: `configs/experiments/*.yaml` (a1-a3, b0-b3, c1-c3, d1-d2, e1-e5, f1/f4, g1, z_submission); kinds and stages in `docs/EXPERIMENT_GUIDE.md`.
- Tests (synthetic, not run): `tests/test_phase5_infrastructure.py`, `tests/test_phase5_science.py`.
- Git: nested repository in this folder (the home-directory repo is untouched). See `docs/NEXT_SESSION_HANDOFF.md` for remote / push status.

## Next steps (user)
1. Local: `pip install -e ".[dev]"`, `pytest` (fix anything red before Colab).
2. Push to GitHub; upload data + `outputs/` to Drive (`docs/GOOGLE_DRIVE_LAYOUT.md`).
3. Colab notebook 00 → 01 (profile features) → 02 (A2, **A1: gated-encoder discovery**, A3).
4. Notebook 03: B0 first, then B1 / B2 sweep; C1-C3.
5. Then D, E, F, G as in `docs/EXPERIMENT_GUIDE.md`; promotion only through saved decisions.

## Known risks
- Nothing in Phase 5 has executed: expect first-run errors (shapes, dtypes, missing cache names); fix in place.
- Phase-5 units use per-unit RNG seeds (resumability), so B0/E1 must be compared with P4_B_gated to measure trainer variance.
- DEM tile corner convention (`upper_left`) is inferred from names; F1 verifies it against real headers before use.
- The official scorer is not available locally: leaderboard parity remains UNVERIFIED until supplied.
