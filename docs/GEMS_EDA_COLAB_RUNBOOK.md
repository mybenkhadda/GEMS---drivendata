# GEMS exhaustive geological EDA — Colab runbook (gde.1-r3)

## 0. Before Colab (your machine)

```
git add -A
git commit -m "gde.1-r3: EDA production readiness"
git push
git rev-parse HEAD
```

Optional: `python scripts/eda/patch_gems_eda_notebook.py --check` should report only OK lines (the committed notebook
already has the r3 cells; the patcher only repairs a diverged copy and never touches outputs).

## 1. Colab, in order

1. Open the notebook from GitHub (Colab badge). Runtime: standard CPU high-RAM if available (no GPU needed).
2. **Config cell**: set `'ref'` to the SHA from step 0 and `EXPECT['commit']` to its first 10 characters. Keep
   `mode='FULL_EDA'`, `reuse_cached=True`, `train_models=False`.
3. **Bootstrap cell** (mounts Drive, clones / checks out the SHA, installs, builds the factory and `E`). If it says
   "gems packages already imported": Runtime -> Restart session, run from the config cell.
4. **Cell 2b**: `verify_checkout` must print `code_rev gde.1-r3`, module under `/content/GEMS`.
5. **Cell 3-4 (preflight)**: read `P.summary()`. Expected on the first r3 run: 00 READY; e01, e02
   READY_FROM_CACHE (if their Drive manifests match); e03 NEEDS_RECOMPUTE (geometry schema 3); e04-e12
   NEEDS_RECOMPUTE; 01 / 03 READY_FROM_CACHE or NEEDS_RECOMPUTE; e13 OPTIONAL_UNAVAILABLE; e14 / e15 READY.
   `P.problems` must be empty. `P.geometry` shows the precision defects of the OLD cached table (19 strikes = 180,
   1 linearity > 1) and `geometry_rev_of_cached_table = absent` -> recompute expected.
6. **Cell 5**: `P.require_ready(); result = E.run()` (several hours in FULL_EDA; progress lines per stage).
7. **Cell 6**: health, current errors, integrity, decision report.

## 2. One-cell entry point (alternative to the bootstrap + cells 2b-6, after the config cell)

```python
import os, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')
CODE = '/content/GEMS'
if not os.path.exists(CODE + '/.git'):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/' + CONFIG['repo'] + '.git', CODE], check=True)
subprocess.run(['git', '-C', CODE, 'fetch', '-q', 'origin'], check=True)
subprocess.run(['git', '-C', CODE, 'checkout', '-q', '--detach', CONFIG['ref']], check=True)
HEAD = subprocess.run(['git', '-C', CODE, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', CODE + '[colab]', 'pytest', 'tabulate'], check=True)
sys.path.insert(0, CODE)
from gems_eda.discovery.run import prepare_eda
E = prepare_eda(CONFIG, HEAD, CODE, EXPECT)          # verifies the checkout, builds factory + E
P = E.preflight(); display(P.summary()); P.require_ready()
result = E.run()
display(E.health())
```

(`CONFIG['ref']` must be a SHA or `origin/main` for the detached checkout.)

## 3. Recovery

* Interrupted / runtime reset: config -> bootstrap -> 2b -> `P = E.preflight()` -> `E.resume()`. Valid stages are
  REUSED_VALID_CACHE; only invalid / missing ones run.
* A stage FAILED: the run summary prints `>>> stage FAILED [CATEGORY]`, the error, file:line, affected output,
  downstream stages and the recovery step. Full traceback: `manifests/history/stage_events.jsonl` (append-only) and
  `manifests/stages/<stage>.json`. After a code fix: push, pin the new SHA, restart, `E.resume()`.
* Historical errors: `DG.current_errors(E)` lists only errors whose stage is still not valid.

## 4. What is checked where

| check | where |
|---|---|
| code rev / stale install | cell 2b, `prepare_eda` |
| raster CRS, transform, shape, dtype, labels, footprint | preflight `inputs()` (metadata + bounded reads) |
| cached outputs (columns, rules, NPZ keys / grid shape) | preflight `cache()` and every stage reuse (`check_outputs`) |
| frozen engine provenance / held-out maps | factory at 01_attach (preflight only reports presence) |
| DEM | optional; preflight reports cached files, never downloads |
