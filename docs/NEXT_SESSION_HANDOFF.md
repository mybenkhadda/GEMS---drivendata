# Next-session handoff (2026-10-21)

## Git

* Nested repository in `Downloads/GEMS` (branch `main`). The home-directory repository
  (`C:/Users/myben`, remote `ImplementationML`) was not touched; GEMS was never tracked there.
* Commits: `c0cb2cb` (platform + Phase 5), then this handoff. **No remote configured, nothing pushed.**
* Push (create an empty GitHub repository first, private recommended; no README / license on creation):

```bash
cd ~/Downloads/GEMS
git remote add origin https://github.com/<owner>/<repository>.git
git push -u origin main
```

Then replace `OWNER/REPOSITORY` in `notebooks/colab/*.ipynb` (variable `REPO`) and `pyproject.toml`
(`[project.urls]`), commit, push. Install from GitHub: `pip install "git+https://github.com/<owner>/<repository>.git@<sha>"`.

* Not in Git by design: raw rasters, `outputs/**` except findings `.md`, feature caches, checkpoints,
  `experiments/`, `submissions/`, `model_registry/`, secrets. `notebooks/01_geological_eda.ipynb` (31 MB of
  outputs) is excluded: strip outputs (`nbstripout notebooks/01_geological_eda.ipynb`) and remove its line from
  `.gitignore` to version it.

## Nothing was executed

No Python, notebook, test or training ran in this session. Expected first-run work: `pytest` locally, then
notebook 00 on Colab. Fix first-run errors in place (keep config hashes stable where possible).

## Ready to execute (in order)

1. `pytest` (synthetic) → `gems run --config configs/experiments/a2_parity.yaml` (CPU).
2. Upload data + outputs to Drive (`docs/GOOGLE_DRIVE_LAYOUT.md`).
3. Colab 00 → 01 (profile features) → 02: **A1 gated-encoder pseudo-discovery** (the critical missing result) + A4 scoreboard, A3 designs.
4. Colab 03: B0 → B1, B2 (π sweep) → C1 → C2, C3. B3 only if B1/B2 pass.

## Deferred (authored, run after the above)

D2 (needs D1 family ablation), E1–E5, F (needs `tnm_urls.json` on Drive; download needs explicit approval),
G (after promoted candidates exist), submission (after a promotion), hierarchical 1 m encoder (only if F4 is
compelling), `src/` layout migration (`docs/ARCHITECTURE.md`).

## Compatibility risks

* Legacy notebooks import from the project root: unchanged behaviour (flat layout kept).
* `gems_bench.data` / `.features` / `.discovery` are now packages; old imports work, but stale `__pycache__`
  `.pyc` files from before the move must not shadow them (deleted this session).
* Phase-4 `configs/phase4` are read from the code checkout when the data workspace lacks them (Colab).
* Torch / CUDA versions on Colab change over time: pin `COMMIT`, record `environment_report()` per session.

## Exact next milestone

**5B execution:** obtain the gated dual encoder's pseudo-discovery scores (A1), the unified scoreboard (A4), metric
parity (A2) and the first PU / geometry comparisons against B0 (B1, B2, C1), then decide which of D / E / F gets
the next GPU budget, based on the promotion decisions.
