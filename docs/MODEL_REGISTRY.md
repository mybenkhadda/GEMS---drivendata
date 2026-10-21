# Model registry

`model_registry/registry.csv` (current state), `transitions.jsonl` (append-only history with evidence),
`evidence/<model>.json` (latest evidence). Implemented in `gems_bench.artifacts.registry.ModelRegistry`.

| status | how a model gets there |
|---|---|
| `baseline` | pre-Phase-5 reference models, registered by the A1 zoo run (`export` stage) |
| `candidate` | automatically after a Phase-5 evaluate stage |
| `validated` | automatically when its scorecard is complete (geographic + both discovery policies) |
| `promoted` | **explicitly**: `gems registry promote --model M --experiment ID`, accepted only if that experiment saved a decision with `promote: true` (section-17 rules). Promoting a model demotes the previous champion to `validated` |
| `rejected` | explicitly (`gems registry reject`), from any status |

Allowed transitions: candidate → validated / rejected; validated → promoted / rejected / candidate;
promoted → validated / rejected; rejected → candidate; baseline → rejected.

The submission run (`z_submission.yaml`) exports the champion; exporting anything else requires
`model.allow_unpromoted: true` (recorded in the provenance JSON). Champion-vs-challenger reports are the
`reports/experiment_summary.md` of each evaluate stage.
