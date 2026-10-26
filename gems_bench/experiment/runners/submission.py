"""Kind ``submission`` (Milestone 5D): full-raster inference of the promoted model + validated GeoTIFF export.

The model is the registry champion (status ``promoted``) unless ``model.allow_unpromoted: true`` names another
one explicitly (e.g. a fallback). Full-region map = average of the champion's CV fold models over the whole
footprint (no retraining). The area fraction is the champion's geographic cross-fitted budget (``geo_frac``),
applied as a fraction of the full footprint. Every known-fault policy variant is exported; which one to upload
is decided by leaderboard probes, recorded by the user in reports/leaderboard_log.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .base import Runner


class SubmissionRunner(Runner):
    kind = "submission"

    def prepare(self):
        reg = self.exp.registry
        name = self.cfg.get("model.name")
        if not name:
            ch = reg.champion()
            if ch is None:
                raise RuntimeError("no promoted model in the registry; promote one or set model.name + allow_unpromoted")
            name = ch["name"]
        row = reg.get(name)
        if row is None:
            raise RuntimeError(f"{name} is not registered")
        if row["status"] != "promoted" and not self.cfg.get("model.allow_unpromoted", False):
            raise RuntimeError(f"{name} has status {row['status']}; set model.allow_unpromoted: true to export it anyway")
        self.rd.save_json(row, "submission_model")
        return {"model": name, "status": row["status"], "experiment": row.get("experiment_id")}

    def _model_row(self) -> dict:
        return json.loads((self.rd.reports / "submission_model.json").read_text(encoding="utf-8"))

    def train(self):
        """Full-region probability map (predictions/full_region.tif)."""
        from ...inference.full_raster import cnn5_full_region, cv_checkpoints

        row = self._model_row()
        ctx = self.ctx
        if self.rd.raster_path("full_region").exists():
            return {"full_region": "exists"}
        exp_dir = self.exp.paths.experiments / str(row["experiment_id"])
        if row["kind"] == "cnn5" and cv_checkpoints(exp_dir):
            p = cnn5_full_region(ctx, exp_dir)
        elif row["kind"] in ("lgbm5", "lgbm"):
            import lightgbm as lgb

            from ...gbm import predict_mask

            run_dir = exp_dir / "runs" / "experiments" / row["name"] if row["kind"] == "lgbm5" else self._legacy_dir(row["name"])
            names = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["features"]
            boosters = [lgb.Booster(model_file=str(f)) for f in sorted((run_dir / "models").glob("fold*.txt"))]
            p = np.nanmean([predict_mask(b, ctx.store, names, ctx.data.valid) for b in boosters], 0).astype(np.float32)
        else:
            raise NotImplementedError(f"full-region inference for kind {row['kind']}: export member maps first")
        self.rd.save_raster({"probability_full_region": p}, "full_region", ctx.data.grid)
        return {"full_region": "written"}

    def _legacy_dir(self, name: str):
        from ...models.zoo import experiment_dir

        return experiment_dir(self.ctx, name)

    def export(self):
        from ...inference.submission5 import export_submission

        row = self._model_row()
        ctx = self.ctx
        frac = self.cfg.get("export.area_frac")
        if frac is None:
            from ... import phase4_eval

            p_oof = ctx.oof(row["name"])
            frac = float(phase4_eval.geo_score(ctx, {row["name"]: p_oof}, 0.05)[0].geo_frac.iloc[0])
        prov = {"model": row["name"], "experiment_id": row.get("experiment_id"), "registry_status": row["status"],
                "config_hash": row.get("config_hash"), "git_commit": self.exp.manifest.data.get("git_commit"),
                "protocol_hash": row.get("protocol_hash")}
        out = export_submission(ctx, self.rd.read_raster("full_region"), self.exp.paths.submissions, row["name"],
                                mode=self.cfg.get("export.mode", "binary_area"), frac=frac,
                                known_policies=tuple(self.cfg.get("export.known_policies", ["model", "suppress_known"])),
                                provenance=prov)
        self.rd.save_table(pd.DataFrame([{k: v for k, v in o.items() if k != "audit"} for o in out]), "submission_files")
        self.push_paths([self.exp.paths.submissions])
        return {"files": [o["file"] for o in out], "area_frac": frac}
