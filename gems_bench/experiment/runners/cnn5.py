"""Kind ``cnn5`` (Experiments B, C, E and the CNN arm of F4): one Phase-5 CNN vs its control.

prepare   resolve the P5Spec (base run + overrides + GPU auto-config), build missing feature channels,
          record parameter counts of candidate and control (capacity must not explain a gain)
train     geographic CV (3 resumable fold units) + pseudo-discovery (3 resumable split units)
evaluate  shared scorecard / scoreboard / promotion vs ``control`` + kind-specific diagnostics:
          geometry metrics (always), orientation error (``export_heads``), PU diagnostics (``pu`` != none)
export    registry update, figures, required-table contract
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from .base import Runner


class Cnn5Runner(Runner):
    kind = "cnn5"

    @property
    def model_name(self) -> str:
        return self.cfg.get("model_name", self.cfg.name)

    def spec(self):
        if getattr(self, "_spec", None) is None:
            from ...training.gpu import autoconfig
            from ...training.spec import spec_from_experiment, with_gpu

            s = spec_from_experiment(self.ctx, self.cfg.get("model"), self.model_name, self.cfg.seed)
            gpu = {**autoconfig(patch_px=s.patch_px, effective_batch=s.batch_size * max(s.grad_accum, 1),
                                n_channels=len(s.channels)), **(self.cfg.get("runtime.gpu", {}) or {})}
            self._gpu = gpu
            self._spec = with_gpu(s, gpu)
        return self._spec

    def prepare(self):
        from ...models.factory import build_model5, count_params
        from ...training.cv import ensure_channels, write_spec
        from ...training.spec import spec_hash

        spec = self.spec()
        ensure_channels(self.ctx, spec)
        write_spec(self.rd, spec)
        rows = [{"model": self.model_name, "n_params": count_params(build_model5(spec)), "n_channels": len(spec.channels),
                 "arch": spec.model, "fusion": spec.gsn_fusion if spec.model == "geostructurenet" else spec.fusion,
                 "spec_hash": spec_hash(spec)}]
        ctrl = self.cfg.get("evaluation.control")
        if ctrl:
            try:
                from ...training.spec import base_spec_dict

                rows.append({"model": ctrl, "n_params": base_spec_dict(self.ctx, ctrl).get("n_params")})
            except Exception:
                pass
        self.rd.save_table(pd.DataFrame(rows), "architecture")
        self.rd.save_json({"gpu_overrides": self._gpu}, "runtime_config")
        return {"spec_hash": spec_hash(spec), "n_params": rows[0]["n_params"]}

    def train(self):
        from ...training.cv import discovery_epochs, prepare_stack5, run_cv5, run_discovery5, runtime_tables

        spec, rd, m = self.spec(), self.rd, self.exp.manifest
        stack = prepare_stack5(self.ctx, spec, rd)
        dl = self.exp.deadline()
        run_cv5(self.ctx, rd, spec, sync=self.exp.drive, deadline=dl, stack=stack, mark=m.mark_unit)
        disc = {}
        if self.cfg.get("discovery", True):
            ep = int(self.cfg.get("discovery_epochs") or discovery_epochs(rd, spec.min_epochs))
            run_discovery5(self.ctx, rd, spec, ep, sync=self.exp.drive, deadline=dl, stack=stack, mark=m.mark_unit)
            disc = {d: rd.raster_path(f"disc{d}") for d in range(self.ctx.splits.n_splits)}
        self.register_maps(self.model_name, rd.raster_path("oof"), disc)
        runtime_tables(rd)
        return {"oof": "predictions/oof.tif", "discovery_splits": len(disc)}

    def _orient_oof(self) -> tuple[np.ndarray, np.ndarray] | None:
        """OOF orientation rasters from the fold models (only for models with an orientation head)."""
        if self.spec().aux_orient <= 0:
            return None
        from ...cnn import device_of
        from ...inference.full_raster import predict_heads_mask
        from ...phase4_training import pooled_context, to_tensor_stack
        from ...training.checkpoint import CheckpointManager
        from ...training.cv import prepare_stack5
        from ...training.trainer import load_unit_model

        if self.rd.raster_path("oof_orient").exists():
            import rasterio

            with rasterio.open(self.rd.raster_path("oof_orient")) as src:
                return src.read(1), src.read(2)
        dev = device_of()
        spec = self.spec()
        g = to_tensor_stack(prepare_stack5(self.ctx, spec), dev)
        gctx = pooled_context(g, spec.context_factor) if spec.context else None
        c2 = np.full(self.ctx.data.shape, np.nan, np.float32)
        s2 = c2.copy()
        for k in self.ctx.fold_ids:
            _, model = load_unit_model(CheckpointManager(self.rd.checkpoints, f"cv_fold{k}"), dev)
            val = self.ctx.folds.val(k)
            h = predict_heads_mask(model, g, val, spec, dev, gctx)
            c2[val], s2[val] = h["ocos"][val], h["osin"][val]
        self.rd.save_raster({"ocos": c2, "osin": s2}, "oof_orient", self.ctx.data.grid)
        return c2, s2

    def evaluate(self):
        from ...evaluation.geometry_metrics import geometry_scorecard, orientation_error, withheld_distance
        from ...evaluation.pu_diagnostics import confidence_mass, training_collapse

        ev = self.cfg.get("evaluation", {}) or {}
        control = ev.get("control")
        name = self.model_name
        out = self.standard_evaluation(list(ev.get("models", [])) + [name], [name], control)
        ctx = self.eval_ctx()
        frac = float(ev.get("match_frac", 0.05))
        names = [n for n in (name, control) if n and ctx.has_oof(n)]
        maps = {n: ctx.oof(n) for n in names}
        self.rd.save_table(geometry_scorecard(ctx, maps, frac), "geometry_metrics")
        self.rd.save_table(withheld_distance(ctx, [n for n in names if ctx.has_discovery(n)], frac), "withheld_distance")
        o = self._orient_oof() if self.cfg.get("export_heads", False) else None
        if o is not None:
            self.rd.save_table(orientation_error(ctx, {name: o}), "orientation_error")
        if self.spec().pu != "none" or self.spec().consistency > 0:
            self.rd.save_table(confidence_mass(ctx, maps, frac=frac), "pu_confidence_mass")
            if self.rd.has_table("training_history"):
                self.rd.save_table(training_collapse(self.rd.load_table("training_history")), "pu_training_collapse")
        d = out["decisions"][0] if out["decisions"] else {}
        return {"promote_recommended": bool(d.get("promote", False)), "criteria": d.get("criteria", {})}

    def export(self):
        import matplotlib.pyplot as plt

        from ...visualization import figures as F

        rd = self.rd
        if rd.has_table("geometry_metrics"):
            fig = F.short_fault_recall(rd.load_table("geometry_metrics"))
            rd.save_fig(fig, "short_fault_recall")
            plt.close(fig)
        if rd.has_table("training_history"):
            h = rd.load_table("training_history")
            if "loss_unlabeled_mean_p" in h:
                fig = F.pu_diagnostics(h)
                rd.save_fig(fig, "pu_diagnostics")
                plt.close(fig)
        rd.save_text(json.dumps(asdict(self.spec()), indent=2, default=str), "spec_final.json")
        return {"figures": sorted(p.name for p in rd.figures.glob("*.png"))}
