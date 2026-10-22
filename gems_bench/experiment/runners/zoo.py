"""Kind ``zoo`` (Experiments A1 + A4): complete missing Phase-4 protocol runs, then score every established
model under one protocol and write the competition-alignment scoreboard.

The gated dual encoder's pseudo-discovery runs are trained with the *Phase-4* code (``run_cnn4_discovery_all``,
spec from its config.json, epochs = median best CV epoch) into ``outputs/phase4/experiments/disc<d>_P4_B_gated``,
i.e. exactly the protocol that produced DL00 / DL02 / RS05 discovery maps. Only throughput knobs (precision,
inference batch) may be adapted to the Colab GPU; they are recorded.
"""
from __future__ import annotations

import json
from dataclasses import fields, replace

from .base import Runner


class ZooRunner(Runner):
    kind = "zoo"

    def _spec(self, name: str):
        from ...phase4_training import P4Spec, spec_from_config
        from ...training.gpu import autoconfig
        from ...training.spec import base_spec_dict

        cfg = base_spec_dict(self.ctx, name)          # legacy config.json, else the committed snapshot
        spec = spec_from_config(cfg, P4Spec)
        gpu = {**autoconfig(patch_px=spec.patch_px, effective_batch=spec.batch_size * max(spec.grad_accum, 1),
                            n_channels=len(spec.channels)), **(self.cfg.get("runtime.gpu", {}) or {})}
        allowed = {f.name for f in fields(P4Spec)} & {"amp_dtype", "infer_batch", "batch_size", "grad_accum",
                                                       "data_on_gpu", "cudnn_benchmark", "channels_last"}
        return replace(spec, **{k: v for k, v in gpu.items() if k in allowed}), gpu

    def prepare(self):
        ctx = self.ctx
        rows = []
        for n in self.cfg.get("complete_discovery", []) or []:
            spec, gpu = self._spec(n)
            missing = [c for c in spec.channels if not ctx.store.has(c)]
            rows.append({"model": n, "channels": len(spec.channels), "missing_channels": ",".join(missing),
                         "has_discovery": ctx.has_discovery(n), "gpu_overrides": json.dumps(gpu)})
            if missing:
                raise FileNotFoundError(f"{n}: feature caches missing (stage them from Drive or build): {missing}")
        import pandas as pd

        self.rd.save_table(pd.DataFrame(rows), "discovery_completion_plan")
        return {"to_complete": [r["model"] for r in rows if not r["has_discovery"]]}

    def train(self):
        from ...models.zoo import BY_NAME
        from ...phase4_training import run_cnn4_discovery_all

        ctx, done = self.ctx, {}
        for n in self.cfg.get("complete_discovery", []) or []:
            if ctx.has_discovery(n):
                done[n] = "already complete"
                continue
            spec, _ = self._spec(n)
            epochs = run_cnn4_discovery_all(ctx, n, spec, BY_NAME[n].experiment)
            self.exp.manifest.mark_unit(f"discovery/{n}", {"epochs": epochs})
            self.push_paths([ctx.paths.experiments / f"disc{d}_{n}" for d in range(ctx.splits.n_splits)])
            done[n] = f"trained ({epochs} epochs)"
        return done

    def evaluate(self):
        ev = self.cfg.get("evaluation", {}) or {}
        models = list(self.cfg.get("models", []))
        out = self.standard_evaluation(models, list(ev.get("candidates", [])), ev.get("control"), register=False)
        return {"n_models": len(out["board"]), "protocol_hash": self.exp.manifest.data["protocol"]["protocol_hash"]}

    def export(self):
        from ...artifacts.registry import RegistryError

        reg, out = self.exp.registry, {}
        from ...models.zoo import BY_NAME

        b = self.rd.load_table("scoreboard").set_index("model")
        for n in self.cfg.get("models", []):
            if n in BY_NAME and n in b.index:
                try:
                    reg.register(n, "baseline", kind=BY_NAME[n].kind, experiment_id=BY_NAME[n].experiment,
                                 protocol_hash=str(b.loc[n, "protocol_hash"]),
                                 metrics={"geo_dti": float(b.loc[n, "geo_dti_area"]),
                                          "disc_dti_excl": b.loc[n].get("disc_dti_excl"),
                                          "disc_dti_retained_fp": b.loc[n].get("disc_dti_retained_fp")},
                                 notes=BY_NAME[n].note, overwrite=True)
                    out[n] = "baseline"
                except RegistryError as e:
                    out[n] = str(e)
        return out
