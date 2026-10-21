"""Kind ``lgbm5`` (Experiment D and the LightGBM arm of F4): LightGBM with Phase-5 feature families.

Exactly the Phase-3/4 LightGBM protocol (``lgbm3.run_variant_cv`` / ``fit_full_variant``: same sampler, variant,
seed, folds, discovery rule), run inside this experiment folder (``runs/``) through a nested context.
``features.base`` names a zoo model whose feature list is the starting point; ``add`` / ``add_prefixes`` append
Phase-5 features; ``ablate_families: true`` adds one drop-family CV run per added family (incremental value).
"""
from __future__ import annotations

import json

import pandas as pd

from .base import Runner


class Lgbm5Runner(Runner):
    kind = "lgbm5"

    @property
    def model_name(self) -> str:
        return self.cfg.get("model_name", self.cfg.name)

    def names(self) -> list[str]:
        from ...training.spec import base_spec_dict

        f = self.cfg.get("features", {}) or {}
        base = list(base_spec_dict(self.ctx, f["base"])["features"]) if f.get("base") else []
        store = self.ctx.store
        add = list(f.get("add", []) or [])
        for pre in f.get("add_prefixes", []) or []:
            add += [n for n in store.specs if n.startswith(pre)]
        drop = set(f.get("drop", []) or [])
        return [n for n in dict.fromkeys(base + add) if n not in drop]

    def families(self, names: list[str]) -> dict[str, list[str]]:
        fam: dict[str, list[str]] = {}
        for n in names:
            s = self.ctx.store.specs[n]
            if n.startswith(("p5:", "dem:")):
                fam.setdefault(getattr(s, "family", n), []).append(n)
        return fam

    def prepare(self):
        names = self.names()
        missing = [n for n in names if not self.ctx.store.has(n)]
        if missing:
            self.ctx.store.build(missing)
        self.rd.save_table(pd.DataFrame({"feature": names}), "feature_list")
        from ...evaluation.spatial_cv import buffer_audit

        self.rd.save_table(buffer_audit(self.ctx, names, float(self.ctx.cfg.buffer_km)), "buffer_audit")
        return {"n_features": len(names), "built": len(missing)}

    def _nctx(self):
        from ...phase5_context import nested_context

        return nested_context(self.ctx, self.rd.runs)

    def train(self):
        import numpy as np

        from ...p3 import lgbm3

        nctx, names, name = self._nctx(), self.names(), self.model_name
        v = lgbm3.LGBM_GRID[self.cfg.get("variant", "D2_conservative")]
        run = lgbm3.run_variant_cv(nctx, name, names, v, tag=name)
        self.exp.manifest.mark_unit("cv", {"exp": name})
        disc = {}
        if self.cfg.get("discovery", True):
            rounds = max(1, int(np.median(json.loads(json.dumps(run.state["summary"]["best_iters"])))))
            for d in range(nctx.splits.n_splits):
                lgbm3.fit_full_variant(nctx, f"disc{d}_{name}", names, v, rounds, ignore_mask=nctx.splits.ignore(d),
                                       predict_area=nctx.splits.eval_zone(d) & nctx.data.valid)
                disc[d] = nctx.paths.experiments / f"disc{d}_{name}" / "pred.tif"
                self.exp.manifest.mark_unit(f"disc{d}")
        self.register_maps(name, run.oof_path, disc)
        if self.cfg.get("ablate_families", False):
            for fam, members in self.families(names).items():
                key = f"{name}_drop_{fam.replace(':', '-')}"
                r = lgbm3.run_variant_cv(nctx, key, [n for n in names if n not in members], v, tag=key)
                self.register_maps(key, r.oof_path, None)
                self.exp.manifest.mark_unit(f"ablation/{fam}")
        return {"oof": self.rel(run.oof_path), "discovery_splits": len(disc)}

    def evaluate(self):
        from ... import phase4_eval

        ev = self.cfg.get("evaluation", {}) or {}
        name = self.model_name
        out = self.standard_evaluation(list(ev.get("models", [])) + [name], [name], ev.get("control"))
        abl = [m for m in self.exp.manifest.data["maps"] if m.startswith(f"{name}_drop_")]
        if abl:
            score = phase4_eval.geo_score(self.eval_ctx(), {m: self.ctx.oof(m) for m in abl + [name]},
                                          float(ev.get("match_frac", 0.05)))[0]
            self.rd.save_table(phase4_eval.comparison_rows(score, [(a, name) for a in abl], "drop_family"), "family_ablation")
        d = out["decisions"][0] if out["decisions"] else {}
        return {"promote_recommended": bool(d.get("promote", False))}
