"""Kind ``ensemble5`` (Experiment G): nested geographic fusion + honest discovery weights, promotion vs best member."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .base import Runner


class Ensemble5Runner(Runner):
    kind = "ensemble5"

    def prepare(self):
        ctx = self.ctx
        mem = list(self.cfg.get("members"))
        rows = [{"member": m, "has_oof": ctx.has_oof(m), "has_discovery": ctx.has_discovery(m)} for m in mem]
        self.rd.save_table(pd.DataFrame(rows), "members")
        bad = [r["member"] for r in rows if not (r["has_oof"] and r["has_discovery"])]
        if bad:
            raise FileNotFoundError(f"members without OOF + all discovery maps: {bad} (complete them first, e.g. A1)")
        return {"members": mem}

    def train(self):
        from ...discovery import ensemble5 as E

        ctx, rd = self.ctx, self.rd
        mem = list(self.cfg.get("members"))
        step = float(self.cfg.get("step", 0.25 if len(mem) > 3 else 0.1))
        U = E.quantiles(ctx, mem)
        dweights = self.cfg.get("discovery_weights", "geo_masked")
        made = {}
        for method in self.cfg.get("methods", ["rank_mean", "weighted_rank", "convex_stack"]):
            name = f"ENS_{method}"
            if self.exp.manifest.unit_done(name):
                continue
            if method in ("mean", "rank_mean"):
                geo = E.geo_simple(ctx, U, {m: ctx.oof(m) for m in mem} if method == "mean" else None, method)
                wsplit = {d: {m: 1.0 / len(mem) for m in mem} for d in range(ctx.splits.n_splits)}
                wtab = pd.DataFrame([{"zone": "all", "weights": json.dumps(wsplit[0])}])
            elif method == "weighted_rank":
                geo, wtab = E.geo_weighted(ctx, U, step)
                wsplit = None
            elif method == "convex_stack":
                geo, wtab = E.geo_convex_stack(ctx, U, np.random.default_rng(self.cfg.seed))
                wsplit = None
            elif method == "uncertainty":
                geo = E.geo_uncertainty(ctx, U, calibration_ok=bool(self.cfg.get("calibration_ok", False)))
                wsplit = {d: {m: 1.0 / len(mem) for m in mem} for d in range(ctx.splits.n_splits)}
                wtab = pd.DataFrame()
            else:
                raise ValueError(method)
            if wsplit is None:
                wsplit = {d: (E.weights_geo_masked(ctx, U, d, step) if dweights == "geo_masked"
                              else E.weights_loso(ctx, mem, d, step)) for d in range(ctx.splits.n_splits)}
            rd.save_table(wtab.assign(method=method), f"weights_geo_{method}")
            rd.save_json({str(d): w for d, w in wsplit.items()}, f"weights_discovery_{method}")
            oof = rd.save_raster({"oof_probability": geo}, f"{name}_oof", ctx.data.grid)
            disc = {}
            for d, p in E.discovery_maps(ctx, mem, wsplit).items():
                disc[d] = rd.save_raster({"probability": p}, f"{name}_disc{d}", ctx.data.grid)
            self.register_maps(name, oof, disc)
            self.exp.manifest.mark_unit(name)
            made[name] = method
        return made

    def evaluate(self):
        from ...discovery.ensemble5 import region_stability

        ev = self.cfg.get("evaluation", {}) or {}
        mem = list(self.cfg.get("members"))
        ens = [n for n in self.exp.manifest.data["maps"] if n.startswith("ENS_")]
        control = ev.get("control")
        if not control:                     # best member on the primary discovery policy, chosen from saved scores
            from ...evaluation.scoreboard import evaluate_models

            sc = evaluate_models(self.eval_ctx(), mem, match_frac=float(ev.get("match_frac", 0.05)))
            pp = (ev.get("promotion") or {}).get("primary_policy", "excl")
            control = sc["score"].sort_values(f"disc_dti_{pp}", ascending=False).model.iloc[0]
        out = self.standard_evaluation(mem, ens, control)
        self.rd.save_table(region_stability(self.ctx, {m: self.ctx.oof(m) for m in ens + [control]}), "region_stability")
        return {"control": control, "promote": {d["candidate"]: d["promote"] for d in out["decisions"]}}
