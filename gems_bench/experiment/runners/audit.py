"""Kinds ``parity`` (Experiment A2) and ``splits`` (Experiment A3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import Runner


class ParityRunner(Runner):
    kind = "parity"
    requires_prepare = False

    def evaluate(self):
        from ...evaluation.parity import convention_audit, official_adapter, reference_dti, run_parity
        from ...metric import dti_components

        off_path = self.cfg.get("parity.official_scorer")
        official = official_adapter(self.exp.paths.root / off_path, self.cfg.get("parity.official_function", "score")) \
            if off_path else None
        cases = run_parity(official)
        self.rd.save_table(cases, "parity_cases")
        ctx = self.ctx
        conv = convention_audit(ctx)
        self.rd.save_table(conv, "conventions")
        # real-data spot checks: local scorer vs the KD-tree reference on small windows of saved OOF maps
        rows = []
        for m in self.cfg.get("parity.real_maps", ["G1"]):
            p = ctx.oof(m)
            for (r, c) in self.cfg.get("parity.windows", [[1500, 1500], [2500, 1200]]):
                sl = (slice(r, r + 150), slice(c, c + 150))
                ev = ctx.data.valid[sl]
                loc = dti_components(p[sl], ctx.data.labels[sl], eval_mask=ev).dti
                ref = reference_dti(p[sl], ctx.data.labels[sl], eval_mask=ev)["dti"]
                rows.append({"model": m, "window": f"{r},{c}", "local": loc, "reference": ref, "abs_diff": abs(loc - ref)})
        real = pd.DataFrame(rows)
        self.rd.save_table(real, "parity_real_windows")
        ok_ref = bool(cases.pass_reference.all()) and bool((real.abs_diff < 1e-9).all() if len(real) else True)
        ok_off = bool(cases.get("pass_official", pd.Series([False])).fillna(False).all()) if official else None
        lines = ["# Metric parity (A2)", "",
                 f"- local vs independent reference implementation: **{'PASS' if ok_ref else 'FAIL'}**",
                 f"- local vs official scorer: **{'PASS' if ok_off else 'FAIL' if ok_off is False else 'UNVERIFIED (no official scorer supplied)'}**",
                 "", "Local DTI may be called leaderboard-equivalent only if the official comparison passed.", "",
                 cases.to_markdown(index=False, floatfmt=".6f"), "", conv.to_markdown(index=False)]
        self.rd.save_text("\n".join(lines), "experiment_summary.md")
        return {"reference_parity": ok_ref, "official_parity": ok_off}

    def export(self):
        from ...evaluation.parity import submission_audit

        sub = self.cfg.get("parity.submission")
        if not sub:
            return {}
        res = submission_audit(self.exp.paths.root / sub, self.ctx.data)
        self.rd.save_json(res, "submission_audit")
        return {"submission_ok": True}


class SplitsRunner(Runner):
    """Builds the A3 designs (saved under ``data/processed/splits/<name>``) and reports their properties.

    Scoring a model under a new design requires retraining it there: point a cnn5 / lgbm5 config at the design
    with ``validation.discovery_dir`` / ``validation.folds_dir``. Only label-free maps (B0) are rescored here.
    """
    kind = "splits"

    def _root(self) -> Path:
        return self.exp.paths.processed / "splits"

    def prepare(self):
        from ...evaluation import spatial_cv as S

        ctx, out = self.ctx, {}
        for d in self.cfg.get("designs", []):
            name, kind = d["name"], d["design"]
            target = self._root() / name
            if (target / "fold_map.tif").exists() or (target / "discovery_map.tif").exists():
                out[name] = "exists"
                continue
            kw = {k: v for k, v in d.items() if k not in ("name", "design")}
            if kind == "system_kfold":
                obj = S.system_kfold(ctx, **kw)
            elif kind == "length_stratified":
                obj = S.length_stratified(ctx, **kw)
            elif kind == "region_loro":
                obj = S.region_loro(ctx, **kw)
            else:
                raise ValueError(kind)
            S.save_design(obj, self._root(), name)
            out[name] = "built"
            self.push_paths([target])
        return out

    def evaluate(self):
        from ...evaluation import spatial_cv as S
        from ...folds import DiscoverySplits, GeoFolds

        ctx, rows = self.ctx, []
        for d in self.cfg.get("designs", []):
            p = self._root() / d["name"]
            if (p / "discovery_map.tif").exists():
                s = DiscoverySplits.load(p, ctx.data.grid)
                for i in range(s.n_splits):
                    rows.append({"design": d["name"], "split": i, "withheld_px": int(s.withheld(i).sum()),
                                 "eval_zone_km2": float(s.eval_zone(i).sum() * 0.01),
                                 "n_systems": int((s.systems.discovery_split == i).sum())})
            else:
                f = GeoFolds.load(p, ctx.data.grid)
                for k in range(f.n_folds):
                    rows.append({"design": d["name"], "split": k, "val_km2": float(f.val(k).sum() * 0.01),
                                 "val_label_px": int((ctx.data.labels & f.val(k)).sum()),
                                 "inner_val_km2": float(f.inner_val(k).sum() * 0.01)})
        self.rd.save_table(pd.DataFrame(rows), "design_summary")
        chans = self.cfg.get("buffer_audit.channels") or []
        if chans:
            self.rd.save_table(S.buffer_audit(ctx, chans, float(self.cfg.get("buffer_audit.buffer_km", 13.0))), "buffer_audit")
        hard_refs = [m for m in self.cfg.get("hard_subregions.models", []) if ctx.has_oof(m)]
        if hard_refs:
            self.rd.save_table(S.hard_subregions(ctx, {m: ctx.oof(m) for m in hard_refs}), "hard_subregions")
        return {"designs": len(self.cfg.get("designs", []))}


__all__ = ["ParityRunner", "SplitsRunner", "np"]
