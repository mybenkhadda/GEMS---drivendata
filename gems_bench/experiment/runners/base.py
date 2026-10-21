"""Runner base class and the shared evaluation / reporting path used by every model-producing kind."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class Runner:
    kind = "base"
    requires_prepare = True

    def __init__(self, exp):
        self.exp = exp
        self.cfg = exp.config
        self.rd = exp.rd

    # -- context (optionally on another validation design) --------------------
    @property
    def ctx(self):
        if getattr(self, "_ctx", None) is None:
            from ...evaluation.spatial_cv import with_folds, with_splits

            ctx = self.exp.ctx
            fd, sd = self.cfg.get("validation.folds_dir"), self.cfg.get("validation.discovery_dir")
            if fd:
                ctx = with_folds(ctx, self.exp.paths.root / fd)
            if sd:
                ctx = with_splits(ctx, self.exp.paths.root / sd)
            self._ctx = ctx
        return self._ctx

    def prepare(self):
        return {}

    def train(self):
        return {}

    def evaluate(self):
        return {}

    def export(self):
        return {}

    # -- helpers --------------------------------------------------------------
    def rel(self, p: Path) -> str:
        return Path(p).resolve().relative_to(self.rd.path.resolve()).as_posix()

    def register_maps(self, name: str, oof: Path | None, disc: dict[int, Path] | None) -> None:
        self.exp.manifest.add_maps(name, self.rel(oof) if oof else None, {d: self.rel(p) for d, p in (disc or {}).items()})
        self.ctx.register_maps(name, oof, disc, self.exp.id)
        if self.ctx is not self.exp.ctx:
            self.exp.ctx.register_maps(name, oof, disc, self.exp.id)

    def push_paths(self, paths: list[Path]) -> None:
        if self.exp.drive is None:
            return
        rels = [Path(p).resolve().relative_to(self.exp.paths.root.resolve()).as_posix() for p in paths]
        self.exp.drive.push(rels)

    def artifacts_ok(self, name: str) -> bool:
        ctx = self.ctx
        try:
            return ctx.has_oof(name) and ctx.has_discovery(name)
        except Exception:
            return False

    def eval_ctx(self):
        """Scoring context: optionally restricted to DEM-covered pixels (Experiment F4)."""
        if self.cfg.get("evaluation.restrict") == "dem_coverage":
            from ...dem.benchmark import coverage_mask, coverage_report, covered_context

            cov = coverage_mask(self.ctx, float(self.cfg.get("evaluation.min_coverage", 0.9)))
            self.rd.save_table(coverage_report(self.ctx, cov), "dem_coverage_report")
            return covered_context(self.ctx, cov)
        return self.ctx

    def standard_evaluation(self, models: list[str], candidates: list[str], control: str | None,
                            register: bool = True) -> dict:
        import matplotlib.pyplot as plt

        from ...discovery.atlas import complementarity, recovery_matrix
        from ...evaluation.protocol import protocol_from_ctx
        from ...evaluation.scoreboard import (champion_report, competition_scoreboard, decision_table, evaluate_models,
                                              load_leaderboard, promotion_decision)
        from ...visualization import figures as F

        ev = self.cfg.get("evaluation", {}) or {}
        frac = float(ev.get("match_frac", 0.05))
        refs = tuple(ev.get("refs", ["G1"]))
        rules = ev.get("promotion", {}) or {}
        ctx = self.eval_ctx()
        proto = protocol_from_ctx(ctx, frac, rules.get("primary_policy", "excl"))
        models = [m for m in dict.fromkeys(models + candidates + ([control] if control else [])) if m]
        sc = evaluate_models(ctx, models, match_frac=frac, refs=refs)
        score = sc["score"]
        rd = self.rd
        rd.save_table(score, "scorecard")
        geo_cols = ["model"] + [c for c in score.columns if c.startswith(("geo_", "dti_fold"))]
        rd.save_table(score[geo_cols], "geographic_dti")
        disc_cols = ["model"] + [c for c in score.columns if c.startswith("disc_")]
        rd.save_table(score[disc_cols], "pseudo_discovery_dti")
        if len(sc["disc_per_split"]):
            rd.save_table(sc["disc_per_split"], "pseudo_discovery_per_split")
        rd.save_table(sc["recovery"], "subgroup_recall")
        rd.save_table(sc["geo_curves"], "predicted_area_curve")
        if len(sc["disc_curves"]):
            rd.save_table(sc["disc_curves"], "discovery_area_curve")
        rd.save_table(sc["unique"], "unique_withheld_systems")
        maps = {m: ctx.oof(m) for m in score.model}
        try:
            pairs, per = complementarity(ctx, maps, frac, refs=refs + ((control,) if control else ()), support=sc["support"])
            rd.save_table(pairs, "model_complementarity")
            rd.save_table(per, "model_complementarity_per_model")
        except Exception as e:                                     # descriptive table only: never block a run
            rd.log(f"complementarity skipped: {e}")
        if sc["disc_objs"]:
            rd.save_table(recovery_matrix(ctx, sc, frac, rules.get("primary_policy", "excl")), "fault_recovery_atlas")
        lb = load_leaderboard(self.exp.paths.reports / "leaderboard_log.csv")
        board = competition_scoreboard(sc, proto.hash(), lb, ref=refs[0])
        rd.save_table(board, "scoreboard")
        decisions = [promotion_decision(sc, c, control, rules, artifacts_ok=self.artifacts_ok(c))
                     for c in candidates if control and c != control]
        if decisions:
            rd.save_table(decision_table(decisions), "promotion_decisions")
            rd.save_json(decisions, "promotion_decisions")
        rd.save_text(champion_report(board, decisions, f"{self.exp.id}: champion vs challengers"), "experiment_summary.md")
        fig = F.experiment_dashboard(board, ref=control)
        rd.save_fig(fig, "experiment_dashboard")
        plt.close(fig)
        fig = F.dti_area_curves(sc["geo_curves"], list(score.model), title="Geographic DTI vs predicted area")
        rd.save_fig(fig, "geo_dti_area_curves")
        plt.close(fig)
        if len(sc["disc_curves"]):
            pp = rules.get("primary_policy", "excl")
            fig = F.dti_area_curves(sc["disc_curves"][sc["disc_curves"].policy == pp], list(score.model),
                                    title=f"Pseudo-discovery DTI vs predicted area ({pp})")
            rd.save_fig(fig, "discovery_dti_area_curves")
            plt.close(fig)
        self.exp.manifest.data["protocol"] = proto.as_dict()
        self.exp.manifest.save()
        if register:
            self._registry_update(board, candidates, proto.hash())
        return {"sc": sc, "board": board, "decisions": decisions}

    def _registry_update(self, board: pd.DataFrame, candidates: list[str], protocol_hash: str) -> None:
        from ...artifacts.registry import RegistryError

        reg = self.exp.registry
        b = board.set_index("model")
        for c in candidates:
            if c not in b.index:
                continue
            m = {"geo_dti": _f(b.loc[c, "geo_dti_area"]), "disc_dti_excl": _f(b.loc[c].get("disc_dti_excl")),
                 "disc_dti_retained_fp": _f(b.loc[c].get("disc_dti_retained_fp"))}
            reg.register(c, "candidate", kind=self.cfg.kind, experiment_id=self.exp.id, config_hash=self.cfg.hash(),
                         git_commit=self.exp.manifest.data.get("git_commit", ""), protocol_hash=protocol_hash, metrics=m)
            if reg.status(c) == "candidate" and all(v is not None for v in m.values()):
                try:
                    reg.transition(c, "validated", reason=f"complete scorecard in {self.exp.id}", metrics=m,
                                   evidence={"protocol_hash": protocol_hash})
                except RegistryError as e:
                    self.rd.log(f"registry: {e}")


def _f(x):
    try:
        x = float(x)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None
