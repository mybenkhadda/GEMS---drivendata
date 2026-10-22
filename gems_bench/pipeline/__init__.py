"""Dependency-aware, resumable master pipeline (``notebooks/colab/GEMS_MASTER_PIPELINE.ipynb``).

    from gems_bench.pipeline import Pipeline
    pipe = Pipeline.from_notebook(CONFIG, DEM_CONFIG, urls=URLS)
    pipe.plan()                       # read-only: what is cached on Drive, what would run, what is blocked
    pipe.run()                        # the configured mode (default RESUME); or pipe.run_stage("feature_store")
    pipe.dashboard()

Modules: ``config`` (validated schema), ``registry`` (stage declarations), ``dependencies`` (graph, modes),
``manifests`` (stage records on Drive), ``executor`` (skip / build / resume / incompatible / blocked),
``stages`` (stage implementations over the library), ``baselines`` (restore-or-retrain catalogue),
``eda_rebuild`` (minimal Phase-1 rebuild), ``experiments`` (Phase-5 research items), ``status`` (dashboard).
"""
from __future__ import annotations

from .config import MODES, ConfigError, PipelineConfig
from .context import GPURequired, PipelineContext, PipelineStop


class Pipeline:
    def __init__(self, cfg: PipelineConfig, urls: dict | None = None):
        from .executor import Executor
        from .stages import build_registry

        self.cfg = cfg
        self.pc = PipelineContext(cfg, urls=dict(urls or {}))
        self.registry = build_registry()
        self.executor = Executor(self.pc, self.registry)
        self.pc._cache["registry"] = self.registry
        self.pc._cache["executor"] = self.executor

    @classmethod
    def from_notebook(cls, config: dict | None = None, dem: dict | None = None, *, urls: dict | None = None,
                      yaml_path=None, overrides: dict | None = None) -> "Pipeline":
        y = yaml_path or PipelineConfig.default_yaml()
        return cls(PipelineConfig.load(y, flat=config, dem=dem, overrides=overrides), urls=urls)

    # -- inspection ------------------------------------------------------------
    def plan(self, mode: str | None = None, root: str = "drive"):
        return self.executor.plan(mode=mode, root=root if self.pc.drive is not None else "local")

    def graph(self):
        return self.registry.table()

    def status(self):
        from .status import stage_table

        return stage_table(self.pc, self.registry, self.executor)

    def dashboard(self, save: bool = True):
        """Summary table + stage figure (displayed in notebooks); saved under reports/pipeline when ``save``."""
        from .status import dashboard_figure, save_fig, summary

        sm = summary(self.pc, self.registry, self.executor)
        fig = dashboard_figure(self.pc, self.registry, self.executor)
        try:
            from IPython.display import display

            display(sm)
            display(fig)
        except ImportError:
            print(sm.to_string(index=False))
        if save:
            sm.to_csv(self.pc.reports / "dashboard_summary.csv", index=False)
            save_fig(self.pc, fig, "pipeline_dashboard")
            self.pc.push(["reports/pipeline"])
        return sm

    def outputs(self, stage: str) -> list[str]:
        """Durable (Drive) locations of a stage's outputs."""
        s = self.registry[stage]
        base = self.pc.drive or self.pc.local
        return [str(base / r) for r in s.outputs(self.pc)]

    def record(self, stage: str) -> dict:
        return self.pc.manifests.load(stage)

    def events(self, last: int = 50):
        return self.pc.manifests.events(last)

    # -- execution -------------------------------------------------------------
    def run(self, mode: str | None = None, targets: list[str] | None = None, force: tuple[str, ...] = ()):
        mode = (mode or self.cfg.mode).upper()
        if mode not in MODES:
            raise ConfigError(f"mode {mode!r} not in {MODES}")
        if mode == "FULL_PIPELINE":
            exp = [s.name for s in self.registry if s.expensive and s.enabled(self.pc)[0]]
            print(f"[pipeline] FULL_PIPELINE: expensive stages enabled -> {exp or 'none'}")
        return self.executor.run(mode=mode, targets=targets, force=tuple(force))

    def run_stage(self, name: str, force: bool = False):
        r = self.executor.run_stage(name, force=force)
        print(f"[{name}] {r.outcome}: {r.reason[:300]}")
        if r.recovery:
            print(f"[{name}] recovery: {r.recovery}")
        if r.usable:
            print(f"[{name}] outputs: {self.outputs(name)[:6]}")
        return r

    def step(self, name: str, force: bool = False):
        """Notebook section cell: run ``name`` if the configured mode includes it, otherwise only report its
        validated state (so SETUP_ONLY / SUBMISSION_ONLY never build data by running the cells in order)."""
        from .dependencies import ANCESTORS_VALIDATE_ONLY, closure, mode_targets

        mode = self.cfg.mode
        targets = mode_targets(self.registry, mode)
        in_mode = name in closure(self.registry, targets)
        if in_mode and not (mode in ANCESTORS_VALIDATE_ONLY and name not in targets):
            return self.run_stage(name, force=force)
        s = self.registry[name]
        try:
            self.pc.stage_in(s.stage_in(self.pc))
            rep = s.validate(self.pc, self.pc.local) if s.validate else None
        except Exception as e:                        # e.g. raw data absent: nothing to validate yet
            print(f"[{name}] cannot validate yet: {type(e).__name__}: {e}")
            return None
        why = "not part of" if not in_mode else "validated only in"
        print(f"[{name}] {why} mode {mode}: outputs {rep.status if rep else 'n/a'} - {rep.summary() if rep else ''}")
        return rep

    def push_all(self) -> None:
        """Push every stage's outputs and records (use before closing a session after manual work)."""
        from .manifests import DIR

        rels = [DIR, "reports"]
        for s in self.registry:
            try:
                rels += s.outputs(self.pc)
            except Exception:
                pass
        self.pc.push(list(dict.fromkeys(rels)))

    # -- shortcuts ---------------------------------------------------------------
    def baselines(self):
        from . import baselines as B
        from .stages import folds_sha, splits_sha

        return B.status_table(self.pc.local, grid=self.pc.grid(), n_splits=self.pc.bench_cfg.n_discovery_folds,
                              folds_sha=folds_sha(self.pc), splits_sha=splits_sha(self.pc))

    def experiments(self):
        import pandas as pd

        from .experiments import catalogue_table, experiment_config, experiment_state

        cat = catalogue_table(list(self.cfg.get("experiments.advanced") or []))
        st = []
        for stem in cat.experiment:
            try:
                st.append(experiment_state(self.pc.drive or self.pc.local, experiment_config(self.pc, stem)))
            except FileNotFoundError as e:
                st.append({"error": str(e)})
        return pd.concat([cat, pd.DataFrame(st)], axis=1)


__all__ = ["Pipeline", "PipelineConfig", "PipelineContext", "PipelineStop", "GPURequired", "ConfigError", "MODES"]
