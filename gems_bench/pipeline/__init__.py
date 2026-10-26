"""Dependency-aware, resumable master pipeline (``notebooks/colab/GEMS_MASTER_PIPELINE.ipynb``).

    from gems_bench.pipeline import Pipeline
    pipe = Pipeline.from_notebook(CONFIG, DEM_CONFIG, urls=URLS)
    pipe.overview()                   # read-only: config, code commit, Drive, GPU, raw data, planned actions
    pipe.plan()                       # read-only: what is cached on Drive, what would run, what is blocked
    pipe.run()                        # the configured mode (default RESUME); or pipe.step("feature_store")
    pipe.explain("geological_eda")    # error, file:line, recovery, traceback of a stage (this or a past session)
    pipe.failure_report()             # root failures and the stages each one blocks
    pipe.dashboard()
    pipe.inventory()                  # read-only: every experiment folder, profile, units, reusability, next action
    pipe.plan_experiments()           # read-only: reuse-first plan at experiments.profile (SCREEN / VALIDATE / FULL)
    pipe.compare_results()            # read-only: metrics comparison, paired deltas, shortlist -> reports/experiment_screening
    pipe.research_dashboard()
    pipe.m6(M6_CONFIG).run()         # M6: audit, paired comparison, calibration, ensembles, errors, gate (no training)

Modules: ``config`` (validated schema), ``registry`` (stage declarations), ``dependencies`` (graph, modes),
``manifests`` (stage records on Drive), ``executor`` (skip / build / resume / incompatible / blocked),
``stages`` (stage implementations over the library), ``baselines`` (restore-or-retrain catalogue),
``eda_rebuild`` (minimal Phase-1 rebuild), ``experiments`` (Phase-5 research items), ``status`` (dashboard).
"""
from __future__ import annotations

from .config import MODES, ConfigError, PipelineConfig
from .context import GPURequired, PipelineContext, PipelineStop, StageFailed


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
        """Run one stage (dependencies validated, not built). Under ``failure_policy: interactive`` a failed
        critical stage raises ``StageFailed`` after printing the full diagnostics, so "Run all" stops there."""
        ex = self.executor
        r = ex.run_stage(name, force=force)
        if r.outcome == "FAILED":
            ex.print_failure(r, full=ex.policy == "interactive")
        elif r.outcome == "BLOCKED":
            root = ", ".join(r.root_causes)
            print(f"[{name}] BLOCKED by {root}: {r.reason[:300]}")
            print(f"[{name}] fix the root stage first (pipe.explain('{r.root_causes[0]}') shows why)"
                  if r.root_causes else "")
        else:
            print(f"[{name}] {r.outcome}: {r.reason[:300]}")
            if r.recovery:
                print(f"[{name}] recovery: {r.recovery}")
        if r.usable:
            print(f"[{name}] outputs: {self.outputs(name)[:6]}")
        if ex.stops(r):
            raise StageFailed(name, r.reason, r.recovery, r.location)
        return r

    def explain(self, stage: str) -> dict:
        """Why a stage is in its state: outcome, error, file:line, recovery and the traceback (this session's
        result, else the record saved on Drive by an earlier session)."""
        r = self.executor.results.get(stage)
        rec = self.pc.manifests.load(stage)
        info = {"stage": stage, "outcome": r.outcome if r else rec.get("status", "never run"),
                "error": (r.reason if r else rec.get("error", "")) or "",
                "location": (r.location if r else rec.get("location", "")) or "",
                "recovery": (r.recovery if r else rec.get("recovery", "")) or "",
                "root_causes": list(r.root_causes) if r else [],
                "traceback": (r.traceback if r else rec.get("traceback", "")) or ""}
        for k in ("stage", "outcome", "error", "location", "recovery", "root_causes"):
            if info[k]:
                print(f"{k:12s}: {info[k]}")
        if info["traceback"]:
            print(info["traceback"].rstrip())
        return info

    def failure_report(self):
        """Root failures of this session and the stages each one blocks."""
        print(self.executor.failure_report_text())
        return self.executor.failure_report()

    def overview(self):
        """First-run status: configuration, code revision, Drive, GPU, raw data, cached artifacts, planned actions.
        Read-only (validates, never builds)."""
        from .._version import provenance
        from .dependencies import closure, mode_targets

        pc, cfg = self.pc, self.cfg
        prov = provenance()
        print(f"mode           : {cfg.mode}   failure_policy: {cfg.get('failure_policy')}")
        print(f"code           : ref '{cfg.get('ref')}' -> commit {prov['git_commit']} (package {prov['version']})")
        print(f"Drive          : {pc.drive if pc.drive is not None else 'NOT MOUNTED - nothing will persist'}")
        g = pc.gpu()
        ok, why = pc.cuda_ready()
        gpu_txt = f"{g.get('name')} ({g.get('vram_gb')} GB, cc {g.get('capability')})" if ok else f"none - {why}"
        print(f"GPU            : {gpu_txt}; torch {g.get('torch')} (CUDA build {g.get('torch_cuda')})")
        order = closure(self.registry, mode_targets(self.registry, cfg.mode))
        gpu_stages = [s.name for s in self.registry if s.resource == "gpu" and s.name in order]
        if not ok and gpu_stages:
            print(f"                 GPU stages in this mode ({', '.join(gpu_stages)}) will be DISABLED; "
                  "CPU preprocessing and LightGBM baselines are unaffected")
        try:
            from .stages import raw_validate

            raw = raw_validate(pc, pc.drive or pc.local)
            print(f"raw data       : {raw.summary()}")
        except Exception as e:
            print(f"raw data       : cannot validate yet ({type(e).__name__}: {e})")
        plan = self.plan()
        print("planned actions: " + ", ".join(f"{r.stage}={r.planned}" for r in plan.itertuples()))
        return plan

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

    # -- research economy (fidelity, reuse, screening) -------------------------
    def inventory(self, save: bool = True):
        """Read-only table of every experiment folder on Drive: profile, units, metrics, reusability, next action."""
        from .inventory import experiment_inventory

        inv = experiment_inventory(self.pc)
        if save:
            inv.to_csv(self.pc.reports / "experiment_inventory.csv", index=False)
            self.pc.push(["reports/pipeline/experiment_inventory.csv"])
        return inv

    def plan_experiments(self, profile: str | None = None, stems: list[str] | None = None):
        """What ``advanced_models`` would do at ``profile`` (default: experiments.profile); read-only."""
        from .experiments import plan_experiments

        return plan_experiments(self.pc, stems, profile)

    def compare_results(self):
        """Read-only metrics comparison + paired deltas + resource usage + shortlist, exported to
        reports/experiment_screening (CSV, PNG/SVG, SCREENING_REPORT.md). No training."""
        from .screening import export_screening_report

        return export_screening_report(self.pc)

    def shortlist(self, from_profile: str = "SCREEN") -> list[str]:
        """Experiments recommended for the next fidelity level (you decide whether to run them)."""
        from .screening import shortlisted

        return shortlisted(self.pc, from_profile)

    def research_dashboard(self, save: bool = True) -> dict:
        from .screening import research_dashboard

        return research_dashboard(self.pc, save=save)

    def ensure_map(self, name: str = "B0_lineament") -> dict:
        """Find / verify / (re)build a label-free dependency map with its original CPU recipe."""
        from .experiments import ensure_label_free_map

        return ensure_label_free_map(self.pc, name, requested_by="manual")

    # -- M6: model selection on existing FULL runs (no training) ------------------
    def m6(self, config: dict | None = None):
        """M6 analysis object (``.run()`` / single phases); reads saved FULL predictions, writes reports/m6."""
        from .m6 import M6Analysis

        return M6Analysis(self.pc, config)

    # -- GEMS 3.0: discovery-first research on existing predictions (no CNN retraining) ---------------------------
    def gems3(self, config: dict | None = None):
        """GEMS3 research object (``.run()`` / single stages); writes reports/gems3 (docs/NEXT_SESSION_HANDOFF.md)."""
        from ..gems3.run import GEMS3Research

        return GEMS3Research(self.pc, config)

    # -- M7: discovery-first research (candidate coverage before ranking; no CNN retraining) ---------------------
    def m7(self, config: dict | None = None):
        """M7 research object (one method per notebook section); writes outputs/m7_discovery_first."""
        from ..m7.run import M7Research

        return M7Research(self.pc, config)


__all__ = ["Pipeline", "PipelineConfig", "PipelineContext", "PipelineStop", "StageFailed", "GPURequired",
           "ConfigError", "MODES"]
