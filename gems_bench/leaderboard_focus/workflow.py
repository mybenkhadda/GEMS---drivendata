"""The single Milestone-1 entry point (``notebooks/leaderboard_focus/GEMS_M1_CHAMPION_TO_SUBMISSION.ipynb``).

    R = Runner(CONFIG, HEAD, M1_CONFIG)       # one master CONFIG; M1_CONFIG = optional m1.yaml overrides
    R.step("artifacts"); R.step("selection"); ...      # one notebook cell per stage (each resumable)
    R.run()                                            # or: every stage of CONFIG['mode'] in order

Modes (``PLAN``): FULL = every stage; RECOVER = inventory + selection (+ promotion view); VALIDATE = inventory +
validation + parity (+ promotion) on the persisted selection; SUBMIT = inventory + promotion + export (+ leaderboard)
on the persisted, validated recipe; PREFLIGHT = the read-only Drive preflight + report only. Every mode starts with
``preflight`` (no engine): when it is BLOCKED, every engine stage records INCOMPLETE before any attach. ``exploratory``
(opt-in) runs after ``submissions`` and is NOT_REQUIRED without an engine attach while ``exploratory.enabled`` is
false. A stage outside the mode is reported NOT_REQUIRED and its persisted manifest is
left untouched. A stage never runs on an unusable upstream result: it records INCOMPLETE (upstream not run / stale
code) or BLOCKED (promotion blocked) and nothing downstream is produced. Programming errors are recorded (FAILED +
traceback, failure report) and re-raised. The P6 engine is attached once per kernel and only for stages that need it.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from . import config as CF
from .config import BLOCKED, COMPLETE, FAILED, INCOMPLETE, NOT_REQUIRED, load_m1_config

PLAN = {
    "FULL": ("preflight", "artifacts", "selection", "validation", "parity", "promotion", "submissions", "exploratory",
             "leaderboard", "report"),
    "RECOVER": ("preflight", "artifacts", "selection", "promotion", "report"),
    "VALIDATE": ("preflight", "artifacts", "validation", "parity", "promotion", "report"),
    "SUBMIT": ("preflight", "artifacts", "promotion", "submissions", "exploratory", "leaderboard", "report"),
    "PREFLIGHT": ("preflight", "report"),
}
NEEDS_ENGINE = ("selection", "validation", "parity", "submissions", "exploratory")


def plan(mode: str) -> tuple[str, ...]:
    return PLAN[str(mode).upper()]


def engine_needed(mode: str) -> bool:
    return any(s in NEEDS_ENGINE for s in plan(mode))


class Runner:
    def __init__(self, CONFIG: dict, head: str, m1_config: dict | None = None):
        self.nb = CF.notebook_config(CONFIG)
        self.head = head
        self.overrides = CF.overrides_from_notebook(self.nb, m1_config)
        self.cfg = load_m1_config(self.overrides)
        self.stages = plan(self.nb["mode"])
        self.sess = None                                   # P6-engine session (attached once per kernel)
        self._ws_sess = None                               # engine-free session (manifests, report, leaderboard)
        self.inventory_ran = False
        self.results: dict[str, dict] = {}

    # ------------------------------------------------------------------------------- sessions
    def _bare(self):
        """Engine-free session on the local M1 root (the files stage 1 just wrote); pushes when sync is on."""
        from ..artifacts.sync import DriveSync
        from .session import M1Session, Workspace

        local = Path(self.nb["local_root"])
        out = self.cfg["outputs"]
        pc = DriveSync(local, Path(self.nb["drive_root"])) if out["sync_to_drive"] else None
        ws = Workspace(local / self.cfg["output_dir"], pc, local, save_figures=out["save_figures"],
                       save_parquet=out["save_parquet"])
        return M1Session(SimpleNamespace(bst=None, p6=None, cfg=None), self.cfg, ws)

    def light(self):
        from .session import M1Session

        if self.sess is not None:
            return self.sess
        if self._ws_sess is None:
            self._ws_sess = self._bare() if self.inventory_ran else M1Session.light(self.nb, self.overrides)
        return self._ws_sess

    def engine(self):
        """Attach the read-only P6 engine once (raises StageBlocked when the inventory is not COMPLETE)."""
        from .session import M1Session

        if self.sess is None:
            try:
                self.sess = M1Session.attach(self.nb, self.overrides)
            except Exception as e:
                from .session import StageBlocked

                if not isinstance(e, StageBlocked):
                    self.light().record("session", FAILED, "", reasons=[f"{type(e).__name__}: {e}"],
                                        traceback=traceback.format_exc()[-3000:])
                raise
            self.sess.record("session", COMPLETE, "", staged=self.sess.staged)     # clears an earlier failed attach
        return self.sess

    # ------------------------------------------------------------------------------- stages
    def step(self, name: str, **kw) -> dict:
        """Run one stage of the mode (cached results are reused inside the stage)."""
        from . import artifacts as AR
        from . import champion as CH
        from . import gates as GT
        from . import parity as PA
        from . import reporting as RP
        from . import submission as SU
        from . import validation as VA
        from .session import StageBlocked

        if name not in self.stages:
            m = {"stage": name, "status": NOT_REQUIRED, "reasons": [f"not part of mode {self.nb['mode']}"]}
            return self._done(name, m)
        try:
            if name == "preflight":
                from . import preflight as PF

                return self._done(name, PF.run(self.nb, self.overrides))
            if name == "exploratory" and not (self.cfg.get("exploratory") or {}).get("enabled"):
                return self._done(name, self.light().record(name, NOT_REQUIRED, "", reasons=[
                    "exploratory.enabled is false (default): no exploratory file, no engine attach"]))
            pf = self.results.get("preflight") or {}
            if name in NEEDS_ENGINE and pf.get("status") == BLOCKED:
                return self._done(name, self.light().record(name, INCOMPLETE, "", reasons=[
                    f"preflight BLOCKED (diagnostics/DRIVE_READINESS_REPORT.md): {(pf.get('reasons') or [''])[0]}"]))
            if name == "artifacts":
                out = AR.run_environment(self.nb, self.head, self.overrides)
                self.inventory_ran = True
                self._ws_sess = None
                return self._done(name, {**self.light().manifest("artifacts"), "inventory": out["inventory"]})
            if name in NEEDS_ENGINE:
                try:
                    s = self.engine()
                except StageBlocked as e:
                    return self._done(name, self.light().record(name, INCOMPLETE, "", reasons=[str(e)]))
            else:
                s = self.light()
            fn = {"selection": CH.recover, "validation": VA.validate, "parity": PA.diagnose, "promotion": GT.promote,
                  "submissions": SU.export, "exploratory": SU.export_exploratory, "report": RP.final_report,
                  "leaderboard": lambda x: RP.record_leaderboard(x, kw.get("entry"))}[name]
            out = fn(s)
            if name == "report":
                return self._done(name, {"stage": "report", "status": "COMPLETE", **out})
            if name == "leaderboard":
                return self._done(name, {**s.manifest("leaderboard"), "messages": out["messages"]})
            if name in ("submissions", "exploratory"):
                return self._done(name, {**s.manifest(name), "submission": out})
            return self._done(name, out)
        except Exception:
            try:
                RP.final_report(self.light())          # failure report on disk before the traceback surfaces
            except Exception as e2:                     # the original error matters more
                print(f"[m1] failure report not written: {type(e2).__name__}: {e2}")
            raise

    def _done(self, name: str, m: dict) -> dict:
        self.results[name] = m
        why = (m.get("reasons") or m.get("warnings") or [""])[0] if isinstance(m, dict) else ""
        cache = " (cached)" if m.get("cache") else ""
        print(f"[m1] {name:<12} {m.get('status')}{cache}" + (f" | {str(why)[:160]}" if why else ""))
        return m

    def run(self, leaderboard_entry: dict | None = None) -> dict:
        """Every stage of the mode in order; a FAILED stage stops the run (after the failure report)."""
        for s in self.stages:
            self.step(s, entry=leaderboard_entry) if s == "leaderboard" else self.step(s)
        return self.results

    def table(self) -> pd.DataFrame:
        """Persisted status of every stage (what the next session will see)."""
        s = self.sess or self.light()
        rows = []
        for st in ("preflight", "artifacts", "selection", "validation", "parity", "promotion", "submissions",
                   "exploratory", "leaderboard"):
            m = s.manifest(st)
            rows.append({"stage": st, "in_mode": st in self.stages, "status": m.get("status") or "not run",
                         "written": m.get("written"), "first_reason": (m.get("reasons") or [""])[0]})
        return pd.DataFrame(rows)

    def blocked(self) -> bool:
        return any(m.get("status") in (BLOCKED, FAILED, INCOMPLETE) for m in self.results.values())
