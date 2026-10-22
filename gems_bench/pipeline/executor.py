"""Dependency-aware, resumable execution of the stage graph.

Per stage, in topological order:

1. **blocked**   a hard dependency has no usable output  -> BLOCKED (names the dependency and its state)
2. stage in      the stage's outputs are copied Drive -> local (unchanged files skipped)
3. validate      content-level checks of the outputs (``artifacts.validation``)
4. decide
   * valid + fingerprint matches (or no record: adopt)     -> SKIP (cached)
   * valid + record made under another configuration       -> INCOMPATIBLE (never overwritten)
   * missing                                               -> BUILD
   * partial / invalid                                     -> RESUME (invalid local files quarantined first)
   * incompatible content (grid / params differ)           -> INCOMPATIBLE
   * disabled by configuration                             -> DISABLED (downstream blocked if outputs missing)
5. run           ``stage.run`` (idempotent; library engines skip completed units)
6. re-validate   only valid outputs mark the stage done; push outputs + record to Drive

Failures are recorded with the exception, the source location (file:line), the traceback and a recovery
instruction. ``KeyboardInterrupt`` and training-budget stops are recorded as *interrupted* (resumable) and end
the run cleanly. What happens next depends on ``failure_policy``:

* ``batch`` (default, backward compatible): stages that do not depend on the failure continue; a blocked
  stage records its *root* failure(s), per-stage BLOCKED lines are suppressed, and one grouped report
  (``failure_report``) closes the run.
* ``interactive`` (notebook): the failure is shown in full (traceback, file:line, recovery). If the stage is
  ``critical`` (data-preparation backbone) the run stops with ``StageFailed``; completed artifacts are kept.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import pandas as pd

from ..artifacts import validation as V
from .context import PipelineContext, PipelineStop, StageFailed
from .dependencies import ANCESTORS_VALIDATE_ONLY, closure, descendants, mode_targets
from .manifests import fp_hash, now, output_fingerprints
from .registry import ALWAYS_RUN, Stage, StageRegistry

# decisions / outcomes
SKIP, BUILD, RESUME, RUN = "SKIP_CACHED", "BUILD", "RESUME", "RUN"
DONE, FAILED, BLOCKED, DISABLED = "DONE", "FAILED", "BLOCKED", "DISABLED"
INCOMPATIBLE, MISSING_PREREQ, INTERRUPTED = "INCOMPATIBLE", "MISSING_PREREQ", "INTERRUPTED"
USABLE = (SKIP, DONE)
STOPPING = (FAILED, INCOMPATIBLE)          # outcomes of a critical stage that end an interactive run
ROOT_OUTCOMES = (FAILED, INCOMPATIBLE, INTERRUPTED, DISABLED, MISSING_PREREQ)


@dataclass
class StageResult:
    stage: str
    decision: str = ""
    outcome: str = ""
    validation: str = ""
    reason: str = ""
    recovery: str = ""
    seconds: float = 0.0
    summary: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    location: str = ""                                   # file:line of the failing frame (our code first)
    traceback: str = ""
    root_causes: list = field(default_factory=list)      # BLOCKED: the upstream stage(s) that actually failed

    @property
    def usable(self) -> bool:
        return self.outcome in USABLE

    def row(self) -> dict:
        return {"stage": self.stage, "decision": self.decision, "outcome": self.outcome,
                "validation": self.validation, "seconds": round(self.seconds, 1), "reason": self.reason[:300],
                "recovery": self.recovery[:300], "location": self.location,
                "root_cause": ", ".join(self.root_causes), "outputs": ", ".join(self.outputs[:3]) +
                (" ..." if len(self.outputs) > 3 else "")}


def _external(filename: str) -> bool:
    """Frames of installed libraries / the standard library (not where a project bug can be fixed)."""
    import sysconfig

    fn = filename.replace("\\", "/")
    std = sysconfig.get_paths().get("stdlib", "").replace("\\", "/")
    return (any(k in fn for k in ("site-packages/", "dist-packages/", "<frozen")) or (bool(std) and fn.startswith(std)))


def error_location(exc: BaseException) -> str:
    """``path:line in function`` of the innermost frame outside installed libraries (the failing project code,
    e.g. a class body in ``eda_rebuild.py``, not the executor that called it); else the innermost frame."""
    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    if not frames:
        return ""
    ours = [f for f in frames if not _external(f.filename)]
    f = (ours or frames)[-1]
    path = f.filename.replace("\\", "/")
    for k in ("gems_bench/", "gems_eda/"):
        if k in path:
            path = k + path.split(k, 1)[1]
            break
    return f"{path}:{f.lineno} in {f.name}"


class Executor:
    def __init__(self, pc: PipelineContext, registry: StageRegistry):
        self.pc = pc
        self.reg = registry
        self.results: dict[str, StageResult] = {}

    # -- planning (read-only) ------------------------------------------------
    def plan(self, mode: str | None = None, targets: list[str] | None = None, root: str = "drive") -> pd.DataFrame:
        """Predicted action per stage from the outputs under Drive (default) or local; nothing is written."""
        pc = self.pc
        r = pc.root(root) or pc.local
        mode = mode or pc.cfg.mode
        order = closure(self.reg, targets or mode_targets(self.reg, mode))
        usable, rows = set(), []
        for name in order:
            s = self.reg[name]
            en, why = s.enabled(pc)
            try:
                rep = s.validate(pc, r) if s.validate else V.Report(name)
                st, detail = rep.status, rep.summary()
            except Exception as e:                       # e.g. raw data absent: validators cannot even start
                st, detail = "unknown", f"{type(e).__name__}: {str(e)[:160]}"
            rec = pc.manifests.load(name)
            blk = [d for d in s.deps if d not in usable]
            if st == V.OK and not self._fp_conflict(s, rec):
                action = SKIP
                usable.add(name)
            elif blk:
                action = f"{BLOCKED} by {', '.join(blk)}"
            elif not en:
                action = f"{DISABLED} ({why})"
            elif st == V.INCOMPATIBLE or self._fp_conflict(s, rec):
                action = INCOMPATIBLE
            else:
                action = RUN if s.reuse == ALWAYS_RUN else (RESUME if st in (V.PARTIAL, V.INVALID) else BUILD)
                usable.add(name)                         # optimistic: downstream planned as runnable
            rows.append({"stage": name, "title": s.title, "outputs": st, "planned": action,
                         "last_status": rec.get("status", ""), "last_origin": rec.get("origin", ""),
                         "last_finished": rec.get("finished", ""), "resource": s.resource,
                         "est_minutes": s.est_minutes, "detail": detail[:240]})
        return pd.DataFrame(rows)

    # -- execution -------------------------------------------------------------
    def run(self, mode: str | None = None, targets: list[str] | None = None, force: tuple[str, ...] = ()) -> pd.DataFrame:
        pc = self.pc
        mode = mode or pc.cfg.mode
        tg = targets or mode_targets(self.reg, mode)
        order = closure(self.reg, tg)
        validate_only = set(order) - set(tg) if mode in ANCESTORS_VALIDATE_ONLY else set()
        if mode == "SETUP_ONLY":
            pc.allow_download = False
        print(f"[pipeline] mode {mode} ({self.policy}): {len(order)} stage(s): {' -> '.join(order)}")
        usable: set[str] = {n for n, r in self.results.items() if r.usable}
        for i, name in enumerate(order):
            res = self._execute(self.reg[name], usable, validate_only=name in validate_only, mode=mode,
                                force=name in force)
            self.results[name] = res
            if res.usable:
                usable.add(name)
            if res.outcome != BLOCKED or self.policy == "interactive":
                print(f"[pipeline] {name:<22s} {res.outcome:<14s} {res.reason[:110]}")
            if res.outcome == FAILED:
                self.print_failure(res, full=self.policy == "interactive")
            if res.outcome == INTERRUPTED:
                print("[pipeline] interrupted: completed work is on Drive; re-run the same cell to resume")
                break
            if self.stops(res):
                rest = [n for n in order[i + 1:] if n in descendants(self.reg, name)]
                print(f"[pipeline] stopped (failure_policy=interactive): critical stage {name} {res.outcome}; "
                      f"not run: {', '.join(rest) or 'nothing downstream'}. Completed artifacts are kept.")
                raise StageFailed(name, res.reason, res.recovery, res.location)
        if any(r.outcome in (FAILED, BLOCKED, INCOMPATIBLE) for r in self.results.values()):
            print(self.failure_report_text())
        return self.table()

    @property
    def policy(self) -> str:
        return str(self.pc.cfg.get("failure_policy") or "batch")

    def stops(self, res: StageResult) -> bool:
        """True when ``res`` must end an interactive run (critical stage failed / incompatible)."""
        return self.policy == "interactive" and res.outcome in STOPPING and self.reg[res.stage].critical

    def run_stage(self, name: str, force: bool = False) -> StageResult:
        """One stage; dependencies are validated (not built) and must be usable."""
        usable, states = set(), {}
        for d in sorted(set(closure(self.reg, [name])) - {name}):
            r = self.results.get(d)
            if r is not None and r.usable:
                usable.add(d)
                continue
            s = self.reg[d]
            try:
                self.pc.stage_in(s.stage_in(self.pc))
                rep = s.validate(self.pc, self.pc.local) if s.validate else V.Report(d, [V.Check(d, V.OK)])
                if rep.ok:
                    usable.add(d)
                elif r is None:
                    states[d] = f"not run in this session; outputs {rep.status}"
            except Exception as e:
                if r is None:
                    states[d] = f"not run in this session; validation error {type(e).__name__}"
        res = self._execute(self.reg[name], usable, validate_only=False, mode="RESUME", force=force, states=states)
        self.results[name] = res
        return res

    # -- one stage ---------------------------------------------------------------
    def _fp_conflict(self, s: Stage, rec: dict) -> bool:
        if not rec or rec.get("status") != "done" or s.reuse == ALWAYS_RUN:
            return False
        old = rec.get("fingerprint_hash")
        return bool(old) and old != fp_hash(s.fingerprint(self.pc))

    def _root_causes(self, blk: list[str]) -> list[str]:
        roots = []
        for d in blk:
            r = self.results.get(d)
            roots += r.root_causes if (r is not None and r.outcome == BLOCKED and r.root_causes) else [d]
        return list(dict.fromkeys(roots))

    def _execute(self, s: Stage, usable: set[str], *, validate_only: bool, mode: str, force: bool,
                 states: dict | None = None) -> StageResult:
        pc, m = self.pc, self.pc.manifests
        res = StageResult(s.name)
        blk = [d for d in s.deps if d not in usable]
        if blk:
            res.decision = res.outcome = BLOCKED
            st = {d: (self.results[d].outcome if d in self.results else (states or {}).get(d, "not run"))
                  for d in blk}
            res.root_causes = self._root_causes(blk)
            res.reason = "missing prerequisite(s): " + ", ".join(f"{d} ({v})" for d, v in st.items())
            direct = [r for r in res.root_causes if r not in blk]
            if direct:
                res.reason += " <- root failure: " + ", ".join(
                    f"{r} ({self.results[r].outcome if r in self.results else 'not run'})" for r in direct)
            res.recovery = "; ".join(dict.fromkeys(filter(None, (self.results[r].recovery for r in res.root_causes
                                                                  if r in self.results))))
            if not res.recovery:
                res.recovery = "run the prerequisite stage(s) first: " + ", ".join(
                    f"pipe.step('{r}')" for r in res.root_causes)
            return res
        enabled, why = s.enabled(pc)
        t0 = time.perf_counter()
        try:
            pc.stage_in(s.stage_in(pc))
            rep = s.validate(pc, pc.local) if s.validate else V.Report(s.name)
        except Exception as e:
            rep = V.Report(s.name, [V.Check(s.name, V.MISSING, f"validation could not run: {type(e).__name__}: {e}")])
        res.validation = rep.status
        rec = m.load(s.name)
        fp = s.fingerprint(pc)
        conflict = self._fp_conflict(s, rec)
        runnable = enabled and not validate_only

        if rep.ok and not conflict and not force and not (s.reuse == ALWAYS_RUN and runnable):
            res.decision = res.outcome = SKIP
            res.reason = rep.summary()
            if rec.get("status") != "done":                     # pre-existing artifacts: adopt with provenance
                self._record(s, "done", "adopted_existing", fp, rep, t0, {})
            res.outputs = s.outputs(pc)
            return res
        if rep.status == V.INCOMPATIBLE or (conflict and rep.ok and not force):
            res.decision = res.outcome = INCOMPATIBLE
            res.reason = (rep.summary() if rep.status == V.INCOMPATIBLE else
                          "valid outputs exist but were produced under another configuration "
                          f"(recorded {rec.get('fingerprint_hash')} vs current {fp_hash(fp)})")
            res.recovery = (s.recovery or "existing artifacts are preserved; restore the previous configuration, or "
                            "move the outputs aside on Drive and re-run (versioned stages use a new folder automatically)")
            m.event(s.name, "incompatible", reason=res.reason)
            return res
        if validate_only:
            res.decision = res.outcome = MISSING_PREREQ
            res.reason = f"{rep.summary()} (mode {mode} does not build prerequisites)"
            res.recovery = s.recovery or f"run mode PREPARE_DATA / RESUME to build {s.name}"
            return res
        if not enabled:
            res.decision = res.outcome = DISABLED
            res.reason = f"{why}; outputs: {rep.status}"
            res.recovery = f"enable it in CONFIG to build {s.name}" if not rep.ok else ""
            return res
        if mode == "RESUME" and not pc.cfg.get("rebuild_missing") and rep.status == V.MISSING \
                and rec.get("status") not in ("running", "interrupted", "failed") and s.reuse != ALWAYS_RUN:
            res.decision = res.outcome = DISABLED
            res.reason = "outputs missing and rebuild_missing is false (RESUME only continues started stages)"
            return res

        res.decision = RUN if s.reuse == ALWAYS_RUN else RESUME if rep.status in (V.PARTIAL, V.INVALID) else BUILD
        self._quarantine(rep)
        self._record(s, "running", "resumed" if res.decision == RESUME else "built", fp, None, t0, {},
                     started=now(), attempts=int(rec.get("attempts", 0)) + 1)
        m.event(s.name, "start", decision=res.decision, validation=rep.status)
        TrainingInterrupted = _training_interrupted()
        try:
            res.summary = s.run(pc) or {}
        except (KeyboardInterrupt, TrainingInterrupted) as e:
            res.outcome, res.reason = INTERRUPTED, f"{type(e).__name__}: {e}"
            res.recovery = "re-run the same cell (completed units are skipped, training resumes from checkpoints)"
            self._finish_failed(s, "interrupted", res, fp, t0)
            return res
        except PipelineStop as e:
            res.outcome, res.reason, res.recovery = FAILED, str(e), e.recovery or s.recovery
            res.location, res.traceback = error_location(e), traceback.format_exc()
            self._finish_failed(s, "failed", res, fp, t0, tb=res.traceback[-4000:])
            return res
        except Exception as e:
            res.outcome, res.reason = FAILED, f"{type(e).__name__}: {str(e)[:500]}"
            res.recovery = s.recovery or ("a code or data error, not a missing artifact: fix the code at the location "
                                          "shown (or update the checkout), then re-run; completed work is kept")
            res.location, res.traceback = error_location(e), traceback.format_exc()
            self._finish_failed(s, "failed", res, fp, t0, tb=res.traceback[-4000:])
            return res
        try:
            rep2 = s.validate(pc, pc.local) if s.validate else V.Report(s.name)
        except Exception as e:
            rep2 = V.Report(s.name, [V.Check(s.name, V.INVALID, f"post-run validation failed: {e}")])
        res.validation = rep2.status
        res.seconds = time.perf_counter() - t0
        res.outputs = s.outputs(pc)
        pc.push(res.outputs)
        if rep2.ok:
            res.outcome = DONE
            res.reason = rep2.summary()
            self._record(s, "done", "resumed" if res.decision == RESUME else "built", fp, rep2, t0, res.summary)
            m.event(s.name, "done", seconds=round(res.seconds, 1))
        else:
            res.outcome = FAILED
            res.reason = f"outputs invalid after run: {rep2.summary()}"
            res.recovery = s.recovery or "inspect the listed files; re-run resumes"
            self._finish_failed(s, "failed", res, fp, t0)
        return res

    # -- helpers -------------------------------------------------------------------
    def _quarantine(self, rep: V.Report) -> None:
        """Rename invalid *local* files (e.g. truncated) so a rebuild cannot mistake them for cached ones.
        Drive copies are never touched here; the verified rebuild replaces them through the atomic push."""
        from pathlib import Path

        for c in rep.checks:
            p = Path(c.path)
            if c.status == V.INVALID and p.is_file() and self.pc.local in p.parents:
                q = p.with_name(p.name + f".invalid-{int(time.time())}")
                p.rename(q)
                print(f"[pipeline] quarantined invalid local file {p.name} -> {q.name} ({c.reason})")

    def _record(self, s: Stage, status: str, origin: str, fp: dict, rep: V.Report | None, t0: float, summary: dict,
                **extra) -> None:
        from .._version import provenance

        pc = self.pc
        rec = {"status": status, "origin": origin, "fingerprint": fp, "fingerprint_hash": fp_hash(fp),
               "resource": s.resource, "requirement": s.requirement, **provenance(),
               "summary": _jsonable(summary), **extra}
        if status == "done":
            rec.update(finished=now(), seconds=round(time.perf_counter() - t0, 1),
                       outputs=output_fingerprints(pc.local, [o for o in s.outputs(pc) if (pc.local / o).exists()]),
                       validation=rep.summary() if rep else "")
        prev = pc.manifests.load(s.name)
        if status != "done":
            rec = {**prev, **rec}
        else:
            rec["attempts"] = prev.get("attempts", 1)
            rec["started"] = prev.get("started", rec.get("finished"))
        pc.manifests.save(s.name, rec)
        if status == "done":
            from .manifests import STAGE_DIR

            pc.push([f"{STAGE_DIR}/{s.name}.json"])

    def _finish_failed(self, s: Stage, status: str, res: StageResult, fp: dict, t0: float, tb: str = "") -> None:
        res.seconds = time.perf_counter() - t0
        try:
            self.pc.push(s.outputs(self.pc))                      # keep partial progress for the next session
        except Exception as e:
            print(f"[pipeline] push after {status} failed: {e}")
        self.pc.manifests.update(s.name, status=status, error=res.reason, recovery=res.recovery, traceback=tb,
                                 location=res.location, finished=now(), fingerprint_hash=fp_hash(fp))
        self.pc.manifests.event(s.name, status, reason=res.reason[:300], location=res.location)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([r.row() for r in self.results.values()])

    # -- failure reporting -------------------------------------------------------------
    def print_failure(self, res: StageResult, full: bool = True) -> None:
        """The failure as the user needs it: error, file:line, recovery (+ full traceback when ``full``)."""
        bar = "=" * 100
        print(bar)
        print(f"STAGE {res.outcome}: {res.stage}")
        print(f"  error    : {res.reason[:1000]}")
        if res.location:
            print(f"  where    : {res.location}")
        if res.recovery:
            print(f"  recovery : {res.recovery}")
        down = sorted(descendants(self.reg, res.stage), key=self.reg.names().index)
        if down:
            print(f"  blocks   : {', '.join(down)}")
        if full and res.traceback:
            print("  traceback:")
            print(res.traceback.rstrip())
        elif res.traceback:
            print(f"  traceback: pipe.explain('{res.stage}')")
        print("  completed artifacts are kept locally and on Drive; re-running resumes after the fix")
        print(bar)

    def failure_report(self) -> pd.DataFrame:
        """One row per root failure, with the stages it blocks (this session)."""
        rows, blocked = [], {n: r for n, r in self.results.items() if r.outcome == BLOCKED}
        roots = list(dict.fromkeys([n for n, r in self.results.items() if r.outcome in ROOT_OUTCOMES]
                                   + [c for r in blocked.values() for c in r.root_causes]))
        for root in roots:
            r = self.results.get(root)
            under = [n for n, b in blocked.items() if root in b.root_causes]
            if r is not None and r.outcome in (DISABLED, MISSING_PREREQ) and not under:
                continue                                  # switched off / validate-only and blocking nothing
            rows.append({"root": root, "outcome": r.outcome if r else "not run", "error": (r.reason if r else "")[:300],
                         "location": r.location if r else "", "recovery": (r.recovery if r else
                                                                           f"run pipe.step('{root}') first")[:300],
                         "blocked": ", ".join(under), "n_blocked": len(under)})
        return pd.DataFrame(rows, columns=["root", "outcome", "error", "location", "recovery", "blocked", "n_blocked"])

    def failure_report_text(self) -> str:
        df = self.failure_report()
        if df.empty:
            return "[pipeline] no failures"
        out = ["[pipeline] failure report (fix the root, then re-run: completed stages are skipped)"]
        for r in df.itertuples():
            out.append(f"  ROOT {r.root} {r.outcome}: {r.error[:200]}" + (f"  [{r.location}]" if r.location else ""))
            if r.recovery:
                out.append(f"       recovery: {r.recovery[:200]}")
            if r.n_blocked:
                out.append(f"       blocked ({r.n_blocked}): {r.blocked}")
        return "\n".join(out)


def _training_interrupted():
    """The trainer's budget-stop exception (a never-raised placeholder when the training stack cannot import)."""
    try:
        from ..training.trainer import TrainingInterrupted

        return TrainingInterrupted
    except Exception:
        class _Never(Exception):
            pass

        return _Never


def _jsonable(d):
    import json

    try:
        json.dumps(d, default=str)
        return json.loads(json.dumps(d, default=str))
    except (TypeError, ValueError):
        return {"repr": str(d)[:2000]}
