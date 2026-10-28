"""Shared M1 setup: one call builds the read-only P6 engine on the CPU and the M1 workspace.

    m1 = M1Session.attach(CONFIG, M1_CONFIG)        # pipeline (cached data only) -> M9 -> P6 -> p6g (mode recovery)

* The p6g engine (``p6.full_gpu.FullGPU``) runs in ``mode: recovery`` (no GPU, no training) and its ``root`` IS the M1
  output root, so ``final_submission`` writes its files under ``outputs/leaderboard_focus/m1_champion_recovery``;
  nothing under ``outputs/p6_fast_track`` or ``outputs/p6_boost`` is written.
* ``attach`` refuses to start when stage 1 has not produced a COMPLETE artifact inventory. It stages exactly the
  inventory's ``staged`` files (``staging_manifest.json``), never whole P6 directories (``fg.stage_inputs`` would copy
  ``seeds/``, the historical submissions and every boost stage).
* Stage manifests ``manifests/<stage>.json``: status, reasons, warnings, fingerprint, ``impl`` (hash of the code the
  stage ran), inputs, commit, seconds; every write is also kept in ``manifests/history/``. ``cached`` reuses only a
  COMPLETE / WARNING result with the same fingerprint and every output present: a BLOCKED result is always
  re-evaluated (its blocker may have been fixed outside the fingerprint). ``require`` refuses upstream results that
  are not COMPLETE / WARNING or that were produced by different stage code.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from . import config as CF
from .config import BLOCKED, COMPLETE, FAILED, INCOMPLETE, PASSING, WARNING, load_m1_config

PKG = Path(__file__).resolve().parents[1]
# Code each stage's numbers depend on (an edit elsewhere never invalidates the stage).
IMPL_MODULES = {
    "selection": ("leaderboard_focus/champion.py", "p6/full_benchmark.py", "p6/export_parity.py"),
    "validation": ("leaderboard_focus/validation.py", "p6/boost.py", "p6/final_submission.py", "gems3/official.py",
                   "gems3/representation.py", "gems3/submission.py"),
    "parity": ("leaderboard_focus/parity.py", "p6/final_submission.py", "p6/stack.py"),
    "submissions": ("leaderboard_focus/submission.py", "p6/final_submission.py", "gems3/submission.py"),
}
DEPENDENCIES = ("numpy", "pandas", "scipy", "rasterio", "lightgbm", "scikit-learn")


class StageBlocked(RuntimeError):
    """A stage cannot run because an upstream stage is not usable (recorded as INCOMPLETE)."""


class StageInvalid(RuntimeError):
    """A required input of the stage is absent or invalid (recorded as BLOCKED with the message)."""


def dependency_versions() -> dict:
    from importlib import metadata

    out = {}
    for d in DEPENDENCIES:
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            out[d] = None
    return out


def impl_fp(stage: str) -> str | None:
    """Hash of the source files of ``IMPL_MODULES[stage]`` + numeric dependency versions (None: not tracked)."""
    mods = IMPL_MODULES.get(stage)
    if not mods:
        return None
    h = hashlib.sha256()
    for m in mods:
        p = PKG / m
        h.update(m.encode() + (p.read_bytes() if p.exists() else b"<absent>"))
    h.update(json.dumps(dependency_versions(), sort_keys=True).encode())
    return h.hexdigest()[:16]


class Workspace:
    """Atomic writes under the M1 root (local) + Drive push of the root (skip unchanged, DriveSync ledger)."""

    def __init__(self, root: Path, pc=None, local_root: Path | None = None, *, save_figures: bool = True,
                 save_parquet: bool = True):
        self.root, self.pc, self.local_root = Path(root), pc, Path(local_root) if local_root else None
        self.save_figures, self.save_parquet = save_figures, save_parquet

    def path(self, rel: str) -> Path:
        return self.root / rel

    def save_json(self, obj, rel: str) -> Path:
        from ..artifacts.atomic import atomic_write_json
        from ..p6.boost import _js

        return atomic_write_json(self.path(rel), json.loads(json.dumps(obj, default=_js)))

    def read_json(self, rel: str) -> dict | None:
        p = self.path(rel)
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    def save_table(self, df: pd.DataFrame, rel: str, parquet: bool = False) -> Path:
        from ..artifacts.atomic import atomic_save

        d = df if df is not None else pd.DataFrame()
        p = self.path(rel)
        atomic_save(p, lambda t: d.to_csv(t, index=False))
        if parquet and self.save_parquet and len(d):
            from ..pipeline.m6 import _parquet_safe

            try:
                atomic_save(p.with_suffix(".parquet"), lambda t: _parquet_safe(d).to_parquet(t, index=False))
            except Exception as e:                       # optional format
                print(f"[m1] parquet skipped for {rel}: {type(e).__name__}: {e}")
        return p

    def load_table(self, rel: str) -> pd.DataFrame:
        """The table, or an empty frame when absent / 0 bytes / no header (a header-only file keeps its columns)."""
        p = self.path(rel)
        if not p.exists() or not p.stat().st_size:
            return pd.DataFrame()
        try:
            return pd.read_csv(p, dtype={"systems": str})
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def require_table(self, rel: str, columns=()) -> pd.DataFrame:
        """A table a stage cannot work without: absent / empty / missing columns -> ``StageInvalid``."""
        T = self.load_table(rel)
        miss = [c for c in columns if c not in T]
        if not len(T) or miss:
            raise StageInvalid(f"{rel} is {'missing columns ' + str(miss) if len(T) else 'absent or empty'}: "
                               "re-run the stage that writes it")
        return T

    def save_text(self, text: str, rel: str) -> Path:
        from ..artifacts.atomic import atomic_write_text

        return atomic_write_text(self.path(rel), text)

    def save_fig(self, fig, rel_png: str, data: pd.DataFrame | None) -> Path | None:
        """PNG + its plotting data (``<name>_data.csv``, always written when given, even with figures disabled)."""
        import matplotlib.pyplot as plt

        from ..artifacts.atomic import atomic_save

        p = self.path(rel_png)
        if self.save_figures:
            atomic_save(p, lambda t: fig.savefig(t, dpi=160, bbox_inches="tight", format="png"))
        if data is not None:
            atomic_save(p.with_name(p.stem + "_data.csv"), lambda t: data.to_csv(t, index=False))
        plt.close(fig)
        return p if self.save_figures else None

    def push(self) -> None:
        if self.pc is None or self.local_root is None:
            return
        try:
            self.pc.push([self.root.resolve().relative_to(self.local_root.resolve()).as_posix()])
        except Exception as e:                           # local copy intact; retried at the next stage
            print(f"[m1] push skipped: {type(e).__name__}: {e}")


def stage_frozen_inputs(m9, ws: Workspace) -> dict:
    """Re-stage exactly the stage-1 staging list (DriveSync skips unchanged files) and verify the local copies exist."""
    man = ws.read_json("staging_manifest.json") or {}
    files = list(man.get("files") or [])
    if not files:
        raise StageBlocked("staging_manifest.json lists no files: run stage 1 (artifact inventory) first")
    m9.stage_in(files)
    local = Path(m9.local)
    miss = [f for f in files if not (local / f).exists()]
    if miss:
        raise StageBlocked(f"{len(miss)} staged input(s) absent locally after staging: {miss[:5]}")
    return {"n_files": len(files), "inventory_fp": man.get("inventory_fp")}


class M1Session:
    def __init__(self, fg, cfg: dict, ws: Workspace):
        self.fg, self.cfg, self.ws = fg, cfg, ws
        self.bst, self.p6 = fg.bst, fg.p6

    # ------------------------------------------------------------------------------- construction
    @classmethod
    def attach(cls, CONFIG: dict, m1_config: dict | None = None, *, require_inventory: bool = True):
        from ..p6 import P6FastTrack
        from ..pipeline import Pipeline

        cfg = load_m1_config(m1_config)
        out = cfg.get("outputs") or {}
        nb_cfg = {"run_mode": "PREPARE_DATA", "failure_policy": "interactive", "download_missing": False,
                  "run_advanced_models": False, "train_baselines": [], "run_full_inference": False,
                  "run_submission": False, "feature_stage": "lazy", "use_gpu": False, "seed": 2026,
                  **{k: CONFIG[k] for k in ("repo", "ref", "drive_root", "local_root") if k in CONFIG}}
        pipe = Pipeline.from_notebook(nb_cfg, {"enabled": False, "mode": "inventory_only", "download": False})
        pipe.step("bootstrap")
        for step in ("raw_data", "native_grid", "spatial_folds", "discovery_splits", "feature_store"):
            pipe.step(step)                                # validate + stage cached data; never rebuilt
        m9 = pipe.m9({"profile": "PILOT", "overwrite_existing": False})
        m9.load()
        p6 = P6FastTrack(m9, {"overwrite_existing": False})
        fg = p6.full_gpu({"mode": "recovery", "output_dir": cfg["output_dir"]}, {},
                         {"geology": {"backfill_missing": False}})
        root = Path(m9.local) / cfg["output_dir"]
        ws = Workspace(root, m9.pc if out.get("sync_to_drive", True) else None, Path(m9.local),
                       save_figures=bool(out.get("save_figures", True)),
                       save_parquet=bool(out.get("save_parquet", True)))
        m9.stage_in([cfg["output_dir"]])                   # earlier M1 outputs (resume)
        s = cls(fg, cfg, ws)
        if require_inventory:
            inv = ws.read_json("artifact_inventory.json") or {}
            if (inv.get("summary") or {}).get("status") != COMPLETE:
                raise StageBlocked("artifact_inventory.json is absent or not COMPLETE: run stage 1 first "
                                   f"(summary: {(inv.get('summary') or {}).get('blocking')})")
        s.staged = stage_frozen_inputs(m9, ws)             # the inventory's staged files only (read-only)
        return s

    @classmethod
    def light(cls, CONFIG: dict, m1_config: dict | None = None,
              subdirs=("submissions", "leaderboard", "manifests", "selection", "validation", "parity", "reports")):
        """Session without the pipeline / P6 engine (leaderboard review, final report): stages only M1 sub-folders."""
        from types import SimpleNamespace

        from ..artifacts.sync import DriveSync

        cfg = load_m1_config(m1_config)
        out = cfg.get("outputs") or {}
        local = Path(CONFIG["local_root"])
        sync = DriveSync(local, Path(CONFIG["drive_root"]))
        sync.stage_in([f"{cfg['output_dir']}/{s}" for s in subdirs if (Path(CONFIG["drive_root"]) / cfg["output_dir"]
                                                                       / s).exists()])
        ws = Workspace(local / cfg["output_dir"], sync if out.get("sync_to_drive", True) else None, local,
                       save_figures=bool(out.get("save_figures", True)),
                       save_parquet=bool(out.get("save_parquet", True)))
        return cls(SimpleNamespace(bst=None, p6=None), cfg, ws)

    # ------------------------------------------------------------------------------- stage records
    def fp(self, *parts) -> str:
        """Stage fingerprint: M1 version + the stage's implementation hash (when tracked) + the given parts. The
        recipe fingerprint (``fp('recipe', ...)``) has no implementation part: the recipe id names the file."""
        from ..pipeline.m6 import fingerprint

        impl = impl_fp(parts[0]) if parts and isinstance(parts[0], str) else None
        return fingerprint(CF.M1_VERSION, *parts, *([impl] if impl else []))

    def inventory_fp(self) -> str | None:
        return ((self.ws.read_json("artifact_inventory.json") or {}).get("summary") or {}).get("fingerprint")

    def manifest(self, stage: str) -> dict:
        return self.ws.read_json(f"manifests/{stage}.json") or {}

    def status(self, stage: str) -> str | None:
        return self.manifest(stage).get("status")

    def require(self, *stages: str) -> None:
        """Every named upstream stage is COMPLETE / WARNING and was produced by the current stage code."""
        bad = {}
        for s in stages:
            m = self.manifest(s)
            if m.get("status") not in PASSING:
                bad[s] = m.get("status") or "not run"
            elif impl_fp(s) is not None and m.get("impl") != impl_fp(s):
                bad[s] = f"{m.get('status')} but produced by different stage code (impl {m.get('impl')} != " \
                         f"{impl_fp(s)}): re-run it"
        if bad:
            raise StageBlocked(f"upstream stages not usable: {bad}")

    def record(self, stage: str, status: str, fp: str, *, reasons=(), inputs: dict | None = None,
               outputs=(), seconds: float | None = None, warnings=(), **extra) -> dict:
        from .._version import provenance

        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        m = {"stage": stage, "status": status, "reasons": list(reasons), "warnings": list(warnings),
             "fingerprint": fp, "impl": impl_fp(stage), "inputs": inputs or {}, "outputs": list(outputs),
             "m1_version": CF.M1_VERSION, "code_rev": CF.M1_CODE_REV, "code": provenance(), "seconds": seconds,
             "written": stamp, **extra}
        self.ws.save_json(m, f"manifests/{stage}.json")
        self.ws.save_json(m, f"manifests/history/{stage}__{stamp.replace(':', '')}.json")    # never replaced
        self.ws.push()
        return m

    def cached(self, stage: str, fp: str, outputs) -> bool:
        m = self.manifest(stage)
        if m.get("fingerprint") != fp or m.get("status") not in PASSING:
            return False
        miss = [o for o in outputs if not self.ws.path(o).exists()]
        if miss:
            print(f"[m1] {stage}: fingerprint matches but {miss} missing -> recomputed")
            return False
        print(f"[m1] {stage}: {CF.SKIP_CACHED} (fingerprint {fp[:10]} unchanged, status {m.get('status')})")
        return True

    def run_stage(self, stage: str, fp: str, outputs, fn, inputs: dict | None = None) -> dict:
        """fn() -> (status, reasons, extra dict[, warnings]). StageBlocked -> INCOMPLETE, StageInvalid -> BLOCKED;
        any other exception is recorded as FAILED (traceback kept) and re-raised."""
        if self.cached(stage, fp, outputs):
            return {**self.manifest(stage), "cache": CF.SKIP_CACHED}
        t0 = time.perf_counter()
        try:
            res = fn()
        except StageBlocked as e:
            return self.record(stage, INCOMPLETE, fp, reasons=[str(e)], inputs=inputs)
        except StageInvalid as e:
            return self.record(stage, BLOCKED, fp, reasons=[str(e)], inputs=inputs)
        except Exception as e:
            self.record(stage, FAILED, fp, reasons=[f"{type(e).__name__}: {e}"], inputs=inputs,
                        traceback=traceback.format_exc()[-3000:])
            raise
        status, reasons, extra = res[:3]
        warns = list(res[3]) if len(res) > 3 else []
        if status == COMPLETE and warns:
            status = WARNING
        return self.record(stage, status, fp, reasons=reasons, inputs=inputs, outputs=outputs, warnings=warns,
                           seconds=round(time.perf_counter() - t0, 1), **(extra or {}))
