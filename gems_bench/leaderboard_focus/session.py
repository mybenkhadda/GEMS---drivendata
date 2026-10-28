"""Shared M1 setup for notebooks 01-04: one call builds the read-only P6 engine on the CPU and the M1 workspace.

    m1 = M1Session.attach(CONFIG, M1_CONFIG)        # pipeline (cached data only) -> M9 -> P6 -> p6g (mode recovery)

* The p6g engine (``p6.full_gpu.FullGPU``) runs in ``mode: recovery`` (no GPU, no training) and its ``root`` IS the M1
  output root, so ``final_submission`` writes its files under ``outputs/leaderboard_focus/m1_champion_recovery``;
  nothing under ``outputs/p6_fast_track`` or ``outputs/p6_boost`` is written.
* ``attach`` refuses to start when notebook 00 has not produced a COMPLETE artifact inventory (or when the inventory
  on Drive changed since: the fingerprint is recomputed lazily by notebook 00 only).
* Stage manifests ``manifests/<stage>.json``: status, reasons, fingerprint, inputs, commit, seconds. ``cached(stage,
  fp, outputs)`` reloads only when the fingerprint matches and every listed output exists.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import pandas as pd

from .config import BLOCKED, COMPLETE, FAILED, INCOMPLETE, M1_CODE_REV, M1_VERSION, load_m1_config


class StageBlocked(RuntimeError):
    """A stage cannot run because an upstream stage is not COMPLETE (recorded as INCOMPLETE)."""


class Workspace:
    """Atomic writes under the M1 root (local) + Drive push of the root (skip unchanged, DriveSync ledger)."""

    def __init__(self, root: Path, pc=None, local_root: Path | None = None):
        self.root, self.pc, self.local_root = Path(root), pc, Path(local_root) if local_root else None

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
        if parquet and len(d):
            from ..pipeline.m6 import _parquet_safe

            try:
                atomic_save(p.with_suffix(".parquet"), lambda t: _parquet_safe(d).to_parquet(t, index=False))
            except Exception as e:                       # optional format
                print(f"[m1] parquet skipped for {rel}: {type(e).__name__}: {e}")
        return p

    def load_table(self, rel: str) -> pd.DataFrame:
        p = self.path(rel)
        if not p.exists() or not p.stat().st_size:
            return pd.DataFrame()
        return pd.read_csv(p, dtype={"systems": str})

    def save_text(self, text: str, rel: str) -> Path:
        from ..artifacts.atomic import atomic_write_text

        return atomic_write_text(self.path(rel), text)

    def save_fig(self, fig, rel_png: str, data: pd.DataFrame | None) -> Path:
        import matplotlib.pyplot as plt

        from ..artifacts.atomic import atomic_save

        p = self.path(rel_png)
        atomic_save(p, lambda t: fig.savefig(t, dpi=160, bbox_inches="tight", format="png"))
        if data is not None:
            atomic_save(p.with_name(p.stem + "_data.csv"), lambda t: data.to_csv(t, index=False))
        plt.close(fig)
        return p

    def push(self) -> None:
        if self.pc is None or self.local_root is None:
            return
        try:
            self.pc.push([self.root.resolve().relative_to(self.local_root.resolve()).as_posix()])
        except Exception as e:                           # local copy intact; retried at the next stage
            print(f"[m1] push skipped: {type(e).__name__}: {e}")


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
        ws = Workspace(root, m9.pc, Path(m9.local))
        m9.stage_in([cfg["output_dir"]])                   # earlier M1 outputs (resume)
        s = cls(fg, cfg, ws)
        if require_inventory:
            inv = ws.read_json("artifact_inventory.json") or {}
            if (inv.get("summary") or {}).get("status") != COMPLETE:
                raise StageBlocked("artifact_inventory.json is absent or not COMPLETE: run notebook 00 first "
                                   f"(summary: {(inv.get('summary') or {}).get('blocking')})")
        need = fg.stage_inputs()                           # W2 / boost / recovery from Drive (read-only)
        s.staged = need
        return s

    @classmethod
    def light(cls, CONFIG: dict, m1_config: dict | None = None, subdirs=("submissions", "leaderboard", "manifests")):
        """Session without the pipeline / P6 engine (notebook 04): stages only the M1 sub-folders it reads."""
        from types import SimpleNamespace

        from ..artifacts.sync import DriveSync

        cfg = load_m1_config(m1_config)
        local = Path(CONFIG["local_root"])
        sync = DriveSync(local, Path(CONFIG["drive_root"]))
        sync.stage_in([f"{cfg['output_dir']}/{s}" for s in subdirs])
        return cls(SimpleNamespace(bst=None, p6=None), cfg, Workspace(local / cfg["output_dir"], sync, local))

    # ------------------------------------------------------------------------------- stage records
    def fp(self, *parts) -> str:
        from ..pipeline.m6 import fingerprint

        return fingerprint(M1_VERSION, *parts)

    def inventory_fp(self) -> str | None:
        return ((self.ws.read_json("artifact_inventory.json") or {}).get("summary") or {}).get("fingerprint")

    def manifest(self, stage: str) -> dict:
        return self.ws.read_json(f"manifests/{stage}.json") or {}

    def status(self, stage: str) -> str | None:
        return self.manifest(stage).get("status")

    def require(self, *stages: str) -> None:
        bad = {s: self.status(s) for s in stages if self.status(s) != COMPLETE}
        if bad:
            raise StageBlocked(f"upstream stages not COMPLETE: {bad} (run the earlier notebooks first)")

    def record(self, stage: str, status: str, fp: str, *, reasons=(), inputs: dict | None = None,
               outputs=(), seconds: float | None = None, **extra) -> dict:
        from .._version import provenance

        m = {"stage": stage, "status": status, "reasons": list(reasons), "fingerprint": fp, "inputs": inputs or {},
             "outputs": list(outputs), "m1_version": M1_VERSION, "code_rev": M1_CODE_REV, "code": provenance(),
             "seconds": seconds, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        self.ws.save_json(m, f"manifests/{stage}.json")
        self.ws.push()
        return m

    def cached(self, stage: str, fp: str, outputs) -> bool:
        m = self.manifest(stage)
        if m.get("fingerprint") != fp or m.get("status") not in (COMPLETE, BLOCKED):
            return False
        miss = [o for o in outputs if not self.ws.path(o).exists()]
        if miss:
            print(f"[m1] {stage}: fingerprint matches but {miss} missing -> recomputed")
            return False
        print(f"[m1] {stage}: fingerprint {fp[:10]} unchanged -> reloaded (status {m.get('status')})")
        return True

    def run_stage(self, stage: str, fp: str, outputs, fn, inputs: dict | None = None) -> dict:
        """fn() -> (status, reasons, extra dict). Exceptions are recorded as FAILED and re-raised; StageBlocked as
        INCOMPLETE."""
        if self.cached(stage, fp, outputs):
            return self.manifest(stage)
        t0 = time.perf_counter()
        try:
            status, reasons, extra = fn()
        except StageBlocked as e:
            return self.record(stage, INCOMPLETE, fp, reasons=[str(e)], inputs=inputs)
        except Exception as e:
            self.record(stage, FAILED, fp, reasons=[f"{type(e).__name__}: {e}"], inputs=inputs,
                        traceback=traceback.format_exc()[-3000:])
            raise
        return self.record(stage, status, fp, reasons=reasons, inputs=inputs, outputs=outputs,
                           seconds=round(time.perf_counter() - t0, 1), **(extra or {}))
