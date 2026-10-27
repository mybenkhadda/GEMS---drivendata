"""P6 orchestration: ``P6FastTrack(m9)`` -> one method per workstream, each restartable from persisted outputs.

    m9 = pipe.m9(M9_CONFIG); m9.load()
    p6 = P6FastTrack(m9)
    p6.w0_probes(); p6.w1_seeds(); p6.w2_stack(); p6.w3_throughput(); p6.w4_dem(); p6.w5_completion(); p6.w6_select()
    bst = p6.boost(BOOST_CONFIG); bst.a_benchmark(); bst.b_geology(); ...    # P6 boost (docs/p6/P6_BOOST.md)

Composition, not duplication: the M9 research object provides the context, units, eligible budget, protocol keys,
member maps and the B0/B1 selection (read-only). M7 / M8 writers receive ``view`` (the M7 engine seen through the P6
workspace) so every P6 file lands under ``outputs/p6_fast_track`` and nothing under M9 is rewritten.

Every workstream records its wall time in ``manifests/compute_ledger.csv`` and refuses to start once its cap in
``budget.workstreams`` is spent (``BudgetExceeded``): the fast track never silently exceeds 12 GPU-h / 20 CPU-h.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

from ..m7.workspace import M7Workspace
from ..m8.run import M7View
from ..pipeline.m6 import fingerprint

P6_VERSION = "p6.1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "p6" / "fast_track.yaml"
WORKSTREAMS = ("w0", "w1", "w2", "w3", "w4", "w5", "w6", "wb")


class BudgetExceeded(RuntimeError):
    """A workstream's GPU / CPU cap is spent: it stops instead of growing."""


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_p6_config(config: dict | None = None, path: Path | None = None) -> dict:
    """YAML defaults + overrides, validated (caps sum within the global budget)."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = _merge(yaml.safe_load(f), config or {})
    b = cfg["budget"]
    for dev in ("gpu", "cpu"):
        s = sum(float(w.get(dev, 0.0)) for w in b["workstreams"].values())
        if s > float(b[f"{dev}_hours"]) + 1e-9:
            raise ValueError(f"workstream {dev} caps sum to {s} h > global {b[f'{dev}_hours']} h")
    return cfg


class Budget:
    """Compute ledger (seconds per workstream and device) persisted as CSV; caps from ``budget``."""

    COLUMNS = ["workstream", "device", "seconds", "what", "written"]

    def __init__(self, path: Path, budget_cfg: dict):
        self.path, self.cfg = Path(path), budget_cfg
        self.rows = pd.read_csv(self.path) if self.path.exists() else pd.DataFrame(columns=self.COLUMNS)

    def spent_h(self, ws: str | None = None, device: str = "cpu") -> float:
        r = self.rows
        if not len(r):
            return 0.0
        m = r.device == device
        if ws is not None:
            m &= r.workstream == ws
        return float(r.loc[m, "seconds"].astype(float).sum()) / 3600.0

    def left_h(self, ws: str, device: str) -> float:
        cap = float(self.cfg["workstreams"].get(ws, {}).get(device, 0.0))
        glob = float(self.cfg[f"{device}_hours"]) - self.spent_h(None, device)
        return max(min(cap - self.spent_h(ws, device), glob), 0.0)

    def check(self, ws: str, device: str, need_h: float = 0.0) -> None:
        left = self.left_h(ws, device)
        cap = float(self.cfg["workstreams"].get(ws, {}).get(device, 0.0))
        if cap > 0 and (left <= 0 or need_h > left + 1e-9):
            raise BudgetExceeded(f"{ws}: {device} budget left {left:.2f} h < needed {need_h:.2f} h (cap {cap} h)")

    def add(self, ws: str, device: str, seconds: float, what: str = "") -> None:
        row = {"workstream": ws, "device": device, "seconds": round(float(seconds), 1), "what": what,
               "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.rows = pd.concat([self.rows, pd.DataFrame([row])], ignore_index=True) if len(self.rows) else \
            pd.DataFrame([row], columns=self.COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        from ..artifacts.atomic import atomic_save

        atomic_save(self.path, lambda t: self.rows.to_csv(t, index=False))

    def table(self) -> pd.DataFrame:
        out = []
        for w in WORKSTREAMS:
            for d in ("gpu", "cpu"):
                out.append({"workstream": w, "device": d, "cap_h": float(self.cfg["workstreams"].get(w, {}).get(d, 0.0)),
                            "spent_h": round(self.spent_h(w, d), 3), "left_h": round(self.left_h(w, d), 3)})
        return pd.DataFrame(out)


class P6Workspace(M7Workspace):
    SUBDIRS = ("manifests", "manifests/units", "manifests/sections", "probes", "seeds", "seeds/maps",
               "stack", "stack/features", "stack/models", "stack/maps", "stack/figures", "throughput", "dem",
               "completion", "selection", "validation/metrics", "validation/maps", "reports", "submissions",
               "vectors", "boost")


class _P6Like:
    """The attributes ``M7View`` reads, served by P6 (selection / caches from M9, writes to the P6 workspace)."""

    def __init__(self, p6):
        self._p6 = p6

    @property
    def m7(self):
        return self._p6.m9.m7

    @property
    def ws(self):
        return self._p6.ws

    def baseline_selection(self):
        return self._p6.m9.baseline_selection()

    def dem_mosaic(self):
        return self._p6.m9.dem_mosaic()

    def dem_layer(self):
        return None

    def read_json(self, rel: str):
        return self._p6.m9.read_json(rel)

    def push(self, rels=None):
        return self._p6.push()


class P6FastTrack:
    def __init__(self, m9, config: dict | None = None, *, config_path: Path | None = None):
        self.m9 = m9
        self.cfg = load_p6_config(config, config_path)
        self.ws = P6Workspace(Path(m9.local) / self.cfg["output_dir"], self.cfg)
        self.budget = Budget(self.ws.root / "manifests" / "compute_ledger.csv", self.cfg["budget"])
        self.view = M7View(_P6Like(self))
        self.log: list[dict] = []

    # ------------------------------------------------------------------------------- context
    @property
    def ctx(self):
        return self.m9.ctx

    @property
    def prim(self) -> str:
        return self.m9.prim

    def say(self, msg: str) -> None:
        print(f"[p6] {msg}")

    def m9_path(self, rel: str) -> Path:
        return Path(self.m9.local) / self.cfg["m9_output_dir"] / rel

    def stage_m9(self, rels: list[str]) -> None:
        """Copy M9 outputs from Drive (read-only use; never written back)."""
        self.m9.stage_in([f"{self.cfg['m9_output_dir']}/{r}" for r in rels])

    def push(self) -> None:
        if not self.cfg.get("push_to_drive", True) or self.m9.pc is None:
            return
        try:
            self.m9.pc.push([Path(self.ws.root).resolve().relative_to(Path(self.m9.local).resolve()).as_posix()])
        except Exception as e:
            self.say(f"push skipped: {type(e).__name__}: {e}")

    def selection(self) -> dict:
        self.stage_m9(["baseline/baseline_selection.json"])
        sel = self.m9.baseline_selection()
        if not sel:
            raise FileNotFoundError("M9 baseline/baseline_selection.json missing: run M9 section 01 first")
        return sel

    def fp(self, *parts) -> str:
        return fingerprint(P6_VERSION, *parts)

    # ------------------------------------------------------------------------------- sections
    def section(self, ws: str, device: str, fn, *, what: str = "", need_h: float = 0.0):
        """Run one workstream step: budget check, timing, manifest, push. Failures are recorded, then re-raised."""
        from .._version import provenance

        self.budget.check(ws, device, need_h)
        t0 = time.perf_counter()
        try:
            out = fn()
            status = "done"
        except BudgetExceeded:
            raise
        except Exception as e:
            status = "failed"
            self.log.append({"workstream": ws, "what": what, "status": status, "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-3000:]})
            raise
        finally:
            dt = time.perf_counter() - t0
            self.budget.add(ws, device, dt, what)
            rec = {"workstream": ws, "what": what, "status": status, "seconds": round(dt, 1), "device": device,
                   "config_fp": self.fp(self.cfg), "code": provenance(), "p6_version": P6_VERSION,
                   "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self.ws.save_json(json.loads(json.dumps(rec, default=str)), f"manifests/sections/{ws}_{what or 'run'}.json")
            self.push()
        return out

    # ------------------------------------------------------------------------------- workstreams
    def w0_probes(self) -> pd.DataFrame:
        from . import probes

        return self.section("w0", "cpu", lambda: probes.run(self), what="probes")

    def w0_interpret(self) -> dict:
        from . import probes

        return probes.interpret_log(self)

    def w1_seeds(self, train: bool = False) -> dict:
        from . import seeds

        if train:
            self.section("w1", "gpu", lambda: seeds.train_missing(self), what="train")
        if self.cfg["seeds"].get("tta_rot180"):
            self.section("w1", "gpu", lambda: seeds.tta_all(self), what="tta")
        out = self.section("w1", "cpu", lambda: seeds.evaluate(self), what="evaluate")
        if (out.get("candidate") or {}).get("gate") == "PROMOTE":
            out["full"] = self.section("w1", "cpu", lambda: seeds.full_region(self), what="full_region")
        return out

    def w2_stack(self, full_region: bool = True) -> dict:
        from . import stack

        out = self.section("w2", "cpu", lambda: stack.run(self), what="crossfit")
        if full_region and out.get("gate", {}).get("decision") == "PROMOTE":
            out["full"] = self.section("w2", "cpu", lambda: stack.full_region(self), what="full_region")
        return out

    def w3_throughput(self) -> pd.DataFrame:
        from . import throughput

        return self.section("w3", "gpu", lambda: throughput.run(self), what="profile")

    def w4_dem(self) -> dict:
        from . import dem_pilot

        return self.section("w4", "cpu", lambda: dem_pilot.run(self), what="pilot")

    def w5_completion(self) -> dict:
        from . import completion, probes

        verdict = probes.interpret_log(self)
        if verdict["verdict"] != self.cfg["completion"]["enabled_if"]:
            msg = f"W5 blocked: W0 verdict is {verdict['verdict']!r}, needs {self.cfg['completion']['enabled_if']!r}"
            self.say(msg)
            return {"status": "BLOCKED", "reason": msg}
        return self.section("w5", "cpu", lambda: completion.run(self), what="completion")

    def boost(self, config: dict | None = None):
        """P6 boost (``p6/boost.py``): matched benchmark, G02 ablation, weak-zone robustness, hybrids, selection."""
        from .boost import Boost

        return Boost(self, config)

    def recovery(self, config: dict | None = None, boost_config: dict | None = None):
        """P6.1 recovery (``p6/boost_diagnostics.py``): deployable W2 policies + export parity, candidate recall,
        segment recovery, G01 / G02 repair, weak-zone robustness, selection, validated export (on the boost cap)."""
        from .boost_diagnostics import Recovery

        return Recovery(self.boost(boost_config), config)

    def w6_select(self) -> dict:
        from . import select

        return self.section("w6", "cpu", lambda: select.run(self), what="select")
