"""M8 orchestration: ``pipe.m8(M8_CONFIG)`` -> one method per notebook section, fingerprinted and resumable.

    m8 = pipe.m8(M8_CONFIG)
    m8.s00_setup(); m8.s01_baseline(); m8.s02_signal_recovery(); ...; m8.run()

Composition, not duplication: an ``M7Research`` engine (read-only) provides the units, the eligible budget domain,
protocol keys, the B0 / B1 member maps and every M7 helper. Its GEMS3 engine is re-pointed to
``outputs/m8_physics_probabilistic/gems3_engine`` so that member-map staging and full-region inference never write
into the M7 tree. M7 helpers that write (held-out maps, DEM tiles, exports, comparison figures) receive ``M7View``:
the M7 engine seen through the M8 workspace. No M7 stage method is ever called.

A stage whose fingerprint (its configuration + upstream stage fingerprints + member-map fingerprints + M8 version)
and outputs match is reused; ``overwrite_existing=True`` recomputes. Every stage writes
``manifests/sections/<stage>.json`` (fingerprint, experiment id, profile, files, Drive root) and pushes to Drive.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..pipeline.m6 import fingerprint
from .config import load_m8_config, m7_overrides
from .workspace import M8_VERSION, M8Workspace


class M7View:
    """The M7 engine with the M8 workspace: writes land in outputs/m8..., reads of selections / DEM go through M8."""

    def __init__(self, m8):
        self._m8 = m8

    def __getattr__(self, k):
        return getattr(self._m8.m7, k)

    @property
    def ws(self):
        return self._m8.ws

    @property
    def cfg(self):
        return self._m8.m7.cfg

    def baseline_selection(self):
        return self._m8.baseline_selection()

    def dem_mosaic(self):
        return self._m8.dem_mosaic()

    def dem_layer(self):
        return self._m8.dem_layer()

    def read_json(self, rel: str):
        return self._m8.read_json(rel)

    def push(self, rels=None):
        return self._m8.push(rels)


class M8Research:
    def __init__(self, pc=None, config: dict | None = None, *, ctx=None, local_root: Path | None = None,
                 data_root: Path | None = None, config_path: Path | None = None):
        from ..gems3.run import GEMS3Research
        from ..m7.run import M7Research

        self.cfg = load_m8_config(config, config_path)
        self.pc = pc
        self.local = Path(local_root or (pc.local if pc is not None else Path.cwd()))
        drive = getattr(pc, "drive", None) if pc is not None else None
        self.cfg["_drive_root"] = str(Path(drive) / self.cfg["output_dir"]) if drive else None
        self.ws = M8Workspace(self.local / self.cfg["output_dir"], self.cfg)
        self.cache = self.local / self.cfg["cache_dir"]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.m7 = M7Research(pc, m7_overrides(self.cfg), ctx=ctx, local_root=local_root, data_root=data_root)
        sub = self.m7.cfg["submission"]
        self.m7.g3 = GEMS3Research(pc, {"models": list(self.m7.cfg["models"]), "tasks": ["geo", "disc_official"],
                                        "primary_task": "disc_official",
                                        "output_dir": f"{self.cfg['output_dir']}/gems3_engine", "cache_dir": "cache/gems3",
                                        "overwrite_existing": False,
                                        "submission": {"source": sub["source"], "device": sub["device"]}},
                                   ctx=ctx, local_root=local_root, data_root=data_root)
        self.m7.cfg["evaluation"]["n_boot"] = int(self.cfg["_profile"]["n_boot"])   # CI precision only (scope)
        self.view = M7View(self)
        self._memo: dict = {}
        self._layers = None
        self.log: list[dict] = []

    # =========================================================================== context helpers
    @property
    def ctx(self):
        return self.m7.ctx

    @property
    def models(self) -> list[str]:
        return self.m7.models

    @property
    def sh(self) -> dict:
        """The shared (M7-schema) configuration: protocol, PU labels, nested grids, promotion rules."""
        return self.m7.cfg

    @property
    def profile(self) -> dict:
        return self.cfg["_profile"]

    @property
    def prim(self) -> str:
        return self.sh["evaluation"]["primary_task"]

    def load(self) -> None:
        self.m7.load()

    def say(self, msg: str) -> None:
        print(f"[m8] {msg}")

    def units_of(self, task: str) -> list:
        return self.m7.units_of(task)

    def unit(self, key: str):
        return self.m7.unit(key)

    def zones(self, task: str) -> list[int]:
        return self.m7.zones(task)

    def full_unit(self):
        return self.m7.full_unit()

    def elig(self, u, ring_px: int = 0) -> np.ndarray:
        return self.m7.elig(u, ring_px)

    def pkey(self, u) -> str:
        return self.m7.pkey(u)

    def cache_get(self, key: str, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    @property
    def layers(self):
        if self._layers is None:
            from .layers import LayerStore

            self.load()
            self._layers = LayerStore(self.ctx, self.cfg, self.cache, compat=self.band_compat())
        return self._layers

    def band_compat(self) -> dict:
        """Derivative-band provenance (computed once, label-free; persisted with the s03 outputs)."""
        j = self.read_json("geophysical_features/band_compat.json")
        if j is not None and not self.cfg.get("overwrite_existing"):
            return j
        from .layers import derivative_audit

        t, comp = derivative_audit(self.ctx, self.cfg, int(self.cfg["seed"]))
        from ..artifacts.atomic import atomic_save, atomic_write_json

        atomic_write_json(self.ws.root / "geophysical_features" / "band_compat.json", comp)
        atomic_save(self.ws.root / "geophysical_features" / "band_provenance.csv", lambda p: t.to_csv(p, index=False))
        return comp

    # -- Drive / files -------------------------------------------------------------
    def stage_in(self, rels: list[str]) -> None:
        if self.pc is not None and rels:
            try:
                self.pc.stage_in([r for r in rels if r])
            except Exception as e:
                self.say(f"stage_in skipped: {type(e).__name__}: {e}")

    def stage_m7(self, subs: list[str]) -> None:
        self.stage_in([f"{self.cfg['m7_output_dir']}/{s}" for s in subs])

    def push(self, rels: list[str] | None = None) -> None:
        if self.pc is None or not self.cfg.get("push_to_drive", True):
            return
        try:
            rels = rels or [Path(self.ws.root).resolve().relative_to(self.local.resolve()).as_posix()]
            self.pc.push(rels)
        except Exception as e:
            self.say(f"push skipped: {type(e).__name__}: {e}")

    def read_json(self, rel: str) -> dict | None:
        p = self.ws.root / rel
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    def m7_json(self, rel: str) -> dict | None:
        p = self.m7.ws.root / rel
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    def m7_experiment_ids(self) -> dict:
        d = self.m7.ws.root / "manifests" / "sections"
        out = {}
        for p in sorted(d.glob("*.json")) if d.exists() else []:
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                out[p.stem] = j.get("experiment_id") or j.get("fingerprint")
            except (OSError, json.JSONDecodeError):
                continue
        return out

    # -- fingerprints / stages -----------------------------------------------------
    def fp(self, *parts) -> str:
        self.load()
        return fingerprint(M8_VERSION, self.m7.g3.m6.map_fp, self.sh["models"], self.sh["evaluation"]["tasks"],
                           self.sh["evaluation"]["budget_domain"], self.sh["evaluation"]["folds"],
                           self.sh["evaluation"]["splits"], *parts)

    def fp_of(self, stage: str) -> str | None:
        r = self.ws.unit_record(stage)
        return r.get("fingerprint") if r else None

    def enabled(self, section: str) -> bool:
        return bool(self.cfg["sections"].get(section))

    def _stage(self, name: str, fp: str, fn, load=None, optional: bool = True):
        from ..m7.run import GateClosed

        from .dem import DemUnavailable

        t0 = time.perf_counter()
        if self.ws.valid(name, fp):
            self.say(f"{name}: reused (fingerprint {fp}) -> {self.ws.root}")
            self.log.append({"stage": name, "outcome": "REUSED", "seconds": 0.0, "fingerprint": fp})
            return load() if load else None
        self.ws.begin(name)
        try:
            out = fn()
        except (GateClosed, DemUnavailable) as e:
            self.say(f"{name} SKIPPED (gate): {e}")
            self.log.append({"stage": name, "outcome": "GATE_CLOSED", "reason": str(e), "seconds": 0.0})
            self._save_log()
            return None
        except Exception as e:
            self.log.append({"stage": name, "outcome": "FAILED", "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-4000:], "seconds": round(time.perf_counter() - t0, 1)})
            self._save_log()
            if not optional:
                raise
            self.say(f"{name} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
            return None
        dt = round(time.perf_counter() - t0, 1)
        self.ws.section_manifest(name, {"fingerprint": fp, "experiment_id": self.ws.experiment_id(name, fp),
                                        "seconds": dt, "profile_scope": self.profile,
                                        "layer_log": self._layers.log[-50:] if self._layers is not None else []})
        self.ws.finish(name, fp, seconds=dt)
        self.log.append({"stage": name, "outcome": "COMPUTED", "seconds": dt, "fingerprint": fp})
        self.say(f"{name}: computed in {dt:.0f}s -> {self.ws.root}")
        self._save_log()
        self.push()
        return out

    def _save_log(self) -> None:
        from ..artifacts.atomic import atomic_write_json

        atomic_write_json(self.ws.root / "manifests" / "stage_log.json", json.loads(json.dumps(self.log, default=str)))

    def gate(self, name: str, sub: str) -> str | None:
        t = self.ws.load_table(name, sub)
        if not len(t):
            return None
        t = t[t.task == self.prim] if "task" in t and (t.task == self.prim).any() else t
        return str(t.decision.iloc[0])

    def _require_coverage(self) -> None:
        from ..m7.run import GateClosed

        g = self.gate("gate2", "candidate_generation/oracle")
        if g == "GO" or self.cfg["ranking"].get("force"):
            return
        if self.cfg.get("stop_ranking_if_coverage_inadequate", True):
            raise GateClosed(f"Gate 2 (candidate coverage) is {g}: ranking is not justified (read "
                             "candidate_generation/oracle/gate2.csv; ranking.force=True overrides)")

    # =========================================================================== shared loaders
    def set_selection(self, sel: dict) -> None:
        self._memo["selection"] = sel

    def baseline_selection(self) -> dict | None:
        if "selection" not in self._memo:
            self._memo["selection"] = self.read_json("baseline/baseline_selection.json")
        return self._memo["selection"]

    def system_table(self) -> pd.DataFrame:
        return self.ws.load_table("system_table", "baseline")

    def attributes(self) -> pd.DataFrame:
        return self.ws.load_table("system_attributes", "baseline")

    def m7_pool(self, key: str):
        k = ("m7pool", key)
        if k not in self._memo:
            try:
                self._memo[k] = self.m7.load_unit_set(key)
            except Exception:
                self._memo[k] = None
        return self._memo[k]

    def _load_set(self, sub: str, k: str):
        from ..gems3.candidates import CandidateSet

        kk = k.replace(":", "_")
        z = self.ws.root / sub / f"{kk}.npz"
        t = self.ws.load_table(kk, sub)
        if not z.exists() or not len(t):
            return None
        return CandidateSet.from_npz(t.sort_values("cand_id").reset_index(drop=True) if "cand_id" in t else t, np.load(z))

    def _save_set(self, cs, sub: str, k: str) -> None:
        kk = k.replace(":", "_")
        self.ws.save_npz(cs.to_npz(), f"{sub}/{kk}.npz")
        self.ws.save_table(cs.table, kk, sub)

    def pool6(self, key: str):
        """Section-06 pool of one unit (crop coordinates)."""
        k = ("pool6", key)
        if k not in self._memo:
            self._memo[k] = self._load_set("candidate_generation/units", key)
        return self._memo[k]

    def pool(self, key: str):
        """Final pool = section-06 pool + section-07 MC-only structures (cand_id continues)."""
        from ..gems3.candidates import CandidateSet

        k = ("pool", key)
        if k not in self._memo:
            base = self.pool6(key)
            ext = self._load_set("monte_carlo/extras", key)
            if base is not None and ext is not None and len(ext):
                t = ext.table.copy()
                t["cand_id"] = np.arange(len(base), len(base) + len(ext))
                t["families"], t["n_families"] = "mc_occupancy", 1
                if "structural_score" not in t:
                    t["structural_score"] = pd.Series(t.conf_lf).rank(pct=True).to_numpy() if "conf_lf" in t else 0.0
                from ..m7.generator import geometry

                g = geometry(ext)
                t = pd.concat([t.reset_index(drop=True), g[[c for c in g.columns if c not in t]]], axis=1)
                base = CandidateSet(pd.concat([base.table, t], ignore_index=True), base.paths + ext.paths)
            self._memo[k] = base
        return self._memo[k]

    def completion_set(self, key: str):
        cs = self._load_set("fault_networks/units", key)
        if cs is None:
            try:
                cs = self.m7.completion_sets().get(key)            # M7 deterministic hypotheses (read-only)
            except Exception:
                cs = None
        return cs

    def completion_occupancy(self, key: str) -> np.ndarray | None:
        p = self.ws.root / "fault_networks" / "units" / f"{key.replace(':', '_')}_occupancy.npz"
        return np.load(p)["occ"].astype(np.float32) if p.exists() else None

    def unit_profiles(self, key: str) -> pd.DataFrame | None:
        t = self.ws.load_table(key.replace(":", "_"), "fault_normal_profiles/units")
        return t if len(t) else None

    def mc_unit(self, key: str) -> pd.DataFrame | None:
        t = self.cache_get("mc_stability", lambda: self.ws.load_table("stability", "monte_carlo/stability"))
        return t[t.unit == key] if len(t) else None

    def mc_occupancy(self) -> np.ndarray | None:
        def build():
            p = self.ws.root / "monte_carlo" / "uncertainty_maps" / "occupancy_fraction.npz"
            if not p.exists():
                return None
            z = np.load(p)
            return np.where(z["covered"], z["occ"].astype(np.float32), np.nan).astype(np.float32)
        return self.cache_get("mc_occupancy", build)

    # -- DEM -------------------------------------------------------------------------
    def dem_mosaic(self):
        from ..m7.dem_morph import DemMosaic

        if "dem_mosaic" in self._memo:
            return self._memo["dem_mosaic"]
        mine = self.ws.root / "dem" / "features" / "m7_dem_features_100m.tif"
        theirs = self.m7.ws.root / "dem" / "features" / "m7_dem_features_100m.tif"
        if not theirs.exists():
            self.stage_in([theirs.relative_to(self.local).as_posix()])
        p = mine if mine.exists() else (theirs if theirs.exists() else None)
        self._memo["dem_mosaic"] = DemMosaic(p) if p is not None else None
        return self._memo["dem_mosaic"]

    def forget_dem(self) -> None:
        for k in ("dem_mosaic", "dem_layer", "dem_full"):
            self._memo.pop(k, None)

    def dem_layer(self) -> np.ndarray | None:
        def build():
            mos = self.dem_mosaic()
            if mos is None:
                return None
            b = next((x for x in self.sh["dem"]["candidate_layers"] if mos.has(x)), None)
            if b is None:
                return None
            a = mos.band(b)
            cov = mos.coverage(self.sh["dem"]["min_coverage"])
            v = np.sort(a[cov & np.isfinite(a)])
            if not v.size:
                return None
            return np.where(cov & np.isfinite(a), np.searchsorted(v, a) / v.size, np.nan).astype(np.float32)
        return self.cache_get("dem_layer", build)

    def set_dem_candidates(self, cs) -> None:
        self._memo["dem_full"] = cs
        if cs is not None and len(cs):
            self._save_set(cs.subset(range(len(cs))) if "cand_id" in cs.table else _with_ids(cs), "dem/candidates",
                           "dem_candidates_full")

    def dem_candidates_full(self):
        def build():
            cs = self._load_set("dem/candidates", "dem_candidates_full")
            if cs is not None:
                return cs
            mos = self.dem_mosaic()
            if mos is None or not self.cfg["generation"].get("include_dem"):
                return None
            from ..m7.dem_value import dem_candidates

            covered = mos.coverage(self.sh["dem"]["min_coverage"]) & np.asarray(self.ctx.data.valid, bool)
            cs, _ = dem_candidates(self.view, mos, covered)
            return _with_ids(cs)
        return self.cache_get("dem_full", build)

    # -- full region -----------------------------------------------------------------
    def recover_full_maps(self) -> pd.DataFrame:
        """Copy full-region member maps cached by M7's engine into M8's engine when their fingerprints still match
        (read-only on the M7 side; GEMS3 legacy maps are handled by ``M7Research.recover_legacy_full_maps``)."""
        import shutil

        from ..gems3.submission import _fp

        src = self.m7.ws.root / "gems3_engine" / "predictions"
        self.stage_m7(["gems3_engine/predictions"])
        dst = self.m7.g3.ws.root / "predictions"
        rows = []
        try:
            members = self.m7.g3.members()
        except Exception as e:
            return pd.DataFrame([{"status": f"members unavailable: {e}"}])
        source = self.sh["submission"]["source"]
        for mem in members:
            t, j = src / f"{mem['model']}__full_region_raw.tif", src / f"{mem['model']}__full_region_raw.json"
            st = "absent"
            if t.exists() and j.exists():
                meta = json.loads(j.read_text(encoding="utf-8"))
                if meta.get("fingerprint") == _fp(mem, source) and meta.get("source") == source:
                    dst.mkdir(parents=True, exist_ok=True)
                    if not (dst / t.name).exists():
                        shutil.copy2(t, dst / t.name)
                        shutil.copy2(j, dst / j.name)
                    st = "recovered from M7 engine"
                else:
                    st = "fingerprint or source differs (not reused)"
            rows.append({"model": mem["model"], "m7_map": str(t), "status": st})
        return pd.DataFrame(rows)

    def full_member_maps(self) -> list:
        self.recover_full_maps()
        fm = self.m7.cached_full_maps()
        if fm is not None:
            return fm
        if self.sh["submission"]["source"] == "fold_average":
            return self.m7.full_maps()[0]
        return [np.asarray(self.ctx.oof(m), np.float32) for m in self.models]

    def build_full_pool(self):
        from . import candidates as C

        lf, _, _ = C.cached_label_free(self)
        fm = self.full_member_maps()
        fu = self.full_unit()
        scores = {self.sh["models"][i]: fm[i] for i in range(2)}
        cs, _ = C.unit_pool(self, fu, lf, scores, [C.dem_extra(self, fu)])
        self._save_set(cs, "candidate_generation/units", "full:0")
        self._memo[("pool6", "full:0")] = cs
        self._memo.pop(("pool", "full:0"), None)
        return cs

    def full_candidate_raster(self) -> np.ndarray | None:
        def build():
            H, W = self.ctx.data.shape
            out = np.zeros((H, W), np.uint8)
            any_ = False
            for t in ("geo", self.prim):
                for u in self.units_of(t):
                    cs = self.pool6(u.key) or self.m7_pool(u.key)
                    if cs is None or not len(cs):
                        continue
                    out[u.sl] = np.maximum(out[u.sl], (cs.raster(u.ev.shape) > 0).astype(np.uint8))
                    any_ = True
            return out if any_ else None
        return self.cache_get("full_cand_raster", build)

    def full_completion_score(self) -> np.ndarray:
        from . import completion as CO
        from .ensemble import _paint_rank

        valid = np.asarray(self.ctx.data.valid, bool)
        ev, strike, _ = CO.evidence(self, valid)
        lab = np.asarray(self.ctx.data.labels, bool)
        cs, _, _ = CO.stochastic(self, lab, valid, ev, strike, prefix="full/", seed=int(self.cfg["seed"]))
        if not len(cs):
            return np.zeros(valid.shape, np.float32)
        return _paint_rank(cs, CO.completion_score(cs.table), valid)

    # =========================================================================== S00
    def s00_setup(self) -> dict:
        """Environment, inventory, compatibility (read-only; always re-run)."""
        from .._version import provenance
        from . import inventory as I

        self.load()
        d = self.ctx.data
        checks = {"crs_epsg": d.grid.crs.to_epsg() if hasattr(d.grid.crs, "to_epsg") else str(d.grid.crs),
                  "shape": list(d.shape), "template_same_grid": bool(d.template_matches()), "n_bands": len(d.band_names),
                  "bands": list(d.band_names), "profile": self.profile, "m7_outputs": str(self.m7.ws.root)}
        fails = []
        if checks["crs_epsg"] != 32611:
            fails.append(f"CRS {checks['crs_epsg']} != EPSG:32611")
        if not checks["template_same_grid"]:
            fails.append("example_submission.tif grid differs from the feature grid")
        need = {s.get("band") for s in self.cfg["generation"]["families"] if s.get("band")} | \
               {f["field"] for f in self.cfg["bands"]["families"].values()} | set(self.cfg["bands"]["context"])
        bands = pd.DataFrame([{"band": b, "available": b in d.band_names, "description": d.band_desc.get(b, "")}
                              for b in sorted(x for x in need if x)])
        self.ws.begin("s00_setup")
        self.ws.save_table(I.table(), "integration_inventory", "manifests")
        self.ws.save_table(bands, "band_availability", "manifests")
        self.ws.save_table(self.m7.protocol_keys(), "protocol_keys", "manifests")
        self.ws.save_json({"checks": checks, "failures": fails, "code": provenance(), "m8_version": M8_VERSION,
                           "map_fingerprints": self.m7.g3.m6.map_fp, "models": self.models,
                           "config": json.loads(json.dumps(self.cfg, default=str)),
                           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, "manifests/setup.json")
        self.ws.save_json(json.loads(json.dumps(self.cfg, default=str)), "manifests/m8_config.json")
        self.ws.section_manifest("s00_setup", {"failures": fails})
        self.ws.finish("s00_setup", fingerprint(time.time()))
        self.push()
        if fails:
            raise RuntimeError(f"setup checks failed: {fails}")
        return {"checks": checks, "bands": bands, "inventory": I.table()}

    # =========================================================================== S01
    def s01_baseline(self) -> dict | None:
        from . import baseline as B

        def fn():
            out = B.run(self)
            sub = "baseline"
            fp = self.fp("s01", self.cfg["baseline"], self.sh["baseline"])
            self.ws.save_immutable_table(out["table"], "baseline_table", sub, fp)
            for k, name in (("per_zone", "baseline_per_zone"), ("ledger", "area_ledger"), ("retained_fp", "retained_fp_rescore"),
                            ("recorded", "recorded_vs_this_run"), ("compatibility", "compatibility"),
                            ("systems", "system_table"), ("attributes", "system_attributes"),
                            ("saved", "heldout_maps_saved"), ("m7_pool_summary", "m7_pool_summary")):
                self.ws.save_table(out[k], name, sub)
            self.ws.save_table(out["m7_gate"], "m7_gate_decision", f"{sub}/recovered_m7")
            self.ws.save_json(json.loads(json.dumps(out["selection"], default=float)), f"{sub}/baseline_selection.json")
            self.ws.save_json(json.loads(json.dumps(out["manifest"], default=str)), f"{sub}/baseline_manifest.json")
            from .figures import bar_panels16

            t = out["table"]
            if len(t):
                d = t[["task", "method", "dti_crossfit", "fold_min"]].copy()
                self.ws.save_fig(bar_panels16(d, "method", "dti_crossfit", "task", ref_col="fold_min",
                                              title="Section 01 - baselines under the shared protocol (tick = worst zone)"),
                                 "baseline_dti", d, sub)
            return out

        def load():
            return {"table": self.ws.load_immutable_table("baseline_table", "baseline"),
                    "recorded": self.ws.load_table("recorded_vs_this_run", "baseline"),
                    "compatibility": self.ws.load_table("compatibility", "baseline"),
                    "selection": self.baseline_selection()}
        return self._stage("s01_baseline", self.fp("s01", self.cfg["baseline"], self.sh["baseline"],
                                                   self.cfg["recorded_m7"]), fn, load, optional=False)

    # =========================================================================== S02
    def s02_signal_recovery(self) -> dict | None:
        from . import signal_recovery as SR
        from .figures import export_cases, stage_funnel

        def fn():
            out = SR.run(self)
            sub = "signal_recovery"
            for k, name in (("long", "stage_long"), ("wide", "stage_matrix"), ("reasons", "reasons"),
                            ("by_stratum", "stage_recall"), ("missed_by_both", "missed_by_both_by_stage"),
                            ("density", "stage_density"), ("summary", "summary"), ("cases", "case_candidates"),
                            ("regional", "regional_flags")):
                self.ws.save_table(out[k], name, sub)
            fig, d = stage_funnel(out["by_stratum"])
            self.ws.save_fig(fig, "stage_funnel", d, sub)
            export_cases(self, out["cases"], f"{sub}/case_studies", int(self.profile["case_studies_per_class"]))
            return out

        def load():
            return {k: self.ws.load_table(n, "signal_recovery") for k, n in
                    (("reasons", "reasons"), ("summary", "summary"), ("by_stratum", "stage_recall"))}
        fp = self.fp("s02", self.cfg["signal_recovery"], self.profile["signal_recovery_tasks"], self.cfg["physics"],
                     self.sh["candidates"], self.fp_of("s01_baseline"), self.layers.compat.get("status"))
        return self._stage("s02_signal_recovery", fp, fn, load)

    # =========================================================================== S03
    def s03_potential_fields(self) -> dict | None:
        from . import potential_fields as PF
        from .layers import parse, resolution_table

        def fn():
            L = self.layers
            sub = "geophysical_features"
            res = resolution_table(L)
            self.ws.save_table(res, "effective_resolution", sub)
            rows = []
            for fam, ops in self.cfg["physics"]["operators"].items():
                for op in ops:
                    ok, why = L.allowed(fam, op)
                    for s in L.scales(fam):
                        rows.append({"family": fam, "operator": op, "scale_m": s, "allowed": ok, "reason": why,
                                     "definition": PF.OPERATOR_INFO.get(op, ""), "layer": L.name(fam, op, s),
                                     "units": (self.cfg["bands"].get("units") or {}).get(L.families[fam]["field"],
                                                                                        "unknown (not verified)")})
            self.ws.save_table(pd.DataFrame(rows), "operator_inventory", sub)
            written = []
            for nm in self.cfg["physics"]["save_layers"]:
                p = parse(nm)
                if p is None or not L.allowed(p[0], p[1])[0]:
                    written.append({"layer": nm, "status": "refused or invalid"})
                    continue
                a = np.asarray(L.get(nm), np.float32)
                self.ws.save_raster({p[1]: a}, f"{sub}/{p[0]}/{p[1]}_{int(p[2])}m.tif", self.ctx.data.grid)
                v = a[np.isfinite(a)]
                written.append({"layer": nm, "status": "ok", "defined_share_valid": float(
                    np.isfinite(a[np.asarray(self.ctx.data.valid, bool)]).mean()),
                    "p50": float(np.median(v)) if v.size else np.nan, "p99": float(np.quantile(v, 0.99)) if v.size else np.nan})
                del a
            self.ws.save_table(pd.DataFrame(written), "saved_layers", sub)
            bp = self.ws.root / sub / "band_provenance.csv"
            bt = pd.read_csv(bp) if bp.exists() else pd.DataFrame()
            self.ws.save_table(bt, "band_provenance", sub)
            self.ws.save_json(L.compat, f"{sub}/band_compat.json")
            self.ws.save_table(pd.DataFrame(L.log), "layer_compute_log", sub)
            L.release()
            return {"resolution": res, "operators": pd.DataFrame(rows), "bands": bt}

        def load():
            sub = "geophysical_features"
            return {"resolution": self.ws.load_table("effective_resolution", sub),
                    "operators": self.ws.load_table("operator_inventory", sub),
                    "bands": self.ws.load_table("band_provenance", sub)}
        return self._stage("s03_potential_fields", self.fp("s03", self.cfg["physics"], self.cfg["bands"]), fn, load)

    def s03b_structural_worms(self) -> dict | None:
        from . import worms as WM

        def fn():
            out = WM.run(self)
            sub = "geophysical_features/structural_worms"
            self.ws.save_table(out["log"], "worm_levels", sub)
            self.ws.save_table(out["summary"], "worm_summary", sub)
            cov = WM.coverage_vs_fixed_scale(self)
            self.ws.save_table(cov, "coverage_vs_fixed_scale", sub)
            from .figures import bar_panels16

            if len(cov):
                d = cov.groupby(["family", "task"])[["recall_persistent_worms", "recall_all_base_worms"]].mean().reset_index()
                m = d.melt(id_vars=["family", "task"], var_name="set", value_name="system_recall")
                m["cat"] = m.family + " | " + m.set
                self.ws.save_fig(bar_panels16(m, "cat", "system_recall", "task", xlabel="withheld-system recall (1-px)",
                                              title="Section 03b - persistent worms vs fixed-scale crests (evaluation)"),
                                 "worms_vs_fixed_scale", m, sub)
            return out

        def load():
            return {"summary": self.ws.load_table("worm_summary", "geophysical_features/structural_worms")}
        return self._stage("s03b_structural_worms", self.fp("s03b", self.cfg["worms"], self.cfg["bands"]["families"]),
                           fn, load)

    def s04_multimodal(self) -> dict | None:
        from . import multimodal as MM

        def fn():
            out = MM.run(self)
            sub = "geophysical_features/multimodal"
            self.ws.save_table(out["mag_grav"], "mag_grav_agreement", sub)
            self.ws.save_table(out["surface_subsurface"], "surface_subsurface", sub)
            self.ws.save_table(out["regions"], "classes_by_region", sub)
            self.ws.save_table(out["context"], "context_layers", "geophysical_features/context")
            return out

        def load():
            return {"mag_grav": self.ws.load_table("mag_grav_agreement", "geophysical_features/multimodal")}
        return self._stage("s04_multimodal", self.fp("s04", self.cfg["multimodal"], self.cfg["bands"]["context"],
                                                     self.fp_of("s03b_structural_worms")), fn, load)

    # =========================================================================== S05
    def s05_fault_normal_profiles(self) -> dict | None:
        from . import profiles as PR
        from .figures import profile_gallery

        def fn():
            sub = "fault_normal_profiles"
            units = [u for t in ("geo", self.prim) for u in self.units_of(t)]
            pools, src = {}, {}
            for u in units:
                cs = self.pool6(u.key)
                src[u.key] = "m8" if cs is not None else "m7"
                pools[u.key] = cs if cs is not None else self.m7_pool(u.key)
            kept, n_keep = [], int(self.profile["persist_transects_max"])
            for u in units:
                cs = pools.get(u.key)
                if cs is None or not len(cs):
                    continue
                df, keep = PR.candidate_profiles(self, u, cs, u.ev, max(n_keep - sum(len(k) for k in kept), 0))
                if "uid" in cs.table:
                    df["uid"] = cs.table.uid.astype(str).to_numpy()[:len(df)]
                df["pool_source"] = src[u.key]
                self.ws.save_table(df, u.key.replace(":", "_"), f"{sub}/units")
                kept.append(keep)
                if keep:
                    self.ws.save_table(PR.transects_long(keep, u.key), f"transects_{u.key.replace(':', '_')}", sub)
                self.say(f"s05 {u.key}: {len(df)} candidates profiled ({src[u.key]} pool)")
            P, feats = PR.gallery(self, [u for u in self.units_of(self.prim)], pools)
            self.ws.save_table(P, "gallery_profiles", f"{sub}/gallery")
            self.ws.save_table(feats, "gallery_candidate_features", f"{sub}/gallery")
            self.ws.save_fig(profile_gallery(P), "gallery_contrast_vs_offset", P, f"{sub}/gallery")
            return {"gallery": P, "pool_source": src}

        def load():
            return {"gallery": self.ws.load_table("gallery_profiles", "fault_normal_profiles/gallery")}
        fp = self.fp("s05", self.cfg["profiles_fn"], self.profile["gallery_per_group"], self.profile["persist_transects_max"],
                     self.fp_of("s06_candidates") or "m7_pool")
        return self._stage("s05_fault_normal_profiles", fp, fn, load)

    # =========================================================================== S06
    def s06_candidates(self) -> dict | None:
        from . import candidates as C
        from . import signal_recovery as SR
        from .figures import recall_density

        if self.cfg["generation"].get("include_completion") and self.enabled("s08_monte_carlo_completion"):
            self.s08a_completion_hypotheses()            # its own stage (never nested inside another stage)

        def fn():
            lf, log, key = C.cached_label_free(self)
            sub = "candidate_generation"
            self.ws.save_table(log, "generator_log", sub)
            self.recover_full_maps()
            fm = self.m7.cached_full_maps()
            sets, inv, mlog = {}, [], []
            units = [u for t in ("geo", self.prim) for u in self.units_of(t)]
            if self.profile["full_region_candidates"]:
                units.append(self.full_unit())
            for u in units:
                scores = self.m7._member_scores(u, fm)
                extra = []
                if self.cfg["generation"].get("include_completion") and u.task != "geo" and u.task != "full":
                    extra.append(C.completion_extra(self, u))
                if self.cfg["generation"].get("include_dem"):
                    extra.append(C.dem_extra(self, u))
                cs, lg = C.unit_pool(self, u, lf, scores, extra)
                self._save_set(cs, f"{sub}/units", u.key)
                self._memo[("pool6", u.key)] = cs
                self._memo.pop(("pool", u.key), None)
                sets[u.key] = cs
                inv.append(cs.table.assign(task=u.task, zone=u.zone, unit=u.key))
                mlog.append(lg)
                self.say(f"s06 {u.key}: {len(cs)} fused candidates")
            invt = pd.concat(inv, ignore_index=True) if inv else pd.DataFrame()
            self.ws.save_table(invt, "candidate_inventory", sub)
            self.ws.save_table(pd.concat(mlog, ignore_index=True) if mlog else pd.DataFrame(), "unit_family_log", sub)
            self._save_vectors(sets)
            res = C.oracle(self, {k: v for k, v in sets.items() if k != "full:0"})
            osub = f"{sub}/oracle"
            for k in ("summary", "curve", "oracle", "by_group", "systems", "low_conf", "pooled_curve", "pooled_summary",
                      "pooled_oracle", "pooled_groups", "incremental"):
                self.ws.save_table(res[k], k, osub)
            self.ws.save_table(res["gate"], "gate2", osub)
            fig, d = recall_density(res["pooled_curve"])
            self.ws.save_fig(fig, "recall_density_m8_vs_m7", d, osub)
            # Gate 1: M8 physics stages vs the M7 chain (section 02 matrix)
            W = self.ws.load_table("stage_matrix", "signal_recovery")
            if len(W):
                M8 = SR.m8_stage_rows(self, sorted(W.task.unique()), sets)
                self.ws.save_table(M8, "m8_stage_rows", "signal_recovery")
                self.ws.save_table(SR.gate1(W, M8, self.cfg["gates"]["g1_signal_recovery"])
                                   .assign(profile=self.profile["name"], provisional=self.profile["provisional"]),
                                   "gate1", "signal_recovery")
            self.ws.save_json({"label_free_cache_key": key, "n_label_free": len(lf)}, f"{sub}/cache.json")
            return {"sets": sets, "gate": res["gate"]}

        def load():
            return {"gate": self.ws.load_table("gate2", "candidate_generation/oracle"),
                    "pooled_curve": self.ws.load_table("pooled_curve", "candidate_generation/oracle")}
        fp = self.fp("s06", self.cfg["generation"], self.cfg["oracle"], C.label_free_key(self),
                     self.profile["full_region_candidates"], self.fp_of("s08a_completion_hypotheses"),
                     self.fp_of("s09_dem_pilot"), self.fp_of("s02_signal_recovery"), self.fp_of("s01_baseline"))
        return self._stage("s06_candidates", fp, fn, load, optional=False)

    def _save_vectors(self, sets: dict) -> None:
        from ..gems3.candidates import to_full

        paths, rows = [], []
        for k, cs in sets.items():
            u = self.unit(k)
            for i, p in enumerate(cs.paths):
                paths.append(to_full(p, u) if k != "full:0" else p)
                r = cs.table.iloc[i]
                rows.append({"unit": k, "uid": r.get("uid"), "source": r.get("source"), "families": r.get("families"),
                             "modality": r.get("modality"), "length_km": r.get("length_km"),
                             "structural_score": r.get("structural_score"), "n_families": r.get("n_families")})
        if paths:
            self.ws.save_lines(paths, pd.DataFrame(rows), "candidate_generation/candidates.gpkg", self.ctx.data.grid,
                               "m8_candidates")

    # =========================================================================== S07
    def s07_monte_carlo_signal(self) -> dict | None:
        from . import montecarlo as MC
        from .figures import occupancy_map

        def fn():
            pools = {u.key: self.pool6(u.key) for t in ("geo", self.prim) for u in self.units_of(t)}
            if self.profile["full_region_candidates"] and self.pool6("full:0") is not None:
                pools["full:0"] = self.pool6("full:0")
            pools = {k: v for k, v in pools.items() if v is not None}
            if not pools:
                raise RuntimeError("no section-06 pool: run section 06 first")
            out = MC.run(self, pools)
            sub = "monte_carlo"
            self.ws.save_table(out["stability"], "stability", f"{sub}/stability")
            self.ws.save_table(out["manifest"], "realization_manifest", f"{sub}/realizations")
            self.ws.save_table(out["regions"], "regions", f"{sub}/realizations")
            self.ws.save_json({"distributions": out["distributions"], "uncertainty_type": out["uncertainty_type"],
                               "n_realizations": out["n_real"], "profile": self.profile["name"],
                               "note": "detection frequency = processing stability, not a calibrated fault probability"},
                              f"{sub}/realizations/assumptions.json")
            R = max(int(out["n_real"]), 1)
            occf = out["occupancy"].astype(np.float32) / R
            self.ws.save_npz({"occ": occf.astype(np.float16), "covered": out["covered"]},
                             f"{sub}/uncertainty_maps/occupancy_fraction.npz")
            lay = {"occupancy_all": np.where(out["covered"], occf, np.nan)}
            for f, a in out["occupancy_family"].items():
                lay[f"occupancy_{f}"] = np.where(out["covered"], a.astype(np.float32) / R, np.nan)
            self.ws.save_raster(lay, f"{sub}/uncertainty_maps/occupancy.tif", self.ctx.data.grid)
            self._memo.pop("mc_occupancy", None)
            self._memo.pop("mc_stability", None)
            ext = MC.extras(self, out, pools)
            for k, cs in ext.items():
                if len(cs):
                    self._save_set(_with_ids(cs), f"{sub}/extras", k)
                self._memo.pop(("pool", k), None)
            ev = MC.evaluate(self, pools, out["stability"], ext)
            self.ws.save_table(ev["stability_vs_truth"], "stability_vs_truth", f"{sub}/stability")
            self.ws.save_table(ev["unique"], "mc_only_coverage", f"{sub}/stability")
            self.ws.save_table(MC.gate3(self, ev["unique"]), "gate3", f"{sub}/stability")
            self.ws.save_fig(occupancy_map(self, occf, out["covered"], f"Monte Carlo occupancy ({R} realisations, "
                                                                     f"{self.profile['name']})"),
                             "occupancy_atlas", None, f"{sub}/uncertainty_maps")
            return {"stability": out["stability"], "evaluation": ev}

        def load():
            return {"stability": self.ws.load_table("stability", "monte_carlo/stability"),
                    "gate": self.ws.load_table("gate3", "monte_carlo/stability")}
        fp = self.fp("s07", self.cfg["monte_carlo"], self.profile["mc_realizations"], self.profile["mc_window"],
                     self.cfg["generation"], self.fp_of("s06_candidates"))
        return self._stage("s07_monte_carlo_signal", fp, fn, load)

    # =========================================================================== S08
    def _evidence(self):
        from . import completion as CO

        return self.cache_get("completion_evidence", lambda: CO.evidence(self, np.asarray(self.ctx.data.valid, bool)))

    def s08a_completion_hypotheses(self) -> dict | None:
        from . import completion as CO

        def fn():
            ev, strike, used = self._evidence()
            out = CO.disc_units(self, ev, strike)
            sub = "fault_networks"
            for k, cs in out["sets"].items():
                self._save_set(cs, f"{sub}/units", k)
                self.ws.save_npz({"occ": out["occupancy"][k].astype(np.float16)},
                                 f"{sub}/units/{k.replace(':', '_')}_occupancy.npz")
            self.ws.save_json({"evidence_layers": used}, f"{sub}/evidence_layers.json")
            self.ws.save_table(CO.coverage_audit(self, out["sets"]), "coverage_audit", sub)
            return out

        def load():
            return {"coverage": self.ws.load_table("coverage_audit", "fault_networks")}
        fp = self.fp("s08a", self.cfg["completion"], self.profile["completion_samples"], self.sh["map_completion"],
                     self.fp_of("s03_potential_fields"), self.fp_of("s09_dem_pilot"))
        return self._stage("s08a_completion_hypotheses", fp, fn, load)

    def s08_monte_carlo_completion(self) -> dict | None:
        """Hypotheses (08a, reused when already built for section 06), System C and the corruption benchmark."""
        from . import completion as CO

        self.s08a_completion_hypotheses()

        def fn():
            sets = {u.key: self.completion_set(u.key) for u in self.units_of(self.prim)}
            sets = {k: v for k, v in sets.items() if v is not None and "completion_score" in v.table}
            sub = "fault_networks"
            sc = CO.system_c(self, sets)
            self.ws.save_table(sc["curve"], "system_c_curve", sub)
            self.ws.save_table(sc["summary"], "system_c_summary", sub)
            ev, strike, _ = self._evidence()
            bm = CO.benchmark(self, ev, strike)
            self.ws.save_table(bm["protocol"], "benchmark_protocol", f"{sub}/benchmark")
            self.ws.save_table(bm["results"], "benchmark_results", f"{sub}/benchmark")
            self.ws.save_table(bm["selection"], "benchmark_selection", f"{sub}/benchmark")
            from .figures import bar_panels16

            s = bm["selection"]
            if len(s):
                self.ws.save_fig(bar_panels16(s.assign(cat=s.completer), "cat", "dti_crossfit", "class",
                                              title="Section 08 - stochastic vs deterministic completion on damaged maps"),
                                 "completion_benchmark", s, f"{sub}/benchmark")
            return {"system_c": sc["summary"], "benchmark": s}

        def load():
            return {"system_c": self.ws.load_table("system_c_summary", "fault_networks"),
                    "benchmark": self.ws.load_table("benchmark_selection", "fault_networks/benchmark")}
        fp = self.fp("s08b", self.cfg["completion"], self.sh["map_completion"], self.profile["corruption_realisations"],
                     self.profile["completion_samples"], self.fp_of("s08a_completion_hypotheses"))
        return self._stage("s08b_completion_evaluation", fp, fn, load)

    # =========================================================================== S09
    def s09_dem_pilot(self) -> dict | None:
        from . import dem as DM

        def fn():
            try:
                out = DM.run(self)
            except DM.DemUnavailable as e:
                self.ws.save_table(pd.DataFrame([{"gate": "G4_dem", "task": self.prim, "decision": "INCONCLUSIVE",
                                                  "reason": str(e), "profile": self.profile["name"]}]),
                                   "gate4", "dem/candidates")
                raise
            sub = "dem/candidates"
            for k, name in (("results", "results"), ("systems", "systems"), ("dem_log", "dem_candidate_log"),
                            ("gate", "gate4"), ("log", "dem_volume_log")):
                self.ws.save_table(out[k], name, sub)
            return out

        def load():
            return {"gate": self.ws.load_table("gate4", "dem/candidates")}
        fp = self.fp("s09", self.cfg["dem_pilot"], self.sh["dem"], self.fp_of("s06_candidates"))
        return self._stage("s09_dem_pilot", fp, fn, load)

    # =========================================================================== S10-S13
    def _profiles_for(self, u, cs) -> pd.DataFrame:
        from . import profiles as PR

        t = self.unit_profiles(u.key)
        if t is not None and "uid" in t and "uid" in cs.table:
            have = set(t.uid.astype(str))
            if set(cs.table.uid.astype(str)) <= have:
                m = dict(zip(t.uid.astype(str), range(len(t))))
                rows = t.iloc[[m[x] for x in cs.table.uid.astype(str)]].drop(columns=["uid", "pool_source"], errors="ignore")
                return rows.assign(cand_id=cs.table.cand_id.to_numpy()).reset_index(drop=True)
        df, _ = PR.candidate_profiles(self, u, cs, u.ev if u.task != "full" else np.asarray(self.ctx.data.valid, bool))
        return df

    def s10_candidate_features(self) -> dict | None:
        from . import features as FT

        def fn():
            self._require_coverage()
            tabs, provs = [], []
            mos = self.dem_mosaic()
            fm = self.m7.cached_full_maps()
            from ..gems3.official import unit_score

            for t in ("geo", self.prim):
                for u in self.units_of(t):
                    cs = self.pool(u.key)
                    if cs is None or not len(cs):
                        continue
                    sc = {"b0": unit_score(self.ctx, self.models[0], u), "b1": unit_score(self.ctx, self.models[1], u)}
                    prof = self._profiles_for(u, cs) if self.cfg["features"]["include_profiles"] else None
                    mc = self.mc_unit(u.key) if self.cfg["features"]["include_mc"] else None
                    T, pv = FT.describe(self, u, cs, with_labels=True, scores=sc, profiles=prof, mc=mc, mos=mos)
                    self.ws.save_table(T, u.key.replace(":", "_"), "feature_tables/units")
                    tabs.append(T)
                    provs.append(pv)
                    self.say(f"s10 {u.key}: {len(T)} rows")
            if self.profile["full_region_candidates"]:
                cs = self.pool("full:0")
                if cs is not None and len(cs):
                    fmap = fm or [np.asarray(self.ctx.oof(m), np.float32) for m in self.models]
                    fu = self.full_unit()
                    T, _ = FT.describe(self, fu, cs, with_labels=False, scores={"b0": fmap[0], "b1": fmap[1]},
                                       profiles=self._profiles_for(fu, cs) if self.cfg["features"]["include_profiles"]
                                       else None, mc=self.mc_unit("full:0"), mos=mos)
                    self.ws.save_table(T, "full_0", "feature_tables/units")
            T = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
            self.ws.save_table(T, "feature_table", "feature_tables")
            g = FT.groups(T) if len(T) else {}
            self.ws.save_table(pd.DataFrame([{"group": k, "feature": c} for k, v in g.items() for c in v]),
                               "feature_groups", "feature_tables")
            self.ws.save_table(pd.concat(provs, ignore_index=True).drop_duplicates() if provs else pd.DataFrame(),
                               "feature_provenance", "feature_tables")
            if len(T):
                self.ws.save_table(T.groupby(["task", "label"]).size().rename("n").reset_index(), "label_balance",
                                   "feature_tables")
                miss = T[[c for c in T.columns if c.startswith("has_")]].mean().rename("share").reset_index()
                self.ws.save_table(miss.rename(columns={"index": "indicator"}), "missingness", "feature_tables")
            return {"table": T}

        def load():
            return {"table": self.ws.load_table("feature_table", "feature_tables")}
        fp = self.fp("s10", self.cfg["features"], self.sh["ranking"]["positive_kd"], self.sh["ranking"]["negative_dist_km"],
                     *[self.fp_of(s) for s in ("s03b_structural_worms", "s04_multimodal", "s05_fault_normal_profiles",
                                               "s06_candidates", "s07_monte_carlo_signal", "s08a_completion_hypotheses",
                                               "s09_dem_pilot")], self.profile["full_region_candidates"])
        return self._stage("s10_candidate_features", fp, fn, load)

    def _tables_and_sets(self) -> tuple[dict, dict]:
        tabs, sets = {}, {}
        for t in ("geo", self.prim):
            for u in self.units_of(t):
                T = self.ws.load_table(u.key.replace(":", "_"), "feature_tables/units")
                cs = self.pool(u.key)
                if len(T) and cs is not None and len(T) == len(cs):
                    tabs[u.key] = T.sort_values("cand_id").reset_index(drop=True)
                    sets[u.key] = cs
        return tabs, sets

    def s11_ranking(self) -> dict | None:
        from ..m7.run import GateClosed

        from . import montecarlo as MC
        from . import ranking as RK

        def fn():
            self._require_coverage()
            tabs, sets = self._tables_and_sets()
            if not tabs:
                raise GateClosed("no feature tables: run section 10")
            out = RK.run(self, tabs, sets)
            sub = "ranking"
            for k, name in (("table", "ranking_scores"), ("curves", "ranking_curves"), ("summary", "ranking_summary"),
                            ("per_zone", "ranking_per_zone"), ("audit", "training_audit"), ("importance", "feature_importance"),
                            ("feature_groups", "feature_groups"), ("gate", "gate5")):
                self.ws.save_table(out[k], name, sub)
            uq = self.ws.load_table("mc_only_coverage", "monte_carlo/stability")
            if len(uq) or self.fp_of("s07_monte_carlo_signal"):
                self.ws.save_table(MC.gate3(self, uq, out["summary"]), "gate3", "monte_carlo/stability")
            from .figures import bar_panels16

            if len(out["summary"]):
                self.ws.save_fig(bar_panels16(out["summary"], "method", "dti_crossfit", "task", ref_col="fold_min",
                                              title="Section 11 - candidate rankers vs label-free / CNN-only controls"),
                                 "ranking_dti", out["summary"], sub)
            return out

        def load():
            return {"summary": self.ws.load_table("ranking_summary", "ranking"), "gate": self.ws.load_table("gate5", "ranking")}
        fp = self.fp("s11", self.cfg["ranking"], self.sh["ranking"], self.sh["rasterization"]["area_fracs"],
                     self.profile["run_pu_sensitivity"], self.fp_of("s10_candidate_features"))
        return self._stage("s11_ranking", fp, fn, load)

    def s12_graph(self) -> dict | None:
        from ..m7.run import GateClosed

        from . import graph as GR
        from .ranking import score_column

        def fn():
            self._require_coverage()
            if self.gate("gate5", "ranking") != "PASS":
                raise GateClosed("Gate 5 did not pass: the graph model would add complexity on top of a ranking that "
                                 "does not beat its controls")
            tabs, sets = self._tables_and_sets()
            T = self.ws.load_table("ranking_scores", "ranking")
            col = score_column(self, self.ws.load_table("ranking_summary", "ranking"))
            out = GR.run(self, T, sets, col)
            for k, name in (("edges", "edges"), ("curves", "graph_curves"), ("summary", "graph_summary"),
                            ("metrics", "graph_metrics"), ("decision", "graph_decision")):
                self.ws.save_table(out[k], name, "graphs")
            return out

        def load():
            return {"summary": self.ws.load_table("graph_summary", "graphs")}
        return self._stage("s12_graph", self.fp("s12", self.cfg["graph"], self.fp_of("s11_ranking")), fn, load)

    def s13_rasterization(self) -> dict | None:
        from ..m7.run import GateClosed

        from . import rasterize as RZ
        from .ranking import score_column

        def fn():
            self._require_coverage()
            T = self.ws.load_table("ranking_scores", "ranking")
            S = self.ws.load_table("ranking_summary", "ranking")
            if not len(T) or not len(S):
                raise GateClosed("no ranking outputs: run section 11")
            col = score_column(self, S)
            _, sets = self._tables_and_sets()
            stab = self.ws.load_table("stability", "monte_carlo/stability")
            out = RZ.run(self, T, sets, col, stab)
            sub = "rasterization"
            for k, name in (("grid", "rasterization_grid"), ("summary", "rasterization_summary"),
                            ("per_zone", "rasterization_per_zone"), ("ablation", "rasterization_ablation"),
                            ("ledger", "area_mass_ledger")):
                self.ws.save_table(out[k], name, sub)
            from .figures import bar_panels16

            ab = out["ablation"]
            if len(ab):
                self.ws.save_fig(bar_panels16(ab.assign(cat=ab.rep), "cat", "dti_crossfit", "task", ref_col="fold_min",
                                              title="Section 13 - representation / regional-budget ablation"),
                                 "representation_ablation", ab, sub)
            return out

        def load():
            return {"summary": self.ws.load_table("rasterization_summary", "rasterization")}
        fp = self.fp("s13", self.cfg["rasterization"], self.sh["rasterization"]["area_fracs"], self.fp_of("s11_ranking"),
                     self.fp_of("s07_monte_carlo_signal"))
        return self._stage("s13_rasterization", fp, fn, load)

    # =========================================================================== S14-S15
    def s14_ensemble(self) -> dict | None:
        from ..m7 import comparison as CP

        from . import ensemble as EN
        from . import gates as GA
        from . import report as RP
        from .ranking import score_column

        def fn():
            T = self.ws.load_table("ranking_scores", "ranking")
            S = self.ws.load_table("ranking_summary", "ranking")
            col = score_column(self, S) if len(S) else None
            comp = {u.key: self.completion_set(u.key) for u in self.units_of(self.prim)}
            comp = {k: v for k, v in comp.items() if v is not None and "completion_score" in v.table}
            fu = EN.run(self, T if len(T) else None, col, comp)
            self.ws.save_table(fu["grid"], "fusion_grid", "ensemble")
            self.ws.save_table(fu["summary"], "fusion_summary", "ensemble")
            self.ws.save_table(fu["per_zone"], "fusion_per_zone", "ensemble")
            cmp = EN.compare(self)
            sub = "validation/metrics"
            for k, name in (("per_zone", "method_per_zone"), ("retained_fp", "retained_fp_per_zone"),
                            ("pooled", "method_pooled"), ("paired", "paired_zone_deltas"), ("bootstrap", "bootstrap"),
                            ("systems", "system_hits"), ("proximity", "known_proximity"), ("regions", "regional_dti"),
                            ("false_length", "false_length")):
                self.ws.save_table(cmp[k], name, sub)
            self.ws.save_table(cmp["verdicts"], "decision_verdicts", "reports")
            CP.figures(self.view, cmp)
            g6 = EN.gate6(self, cmp["verdicts"], cmp["pooled"])
            self.ws.save_table(g6, "gate6", "ensemble")
            gates = GA.collect(self)
            self.ws.save_table(gates, "gates", "reports")
            self.ws.save_json(json.loads(gates.to_json(orient="records")), "reports/gates.json")
            ans = pd.DataFrame(RP.answers(self))
            self.ws.save_table(ans, "research_answers", "reports")
            self.ws.save_text(RP.report(self, gates), "reports/M8_PHYSICS_PROBABILISTIC_REPORT.md")
            return {"gates": gates, "verdicts": cmp["verdicts"], "pooled": cmp["pooled"], "answers": ans}

        def load():
            return {"gates": self.ws.load_table("gates", "reports"), "verdicts": self.ws.load_table("decision_verdicts", "reports"),
                    "pooled": self.ws.load_table("method_pooled", "validation/metrics"),
                    "answers": self.ws.load_table("research_answers", "reports")}
        fp = self.fp("s14", self.cfg["ensemble"], self.sh["decision"], self.sh["comparison"], self.profile["n_boot"],
                     *[self.fp_of(s) for s in ("s01_baseline", "s02_signal_recovery", "s06_candidates",
                                               "s07_monte_carlo_signal", "s08b_completion_evaluation", "s09_dem_pilot",
                                               "s11_ranking", "s12_graph", "s13_rasterization")])
        return self._stage("s14_ensemble", fp, fn, load)

    def s15_submission(self) -> dict | None:
        from . import submission as SU

        def fn():
            out = SU.run(self)
            self.ws.save_table(out["index"], "submission_index", "submissions")
            return out

        def load():
            return {"index": self.ws.load_table("submission_index", "submissions")}
        fp = self.fp("s15", self.cfg["submission"], self.sh["submission"], self.fp_of("s14_ensemble"))
        return self._stage("s15_submission", fp, fn, load)

    # =========================================================================== run + summary
    ORDER = ("s01_baseline", "s02_signal_recovery", "s03_potential_fields", "s03b_structural_worms", "s04_multimodal",
             "s06_candidates", "s05_fault_normal_profiles", "s07_monte_carlo_signal", "s08_monte_carlo_completion",
             "s09_dem_pilot", "s10_candidate_features", "s11_ranking", "s12_graph", "s13_rasterization",
             "s14_ensemble", "s15_submission")

    def run(self) -> dict:
        """Every enabled section in dependency order (06 before 05: profiles describe the M8 pool)."""
        if not self.cfg.get("enabled", True):
            self.say("disabled (M8_CONFIG['enabled'] = False)")
            return {}
        t0 = time.perf_counter()
        out = {"s00": self.s00_setup()}
        for name in self.ORDER:
            if self.enabled(name):
                out[name] = getattr(self, name)()
        self.say(f"finished in {time.perf_counter() - t0:.0f}s -> {self.ws.root}")
        self.paths()
        return out

    def summary(self) -> pd.DataFrame:
        p = self.ws.load_table("method_pooled", "validation/metrics")
        if len(p):
            v = self.ws.load_table("decision_verdicts", "reports")
            return p.merge(v[["method", "verdict"]], on="method", how="left") if len(v) else p
        return self.ws.load_immutable_table("baseline_table", "baseline")

    def paths(self) -> dict:
        drive = self.cfg.get("_drive_root")
        rows = {}
        for s in sorted((self.ws.root / "manifests" / "sections").glob("*.json")):
            rows[s.stem] = {"local": str(self.ws.root), "drive": drive, "manifest": str(s.relative_to(self.ws.root))}
        print(f"[m8] outputs (local): {self.ws.root}")
        if drive:
            print(f"[m8] outputs (Drive): {drive}")
        for k, v in rows.items():
            print(f"  {k:<32} {v['manifest']}")
        return rows


def _with_ids(cs):
    from ..gems3.candidates import CandidateSet

    t = cs.table.reset_index(drop=True).copy()
    t["cand_id"] = np.arange(len(cs))
    return CandidateSet(t, list(cs.paths))
