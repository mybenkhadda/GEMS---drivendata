"""M9 orchestration: ``pipe.m9(M9_CONFIG)`` -> one method per notebook section, fingerprinted and resumable.

    m9 = pipe.m9(M9_CONFIG)
    m9.s00_setup(); m9.s01_baseline(); ...; m9.run(stage="PREPARE"); m9.run()       # stages or everything

Composition, not duplication: an ``M7Research`` engine (read-only) provides units, eligible budget, protocol keys, the
B0 / B1 maps and every M7 helper; its GEMS3 engine is re-pointed to ``outputs/m9_geofault_production/gems3_engine``.
M7 / M8 helpers that write receive ``M9View`` (the M7 engine seen through the M9 workspace, with the comparison
restricted to the zones this profile trains). Section 01 reuses M8's baseline recovery through ``_M8Like``.

A stage whose fingerprint (its configuration + upstream fingerprints + member-map fingerprints + profile scope + M9
version) and outputs match is reused; ``overwrite_existing=True`` recomputes. Training resumes from checkpoints
regardless of stage fingerprints. Every stage writes ``manifests/sections/<stage>.json`` and pushes to Drive.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..m8.run import M7View
from ..pipeline.m6 import fingerprint
from .config import load_m9_config, m7_overrides, resolve_variant, sections_of, zones_of
from .workspace import M9_VERSION, M9Workspace


class M9View(M7View):
    """M7 engine seen through the M9 workspace; comparisons only see the zones trained by the profile."""

    def units_of(self, task: str) -> list:
        return self._m8.units_of(task)

    def zones(self, task: str) -> list[int]:
        return [u.zone for u in self._m8.units_of(task)]


class _M8Like:
    """The attributes ``m8.baseline.run`` reads, served by M9 (all zones: the champion is rebuilt everywhere)."""

    def __init__(self, m9):
        self._m9 = m9
        self.cfg = {"baseline": {"reuse_m7_selection": m9.cfg["baseline"]["reuse_m7_selection"],
                                 "champion": m9.cfg["baseline"]["champion"]},
                    "recorded_m7": m9.cfg["recorded"]}

    def __getattr__(self, k):
        return getattr(self._m9, k)

    @property
    def view(self):
        return self._m9.m7view_all                      # all zones: the champion's selection is nested over them

    def units_of(self, task: str) -> list:
        return self._m9.m7.units_of(task)


class GateClosed(RuntimeError):
    """A section refuses to run because its scientific gate is closed (not a failure)."""


class M9Research:
    def __init__(self, pc=None, config: dict | None = None, *, ctx=None, local_root: Path | None = None,
                 data_root: Path | None = None, config_path: Path | None = None):
        from ..gems3.run import GEMS3Research
        from ..m7.run import M7Research
        from .features import LayerCache

        self.cfg = load_m9_config(config, config_path)
        self.pc = pc
        self.local = Path(local_root or (pc.local if pc is not None else Path.cwd()))
        drive = getattr(pc, "drive", None) if pc is not None else None
        self.cfg["_drive_root"] = str(Path(drive) / self.cfg["output_dir"]) if drive else None
        self.ws = M9Workspace(self.local / self.cfg["output_dir"], self.cfg)
        self.cache = self.local / self.cfg["cache_dir"]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.layers = LayerCache(self.cache)
        self.m7 = M7Research(pc, m7_overrides(self.cfg), ctx=ctx, local_root=local_root, data_root=data_root)
        sub = self.m7.cfg["submission"]
        self.m7.g3 = GEMS3Research(pc, {"models": list(self.m7.cfg["models"]), "tasks": ["geo", "disc_official"],
                                        "primary_task": "disc_official",
                                        "output_dir": f"{self.cfg['output_dir']}/gems3_engine", "cache_dir": "cache/gems3",
                                        "overwrite_existing": False,
                                        "submission": {"source": sub["source"], "device": sub["device"]}},
                                   ctx=ctx, local_root=local_root, data_root=data_root)
        self.m7.cfg["evaluation"]["n_boot"] = int(self.cfg["_profile"]["n_boot"])
        self.m7view = M9View(self)                      # comparisons: zones trained by the profile
        self.m7view_all = M7View(self)                  # baseline recovery / fallback export: every zone
        self._memo: dict = {}
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
        print(f"[m9] {msg}")

    def train_folds(self) -> list[int]:
        self.load()
        return zones_of(self.profile["train_folds"], list(self.ctx.fold_ids))

    def train_splits(self) -> list[int]:
        self.load()
        return zones_of(self.profile["train_splits"], list(range(self.ctx.splits.n_splits)))

    def units_of(self, task: str) -> list:
        """Units of the zones trained by the profile (every comparison is paired on these zones)."""
        keep = self.train_folds() if task == "geo" else self.train_splits() if task.startswith("disc") else None
        us = self.m7.units_of(task)
        return [u for u in us if keep is None or u.zone in keep]

    def unit(self, key: str):
        return self.m7.unit(key)

    def full_unit(self):
        return self.m7.full_unit()

    def elig(self, u, ring_px: int = 0) -> np.ndarray:
        return self.m7.elig(u, ring_px)

    def pkey(self, u) -> str:
        return self.m7.pkey(u)

    def cache_get(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    # -- variants ---------------------------------------------------------------------
    def variants(self) -> list[str]:
        out = list(self.profile["variants"])
        out += [f"core@seed{i}" for i in range(1, int(self.profile.get("n_seeds", 1)))]
        return out

    def variant_cfg(self, name: str) -> dict:
        base, _, s = name.partition("@seed")
        v = resolve_variant(self.cfg, base)
        if s:
            v["seed"] = int(v["seed"]) + 7919 * int(s)
        v["_variant"] = name
        return v

    def cube_for(self, vcfg: dict):
        """(meta, local cube, pooled cube) of a variant's channel groups (built once, memory-mapped)."""
        from .dataset import build_cube, open_cube
        from .features import channel_groups

        groups = channel_groups(self, vcfg)
        meta = build_cube(self, groups)
        k = ("cube", meta["key"])
        if k not in self._memo:
            self._memo[k] = open_cube(meta)
        loc, pool = self._memo[k]
        return meta, loc, pool

    def cube_keys(self) -> dict:
        from .dataset import channel_order, cube_key
        from .features import channel_groups

        out = {}
        for v in self.variants():
            try:
                g = channel_groups(self, self.variant_cfg(v))
                out[v] = cube_key(*channel_order(g), int(self.cfg["data"]["context_factor"]))
            except Exception:
                out[v] = ""
        return out

    def spectral_selected(self) -> list[str]:
        t = self.ws.load_table("spectral_registry", "spectral")
        return list(t[t.selected.astype(bool)].name) if len(t) and "selected" in t else []

    def struct_selected(self) -> list[str]:
        t = self.ws.load_table("selected_registry", "features")
        return list(t[t.selected.astype(bool) & (t.family != "raw")].name) if len(t) else []

    def domains(self) -> np.ndarray | None:
        def build():
            p = self.ws.root / "datasets" / "cv" / "domains.npz"
            return np.load(p)["domain"] if p.exists() else None
        return self.cache_get("domains", build)

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

    # -- selection / DEM (M7View interface) ---------------------------------------------
    def set_selection(self, sel: dict) -> None:
        self._memo["selection"] = sel

    def baseline_selection(self) -> dict | None:
        if "selection" not in self._memo:
            self._memo["selection"] = self.read_json("baseline/baseline_selection.json")
        return self._memo["selection"]

    def dem_mosaic(self):
        from ..m7.dem_morph import DemMosaic

        if "dem_mosaic" not in self._memo:
            p = self.m7.ws.root / "dem" / "features" / "m7_dem_features_100m.tif"
            if not p.exists():
                self.stage_in([p.relative_to(self.local).as_posix()] if str(p).startswith(str(self.local)) else [])
            self._memo["dem_mosaic"] = DemMosaic(p) if p.exists() else None
        return self._memo["dem_mosaic"]

    def dem_layer(self):
        return None

    # -- loaders shared by sections -----------------------------------------------------
    def trace_set(self, key: str):
        from ..gems3.candidates import CandidateSet

        k = ("trace", key)
        if k not in self._memo:
            kk = key.replace(":", "_")
            z = self.ws.root / "tracing" / "units" / f"{kk}.npz"
            t = self.ws.load_table(kk, "tracing/units")
            self._memo[k] = CandidateSet.from_npz(t.sort_values("cand_id").reset_index(drop=True), np.load(z)) \
                if z.exists() and len(t) else None
        return self._memo[k]

    def save_trace_set(self, cs, key: str) -> None:
        kk = key.replace(":", "_")
        self.ws.save_npz(cs.to_npz(), f"tracing/units/{kk}.npz")
        self.ws.save_table(cs.table, kk, "tracing/units")
        self._memo[("trace", key)] = cs

    def full_traces(self):
        return self.trace_set("full:0")

    # -- fingerprints / stages -----------------------------------------------------
    def fp(self, *parts) -> str:
        self.load()
        return fingerprint(M9_VERSION, self.m7.g3.m6.map_fp, self.sh["models"], self.sh["evaluation"]["tasks"],
                           self.sh["evaluation"]["budget_domain"], *parts)

    def scope(self) -> dict:
        return {k: self.profile[k] for k in ("train_folds", "train_splits", "variants", "n_seeds", "epochs",
                                              "steps_per_epoch")}

    def fp_of(self, stage: str) -> str | None:
        r = self.ws.unit_record(stage)
        return r.get("fingerprint") if r else None

    def enabled(self, section: str) -> bool:
        return bool(self.cfg["sections"].get(section))

    def _stage(self, name: str, fp: str, fn, load=None, optional: bool = True):
        from ..m7.run import GateClosed as G7
        from ..training.trainer import TrainingInterrupted

        t0 = time.perf_counter()
        if self.ws.valid(name, fp):
            self.say(f"{name}: reused (fingerprint {fp}) -> {self.ws.root}")
            self.log.append({"stage": name, "outcome": "REUSED", "seconds": 0.0, "fingerprint": fp})
            return load() if load else None
        self.ws.begin(name)
        try:
            out = fn()
        except (GateClosed, G7) as e:
            self.say(f"{name} SKIPPED (gate): {e}")
            self.log.append({"stage": name, "outcome": "GATE_CLOSED", "reason": str(e), "seconds": 0.0})
            self._save_log()
            return None
        except TrainingInterrupted as e:
            self.say(f"{name} INTERRUPTED (runtime budget, resumable): {e}")
            self.log.append({"stage": name, "outcome": "INTERRUPTED", "reason": str(e),
                             "seconds": round(time.perf_counter() - t0, 1)})
            self._save_log()
            self.push()
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
                                        "seconds": dt, "profile_scope": self.profile})
        self.ws.finish(name, fp, seconds=dt)
        self.log.append({"stage": name, "outcome": "COMPUTED", "seconds": dt, "fingerprint": fp})
        self.say(f"{name}: computed in {dt:.0f}s -> {self.ws.root}")
        self._save_log()
        self.push()
        return out

    def _save_log(self) -> None:
        from ..artifacts.atomic import atomic_write_json

        atomic_write_json(self.ws.root / "manifests" / "stage_log.json", json.loads(json.dumps(self.log, default=str)))

    def gate(self, name: str, sub: str, component: str | None = None) -> str | None:
        t = self.ws.load_table(name, sub)
        if not len(t):
            return None
        if component is not None and "component" in t:
            t = t[t.component == component]
        t = t[t.task == self.prim] if "task" in t and (t.task == self.prim).any() else t
        return str(t.decision.iloc[0]) if len(t) else None

    def _fig(self, out, name: str, sub: str) -> None:
        fig, data = out if isinstance(out, tuple) else (out, None)
        if fig is not None:
            self.ws.save_fig(fig, name, data, sub)
        elif data is not None and len(data):
            self.ws.save_table(data, f"{name}_data", sub)

    # =========================================================================== S00
    def s00_setup(self) -> dict:
        """Environment, reuse matrix, compatibility, M8 gate records (read-only; always re-run)."""
        from .._version import provenance
        from ..training.gpu import detect
        from . import inventory as I

        self.load()
        d = self.ctx.data
        checks = {"crs_epsg": d.grid.crs.to_epsg() if hasattr(d.grid.crs, "to_epsg") else str(d.grid.crs),
                  "shape": list(d.shape), "template_same_grid": bool(d.template_matches()), "n_bands": len(d.band_names),
                  "bands": list(d.band_names), "profile": self.profile, "gpu": detect(),
                  "train_folds": self.train_folds(), "train_splits": self.train_splits(), "variants": self.variants()}
        fails = []
        if checks["crs_epsg"] != 32611:
            fails.append(f"CRS {checks['crs_epsg']} != EPSG:32611")
        if not checks["template_same_grid"]:
            fails.append("example_submission.tif grid differs from the feature grid")
        need = set(self.cfg["features"]["raw_local"]) | set(self.cfg["features"]["raw_context"]) | \
            set(self.cfg["spectral"]["fields"].values())
        bands = pd.DataFrame([{"band": b, "available": b in d.band_names, "description": d.band_desc.get(b, "")}
                              for b in sorted(need)])
        self.ws.begin("s00_setup")
        self.ws.save_table(I.table(), "reuse_matrix", "manifests")
        self.ws.save_table(bands, "band_availability", "manifests")
        self.ws.save_table(self.m7.protocol_keys(), "protocol_keys", "manifests")
        m8 = self.recover_m8()
        self.ws.save_table(m8, "m8_saved_outputs", "baseline")
        self.ws.save_json({"checks": checks, "failures": fails, "code": provenance(), "m9_version": M9_VERSION,
                           "map_fingerprints": self.m7.g3.m6.map_fp, "models": self.models,
                           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, "manifests/setup.json")
        self.ws.save_json(json.loads(json.dumps(self.cfg, default=str)), "manifests/m9_config.json")
        self.ws.section_manifest("s00_setup", {"failures": fails})
        self.ws.finish("s00_setup", fingerprint(time.time()))
        self.push()
        if fails:
            raise RuntimeError(f"setup checks failed: {fails}")
        return {"checks": checks, "bands": bands, "reuse": I.table(), "m8": m8}

    def recover_m8(self) -> pd.DataFrame:
        """M8's saved gate / comparison tables, copied verbatim (read-only); 'absent' when M8 never ran."""
        if not self.cfg["baseline"].get("read_m8_gates", True):
            return pd.DataFrame()
        root = self.local / self.cfg["m8_output_dir"]
        rels = ["reports/gates.csv", "ensemble/gate6.csv", "validation/metrics/method_pooled.csv",
                "candidate_generation/oracle/gate2.csv", "reports/decision_verdicts.csv"]
        self.stage_in([f"{self.cfg['m8_output_dir']}/{r}" for r in rels])
        rows = []
        for r in rels:
            p = root / r
            if p.exists():
                t = pd.read_csv(p)
                self.ws.save_table(t, Path(r).stem, "baseline/recovered_m8")
                rows.append({"m8_table": r, "status": "recovered (read-only)", "rows": len(t),
                             "decisions": ";".join(f"{a}={b}" for a, b in zip(t.get("gate", t.get("method", [])),
                                                                               t.get("decision", t.get("verdict", []))))
                             if ("decision" in t or "verdict" in t) else ""})
            else:
                rows.append({"m8_table": r, "status": "absent (M8 not run or not on Drive): nothing is assumed",
                             "rows": 0, "decisions": ""})
        return pd.DataFrame(rows)

    # =========================================================================== S01
    def s01_baseline(self) -> dict | None:
        from ..m8 import baseline as B

        from . import gates as GA

        def fn():
            out = B.run(_M8Like(self))
            sub = "baseline"
            fp = self.fp("s01", self.cfg["baseline"], self.sh["baseline"])
            self.ws.save_immutable_table(out["table"], "baseline_table", sub, fp)
            for k, name in (("per_zone", "baseline_per_zone"), ("ledger", "area_ledger"), ("retained_fp", "retained_fp_rescore"),
                            ("recorded", "recorded_vs_this_run"), ("compatibility", "compatibility"),
                            ("systems", "system_table"), ("attributes", "system_attributes"), ("saved", "heldout_maps_saved")):
                self.ws.save_table(out[k], name, sub)
            self.ws.save_json(json.loads(json.dumps(out["selection"], default=float)), f"{sub}/baseline_selection.json")
            self.ws.save_json(json.loads(json.dumps(out["manifest"], default=str)), f"{sub}/baseline_manifest.json")
            info, err = None, None
            if self.cfg["baseline"]["gate0_export_fallback"]:
                try:
                    from ..m7 import export as E

                    champ = self.cfg["baseline"]["champion"]
                    info = E.export_baseline(self.m7view_all, f"m9_fallback_{champ}", champ, ring_px=0)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
            self.ws.save_table(GA.gate0(self, info, err), "gate0", sub)
            from .figures import bar_panels16

            t = out["table"]
            if len(t):
                d = t[["task", "method", "dti_crossfit", "fold_min"]].copy()
                self.ws.save_fig(bar_panels16(d, "method", "dti_crossfit", "task", ref_col="fold_min",
                                              title="Section 01 - B0/B1 baselines under the shared protocol"),
                                 "baseline_dti", d, sub)
            return out

        def load():
            return {"table": self.ws.load_immutable_table("baseline_table", "baseline"),
                    "recorded": self.ws.load_table("recorded_vs_this_run", "baseline"),
                    "compatibility": self.ws.load_table("compatibility", "baseline"),
                    "gate0": self.ws.load_table("gate0", "baseline")}
        return self._stage("s01_baseline", self.fp("s01", self.cfg["baseline"], self.sh["baseline"],
                                                   self.cfg["recorded"], self.sh["submission"]), fn, load, optional=False)

    # =========================================================================== S02
    def s02_features(self) -> dict | None:
        from . import features as FE

        def fn():
            sub = "features"
            inv = FE.inventory(self)
            self.ws.save_table(inv, "inventory", sub)
            new = FE.crossmodal_layers(self)
            self.ws.save_table(new, "crossmodal_new", sub)
            dec, pairs = FE.redundancy_select(self, inv)
            if len(new):
                dec = pd.concat([dec, new.assign(selected=True, reason="new tolerant cross-modal evidence (Gate 1 decides)")
                                 [["name", "family", "selected", "reason"]]], ignore_index=True)
            meta = inv.set_index("name")
            for c in ("modality", "source", "units", "scale_km", "support_km", "origin", "phase", "fingerprint"):
                dec[c] = dec.name.map(meta[c]) if c in meta else None
            dec["source_resolution_m"] = dec.name.map(lambda n: 100 if not n.startswith("m9:") else 100)
            self.ws.save_table(dec, "selected_registry", sub)
            self.ws.save_table(pairs, "redundancy_pairs", sub)
            fam = dec.groupby(["family", "selected"]).size().rename("n").reset_index()
            self.ws.save_table(fam, "family_counts", sub)
            corr = FE.correlation_matrix(self, list(dec[dec.selected.astype(bool)].name)[:40])
            self.ws.save_table(corr, "selected_spearman", sub)
            from .figures import bar_panels16

            if len(fam):
                f2 = fam.assign(state=np.where(fam.selected, "selected", "dropped"))
                self.ws.save_fig(bar_panels16(f2, "family", "n", "state", xlabel="channels",
                                              title="Feature families: selected vs dropped (label-free redundancy)"),
                                 "feature_families", f2, f"{sub}/figures")
            return {"registry": dec, "inventory": inv}

        def load():
            return {"registry": self.ws.load_table("selected_registry", "features"),
                    "inventory": self.ws.load_table("inventory", "features")}
        return self._stage("s02_features", self.fp("s02", self.cfg["features"]), fn, load, optional=False)

    # =========================================================================== S03
    def s03_spectral(self) -> dict | None:
        from . import features as FE
        from . import gates as GA
        from . import spectral as SP
        from . import figures as FG

        def fn():
            sub = "spectral"
            if not self.cfg["spectral"].get("enabled", True):
                self.ws.save_table(pd.DataFrame([{"gate": "G1_representation", "component": "spectral",
                                                  "decision": "EXCLUDE", "reason": "spectral.enabled = false"}]),
                                   "gate1", sub)
                return {}
            reg, prods = SP.build(self)
            reg = SP.select(self, reg, int(self.cfg["spectral"]["max_channels"]))
            self.ws.save_table(reg, "spectral_registry", sub)
            self.ws.save_table(SP.compare_existing(self, list(reg.name)), "spectral_vs_existing", sub)
            for fam, pr in prods.items():
                self._fig(FG.fourier_spectra(pr, fam, seed=int(self.cfg["seed"])), f"fourier_{fam}", f"{sub}/figures")
                self.ws.save_npz({k: np.asarray(v, np.float32) for k, v in pr.items()}, f"{sub}/fft_windows_{fam}.npz")
            sel = FE.selected_registry(self)
            base = list(sel[sel.selected.astype(bool)].name)
            raw = [n for n in base if n.startswith("raw:")]
            sets = {"raw": raw, "base": base, "base_no_new": [n for n in base if not n.startswith("m9:xmod:")],
                    "base_spectral": base + list(reg[reg.selected.astype(bool)].name)}
            if self.cfg["features"]["family_screen"].get("enabled", True):
                t = FE.family_screen(self, sets)
                pooled = FE.pooled_screen(t)
                hits = t.attrs.get("hits", pd.DataFrame())
                uniq = pd.concat([FE.unique_systems(hits, "base_spectral", "base"),
                                  FE.unique_systems(hits, "base", "base_no_new")], ignore_index=True)
                ming = float(self.cfg["gates"]["g1_representation"]["min_conditional_gain"])
                nested = pd.concat([GA.nested_component(t, "spectral", "base_spectral", "base", ming),
                                    GA.nested_component(t, "crossmodal_new", "base", "base_no_new", ming)], ignore_index=True)
                self.ws.save_table(t, "screen_per_zone", sub)
                self.ws.save_table(pooled, "screen_pooled", sub)
                self.ws.save_table(uniq, "screen_unique_systems", sub)
                self.ws.save_table(nested, "screen_nested_consistency", sub)
                self.ws.save_table(GA.gate1(self, pooled, uniq, nested), "gate1", sub)
                self._fig(FG.feature_family_comparison(pooled), "feature_set_comparison", f"{sub}/figures")
                self._fig(FG.raw_vs_structural(pooled), "raw_vs_structural", f"{sub}/figures")
            c = FG.unit_center(self.units_of(self.prim)[0]) if self.units_of(self.prim) else (1800, 1600)
            tol = int(min(self.cfg["features"]["crossmodal"]["tolerance_m"]))
            self._fig(FG.agreement_map(self, f"m9:xmod:mag_grav_tol@{tol}m_{self.cfg['features']['crossmodal']['scale_km'][0]:g}km",
                                       c), "mag_grav_agreement", f"{sub}/figures")
            for fam in self.cfg["spectral"]["fields"]:
                self._fig(FG.wavelet_maps(self, fam, c), f"wavelet_{fam}", f"{sub}/figures")
            return {"registry": reg}

        def load():
            return {"registry": self.ws.load_table("spectral_registry", "spectral"),
                    "gate": self.ws.load_table("gate1", "spectral")}
        fp = self.fp("s03", self.cfg["spectral"], self.cfg["features"]["family_screen"], self.cfg["gates"]["g1_representation"],
                     self.scope(), self.fp_of("s02_features"))
        return self._stage("s03_spectral", fp, fn, load)

    # =========================================================================== S04
    def s04_dataset(self) -> dict | None:
        from . import dataset as DS
        from .features import channel_groups

        def fn():
            sub = "datasets"
            for k, t in DS.cv_manifests(self).items():
                self.ws.save_table(t, k, f"{sub}/cv")
            dom, blocks = DS.geological_domains(self)
            self.ws.save_npz({"domain": dom.astype(np.int16)}, f"{sub}/cv/domains.npz")
            self.ws.save_raster({"domain": np.where(dom >= 0, dom, np.nan)}, f"{sub}/cv/domains.tif", self.ctx.data.grid)
            self.ws.save_table(blocks, "cv_e_domain_blocks", f"{sub}/cv")
            self._memo.pop("domains", None)
            rows, norm = [], {}
            for v in self.variants():
                vc = self.variant_cfg(v)
                g = channel_groups(self, vc)
                meta, cube, pooled = self.cube_for(vc)
                self.ws.save_json({"variant": v, "groups": g, "cube_key": meta["key"], "local": meta["local"],
                                   "pooled": meta["pooled"]}, f"{sub}/channels_{v.replace('@', '_')}.json")
                for tu in DS.training_units(self):
                    sc = DS.FoldScaler.fit(cube, meta["local"], tu.allowed & np.asarray(self.ctx.data.valid, bool),
                                           int(vc["features"]["scaler"]["sample_px"]), float(vc["features"]["scaler"]["clip"]),
                                           int(vc["seed"]))
                    norm[f"{v}/{tu.name}"] = sc.to_json()
                    rows.append({"variant": v, "unit": tu.name, "kind": tu.kind, "zone": tu.zone,
                                 "allowed_px": int(tu.allowed.sum()), "visible_fault_px": int(tu.visible.sum()),
                                 "weighted_px": int(tu.weight_mask.sum()),
                                 "predict_px": int(tu.predict.sum()) if tu.predict is not None else 0,
                                 "n_local": len(meta["local"]), "n_pooled": len(meta["pooled"]), "cube_key": meta["key"],
                                 "scaler_hash": sc.to_json()["hash"]})
            self.ws.save_table(pd.DataFrame(rows), "training_units", sub)
            self.ws.save_json(norm, f"{sub}/normalization_manifest.json")
            return {"units": pd.DataFrame(rows)}

        def load():
            return {"units": self.ws.load_table("training_units", "datasets")}
        fp = self.fp("s04", self.cfg["data"], self.cfg["features"], self.scope(), self.fp_of("s02_features"),
                     self.fp_of("s03_spectral"))
        return self._stage("s04_dataset", fp, fn, load, optional=False)

    # =========================================================================== S05
    def s05_corruption(self) -> dict | None:
        from . import corruption as CO
        from . import dataset as DS

        def fn():
            sub = "corruption"
            sets, proto = CO.benchmark_sets(self)
            CO.save_sets(self.ws, sets)
            self.ws.save_table(proto, "benchmark_protocol", sub)
            vc = self.variant_cfg("core")
            meta, cube, pooled = self.cube_for(vc)
            tus = DS.training_units(self)
            stats = []
            if tus:
                tu = tus[0]
                sc = DS.FoldScaler.fit(cube, meta["local"], tu.allowed, 50_000, 6.0, int(vc["seed"]))
                psc = DS.pooled_scaler(sc, meta["pooled"], pooled, np.ones(pooled.shape[1:], bool), int(vc["seed"]))
                b = DS.BatchBuilder(self, tu, vc, cube, pooled, sc, psc, DS.build_targets(self, tu),
                                    np.random.default_rng(int(vc["seed"]) + 55))
                arr, st = CO.audit_samples(self, b, int(self.profile["audit_samples"]))
                self.ws.save_npz(arr, f"{sub}/samples/{tu.name}.npz")
                self.ws.save_table(st, f"samples_{tu.name}", f"{sub}/samples")
                stats.append(pd.DataFrame([{"operation": k, "count": v} for k, v in b.stats.items()]).assign(unit=tu.name))
                if (st.context_on_non_visible_px > 0).any() and not ("local_displacement" in vc["corruption"]["operators"]):
                    raise RuntimeError("leakage audit: context shows pixels that are not visible labels")
                self._fig(self._corruption_panels(arr, tu.name), "corruption_samples", f"{sub}/samples")
            self.ws.save_table(pd.concat(stats, ignore_index=True) if stats else pd.DataFrame(), "operation_counts", sub)
            self.ws.save_json({"mixture": vc["corruption"]["mixture"], "operators": vc["corruption"]["operators"],
                               "context_pixel_weight": vc["corruption"]["context_pixel_weight"],
                               "reconstruction_weight": vc["corruption"]["reconstruction_weight"],
                               "leakage_rule": "context = visible labels of the unit only (permitted geography, ignored "
                                               "systems removed); features label-free"}, f"{sub}/corruption_manifest.json")
            return {"protocol": proto}

        def load():
            return {"protocol": self.ws.load_table("benchmark_protocol", "corruption")}
        fp = self.fp("s05", self.cfg["corruption"], self.scope(), self.fp_of("s04_dataset"))
        return self._stage("s05_corruption", fp, fn, load)

    def _corruption_panels(self, arr: dict, name: str):
        import matplotlib.pyplot as plt

        n = min(6, arr["seg"].shape[0])
        if not n:
            return None, None
        P = arr["seg"].shape[-1] * 8
        un = {k: np.unpackbits(v, axis=-1)[..., :P].astype(bool) for k, v in arr.items()}
        fig, axes = plt.subplots(3, n, figsize=(16, 9), squeeze=False)
        for i in range(n):
            for j, (k, t) in enumerate((("seg", "visible faults (target)"), ("context", "context shown"),
                                        ("removed", "removed (L6 target)"))):
                ax = axes[j][i]
                ax.imshow(un[k][i], cmap="Greys", interpolation="nearest")
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_ylabel(t, fontsize=8)
        fig.suptitle(f"Corruption audit samples ({name})", x=0.01, ha="left")
        fig.tight_layout()
        return fig, pd.DataFrame({"sample": range(n)})

    # =========================================================================== S06 / S07
    def s06_architecture(self) -> dict | None:
        def fn():
            from .features import channel_groups
            from .model import build, count_params, net_spec

            rows = []
            for v in self.variants():
                vc = self.variant_cfg(v)
                meta, _, _ = self.cube_for(vc)
                spec = net_spec(meta, channel_groups(self, vc), vc)
                model = build(spec)
                rows.append({"variant": v, "n_params": count_params(model), **{f"n_{g}": n for g, n in spec["group_sizes"].items()},
                             "n_pooled": spec["n_pool"], "patch_px": spec["patch_px"], "context_factor": spec["factor"],
                             "widths": str(vc["model"]["widths"]), "fusion": vc["model"]["fusion"]})
                self.ws.save_json(json.loads(json.dumps(spec, default=str)), f"models/net_spec_{v.replace('@', '_')}.json")
                del model
            t = pd.DataFrame(rows)
            self.ws.save_table(t, "architecture", "models")
            return {"architecture": t}

        def load():
            return {"architecture": self.ws.load_table("architecture", "models")}
        return self._stage("s06_architecture", self.fp("s06", self.cfg["model"], self.cfg["variants"], self.scope(),
                                                       self.fp_of("s04_dataset")), fn, load)

    def s07_losses(self) -> dict | None:
        def fn():
            from .losses import active_terms

            rows = []
            for v in self.variants():
                lc = self.variant_cfg(v)["loss"]
                for st in ("A", "B", "C"):
                    rows.append({"variant": v, "stage": st, **active_terms(st, lc)})
            t = pd.DataFrame(rows)
            self.ws.save_table(t, "loss_schedule", "training")
            return {"schedule": t}

        def load():
            return {"schedule": self.ws.load_table("loss_schedule", "training")}
        return self._stage("s07_losses", self.fp("s07", self.cfg["loss"], self.cfg["variants"], self.scope()), fn, load)

    # =========================================================================== S08
    def s08_training(self) -> dict | None:
        """Every (variant, unit) of the profile: cv units first, then discovery units (fixed epochs). Resumable."""
        from . import dataset as DS
        from . import train as TR
        from .features import channel_groups
        from .figures import loss_components

        def fn():
            mx = float(self.cfg["training"]["max_runtime_min"])
            deadline = time.monotonic() + mx * 60 if mx > 0 else None
            recs = []
            prev = self.ws.load_table("training_units", "training")
            for v in self.variants():
                vc = self.variant_cfg(v)
                groups = channel_groups(self, vc)
                meta, cube, pooled = self.cube_for(vc)
                tus = DS.training_units(self)
                for tu in [t for t in tus if t.kind == "cv"]:
                    recs.append(TR.train_unit(self, v, vc, tu, meta, cube, pooled, groups, deadline=deadline))
                ep = TR.discovery_epochs(self, v, int(self.profile["epochs"]))
                for tu in [t for t in tus if t.kind != "cv"]:
                    recs.append(TR.train_unit(self, v, vc, tu, meta, cube, pooled, groups, fixed_epochs=ep,
                                              deadline=deadline))
            R = pd.DataFrame(recs)
            if len(prev):
                R = pd.concat([prev[~prev.set_index(["variant", "unit"]).index.isin(R.set_index(["variant", "unit"]).index)], R],
                              ignore_index=True)
            self.ws.save_table(R, "training_units", "training")
            self.ws.save_table(R, "model_registry", "reports/model_registry")
            logs = []
            for r in R.itertuples():
                h = self.ws.load_table(f"{r.variant}__{r.unit}", "training/logs")
                if len(h):
                    logs.append(h.assign(run=f"{r.variant}/{r.unit}"))
            if logs:
                H = pd.concat(logs, ignore_index=True)
                self._fig(loss_components(H[H.run.str.startswith("core/")]), "loss_components", "training")
            return {"units": R}

        def load():
            return {"units": self.ws.load_table("training_units", "training")}
        fp = self.fp("s08", self.cfg["training"], self.cfg["loss"], self.cfg["corruption"], self.cfg["model"],
                     self.cfg["variants"], self.scope(), self.fp_of("s04_dataset"))
        return self._stage("s08_training", fp, fn, load)

    # =========================================================================== S09
    def s09_validation(self) -> dict | None:
        from . import figures as FG
        from . import fusion as FU
        from . import validation as VA

        def fn():
            sub = "validation/metrics"
            idx = []
            for v in self.variants():
                idx.append(VA.heldout(self, v))
                VA.stitch_rasters(self, v)
            self.ws.save_table(pd.concat(idx, ignore_index=True) if idx else pd.DataFrame(), "heldout_index", "validation")
            if int(self.profile.get("n_seeds", 1)) > 1:
                self._ensemble_preds("core", [v for v in self.variants() if v == "core" or v.startswith("core@seed")])
            sel = {}
            names = self.variants() + (["core_ens"] if int(self.profile.get("n_seeds", 1)) > 1 else [])
            grids = []
            for v in names:
                out = VA.dense_methods(self, v)
                for m, o in out.items():
                    sel[m] = o["summary"]
                    grids.append(o["grid"])
            self.ws.save_table(pd.concat(grids, ignore_index=True) if grids else pd.DataFrame(), "dense_grid", "validation")
            self.ws.save_json(json.loads(json.dumps(sel, default=float)), "validation/dense_selection.json")
            rec = pd.concat([VA.reconstruction(self, v) for v in ("core", "recon_off") if v in self.variants()],
                            ignore_index=True)
            self.ws.save_table(rec, "reconstruction", sub)
            if len(rec):
                self.ws.save_table(rec.groupby(["variant", "class", "context", "frac"])[["recall_w", "dti"]].mean().reset_index(),
                                   "reconstruction_pooled", sub)
            cmp = FU.compare(self)
            self._save_cmp(cmp, sub)
            strata = VA.length_strata(self, cmp["systems"])
            self.ws.save_table(strata, "length_strata", sub)
            meths = [m for m in cmp["pooled"].method.unique()] if len(cmp["pooled"]) else []
            dd = VA.domain_dti(self, meths)
            self.ws.save_table(dd, "domain_dti", sub)
            self.ws.save_table(VA.variant_deltas(cmp["pooled"]), "variant_deltas", sub)
            self.ws.save_table(VA.gate2(self, cmp["verdicts"], cmp["systems"]), "gate2", "validation")
            self.ws.save_table(VA.gate3(self, cmp["pooled"], rec), "gate3", "validation")
            fsub = "validation/figures"
            self._fig(FG.fold_performance(cmp["per_zone"], meths), "geographic_fold_performance", fsub)
            self._fig(FG.system_recall(cmp["pooled"]), "whole_system_recall", fsub)
            self._fig(FG.short_fault(strata), "short_fault_behaviour", fsub)
            self._fig(FG.known_distance(cmp["proximity"]), "known_fault_distance", fsub)
            self._fig(FG.failure_modes(dd, cmp["regions"]), "geographic_failure_modes", fsub)
            return {"pooled": cmp["pooled"], "verdicts": cmp["verdicts"]}

        def load():
            return {"pooled": self.ws.load_table("method_pooled", "validation/metrics"),
                    "verdicts": self.ws.load_table("decision_verdicts", "reports")}
        fp = self.fp("s09", self.cfg["validation"], self.cfg["gates"], self.sh["decision"], self.scope(),
                     self.fp_of("s08_training"), self.fp_of("s01_baseline"), self.fp_of("s05_corruption"))
        return self._stage("s09_validation", fp, fn, load)

    def _save_cmp(self, cmp: dict, sub: str) -> None:
        for k, name in (("per_zone", "method_per_zone"), ("retained_fp", "retained_fp_per_zone"), ("pooled", "method_pooled"),
                        ("paired", "paired_zone_deltas"), ("bootstrap", "bootstrap"), ("systems", "system_hits"),
                        ("proximity", "known_proximity"), ("regions", "regional_dti"), ("false_length", "false_length")):
            self.ws.save_table(cmp.get(k, pd.DataFrame()), name, sub)
        self.ws.save_table(cmp["verdicts"], "decision_verdicts", "reports")

    def _ensemble_preds(self, base: str, members: list[str]) -> None:
        from .validation import MODES, load_pred

        for task in ("geo", self.prim):
            for u in self.units_of(task):
                for md in (("none",) if task == "geo" else MODES):
                    ps = [load_pred(self, v, u, md) for v in members]
                    ps = [p for p in ps if p is not None]
                    if not ps:
                        continue
                    arr = {h: np.mean([p[h] for p in ps], 0).astype(np.float16) for h in ps[0]}
                    self.ws.save_npz({**arr, "protocol_key": np.array(self.pkey(u)), "members": np.array(len(ps))},
                                     f"predictions/{base}_ens/{task}_{u.zone}__{md}.npz")

    # =========================================================================== S10
    def s10_dense_analysis(self) -> dict | None:
        from scipy import ndimage

        from ..geology.orientation import axial_difference
        from ..geology.skeleton import branch_points, endpoints, orientation_targets, skeletonize

        from .infer import strike_of
        from .validation import load_pred

        def fn():
            rows_o, rows_t, rows_c = [], [], []
            grid = self.ctx.data.grid
            for task in ("geo", self.prim):
                for u in self.units_of(task):
                    p = load_pred(self, "core", u, "none" if task == "geo" else self.cfg["validation"]["production_context"])
                    if p is None:
                        continue
                    G = u.eval_truth()
                    if not G.any():
                        continue
                    ot = orientation_targets(G, grid, scale_km=0.3, min_coherence=0.7)
                    ok = ot["ovalid"] > 0
                    st_pred, conf = strike_of(p["ocos"], p["osin"])
                    st_true = (np.degrees(0.5 * np.arctan2(ot["osin"], ot["ocos"])) % 180.0)
                    err = axial_difference(st_pred[ok], st_true[ok])
                    q = np.digitize(conf[ok], [0.25, 0.5, 0.75])
                    for b in range(4):
                        m = q == b
                        rows_o.append({"task": task, "zone": u.zone, "confidence_bin": b, "n_px": int(m.sum()),
                                       "median_error_deg": float(np.median(err[m])) if m.any() else np.nan,
                                       "share_within_15deg": float((err[m] <= 15).mean()) if m.any() else np.nan})
                    sk = skeletonize(G)
                    for head, pts in (("ep", endpoints(sk)), ("jn", branch_points(sk))):
                        h = np.nan_to_num(p[head])
                        dist_t = ndimage.distance_transform_edt(~pts) if pts.any() else np.full(G.shape, np.inf)
                        for thr in (0.3, 0.5):
                            peaks = (h >= thr) & (h == ndimage.maximum_filter(h, size=5)) & u.ev
                            dist_p = ndimage.distance_transform_edt(~peaks) if peaks.any() else np.full(G.shape, np.inf)
                            rows_t.append({"task": task, "zone": u.zone, "head": head, "threshold": thr,
                                           "n_truth": int(pts.sum()), "n_pred": int(peaks.sum()),
                                           "recall_2px": float((dist_p[pts] <= 2).mean()) if pts.any() else np.nan,
                                           "precision_2px": float((dist_t[peaks] <= 2).mean()) if peaks.any() else np.nan})
                    kern = ndimage.distance_transform_edt(~G) <= 3
                    s = np.nan_to_num(p["seg"])[u.ev]
                    k = kern[u.ev]
                    bins = np.linspace(0, 1, 11)
                    bi = np.clip(np.digitize(s, bins) - 1, 0, 9)
                    for b in range(10):
                        m = bi == b
                        rows_c.append({"task": task, "zone": u.zone, "p_lo": bins[b], "p_hi": bins[b + 1], "n_px": int(m.sum()),
                                       "share_within_300m_of_truth": float(k[m].mean()) if m.any() else np.nan})
            sub = "dense_analysis"
            O, T, C = pd.DataFrame(rows_o), pd.DataFrame(rows_t), pd.DataFrame(rows_c)
            for t, n in ((O, "orientation_error"), (T, "endpoint_junction"), (C, "reliability")):
                self.ws.save_table(t, n, sub)
            from .figures import line_panels16

            if len(C):
                c = C.groupby(["task", "p_lo"]).share_within_300m_of_truth.mean().reset_index().assign(series="M9 core")
                self.ws.save_fig(line_panels16(c, "p_lo", "share_within_300m_of_truth", "series", "task",
                                               xlabel="predicted probability bin", ylabel="share within 300 m of truth",
                                               title="Reliability of the dense map (NOT calibrated probability: "
                                                     "unlabelled faults count as misses)"), "reliability", c, sub)
            return {"orientation": O, "topology": T, "reliability": C}

        def load():
            return {k: self.ws.load_table(n, "dense_analysis") for k, n in
                    (("orientation", "orientation_error"), ("topology", "endpoint_junction"), ("reliability", "reliability"))}
        return self._stage("s10_dense_analysis", self.fp("s10", self.scope(), self.fp_of("s09_validation")), fn, load)

    # =========================================================================== S11
    def s11_tracing(self) -> dict | None:
        from ..gems3.representation import rank01

        from . import figures as FG
        from . import tracing as TC
        from .validation import load_pred

        def fn():
            curves, logs, hits, sets = [], [], [], {}
            for task in ("geo", self.prim):
                for u in self.units_of(task):
                    p = load_pred(self, "core", u, "none" if task == "geo" else self.cfg["validation"]["production_context"])
                    if p is None:
                        continue
                    cs, lg, fl = TC.unit_candidates(self, u, p, stochastic=bool(self.cfg["tracing"]["stochastic"]["enabled"]),
                                                    seed=int(self.cfg["seed"]) + u.zone)
                    self.save_trace_set(cs, u.key)
                    sets[u.key] = cs
                    logs.append(lg)
                    cu, h = TC.oracle_unit(self, u, cs, rank01(np.nan_to_num(p["seg"]), u.ev))
                    curves.append(cu)
                    if len(h):
                        hits.append(h)
                        missed = h[~h.recovered].system_id.astype(int).tolist()[:6]
                        if task == self.prim and missed:
                            self._fig(FG.tracing_failures(self, u, missed, fl["E"], cs), f"tracing_failures_{u.key.replace(':', '_')}",
                                      "tracing")
                    self.say(f"s11 {u.key}: {len(cs)} traced candidates")
            C = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
            P = TC.pooled_oracle(C)
            osub = "tracing/oracle"
            self.ws.save_table(C, "oracle_curve", osub)
            self.ws.save_table(P, "pooled_oracle", osub)
            self.ws.save_table(pd.concat(hits, ignore_index=True) if hits else pd.DataFrame(), "systems_at_max_density", osub)
            self.ws.save_table(TC.gate4(self, P), "gate4", osub)
            self.ws.save_table(pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(), "tracer_log", "tracing")
            self._save_vectors(sets, "tracing/candidates.gpkg")
            from .figures import line_panels16

            if len(P):
                self.ws.save_fig(line_panels16(P.assign(series="traces"), "density_cap", "system_recall", "series", "task",
                                               xlabel="density cap (km / 100 km^2)", ylabel="system recall (oracle)",
                                               logx=True, title="Tracing oracle: withheld-system recall vs candidate density"),
                                 "oracle_curve", P, osub)
            return {"gate": self.ws.load_table("gate4", osub), "pooled": P}

        def load():
            return {"gate": self.ws.load_table("gate4", "tracing/oracle"),
                    "pooled": self.ws.load_table("pooled_oracle", "tracing/oracle")}
        fp = self.fp("s11", self.cfg["tracing"], self.cfg["gates"]["g4_tracing"], self.profile["max_seeds_per_unit"],
                     self.scope(), self.fp_of("s09_validation"))
        return self._stage("s11_tracing", fp, fn, load)

    def _save_vectors(self, sets: dict, rel: str) -> None:
        from ..gems3.candidates import to_full

        paths, rows = [], []
        for k, cs in sets.items():
            u = self.unit(k) if k != "full:0" else None
            for i, p in enumerate(cs.paths):
                paths.append(to_full(p, u) if u is not None else p)
                r = cs.table.iloc[i]
                rows.append({"unit": k, "uid": r.get("uid"), "source": r.get("source"), "length_km": r.get("length_km"),
                             "e_mean": r.get("e_mean"), "confidence_lf": r.get("confidence_lf"),
                             "score": r.get("score_full", np.nan)})
        if paths:
            self.ws.save_lines(paths, pd.DataFrame(rows), rel, self.ctx.data.grid, "m9_candidates")

    # =========================================================================== S12
    def s12_monte_carlo(self) -> dict | None:
        from . import figures as FG
        from . import montecarlo as MC
        from . import tracing as TC
        from ..gems3.representation import rank01
        from .validation import load_pred

        def fn():
            sub = "monte_carlo"
            stab, conv, prm, extra = [], [], [], []
            for u in self.units_of(self.prim):
                p = load_pred(self, "core", u, self.cfg["validation"]["production_context"])
                cs = self.trace_set(u.key)
                if p is None or cs is None:
                    continue
                out = MC.run_unit(self, u, p, cs)
                stab.append(out["stability"])
                conv.append(out["convergence"])
                prm.append(out["params"])
                self.ws.save_npz({"occA": out["occA"].astype(np.float16), "occB": out["occB"].astype(np.float16)},
                                 f"{sub}/occupancy_{u.key.replace(':', '_')}.npz")
                full = np.full(self.ctx.data.shape, np.nan, np.float32)
                full[u.sl] = np.where(u.ev, out["occB"], np.nan)
                self.ws.save_raster({"mc_b_path_occupancy": full}, f"{sub}/occupancy_{u.key.replace(':', '_')}.tif",
                                    self.ctx.data.grid)
                xo = MC.mc_only_pool(self, u, out["occB"], cs)
                if len(xo):
                    cu, _ = TC.oracle_unit(self, u, xo, rank01(np.nan_to_num(p["seg"]), u.ev))
                    extra.append(cu.assign(pool="mc_only"))
                self._fig(FG.mc_uncertainty(self, u, out["occB"], out["stability"]), f"mc_{u.key.replace(':', '_')}", sub)
            S = pd.concat(stab, ignore_index=True) if stab else pd.DataFrame()
            self.ws.save_table(S, "stability", sub)
            self.ws.save_table(pd.concat(conv, ignore_index=True) if conv else pd.DataFrame(), "convergence", sub)
            self.ws.save_table(pd.concat(prm, ignore_index=True) if prm else pd.DataFrame(), "realization_params", sub)
            E = pd.concat(extra, ignore_index=True) if extra else pd.DataFrame()
            dens = float(self.cfg["tracing"]["oracle"]["gate_density"])
            E = E[E.density_cap == dens] if len(E) else E
            self.ws.save_table(E, "mc_only_coverage", sub)
            self.ws.save_json({"mc_a": self.cfg["monte_carlo"]["mc_a"], "realizations": self.profile["mc_realizations"],
                               "note": "heuristic sensitivity analysis; occupancy is processing stability, not a "
                                       "calibrated fault probability"}, f"{sub}/assumptions.json")
            self._gate5_rows(mc=MC.gate5_mc(self, E, None))
            return {"stability": S}

        def load():
            return {"stability": self.ws.load_table("stability", "monte_carlo")}
        fp = self.fp("s12", self.cfg["monte_carlo"], self.profile["mc_realizations"], self.fp_of("s11_tracing"))
        return self._stage("s12_monte_carlo", fp, fn, load)

    def _gate5_rows(self, mc: dict | None = None, rl: dict | None = None) -> None:
        t = self.ws.load_table("gate5", "reports")
        rows = {r["component"]: r for r in t.to_dict("records")} if len(t) else {}
        if mc is not None:
            rows["monte_carlo"] = mc
        if rl is not None:
            rows["rl"] = rl
        self.ws.save_table(pd.DataFrame(list(rows.values())), "gate5", "reports")

    # =========================================================================== S13
    def s13_rl(self) -> dict | None:
        from ..gems3.representation import rank01

        from . import rl as RL
        from . import tracing as TC
        from .validation import load_pred

        def fn():
            preds = {}
            for task in ("geo", self.prim):
                for u in self.units_of(task):
                    p = load_pred(self, "core", u, "none" if task == "geo" else self.cfg["validation"]["production_context"])
                    if p is not None:
                        preds[u.key] = p
            rows, rng = [], np.random.default_rng(int(self.cfg["seed"]) + 1300)
            tc = self.cfg["tracing"]
            min_px = max(int(round(float(tc["min_len_km"]) / 0.112)), 2)
            for u in self.units_of(self.prim):
                if u.key not in preds:
                    continue
                src = RL.episode_sources(self, u, preds)
                ev = u.scored_mask()
                E, theta, conf = TC.fields(preds[u.key], ev, float(tc["evidence_sigma_px"]))
                gap_thr = float(np.quantile(E[ev], float(tc["gap_quantile"]))) if ev.any() else 0.0
                sd = TC.seeds(E, ev, float(tc["seed_quantile"]), int(tc["seed_nms_px"]), int(self.profile["max_seeds_per_unit"]))
                rc = {**self.cfg["rl"], "max_gap_px": tc["max_gap_px"], "gap_thr": gap_thr}
                pools = {"beam": self.trace_set(u.key)}
                if src:
                    X, y = RL.collect_bc(self, src, int(self.profile["rl_episodes"]), rng)
                    bc = RL.train_bc(X, y, int(self.cfg["seed"])) if len(X) else None
                    if bc is not None:
                        paths = RL.rollout(lambda o: bc.predict(o[None]), E, theta, ev, sd, rc)
                        pools["bc"] = self._pool(paths, "rl_bc", E, theta, conf, preds[u.key], u, gap_thr, min_px)
                    if self.cfg["rl"]["algorithm"] in ("ppo", "dqn"):
                        mdl = RL.train_sb3(self, src, self.cfg["rl"]["algorithm"], int(self.profile["rl_episodes"]) * 50,
                                           int(self.cfg["seed"]))
                        if mdl is not None:
                            paths = RL.rollout(lambda o: mdl.predict(o, deterministic=True)[0], E, theta, ev, sd, rc)
                            pools[self.cfg["rl"]["algorithm"]] = self._pool(paths, "rl_sb3", E, theta, conf, preds[u.key], u,
                                                                            gap_thr, min_px)
                    if self.cfg["rl"].get("learned_edge_cost"):
                        lr = RL.learned_edge_cost(src, preds[u.key], int(self.cfg["seed"]))
                        if lr is not None:
                            El = RL.apply_edge_cost(lr, E)
                            paths, r_ = TC.trace_unit(El, theta, ev, tc, int(self.profile["max_seeds_per_unit"]),
                                                      source="learned_cost")
                            pools["learned_edge_cost"] = self._pool(paths, "learned_cost", E, theta, conf, preds[u.key], u,
                                                                    gap_thr, min_px)
                dense = rank01(np.nan_to_num(preds[u.key]["seg"]), u.ev)
                for name, cs in pools.items():
                    if cs is None or not len(cs):
                        continue
                    cu, _ = TC.oracle_unit(self, u, cs, dense)
                    rows.append(cu.assign(tracer=name))
            C = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            if len(C):
                C = C.groupby(["task", "tracer", "density_cap"]).agg(systems_recovered=("systems_recovered", "sum"),
                                                                    n_systems=("n_systems", "sum")).reset_index()
                C["system_recall"] = C.systems_recovered / C.n_systems.clip(lower=1)
            self.ws.save_table(C, "tracer_comparison", "rl")
            self._gate5_rows(rl=RL.gate5_rl(self, C))
            return {"comparison": C}

        def load():
            return {"comparison": self.ws.load_table("tracer_comparison", "rl")}
        fp = self.fp("s13", self.cfg["rl"], self.profile["rl_episodes"], self.fp_of("s11_tracing"))
        return self._stage("s13_rl", fp, fn, load)

    def _pool(self, paths, source, E, theta, conf, pred, u, gap_thr, min_px):
        from . import tracing as TC

        rows = [{"source": source, "seed_r": int(p[0, 0]) if len(p) else 0, "seed_c": int(p[0, 1]) if len(p) else 0,
                 "beam_score": np.nan, "parent": -1} for p in paths]
        T = TC.describe(paths, rows, E, theta, conf, np.nan_to_num(pred["ep"]), np.nan_to_num(pred["jn"]), u.key,
                        (u.sl[0].start, u.sl[1].start), gap_thr)
        return TC.dedup(paths, T, E.shape, int(self.cfg["tracing"]["dedup"]["radius_px"]),
                        float(self.cfg["tracing"]["dedup"]["overlap"]), min_px)

    # =========================================================================== S14
    def _require_tracing(self) -> None:
        g = self.gate("gate4", "tracing/oracle")
        if g == "CONTINUE" or self.cfg["tracing"].get("force_rank"):
            return
        raise GateClosed(f"Gate 4 (structured tracing) is {g}: candidate rankers are not justified (read "
                         "tracing/oracle/gate4.csv; tracing.force_rank=True overrides)")

    def s14_candidate_features(self) -> dict | None:
        from ..gems3.official import unit_score

        from . import candidates as CF
        from . import figures as FG
        from .validation import load_pred

        def fn():
            self._require_tracing()
            mc = self.ws.load_table("stability", "monte_carlo")
            tabs, provs, sets = [], [], {}
            for task in ("geo", self.prim):
                for u in self.units_of(task):
                    cs = self.trace_set(u.key)
                    if cs is None or not len(cs):
                        continue
                    pp = load_pred(self, "core", u, "none" if task == "geo" else self.cfg["validation"]["production_context"])
                    pn = load_pred(self, "core", u, "none")
                    leg = {"b0": unit_score(self.ctx, self.models[0], u), "b1": unit_score(self.ctx, self.models[1], u)}
                    leg = {k: v for k, v in leg.items() if v is not None}
                    T, pv = CF.describe(self, u, cs, pp, pn, with_labels=True, mc=mc[mc.unit == u.key] if len(mc) else None,
                                        legacy=leg)
                    self.ws.save_table(T, u.key.replace(":", "_"), "candidates/units")
                    tabs.append(T)
                    provs.append(pv)
                    sets[u.key] = cs
                    self.say(f"s14 {u.key}: {len(T)} rows")
            T = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
            self.ws.save_table(T, "candidate_table", "candidates")
            g = CF.groups(T) if len(T) else {}
            self.ws.save_table(pd.DataFrame([{"group": k, "feature": c} for k, v in g.items() for c in v]),
                               "feature_groups", "candidates")
            self.ws.save_table(pd.concat(provs, ignore_index=True).drop_duplicates("feature") if provs else pd.DataFrame(),
                               "feature_provenance", "candidates")
            if len(T):
                self.ws.save_table(T.groupby(["task", "label"]).size().rename("n").reset_index(), "label_balance", "candidates")
                self._fig(FG.fault_normal_profiles(T), "candidate_fault_normal_profiles", "candidates")
            self._save_vectors(sets, "candidates/candidates.gpkg")
            return {"table": T}

        def load():
            return {"table": self.ws.load_table("candidate_table", "candidates")}
        fp = self.fp("s14", self.cfg["candidates"], self.fp_of("s11_tracing"), self.fp_of("s12_monte_carlo"),
                     self.fp_of("s03_spectral"))
        return self._stage("s14_candidate_features", fp, fn, load)

    # =========================================================================== S15
    def s15_gbm_benchmark(self) -> dict | None:
        from . import figures as FG
        from . import gbm as GB

        def fn():
            self._require_tracing()
            T = self.ws.load_table("candidate_table", "candidates")
            if not len(T):
                raise GateClosed("no candidate table: run section 14")
            sets = {k: self.trace_set(k) for k in T.unit.unique()}
            sets = {k: v for k, v in sets.items() if v is not None}
            out = GB.run(self, T, sets)
            sub = "gbm"
            for k in ("candidate_metrics", "system_recall", "grid", "summary", "per_zone", "audit", "availability",
                      "stability", "choice"):
                self.ws.save_table(out[k], k, sub)
            self.ws.save_table(out["table"], "candidate_scores", sub)
            self.ws.save_json(out["methods"], f"{sub}/score_columns.json")
            from .candidates import feature_sets

            fs = feature_sets(T).get("all", [])
            imps = []
            for lib in [a.library for a in out["availability"].itertuples() if a.available]:
                m = GB.fit_full(T, lib, fs, self.cfg["gbm"], self.prim, int(self.cfg["seed"]))
                if m is None:
                    continue
                import joblib

                from ..artifacts.atomic import atomic_save

                atomic_save(self.ws.root / sub / "models" / f"{lib}_all.joblib", lambda t, m=m: joblib.dump(m, t))
                imps.append(GB.importance(m, fs).assign(library=lib))
            self.ws.save_table(pd.concat(imps, ignore_index=True) if imps else pd.DataFrame(), "feature_importance", sub)
            self._fig(FG.gbm_comparison(out["summary"]), "library_comparison", sub)
            return {"summary": out["summary"], "choice": out["choice"]}

        def load():
            return {"summary": self.ws.load_table("summary", "gbm"), "choice": self.ws.load_table("choice", "gbm")}
        fp = self.fp("s15", self.cfg["gbm"], self.cfg["validation"]["area_fracs"], self.fp_of("s14_candidate_features"))
        return self._stage("s15_gbm_benchmark", fp, fn, load)

    # =========================================================================== S16
    def s16_fusion(self) -> dict | None:
        from . import figures as FG
        from . import fusion as FU
        from . import validation as VA

        def fn():
            T = self.ws.load_table("candidate_scores", "gbm")
            ch = self.ws.load_table("choice", "gbm")
            cols = self.read_json("gbm/score_columns.json") or {}
            col = cols.get(str(ch.method.iloc[0])) if len(ch) and isinstance(ch.method.iloc[0], str) else None
            out = FU.run(self, T if len(T) else None, col)
            sub = "fusion"
            for k in ("grid", "summary", "per_zone", "ring"):
                self.ws.save_table(out[k], k, sub)
            cmp = FU.compare(self)
            self._save_cmp(cmp, "validation/metrics")
            g6 = FU.gate6(self, cmp["verdicts"])
            self.ws.save_table(g6, "gate6", sub)
            S = out["summary"]
            gain = None
            pooled = cmp["pooled"]
            if len(pooled):
                pp = pooled[pooled.task == self.prim].set_index("method").dti_pooled
                ref = f"s01_{self.cfg['baseline']['champion']}"
                if "m9_fuse_D+L" in pp and ref in pp:
                    gain = float(pp["m9_fuse_D+L"] - pp[ref])
            self.ws.save_table(VA.gate2(self, cmp["verdicts"], cmp["systems"], gain), "gate2", "validation")
            gs = self.ws.load_table("summary", "gbm")
            if len(gs):
                lib = str(ch.library.iloc[0]) if len(ch) and "library" in ch else ""
                a = gs[(gs.task == self.prim) & (gs.method == f"{lib}_all")]
                b = gs[(gs.task == self.prim) & (gs.method == f"{lib}_all_mc")]
                if len(a) and len(b):
                    from .montecarlo import gate5_mc

                    self._gate5_rows(mc=gate5_mc(self, self.ws.load_table("mc_only_coverage", "monte_carlo"),
                                                 float(b.dti_crossfit.iloc[0] - a.dti_crossfit.iloc[0])))
            params = self.read_json("validation/dense_selection.json") or {}
            for r in (S.to_dict("records") if len(S) else []):
                params.setdefault(r["method"], {})[r["task"]] = {"params_all": json.loads(r["params_all"]),
                                                                 "dti_crossfit": r["dti_crossfit"]}
            if len(gs) and len(ch) and isinstance(ch.method.iloc[0], str):
                for r in gs[gs.method == ch.method.iloc[0]].to_dict("records"):
                    params.setdefault("m9_structured", {})[r["task"]] = {"params_all": json.loads(r.get("params_all") or "{}"),
                                                                         "dti_crossfit": r["dti_crossfit"]}
            self.ws.save_json(json.loads(json.dumps(params, default=float)), f"{sub}/method_parameters.json")
            u = self.units_of(self.prim)[0] if self.units_of(self.prim) else None
            if u is not None:
                from ..m7.rasterize import load_heldout
                from .validation import load_pred

                p = load_pred(self, "core", u, self.cfg["validation"]["production_context"])
                self._fig(FG.dense_vs_structured(self, u, None if p is None else np.nan_to_num(p["seg"]),
                                                 load_heldout(self.ws, "m9_structured", u)), "dense_vs_structured", sub)
            return {"summary": S, "gate6": g6, "verdicts": cmp["verdicts"]}

        def load():
            return {"summary": self.ws.load_table("summary", "fusion"), "gate6": self.ws.load_table("gate6", "fusion"),
                    "verdicts": self.ws.load_table("decision_verdicts", "reports")}
        fp = self.fp("s16", self.cfg["fusion"], self.sh["decision"], *[self.fp_of(s) for s in
                     ("s09_validation", "s11_tracing", "s15_gbm_benchmark", "s12_monte_carlo", "s13_rl")])
        return self._stage("s16_fusion", fp, fn, load)

    # =========================================================================== S17
    def s17_inference(self) -> dict | None:
        from . import candidates as CF
        from . import gbm as GB
        from . import submission as SU
        from . import tracing as TC

        def fn():
            out = SU.full_dense(self, "core")
            if out is None:
                raise GateClosed("no trained core checkpoint: run section 08")
            info = {"units": out["units"], "context": out["context"]}
            if self.profile["full_region_traces"]:
                fu = SU.full_unit_for_tracing(self)
                cs, lg, _ = TC.unit_candidates(self, fu, out["heads"], stochastic=False)
                ch = self.ws.load_table("choice", "gbm")
                T = self.ws.load_table("candidate_table", "candidates")
                if len(ch) and isinstance(ch.method.iloc[0], str) and len(T):
                    lib = str(ch.library.iloc[0])
                    fs = CF.feature_sets(T).get("all", [])
                    maps, _ = self.m7view.full_maps()
                    D, _ = CF.describe(self, fu, cs, out["heads"], None, with_labels=False,
                                       legacy={"b0": maps[0], "b1": maps[1]})
                    m = GB.fit_full(T, lib, fs, self.cfg["gbm"], self.prim, int(self.cfg["seed"])) \
                        if lib in ("xgboost", "lightgbm", "catboost") else None
                    if m is not None and len(D):
                        X = D.reindex(columns=fs).astype(float).to_numpy()
                        cs.table["score_full"] = m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.predict(X)
                    else:
                        cs.table["score_full"] = cs.table.confidence_lf
                self.save_trace_set(cs, "full:0")
                self._save_vectors({"full:0": cs}, "tracing/full_region.gpkg")
                info["n_full_candidates"] = len(cs)
            self.ws.save_json(info, "inference/inference_manifest.json")
            return info

        def load():
            return self.read_json("inference/inference_manifest.json")
        fp = self.fp("s17", self.cfg["inference"], self.profile["full_region_traces"], self.fp_of("s08_training"),
                     self.fp_of("s15_gbm_benchmark"))
        return self._stage("s17_inference", fp, fn, load)

    # =========================================================================== S18 / S19
    def s18_submission(self) -> dict | None:
        from . import submission as SU

        def fn():
            out = SU.run(self)
            self.ws.save_table(out["index"], "submission_index", "submissions")
            return out

        def load():
            return {"index": self.ws.load_table("submission_index", "submissions")}
        fp = self.fp("s18", self.cfg["submission"], self.sh["submission"], self.fp_of("s16_fusion"), self.fp_of("s17_inference"))
        return self._stage("s18_submission", fp, fn, load)

    def s19_reports(self) -> dict:
        """Gates, report, model registry (always recomputed from saved tables)."""
        from . import gates as GA
        from . import report as RP

        self.ws.begin("s19_reports")
        gates = GA.collect(self)
        self.ws.save_table(gates, "gates", "reports")
        self.ws.save_json(json.loads(gates.to_json(orient="records")), "reports/gates.json")
        reg = RP.registry(self)
        self.ws.save_table(reg, "model_registry", "reports/model_registry")
        self.ws.save_json(json.loads(reg.to_json(orient="records")), "reports/model_registry/model_registry.json")
        self.ws.save_text(RP.report(self, gates), "reports/M9_GEOFAULT_PRODUCTION_REPORT.md")
        self.ws.section_manifest("s19_reports", {})
        self.ws.finish("s19_reports", fingerprint(time.time()))
        self.push()
        return {"gates": gates, "registry": reg}

    # =========================================================================== run + summary
    def run(self, stage: str | None = None) -> dict:
        """Every enabled section of ``stage`` (PREPARE ... SUBMIT; None = everything) in order."""
        if not self.cfg.get("enabled", True):
            self.say("disabled (M9_CONFIG['enabled'] = False)")
            return {}
        t0 = time.perf_counter()
        out = {"s00": self.s00_setup()}
        for name in sections_of(self.cfg, stage):
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
        print(f"[m9] outputs (local): {self.ws.root}")
        if drive:
            print(f"[m9] outputs (Drive): {drive}")
        for k, v in rows.items():
            print(f"  {k:<28} {v['manifest']}")
        return rows
