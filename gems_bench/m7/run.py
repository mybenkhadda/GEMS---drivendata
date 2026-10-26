"""M7 orchestration: ``pipe.m7(M7_CONFIG)`` -> one method per notebook section, fingerprinted and resumable.

    m7 = pipe.m7(M7_CONFIG)
    m7.s00_setup(); m7.s01_baseline(); m7.s02_failure_analysis(); m7.s03_candidates(); m7.s04_oracle(); ...
    m7.run()                       # every enabled section in order (stops 10-12 after an oracle NO-GO)

Artifact recovery (read-only): B0 / B1 FULL OOF + discovery maps through the M6 inventory (memory-mapped, staged from
Drive); full-region maps from an earlier GEMS3 run (``legacy_gems3_dir``) when their fingerprints match the members'
checkpoints, otherwise fold-average inference (GPU) or the stitched OOF fallback. Nothing is retrained.

A stage whose fingerprint (its configuration + upstream stage fingerprints + map fingerprints + M7 version) and
outputs match is reused; ``overwrite_existing=True`` recomputes. Every stage writes ``manifests/sections/<s>.json``
(fingerprint, files, Drive paths) and pushes its folder to Drive.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..pipeline.m6 import fingerprint
from .config import load_m7_config
from .workspace import M7_VERSION, M7Workspace


class GateClosed(RuntimeError):
    """A section refuses to run because its scientific gate is closed (not a failure)."""


class M7Research:
    def __init__(self, pc=None, config: dict | None = None, *, ctx=None, local_root: Path | None = None,
                 data_root: Path | None = None, config_path: Path | None = None):
        from ..gems3.run import GEMS3Research

        self.cfg = load_m7_config(config, config_path)
        self.pc = pc
        self.local = Path(local_root or (pc.local if pc is not None else Path.cwd()))
        drive = getattr(pc, "drive", None) if pc is not None else None
        self.cfg["_drive_root"] = str(Path(drive) / self.cfg["output_dir"]) if drive else None
        self.ws = M7Workspace(self.local / self.cfg["output_dir"], self.cfg)
        self.cache = self.local / self.cfg["cache_dir"]
        self.cache.mkdir(parents=True, exist_ok=True)
        sub = self.cfg["submission"]
        self.g3 = GEMS3Research(pc, {"models": list(self.cfg["models"]), "tasks": ["geo", "disc_official"],
                                     "primary_task": "disc_official",
                                     "output_dir": f"{self.cfg['output_dir']}/gems3_engine", "cache_dir": "cache/gems3",
                                     "overwrite_existing": bool(self.cfg.get("overwrite_existing")),
                                     "submission": {"source": sub["source"], "device": sub["device"]}},
                                ctx=ctx, local_root=local_root, data_root=data_root)
        self._units = None
        self._memo: dict = {}
        self.log: list[dict] = []

    # =========================================================================== context helpers
    @property
    def ctx(self):
        return self.g3.ctx

    @property
    def models(self) -> list[str]:
        return self.g3.models

    def load(self) -> None:
        self.g3.load()

    def _say(self, msg: str) -> None:
        print(f"[m7] {msg}")

    @property
    def units(self) -> list:
        if self._units is None:
            from .protocol import m7_units

            self.load()
            self._units = m7_units(self.ctx, self.cfg)
        return self._units

    def units_of(self, task: str) -> list:
        return [u for u in self.units if u.task == task]

    def unit(self, key: str):
        if key == "full:0":
            return self.full_unit()
        return next((u for u in self.units if u.key == key), None)

    def zones(self, task: str) -> list[int]:
        return [u.zone for u in self.units_of(task)]

    def full_unit(self):
        from ..gems3.official import Unit

        if "full_unit" not in self._memo:
            H, W = self.ctx.data.shape
            self._memo["full_unit"] = Unit("full", 0, (slice(0, H), slice(0, W)), np.asarray(self.ctx.data.valid, bool),
                                           np.asarray(self.ctx.data.labels, bool), None, True,
                                           np.asarray(self.ctx.data.labels, bool))
        return self._memo["full_unit"]

    def template(self, u) -> np.ndarray:
        k = ("tmpl", u.key)
        if k not in self._memo:
            from .protocol import template_crop

            self._memo[k] = template_crop(self.ctx, u)
        return self._memo[k]

    def elig(self, u, ring_px: int = 0) -> np.ndarray:
        """Eligible budget domain of unit ``u`` with a ring of ``ring_px`` (cached)."""
        from .protocol import eligible

        k = ("elig", u.key, int(ring_px or 0))
        if k not in self._memo:
            ev = self.cfg["evaluation"]
            self._memo[k] = eligible(u, self.template(u), int(ring_px or 0), ev["budget_domain"],
                                     bool(ev["include_template_mask"]))
        return self._memo[k]

    def pkey(self, u) -> str:
        from .protocol import protocol_key

        k = ("pkey", u.key)
        if k not in self._memo:
            ev = self.cfg["evaluation"]
            self._memo[k] = protocol_key(u, self.template(u), ev["budget_domain"], bool(ev["include_template_mask"]))
        return self._memo[k]

    def protocol_keys(self) -> pd.DataFrame:
        from .protocol import protocol_table

        ev = self.cfg["evaluation"]
        return protocol_table(self.units, {u.key: self.template(u) for u in self.units}, ev["budget_domain"],
                              bool(ev["include_template_mask"]))

    def transfer(self, q: np.ndarray, u_from, u_to) -> np.ndarray:
        if u_from.sl == u_to.sl:
            return q
        full = np.zeros(self.ctx.data.shape, np.float32)
        full[u_from.sl] = q
        return full[u_to.sl]

    def cache_get(self, key: str, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    # -- Drive -------------------------------------------------------------------
    def stage_in(self, rels: list[str]) -> None:
        if self.pc is not None and rels:
            try:
                self.pc.stage_in([r for r in rels if r])
            except Exception as e:
                self._say(f"stage_in skipped: {type(e).__name__}: {e}")

    def push(self, rels: list[str] | None = None) -> None:
        if self.pc is None or not self.cfg.get("push_to_drive", True):
            return
        try:
            rels = rels or [Path(self.ws.root).resolve().relative_to(self.local.resolve()).as_posix()]
            self.pc.push(rels)
        except Exception as e:                                         # never lose results to a sync error
            self._say(f"push skipped: {type(e).__name__}: {e}")

    def read_json(self, rel: str) -> dict | None:
        p = self.ws.root / rel
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    # -- fingerprints / stages -----------------------------------------------------
    def fp(self, *parts) -> str:
        self.load()
        return fingerprint(M7_VERSION, self.g3.m6.map_fp, self.cfg["models"], self.cfg["evaluation"], *parts)

    def fp_of(self, stage: str) -> str | None:
        r = self.ws.unit_record(stage)
        return r.get("fingerprint") if r else None

    def _stage(self, name: str, fp: str, fn, load=None, optional: bool = True):
        t0 = time.perf_counter()
        if self.ws.valid(name, fp):
            self._say(f"{name}: reused (fingerprint {fp}) -> {self.ws.root}")
            self.log.append({"stage": name, "outcome": "REUSED", "seconds": 0.0, "fingerprint": fp})
            return load() if load else None
        self.ws.begin(name)
        try:
            out = fn()
        except GateClosed as e:
            self._say(f"{name} SKIPPED (gate): {e}")
            self.log.append({"stage": name, "outcome": "GATE_CLOSED", "reason": str(e), "seconds": 0.0})
            self._save_log()
            return None
        except Exception as e:
            self.log.append({"stage": name, "outcome": "FAILED", "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-4000:], "seconds": round(time.perf_counter() - t0, 1)})
            self._save_log()
            if not optional:
                raise
            self._say(f"{name} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
            return None
        dt = round(time.perf_counter() - t0, 1)
        self.ws.section_manifest(name, {"fingerprint": fp, "experiment_id": self.ws.experiment_id(name, fp),
                                        "seconds": dt, "config": self._section_cfg(name)})
        self.ws.finish(name, fp, seconds=dt)
        self.log.append({"stage": name, "outcome": "COMPUTED", "seconds": dt, "fingerprint": fp})
        self._say(f"{name}: computed in {dt:.0f}s -> {self.ws.root}")
        self._save_log()
        self.push()
        return out

    def _section_cfg(self, name: str) -> dict:
        keys = {"s01": ["baseline"], "s02": ["failure"], "s03": ["candidates"], "s04": ["oracle"], "s05": ["dem"],
                "s06": ["dem"], "s07": ["dem"], "s08": ["map_completion"], "s09": ["map_completion"],
                "s10": ["features"], "s11": ["ranking"], "s12": ["rasterization"], "s13": ["comparison"],
                "s14": ["decision"], "s15": ["submission"], "s16": ["probes"]}.get(name[:3], [])
        return json.loads(json.dumps({k: self.cfg.get(k) for k in keys + ["evaluation"]}, default=str))

    def _save_log(self) -> None:
        from ..artifacts.atomic import atomic_write_json

        atomic_write_json(self.ws.root / "manifests" / "stage_log.json", self.log)

    def enabled(self, section: str) -> bool:
        return bool(self.cfg["sections"].get(section))

    # =========================================================================== shared loaders
    def baseline_selection(self) -> dict | None:
        if "s01_selection" not in self._memo:
            self._memo["s01_selection"] = self.read_json("recovered_baselines/baseline_selection.json")
        return self._memo["s01_selection"]

    def best_baseline_dti(self, task: str) -> float:
        t = self.ws.load_immutable_table("baseline_table", "recovered_baselines")
        if not len(t):
            return np.nan
        g = t[(t.task == task) & ~t.method.isin(["b0_raw", "b1_raw"])]
        return float(g.dti_crossfit.max()) if len(g) else np.nan

    def load_unit_set(self, key: str):
        from ..gems3.candidates import CandidateSet

        k = key.replace(":", "_")
        z = self.ws.root / "candidate_generation" / "vectors" / "units" / f"{k}.npz"
        t = self.ws.load_table(f"units/{k}", "candidate_generation/tables")
        if not z.exists() or not len(t):
            return None
        return CandidateSet.from_npz(t.sort_values("cand_id").reset_index(drop=True), np.load(z))

    def unit_sets(self, tasks=None) -> dict:
        tasks = tasks or ["geo", self.cfg["evaluation"]["primary_task"]]
        out = {}
        for t in tasks:
            for u in self.units_of(t):
                cs = self.load_unit_set(u.key)
                if cs is not None:
                    out[u.key] = cs
        return out

    def dem_mosaic(self):
        from .dem_morph import DemMosaic, mosaic_path

        p = mosaic_path(self)
        if not p.exists():
            self.stage_in([p.relative_to(self.local).as_posix()])
        return DemMosaic(p) if p.exists() else None

    def dem_layer(self) -> np.ndarray | None:
        """Percentile-normalised DEM scarp evidence on the full grid (NaN outside coverage), or None."""
        def build():
            mos = self.dem_mosaic()
            if mos is None:
                return None
            b = next((x for x in self.cfg["dem"]["candidate_layers"] if mos.has(x)), None)
            if b is None:
                return None
            a = mos.band(b)
            cov = mos.coverage(self.cfg["dem"]["min_coverage"])
            v = np.sort(a[cov & np.isfinite(a)])
            if not v.size:
                return None
            return np.where(cov & np.isfinite(a), np.searchsorted(v, a) / v.size, np.nan).astype(np.float32)
        return self.cache_get("dem_layer", build)

    def recover_legacy_full_maps(self) -> pd.DataFrame:
        """Copy earlier GEMS3 full-region member maps into the engine cache when their fingerprints still match."""
        from ..gems3.submission import _fp

        legacy = self.local / self.cfg["legacy_gems3_dir"] / "predictions"
        self.stage_in([f"{self.cfg['legacy_gems3_dir']}/predictions"])
        dst = self.g3.ws.root / "predictions"
        rows = []
        try:
            members = self.g3.members()
        except Exception as e:
            return pd.DataFrame([{"status": f"members unavailable: {e}"}])
        for mem in members:
            src_t, src_j = legacy / f"{mem['model']}__full_region_raw.tif", legacy / f"{mem['model']}__full_region_raw.json"
            st = "absent"
            if src_t.exists() and src_j.exists():
                meta = json.loads(src_j.read_text(encoding="utf-8"))
                want = _fp(mem, meta.get("source", self.cfg["submission"]["source"]))
                if meta.get("fingerprint") == want and meta.get("source") == self.cfg["submission"]["source"]:
                    dst.mkdir(parents=True, exist_ok=True)
                    if not (dst / src_t.name).exists():
                        shutil.copy2(src_t, dst / src_t.name)
                        shutil.copy2(src_j, dst / src_j.name)
                    st = "recovered"
                else:
                    st = "fingerprint or source differs (not reused)"
            rows.append({"model": mem["model"], "legacy_map": str(src_t), "status": st})
        return pd.DataFrame(rows)

    def full_maps(self):
        """Full-region member maps for export: cached / recovered maps, else fold-average inference (GPU) or the
        stitched OOF maps (``submission.source = oof_stitched``)."""
        if "full_maps" not in self._memo:
            self.recover_legacy_full_maps()
            self._memo["full_maps"] = self.g3.full_region_maps()
        return self._memo["full_maps"]

    def cached_full_maps(self) -> list | None:
        """Full-region maps only if they already exist (engine cache or recovered GEMS3 maps); never runs a CNN.
        Analysis sections use this; ``None`` means: use the stitched OOF maps."""
        from ..ensemble import read_map

        self.recover_legacy_full_maps()
        d = self.g3.ws.root / "predictions"
        paths = [d / f"{m}__full_region_raw.tif" for m in self.models]
        return [read_map(p) for p in paths] if all(p.exists() for p in paths) else None

    # =========================================================================== S00
    def s00_setup(self) -> dict:
        """Environment, artifact recovery and compatibility checks (read-only, always re-run)."""
        self.load()
        ctx, d = self.ctx, self.ctx.data
        gpu = self.pc.gpu() if self.pc is not None else {}
        checks = {"crs_epsg": d.grid.crs.to_epsg() if hasattr(d.grid.crs, "to_epsg") else str(d.grid.crs),
                  "shape": list(d.shape), "pixel_m": [d.grid.pixel_km_x * 1000, d.grid.pixel_km_y * 1000],
                  "template_same_grid": bool(d.template_matches()), "n_bands": len(d.band_names),
                  "valid_frac": float(d.valid.mean()), "template_valid_frac": float(np.asarray(d.template_valid).mean()),
                  "valid_but_template_nan_px": int((d.valid & ~np.asarray(d.template_valid, bool)).sum()),
                  "template_finite_but_invalid_px": int((~d.valid & np.asarray(d.template_valid, bool)).sum()),
                  "label_px": int(d.labels.sum()), "n_folds": int(ctx.folds.n_folds),
                  "n_splits": int(ctx.splits.n_splits)}
        fails = []
        if checks["crs_epsg"] != 32611:
            fails.append(f"CRS {checks['crs_epsg']} != EPSG:32611")
        if not checks["template_same_grid"]:
            fails.append("example_submission.tif grid differs from the feature grid")
        need = sorted({s["feature"] for s in self.cfg["candidates"]["sources"] if s.get("feature")} |
                      set(self.cfg["map_completion"]["evidence_layers"]) |
                      set((self.cfg["failure"].get("gradient_features") or {}).values()) |
                      set((self.cfg["failure"].get("morphology_features") or {}).values()))
        feats = pd.DataFrame([{"feature": f, "in_catalogue": f in getattr(ctx.store, "specs", {}),
                               "cached": bool(ctx.store.has(f))} for f in need])
        bands = pd.DataFrame([{"band": s.get("band"), "source": s["name"],
                               "available": s.get("band") in d.band_names} for s in self.cfg["candidates"]["sources"]
                              if s.get("band")])
        legacy = self.local / self.cfg["legacy_gems3_dir"]
        self.stage_in([f"{self.cfg['legacy_gems3_dir']}/manifests", f"{self.cfg['legacy_gems3_dir']}/proposals"])
        leg = pd.DataFrame([{"path": str(p.relative_to(self.local)), "bytes": p.stat().st_size}
                            for p in sorted(legacy.rglob("*")) if p.is_file()][:500]) if legacy.exists() else pd.DataFrame()
        self.ws.begin("s00_setup")
        self.ws.save_table(self.g3.m6.inv if self.g3.m6.inv is not None else pd.DataFrame(), "model_inventory", "manifests")
        self.ws.save_table(feats, "feature_compatibility", "manifests")
        self.ws.save_table(bands, "band_compatibility", "manifests")
        self.ws.save_table(leg, "legacy_gems3_artifacts", "manifests")
        self.ws.save_table(self.protocol_keys(), "protocol_keys", "manifests")
        from .._version import provenance

        self.ws.save_json({"checks": checks, "failures": fails, "gpu": gpu, "code": provenance(),
                           "map_fingerprints": self.g3.m6.map_fp, "models": self.models,
                           "m7_version": M7_VERSION, "config": json.loads(json.dumps(self.cfg, default=str)),
                           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, "manifests/setup.json")
        self.ws.save_json(json.loads(json.dumps(self.cfg, default=str)), "manifests/m7_config.json")
        self.ws.section_manifest("s00_setup", {"failures": fails})
        self.ws.finish("s00_setup", fingerprint(time.time()))
        self.push()
        if fails:
            raise RuntimeError(f"setup checks failed: {fails}")
        return {"checks": checks, "features": feats, "bands": bands, "legacy": leg, "gpu": gpu}

    # =========================================================================== S01
    def s01_baseline(self) -> dict | None:
        from . import baseline as B
        from .rasterize import save_heldout

        cfg = self.cfg

        def fn():
            out = B.run(self)
            sub = "recovered_baselines"
            fp = self.fp("s01", cfg["baseline"])
            self.ws.save_table(out["grid"], "baseline_grid_components", sub)
            self.ws.save_table(out["per_zone"], "baseline_per_zone", sub)
            self.ws.save_immutable_table(out["table"], "baseline_table", sub, fp)
            self.ws.save_table(out["recorded"], "recorded_vs_recomputed", sub)
            self.ws.save_table(out["ledger"], "area_ledger", sub)
            self.ws.save_table(out["retained_fp"], "retained_fp_rescore", sub)
            self.ws.save_table(out["protocol"], "protocol_keys", sub)
            self.ws.save_json(json.loads(json.dumps(out["selection"], default=float)), f"{sub}/baseline_selection.json")
            self._memo["s01_selection"] = out["selection"]
            mm = B.MemberMaps(self.ctx, self.models)
            H, W = self.ctx.data.shape
            for task, ms in out["selection"].items():
                for m in ms:
                    layers = {}
                    geo_full = np.full((H, W), np.nan, np.float32) if task == "geo" else None
                    for u in self.units_of(task):
                        q = B.heldout_map(self, task, m, u.zone, mm)
                        if q is None:
                            continue
                        save_heldout(self.ws, f"s01_{m}", u, q, self.pkey(u))
                        if m in B.RAW:
                            continue
                        tgt = geo_full if geo_full is not None else np.full((H, W), np.nan, np.float32)
                        v = tgt[u.sl]
                        v[u.ev] = q[u.ev]
                        if geo_full is None:
                            layers[f"{task}_split{u.zone}"] = tgt
                    if geo_full is not None:
                        layers[f"{task}_heldout"] = geo_full
                    if layers:
                        self.ws.save_raster(layers, f"{sub}/heldout/{task}__{m}.tif", self.ctx.data.grid)
            B.figures(self, out)
            return out

        def load():
            return {"table": self.ws.load_immutable_table("baseline_table", "recovered_baselines"),
                    "recorded": self.ws.load_table("recorded_vs_recomputed", "recovered_baselines"),
                    "selection": self.baseline_selection()}
        return self._stage("s01_baseline", self.fp("s01", cfg["baseline"]), fn, load, optional=False)

    # =========================================================================== S02
    def s02_failure_analysis(self) -> dict | None:
        from . import failure as F

        def fn():
            out = F.run(self)
            sub = "failure_analysis/tables"
            for k, name in (("attributes", "system_attributes"), ("detection", "system_detection"),
                            ("systems", "system_table"), ("stratified", "recall_by_stratum"),
                            ("density", "prediction_density_by_region"), ("geo_vs_disc", "geo_vs_disc")):
                self.ws.save_table(out[k], name, sub)
            self.ws.save_table(pd.DataFrame([{"task": t, **v} for t, v in out["summary"].items()]), "summary", sub)
            F.figures(self, out)
            F.export_case_studies(self, out["systems"])
            return out

        def load():
            return {k: self.ws.load_table(n, "failure_analysis/tables") for k, n in
                    (("systems", "system_table"), ("stratified", "recall_by_stratum"), ("summary", "summary"))}
        return self._stage("s02_failure_analysis", self.fp("s02", self.cfg["failure"], self.fp_of("s01_baseline")),
                           fn, load)

    def s02b_case_studies_with_candidates(self) -> pd.DataFrame | None:
        """Re-export the case studies with the section-03 candidate lines drawn (after section 03)."""
        from . import failure as F

        def fn():
            systab = self.ws.load_table("system_table", "failure_analysis/tables")
            H, W = self.ctx.data.shape
            cand = {}
            for k, cs in self.unit_sets().items():
                u = self.unit(k)
                full = np.zeros((H, W), np.float32)
                full[u.sl] = cs.raster(u.ev.shape)
                cand[k] = full
            return F.export_case_studies(self, systab, cand)
        return self._stage("s02b_case_studies", self.fp("s02b", self.fp_of("s02_failure_analysis"),
                                                        self.fp_of("s03_candidates")), fn,
                           lambda: self.ws.load_table("case_study_index", "failure_analysis/case_studies"))

    # =========================================================================== S03
    def _label_free(self):
        from ..artifacts.atomic import atomic_save
        from ..gems3.candidates import CandidateSet
        from . import generator as G

        c = self.cfg["candidates"]
        key = fingerprint(M7_VERSION, c["sources"], c["scales_m"], c["top_frac_per_scale"], c["strong_frac"],
                          c["min_coherence"], c["min_persistence"], c["link_gap_px"], c["link_angle_deg"],
                          c["min_len_km"], c["max_candidates_per_source"])
        npz, tab, logp = (self.cache / f"label_free_{key}{e}" for e in (".npz", ".parquet", "_log.csv"))
        if npz.exists() and tab.exists() and logp.exists() and not self.cfg.get("overwrite_existing"):
            self._say(f"s03 label-free candidates reused from {npz.name}")
            return CandidateSet.from_npz(pd.read_parquet(tab), np.load(npz)), pd.read_csv(logp), key
        lf, log = G.label_free(self.ctx, self.cfg)
        atomic_save(npz, lambda t: np.savez_compressed(t, **lf.to_npz()))
        atomic_save(tab, lambda t: lf.table.to_parquet(t, index=False))
        atomic_save(logp, lambda t: log.to_csv(t, index=False))
        return lf, log, key

    def _member_scores(self, u, full_maps=None) -> dict:
        from ..gems3.official import unit_score

        out = {}
        for i, m in enumerate(self.models):
            stem = self.cfg["models"][i]
            if u.task == "full":
                out[stem] = full_maps[i] if full_maps is not None else np.asarray(self.ctx.oof(m), np.float32)
            else:
                out[stem] = unit_score(self.ctx, m, u)
        return out

    def s03_candidates(self) -> dict | None:
        from . import generator as G

        cfg = self.cfg

        def fn():
            lf, log, key = self._label_free()
            sub_t, sub_v = "candidate_generation/tables", "candidate_generation/vectors"
            self.ws.save_table(log, "generator_log", sub_t)
            fm = self.cached_full_maps()
            if fm is None:
                self._say("s03 no cached full-region maps: the stitched OOF maps feed the 'full' unit's model crests")
            sets, inv, mlog = {}, [], []
            units = [u for t in ("geo", cfg["evaluation"]["primary_task"]) for u in self.units_of(t)] + [self.full_unit()]
            H, W = self.ctx.data.shape
            conf = np.zeros((H, W), np.float32)
            for u in units:
                cs, lg = G.unit_candidates(u, lf, self._member_scores(u, fm), cfg)
                gt = G.geometry(cs)
                tab = pd.concat([cs.table.reset_index(drop=True), gt], axis=1)
                tab["structural_score"] = G.structural_score(tab)
                cs.table = tab
                k = u.key.replace(":", "_")
                self.ws.save_npz(cs.to_npz(), f"{sub_v}/units/{k}.npz")
                self.ws.save_table(tab, f"units/{k}", sub_t)
                inv.append(tab.assign(task=u.task, zone=u.zone, unit=u.key))
                mlog.append(lg)
                sets[u.key] = cs
                if u.task == "full" and len(cs):
                    conf = np.maximum(conf, cs.raster((H, W), tab.structural_score.to_numpy()))
                    self.ws.save_lines(cs.paths, tab[[c for c in ("uid", "cand_id", "source", "modality", "kind",
                                                                  "sources", "n_sources", "evidence_class", "network_id",
                                                                  "length_km", "strike_deg", "structural_score",
                                                                  "n_agree_modalities") if c in tab]],
                                       f"{sub_v}/full_region_candidates.gpkg", self.ctx.data.grid, "candidates")
                self._say(f"s03 {u.key}: {len(cs)} fused candidates")
            invt = pd.concat(inv, ignore_index=True)
            self.ws.save_table(invt, "candidate_inventory", sub_t)
            self.ws.save_table(pd.concat(mlog, ignore_index=True) if mlog else pd.DataFrame(), "model_source_log", sub_t)
            self._save_unit_vectors(sets)
            self.ws.save_raster({"structural_score_max": np.where(conf > 0, conf, np.nan)},
                                "candidate_generation/rasters/full_region_candidate_confidence.tif", self.ctx.data.grid)
            self.ws.save_json({"label_free_cache_key": key, "n_label_free": len(lf)},
                              "candidate_generation/tables/cache.json")
            self._fig_candidates(invt)
            return sets

        fp = self.fp("s03", cfg["candidates"], self.cfg["submission"]["source"])
        return self._stage("s03_candidates", fp, fn, self.unit_sets, optional=False)

    def _save_unit_vectors(self, sets: dict) -> None:
        """All evaluation-unit candidates in full-grid coordinates, one GeoPackage (unit attribute)."""
        from ..gems3.candidates import to_full

        paths, rows = [], []
        for k, cs in sets.items():
            if k == "full:0":
                continue
            u = self.unit(k)
            for i, p in enumerate(cs.paths):
                paths.append(to_full(p, u))
                r = cs.table.iloc[i]
                rows.append({"unit": k, "uid": r.get("uid"), "source": r.get("source"), "modality": r.get("modality"),
                             "length_km": r.get("length_km"), "structural_score": r.get("structural_score")})
        if paths:
            self.ws.save_lines(paths, pd.DataFrame(rows), "candidate_generation/vectors/unit_candidates.gpkg",
                               self.ctx.data.grid, "unit_candidates")

    def _fig_candidates(self, inv: pd.DataFrame) -> None:
        from .figures import bar_panels16

        if not len(inv):
            return
        g = inv.groupby(["task", "modality"]).agg(n=("cand_id", "size"), km=("length_km", "sum")).reset_index()
        self.ws.save_fig(bar_panels16(g, "modality", "km", "task", xlabel="candidate length (km)",
                                      title="Section 03 - candidate inventory by modality"), "candidate_inventory_km", g,
                         "candidate_generation/tables")

    # =========================================================================== S04
    def s04_oracle(self) -> dict | None:
        from . import coverage as C

        def fn():
            sets = self.unit_sets()
            if not sets:
                raise RuntimeError("no candidate sets: run section 03 first")
            res = C.run(self, sets)
            sub = "candidate_generation/oracle"
            for k in ("summary", "curve", "oracle", "by_group", "systems", "truth_stats", "pooled_curve", "pooled_oracle",
                      "pooled_groups", "incremental", "by_stratum", "miss_diagnostics", "miss_summary"):
                self.ws.save_table(res[k], k, sub)
            self.ws.save_table(res["decision"], "gate_decision", sub)
            self.ws.save_json(json.loads(res["decision"].to_json(orient="records")), f"{sub}/gate_decision.json")
            C.figures(self, res)
            return res

        def load():
            return {"decision": self.ws.load_table("gate_decision", "candidate_generation/oracle"),
                    "pooled_curve": self.ws.load_table("pooled_curve", "candidate_generation/oracle")}
        fp = self.fp("s04", self.cfg["oracle"], self.fp_of("s03_candidates"), self.fp_of("s02_failure_analysis"))
        return self._stage("s04_oracle", fp, fn, load, optional=False)

    def gate(self) -> str | None:
        d = self.ws.load_table("gate_decision", "candidate_generation/oracle")
        prim = self.cfg["evaluation"]["primary_task"]
        d = d[d.task == prim] if len(d) else d
        return str(d.decision.iloc[0]) if len(d) else None

    def _require_gate(self) -> None:
        g = self.gate()
        if g == "GO" or self.cfg["ranking"].get("force"):
            return
        if self.cfg.get("stop_after_oracle_if_inadequate", True):
            raise GateClosed(f"oracle decision for the primary task is {g}: candidate coverage does not justify a "
                             "ranker (read candidate_generation/oracle/miss_diagnostics.csv; ranking.force=True "
                             "overrides)")

    # =========================================================================== S05-S07
    def s05_dem_inventory(self) -> dict | None:
        from . import dem_pilot as DP

        def fn():
            out = DP.run(self)
            if out["status"] == "NO_INVENTORY":
                raise GateClosed(out["reason"])
            self.ws.save_table(out["inventory"], "inventory", "dem/inventory")
            self.ws.save_table(out["tiles"], "tiles_one_per_tile", "dem/inventory")
            self.ws.save_table(out["descriptors"], "tile_descriptors", "dem/pilot")
            self.ws.save_table(out["pilot"], "pilot_tiles", "dem/pilot")
            self.ws.save_json({"status": out["status"], "provenance": out["provenance"], **out["storage"]},
                              "dem/pilot/storage_estimate.json")
            self._fig_pilot(out["descriptors"], out["pilot"])
            return out

        def load():
            return {"pilot": self.ws.load_table("pilot_tiles", "dem/pilot"),
                    "storage": self.read_json("dem/pilot/storage_estimate.json")}
        fp = self.fp("s05", self.cfg["dem"], self.fp_of("s02_failure_analysis"))
        return self._stage("s05_dem_inventory", fp, fn, load)

    def _fig_pilot(self, desc: pd.DataFrame, sel: pd.DataFrame) -> None:
        from .figures import overview_map

        if not len(sel):
            return
        w = sel.window.map(json.loads)
        pts = sel.assign(r=[(a[0] + a[1]) / 2 for a in w], c=[(a[2] + a[3]) / 2 for a in w], size=sel.label_density * 1e4)
        self.ws.save_fig(overview_map(self.ctx.data.grid, self.ctx.data.valid, pts, color_col="reason",
                                      title="DEM pilot tiles by selection reason"), "pilot_tiles_map",
                         pts[["tile_id", "reason", "r", "c", "label_density"]], "dem/pilot")

    def s06_dem_morphology(self) -> dict | None:
        from . import dem_morph as DM
        from . import dem_pilot as DP

        def fn():
            sel = self.ws.load_table("pilot_tiles", "dem/pilot")
            if not len(sel):
                raise GateClosed("no pilot tiles: run section 05")
            if self.cfg["dem"]["download"]:
                log = DP.download(self, sel)
                self.ws.save_table(log, "download_log", "dem/pilot")
            out = DM.run(self, sel)
            self.ws.save_table(out["timing"], "tile_timing", "dem/pilot")
            self.ws.save_json(out["estimate"], "dem/pilot/dem_cost_estimate.json")
            self.ws.save_json({"bands": out["bands"], "mosaic": str(out["mosaic"]) if out["mosaic"] else None},
                              "dem/features/mosaic_bands.json")
            if out["mosaic"]:
                self.ws._track(Path(out["mosaic"]))
            DM.save_vectors(self, out["vectors"])
            self._memo.pop("dem_layer", None)
            if not out["mosaic"]:
                raise GateClosed("no tile processed (raw tiles missing): approve dem.download or stage the tiles")
            return out
        fp = self.fp("s06", self.cfg["dem"]["morphology"], self.fp_of("s05_dem_inventory"))
        return self._stage("s06_dem_morphology", fp, fn,
                           lambda: {"estimate": self.read_json("dem/pilot/dem_cost_estimate.json")})

    def s07_dem_increment(self) -> dict | None:
        from . import dem_value as DV

        def fn():
            mos = self.dem_mosaic()
            if mos is None:
                raise GateClosed("no DEM mosaic: run section 06")
            out = DV.run(self, self.unit_sets(), mos)
            sub = "dem/candidates"
            for k in ("results", "pooled", "systems", "venn", "baseline_restricted", "dem_candidate_log", "decision"):
                self.ws.save_table(out[k], k, sub)
            B = out["dem_candidates_full"]
            if len(B):
                self.ws.save_lines(B.paths, B.table[[c for c in ("uid", "source", "scale_m", "strength_mean",
                                                                 "coherence_mean", "persistence_mean") if c in B.table]],
                                   f"{sub}/dem_candidates.gpkg", self.ctx.data.grid, "dem_candidates")
            self._fig_dem(out)
            return out
        fp = self.fp("s07", self.cfg["dem"], self.fp_of("s06_dem_morphology"), self.fp_of("s03_candidates"))
        return self._stage("s07_dem_increment", fp, fn, lambda: {"decision": self.ws.load_table("decision", "dem/candidates")})

    def _fig_dem(self, out: dict) -> None:
        from .figures import bar_panels16

        p = out["pooled"]
        if len(p):
            self.ws.save_fig(bar_panels16(p, "set", "system_recall", "task", xlabel="system recall (DEM-covered, matched "
                                          "density)", title="Section 07 - geophysics vs DEM vs combined candidates"),
                             "dem_increment_system_recall", p, "dem/candidates")
        v = out["venn"]
        if len(v):
            g = v.groupby(["task", "venn"]).size().rename("n").reset_index()
            self.ws.save_fig(bar_panels16(g, "venn", "n", "task", xlabel="withheld systems",
                                          title="Coverage decomposition (Venn counts)"), "dem_venn", g, "dem/candidates")

    # =========================================================================== S08-S09
    def s08_network_analysis(self) -> dict | None:
        from . import network as N

        def fn():
            out = N.run(self)
            sub = "map_completion/candidates"
            for k, t in out["tables"].items():
                self.ws.save_table(t, f"full_{k}", sub)
            self.ws.save_table(out["summary"], "summary", sub)
            full = out["full"]
            if len(full):
                self.ws.save_lines(full.paths, full.table, f"{sub}/full_region_completion_hypotheses.gpkg",
                                   self.ctx.data.grid, "completion_hypotheses")
            for k, cs in out["per_unit"].items():
                kk = k.replace(":", "_")
                self.ws.save_npz(cs.to_npz(), f"{sub}/units/{kk}.npz")
                self.ws.save_table(cs.table, f"units/{kk}", sub)
            self.ws.save_table(N.unit_completion_audit(self, out["per_unit"]), "completion_coverage_audit", sub)
            base = N.completion_baseline(self, out["per_unit"])
            self.ws.save_table(base["curve"], "completion_baseline_curve", "map_completion/metrics")
            self.ws.save_table(base["summary"], "completion_baseline_summary", "map_completion/metrics")
            self._memo["mc_evidence"] = out["evidence"]
            return out
        fp = self.fp("s08", self.cfg["map_completion"], self.cfg["candidates"]["tip_tail_px"])
        return self._stage("s08_network_analysis", fp, fn,
                           lambda: {"summary": self.ws.load_table("summary", "map_completion/candidates")})

    def completion_sets(self) -> dict:
        from ..gems3.candidates import CandidateSet

        out = {}
        for u in self.units_of(self.cfg["evaluation"]["primary_task"]):
            kk = u.key.replace(":", "_")
            z = self.ws.root / "map_completion" / "candidates" / "units" / f"{kk}.npz"
            t = self.ws.load_table(f"units/{kk}", "map_completion/candidates")
            if z.exists() and len(t):
                out[u.key] = CandidateSet.from_npz(t, np.load(z))
        return out

    def s09_corruption_benchmark(self) -> dict | None:
        from . import corruption as K
        from .network import evidence_layer

        def fn():
            ev = self._memo.get("mc_evidence")
            if ev is None:
                ev = evidence_layer(self.ctx, self.cfg["map_completion"]["evidence_layers"], self.ctx.data.valid,
                                    int(self.cfg["seed"]))
            out = K.evaluate(self, ev)
            sub = "map_completion/metrics"
            self.ws.save_table(out["protocol"], "corruption_protocol", "map_completion/corruptions")
            self.ws.save_table(out["results"], "results", sub)
            self.ws.save_table(out["selection"], "selection", sub)
            self.ws.save_table(out["decision"], "decision", sub)
            self.ws.save_table(out["proposals"], "proposals_labelled", "map_completion/corruptions")
            if "learned" in out:
                self.ws.save_table(out["learned"], "learned_completer", sub)
            self._fig_mc(out)
            return out
        fp = self.fp("s09", self.cfg["map_completion"])
        return self._stage("s09_corruption_benchmark", fp, fn,
                           lambda: {"selection": self.ws.load_table("selection", "map_completion/metrics"),
                                    "decision": self.ws.load_table("decision", "map_completion/metrics")})

    def _fig_mc(self, out: dict) -> None:
        from .figures import bar_panels16

        s = out["selection"]
        if len(s):
            self.ws.save_fig(bar_panels16(s.assign(cat=s.completer), "cat", "recall_crossfit", "class",
                                          xlabel="cross-fitted weighted recall of the removed geometry",
                                          title="Section 09 - deterministic completion on damaged maps"),
                             "completion_recall", s, "map_completion/metrics")

    # =========================================================================== S10-S12
    def s10_candidate_features(self) -> dict | None:
        from ..gems3.candidates import CandidateSet
        from ..gems3.official import unit_score
        from . import cand_features as CF

        def fn():
            self._require_gate()
            sets = self.unit_sets()
            comp = self.completion_sets()
            mos = self.dem_mosaic()
            tabs, ctabs = [], []
            for k, cs in sets.items():
                u = self.unit(k)
                sc = {"b0": unit_score(self.ctx, self.models[0], u), "b1": unit_score(self.ctx, self.models[1], u)}
                df = CF.describe(self, u, cs, sc, with_labels=True, dem=mos)
                self.ws.save_table(df, k.replace(":", "_"), "candidate_ranking/features")
                tabs.append(df)
                if k in comp:
                    c = comp[k]
                    t = pd.concat([cs.table, c.table.assign(cand_id=np.arange(len(cs), len(cs) + len(c)))],
                                  ignore_index=True)
                    both = CandidateSet(t, cs.paths + c.paths)
                    d2 = CF.describe(self, u, both, sc, with_labels=True, dem=mos)
                    self.ws.save_table(d2, f"{k.replace(':', '_')}__with_completion", "candidate_ranking/features")
                    ctabs.append(d2)
                self._say(f"s10 {k}: {len(df)} rows")
            fu = self.full_unit()
            fcs = self.load_unit_set("full:0")
            if fcs is not None and len(fcs):
                fm = self.cached_full_maps() or [np.asarray(self.ctx.oof(m), np.float32) for m in self.models]
                ff = CF.describe(self, fu, fcs, {"b0": fm[0], "b1": fm[1]}, with_labels=False, dem=mos)
                self.ws.save_table(ff, "full_0", "candidate_ranking/features")
            T = pd.concat(tabs, ignore_index=True)
            self.ws.save_table(T, "ranking_table", "candidate_ranking/features")
            g = CF.groups(T)
            self.ws.save_table(pd.DataFrame([{"group": k, "feature": f} for k, v in g.items() for f in v]),
                               "feature_groups", "candidate_ranking/features")
            self.ws.save_table(T.groupby(["task", "label"]).size().rename("n").reset_index(), "label_balance",
                               "candidate_ranking/features")
            return {"table": T, "with_completion": pd.concat(ctabs, ignore_index=True) if ctabs else pd.DataFrame()}
        fp = self.fp("s10", self.cfg["features"], self.cfg["ranking"]["positive_kd"], self.cfg["ranking"]["negative_dist_km"],
                     self.fp_of("s03_candidates"), self.fp_of("s08_network_analysis"), self.fp_of("s06_dem_morphology"))
        return self._stage("s10_candidate_features", fp, fn,
                           lambda: {"table": self.ws.load_table("ranking_table", "candidate_ranking/features")})

    def _feature_tables(self) -> tuple[dict, dict]:
        from ..gems3.candidates import CandidateSet

        sets = self.unit_sets()
        comp = self.completion_sets()
        tabs, csets = {}, {}
        for k, cs in sets.items():
            t = self.ws.load_table(k.replace(":", "_"), "candidate_ranking/features")
            if len(t):
                tabs[k] = t.sort_values("cand_id").reset_index(drop=True)
            t2 = self.ws.load_table(f"{k.replace(':', '_')}__with_completion", "candidate_ranking/features")
            if len(t2) and k in comp:
                c = comp[k]
                csets[k] = (CandidateSet(t2.sort_values("cand_id").reset_index(drop=True), cs.paths + c.paths),
                            t2.sort_values("cand_id").reset_index(drop=True))
        return tabs, csets

    def s11_candidate_ranking(self) -> dict | None:
        from . import ranking as RK

        def fn():
            self._require_gate()
            tabs, csets = self._feature_tables()
            if not tabs:
                raise GateClosed("no candidate feature tables: run section 10")
            out = RK.run(self, tabs, self.unit_sets(), csets)
            sub = "candidate_ranking"
            self.ws.save_table(out["table"], "ranking_scores", f"{sub}/features")
            self.ws.save_table(out["completion_table"], "ranking_scores_with_completion", f"{sub}/features")
            self.ws.save_table(out["curves"], "ranking_curves", f"{sub}/models")
            self.ws.save_table(out["summary"], "ranking_summary", f"{sub}/models")
            self.ws.save_table(out["per_zone"], "ranking_per_zone", f"{sub}/models")
            self.ws.save_table(out["audit"], "training_audit", f"{sub}/models")
            self.ws.save_table(out["importance"], "feature_importance", f"{sub}/models")
            from .figures import bar_panels16

            if len(out["summary"]):
                self.ws.save_fig(bar_panels16(out["summary"], "method", "dti_crossfit", "task", ref_col="fold_min",
                                              title="Section 11 - candidate rankers (1-px lines, eligible budget)"),
                                 "ranking_dti", out["summary"], f"{sub}/models")
            return out
        fp = self.fp("s11", self.cfg["ranking"], self.cfg["rasterization"]["area_fracs"], self.fp_of("s10_candidate_features"))
        return self._stage("s11_candidate_ranking", fp, fn,
                           lambda: {"summary": self.ws.load_table("ranking_summary", "candidate_ranking/models")})

    def s12_rasterization(self) -> dict | None:
        from . import rasterize as RZ

        def fn():
            self._require_gate()
            T = self.ws.load_table("ranking_scores", "candidate_ranking/features")
            S = self.ws.load_table("ranking_summary", "candidate_ranking/models")
            if not len(T) or not len(S):
                raise GateClosed("no ranking outputs: run section 11")
            out = RZ.experiment(self, {"table": T, "summary": S}, self.unit_sets())
            sub = "candidate_ranking/predictions"
            self.ws.save_table(out["grid"], "rasterization_grid", sub)
            self.ws.save_table(out["summary"], "rasterization_summary", sub)
            self.ws.save_table(out["per_zone"], "rasterization_per_zone", sub)
            self.ws.save_table(out["area"], "area_after_masking", sub)
            self.ws.save_table(out["rep_ablation"], "representation_ablation", sub)
            self.ws.save_json({"best_ranker": out["best_ranker"]}, f"{sub}/best_ranker.json")
            from .figures import bar_panels16

            ra = out["rep_ablation"]
            if len(ra):
                self.ws.save_fig(bar_panels16(ra.assign(cat=ra.method + " | " + ra.rep), "cat", "dti_crossfit", "task",
                                              title="Section 12 - representation and ring ablation"),
                                 "representation_ablation", ra, sub)
            return out
        fp = self.fp("s12", self.cfg["rasterization"], self.fp_of("s11_candidate_ranking"))
        return self._stage("s12_rasterization", fp, fn,
                           lambda: {"summary": self.ws.load_table("rasterization_summary", "candidate_ranking/predictions")})

    # =========================================================================== S13-S16
    def s13_comparison(self) -> dict | None:
        from . import comparison as CP

        def fn():
            out = CP.evaluate(self)
            sub = "validation/metrics"
            for k, name in (("per_zone", "method_per_zone"), ("retained_fp", "retained_fp_per_zone"),
                            ("pooled", "method_pooled"), ("paired", "paired_zone_deltas"), ("bootstrap", "bootstrap"),
                            ("systems", "system_hits"), ("proximity", "known_proximity"), ("regions", "regional_dti")):
                self.ws.save_table(out[k], name, sub)
            self.ws.save_json({"champion_reference": out["champion_reference"]}, f"{sub}/champion.json")
            CP.figures(self, out)
            return out

        def load():
            sub = "validation/metrics"
            return {"pooled": self.ws.load_table("method_pooled", sub), "paired": self.ws.load_table("paired_zone_deltas", sub),
                    "bootstrap": self.ws.load_table("bootstrap", sub), "systems": self.ws.load_table("system_hits", sub),
                    "champion_reference": (self.read_json(f"{sub}/champion.json") or {}).get("champion_reference")}
        deps = [self.fp_of(s) for s in ("s01_baseline", "s08_network_analysis", "s11_candidate_ranking",
                                        "s12_rasterization")]
        return self._stage("s13_comparison", self.fp("s13", self.cfg["comparison"], deps), fn, load)

    def s14_decision_report(self) -> dict | None:
        from . import decision as DC

        def fn():
            cmp = self.s13_comparison() or {}
            if not cmp:
                raise RuntimeError("section 13 produced no comparison")
            out = DC.run(self, cmp)
            self.ws.save_table(out["verdicts"], "decision_verdicts", "reports")
            self.ws.save_table(out["answers"], "research_answers", "reports")
            self.ws.save_json(out["decision"], "reports/decision.json")
            self.ws.save_text(out["report"], "reports/M7_DISCOVERY_FIRST_REPORT.md")
            return out
        fp = self.fp("s14", self.cfg["decision"], self.fp_of("s13_comparison"), self.fp_of("s04_oracle"),
                     self.fp_of("s07_dem_increment"), self.fp_of("s09_corruption_benchmark"))
        return self._stage("s14_decision_report", fp, fn,
                           lambda: {"verdicts": self.ws.load_table("decision_verdicts", "reports"),
                                    "answers": self.ws.load_table("research_answers", "reports")})

    def s15_submission_export(self) -> pd.DataFrame | None:
        from . import export as E

        def fn():
            rows = []
            sel = self.baseline_selection() or {}
            prim = self.cfg["evaluation"]["primary_task"]
            for m in ("fusion_thin", "fusion_binary"):
                if m in sel.get(prim, {}):
                    info = E.export_baseline(self, f"m7_s01_{m}", m)
                    rows.append({"candidate": f"s01_{m}", "file": info["submission"],
                                 "frac_of_eligible": info["final_frac_of_eligible"],
                                 "checks_failed": ";".join(info["validation"]["checks_failed"])})
            pm = E.promoted_method(self)
            if pm and (pm.startswith("s12_") or pm.startswith("rank_")):
                name = pm[4:] if pm.startswith("s12_") else f"cand_{pm[5:]}"
                try:
                    info = E.export_candidate_model(self, f"m7_{name}", name)
                    rows.append({"candidate": pm, "file": info["submission"],
                                 "frac_of_eligible": info["final_frac_of_eligible"],
                                 "checks_failed": ";".join(info["validation"]["checks_failed"])})
                except (KeyError, FileNotFoundError) as e:
                    self._say(f"s15 {pm}: not exported ({e})")
            t = pd.DataFrame(rows)
            self.ws.save_table(t, "submission_index", "submissions")
            return t
        fp = self.fp("s15", self.cfg["submission"], self.fp_of("s01_baseline"), self.fp_of("s14_decision_report"))
        return self._stage("s15_submission_export", fp, fn, lambda: self.ws.load_table("submission_index", "submissions"))

    def s16_probe_packages(self) -> pd.DataFrame | None:
        from . import export as E

        def fn():
            t = E.probes(self)
            self.ws.save_table(t, "probe_packages", "submissions")
            return t
        fp = self.fp("s16", self.cfg["probes"], self.cfg["submission"], self.fp_of("s14_decision_report"))
        return self._stage("s16_probe_packages", fp, fn, lambda: self.ws.load_table("probe_packages", "submissions"))

    # =========================================================================== run + summary
    def run(self) -> dict:
        if not self.cfg.get("enabled", True):
            self._say("disabled (M7_CONFIG['enabled'] = False)")
            return {}
        t0 = time.perf_counter()
        out = {"s00": self.s00_setup()}
        order = [("s01_baseline", self.s01_baseline), ("s02_failure_analysis", self.s02_failure_analysis),
                 ("s03_candidates", self.s03_candidates), ("s04_oracle", self.s04_oracle),
                 ("s05_dem_inventory", self.s05_dem_inventory), ("s06_dem_morphology", self.s06_dem_morphology),
                 ("s07_dem_increment", self.s07_dem_increment), ("s08_network_analysis", self.s08_network_analysis),
                 ("s09_corruption_benchmark", self.s09_corruption_benchmark),
                 ("s10_candidate_features", self.s10_candidate_features),
                 ("s11_candidate_ranking", self.s11_candidate_ranking), ("s12_rasterization", self.s12_rasterization),
                 ("s13_comparison", self.s13_comparison), ("s14_decision_report", self.s14_decision_report),
                 ("s15_submission_export", self.s15_submission_export), ("s16_probe_packages", self.s16_probe_packages)]
        for name, fn in order:
            if self.enabled(name):
                out[name] = fn()
                if name == "s03_candidates" and self.enabled("s02_failure_analysis"):
                    self.s02b_case_studies_with_candidates()
        self._say(f"finished in {time.perf_counter() - t0:.0f}s -> {self.ws.root}")
        self.paths()
        return out

    def summary(self) -> pd.DataFrame:
        """Compact cross-section view: one row per method and task from section 13 (else section 01)."""
        p = self.ws.load_table("method_pooled", "validation/metrics")
        if len(p):
            v = self.ws.load_table("decision_verdicts", "reports")
            if len(v):
                p = p.merge(v[["method", "verdict"]], on="method", how="left")
            return p
        return self.ws.load_immutable_table("baseline_table", "recovered_baselines")

    def paths(self) -> dict:
        """Exact local and Drive locations of every section's outputs (printed for the next session)."""
        drive = self.cfg.get("_drive_root")
        rows = {}
        for s in sorted((self.ws.root / "manifests" / "sections").glob("*.json")):
            rows[s.stem] = {"local": str(self.ws.root), "drive": drive,
                            "manifest": str(s.relative_to(self.ws.root))}
        print(f"[m7] outputs (local): {self.ws.root}")
        if drive:
            print(f"[m7] outputs (Drive): {drive}")
        for k, v in rows.items():
            print(f"  {k:<28} {v['manifest']}")
        return rows
