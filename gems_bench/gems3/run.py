"""GEMS3 orchestration: ``pipe.gems3(GEMS3_CONFIG)`` -> stages with fingerprints, reuse and failure isolation.

    g3 = pipe.gems3(GEMS3_CONFIG)
    g3.run()                       # every enabled stage, in priority order
    g3.verify_official_metric(); g3.metric_optimization(); g3.submission(); g3.candidates(); g3.oracle()
    g3.ranker(); g3.dem_gate(); g3.map_completion(); g3.report(); g3.summary()

Maps are loaded through M6 (read-only inventory, staging of prediction rasters only, memory-mapped cache shared
with M6 in ``cache/m6``); the loader writes its inventory to ``reports/gems3/inventory`` so M6's own reports are
never touched. A stage whose fingerprint (configuration section + input-map fingerprints + GEMS3 version) and
outputs match is reused; ``overwrite_existing=True`` recomputes. Every stage after the metric audit is isolated:
its failure is logged (``manifests/stage_log.json``) and the next stage still runs.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..pipeline.m6 import M6Analysis, fingerprint
from .config import load_gems3_config
from .workspace import GEMS3_VERSION, G3Workspace


class GEMS3Research:
    def __init__(self, pc=None, config: dict | None = None, *, ctx=None, local_root: Path | None = None,
                 data_root: Path | None = None, config_path: Path | None = None):
        self.cfg = load_gems3_config(config, config_path)
        self.pc = pc
        self.local = Path(local_root or (pc.local if pc is not None else Path.cwd()))
        self.ws = G3Workspace(self.local / self.cfg["output_dir"], self.cfg)
        self.cache = self.local / self.cfg["cache_dir"]
        self.cache.mkdir(parents=True, exist_ok=True)
        stems = list(self.cfg["models"])
        self.m6 = M6Analysis(pc, {"models": stems, "primary_pair": stems, "controls": stems, "optional_models": [],
                                  "baselines": [], "compare_baselines": False, "output_dir": f"{self.cfg['output_dir']}/inventory",
                                  "ensemble": {"extended_sets": []}, "save_parquet": False},
                             ctx=ctx, local_root=local_root, data_root=data_root)
        self._loaded = False
        self._units = None
        self._sets = None
        self.log: list[dict] = []

    # -- helpers ---------------------------------------------------------------
    @property
    def ctx(self):
        return self.m6.ctx

    @property
    def models(self) -> list[str]:
        return [self.m6.m(s) for s in self.cfg["models"]]

    def stem_to_model(self, stem: str) -> str:
        return self.m6.m(stem)

    def _say(self, msg: str) -> None:
        print(f"[gems3] {msg}")

    def _push(self) -> None:
        if self.pc is not None:
            try:
                self.pc.push([Path(self.ws.root).resolve().relative_to(self.local.resolve()).as_posix()])
            except (ValueError, Exception) as e:                      # never lose results to a sync error
                self._say(f"push skipped: {type(e).__name__}: {e}")

    def load(self) -> None:
        """M6 read-only inventory + memory-mapped OOF / discovery maps of both members (fails clearly if absent)."""
        if not self._loaded:
            self.m6.audit()
            self._loaded = True

    @property
    def units(self):
        if self._units is None:
            from .official import build_units

            self.load()
            self._units = build_units(self.ctx, self.cfg["tasks"])
        return self._units

    def _fp(self, *parts) -> str:
        self.load()
        return fingerprint(GEMS3_VERSION, self.m6.map_fp, self.cfg["models"], self.cfg["tasks"], *parts)

    def _stage(self, name: str, fp: str, fn, load=None, optional: bool = True):
        t0 = time.perf_counter()
        if self.ws.valid(name, fp):
            self._say(f"{name}: reused (fingerprint {fp})")
            self.log.append({"stage": name, "outcome": "REUSED", "seconds": 0.0})
            return load() if load else None
        self.ws.begin(name)
        try:
            out = fn()
        except Exception as e:
            if not optional:
                raise
            self._say(f"{name} FAILED: {type(e).__name__}: {e}")
            self.log.append({"stage": name, "outcome": "FAILED", "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-3000:],
                             "seconds": round(time.perf_counter() - t0, 1)})
            self._save_log()
            return None
        dt = round(time.perf_counter() - t0, 1)
        self.ws.finish(name, fp, seconds=dt)
        self.log.append({"stage": name, "outcome": "COMPUTED", "seconds": dt})
        self._say(f"{name}: computed in {dt:.0f}s")
        self._save_log()
        self._push()
        return out

    def _save_log(self) -> None:
        from ..artifacts.atomic import atomic_write_json

        atomic_write_json(self.ws.root / "manifests" / "stage_log.json", self.log)

    def selection(self) -> dict | None:
        p = self.ws.root / "manifests" / "selected_representation.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    # =========================================================================== 1. official metric audit
    def verify_official_metric(self) -> dict:
        from .official import contract_table, masking_audit, official_vs_internal

        def fn():
            out = {"contract": contract_table(), "masking": masking_audit(),
                   "comparison": official_vs_internal(self.ctx, self.models)}
            self.ws.save_table(out["contract"], "official_contract", "audit")
            self.ws.save_table(out["masking"], "masking_audit", "audit")
            self.ws.save_table(out["comparison"], "official_metric_comparison", "audit")
            inv = self.m6.inv
            self.ws.save_table(inv if inv is not None else pd.DataFrame(), "artifact_inventory", "audit")
            self._fig_conventions(out["comparison"])
            return out

        def load():
            return {k: self.ws.load_table(n, "audit") for k, n in
                    (("contract", "official_contract"), ("masking", "masking_audit"),
                     ("comparison", "official_metric_comparison"))}
        return self._stage("official_metric", self._fp("audit"), fn, load, optional=False)

    def _fig_conventions(self, comp: pd.DataFrame) -> None:
        from .figures import line_panels

        if not len(comp):
            return
        d = comp[comp.unit == "pooled_splits"].copy()
        d["series"] = d.task
        self.ws.save_fig(line_panels(d, "target_frac", "dti", "series", "model", xlabel="predicted area (share of zone)",
                                     logx=True, title="discovery conventions: "), "official_vs_internal_conventions",
                         d, sub="audit")

    # =========================================================================== 2-4. metric optimisation + fusion
    def metric_optimization(self) -> dict | None:
        from . import metric_opt as MO

        cfg = self.cfg

        def fn():
            out = MO.run(self.ctx, self.units, self.models, cfg)
            curves = MO.pooled_curves(out)
            sub = "metric_optimization"
            for name, df in curves.items():
                self.ws.save_table(df, name, sub)
            self.ws.save_table(out["raw"], "raw_scores", sub)
            self.ws.save_table(out["missing"], "missing_maps", sub)
            self.ws.save_table(out["sel1"].get("per_w", pd.DataFrame()) if out["sel1"] else pd.DataFrame(),
                               "ensemble_ablation", sub)
            self.ws.save_table(out["selection"], "selected_parameters", sub)
            self.ws.save_table(out["per_zone"], "selected_parameters_per_zone", sub)
            self.ws.save_table(out["rep_ablation"], "representation_ablation", sub)
            self.ws.save_table(out["candidates"], "stage2_candidates", sub)
            self.ws.save_table(out["paired"], "paired_fold_deltas", "ensembles")
            if len(out["area"]):
                a = curves["area_budget_curves"]
                self.ws.save_table(a[a.w.isin([0.0, 1.0])], "b0_b1_comparison", "ensembles")
            self.ws.save_table(out["sel1"].get("thresholds", pd.DataFrame()) if out["sel1"] else pd.DataFrame(),
                               "calibration_summary", "ensembles")
            MO.figures(self.ws, out, curves, self.models)
            sel = {"choice": out["choice"], "geo_frac": self._geo_frac(out), "models": self.models,
                   "primary_task": cfg["primary_task"], "gems3_version": GEMS3_VERSION}
            self.ws.save_json(json.loads(json.dumps(sel, default=float)), "manifests/selected_representation.json")
            for task, ch in out["choice"].items():
                layers = MO.heldout_maps(self.ctx, self.units, self.models, cfg, task, ch["per_zone"])
                if layers:
                    self.ws.save_raster(layers, f"predictions/metric_opt_{task}_heldout.tif", self.ctx.data.grid)
            from .report import metric_report

            self.ws.save_text(metric_report(self), "metric_optimization/metric_optimization_report.md")
            return sel

        return self._stage("metric_optimization", self._fp(cfg["metric_opt"], cfg["primary_task"]), fn, self.selection)

    @staticmethod
    def _geo_frac(out: dict) -> dict:
        """All-geo-zone best area fraction per (representation, weight): the submission budget rule 'geo'."""
        res = {}
        a = out["area"]
        if len(a):
            g = a[a.task == "geo"].groupby(["w", "frac"])[["tp_w", "fp_w", "fn_w"]].sum().reset_index()
            g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
            for w, gw in g.groupby("w"):
                res[f"binary|{w:g}"] = float(gw.loc[gw.dti.idxmax(), "frac"])
        t2 = out["stage2"]
        if len(t2):
            g = t2[t2.task == "geo"].groupby(["rep", "w", "frac"])[["tp_w", "fp_w", "fn_w"]].sum().reset_index()
            g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
            for (rep, w), gw in g.groupby(["rep", "w"]):
                res[f"{rep}|{w:g}"] = float(gw.loc[gw.dti.idxmax(), "frac"])
        return res

    # =========================================================================== 3. submission candidate
    def members(self) -> list[dict]:
        from .submission import member_record

        self.load()
        inv = self.m6.inv.set_index("model")
        return [member_record({**inv.loc[m].to_dict(), "model": m}) for m in self.models]

    def full_region_maps(self) -> tuple[list[np.ndarray], list[dict]]:
        from .submission import member_full_map

        sub = self.cfg["submission"]
        maps, metas = [], []
        for mem in self.members():
            if sub["source"] == "fold_average" and self.pc is not None and self.pc.drive is not None:
                rels = [self.m6._rel(Path(mem["run_dir"]) / "checkpoints" / c["file"]) for c in mem["checkpoints"]]
                self.pc.stage_in([r for r in rels if r])
                mem = {**mem, "run_dir": str(self.m6._local(Path(mem["run_dir"])))}
            dev = None if sub["device"] == "auto" else sub["device"]
            p, meta = member_full_map(self.ctx, self.ws, mem, sub["source"], dev)
            maps.append(p)
            metas.append({**mem, **meta})
        return maps, metas

    def submission(self) -> dict | None:
        from . import submission as S

        cfg, sub = self.cfg, self.cfg["submission"]

        def fn():
            sel = self.selection()
            if sel is None:
                raise RuntimeError("no selected representation: run metric_optimization first")
            params = S.resolve_parameters(sel, sub, cfg["primary_task"])
            reps = cfg["metric_opt"]["representations"]
            rep = reps.get(params["rep"], {"select": "top"})
            maps, metas = self.full_region_maps()
            from .._version import provenance

            out = []
            for fill in sorted({False, bool(sub["known_fill"])}):
                score, q = S.transform(maps, self.ctx.data, params, rep, known_fill=fill,
                                       sigma_px=cfg["metric_opt"].get("ridge_sigma_px", 1.0))
                name = S.candidate_name(params, fill)
                man = {"members": metas, "parameters": params, "representation": rep, "known_fill": fill,
                       "selection": {"task": cfg["primary_task"],
                                     "dti_crossfit": sel["choice"][cfg["primary_task"]]["dti_crossfit"],
                                     "dti_insample": sel["choice"][cfg["primary_task"]]["dti_insample"]},
                       "code": provenance(), "protocol_hash": self.m6.protocol_hash,
                       "submission_source": sub["source"], "uploaded": False}
                info = S.export(self.ws, name, score, q, self.ctx.data, man, sub["outside_valid"])
                out.append({"candidate": name, "file": info["submission"], "raw_score": info["raw_score"],
                            "known_fill": fill, **{k: v for k, v in params.items()},
                            "positive_frac_valid": info["validation"]["positive_frac_valid"],
                            "checks_failed": ";".join(info["validation"]["checks_failed"])})
            tab = pd.DataFrame(out)
            self.ws.save_table(tab, "submission_candidates", "submissions")
            return tab

        fp = self._fp(sub, cfg["metric_opt"]["representations"], (self.selection() or {}).get("choice"))
        return self._stage("submission", fp, fn, lambda: self.ws.load_table("submission_candidates", "submissions"))

    # =========================================================================== 5. candidates
    def _candidate_units(self):
        from .official import Unit

        us = [u for u in self.units if u.task in ("geo", self.cfg["primary_task"])]
        H, W = self.ctx.data.shape
        full = Unit("full", 0, (slice(0, H), slice(0, W)), self.ctx.data.valid, self.ctx.data.labels, None, True,
                    self.ctx.data.labels)
        return us, full

    def _model_scores(self, u, full_maps=None) -> dict:
        from .official import unit_score

        out = {}
        for src in self.cfg["candidates"].get("model_sources") or []:
            stem = src["model"]
            if u.task == "full":
                if full_maps is not None and stem in self.cfg["models"]:
                    out[stem] = full_maps[self.cfg["models"].index(stem)]
            else:
                out[stem] = unit_score(self.ctx, self.stem_to_model(stem), u)
        return out

    def _full_maps_cached(self):
        """Full-region member maps if the submission stage produced them, else the stitched OOF maps."""
        from ..ensemble import read_map

        maps = []
        for m in self.models:
            p = self.ws.root / "predictions" / f"{m}__full_region_raw.tif"
            maps.append(read_map(p) if p.exists() else np.asarray(self.ctx.oof(m), np.float32))
        return maps

    def candidates(self) -> dict | None:
        from . import candidates as C

        cfg = self.cfg

        def fn():
            lf, log = self._label_free()
            us, full = self._candidate_units()
            full_maps = self._full_maps_cached()
            sets, inv = {}, []
            for u in us + [full]:
                cs = C.unit_candidates(u, lf, self._model_scores(u, full_maps), cfg)
                sets[u.key] = cs
                self.ws.save_npz(cs.to_npz(), f"proposals/units/{u.key.replace(':', '_')}.npz")
                t = pd.concat([cs.table.reset_index(drop=True), C.geometry_table(cs)], axis=1)
                inv.append(t.assign(task=u.task, zone=u.zone, unit=u.key))
                self.ws.save_table(t, f"units/{u.key.replace(':', '_')}", "proposals")
                self._say(f"candidates {u.key}: {len(cs)}")
            inv = pd.concat(inv, ignore_index=True)
            self.ws.save_table(inv, "candidate_inventory", "proposals")
            self.ws.save_table(log, "label_free_sources", "proposals")
            fc = sets[full.key]
            if len(fc):
                self.ws.save_lines(fc.paths, fc.table, "proposals/candidate_geometries.gpkg", self.ctx.data.grid,
                                   "full_region_candidates")
            self._sets = sets
            return sets

        full_fps = [json.loads(p.read_text(encoding="utf-8")).get("fingerprint")
                    for p in sorted((self.ws.root / "predictions").glob("*__full_region_raw.json"))]
        return self._stage("candidates", self._fp(cfg["candidates"], full_fps), fn, self.load_candidates)

    def _label_free(self):
        """Label-free candidates on the full grid, cached in cache_dir by their configuration."""
        from . import candidates as C

        key = fingerprint(GEMS3_VERSION, self.cfg["candidates"]["sources"], self.cfg["candidates"]["min_len_km"])
        npz, tab = self.cache / f"label_free_{key}.npz", self.cache / f"label_free_{key}.parquet"
        logp = self.cache / f"label_free_{key}_log.csv"
        if npz.exists() and tab.exists() and logp.exists() and not self.cfg.get("overwrite_existing"):
            return C.CandidateSet.from_npz(pd.read_parquet(tab), np.load(npz)), pd.read_csv(logp)
        lf, log = C.label_free(self.ctx, self.cfg)
        from ..artifacts.atomic import atomic_save

        atomic_save(npz, lambda t: np.savez_compressed(t, **lf.to_npz()))
        atomic_save(tab, lambda t: lf.table.to_parquet(t, index=False))
        atomic_save(logp, lambda t: log.to_csv(t, index=False))
        return lf, log

    def load_candidates(self) -> dict:
        from .candidates import CandidateSet

        if self._sets is not None:
            return self._sets
        us, full = self._candidate_units()
        sets = {}
        for u in us + [full]:
            k = u.key.replace(":", "_")
            z, t = self.ws.root / "proposals" / "units" / f"{k}.npz", self.ws.load_table(f"units/{k}", "proposals")
            if z.exists():
                tt = t[[c for c in t.columns if c in ("source", "kind", "scale_m", "cand_id", "sources", "n_sources",
                                                       "n_agree") or c.startswith("agree_")]]
                sets[u.key] = CandidateSet.from_npz(tt, np.load(z))
        self._sets = sets
        return sets

    # =========================================================================== 6. oracle
    def oracle(self) -> dict | None:
        from . import oracle as O

        cfg = self.cfg

        def fn():
            sets = self.load_candidates()
            us, _ = self._candidate_units()
            parts = {"summary": [], "curve": [], "by_source": [], "stats": []}
            for u in us:
                cs = sets.get(u.key)
                if cs is None:
                    continue
                r = O.unit_oracle(self.ctx, u, cs, cfg)
                for k in parts:
                    parts[k].append(r[k])
                self._say(f"oracle {u.key}: done")
            res = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in parts.items()}
            pooled = O.pooled_oracle(res["curve"]) if len(res["curve"]) else pd.DataFrame()
            sel = self.selection() or {"choice": {}}
            best = {t: c.get("dti_crossfit", np.nan) for t, c in sel["choice"].items()}
            dec = O.decision(pooled, res["summary"], best, cfg) if len(pooled) else pd.DataFrame()
            self.ws.save_table(pd.concat([res["summary"], pooled.assign(subset="pooled_oracle")], ignore_index=True),
                               "oracle_recall", "proposals")
            self.ws.save_table(res["curve"], "oracle_curve", "proposals")
            self.ws.save_table(res["by_source"], "oracle_by_source", "proposals")
            self.ws.save_table(res["stats"], "candidate_truth_stats", "proposals")
            self.ws.save_table(dec, "oracle_decision", "proposals")
            self._fig_oracle(pooled, res["by_source"])
            return {"decision": dec, "pooled": pooled}

        def load():
            return {"decision": self.ws.load_table("oracle_decision", "proposals")}
        sel = self.selection() or {}
        return self._stage("oracle", self._fp(cfg["candidates"], cfg["oracle"], sel.get("choice")), fn, load)

    def _fig_oracle(self, pooled: pd.DataFrame, by_src: pd.DataFrame) -> None:
        from .figures import bar_panels, line_panels

        if len(pooled):
            d = pooled.assign(series="oracle (reads truth)")
            self.ws.save_fig(line_panels(d, "kd_threshold", "dti", "series", "task",
                                         xlabel="kept if mean k(dist to truth) >= t"), "oracle_dti_curve", d, "proposals")
        if len(by_src):
            g = by_src[by_src.subset == "all"].groupby(["task", "source"])[["tp_w", "n_truth"]].sum().reset_index()
            g["recall_w"] = g.tp_w / g.n_truth.clip(lower=1)
            self.ws.save_fig(bar_panels(g, "source", "recall_w", "task", ylabel="weighted recall (all candidates)"),
                             "oracle_recall_by_source", g, "proposals")

    # =========================================================================== 7. ranker
    def ranker(self) -> dict | None:
        from . import descriptors as D
        from . import ranker as RK

        cfg = self.cfg

        def fn():
            dec = self.ws.load_table("oracle_decision", "proposals")
            prim = dec[dec.task == cfg["primary_task"]] if len(dec) else dec
            if not cfg["ranker"].get("force") and (not len(prim) or prim.decision.iloc[0] != "CONTINUE"):
                raise RuntimeError("oracle decision for the primary task is not CONTINUE: the ranker is not justified "
                                   "(set ranker.force=True to run it anyway)")
            sets = self.load_candidates()
            stats = self.ws.load_table("candidate_truth_stats", "proposals")
            us, full = self._candidate_units()
            layers_full = self._layers()
            elev = self._elevation()
            rows = []
            for u in us:
                cs = sets.get(u.key)
                if cs is None or not len(cs):
                    continue
                st = stats[(stats.task == u.task) & (stats.zone == u.zone)].sort_values("cand_id")
                df = D.describe(u, cs, scores=self._model_scores(u), layers={k: u.crop(v) for k, v in layers_full.items()},
                                elevation=None if elev is None else u.crop(elev), truth_stats=st, cfg=cfg)
                rows.append(df)
            table = pd.concat(rows, ignore_index=True)
            ev = RK.evaluate(table, sets, [u for u in us if u.key in sets], self.ctx.data.shape, cfg)
            self.ws.save_table(ev["table"], "candidate_descriptors", "proposals")
            self.ws.save_table(ev["curves"], "ranker_curves", "ranker")
            self.ws.save_table(ev["summary"], "ranker_summary", "ranker")
            self.ws.save_table(ev["audit"], "training_audit", "ranker")
            self.ws.save_table(self._ranker_comparison(ev["summary"]), "ranker_vs_pixel_models", "ranker")
            # full-region ranking (primary-task model; candidates use all mapped faults as known)
            ptab = table[table.task == cfg["primary_task"]]
            model, feats, imp = RK.fit_full(ptab, cfg)
            self.ws.save_table(imp, "feature_importance", "ranker")
            fc = sets.get(full.key)
            if fc is not None and len(fc):
                fdf = D.describe(full, fc, scores=self._model_scores(full, self._full_maps_cached()),
                                 layers=layers_full, elevation=elev, truth_stats=None, cfg=cfg)
                for c in feats:
                    if c not in fdf:
                        fdf[c] = np.nan
                fdf["score_lgbm"] = model.predict_proba(fdf[feats])[:, 1]
                self.ws.save_table(fdf, "full_region_candidates_ranked", "ranker")
                self.ws.save_lines(fc.paths, fdf[["cand_id", "source", "sources", "length_km", "score_lgbm"]],
                                   "ranker/full_region_ranked.gpkg", self.ctx.data.grid, "ranked_candidates")
                r = RK.score_raster(fc, fdf.score_lgbm.to_numpy(), self.ctx.data.shape)
                self.ws.save_raster({"candidate_score": np.where(np.isfinite(r), r, 0.0)},
                                    "predictions/ranker_full_region.tif", self.ctx.data.grid)
            return ev["summary"]

        sel = self.selection() or {}
        return self._stage("ranker", self._fp(cfg["candidates"], cfg["oracle"], cfg["ranker"], sel.get("choice")), fn,
                           lambda: self.ws.load_table("ranker_summary", "ranker"))

    def _layers(self) -> dict:
        from .candidates import feature_layer

        out = {}
        for src in self.cfg["candidates"]["sources"]:
            a = feature_layer(self.ctx, src)
            if a is not None:
                out[src["name"]] = a
        return out

    def _elevation(self):
        for n in ("raw:det_elev", "raw:elevation"):
            if self.ctx.store.has(n):
                return np.asarray(self.ctx.store.get(n), np.float32)
        return None

    def _ranker_comparison(self, summ: pd.DataFrame) -> pd.DataFrame:
        sel = self.selection() or {"choice": {}}
        rows = list(summ.to_dict("records"))
        for task, c in sel["choice"].items():
            rows.append({"task": task, "method": "best_representation (metric stage)", "dti_crossfit": c["dti_crossfit"]})
        ab = self.ws.load_table("ensemble_ablation", "metric_optimization")
        for r in ab.to_dict("records"):
            if r.get("w") in (0.0, 1.0):
                rows.append({"task": r["task"], "method": f"{self.models[0] if r['w'] == 1.0 else self.models[1]}_binary",
                             "dti_crossfit": r["dti_crossfit"]})
        return pd.DataFrame(rows)

    # =========================================================================== 8-9. DEM gate, map completion
    def dem_gate(self) -> dict | None:
        from . import dem_gate as DG

        def fn():
            out = DG.run(self.ctx, self.models, self.cfg)
            for k, df in out.items():
                self.ws.save_table(df, k, "dem")
            return out
        return self._stage("dem_gate", self._fp(self.cfg["dem_gate"]), fn,
                           lambda: {"decision": self.ws.load_table("decision", "dem")})

    def map_completion(self) -> dict | None:
        from . import map_completion as MC

        def fn():
            out = MC.evaluate(self.ctx, self.cfg)
            self.ws.save_table(out["protocol"], "corruption_protocol", "map_completion")
            self.ws.save_table(out["results"], "validation_results", "map_completion")
            self.ws.save_table(out["selection"], "selection", "map_completion")
            return out
        return self._stage("map_completion", self._fp(self.cfg["map_completion"]), fn,
                           lambda: {"selection": self.ws.load_table("selection", "map_completion")})

    # =========================================================================== report + run
    def report(self) -> Path:
        from .report import research_report

        p = self.ws.root / "GEMS3_RESEARCH_REPORT.md"
        from ..artifacts.atomic import atomic_write_text

        atomic_write_text(p, research_report(self))
        self._push()
        return p

    def run(self) -> dict:
        cfg = self.cfg
        if not cfg.get("enabled", True):
            self._say("disabled (GEMS3_CONFIG['enabled'] = False)")
            return {}
        t0 = time.perf_counter()
        out = {}
        steps = [("verify_official_metric", self.verify_official_metric),
                 ("run_metric_optimization", self.metric_optimization),
                 ("prepare_submission_candidate", self.submission),
                 ("run_candidate_generation", self.candidates),
                 ("run_candidate_oracle", self.oracle),
                 ("run_candidate_ranker", self.ranker),
                 ("run_dem_gate", self.dem_gate),
                 ("run_map_completion", self.map_completion)]
        for switch, fn in steps:
            if cfg.get(switch):
                out[switch] = fn()
        out["report"] = self.report()
        self._say(f"finished in {time.perf_counter() - t0:.0f}s -> {self.ws.root}")
        return out

    def summary(self) -> pd.DataFrame:
        """Held-out (cross-fitted) DTI of every method per task: the comparison the decisions rest on."""
        from .report import summary_table

        return summary_table(self)
