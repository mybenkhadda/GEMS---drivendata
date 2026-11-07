"""``RecoveryRun`` - the object behind every cell of GEMS_GEOREGIME_RECOVERY_V2.ipynb.

Inference-only: no method trains, fine-tunes or calls ``GeoRegimeExperiment.train_unit`` / ``predict`` /
``export``. V1 and Atlas trees are read-only (every write is checked by ``RecoveryPaths.check_writable``).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from . import crossfit as CF
from . import expert_analysis as EA
from . import fault_proximity as FPX
from . import figures as FG
from . import geological_evidence as GE
from . import metrics as MX
from . import prediction_diagnostics as PD
from . import reports as RP
from . import split_audit as SA
from . import structural_analysis as ST
from . import submission as SUB
from . import transforms as TR
from .cache import (HashCache, StageStore, atomic_write_csv, atomic_write_geotiff, atomic_write_json,
                    atomic_write_parquet, read_json, read_raster, set_guard)
from .config import code_hash, config_hash, package_files, resolve, stable_hash
from .contracts import StageBlocked, StageResult, check_schema
from .inference import FullAreaInference, artifact_table
from .paths import RecoveryPaths
from .preflight import run_preflight


class RecoveryRun:
    def __init__(self, cfg: dict | None = None):
        self.cfg = resolve(cfg)
        self.cfg_hash = config_hash(self.cfg)
        self.paths = RecoveryPaths(self.cfg).ensure()
        set_guard(self.paths)
        self.output_root = self.paths.root
        self.code_root = Path(__file__).resolve().parents[2]
        self.code_hash = code_hash(package_files())
        self.hashes = HashCache(self.paths.d("cache", "hash") / "sha256.jsonl")
        self.stages = StageStore(self.paths.d("manifests"), self.hashes)
        self.scorer = MX.check_constants(self.cfg)
        self.bank = None
        self.src_cfg: dict = {}
        self.ckpt: dict = {}
        self.v1pred: dict = {}
        self.ckpt_pred_sha: dict = {}
        self.units = None
        self.split_verified = False
        self.inference: FullAreaInference | None = None
        self.evidence = None
        self.source_fp = None
        self.results: dict = {}
        self.interpretations: dict = {}
        self.figures: dict = {}
        self._mem: dict = {}
        self._arrays: OrderedDict = OrderedDict()
        self._bank_ready = False

    # ================================================================== utilities required by the notebook
    def describe(self) -> str:
        c, P = self.cfg, self.paths
        return "\n".join([
            "GEMS GeoRegime V2 recovery - inference only (train_models=False)",
            f"  code commit       : {c.get('code_commit') or 'unknown (set by the bootstrap cell)'}",
            f"  code hash         : {self.code_hash}   config hash: {self.cfg_hash}",
            f"  source V1 run     : {P.source_run}   (read-only)",
            f"  Atlas             : {P.atlas}   (read-only)",
            f"  output namespace  : {P.root}",
            f"  scratch           : {P.scratch}",
            f"  units             : {c['units']}",
            f"  metric            : {self.scorer['radius_m']} m kernel, alpha {self.scorer['alpha']}, beta "
            f"{self.scorer['beta']}, exclude_supports_tp={self.scorer['exclude_supports_tp']} (unverified reading)",
            f"  full-area scoring : primary exclusion {c['scoring']['primary_exclusion']}, sensitivity "
            f"{c['scoring']['sensitivity_exclusion']}",
            f"  policies          : {[p['name'] for p in c['transforms']['policies']]}",
            f"  reuse caches      : {c['reuse_valid_cache']}   full-area inference: {c['compute_full_area_holdout_inference']}",
            f"  submission export : {'ENABLED (guarded)' if c['enable_submission_export'] else 'disabled'}",
        ])

    def save_stage_manifest(self, name: str, result: StageResult) -> Path:
        rec = {**result.summary_dict(), "config_hash": self.cfg_hash, "code_hash": self.code_hash,
               "code_commit": self.cfg.get("code_commit"), "source_fingerprint": self.source_fp,
               "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        return atomic_write_json(self.paths.d("manifests") / f"{name}_result.json", rec)

    def plot_group(self, name: str) -> dict:
        if name not in FG.GROUPS:
            raise KeyError(f"unknown figure group {name}; known: {sorted(FG.GROUPS)}")
        stage = RP.GROUP_STAGE[name]
        R = self.results.get(stage)
        if R is None:
            raise StageBlocked(f"plot_group('{name}') needs RUN.{stage}() first")
        W = FG.Writer(self, name)
        FG.GROUPS[name](self, W, R)
        b = RP.interpret(name, self)
        self.interpretations[name] = b
        self.figures[name] = W.saved
        md = RP.to_markdown(name, b)
        try:
            from IPython.display import Markdown, display

            display(Markdown(md))
        except ImportError:
            print(md)
        man = read_json(self.paths.d("07_figures") / "figure_manifest.json") or {}
        man[name] = {"files": [str(p) for p in W.saved], "skipped": W.skipped, "interpretation": b,
                     "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_write_json(self.paths.d("07_figures") / "figure_manifest.json", man)
        return {"saved": W.saved, "skipped": W.skipped, "interpretation": b}

    def artifact_inventory(self) -> pd.DataFrame:
        rows = []
        for p in sorted(self.paths.root.rglob("*")):
            if p.is_file() and ".partial" not in p.name:
                rel = p.relative_to(self.paths.root).as_posix()
                rows.append({"path": rel, "folder": rel.split("/")[0], "bytes": p.stat().st_size, "type": p.suffix.lstrip("."),
                             "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))})
        df = pd.DataFrame(rows, columns=["path", "folder", "bytes", "type", "modified_utc"])
        atomic_write_csv(self.paths.d("09_reports") / "artifact_inventory.csv", df)
        return df

    # ================================================================== internal helpers
    def _need(self, what: str):
        if self.bank is None:
            raise StageBlocked(f"{what} needs RUN.preflight() first")

    def _tables(self, folder: Path, tables: dict) -> list[Path]:
        outs = []
        for stem, df in tables.items():
            if not isinstance(df, pd.DataFrame):
                continue
            check_schema(stem, df)
            if self.cfg["save_csv"] or not self.cfg["save_parquet"]:
                outs.append(atomic_write_csv(folder / f"{stem}.csv", df))
            if self.cfg["save_parquet"]:
                outs.append(atomic_write_parquet(folder / f"{stem}.parquet", df))
        return outs

    def _store(self, res: StageResult) -> StageResult:
        self.results[res.stage] = res
        return res

    def _cached(self, stage: str, fp: str, folder: Path, compute):
        """Reuse a COMPLETE stage record with the same fingerprint (tables reloaded) or compute it."""
        ok, why = self.stages.valid(stage, fp) if self.cfg["reuse_valid_cache"] else (False, "cache reuse disabled")
        if ok:
            rec = self.stages.load(stage)
            tables = {}
            for k, p in rec["inputs"].get("tables", {}).items():
                p = Path(p)
                if p.suffix == ".csv" and p.is_file():
                    tables[k] = pd.read_csv(p)
                elif p.with_suffix(".parquet").is_file():
                    tables[k] = pd.read_parquet(p.with_suffix(".parquet"))
            print(f"[{stage}] CACHED - fingerprint {fp} (outputs re-verified by size and sha256)")
            return StageResult(stage, rec["stats"].get("_status", "COMPLETE"), tables, rec["stats"],
                               [o["path"] for o in rec["outputs"]], rec["inputs"].get("limitations", []), cached=True)
        print(f"[{stage}] COMPUTING ({why})")
        t0 = time.time()
        res, names = compute()
        outs = list(res.outputs)
        loc = {}
        for key, stem in names.items():
            if key in res.tables:
                outs += self._tables(folder, {stem: res.tables[key]})
                ext = ".csv" if (self.cfg["save_csv"] or not self.cfg["save_parquet"]) else ".parquet"
                loc[key] = str(folder / f"{stem}{ext}")
        res.outputs = outs
        self.stages.record(stage, fp, outs, {**res.stats, "_status": res.status},
                           {"tables": loc, "limitations": res.limitations, "seconds": round(time.time() - t0, 1)})
        print(f"[{stage}] done in {time.time() - t0:.0f} s")
        return res

    def _fp(self, *extra) -> str:
        return stable_hash([self.source_fp, self.cfg_hash, self.code_hash, *extra])

    def _arr(self, key, loader):
        if key in self._arrays:
            self._arrays.move_to_end(key)
            return self._arrays[key]
        a = loader()
        self._arrays[key] = a
        while len(self._arrays) > 10:
            self._arrays.popitem(last=False)
        return a

    def prepare_bank(self):
        """Stage the V1 feature/context bank locally: read-only reuse of the V1 Drive cache, else rebuild locally."""
        from gems_bench.georegime.data import stage_file

        self._need("prepare_bank")
        if self._bank_ready:
            return self.bank
        b = self.bank
        for name, local, producer, attr in ((f"bank_{b.key}.npy", b.bank_local, b._make_bank, "array"),
                                            (f"context_{b.key}.npy", b.aux_local, b._make_context, "context")):
            src = self.paths.v1_cache / name
            if local.is_file() and (not src.is_file() or src.stat().st_size == local.stat().st_size):
                print(f"[bank] local {local.name} reused")
            elif src.is_file():
                print(f"[bank] staging read-only V1 cache {src.name}")
                stage_file(src, local.parent)
            else:
                print(f"[bank] {name} absent from the V1 cache - rebuilding from the Atlas into local scratch only")
                producer()
            setattr(b, attr, np.load(local, mmap_mode="r"))
        if b.array.shape != (len(b.names), *b.shape) or b.context.shape != (3, *b.shape):
            raise RuntimeError("staged bank / context has an unexpected shape")
        self._bank_ready = True
        return b

    # ---- arrays used by stages and figures
    def v1_full_prob(self) -> np.ndarray:
        self._need("v1_full_prob")
        return self._arr("v1_full", lambda: read_raster(self.paths.v1_prediction(self.cfg["units"]["full"])))

    def full_area(self, unit: str, ch: str = "prob") -> np.ndarray:
        return self._arr(("fa", unit, ch), lambda: self.inference.load(unit, ch))

    def dist_mapped(self) -> np.ndarray:
        from scipy import ndimage

        return self._arr("dist_mapped", lambda: (ndimage.distance_transform_edt(~self.bank.labels) * 100.0).astype(np.float32))

    def region_map(self) -> np.ndarray:
        def make():
            r = np.full(self.bank.shape, -1, np.int16)
            for i, u in enumerate(self.cfg["units"]["geographic"]):
                r[self.units[u].zone] = i
            return r
        self._ensure_units()
        return self._arr("region", make)

    @property
    def sample(self):
        return self._arr("sample", lambda: EA.block_sample(self.bank.valid, int(self.cfg["diagnostics"]["sample_block_px"]),
                                                           int(self.cfg["seed"])))

    def regime_prior(self) -> np.ndarray:
        self.prepare_bank()
        return self._arr("prior", lambda: EA.regime_prior(self.bank))

    def gravity_context(self):
        try:
            self.prepare_bank()
        except Exception as e:                                       # noqa: BLE001
            print(f"[gravity context] unavailable: {e}")
            return None
        return self._arr("ctx", lambda: np.where(self.bank.valid, np.asarray(self.bank.context[1], np.float32), np.nan))

    def support_raster(self):
        pol = next((p for p in self.cfg["transforms"]["policies"] if p["kind"] == "support"), None)
        if pol is None or self.evidence is None:
            return None
        if "support" not in self._mem:
            S, info = TR.support_raster(self.evidence, self.bank.valid, pol)
            self._mem["support"], self._mem["support_info"] = S, info
            if S is not None:
                atomic_write_geotiff(self.paths.d("06_postprocessing") / "geo_support_S.tif", S, self.bank.profile,
                                     ["label-free structural support S (0-1)"])
                atomic_write_json(self.paths.d("06_postprocessing") / "geo_support_S.json", info)
        return self._mem["support"]

    def error_maps(self, unit: str):
        d = self.paths.d("03_validation", "tp_fp_fn_spatial")
        files = {k: d / f"{unit}_{k}.tif" for k in ("tp_support", "fp_mass", "fn_support")}
        if not all(f.is_file() for f in files.values()):
            return None

        def load():
            m = {k: read_raster(f) for k, f in files.items()}
            m["truth"] = np.isfinite(m["tp_support"])
            m["tp_support"] = np.nan_to_num(m["tp_support"])
            m["fn_support"] = np.nan_to_num(m["fn_support"])
            m["fp_mass"] = np.nan_to_num(m["fp_mass"])
            return m
        return self._arr(("err", unit), load)

    def _ensure_units(self):
        self._need("split reconstruction")
        if self.units is None:
            self.units = SA.reconstruct(self.bank, self.src_cfg, self.cfg["units"]["full"])

    def _require_splits(self):
        if not self.split_verified:
            print("[splits] verification not run in this session - running the split audit first")
            self.audit_original_splits()

    def _disc(self):
        return [u for u in self.cfg["units"]["discovery"] if self.inference.cached(u)[0]]

    def _geo(self):
        return [u for u in self.cfg["units"]["geographic"] if self.inference.cached(u)[0]]

    # ================================================================== 02
    def preflight(self) -> StageResult:
        res = run_preflight(self)
        t = res.table
        self.ckpt_pred_sha = {r.key: r.sha256 for r in t.itertuples() if r.role == "v1_prediction" and r.status == "OK"}
        self.inference = FullAreaInference(self)
        self.evidence = GE.AtlasEvidence(self.paths.atlas, self.bank.valid)
        if self.evidence.error:
            res.limitations.append(f"Atlas reload API unavailable: {self.evidence.error}")
        return self._store(res)

    # ================================================================== 03
    def baseline_diagnostics(self) -> StageResult:
        self._need("baseline_diagnostics")
        out = PD.compute(self.v1_full_prob(), self.bank.valid, self.bank.labels, self.cfg)
        ref = self.cfg["reference_values"]
        s = out["stats"]
        cmp_rows = pd.DataFrame([
            {"quantity": "reference: eligible pixels outside mapped (user-reported)", "value": ref["eligible_outside_mapped"],
             "unit": "px", "note": f"recomputed {s['eligible_pixels_outside_mapped']}"},
            {"quantity": "reference: mass outside mapped (user-reported)", "value": ref["mass_outside_mapped"], "unit": "sum p",
             "note": f"recomputed {s['total_mass_outside_mapped']:.1f}"}])
        summary = pd.concat([out["summary"], cmp_rows], ignore_index=True)
        tables = {"summary": summary, "thresholds": out["thresholds"], "lorenz": out["lorenz"],
                  "histogram": out["histogram"], "exceedance": out["exceedance"]}
        v1d = self.paths.src("tables", "prediction_mass_diagnostic.csv")
        if v1d.is_file():
            tables["v1_prediction_mass_diagnostic"] = pd.read_csv(v1d)
        d = self.paths.d("08_tables")
        outs = self._tables(d, {"baseline_summary": summary, "baseline_thresholds": out["thresholds"],
                                "baseline_lorenz": out["lorenz"], "baseline_histogram": out["histogram"],
                                "baseline_exceedance": out["exceedance"]})
        lim = [] if s["nan_inside_eligible"] == 0 else [f"{s['nan_inside_eligible']} NaN pixels inside the footprint"]
        return self._store(StageResult("baseline_diagnostics", "COMPLETE", tables, s, outs, lim))

    # ================================================================== 04
    def known_fault_proximity_diagnostics(self) -> StageResult:
        self._need("known_fault_proximity_diagnostics")
        out = FPX.compute(self.v1_full_prob(), self.bank.valid, self.bank.labels, self.bank.transform, self.cfg)
        self._arrays["dist_mapped"] = out["_dist"]
        self._mem["conc_grid"] = out["_conc_grid"]
        tables = {"table": out["table"], "cumulative": out["cumulative"], "endpoints": out["endpoints"],
                  "far_components": out["far_components"], "blocks": out["blocks"]}
        outs = self._tables(self.paths.d("08_tables"), {
            "proximity_bands": out["table"], "proximity_cumulative": out["cumulative"],
            "proximity_endpoints": out["endpoints"], "high_confidence_far_components": out["far_components"],
            "proximity_blocks": out["blocks"]})
        return self._store(StageResult("known_fault_proximity_diagnostics", "COMPLETE", tables, out["stats"], outs,
                                       ["mapped-fault distance is a diagnostic only; never a model or postprocessing input"]))

    # ================================================================== 05
    def audit_original_splits(self) -> StageResult:
        self._ensure_units()
        res = SA.audit(self)
        return self._store(res)

    # ================================================================== 06
    def recompute_original_metrics(self) -> StageResult:
        self._ensure_units()
        self._require_splits()
        v1t = self.paths.src("tables", "validation_components.csv")
        v1 = pd.read_csv(v1t) if v1t.is_file() else pd.DataFrame()
        rows, lim, comps = [], [], {}
        for u in self.cfg["units"]["geographic"] + self.cfg["units"]["discovery"]:
            U = self.units[u]
            tif = self.paths.v1_prediction(u)
            if not tif.is_file():
                lim.append(f"{u}: V1 prediction raster missing - original score not reproducible")
                continue
            pred = read_raster(tif)
            ev = U.zone & self.bank.valid
            fin = np.isfinite(pred) & self.bank.valid
            meta = (self.v1pred.get(u) or {}).get("meta") or {}
            if not np.array_equal(fin, ev):
                roi = "ROI_MISMATCH (raster footprint != this unit's zone: V1 ROI-blind cache)"
            elif meta.get("pixels") not in (None, int(ev.sum())):
                roi = "META_PIXELS_MISMATCH"
            else:
                roi = "ROI_MATCH"
            c = MX.score(pred, U.truth, ev, U.v1_exclude)
            comps.setdefault(U.protocol, []).append(c)
            r = MX.row(c, unit=u, protocol=U.protocol, source="recomputed", roi_status=roi,
                       raster_finite_px=int(fin.sum()), zone_px=int(ev.sum()), v1_meta_pixels=meta.get("pixels"))
            if len(v1):
                m = v1[(v1.unit == u) & (v1.protocol == U.protocol)]
                if len(m):
                    last = m.iloc[-1]
                    r.update(dti_v1_table=float(last.dti), abs_diff_dti=abs(float(last.dti) - c.dti),
                             abs_diff_tp=abs(float(last.tp_w) - c.tp), abs_diff_fp=abs(float(last.fp_w) - c.fp),
                             n_v1_table_rows=int(len(m)))
            rows.append(r)
        comp = pd.DataFrame(rows)
        pooled, stats = [], {}
        for proto, cs in comps.items():
            pc = MX.pool(cs)
            js = self.paths.src("tables", "geographic_dti.json" if proto == "GEOGRAPHIC" else "discovery_dti.json")
            ref = read_json(js) or {}
            pooled.append(MX.row(pc, protocol=proto, n_units=len(cs), v1_json_dti=ref.get("dti"),
                                 abs_diff_vs_v1_json=abs(ref["dti"] - pc.dti) if "dti" in ref else None))
            stats[f"pooled_{proto}"] = pc.dti
        stats["max_abs_diff_vs_v1_table"] = float(comp.abs_diff_dti.max()) if "abs_diff_dti" in comp else None
        stats["n_roi_mismatch"] = int((comp.roi_status != "ROI_MATCH").sum()) if len(comp) else 0
        pooled = pd.DataFrame(pooled)
        d = self.paths.d("03_validation")
        outs = self._tables(d, {"original_metrics": comp, "original_pooled": pooled})
        outs.append(atomic_write_json(d / "scorer_settings.json", {**self.scorer, "original_protocols": {
            "GEOGRAPHIC": "eval = zone & valid; truth = labels & zone; no exclusion",
            "WITHHELD_SYSTEM": "eval = dilate(held, 30 px) & valid; truth = held; exclude = labels & ~held (pixel-exact)"}}))
        return self._store(StageResult("recompute_original_metrics", "COMPLETE",
                                       {"components": comp, "pooled": pooled}, stats, outs, lim))

    # ================================================================== 07
    def ensure_full_area_predictions(self) -> StageResult:
        self._need("ensure_full_area_predictions")
        self._ensure_units()
        self._require_splits()
        if self.cfg["train_models"]:
            raise RuntimeError("training is forbidden in this milestone")
        units = self.cfg["units"]["geographic"] + self.cfg["units"]["discovery"] + [self.cfg["units"]["full"]]
        lim = []
        for u in units:
            ok, why = self.inference.cached(u)
            if ok:
                print(f"[infer {u}] CACHED - FULL_AREA identity {self.inference.key(u)} verified")
                continue
            if not self.cfg["compute_full_area_holdout_inference"]:
                print(f"[infer {u}] SKIPPED ({why}); compute_full_area_holdout_inference=False")
                lim.append(f"{u}: full-area inference not available ({why})")
                continue
            print(f"[infer {u}] COMPUTING ({why})")
            self.prepare_bank()
            self.inference.run_unit(u)
        t = artifact_table(self.inference, units)
        out = self._tables(self.paths.d("02_full_inference"), {"inference_artifacts": t})
        n_ok = int((t.status == "COMPLETE").sum())
        rep = t.reproduction_status.fillna("n/a").tolist()
        stats = {"n_complete": n_ok, "n_units": len(units), "reproduction": dict(zip(t.unit, rep)),
                 "any_differs": bool((t.reproduction_status == "DIFFERS").any())}
        return self._store(StageResult("ensure_full_area_predictions", "COMPLETE" if n_ok == len(units) else "PARTIAL",
                                       {"artifact_table": t}, stats, out, lim))

    # ================================================================== 08
    def score_full_area_holdouts(self) -> StageResult:
        self._need("score_full_area_holdouts")
        self._ensure_units()
        prim, sens = self.cfg["scoring"]["primary_exclusion"], self.cfg["scoring"]["sensitivity_exclusion"]
        disc, geo = self._disc(), self._geo()
        if not disc:
            raise StageBlocked("no complete full-area discovery prediction - run section 07")

        def compute():
            rows, comps = [], {}
            valid = self.bank.valid
            for u in disc + geo:
                U = self.units[u]
                p = self.full_area(u)
                tasks = []
                if U.protocol == "WITHHELD_SYSTEM":
                    v1 = self.paths.v1_prediction(u)
                    if v1.is_file():
                        tasks.append(("WITHHELD_SYSTEM_NARROW_V1", "v1", read_raster(v1), U.zone & valid, U.v1_exclude))
                    tasks += [("DISCOVERY_NARROW_REINFERRED", "v1", p, U.zone & valid, U.v1_exclude),
                              ("DISCOVERY_FULL_AREA", prim, p, valid, U.exclusion(prim)),
                              ("DISCOVERY_FULL_AREA", sens, p, valid, U.exclusion(sens))]
                else:
                    tasks += [("GEOGRAPHIC_ZONE_REINFERRED", "none", p, U.zone & valid, U.v1_exclude),
                              ("GEOGRAPHIC_FULL_AREA", prim, p, valid, U.exclusion(prim)),
                              ("GEOGRAPHIC_FULL_AREA", sens, p, valid, U.exclusion(sens))]
                for proto, ex, pred, ev, excl in tasks:
                    c = MX.score(pred, U.truth, ev, excl)
                    comps.setdefault((proto, ex), []).append(c)
                    rows.append(MX.row(c, unit=u, protocol=proto, exclusion=ex, eval_px=int(ev.sum()),
                                       excluded_px=int((excl & ev).sum()),
                                       fp_share_of_denominator=self.scorer["alpha"] * c.fp /
                                       max(c.tp + self.scorer["alpha"] * c.fp + self.scorer["beta"] * c.fn, 1e-12)))
            comp = pd.DataFrame(rows)
            pooled, stats = [], {}
            for (proto, ex), cs in sorted(comps.items(), key=lambda kv: (kv[0][0], kv[0][1] not in ("v1", prim, "none"))):
                pc = MX.pool(cs)
                pooled.append(MX.row(pc, protocol=proto, exclusion=ex, n_units=len(cs)))
                key = f"pooled_{proto}" + ("_sensitivity" if ex == sens else "")
                stats[key] = pc.dti
                if proto == "DISCOVERY_FULL_AREA" and ex == prim:
                    stats["full_area_fp_share"] = self.scorer["alpha"] * pc.fp / max(
                        pc.tp + self.scorer["alpha"] * pc.fp + self.scorer["beta"] * pc.fn, 1e-12)
            pooled = pd.DataFrame(pooled)
            nar = stats.get("pooled_DISCOVERY_NARROW_REINFERRED")
            stats["full_to_narrow_ratio"] = stats.get("pooled_DISCOVERY_FULL_AREA", np.nan) / nar if nar else np.nan
            return (StageResult("score_full_area_holdouts", "COMPLETE", {"components": comp, "pooled": pooled}, stats, [],
                                ["full-area pseudo-discovery: unmapped real faults count as false positives",
                                 "held labels are used only as scoring truth, never as inference input"]),
                    {"components": "full_area_metrics", "pooled": "pooled_metrics"})

        fp = self._fp([self.inference.key(u) for u in disc + geo], prim, sens)
        return self._store(self._cached("score_full_area_holdouts", fp, self.paths.d("03_validation"), compute))

    # ================================================================== 09
    def spatial_error_atlas(self) -> StageResult:
        self._need("spatial_error_atlas")
        self._ensure_units()
        disc = self._disc()
        if not disc:
            raise StageBlocked("no complete full-area discovery prediction - run section 07")
        prim = self.cfg["scoring"]["primary_exclusion"]
        from scipy import ndimage

        def compute():
            d = self.paths.d("03_validation", "tp_fp_fn_spatial")
            valid, region = self.bank.valid, self.region_map()
            outs, reg_rows, blocks, regions, fpd, prof = [], [], [], [], [], []
            bid, _ = MX.block_ids(valid.shape, int(self.cfg["diagnostics"]["region_block_px"]))
            nb = int(bid.max()) + 1
            for u in disc:
                U = self.units[u]
                p = self.full_area(u)
                cm = MX.ComponentMaps(U.truth, valid, U.exclusion(prim))
                mp = cm.maps(p)                                     # raises if sums disagree with MaskedScorer
                G = mp["truth"]
                for k, arr, m in (("tp_support", mp["tp_support"], G), ("fp_mass", mp["fp_mass"], valid),
                                  ("fn_support", mp["fn_support"], G)):
                    outs.append(atomic_write_geotiff(d / f"{u}_{k}.tif", np.where(m, arr, np.nan), self.bank.profile,
                                                     [f"{u} {k} (full-area discovery, {prim} exclusion)"]))
                self._arrays.pop(("err", u), None)
                rid = np.where(region >= 0, region, int(region.max()) + 1).astype(np.int32)
                rs = MX.block_sums(mp, rid, int(rid.max()) + 1)
                for r in range(len(rs["tp_w"])):
                    tp, fp_, fn, n = rs["tp_w"][r], rs["fp_w"][r], rs["fn_w"][r], rs["n_truth"][r]
                    reg_rows.append({"unit": u, "region": f"zone {r}" if r < len(rs["tp_w"]) - 1 else "outside zones",
                                     "dti": MX.dti_from(tp, fp_, fn), "tp_w": tp, "fp_w": fp_, "fn_w": fn, "n_truth": int(n),
                                     "pred_mass": float(np.nan), "recall_w": tp / n if n else np.nan,
                                     "precision_w": tp / (tp + fp_) if tp + fp_ > 0 else np.nan})
                bs = MX.block_sums(mp, bid, nb)
                keep = np.flatnonzero((bs["n_truth"] > 0) | (bs["fp_w"] > 0))
                nbc = -(-valid.shape[1] // int(self.cfg["diagnostics"]["region_block_px"]))
                B = int(self.cfg["diagnostics"]["region_block_px"])
                blocks.append(pd.DataFrame({"unit": u, "block": keep, "row0": (keep // nbc) * B, "col0": (keep % nbc) * B,
                                            **{k: v[keep] for k, v in bs.items()}}))
                regions += GE.regions_for_unit(u, U, p, mp, self.evidence, valid, self.cfg, self.bank.transform)
                # confidence and FP mass vs distance to training-visible traces (diagnostic)
                dv = ndimage.distance_transform_edt(~U.training_visible) * 100.0
                scored = valid & ~U.exclusion(prim)
                edges = np.r_[0, np.logspace(2, np.log10(40000), 25)]
                idx = np.digitize(dv[scored], edges) - 1
                pv, fv = np.nan_to_num(p[scored]), mp["fp_mass"][scored]
                cnt = np.bincount(idx, minlength=len(edges))
                sp = np.bincount(idx, weights=pv, minlength=len(edges))
                sf = np.bincount(idx, weights=fv, minlength=len(edges))
                for i in range(len(edges) - 1):
                    if cnt[i]:
                        fpd.append({"unit": u, "d_lo_m": edges[i], "d_hi_m": edges[i + 1],
                                    "d_mid_m": np.sqrt(max(edges[i], 50) * edges[i + 1]), "n_px": int(cnt[i]),
                                    "mean_p": sp[i] / cnt[i], "fp_mass": sf[i], "fp_mass_share": sf[i] / max(fv.sum(), 1e-12)})
                # fault-normal profiles: mean p vs distance from held truth and from visible traces
                dh = ndimage.distance_transform_edt(~G) * 100.0
                pe = np.arange(0, float(self.cfg["diagnostics"]["profile_max_m"]) + 100, 100)
                for ref, dist in (("held truth", dh), ("training-visible trace", dv)):
                    m = valid & (dist <= pe[-1])
                    k = np.minimum((dist[m] // 100).astype(int), len(pe) - 1)
                    c_ = np.bincount(k, minlength=len(pe))
                    s_ = np.bincount(k, weights=np.nan_to_num(p[m]), minlength=len(pe))
                    for i in range(len(pe)):
                        if c_[i]:
                            prof.append({"unit": u, "reference": ref, "d_mid_m": pe[i], "mean_p": s_[i] / c_[i], "n_px": int(c_[i])})
                del mp, cm
            rt, rg, fpdt = pd.DataFrame(reg_rows), pd.DataFrame(regions), pd.DataFrame(fpd)
            bl = pd.concat(blocks, ignore_index=True)
            stats = {"n_false_positive": int((rg.kind == "false_positive").sum()) if len(rg) else 0,
                     "n_missed": int((rg.kind == "missed").sum()) if len(rg) else 0,
                     "n_correct": int((rg.kind == "correct").sum()) if len(rg) else 0}
            if len(fpdt):
                tot = fpdt.fp_mass.sum()
                stats["fp_share_within_1km_visible"] = float(fpdt[fpdt.d_hi_m <= 1000].fp_mass.sum() / tot) if tot else np.nan
                stats["fp_share_beyond_3km_visible"] = float(fpdt[fpdt.d_lo_m >= 3000].fp_mass.sum() / tot) if tot else np.nan
            return (StageResult("spatial_error_atlas", "COMPLETE",
                                {"region_table": rt, "blocks": bl, "regions": rg, "fp_by_visible_distance": fpdt,
                                 "profiles": pd.DataFrame(prof)}, stats, outs,
                                ["FP regions may contain real unmapped faults (held labels are incomplete)"]),
                    {"region_table": "regional_metrics", "blocks": "block_metrics", "regions": "error_regions",
                     "fp_by_visible_distance": "fp_by_visible_distance", "profiles": "fault_normal_profiles"})

        fp = self._fp([self.inference.key(u) for u in disc], prim, "regions-v1")
        return self._store(self._cached("spatial_error_atlas", fp, self.paths.d("03_validation"), compute))

    # ================================================================== 10
    def expert_diagnostics(self) -> StageResult:
        self._need("expert_diagnostics")
        self._ensure_units()
        units = self._disc() + ([self.cfg["units"]["full"]] if self.inference.cached(self.cfg["units"]["full"])[0] else [])
        if not units:
            raise StageBlocked("no cached full-area heads - run section 07")
        prim = self.cfg["scoring"]["primary_exclusion"]

        def compute():
            prior, region, smp = self.regime_prior(), self.region_map(), self.sample
            acc = {k: [] for k in ("similarity", "diversity", "routing", "performance", "environment")}
            summ = []
            for u in units:
                heads = {h: self.inference.load(u, h) for h in ("prob", "global", "expert0", "expert1", "route0",
                                                                "specialist_weight")}
                out = EA.analyse_unit(u, heads, self.bank.valid, self.bank.labels, region, prior, smp, self.cfg,
                                      self.units, prim)
                for k in acc:
                    acc[k] += out[k]
                summ.append(out["summary"])
                del heads
            t = {k: pd.DataFrame(v) for k, v in acc.items()}
            t["summary"] = pd.DataFrame(summ)
            st = {"verdicts": dict(zip(t["summary"].unit, t["summary"].verdict))}
            return (StageResult("expert_diagnostics", "COMPLETE", t, st, [],
                                ["specialisation labels are descriptive thresholds fixed in config",
                                 "V1 label-derived cluster assignments are not used as routing targets"]),
                    {"similarity": "expert_similarity", "diversity": "expert_diversity", "routing": "routing_statistics",
                     "performance": "expert_performance", "environment": "expert_environment", "summary": "expert_summary"})

        fp = self._fp([self.inference.key(u) for u in units], prim)
        return self._store(self._cached("expert_diagnostics", fp, self.paths.d("04_expert_analysis"), compute))

    # ================================================================== 11
    def geological_disagreement_diagnostics(self) -> StageResult:
        self._need("geological_disagreement_diagnostics")
        self._ensure_units()
        E = self.results.get("spatial_error_atlas")
        if E is None:
            raise StageBlocked("run RUN.spatial_error_atlas() first")
        ev = self.evidence
        if ev is None or not ev.available:
            t = pd.DataFrame(columns=["kind", "flag", "share", "n"])
            return self._store(StageResult("geological_disagreement_diagnostics", "UNAVAILABLE", {"summary": t}, {}, [],
                                           [f"Atlas registry unavailable: {getattr(ev, 'error', None)}"]))
        prim = self.cfg["scoring"]["primary_exclusion"]

        def compute():
            avail = ev.table()
            regions = E.regions.copy() if "regions" in E.tables else pd.DataFrame()
            prof, resp = [], []
            for u in self._disc():
                mp = self.error_maps(u)
                if mp is None:
                    continue
                prof += GE.class_profiles(u, self.full_area(u), mp, ev, self.bank.valid,
                                          self.units[u].exclusion(prim), self.sample, self.cfg)
            p = self.v1_full_prob()
            elig = self.bank.valid & ~self.bank.labels
            for name in ("mag_persistence", "grav_thd_uc5000", "terrain_slope_break", "xm_evidence_mean",
                         "magnetic_edge", "gravity_edge"):
                a = ev.get(name)
                if a is None:
                    continue
                rc = GE.response_curve(p, a, elig, self.sample, seed=int(self.cfg["seed"]))
                if len(rc):
                    resp.append(rc.assign(evidence=name))
            ff = []
            if len(regions):
                for kind, g in regions.groupby("kind"):
                    fl = g.flags.str.split("; ").explode()
                    vc = fl.value_counts()
                    for flag, n in vc.items():
                        ff.append({"kind": kind, "flag": flag, "n": int(n), "share": float(n / len(g))})
            fft = pd.DataFrame(ff, columns=["kind", "flag", "n", "share"])
            tables = {"summary": fft, "regions": regions, "class_profiles": pd.DataFrame(prof),
                      "response": pd.concat(resp, ignore_index=True) if resp else pd.DataFrame(), "availability": avail}
            lim = [f"missing Atlas layer {k}" for k in ev.missing] + (
                ["no DEM coverage raster in the Atlas (DEM stage incomplete): DEM evidence not used"] if ev.dem is None else [])
            return (StageResult("geological_disagreement_diagnostics", "COMPLETE", tables,
                                {"n_regions": int(len(regions)), "evidence_sources": ev.source}, [], lim),
                    {"summary": "flag_frequency", "regions": "geological_disagreement", "class_profiles": "evidence_class_profiles",
                     "response": "confidence_vs_evidence", "availability": "evidence_availability"})

        fp = self._fp(self.stages.load("spatial_error_atlas").get("fingerprint") if self.stages.load("spatial_error_atlas")
                      else None, sorted(GE.EVIDENCE))
        return self._store(self._cached("geological_disagreement_diagnostics", fp, self.paths.d("05_geological_evidence"),
                                        compute))

    # ================================================================== 12
    def structural_stratification(self) -> StageResult:
        self._need("structural_stratification")
        self._ensure_units()
        disc = [u for u in self._disc() if self.error_maps(u) is not None]
        if not disc:
            raise StageBlocked("run RUN.spatial_error_atlas() first")

        def compute():
            comp, sa, rg, sk = [], [], [], []
            sstats = []
            B = int(self.cfg["diagnostics"]["region_block_px"])
            for u in disc:
                mp = self.error_maps(u)
                comp.append(ST.components_table(u, self.units[u], mp, self.bank.transform, self.cfg))
                t, s = ST.strike_agreement(u, self.full_area(u), mp, self.cfg)
                sa.append(t)
                rng = np.random.default_rng(int(self.cfg["seed"]))
                for kind, arr in (("mapped", s.pop("mapped_strikes")), ("predicted", s.pop("predicted_strikes"))):
                    arr = arr if arr.size <= 50000 else rng.choice(arr, 50000, replace=False)
                    sk.append(pd.DataFrame({"unit": u, "kind": kind, "strike_deg": arr}))
                sstats.append(s)
                rg.append(ST.regional_recall(u, mp, self.bank.valid, B, int(self.cfg["diagnostics"]["min_truth_per_block"])))
            comp = pd.concat(comp, ignore_index=True)
            lc = ST.class_summary(comp, "length_class")
            comp["linearity_class"] = pd.cut(comp.linearity, [0, 0.9, 0.97, 1.0001], labels=["<0.90", "0.90-0.97", ">=0.97"])
            lin = ST.class_summary(comp, "linearity_class")
            ss = pd.DataFrame(sstats)
            sat = pd.concat(sa, ignore_index=True)
            allsa = sat.groupby("diff_lo_deg").n_px.sum()
            stats = {"median_strike_diff_deg": float(np.average(ss.median_diff_deg, weights=ss.n_px.clip(lower=1)))
                     if len(ss) else np.nan,
                     "share_within_20deg": float(allsa[allsa.index < 20].sum() / max(allsa.sum(), 1))}
            return (StageResult("structural_stratification", "COMPLETE",
                                {"table": lc, "components": comp, "linearity": lin, "strike_agreement": sat,
                                 "strike_stats": ss, "regional": pd.concat(rg, ignore_index=True),
                                 "strikes": pd.concat(sk, ignore_index=True)}, stats, [],
                                ["length classes are 8-connected raster fragments, not mapped fault lengths"]),
                    {"table": "length_class_recall", "components": "fault_length_recall", "linearity": "linearity_recall",
                     "strike_agreement": "strike_agreement", "strike_stats": "strike_stats", "regional": "regional_recall",
                     "strikes": "strike_samples"})

        rec = self.stages.load("spatial_error_atlas")
        fp = self._fp(rec.get("fingerprint") if rec else None)
        return self._store(self._cached("structural_stratification", fp, self.paths.d("03_validation"), compute))

    # ================================================================== 13
    def candidate_lineament_review(self) -> StageResult:
        from .candidate_review import review

        self._need("candidate_lineament_review")
        lim = []
        try:
            lin = self.evidence.atlas.load_lineaments() if self.evidence is not None and self.evidence.atlas else None
        except (FileNotFoundError, ImportError) as e:
            lin = None
            lim.append(f"Atlas candidate lineaments unavailable: {e}")
        if lin is None or not len(lin):
            empty = pd.DataFrame(columns=["candidate_id", "length_km", "label_free_support", "model_mean_p",
                                          "model_pct_rank", "agreement_class"])
            return self._store(StageResult("candidate_lineament_review", "UNAVAILABLE",
                                           {"top_candidates": empty, "table": empty}, {"n_candidates": 0}, [], lim))
        extra = {u: self.full_area(u) for u in self._disc()} if self.inference is not None else {}
        df, aux = review(lin, self.v1_full_prob(), self.bank.valid, self.bank.labels, self.bank.transform, extra)
        self._mem["lineaments"], self._mem["lineament_pixels"] = lin, aux["pixels"]
        cols = [c for c in ("candidate_id", "length_km", "strike_deg", "evidence_count", "continuity",
                            "orientation_agreement", "multiscale_persistence", "label_free_support", "model_mean_p",
                            "model_q90_p", "model_pct_rank", "agreement_class", "mapped_fault_distance_m",
                            "coverage_quality", "limitations") if c in df.columns]
        top = df.sort_values("label_free_support", ascending=False)[cols].head(25).reset_index(drop=True)
        outs = self._tables(self.paths.d("05_geological_evidence"), {"candidate_model_agreement": df})
        st = aux["stats"]
        st["reference_n_lineaments"] = self.cfg["reference_values"]["n_lineaments"]
        lim.append("mapped_fault_distance_m is an Atlas diagnostic annotation; candidates are never labels")
        return self._store(StageResult("candidate_lineament_review", "COMPLETE", {"top_candidates": top, "table": df}, st,
                                       outs, lim))

    # ================================================================== 14
    def evaluate_fixed_transform_shortlist(self) -> StageResult:
        self._need("evaluate_fixed_transform_shortlist")
        self._ensure_units()
        units = self._disc() + self._geo()
        if not self._disc():
            raise StageBlocked("no complete full-area discovery prediction - run section 07")
        prim = self.cfg["scoring"]["primary_exclusion"]
        policies, lim = [], []
        S = self.support_raster()
        for p in self.cfg["transforms"]["policies"]:
            if p["kind"] == "support" and S is None:
                lim.append(f"policy {p['name']} dropped: {self._mem.get('support_info', {}).get('reason', 'no evidence')}")
                continue
            policies.append(p)
        # freeze the policy registry BEFORE any held-out score is computed
        reg = self.paths.d("06_postprocessing") / "policy_registry.json"
        frozen = read_json(reg)
        cur = {"policies": policies, "config_hash": self.cfg_hash, "support": self._mem.get("support_info")}
        if frozen is None or stable_hash(frozen.get("policies")) != stable_hash(policies):
            atomic_write_json(reg, {**cur, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "note": "written before scoring; a later change is visible as a new frozen_utc"})

        def compute():
            preds = {u: self.full_area(u) for u in units}
            comp, blocks, regions = TR.evaluate(self.units, preds, policies, S, self.bank.valid, self.region_map(), prim,
                                                int(self.cfg["diagnostics"]["region_block_px"]))
            rc = TR.response_curves(policies)
            sel = comp[comp.protocol == self.cfg["transforms"]["selection_protocol"]]
            pooled = {p["name"]: MX.dti_from(*sel[sel.policy == p["name"]][["tp_w", "fp_w", "fn_w"]].sum().to_numpy())
                      for p in policies}
            return (StageResult("evaluate_fixed_transform_shortlist", "COMPLETE",
                                {"components_by_fold": comp, "blocks": blocks, "regions": regions, "response_curves": rc},
                                {"pooled_by_policy": pooled, "policies": [p["name"] for p in policies]}, [], lim),
                    {"components_by_fold": "transform_components", "blocks": "transform_blocks",
                     "regions": "transform_regions", "response_curves": "transform_response_curves"})

        fp = self._fp([self.inference.key(u) for u in units], policies, prim)
        return self._store(self._cached("evaluate_fixed_transform_shortlist", fp, self.paths.d("06_postprocessing"), compute))

    # ================================================================== 15
    def crossfit_recovery_decision(self) -> StageResult:
        T = self.results.get("evaluate_fixed_transform_shortlist")
        if T is None:
            raise StageBlocked("run RUN.evaluate_fixed_transform_shortlist() first")
        pol_names = T.stats["policies"]
        policies = [p for p in self.cfg["transforms"]["policies"] if p["name"] in pol_names]
        if any(p["kind"] == "support" for p in policies):
            self.support_raster()                       # rebuilds provenance after a restart (label-free check)
        info = self._mem.get("support_info") or {}
        safe = {p["name"]: (p["kind"] != "support") or bool(info.get("label_free") and info.get("status") == "OK")
                for p in policies}
        reps = {u: (read_json(self.paths.d("02_full_inference", "inference_manifests") / f"{u}_full_area.json") or {})
                .get("reproduction_vs_v1", {}).get("status") for u in self.cfg["units"]["discovery"]}
        prov_ok = self.split_verified and all(self.inference.cached(u)[0] for u in self.cfg["units"]["discovery"]) and \
            all(s in ("REPRODUCED", "V1_RASTER_ABSENT") for s in reps.values())
        detail = f"splits verified={self.split_verified}; reproduction {reps}"
        out = CF.decide(T.components_by_fold, T.blocks, T.regions, policies, self.cfg, prov_ok, detail, safe)
        d = self.paths.d("06_postprocessing")
        tables = {"decision_table": out["decision_table"], "gates": out["gates"], "regional": out["regional"],
                  "bootstrap": out["bootstrap"]}
        outs = self._tables(d, {"crossfit_results": out["decision_table"], "promotion_gates": out["gates"],
                                "crossfit_regional": out["regional"], "crossfit_bootstrap": out["bootstrap"]})
        outs.append(atomic_write_json(d / "recovery_decision.json", {
            **out["stats"], "gates": out["gates"].to_dict("records"), "policies": policies, "config_hash": self.cfg_hash,
            "gates_config": self.cfg["gates"], "metric_label": MX.LABEL,
            "decided_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        print(f"[decision] {out['stats']['decision']}")
        return self._store(StageResult("crossfit_recovery_decision", "PROMOTE" if out["stats"]["promote"] else "NO_PROMOTION",
                                       tables, out["stats"], outs, ["three folds: limited statistical power"]))

    # ================================================================== 16
    def export_guarded_submission(self, decision: StageResult | None = None) -> StageResult:
        decision = decision or self.results.get("crossfit_recovery_decision")
        out = SUB.export(self, decision)
        st = out["stats"]
        print(f"[submission] {st['status']}")
        res = StageResult("export_guarded_submission", st["status"], {"validation_table": out["validation_table"]}, st,
                          [p for p in [st.get("path")] if p])
        res.path = Path(st["path"]) if st.get("path") else None
        return self._store(res)

    # ================================================================== 17
    def build_final_report(self) -> StageResult:
        out = RP.build(self)
        d = self.paths.d("09_reports")
        res = StageResult("build_final_report", "COMPLETE", {"summary": out["summary"], "inventory": out["inventory"]},
                          {"n_files": int(len(out["inventory"]))},
                          [d / n for n in ("executive_summary.md", "scientific_findings.md", "limitations.md",
                                           "artifact_manifest.json")])
        return self._store(res)
