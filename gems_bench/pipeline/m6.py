"""M6 - scientific model selection on EXISTING FULL experiments (no training).

    from gems_bench.pipeline.m6 import M6Analysis
    m6 = M6Analysis(pipe.pc, M6_CONFIG)     # or pipe.m6(M6_CONFIG)
    res = m6.run()                           # every enabled phase; resumable (fingerprinted analysis units)
    m6.audit(); m6.paired(); m6.calibration(); m6.ensembles(); m6.complementarity(); m6.postprocessing(); m6.gate()

Phases (docs/M6_RESEARCH_PLAN.md)::

    A  audit            artifact inventory, provenance, compatibility, missing artifacts (read-only); fails clearly
                        when a required OOF / discovery map is absent
       scorecard        ONE consolidated section-17 scorecard of every analysed model (same folds, splits, budgets)
       paired           per-fold / per-split deltas vs B0 and B1, spatial-block bootstrap, figures
    B  calibration      score / area-budget calibration strategies, threshold transfer, area-budget curves
       ensembles        B0 + B1 (mean, rank mean, nested weighted rank / robust; geo-masked and LOSO discovery weights),
                        then larger sets that must justify their members
    C  complementarity  equal-area overlap, recovery classes by length / fold / strike / contrast / junctions,
                        discovery systems, false-positive distance profile, error attribution, case-study panels
       postprocessing   per-operation nested ablation (+ a combination of operations that individually helped)
    D  gate             section-17 gate extended (M6.7) against B0 and B1; recommendation; M6_RESEARCH_REPORT.md

Outputs: ``<workspace>/reports/m6`` (pushed to ``<drive_root>/reports/m6``). FULL experiment folders are only read.
Each unit records ``manifests/units/<unit>.json`` (fingerprint of its inputs + configuration + code version and the
files it wrote); a unit whose fingerprint matches and whose outputs exist is reused unless ``overwrite_existing``.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts.atomic import atomic_save, atomic_write_json, atomic_write_text
from ..evaluation.comparative import (ModelSpec, M6ArtifactError, artifact_inventory, file_fp, grid_signature,
                                      require_complete, resolve_model_name)

M6_VERSION = "m6.1"
CONFIG_PARTS = ("m6", "model_selection.yaml")
FREE_FORM = ("reference_observations", "gate.rules", "postprocessing.grid")


class M6ConfigError(ValueError):
    pass


# =============================================================================== configuration
def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        here = f"{path}.{k}" if path else k
        if k not in base and path not in FREE_FORM:
            close = difflib.get_close_matches(k, list(base), n=1)
            raise M6ConfigError(f"unknown M6 config key '{here}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        if isinstance(v, dict) and isinstance(out.get(k), dict) and here not in FREE_FORM:
            out[k] = _merge(out[k], v, here)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_m6_config(overrides: dict | None = None, path: Path | None = None) -> dict:
    """YAML defaults (configs/m6/model_selection.yaml) + validated notebook overrides."""
    import yaml

    from ..paths import config_path

    p = Path(path) if path else config_path(*CONFIG_PARTS)
    base = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg = _merge(base, overrides or {})
    validate_m6_config(cfg)
    return cfg


def validate_m6_config(cfg: dict) -> None:
    if str(cfg.get("mode")).upper() != "ANALYZE_EXISTING":
        raise M6ConfigError(f"M6 mode must be ANALYZE_EXISTING (got {cfg.get('mode')!r}): M6 never trains")
    if cfg.get("train_models"):
        raise M6ConfigError("train_models=True is refused: M6 reuses saved FULL predictions only; train through "
                            "sections 14-15 (pipe.plan_experiments / pipe.step('advanced_models'))")
    pair = list(cfg.get("primary_pair") or [])
    if len(pair) != 2 or len(set(pair)) != 2:
        raise M6ConfigError(f"primary_pair needs two distinct experiment stems, got {pair}")
    for s in pair + list(cfg.get("controls") or []):
        if s not in cfg["models"]:
            raise M6ConfigError(f"{s} (primary_pair / controls) must also be listed in models")
    for s in cfg["ensemble"].get("extended_sets") or []:
        if not set(pair) <= set(s):
            raise M6ConfigError(f"extended ensemble set {s} must contain the primary pair {pair}")
    if cfg.get("save_rasters") not in ("selected", "all", "none"):
        raise M6ConfigError("save_rasters must be selected | all | none")
    if cfg["primary_policy"] not in cfg["policies"]:
        raise M6ConfigError("primary_policy must be one of policies")
    w = cfg["ensemble"]["weights"]
    if not w or min(w) < 0 or max(w) > 1:
        raise M6ConfigError("ensemble.weights must lie in [0, 1]")
    from ..postprocess.topology import OPERATIONS

    bad = [k for k in (cfg["postprocessing"].get("grid") or {}) if k not in OPERATIONS]
    if bad:
        raise M6ConfigError(f"unknown postprocessing operations {bad}; known: {sorted(OPERATIONS)}")


def fingerprint(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


# =============================================================================== workspace + cache
def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _s(x) -> str:
    """Table cell as a string ('' for None / NaN: a NaN path or hash is truthy and must never be used)."""
    return x if isinstance(x, str) else ""


class Workspace:
    """``reports/m6`` writer: atomic files, per-unit fingerprints and output lists, figure data next to figures."""

    SUBDIRS = ("manifests", "manifests/units", "tables", "processed", "figures", "figures/geological_case_studies")

    def __init__(self, root: Path, cfg: dict):
        self.root = Path(root)
        self.cfg = cfg
        for s in self.SUBDIRS:
            (self.root / s).mkdir(parents=True, exist_ok=True)
        self._written: list[str] = []
        self._unit: str | None = None

    # -- units -----------------------------------------------------------------
    def unit_record(self, name: str) -> dict | None:
        return _read_json(self.root / "manifests" / "units" / f"{name}.json")

    def valid(self, name: str, fp: str) -> bool:
        if self.cfg.get("overwrite_existing"):
            return False
        r = self.unit_record(name)
        return bool(r and r.get("fingerprint") == fp and r.get("status") == "done"
                    and all((self.root / o).exists() for o in r.get("outputs", [])))

    def begin(self, name: str) -> None:
        self._unit, self._written = name, []

    def finish(self, name: str, fp: str, info: dict | None = None, seconds: float | None = None) -> None:
        atomic_write_json(self.root / "manifests" / "units" / f"{name}.json",
                          {"unit": name, "fingerprint": fp, "status": "done", "m6_version": M6_VERSION,
                           "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": seconds,
                           "outputs": sorted(set(self._written)), "info": info or {}})
        self._unit = None

    def _track(self, p: Path) -> Path:
        self._written.append(Path(p).relative_to(self.root).as_posix())
        return p

    # -- writers ---------------------------------------------------------------
    def save_table(self, df: pd.DataFrame, name: str, sub: str = "tables") -> Path:
        df = df if df is not None else pd.DataFrame()
        p = self.root / sub / f"{name}.csv"
        if self.cfg.get("save_csv", True) or sub != "tables":
            atomic_save(p, lambda t: df.to_csv(t, index=False))
            self._track(p)
        if self.cfg.get("save_parquet") and sub == "tables" and len(df):
            q = p.with_suffix(".parquet")
            try:
                atomic_save(q, lambda t: _parquet_safe(df).to_parquet(t, index=False))
                self._track(q)
            except Exception as e:                                  # optional format: never block the analysis
                print(f"[m6] parquet skipped for {name}: {type(e).__name__}: {e}")
        return p

    def load_table(self, name: str, sub: str = "tables") -> pd.DataFrame:
        p = self.root / sub / f"{name}.csv"
        if not p.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def save_fig(self, fig, name: str, data=None, sub: str = "figures") -> Path | None:
        import matplotlib.pyplot as plt

        if fig is None:
            return None
        p = self.root / sub / f"{name}.png"
        if self.cfg.get("save_figures", True):
            atomic_save(p, lambda t: fig.savefig(t, dpi=170, bbox_inches="tight", format="png"))
            self._track(p)
        if data is not None:                                        # every figure's plotting data, as CSV
            d = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            q = self.root / sub / f"{name}_data.csv"
            atomic_save(q, lambda t: d.to_csv(t, index=False))
            self._track(q)
        plt.close(fig)
        return p

    def save_json(self, obj, rel: str) -> Path:
        return self._track(atomic_write_json(self.root / rel, obj))

    def save_text(self, text: str, rel: str) -> Path:
        return self._track(atomic_write_text(self.root / rel, text))

    def save_npz(self, arrays: dict, rel: str) -> Path:
        return self._track(atomic_save(self.root / rel, lambda t: np.savez_compressed(t, **arrays)))

    def save_raster(self, layers: dict, rel: str, grid) -> Path:
        from gems_eda.io import write_geotiff

        return self._track(atomic_save(self.root / rel, lambda t: write_geotiff(
            t, {k: np.asarray(v, np.float32) for k, v in layers.items()}, grid)))


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda x: x if x is None or isinstance(x, str) else str(x))
    return out


class PredictionCache:
    """Local ``.npy`` copy of each prediction GeoTIFF, opened memory-mapped (read-only). Keyed by the source path;
    rebuilt when the source fingerprint (size, first-MiB hash, mtime) changes."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, path: Path) -> np.ndarray:
        from ..ensemble import read_map

        path = Path(path)
        if not path.exists():
            raise M6ArtifactError(f"prediction raster missing: {path}")
        fp = f"{file_fp(path)}:{int(path.stat().st_mtime)}"
        key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
        npy, meta = self.root / f"{key}.npy", self.root / f"{key}.json"
        m = _read_json(meta)
        if not (m and m.get("fp") == fp and npy.exists()):
            arr = read_map(path)
            atomic_save(npy, lambda t: np.save(t, arr))
            atomic_write_json(meta, {"source": str(path), "fp": fp, "shape": list(arr.shape), "dtype": str(arr.dtype)})
        return np.load(npy, mmap_mode="r")


# =============================================================================== scorecard (de)serialisation
SC_TABLES = ("score", "geo_curves", "disc_curves", "disc_summary", "disc_per_split", "recovery", "unique", "raw_geo")


def save_scorecard(ws: Workspace, sc: dict, rel: str) -> None:
    for k in SC_TABLES:
        ws.save_table(sc.get(k, pd.DataFrame()), k, sub=rel)
    ws.save_npz({m: np.asarray(v, np.float32) for m, v in sc.get("support", {}).items()}, f"{rel}/support.npz")


def load_scorecard(ws: Workspace, rel: str) -> dict:
    sc = {k: ws.load_table(k, sub=rel) for k in SC_TABLES}
    p = ws.root / rel / "support.npz"
    sc["support"] = dict(np.load(p)) if p.exists() else {}
    sc["disc_objs"] = {}
    return sc


def merge_scorecards(a: dict, b: dict) -> dict:
    """Rows of ``b`` for models absent from ``a`` (same protocol: both come from ``evaluate_models`` on one context)."""
    out = {}
    have = set(a["score"].model) if len(a.get("score", [])) else set()
    for k in SC_TABLES:
        ta, tb = a.get(k, pd.DataFrame()), b.get(k, pd.DataFrame())
        if len(tb) and "model" in tb:
            tb = tb[~tb.model.isin(have)]
        out[k] = pd.concat([ta, tb], ignore_index=True) if len(tb) else ta
    out["support"] = {**b.get("support", {}), **a.get("support", {})}
    out["disc_objs"] = {**b.get("disc_objs", {}), **a.get("disc_objs", {})}
    return out


# =============================================================================== the analysis
class M6Analysis:
    def __init__(self, pc=None, config: dict | None = None, *, ctx=None, data_root: Path | None = None,
                 local_root: Path | None = None, config_path: Path | None = None):
        self.cfg = load_m6_config(config, config_path)
        self.pc = pc
        self.local = Path(local_root or (pc.local if pc is not None else Path.cwd()))
        self.data_root = Path(data_root or ((pc.drive or pc.local) if pc is not None else self.local))
        self._ctx = ctx
        self.ws = Workspace(self.local / self.cfg["output_dir"], self.cfg)
        self.cache = PredictionCache(self.local / self.cfg["cache_dir"])
        code = pc.code_root if pc is not None else None
        self.specs = [ModelSpec(s, resolve_model_name(s, code)) for s in self.cfg["models"]]
        self.extra_specs = [ModelSpec(s, resolve_model_name(s, code)) for s in self.cfg.get("optional_models") or []
                            if s not in self.cfg["models"]]
        self.baseline_specs = [ModelSpec(b, b, "baseline") for b in self.cfg.get("baselines") or []]
        self.name = {s.stem: s.model for s in self.specs + self.extra_specs}
        self.inv: pd.DataFrame | None = None
        self.map_fp: dict[str, str] = {}
        self._sc: dict | None = None
        self._sc_all: dict | None = None
        self._ens: dict = {}
        self.fp: dict[str, str] = {}
        self.log: list[dict] = []

    # -- helpers ---------------------------------------------------------------
    @property
    def ctx(self):
        if self._ctx is None:
            if self.pc is None:
                raise RuntimeError("M6Analysis needs a pipeline context (pc) or ctx=")
            self._ctx = self.pc.ctx5()
        return self._ctx

    def m(self, stem: str) -> str:
        return self.name.get(stem, stem)

    @property
    def pair(self) -> list[str]:
        return [self.m(s) for s in self.cfg["primary_pair"]]

    @property
    def controls(self) -> list[str]:
        return [self.m(s) for s in self.cfg.get("controls") or self.cfg["primary_pair"]]

    @property
    def ref(self) -> str:
        return self.controls[0]

    @property
    def analysed(self) -> list[str]:
        return [s.model for s in self.specs]

    def comparison_models(self) -> list[str]:
        names = list(self.analysed)
        if self.cfg.get("compare_baselines") and self.inv is not None:
            ok = self.inv[(self.inv.kind == "baseline") & (self.inv.status == "COMPLETE")].model
            names += [b for b in ok if b not in names]
        return names

    def _say(self, msg: str) -> None:
        print(f"[m6] {msg}")

    def _push(self) -> None:
        if self.pc is not None:
            try:
                rel = Path(self.ws.root).resolve().relative_to(self.local.resolve()).as_posix()
                self.pc.push([rel])
            except ValueError:
                pass

    def _unit(self, name: str, fp: str, fn, load=None):
        """Run ``fn`` unless the unit is valid; ``load`` re-reads its outputs. Records time and outcome."""
        t0 = time.perf_counter()
        self.fp[name] = fp
        if self.ws.valid(name, fp):
            self._say(f"{name}: reused (fingerprint {fp})")
            self.log.append({"unit": name, "outcome": "REUSED", "seconds": 0.0, "fingerprint": fp})
            return load() if load else None
        self.ws.begin(name)
        out = fn()
        dt = round(time.perf_counter() - t0, 1)
        self.ws.finish(name, fp, seconds=dt)
        self.log.append({"unit": name, "outcome": "COMPUTED", "seconds": dt, "fingerprint": fp})
        self._say(f"{name}: computed in {dt:.0f}s")
        self._push()
        return out

    # =========================================================================== A. audit
    def audit(self) -> dict:
        from ..evaluation.comparative import RegistryView
        from ..evaluation.protocol import protocol_from_ctx

        ctx, cfg = self.ctx, self.cfg
        proto = protocol_from_ctx(ctx, cfg["match_frac"], cfg["primary_policy"]).as_dict()
        reg = RegistryView(self.data_root / "model_registry")
        fsha = ssha = bgrid = None
        if self.pc is not None:
            from .stages import folds_sha, splits_sha

            fsha, ssha, bgrid = folds_sha(self.pc), splits_sha(self.pc), self.pc.grid(self.data_root)
        tabs = artifact_inventory(self.data_root, self.specs + self.extra_specs + self.baseline_specs,
                                  ref_grid=grid_signature(ctx.data.grid), n_folds=ctx.folds.n_folds,
                                  n_splits=ctx.splits.n_splits, current_protocol=proto, registry=reg,
                                  folds_sha=fsha, splits_sha=ssha, baseline_grid=bgrid)
        inv = tabs["inventory"]
        ref_hash = _s(inv.set_index("model").get("protocol_hash", pd.Series(dtype=object)).get(self.ref, ""))
        if ref_hash and "protocol_hash" in inv:          # one evaluation protocol for every compared number
            bad = inv.protocol_hash.notna() & (inv.protocol_hash != "") & (inv.protocol_hash != ref_hash) & \
                inv.status.isin(["COMPLETE", "PARTIAL"])
            inv.loc[bad, "reasons"] = inv.loc[bad, "reasons"].astype(str) + f"; protocol hash differs from {self.ref}"
            inv.loc[bad, "status"] = "INCOMPATIBLE"
        self.inv = inv
        self.ws.begin("audit")
        for name, df in (("artifact_inventory", inv), ("provenance_audit", tabs["provenance"]),
                         ("model_compatibility", tabs["compatibility"]), ("missing_artifacts", tabs["missing"])):
            self.ws.save_table(df, name, sub=".")
        from .._version import provenance

        self.ws.save_json(json.loads(inv.to_json(orient="records")), "manifests/artifact_inventory.json")
        self.ws.save_json({"code": provenance(), "protocol": proto, "m6_version": M6_VERSION,
                           "data_root": str(self.data_root), "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
                          "manifests/provenance.json")
        self.ws.save_json(self.cfg, "manifests/analysis_config.json")
        self.ws.finish("audit", fingerprint(time.time()))                 # the audit is always re-run (read-only)
        self._push()
        print(inv[[c for c in ("model", "status", "experiment_id", "profile", "n_disc_maps", "reasons")
                   if c in inv]].to_string(index=False))
        require_complete(inv, self.analysed, n_splits=ctx.splits.n_splits)   # fails clearly, names every gap
        self._stage_and_load()
        return tabs

    def _rel(self, p) -> str | None:
        try:
            return Path(p).resolve().relative_to(self.data_root.resolve()).as_posix()
        except (ValueError, OSError, TypeError):
            return None

    def _local(self, p) -> Path:
        r = self._rel(p)
        return self.local / r if (r and self.pc is not None and self.pc.drive is not None) else Path(p)

    def _stage_and_load(self) -> None:
        """Stage only prediction rasters + manifests / metrics (never checkpoints); load them memory-mapped."""
        ctx = self.ctx
        n = ctx.splits.n_splits
        inv = self.inv.set_index("model")
        names = self.comparison_models()
        rels = []
        for mname in names:
            r = inv.loc[mname]
            disc = [_s(r.get(f"disc{d}")) for d in range(n)]
            if r.kind == "experiment":
                run = self._rel(r.run_dir)
                rels += [f"{run}/manifest.json", f"{run}/config.yaml", f"{run}/metrics", f"{run}/reports"] if run else []
                rels += [x for x in [self._rel(_s(r.oof))] + [self._rel(p) for p in disc if p] if x]
            elif _s(r.oof):
                rels += [x for x in [self._rel(r.oof)] if x]
        if self.pc is not None and rels:
            self.pc.stage_in(list(dict.fromkeys(rels)))
            if hasattr(ctx, "refresh_zoo"):
                ctx.refresh_zoo()
        for mname in names:
            r = inv.loc[mname]
            if not _s(r.oof):
                continue
            ctx._maps[mname] = self.cache.load(self._local(r.oof))
            fps = [file_fp(self._local(r.oof))]
            if r.kind == "experiment":
                for d in range(n):
                    p = _s(r.get(f"disc{d}"))
                    if p:
                        ctx._maps[ctx.disc_key(mname, d)] = self.cache.load(self._local(p))
                        fps.append(file_fp(self._local(p)))
            self.map_fp[mname] = fingerprint(fps)
        self._say(f"maps loaded (memory-mapped): {names}")

    @property
    def protocol_hash(self) -> str:
        from ..evaluation.protocol import protocol_from_ctx

        return protocol_from_ctx(self.ctx, self.cfg["match_frac"], self.cfg["primary_policy"]).hash()

    # =========================================================================== A. scorecard + paired
    @property
    def sc(self) -> dict:
        if self._sc is None:
            self.scorecard()
        return self._sc

    def scorecard(self) -> dict:
        from ..evaluation.scoreboard import evaluate_models

        if self.inv is None:
            self.audit()
        cfg, models = self.cfg, self.comparison_models()
        fp = fingerprint(M6_VERSION, "scorecard", sorted(self.map_fp.items()), self.protocol_hash, cfg["match_frac"],
                         cfg["policies"], self.controls)

        def run():
            sc = evaluate_models(self.ctx, models, match_frac=cfg["match_frac"], refs=tuple(self.controls),
                                 policies=tuple(cfg["policies"]))
            save_scorecard(self.ws, sc, "processed/scorecard")
            return sc

        self._sc = self._unit("scorecard", fp, run, lambda: load_scorecard(self.ws, "processed/scorecard"))
        return self._sc

    def paired(self) -> dict:
        from ..evaluation import comparative as C
        from ..evaluation.scoreboard import competition_scoreboard
        from ..visualization.figures import experiment_dashboard

        sc, cfg, ctx = self.sc, self.cfg, self.ctx
        models = self.comparison_models()
        fp = fingerprint(M6_VERSION, "paired", self.fp["scorecard"], cfg["uncertainty"], cfg.get("reference_observations"))

        def run():
            board = competition_scoreboard(sc, self.protocol_hash, None, ref=self.ref)
            self.ws.save_table(board, "model_comparison")
            ft = C.fold_metrics(ctx, sc)
            dt = C.discovery_metrics(sc)
            sf = C.short_recall_by_fold(ctx, sc, models)
            self.ws.save_table(ft, "fold_metrics")
            self.ws.save_table(dt, "discovery_metrics")
            self.ws.save_table(sf, "short_recall_by_fold")
            per_all, summ_all, boot_rows = [], [], []
            block_km = cfg["uncertainty"].get("block_km") or float(ctx.folds.params.get("block_km", 20.0))
            block_px = max(int(round(block_km / ctx.data.grid.pixel_km)), 8)
            gb = C.geo_block_table(ctx, {m: ctx.oof(m) for m in models}, ft, block_px)
            db = C.discovery_block_table(ctx, dt, block_px, cfg["primary_policy"]) if len(dt) else pd.DataFrame()
            self.ws.save_table(gb, "geo_block_components", sub="processed/uncertainty")
            self.ws.save_table(db, "discovery_block_components", sub="processed/uncertainty")
            for ref in self.controls:
                p, s = C.paired_deltas(ft, ref, "fold")
                per_all.append(p.assign(task="geographic"))
                summ_all.append(s.assign(task="geographic"))
                if len(dt):
                    p, s = C.paired_deltas(dt, ref, "split", by=("policy",))
                    per_all.append(p.assign(task="discovery"))
                    summ_all.append(s.assign(task="discovery"))
                for mname in models:
                    if mname == ref:
                        continue
                    boot_rows.append({**C.block_bootstrap(gb, mname, ref, n_boot=cfg["uncertainty"]["n_boot"],
                                                          seed=cfg["uncertainty"]["seed"]), "task": "geographic",
                                      "block_km": block_km})
                    if len(db):
                        boot_rows.append({**C.block_bootstrap(db, mname, ref, n_boot=cfg["uncertainty"]["n_boot"],
                                                              seed=cfg["uncertainty"]["seed"]),
                                          "task": f"discovery_{cfg['primary_policy']}", "block_km": block_km})
            per, summ, boot = (pd.concat(x, ignore_index=True) if x else pd.DataFrame() for x in
                               (per_all, summ_all, [pd.DataFrame(boot_rows)]))
            self.ws.save_table(per, "paired_deltas")
            self.ws.save_table(summ, "paired_summary")
            self.ws.save_table(boot, "spatial_bootstrap")
            persisted = C.persisted_scores(self.inv)
            self.ws.save_table(persisted, "persisted_scores")
            self.ws.save_table(C.reference_discrepancies(cfg.get("reference_observations") or {}, persisted, board),
                               "reference_check")
            fig, data = C.fig_geo_dti_per_fold(ft, models)
            self.ws.save_fig(fig, "geo_dti_per_fold", data)
            fig, data = C.fig_geo_dti_per_fold(ft, models)
            self.ws.save_fig(fig, "fold_dti", data)
            gper = per[per.task == "geographic"] if len(per) else per
            gboot = boot[boot.task == "geographic"] if len(boot) else boot
            if len(gper):
                fig, data = C.fig_paired_differences(gper[gper.reference == self.ref], gboot[gboot.reference == self.ref]
                                                     if len(gboot) else None, self.ref, "fold")
                self.ws.save_fig(fig, "paired_dti_differences", data)
            if len(dt):
                for name in ("discovery_dti_per_split", "discovery_dti"):
                    fig, data = C.fig_discovery_per_split(dt, models, tuple(cfg["policies"]))
                    self.ws.save_fig(fig, name, data)
            fig, data = C.fig_worst_fold(ft, dt, models, cfg["primary_policy"])
            self.ws.save_fig(fig, "worst_fold_comparison", data)
            fig, data = C.fig_short_fault_recall(sc["score"], sf, models)
            self.ws.save_fig(fig, "short_fault_recall_comparison", data)
            fig, data = C.fig_score_sensitivity(sc, models, cfg["primary_policy"])
            self.ws.save_fig(fig, "score_sensitivity", data)
            self.ws.save_fig(experiment_dashboard(board, ref=self.ref), "model_comparison", board)
            return True

        self._unit("paired", fp, run)
        return {k: self.ws.load_table(k) for k in ("model_comparison", "fold_metrics", "discovery_metrics",
                                                   "paired_summary", "spatial_bootstrap", "reference_check")}

    # =========================================================================== B. calibration
    def calibration(self) -> dict:
        from ..evaluation import calibration as K
        from ..p3.transforms import FoldScorers

        cfg, ctx, sc = self.cfg, self.ctx, self.sc
        cm = self.pair if cfg["calibration"]["models"] == "primary_pair" else self.analysed
        fp = fingerprint(M6_VERSION, "calibration", self.fp["scorecard"], cfg["calibration"], cm)

        def run():
            scorers = FoldScorers(ctx.data.labels, ctx.folds.fold_map, ctx.data.valid)
            rng = np.random.default_rng(cfg["uncertainty"]["seed"])
            tabs, transfer = [], []
            for mname in cm:
                p = ctx.oof(mname)
                tabs.append(K.calibration_table(ctx, mname, p, families=tuple(cfg["calibration"]["families"]),
                                                scorers=scorers, rng=rng))
                if cfg["calibration"].get("threshold_transfer", True):
                    transfer.append(K.threshold_transfer(ctx, mname, p))
            table = pd.concat(tabs, ignore_index=True)
            summ, per = K.calibration_summary(table)
            tr = pd.concat(transfer, ignore_index=True) if transfer else pd.DataFrame()
            self.ws.save_table(table, "calibration_components", sub="processed/calibration")
            self.ws.save_table(summ, "calibration_results")
            self.ws.save_table(per, "calibration_per_fold")
            self.ws.save_table(K.invariance_check(table), "calibration_invariance")
            self.ws.save_table(tr, "threshold_transfer")
            self.ws.save_table(K.selection_audit(per), "calibration_selection_audit")
            self.ws.save_table(K.score_distribution(ctx, {m: ctx.oof(m) for m in cm}), "score_distribution")
            curves = K.area_budget_curves(sc, self.comparison_models())
            s = sc["score"].set_index("model")
            cf = {("geographic", m): s.loc[m, "geo_frac"] for m in s.index if "geo_frac" in s}
            for pol in cfg["policies"]:
                if f"disc_frac_{pol}" in s:
                    cf.update({(f"discovery_{pol}", m): s.loc[m, f"disc_frac_{pol}"] for m in s.index
                               if pd.notna(s.loc[m, f"disc_frac_{pol}"])})
            self.ws.save_fig(K.fig_area_budget_curves(curves, self.comparison_models(), cf), "area_budget_curves", curves)
            self.ws.save_fig(K.fig_calibration_summary(summ, tr), "calibration_summary", summ)
            if cfg["save_rasters"] != "none":                    # the area-budget calibrated OOF map (cross-fitted)
                from ..p3.budget import binary_mask

                ft = self.ws.load_table("fold_metrics")
                for mname in cm:
                    out = np.full(ctx.data.shape, np.nan, np.float32)
                    for r in ft[ft.model == mname].itertuples():
                        z = ctx.data.valid & (ctx.folds.fold_map == r.fold)
                        out[z] = binary_mask(np.where(z, ctx.oof(mname), np.nan), z, None,
                                             r.frac_selected_on_other_folds)[z].astype(np.float32)
                    self.ws.save_raster({"area_budget_mask": out},
                                        f"processed/calibrated_predictions/{mname}_area_budget_oof.tif", ctx.data.grid)
            return True

        self._unit("calibration", fp, run)
        return {k: self.ws.load_table(k) for k in ("calibration_results", "threshold_transfer", "calibration_invariance")}

    # =========================================================================== B. ensembles
    def _ensemble_specs(self) -> list[dict]:
        e = self.cfg["ensemble"]
        dws = list(e["discovery_weights"])
        out = [{"members": self.pair, "method": "mean", "normalisation": "none", "dw": "fixed"},
               {"members": self.pair, "method": "rank_mean", "normalisation": "rank", "dw": "fixed"}]
        out += [{"members": self.pair, "method": "weighted", "normalisation": n, "dw": dws[0]} for n in e["normalisations"]]
        out += [{"members": self.pair, "method": "weighted", "normalisation": "rank", "dw": d} for d in dws[1:]]
        usable = set(self.inv[self.inv.status == "COMPLETE"].model)
        for s in e.get("extended_sets") or []:
            mem = [self.m(x) for x in s]
            if set(mem) <= usable:
                out.append({"members": mem, "method": "weighted", "normalisation": "rank", "dw": dws[0]})
            else:
                self._say(f"extended set {mem} skipped: members not COMPLETE ({sorted(set(mem) - usable)})")
        return out

    def ensembles(self) -> dict:
        from ..discovery import nested_selection as N
        from ..evaluation import comparative as C
        from ..evaluation.scoreboard import evaluate_models

        cfg, ctx = self.cfg, self.ctx
        e = cfg["ensemble"]
        specs = self._ensemble_specs()
        fp = fingerprint(M6_VERSION, "ensembles", self.fp["scorecard"], e, specs, cfg["save_rasters"])

        def run():
            maps = {m: ctx.oof(m) for m in self.analysed}
            cands, geo_rows, disc_rows, audit_rows = [], [], [], []
            for s in specs:
                c = N.build_candidate(ctx, maps, s["members"], s["method"], s["normalisation"],
                                      discovery_weights=s["dw"], alphas=tuple(e["weights"]),
                                      simplex_step=e["simplex_step"], refine=e["refine"], refine_step=e["refine_step"],
                                      refine_halfwidth=e["refine_halfwidth"], policy=cfg["primary_policy"])
                cands.append(c)
                if len(c.geo_choice):
                    geo_rows.append(c.geo_choice.assign(candidate=c.name))
                if len(c.disc_choice):
                    disc_rows.append(c.disc_choice.assign(candidate=c.name))
                if len(c.audit):
                    audit_rows.append(c.audit.assign(candidate=c.name))
                self._say(f"ensemble {c.name}: built")
            ctx_e = replace(ctx, zoo5=dict(getattr(ctx, "zoo5", {}) or {}), _maps=dict(ctx._maps))
            for c in cands:
                ctx_e._maps[c.name] = c.oof
                for d, arr in c.disc.items():
                    ctx_e._maps[ctx_e.disc_key(c.name, d)] = arr
            names = [c.name for c in cands]
            sc_c = evaluate_models(ctx_e, names, match_frac=cfg["match_frac"], refs=tuple(self.controls),
                                   policies=tuple(cfg["policies"]))
            save_scorecard(self.ws, sc_c, "processed/scorecard_ensembles")
            allsc = merge_scorecards(self.sc, sc_c)
            inv = self.inv.set_index("model")
            cost = {m: {"n_checkpoints": int(inv.loc[m].get("n_checkpoints", 0) or 0),
                        "checkpoint_mb": float(inv.loc[m].get("checkpoint_mb", 0.0) or 0.0)} for m in self.analysed}
            tab = N.candidate_table(allsc["score"], cands, {s.stem: s.model for s in self.specs}, self.controls, cost)
            tab = self._justify_extensions(tab)
            self.ws.save_table(tab, "ensemble_candidates")
            self.ws.save_table(pd.concat(geo_rows, ignore_index=True) if geo_rows else pd.DataFrame(),
                               "ensemble_selection_geographic")
            self.ws.save_table(pd.concat(disc_rows, ignore_index=True) if disc_rows else pd.DataFrame(),
                               "ensemble_selection_discovery")
            self.ws.save_table(pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame(),
                               "ensemble_index_audit")
            self.ws.save_json({c.name: {**c.record(), "summary": c.summary} for c in cands}, "manifests/ensemble_weights.json")
            # held-out per-fold metrics + paired block bootstrap of every candidate vs both controls (gate evidence)
            ft = C.fold_metrics(ctx_e, allsc)
            self.ws.save_table(ft[ft.model.isin(names)], "ensemble_fold_metrics")
            block_km = cfg["uncertainty"].get("block_km") or float(ctx.folds.params.get("block_km", 20.0))
            block_px = max(int(round(block_km / ctx.data.grid.pixel_km)), 8)
            gb = C.geo_block_table(ctx_e, {m: ctx_e.oof(m) for m in names + self.controls},
                                   ft[ft.model.isin(names + self.controls)], block_px)
            boot = [{**C.block_bootstrap(gb, n, ref, n_boot=cfg["uncertainty"]["n_boot"], seed=cfg["uncertainty"]["seed"]),
                     "task": "geographic"} for n in names for ref in self.controls]
            self.ws.save_table(pd.DataFrame(boot), "ensemble_bootstrap")
            primary = self._primary_name(cands)
            for c in cands:
                if cfg["save_rasters"] == "all" or (cfg["save_rasters"] == "selected" and c.name == primary):
                    self.ws.save_raster({"oof_probability": c.oof}, f"processed/ensemble_predictions/{c.name}/oof.tif",
                                        ctx.data.grid)
                    for d, arr in c.disc.items():
                        self.ws.save_raster({"probability": arr}, f"processed/ensemble_predictions/{c.name}/disc{d}.tif",
                                            ctx.data.grid)
            self.ws.save_json({"primary": primary}, "manifests/ensemble_primary.json")
            self._ens = {c.name: c for c in cands}
            self._fig_ensembles(tab, cands)
            return True

        self._unit("ensembles", fp, run)
        self._sc_all = merge_scorecards(self.sc, load_scorecard(self.ws, "processed/scorecard_ensembles"))
        return {"candidates": self.ws.load_table("ensemble_candidates"),
                "bootstrap": self.ws.load_table("ensemble_bootstrap")}

    def _primary_name(self, cands) -> str:
        """The prespecified primary candidate: B0 + B1, weighted rank, primary discovery weights."""
        dw = self.cfg["ensemble"]["discovery_weights"][0]
        for c in cands:
            if c.members == self.pair and c.method == "weighted" and c.normalisation == "rank" and c.discovery_weights == dw:
                return c.name
        return cands[0].name

    def _justify_extensions(self, tab: pd.DataFrame) -> pd.DataFrame:
        """Larger member sets vs the best B0 + B1 candidate (primary discovery DTI, the protocol's objective)."""
        pp = f"disc_dti_{self.cfg['primary_policy']}"
        pair_key = "+".join(self.pair)
        pairs = tab[(tab.members == pair_key) & (tab.n_members == 2)]
        tab = tab.copy()
        tab["added_geo_vs_best_pair"] = np.nan
        tab["added_disc_vs_best_pair"] = np.nan
        tab["extension_justified"] = None
        if not len(pairs):
            return tab
        best = pairs.sort_values(pp, ascending=False).iloc[0]
        ext = tab.n_members > 2
        tab.loc[ext, "added_geo_vs_best_pair"] = tab.loc[ext, "geo_dti"] - best.geo_dti
        tab.loc[ext, "added_disc_vs_best_pair"] = tab.loc[ext, pp] - best[pp]
        tab.loc[ext, "extension_justified"] = (tab.loc[ext, "added_geo_vs_best_pair"] >= self.cfg["ensemble"]["min_member_gain"]) & \
            (tab.loc[ext, "added_disc_vs_best_pair"] >= 0)
        tab["best_pair_candidate"] = best.candidate
        return tab

    def _fig_ensembles(self, tab: pd.DataFrame, cands) -> None:
        import matplotlib.pyplot as plt

        from ..evaluation.comparative import _colors, _style

        pp = f"disc_dti_{self.cfg['primary_policy']}"
        t = tab.copy()
        t["relative_inference_cost"] = t.get("relative_inference_cost", t.n_members)
        col = _colors(list(t.candidate))
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
        for ax, metric in zip(axs, ("geo_dti", pp)):
            for r in t.to_dict("records"):
                x, y = r["relative_inference_cost"], r[metric]
                ax.scatter(x, y, color=col[r["candidate"]], s=40)
                ax.annotate(r["candidate"].replace("M6_ENS_", ""), (x, y), fontsize=6, xytext=(3, 2),
                            textcoords="offset points")
            _style(ax, "relative inference cost (single model = 1)", metric, f"{metric} vs cost")
        self.ws.save_fig(fig, "score_vs_cost", t[["candidate", "members", "relative_inference_cost", "geo_dti", pp,
                                                 "short_recall_w"]])
        prim = next((c for c in cands if c.name == self._primary_name(cands)), None)
        if prim is not None and prim.geo_choice is not None and len(prim.geo_choice):
            sel = self.ws.load_table("ensemble_selection_geographic")
            sel = sel[sel.candidate == prim.name]
            fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
            a = self.pair[0]
            for r in sel.itertuples():
                w = json.loads(r.w)[a]
                ax.scatter(w, r.dti_heldout, s=40, label=f"fold {r.zone}: dev DTI {r.dti_dev:.4f}")
            ax.set_xlim(-0.05, 1.05)
            _style(ax, f"weight of {a} (dev-selected per held-out fold)", "held-out DTI",
                   f"{prim.name}: nested weight choice per fold")
            ax.legend(fontsize=6, frameon=False)
            self.ws.save_fig(fig, "ensemble_weight_choices", sel)

    def primary_ensemble(self) -> tuple[str | None, np.ndarray | None, dict]:
        """(name, OOF map, {split: map}) of the primary ensemble: in memory, else saved rasters, else rebuilt."""
        rec = _read_json(self.ws.root / "manifests" / "ensemble_primary.json") or {}
        name = rec.get("primary")
        if not name:
            return None, None, {}
        if name in self._ens:
            c = self._ens[name]
            return name, c.oof, c.disc
        base = self.ws.root / "processed" / "ensemble_predictions" / name
        if (base / "oof.tif").exists():
            disc = {d: self.cache.load(base / f"disc{d}.tif") for d in range(self.ctx.splits.n_splits)
                    if (base / f"disc{d}.tif").exists()}
            return name, self.cache.load(base / "oof.tif"), disc
        from ..discovery import nested_selection as N

        e = self.cfg["ensemble"]
        c = N.build_candidate(self.ctx, {m: self.ctx.oof(m) for m in self.pair}, self.pair, "weighted", "rank",
                              discovery_weights=e["discovery_weights"][0], alphas=tuple(e["weights"]),
                              refine=e["refine"], refine_step=e["refine_step"], refine_halfwidth=e["refine_halfwidth"],
                              policy=self.cfg["primary_policy"])
        return c.name, c.oof, c.disc

    # =========================================================================== C. complementarity
    def complementarity(self) -> dict:
        from ..evaluation import complementarity as Q
        from ..evaluation import geological_errors as GE
        from ..p3 import discovery3
        from ..phase4_eval import fig_disagreement, label_pixel_table

        cfg, ctx = self.cfg, self.ctx
        cc = cfg["complementarity"]
        a, b = self.pair
        fp = fingerprint(M6_VERSION, "complementarity", self.fp["scorecard"], self.fp.get("ensembles"), cc,
                         cfg["budgets"])

        def run():
            maps = {m: ctx.oof(m) for m in (a, b)}
            ens, ens_oof, ens_disc = self.primary_ensemble()
            sc_all = self._sc_all or self.sc
            support = dict(sc_all.get("support", {}))
            if ens and ens_oof is not None:
                maps[ens] = ens_oof
            self.ws.save_table(Q.pair_statistics(ctx, maps, a, b, tuple(cfg["budgets"]), cc["sample_px"],
                                                 cfg["uncertainty"]["seed"]), "model_complementarity")
            bands = GE.pick_bands(ctx.data.band_names, cc.get("contrast_bands") or None)
            seg = GE.segment_attributes(ctx, contrast_bands=bands)
            lpt = label_pixel_table(ctx)
            cls = GE.recovery_classes(seg, support, lpt, a, b, cc["recovered_k"], extra=(ens,) if ens in support else ())
            for col in [c for c in cls.columns if c.startswith("contrast_")] + ["local_density", "strike_share_other_folds"]:
                if col in cls and cls[col].notna().sum() >= 6:
                    cls[f"{col}_tercile"] = pd.qcut(cls[col].rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
            cls["edge_bin"] = pd.cut(cls.edge_km, [-np.inf, 2, 5, np.inf], labels=["<2 km", "2-5 km", ">5 km"]).astype(str)
            by = ["length_bin", "short_fault", "fold", "strike_sector", "sinuosity_bin", "junction", "edge_bin"] + \
                [c for c in cls.columns if c.endswith("_tercile")]
            summ = GE.class_summary(cls, by)
            self.ws.save_table(cls, "fault_system_analysis")
            self.ws.save_table(summ, "fault_system_summary")
            self.ws.save_table(GE.miss_attribution(cls), "error_attribution")
            masks = {m: Q.budget_masks(ctx, maps[m], cfg["match_frac"]) for m in (a, b)}
            self.ws.save_table(Q.fp_distance_profile(ctx, masks, a, b), "fp_distance_profile")
            objs = (self.sc.get("disc_objs") or {})
            if not any(k[1] in (a, b) for k in objs):
                _, objs = discovery3.discovery_curves(ctx, [a, b], policies=(cfg["primary_policy"],), variants=("model",))
            dcls = Q.discovery_classes(ctx, objs, a, b, cfg["match_frac"], cfg["primary_policy"], cc["recovered_k"])
            self.ws.save_table(dcls, "discovery_system_classes")
            cs = cc["case_studies"]
            win = Q.case_windows(ctx, cls, a, b, cfg["match_frac"], masks=masks, sys_cls=dcls,
                                 n_per_kind=cs["n_per_kind"], half_px=cs["half_px"], ens=ens)
            self.ws.save_table(win, "case_study_windows")
            fig = fig_disagreement(ctx, maps, a, b, cfg["match_frac"])
            ps = self.ws.load_table("model_complementarity")
            self.ws.save_fig(fig, "model_disagreement", ps[np.isclose(ps.budget, cfg["match_frac"])] if len(ps) else ps)
            rec = sc_all["recovery"]
            fig, data = Q.fig_short_fault_analysis(rec, summ, [m for m in (a, b, ens) if m and m in set(rec.model)])
            self.ws.save_fig(fig, "short_fault_analysis", data)
            geo_band = cs.get("geophysics_band") or (bands[0] if bands else None)
            disc_maps = {(ens, d): v for d, v in (ens_disc or {}).items()} if ens else {}
            for w in win.to_dict("records"):
                fig = Q.fig_case_study(ctx, w, maps, disc_maps=disc_maps, context_band=cs.get("context_band"),
                                       geophysics_band=geo_band)
                self.ws.save_fig(fig, f"{w['kind']}_{w['source']}_{w['id']}", pd.DataFrame([w]),
                                 sub="figures/geological_case_studies")
            return True

        self._unit("complementarity", fp, run)
        return {k: self.ws.load_table(k) for k in ("model_complementarity", "fault_system_summary",
                                                   "discovery_system_classes", "error_attribution", "case_study_windows")}

    # =========================================================================== C. postprocessing
    def postprocessing(self) -> dict:
        from ..p3.transforms import FoldScorers
        from ..phase4_eval import recovery_table
        from ..postprocess import topology as T

        cfg, ctx = self.cfg, self.ctx
        pc_ = cfg["postprocessing"]
        fp = fingerprint(M6_VERSION, "postprocessing", self.fp["scorecard"], self.fp.get("ensembles"), pc_)

        def run():
            models = {m: ctx.oof(m) for m in self.pair}
            if pc_["models"] == "primary_pair+ensemble":
                ens, oof, _ = self.primary_ensemble()
                if ens:
                    models[ens] = oof
            scorers = FoldScorers(ctx.data.labels, ctx.folds.fold_map, ctx.data.valid)
            far = ctx.dist_to_labels_km() > 1.0
            pkm = ctx.data.grid.pixel_km
            rows, tables, sels = [], [], []
            for mname, p in models.items():
                ab = T.ablation(ctx, mname, p, fracs=tuple(pc_["fracs"]), grid=pc_.get("grid"), scorers=scorers)
                tables.append(ab["table"])
                sels.append(ab["selection"])
                summ = ab["summary"].set_index("operation")
                masks = {op: T.stitched_mask(ctx, p, ch) for op, ch in ab["choices"].items()}
                metr = self._mask_metrics(ctx, mname, p, masks, far, pkm, recovery_table)
                for op in masks:
                    rows.append({"model": mname, "operation": op, **summ.loc[op].to_dict(), **metr[op]})
                base = rows[-len(masks):]
                ident = next(r for r in base if r["operation"] == "identity")
                good = [r["operation"] for r in base if r["operation"] != "identity" and r["delta_vs_identity"] > 0
                        and (r["short_recall_w"] - ident["short_recall_w"]) >= -pc_["max_short_recall_drop"]]
                if len(good) >= 2:
                    ch = T.combine_choices(ab["choices"], good)
                    mk = T.stitched_mask(ctx, p, ch)
                    comp = scorers.table(mk.astype(np.float32))
                    from ..p3.curves import dti_from

                    dt = pd.DataFrame(comp)
                    m2 = self._mask_metrics(ctx, mname, p, {"combined": mk}, far, pkm, recovery_table)["combined"]
                    rows.append({"model": mname, "operation": "combined:" + "+".join(good), "dti_crossfit": dti_from(dt),
                                 "fold_min": float(dt.groupby("zone").apply(dti_from).min()),
                                 "delta_vs_identity": dti_from(dt) - ident["dti_crossfit"],
                                 "n_folds_applied": len(ch), **m2})
                    masks["combined"] = mk
                if cfg["save_rasters"] != "none":
                    best = max((r for r in rows if r["model"] == mname), key=lambda r: r["dti_crossfit"])
                    key = "combined" if best["operation"].startswith("combined") else best["operation"]
                    self.ws.save_raster({"mask": masks[key].astype(np.float32)},
                                        f"processed/postprocessed_predictions/{mname}_{key}_oof.tif", ctx.data.grid)
            res = pd.DataFrame(rows)
            for c in ("short_recall_w", "skeleton_km", "n_components", "fragmentation_per_100km", "n_junctions",
                      "orientation_coherence_area_w", "area_px", "far_share"):
                if c in res:
                    res[f"delta_{c}"] = res[c] - res.groupby("model")[c].transform(
                        lambda s: s[res.loc[s.index, "operation"] == "identity"].iloc[0])
            self.ws.save_table(res, "postprocessing_ablation")
            self.ws.save_table(pd.concat(tables, ignore_index=True), "postprocessing_components", sub="processed/postprocessing")
            self.ws.save_table(pd.concat(sels, ignore_index=True), "postprocessing_selection_audit")
            self.ws.save_fig(self._fig_postprocessing(res), "postprocessing_ablation", res)
            return True

        self._unit("postprocessing", fp, run)
        return {"ablation": self.ws.load_table("postprocessing_ablation")}

    @staticmethod
    def _mask_metrics(ctx, mname, p, masks, far, pkm, recovery_table) -> dict:
        from ..postprocess.topology import topology_metrics

        out = {}
        for op, mk in masks.items():
            rec, _ = recovery_table(ctx, {f"{mname}|{op}": p}, 0.0, strata=("short_fault",),
                                    masks={f"{mname}|{op}": mk})
            sr = rec[(rec.stratum == "short_fault") & (rec.value == "<2 km")]
            out[op] = {"short_recall_w": float(sr.weighted_recall.iloc[0]) if len(sr) else np.nan,
                       "far_share": float((mk & far).sum() / max(mk.sum(), 1)), **topology_metrics(mk, pkm)}
        return out

    @staticmethod
    def _fig_postprocessing(res: pd.DataFrame):
        import matplotlib.pyplot as plt

        from ..evaluation.comparative import _colors, _style

        col = _colors(list(dict.fromkeys(res.model)))
        fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
        for r in res[res.operation != "identity"].itertuples():
            ax.scatter(r.delta_short_recall_w, r.delta_vs_identity, color=col[r.model], s=34)
            ax.annotate(r.operation, (r.delta_short_recall_w, r.delta_vs_identity), fontsize=6, xytext=(3, 2),
                        textcoords="offset points")
        ax.axhline(0, color="#8a8985", lw=0.8, ls="--")
        ax.axvline(0, color="#8a8985", lw=0.8, ls="--")
        _style(ax, "change in < 2 km weighted recall", "change in nested geographic DTI",
               "Postprocessing: each operation vs the unprocessed mask")
        return fig

    # =========================================================================== D. gate + report
    def gate(self) -> dict:
        from ..evaluation.selection_gate import gate

        cfg = self.cfg
        sc_all = self._sc_all or merge_scorecards(self.sc, load_scorecard(self.ws, "processed/scorecard_ensembles"))
        cands = [m for m in self.analysed if m != self.ref]
        et = self.ws.load_table("ensemble_candidates")
        ens = [c for c in (et.candidate if len(et) else []) if c.startswith("M6_ENS_")]
        cands += ens
        rules = self._protocol_rules()
        rules.update(cfg["gate"].get("rules") or {})
        inv = self.inv.set_index("model")
        audit = self.ws.load_table("ensemble_index_audit")
        dsel = self.ws.load_table("ensemble_selection_discovery")
        boot = pd.concat([self.ws.load_table("spatial_bootstrap"), self.ws.load_table("ensemble_bootstrap")],
                         ignore_index=True)
        evidence = {}
        for c in cands:
            mem = et.set_index("candidate").loc[c, "members"].split("+") if c in ens else [c]
            full = all(inv.loc[m].get("profile") == "FULL" for m in mem if m in inv.index)
            comp = all(m in inv.index and inv.loc[m].status == "COMPLETE" for m in mem)
            leak_ok = True
            if c in ens:
                a = audit[audit.candidate == c] if len(audit) else audit
                d = dsel[dsel.candidate == c] if len(dsel) else dsel
                leak_ok = not (len(a) and a.heldout_labels_used_for_selection.astype(bool).any()) and \
                    not (len(d) and d.uses_split_labels.astype(bool).any())
            bp = {}
            if len(boot):
                g = boot[(boot.model == c) & (boot.task == "geographic")]
                bp = dict(zip(g.reference, g.p_delta_positive))
            cost = float(et.set_index("candidate").loc[c, "relative_inference_cost"]) if c in ens and \
                "relative_inference_cost" in et else 1.0
            evidence[c] = {"provenance_ok": comp, "members_full": full, "leakage_ok": leak_ok, "boot_p": bp,
                           "cost": cost, "provenance_note": ", ".join(f"{m}={inv.loc[m].status}" for m in mem
                                                                     if m in inv.index)}
        fp = fingerprint(M6_VERSION, "gate", self.fp.get("scorecard"), self.fp.get("ensembles"), rules, cfg["gate"],
                         evidence)

        def run():
            table, rec = gate(sc_all, cands, self.controls, rules=rules, gate_cfg=cfg["gate"], evidence=evidence,
                              primary_policy=cfg["primary_policy"])
            self.ws.save_table(table, "promotion_decisions")
            self.ws.save_json({**rec, "candidates": cands, "controls": self.controls, "rules": rules,
                               "gate": cfg["gate"]}, "manifests/recommendation.json")
            return True

        self._unit("gate", fp, run)
        return {"decisions": self.ws.load_table("promotion_decisions"),
                "recommendation": _read_json(self.ws.root / "manifests" / "recommendation.json")}

    def _protocol_rules(self) -> dict:
        """Section-17 guardrails of the repository (configs/validation/phase5_protocol.yaml)."""
        import yaml

        from ..paths import config_path

        try:
            y = yaml.safe_load(config_path("validation", "phase5_protocol.yaml").read_text(encoding="utf-8"))
            return dict((y.get("evaluation") or {}).get("promotion") or {})
        except Exception:
            return {}

    def report(self) -> Path:
        from .m6_report import build_report

        self.ws.begin("report")
        self.ws.save_table(pd.DataFrame(self.log), "unit_log", sub="manifests")
        p = self.ws.save_text(build_report(self), "M6_RESEARCH_REPORT.md")
        self.ws.finish("report", fingerprint(time.time()))
        self._push()
        return p

    # =========================================================================== run
    def run(self) -> dict:
        cfg = self.cfg
        if not cfg.get("enabled", True):
            self._say("disabled (M6_CONFIG['enabled'] = False)")
            return {}
        t0 = time.perf_counter()
        out = {"audit": self.audit()}
        self.scorecard()
        out["paired"] = self.paired()
        if cfg.get("run_calibration"):
            out["calibration"] = self.calibration()
        if cfg.get("run_ensemble"):
            out["ensembles"] = self.ensembles()
        if cfg.get("run_complementarity"):
            out["complementarity"] = self.complementarity()
        if cfg.get("run_postprocessing"):
            out["postprocessing"] = self.postprocessing()
        if cfg.get("run_promotion_audit"):
            if not cfg.get("run_ensemble") and not (self.ws.root / "processed/scorecard_ensembles/score.csv").exists():
                self._say("gate without ensembles: only single FULL models are assessed")
                self.ws.begin("ensembles_stub")
                save_scorecard(self.ws, {"score": pd.DataFrame(columns=["model"])}, "processed/scorecard_ensembles")
                self.ws.finish("ensembles_stub", "stub")
            out["gate"] = self.gate()
        out["report"] = self.report()
        self._say(f"M6 finished in {time.perf_counter() - t0:.0f}s -> {self.ws.root}")
        return out

    def summary(self) -> pd.DataFrame:
        """Compact end-of-notebook table: one row per assessed model / candidate."""
        mc = self.ws.load_table("model_comparison")
        ec = self.ws.load_table("ensemble_candidates")
        pdx = self.ws.load_table("promotion_decisions")
        cols = ["model", "geo_dti_area", "geo_fold_min", "disc_dti_excl", "disc_dti_retained_fp", "short_recall_w"]
        rows = mc[[c for c in cols if c in mc]].rename(columns={"geo_dti_area": "geo_dti"}) if len(mc) else pd.DataFrame()
        if len(ec):
            e = ec[ec.candidate.str.startswith("M6_ENS_")].rename(columns={"candidate": "model"})
            rows = pd.concat([rows, e[[c for c in ["model", "geo_dti", "geo_fold_min", "disc_dti_excl",
                                                   "disc_dti_retained_fp", "short_recall_w"] if c in e]]],
                             ignore_index=True)
        if len(pdx):
            v = pdx[pdx.control == "ALL"].set_index("candidate").verdict
            rows["verdict"] = rows.model.map(v).fillna("control" if len(rows) else "")
        return rows
