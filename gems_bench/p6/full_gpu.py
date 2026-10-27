"""P6 full GPU run: one Colab session from the saved P6 artifacts to a validated submission (docs/p6/P6_FULL_GPU.md).

    fg = p6.full_gpu(FULL_GPU_CONFIG, RECOVERY_CONFIG, BOOST_CONFIG)
    fg.startup()           # CUDA detection (GPURequired in mode 'full' without a GPU), seeds, dependency record
    fg.stage_inputs()      # Drive -> local: stack/, boost/, recovery/, seeds/ and earlier outputs/p6_boost
    fg.audit()             # artifact classes REUSE / RECOMPUTE / INCOMPATIBLE / UNKNOWN            (audit/)
    fg.gpu()               # W1 extra CNN seeds + rot180 TTA on the W1 GPU cap; fold-average maps opt-in   (gpu/)
    fg.boost_variants(...) # optional boost B / C / D (+ the explicit G02 zone backfill); reload when finished
    fg.benchmark()         # candidate / zone metrics, rankings, ablations, manifest, model selection    (benchmark/)
    fg.submission()        # zone-stitched GeoTIFF(s) + validation gate, never uploaded                (submissions/)
    fg.figures()           # PNG + SVG + plot data                                                     (figures/)
    fg.report()            # reports/full_gpu_report.md + experiment_status.csv

Composition: the P6 workspace, boost (``bst``) and recovery (``rc``) objects are read-only sources; everything new
lands under ``<work root>/<output_dir>`` (default outputs/p6_boost) and is pushed to Drive after every stage. A stage
reloads only when its fingerprint matches and every required output exists; a missing output is never COMPLETE.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import boost as B
from . import gpu_runtime as GR

FULL_VERSION = "p6g.1"                       # enters every stage fingerprint
FULL_CODE_REV = "p6g.1-r1"                   # checked by the notebook start-up cell (not fingerprinted)
DEFAULT_FULL_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "p6" / "full_gpu.yaml"
MODES = ("full", "recovery")
OUTPUTS = {
    "startup": ("runtime.json",),
    "audit": ("artifact_audit.csv", "decision.json"),
    "gpu": ("gpu_log.csv", "decision.json"),
    "benchmark": ("candidate_metrics.csv", "zone_metrics.csv", "candidate_rankings.csv", "candidate_ablations.csv",
                  "experiment_manifest.json", "model_selection.json", "protocol_consistency.csv"),
    "submissions": ("official_spec.json", "submission_manifest.json", "submission_validation.json",
                    "submission_preview.png", "submission_diagnostics.csv", "submission_sha256.txt",
                    "submission_index.csv"),
    "reports": ("full_gpu_report.md", "experiment_status.csv"),
}
STAGES = ("startup", "audit", "gpu", "benchmark", "submissions", "figures", "reports")


def load_full_config(config: dict | None = None, path: Path | None = None) -> dict:
    """YAML + overrides, then ``modes.<mode>`` merged over the result (explicit overrides win last)."""
    from .run import _merge

    with open(path or DEFAULT_FULL_CONFIG, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    cfg = _merge(base, config or {})
    if cfg["mode"] not in MODES:
        raise ValueError(f"mode {cfg['mode']!r}; known {MODES}")
    cfg = _merge(_merge(cfg, cfg["modes"][cfg["mode"]]), {k: v for k, v in (config or {}).items() if k != "mode"})
    sel = cfg["selection"]
    if sel["objective"] not in ("disc", "disc_geo_floor", "joint"):
        raise ValueError(f"selection objective {sel['objective']!r} is not a single deployable policy")
    if sel["geo_rule"]["mode"] not in ("relative", "gate", "report"):
        raise ValueError(f"geo_rule.mode {sel['geo_rule']['mode']!r}")
    if cfg["submission"]["construction"] != "zone_stitched":
        raise ValueError("only the validated zone_stitched construction can produce a submission")
    return cfg


def missing_outputs(stage_dir: Path, stage: str) -> list[str]:
    d = Path(stage_dir)
    return [n for n in OUTPUTS.get(stage, ()) if not (d / n).exists() or (d / n).stat().st_size == 0]


def classify_artifact(exists: bool, compatible: bool | None, needed_by: str, recompute_ok: bool) -> str:
    """REUSE (present + compatible), INCOMPATIBLE (present, a recorded check failed), RECOMPUTE (absent and a stage of
    this run can rebuild it), UNKNOWN (absent / unchecked and not rebuildable here). Pure."""
    if exists and compatible is True:
        return "REUSE"
    if exists and compatible is False:
        return "INCOMPATIBLE"
    if not exists and recompute_ok:
        return "RECOMPUTE"
    return "UNKNOWN"


class FullGPU:
    def __init__(self, rc, config: dict | None = None, *, config_path: Path | None = None):
        self.rc, self.bst, self.p6 = rc, rc.bst, rc.p6
        self.cfg = load_full_config(config, config_path)
        self.root = Path(self.p6.m9.local) / self.cfg["output_dir"]
        self.device: str | None = None
        self.runtime: dict = {}

    # ------------------------------------------------------------------------------- plumbing
    @property
    def prim(self) -> str:
        return self.p6.prim

    def say(self, msg: str) -> None:
        print(f"[p6-full] {msg}")

    def fp(self, *parts) -> str:
        from ..pipeline.m6 import fingerprint

        return fingerprint(FULL_VERSION, *parts)

    def save_table(self, df: pd.DataFrame, name: str, stage: str) -> Path:
        from ..artifacts.atomic import atomic_save

        p = self.root / stage / f"{name}.csv"
        d = df if df is not None else pd.DataFrame()
        atomic_save(p, lambda t: d.to_csv(t, index=False))
        return p

    def save_parquet(self, df: pd.DataFrame, name: str, stage: str) -> Path:
        from ..artifacts.atomic import atomic_save
        from ..pipeline.m6 import _parquet_safe

        p = self.root / stage / f"{name}.parquet"
        d = _parquet_safe(df if df is not None and len(df) else pd.DataFrame({"_empty": []}))
        atomic_save(p, lambda t: d.to_parquet(t, index=False))
        return p

    def save_json(self, obj, stage: str, name: str) -> Path:
        from ..artifacts.atomic import atomic_write_json

        return atomic_write_json(self.root / stage / f"{name}.json", json.loads(json.dumps(obj, default=B._js)))

    def read_json(self, stage: str, name: str) -> dict | None:
        p = self.root / stage / f"{name}.json"
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except (OSError, json.JSONDecodeError) as e:
            self.say(f"WARNING {p} unreadable ({type(e).__name__}): treated as absent")
            return None

    def load_table(self, name: str, stage: str) -> pd.DataFrame:
        p = self.root / stage / f"{name}.csv"
        if not p.exists() or not p.stat().st_size:
            return pd.DataFrame()
        try:
            return pd.read_csv(p, dtype={"systems": str})
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def save_fig(self, fig, name: str, data, stage: str) -> Path:
        import matplotlib.pyplot as plt

        from ..artifacts.atomic import atomic_save

        fc = self.cfg["figures"]
        p = self.root / stage / f"{name}.png"
        atomic_save(p, lambda t: fig.savefig(t, dpi=int(fc["dpi"]), bbox_inches="tight", format="png"))
        if fc.get("svg"):
            atomic_save(p.with_suffix(".svg"), lambda t: fig.savefig(t, bbox_inches="tight", format="svg"))
        if data is not None:
            d = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            atomic_save(p.with_name(f"{name}_data.csv"), lambda t: d.to_csv(t, index=False))
        plt.close(fig)
        return p

    def save_raster(self, layers: dict, rel: str, grid) -> Path:
        from gems_eda.io import write_geotiff

        from ..artifacts.atomic import atomic_save

        return atomic_save(self.root / rel, lambda t: write_geotiff(
            t, {k: np.asarray(v, np.float32) for k, v in layers.items()}, grid))

    def copy_canonical_preview(self, stem: str) -> None:
        d = self.root / "submissions"
        for a, b in ((f"{stem}_preview.png", "submission_preview.png"),
                     (f"{stem}_preview_data.csv", "submission_preview_data.csv")):
            if (d / a).exists():
                shutil.copy2(d / a, d / b)

    def push(self) -> None:
        pc = self.p6.m9.pc
        if pc is None or not self.p6.cfg.get("push_to_drive", True):
            return
        try:
            pc.push([Path(self.root).resolve().relative_to(Path(self.p6.m9.local).resolve()).as_posix()])
        except Exception as e:
            self.say(f"push skipped: {type(e).__name__}: {e}")

    def data_fingerprint(self) -> str:
        """Cheap fingerprint of the competition inputs: grid, template, label / footprint counts, raw-file size+mtime."""
        from .boost_audit import file_hash

        d = self.p6.ctx.data
        t = d.template_profile
        raw = {}
        for k in ("features", "labels", "submission"):
            try:
                raw[k] = file_hash(Path(d.paths.raw(k)), 0)
            except Exception:                                   # unknown raw key: recorded, never fatal
                raw[k] = None
        return self.fp(str(t["crs"]), str(t["transform"]), int(t["height"]), int(t["width"]),
                       int(np.asarray(d.labels, bool).sum()), int(np.asarray(d.valid, bool).sum()), d.band_names, raw)

    # ------------------------------------------------------------------------------- stage machinery
    def set_status(self, stage: str, status: str, **info) -> None:
        if status not in B.STATUSES:
            raise ValueError(f"status {status!r} not in {B.STATUSES}")
        self.save_json({"stage": stage, "status": status, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "code_rev": FULL_CODE_REV, **info}, stage, "status")

    def done(self, stage: str, fp: str) -> bool:
        m = self.read_json(stage, "manifest") or {}
        if m.get("fingerprint") != fp or m.get("status") != "done":
            return False
        miss = missing_outputs(self.root / stage, stage)
        if miss:
            self.say(f"{stage}: fingerprint matches but {miss} missing -> recomputed")
            return False
        self.say(f"{stage}: fingerprint {fp[:10]} unchanged -> reloaded (nothing recomputed)")
        return True

    def begin(self, stage: str, fp: str) -> float:
        self.set_status(stage, "RUNNING", fingerprint=fp)
        GR.reset_gpu_peak()
        return time.perf_counter()

    def finish(self, stage: str, fp: str, t0: float, inputs: dict, decisions: dict, *, status: str = "COMPLETE",
               reason: str = "") -> None:
        from .._version import provenance

        miss = missing_outputs(self.root / stage, stage)
        if miss and status == "COMPLETE":
            status, reason = "FAILED", f"required outputs missing: {miss}"
        self.save_json({"stage": stage, "status": "done" if status == "COMPLETE" else status.lower(),
                        "stage_status": status, "fingerprint": fp, "full_version": FULL_VERSION,
                        "code_rev": FULL_CODE_REV, "mode": self.cfg["mode"], "config": self.cfg, "inputs": inputs,
                        "decisions": decisions, "code": provenance(), "device": self.device,
                        "memory": GR.memory_report(), "seconds": round(time.perf_counter() - t0, 1),
                        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "outputs": sorted(p.name for p in (self.root / stage).glob("*") if p.is_file())},
                       stage, "manifest")
        self.set_status(stage, status, fingerprint=fp, reason=reason, source_commit=provenance()["git_commit"])
        self.push()

    def section(self, stage: str, device: str, fn):
        """Budgeted step (CPU: ``cpu_workstream``; GPU: ``gpu.workstream``); failures leave status FAILED."""
        from .run import BudgetExceeded

        ws = self.cfg["gpu"]["workstream"] if device == "gpu" else self.cfg["cpu_workstream"]
        try:
            return self.p6.section(ws, device, fn, what=f"full_{stage}")
        except BudgetExceeded as e:
            self.set_status(stage, "BLOCKED", reason=f"BudgetExceeded: {e}")
            raise
        except Exception as e:
            self.set_status(stage, "FAILED", reason=f"{type(e).__name__}: {e}")
            raise

    def load(self, stage: str) -> dict:
        d = self.root / stage
        out = {"status": self.read_json(stage, "status")}
        for p in sorted(d.glob("*.json")) if d.exists() else []:
            if p.stem not in ("status", "manifest"):
                out[p.stem] = self.read_json(stage, p.stem)
        for p in sorted(d.glob("*.csv")) if d.exists() else []:
            if not p.name.endswith("_data.csv"):
                out[p.stem] = self.load_table(p.stem, stage)
        return out

    # ------------------------------------------------------------------------------- 0 start-up
    def startup(self) -> dict:
        """Device detection, the GPU requirement of the mode, seeds, dependency record (startup/runtime.json)."""
        info = GR.detect()
        self.device = GR.require_gpu(info, self.cfg["gpu"])     # raises GPURequired in mode full without CUDA
        seeds = GR.seed_everything(int(self.cfg["seed"]), bool(self.cfg["deterministic"]))
        self.runtime = {"mode": self.cfg["mode"], "gpu": info, "device": self.device,
                        "gpu_stages": GR.gpu_stages(self.cfg["gpu"], self.device), "seeds": seeds,
                        "dependencies": GR.dependency_record(), "memory": GR.memory_report(),
                        "precision": self.cfg["gpu"]["amp"], "code_rev": FULL_CODE_REV}
        self.save_json(self.runtime, "startup", "runtime")
        self.say(f"mode {self.cfg['mode']}: device {self.device or 'none (CPU stages only)'}; GPU stages "
                 f"{self.runtime['gpu_stages']}")
        return self.runtime

    def stage_inputs(self) -> dict:
        """Frozen P6 inputs from Drive (read-only) + earlier outputs of this run (resume after a disconnect)."""
        need = self.bst.stage()
        out = self.p6.cfg["output_dir"]
        self.p6.m9.stage_in([f"{out}/{s}" for s in (self.rc.cfg["subdir"], "seeds")] + [self.cfg["output_dir"]])
        return need

    # ------------------------------------------------------------------------------- 1 audit
    def audit(self) -> dict:
        """Runtime classification of every artifact the run depends on (``docs/p6/P6_RECOVERY_TO_BOOST_AUDIT.md``
        is the authoring-time version)."""
        from .boost_audit import file_hash

        stage = "audit"
        t0 = self.begin(stage, "audit")
        root, bst, rc = self.p6.ws.root, self.bst, self.rc
        rows = []

        def add(name, rel, compatible, needed_by, recompute_ok, note=""):
            p = root / rel
            ex = p.exists()
            rows.append({"artifact": name, "path": rel, "exists": ex, "sha256_or_size": file_hash(p, 1024) if ex and
                         p.is_file() else None, "class": classify_artifact(ex, compatible if ex else None, needed_by,
                                                                            recompute_ok),
                         "needed_by": needed_by, "note": note})

        cand = json.loads((root / "stack" / "candidate.json").read_text(encoding="utf-8")) \
            if (root / "stack" / "candidate.json").exists() else {}
        w2_ok = cand.get("gate") == "PROMOTE"
        add("W2 candidate", "stack/candidate.json", w2_ok, "stack candidates", False, f"gate {cand.get('gate')}")
        add("W2 training rows", "stack/features/rows.parquet", w2_ok, "W2 provenance", False)
        for u in bst.units():
            add(f"W2 held-out map {u.key}", f"stack/maps/{u.task}_{u.zone}.npy", w2_ok, "zone-stitched W2 / benchmark",
                False)
        par = rc.load_table("train_inference_parity", "export_parity")
        blocked = bool(len(par)) and bool(((par.status == "BLOCKED") & (par.feature == "w2_score")).any())
        add("W2 full-region score (footprint-ranked refit)", "stack/maps/full_score.tif", (not blocked) if len(par)
            else None, "historical export only", False, "recovery parity: w2_score BLOCKED on geo units" if blocked
            else "parity not run")
        bm = bst.read_json("benchmark", "manifest") or {}
        pk_ok = (bm.get("inputs") or {}).get("protocol_keys") in (None, bst.protocol_keys())
        add("boost A grid", "boost/benchmark/grid.csv", bm.get("status") == "done" and pk_ok, "benchmark", False,
            "protocol keys unchanged" if pk_ok else "protocol keys changed since boost A")
        for st in ("geology", "robustness", "hybrid"):
            m = bst.read_json(st, "manifest") or {}
            add(f"boost {st} grid", f"boost/{st}/grid.csv", m.get("status") == "done", "benchmark", True,
                "opt-in fg.boost_variants()")
        for st in ("export_parity", "candidate_recall", "segments", "geology", "robustness", "selection"):
            s = rc.read_json(st, "status") or {}
            add(f"recovery {st}", f"{rc.cfg['subdir']}/{st}/status.json", s.get("status") == "COMPLETE",
                "audit / benchmark", False, str(s.get("status")))
        add("W1 seed grid", "seeds/grid.csv", True, "benchmark (seed ensembles)", True, "fg.gpu() in mode full")
        add("historical W2 submission (mixed rep / frac)", "submissions/p6_stack_fusion.tif", False,
            "comparison only", False, "rep from the disc choice, frac from the geo choice: never re-exported")
        geo = rc.read_json("geology", "decision") or {}
        rows.append({"artifact": "G02 unit rasters", "path": "G02 run " + str(bst.cfg["geology"]["g02_run_id"]),
                     "exists": bool(geo), "sha256_or_size": None,
                     "class": "RECOMPUTE" if geo.get("repair_status") == "READY_FOR_BACKFILL" else
                     ("REUSE" if geo.get("repair_status") == "NOTHING_TO_DO" else "UNKNOWN"),
                     "needed_by": "boost B (geological prior)", "note": json.dumps(geo.get("blockers"), default=str)})
        for m in list(self.p6.m9.models)[:2]:
            try:
                have = self.p6.ctx.oof(m) is not None and self.p6.ctx.has_discovery(m)
            except (FileNotFoundError, KeyError):
                have = False
            rows.append({"artifact": f"CNN honest maps {m}", "path": "M6 inventory (oof + disc0-2)", "exists": have,
                         "sha256_or_size": None, "class": "REUSE" if have else "UNKNOWN",
                         "needed_by": "every candidate", "note": "OOF on geo zones, discovery maps on disc units"})
        A = pd.DataFrame(rows)
        self.save_table(A, "artifact_audit", stage)
        dec = {"counts": A["class"].value_counts().to_dict(), "blocking": A[(A["class"] != "REUSE") &
                                                                            A.needed_by.isin(["every candidate"])]
               .artifact.tolist(), "w2_promoted": w2_ok, "full_score_parity_blocked": blocked}
        self.save_json(dec, stage, "decision")
        self.finish(stage, "audit", t0, {}, dec)
        return {"table": A, "decision": dec}

    # ------------------------------------------------------------------------------- 2 GPU
    def gpu(self) -> dict:
        """W1 on the GPU cap: extra seeds (resumable experiments), rot180 TTA, seed-ensemble grid; optional
        fold-average full maps (diagnostic). Skipped (recorded) in mode recovery or without a device."""
        stage = "gpu"
        on = GR.gpu_stages(self.cfg["gpu"], self.device)
        t0 = self.begin(stage, f"gpu:{json.dumps(on)}")
        log = []
        if not any(on.values()):
            log.append({"step": "gpu", "status": "SKIPPED", "reason": f"mode {self.cfg['mode']} / device "
                                                                       f"{self.device}"})
        else:
            self.p6.cfg["seeds"]["tta_rot180"] = bool(on["tta"])        # recorded in gpu/decision.json
            from . import seeds as SE

            # W1 components, not p6.w1_seeds(): that would also export its own full-region file into p6_fast_track
            if on["train_seeds"]:
                log.append(self._timed("train_seeds", lambda: self.p6.section(
                    "w1", "gpu", lambda: SE.train_missing(self.p6), what="train")))
            if on["tta"]:
                log.append(self._timed("tta", lambda: self.p6.section("w1", "gpu", lambda: SE.tta_all(self.p6),
                                                                      what="tta")))
            if on["train_seeds"] or on["tta"]:
                log.append(self._timed("evaluate", lambda: self.p6.section("w1", "cpu", lambda: SE.evaluate(self.p6),
                                                                           what="evaluate")))
            if on["fold_average_maps"]:
                log.append(self._timed("fold_average_maps", lambda: self.p6.section("w1", "gpu", self._fold_average,
                                                                                    what="fold_average")))
        L = pd.DataFrame(log)
        self.save_table(L, "gpu_log", stage)
        cand = self.p6.ws.root / "seeds" / "candidate.json"
        dec = {"gpu_stages": on, "device": self.device, "steps": log,
               "seed_candidate": json.loads(cand.read_text(encoding="utf-8")) if cand.exists() else None,
               "tta_rot180": self.p6.cfg["seeds"].get("tta_rot180"),
               "note": "seed ensembles enter the benchmark through seeds/grid.csv (same protocol); fold-average maps "
                       "are diagnostics only"}
        self.save_json(dec, stage, "decision")
        failed = [r["step"] for r in log if r.get("status") == "FAILED"]
        self.finish(stage, f"gpu:{json.dumps(on)}", t0, {}, {"failed": failed},
                    status="FAILED" if failed else "COMPLETE", reason="; ".join(failed))
        return dec

    def _timed(self, step: str, fn) -> dict:
        t0 = time.perf_counter()
        GR.reset_gpu_peak()
        try:
            fn()
            st, why = "COMPLETE", ""
        except Exception as e:                                  # recorded; the stage status becomes FAILED
            st, why = "FAILED", f"{type(e).__name__}: {e}"
            self.say(f"{step}: {why}")
        return {"step": step, "status": st, "reason": why, "seconds": round(time.perf_counter() - t0, 1),
                **{f"mem_{k}": v for k, v in GR.memory_report().items()}}

    def _fold_average(self) -> list[str]:
        """Fold-average B0 / B1 maps over the full footprint (GPU, tiled with overlap windows), cached as GeoTIFF."""
        import torch

        from ..inference.full_raster import cnn5_full_region

        out = []
        for mem in self.p6.m9.m7.g3.members():
            p = self.root / "gpu" / "maps" / f"{mem['model']}__fold_average.tif"
            if not p.exists():
                a = cnn5_full_region(self.p6.ctx, mem["run_dir"], torch.device(self.device))
                if not np.isfinite(a[np.asarray(self.p6.ctx.data.valid, bool)]).all():
                    raise FloatingPointError(f"{mem['model']}: non-finite fold-average values inside the footprint")
                self.save_raster({"fold_average": a}, f"gpu/maps/{p.name}", self.p6.ctx.data.grid)
            out.append(p.name)
        return out

    # ------------------------------------------------------------------------------- 2b boost variants
    def boost_variants(self, backfill: bool = False, geology: bool = False, robustness: bool = False,
                       hybrid: bool = False) -> dict:
        """Opt-in boost stages B / C / D (each reloads when finished; their grids join the benchmark). B needs every
        G02 unit raster: ``backfill=True`` runs the explicit zone backfill first (rc.g02_backfill)."""
        out = {}
        if backfill:
            st = (self.rc.geology_audit() or {}).get("repair_status")
            out["backfill"] = self.rc.g02_backfill() if st == "READY_FOR_BACKFILL" else f"not run: {st}"
        for flag, fn, name in ((geology, self.bst.b_geology, "geology"), (robustness, self.bst.c_robustness,
                                                                          "robustness"),
                               (hybrid, self.bst.d_hybrid, "hybrid")):
            if flag:
                try:
                    fn()
                    out[name] = (self.bst.read_json(name, "status") or {}).get("status")
                except Exception as e:                          # recorded; the benchmark reports the grid absent
                    out[name] = f"FAILED: {type(e).__name__}: {e}"
        return out

    # ------------------------------------------------------------------------------- 3-6
    def benchmark(self) -> dict:
        from . import full_benchmark as FB

        return self.section("benchmark", "cpu", lambda: FB.run(self))

    def submission(self) -> dict:
        from . import final_submission as FS

        return self.section("submissions", "cpu", lambda: FS.run(self))

    def figures(self) -> list[str]:
        from . import full_figures as FF

        t0 = self.begin("figures", "figures")
        done = FF.run(self)
        self.finish("figures", "figures", t0, {}, {"figures": done})
        return done

    def report(self) -> dict:
        """reports/full_gpu_report.md + experiment_status.csv from saved outputs only."""
        from ..artifacts.atomic import atomic_write_text

        stage = "reports"
        t0 = self.begin(stage, "report")
        rows = []
        for st in STAGES:
            s = self.read_json(st, "status") or {}
            rows.append({"stage": st, "status": s.get("status", "PENDING"), "reason": s.get("reason", ""),
                         "updated": s.get("updated"), "source_commit": s.get("source_commit")})
        ST = pd.DataFrame(rows)
        self.save_table(ST, "experiment_status", stage)
        ms = self.read_json("benchmark", "model_selection") or {}
        sm = self.read_json("submissions", "submission_manifest") or {}
        R = self.load_table("candidate_rankings", "benchmark")
        lines = ["# P6 full GPU run: report", "", f"Mode `{self.cfg['mode']}`, device `{self.device}`, code "
                 f"`{FULL_CODE_REV}`. Every number is a nested held-out estimate from saved grids (official "
                 "disc_official / geo components), NOT a leaderboard score. Nothing was uploaded.", "",
                 "## Stages", "", "| stage | status | reason |", "|---|---|---|"]
        lines += [f"| {r.stage} | {r.status} | {str(r.reason)[:160]} |" for r in ST.itertuples()]
        lines += ["", "## Selection", "", f"Objective `{ms.get('objective')}`; decision `{ms.get('decision')}`; "
                  f"winner `{ms.get('winner')}`; baseline `{ms.get('baseline')}`; gain vs baseline "
                  f"{ms.get('gain_vs_baseline')}; slot-worthy {ms.get('slot_worthy')}.", ""]
        if ms.get("meta_nested"):
            m = ms["meta_nested"]
            lines.append(f"META (choosing the candidate, nested): disc {m.get('disc_dti')}, worst {m.get('disc_worst')}"
                         f", chosen per zone {m.get('chosen_per_zone')}.")
        if len(R):
            lines += ["", "| rank | candidate | status | eligible | disc | worst | geo | reasons |",
                      "|---:|---|---|---|---:|---:|---:|---|"]
            for r in R.head(25).to_dict("records"):
                f = (lambda k: f"{float(r[k]):.4f}" if pd.notna(r.get(k)) else "")
                lines.append(f"| {r['rank']} | {r['name']} | {r.get('status')} | {r.get('eligible')} | {f('disc_dti')} "
                             f"| {f('disc_worst')} | {f('geo_dti')} | {str(r.get('eligibility_reasons', ''))[:140]} |")
        lines += ["", "## Submission", "", f"Winner file `{sm.get('winner_file')}`: **{sm.get('winner_status')}**.",
                  "", "Upload manually (docs/p6/P6_FULL_GPU.md section 5), then `lb.record(fg, ...)`."]
        atomic_write_text(self.root / stage / "full_gpu_report.md", "\n".join(lines) + "\n")
        self.finish(stage, "report", t0, {}, {"winner": ms.get("winner"), "winner_status": sm.get("winner_status")})
        return {"status": ST, "markdown": "\n".join(lines)}


def startup_check(expect: dict, head: str, config: dict | None = None) -> dict:
    """Notebook version lock: code revisions, resolved commit, config parse (no data touched)."""
    from ..pipeline.m6 import fingerprint
    from . import boost as Bm
    from .boost_diagnostics import RECOVERY_CODE_REV

    cfg = load_full_config(config)
    probs = []
    for k, have in (("code_rev", Bm.BOOST_CODE_REV), ("recovery_code_rev", RECOVERY_CODE_REV),
                    ("full_code_rev", FULL_CODE_REV)):
        if expect.get(k) and expect[k] != have:
            probs.append(f"{k}: expected {expect[k]}, installed {have}")
    if expect.get("commit") and not str(head).startswith(str(expect["commit"])):
        probs.append(f"commit: expected {expect['commit']}, resolved {head}")
    out = {"resolved_commit": head, "boost_code_rev": Bm.BOOST_CODE_REV, "recovery_code_rev": RECOVERY_CODE_REV,
           "full_code_rev": FULL_CODE_REV, "mode": cfg["mode"], "config_fp": fingerprint(FULL_VERSION, cfg)[:12],
           "problems": probs}
    if probs:
        raise RuntimeError(f"start-up check failed: {probs}")
    return out
