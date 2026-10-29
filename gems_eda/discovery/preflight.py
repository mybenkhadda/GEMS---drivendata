"""Colab preflight of the exhaustive EDA: decides, before any expensive computation, which stages can run, which are
reusable from cache, which must recompute, and which are blocked or optionally unavailable (with the exact reason).

    P = E.preflight()            # == Preflight(E).check(); writes audit/preflight_* and reports/PREFLIGHT_READINESS.md
    display(P.summary()); P.require_ready()

Cost: raster METADATA (+ the 12 MB fault mask, the 49 MB template footprint and one 512 x 512 window of the features
raster), manifests, and the same cached-output validation the stage cache uses (``ExhaustiveEDA.check_outputs`` ->
``artifacts.validate``). Nothing is resampled, trained, inferred or downloaded.

Stage statuses: READY (runs now; forced stages always run), READY_FROM_CACHE (fingerprint matches, every contracted
output validates), NEEDS_RECOMPUTE (will run: no / stale / invalid cache), OPTIONAL_UNAVAILABLE (conditional or
optional prerequisite absent: reported, never concluded), BLOCKED (an upstream stage cannot become valid), INVALID
(its own inputs are broken).
"""
from __future__ import annotations

import importlib
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import EDA_CODE_REV
from . import artifacts as AR

READY, FROM_CACHE, RECOMPUTE = "READY", "READY_FROM_CACHE", "NEEDS_RECOMPUTE"
OPT_UNAVAILABLE, BLOCKED, INVALID = "OPTIONAL_UNAVAILABLE", "BLOCKED", "INVALID"
RUNNABLE = (READY, FROM_CACHE, RECOMPUTE)
FORCED = ("00_tests", "e14_integrity", "e15_synthesis")
RESOURCE = {"00_tests": "light (pytest, ~1 min)", "e01_inventory": "io: reads every band once (~0.4 GB)",
            "e02_landscape": "heavy cpu/ram: 19 bands, 1 / 10 km filters, k-means, ~3 GB peak",
            "e03_geometry": "light-medium: skeleton tracing, 10 figures",
            "e04_profiles": "heavy: ~12 z-scored grids (~0.6 GB) + profile cube (~0.3 GB), chunked metrics",
            "e05_signatures": "cpu: cluster bootstrap over fault systems", "e06_scales": "heavy cpu: Hessian crests "
            "x 5 modalities x scales", "e07_crossmodal": "medium-heavy: skeleton + lineament inventory",
            "e08_label_bias": "light", "e09_regional": "cpu", "01_attach": "drive io (factory staging)",
            "03_scenarios": "drive io heavy (~4 GB layer cache staging)", "e11_forensics": "medium (honest maps)",
            "e12_candidates": "medium (panel rendering)", "e13_dem": "optional", "e14_integrity": "light",
            "e15_synthesis": "light"}
PACKAGES = ("numpy", "pandas", "scipy", "skimage", "sklearn", "rasterio", "matplotlib", "pyarrow", "affine")
OPTIONAL_PACKAGES = ("ipywidgets", "geopandas", "shapely", "tabulate")
GRID_PX = AR.GRID["height"] * AR.GRID["width"]


class PreflightBlocked(RuntimeError):
    pass


@dataclass
class PreflightReport:
    env: pd.DataFrame
    inputs: pd.DataFrame
    cache: pd.DataFrame
    stages: pd.DataFrame
    geometry: pd.DataFrame
    models: pd.DataFrame
    problems: list = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        return self.stages[["stage", "readiness", "can_run_now", "missing_or_invalid", "upstream", "resource",
                            "recovery"]]

    @property
    def ready(self) -> bool:
        return not self.problems

    def require_ready(self) -> None:
        if self.problems:
            raise PreflightBlocked("preflight found blocking problems:\n  - " + "\n  - ".join(self.problems))
        print("preflight: READY (core stages can run; see summary() for cache reuse and optional stages)")


class Preflight:
    def __init__(self, E, deep: bool = False):
        self.E, self.deep = E, deep
        self.root = Path(E.c["drive_root"])

    # =========================================================================== A. environment
    def env(self) -> pd.DataFrame:
        E = self.E
        rows = [{"check": "python", "value": platform.python_version(), "ok": sys.version_info >= (3, 10)},
                {"check": "eda_code_rev", "value": EDA_CODE_REV, "ok": True},
                {"check": "commit", "value": E.head, "ok": bool(E.head)},
                {"check": "configured_ref", "value": E.c.get("ref"),
                 "ok": True, "note": "pinned SHA" if _is_sha(E.c.get("ref")) else "branch (pin a SHA for an "
                                                                                     "immutable run)"},
                {"check": "ref_matches_commit", "value": E.c.get("ref"),
                 "ok": (not _is_sha(E.c.get("ref"))) or str(E.head).startswith(str(E.c.get("ref")))}]
        import gems_eda

        mod = Path(gems_eda.__file__).resolve()
        rows.append({"check": "package_path", "value": str(mod), "ok": True})
        for p in PACKAGES + OPTIONAL_PACKAGES:
            try:
                m = importlib.import_module(p)
                rows.append({"check": f"import {p}", "value": getattr(m, "__version__", "?"), "ok": True})
            except Exception as e:
                rows.append({"check": f"import {p}", "value": f"{type(e).__name__}", "ok": p in OPTIONAL_PACKAGES,
                             "note": "optional" if p in OPTIONAL_PACKAGES else "required"})
        rows.append({"check": "drive_root_mounted", "value": str(self.root), "ok": self.root.exists()})
        try:
            t = E.ws.path("audit/.preflight_write_test")
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text("ok")
            t.unlink()
            rows.append({"check": "local_output_writable", "value": str(E.ws.root), "ok": True})
        except OSError as e:
            rows.append({"check": "local_output_writable", "value": f"{e}", "ok": False})
        for name, p in (("local_disk_free_gb", Path(E.local)), ("drive_free_gb", self.root)):
            try:
                rows.append({"check": name, "value": round(shutil.disk_usage(p).free / 2 ** 30, 1),
                             "ok": shutil.disk_usage(p).free > (8 if name.startswith("local") else 1) * 2 ** 30,
                             "note": "03_scenarios stages ~4 GB locally" if name.startswith("local") else ""})
            except OSError:
                rows.append({"check": name, "value": None, "ok": False})
        mem = _meminfo_gb()
        need = round((19 * GRID_PX * 4 + 14 * GRID_PX * 4) / 2 ** 30 + 2.0, 1)
        rows.append({"check": "ram_total_gb", "value": mem.get("MemTotal"), "ok": (mem.get("MemTotal") or 99) >= need,
                     "note": f"estimated peak ~{need} GB (19-band float32 stack + ~14 full-grid float32 work arrays "
                             "+ 2 GB margin)"})
        rows.append({"check": "gpu", "value": "not required", "ok": True,
                     "note": "no EDA stage trains or infers a network"})
        return pd.DataFrame(rows)

    # =========================================================================== B. raw inputs
    def inputs(self) -> pd.DataFrame:
        from gems_eda.config import RAW_FILES, RAW_SUBDIRS, resolve_raw

        rows, ref = [], None
        for key, spec in AR.RAW.items():
            p = resolve_raw(self.root, key) if key in ("features", "faults", "submission") else self.root / spec["path"]
            r = {"input": key, "path": str(p), "required": spec["required"], "exists": p.exists(),
                 "size_bytes": p.stat().st_size if p.exists() else 0}
            if key in RAW_FILES:                         # duplicate copies in several layouts = ambiguous source
                cands = [self.root / s / RAW_FILES[key] for s in RAW_SUBDIRS.get(key, ("",))]
                cands = [c for c in dict.fromkeys(cands) if c.exists()]
                r["candidate_paths"] = ";".join(str(c) for c in cands)
                if len({c.stat().st_size for c in cands}) > 1:
                    r["ambiguity"] = "copies with different sizes: the loader uses the first (resolve_raw order)"
            if p.suffix.lower() != ".tif":
                r["ok"] = p.exists() and r["size_bytes"] > 0 or not spec["required"]
                r["note"] = "present" if p.exists() else "absent (optional)"
                rows.append(r)
                continue
            if not p.exists() or not r["size_bytes"]:
                rows.append({**r, "ok": False, "note": "missing or empty"})
                continue
            try:
                import rasterio
                from rasterio.windows import Window

                with rasterio.open(p) as s:
                    grid = (s.height, s.width, tuple(round(x, 6) for x in tuple(s.transform)[:6]),
                            s.crs.to_epsg() if s.crs else None)
                    r.update(epsg=grid[3], height=s.height, width=s.width, count=s.count, dtype=s.dtypes[0],
                             nodata=s.nodata, res_x=s.res[0], res_y=s.res[1], bounds=str(tuple(s.bounds)),
                             transform=str(grid[2]))
                    checks = {"epsg": grid[3] == AR.GRID["epsg"], "res": abs(s.res[0] - AR.GRID["res_m"]) < 1e-6
                              and abs(s.res[1] - AR.GRID["res_m"]) < 1e-6,
                              "shape": (s.height, s.width) == (AR.GRID["height"], AR.GRID["width"]),
                              "count": s.count == spec["count"], "dtype": s.dtypes[0] == spec["dtype"]}
                    if key == "features":
                        ref = grid
                        c0, r0 = max(s.width // 2 - 256, 0), max(s.height // 2 - 256, 0)
                        w = s.read(1, window=Window(c0, r0, min(512, s.width - c0), min(512, s.height - r0)),
                                   masked=True).astype("float64").filled(np.nan)      # bounded sample only
                        r["sample_finite_share"] = float(np.isfinite(w).mean())
                        r["band_names"] = ";".join((d or f"band_{i + 1}").split(" - ")[0].strip()
                                                   for i, d in enumerate(s.descriptions))
                    else:
                        checks["same_grid_as_features"] = ref is not None and grid == ref
                    if key == "faults":
                        a = s.read(1)
                        vals = set(np.unique(a).tolist())
                        r["values"] = str(sorted(vals))
                        r["positive_px"] = int((a > 0).sum())
                        checks["labels_binary"] = vals <= {0, 1, int(s.nodata) if s.nodata is not None else 0}
                        checks["positive_px"] = r["positive_px"] == AR.GRID["fault_px"]
                    if key == "submission":
                        a = s.read(1, masked=True)
                        fp = ~np.ma.getmaskarray(a) & np.isfinite(np.ma.filled(a.astype("float64"), np.nan))
                        r["footprint_share"] = float(fp.mean())
                        checks["footprint_nonempty"] = bool(fp.any())
                r.update({f"check_{k}": v for k, v in checks.items()})
                failed = [k for k, v in checks.items() if not v]
                hard = [k for k in failed if k not in ("positive_px",)]          # count drift = warning only
                r["ok"] = not hard
                r["note"] = f"failed checks {failed}" if failed else "all checks pass"
            except Exception as e:
                r.update(ok=False, note=f"unreadable: {type(e).__name__}: {e}")
            rows.append(r)
        return pd.DataFrame(rows)

    # =========================================================================== C. cached products + manifests
    def cache(self) -> pd.DataFrame:
        E = self.E
        rows = []
        for a in AR.registry(type(E)):
            p = E.ws.path(a.rel)
            ok, bad = E.check_outputs([(a.rel, a.required or None)])
            n = None
            if p.exists() and a.kind in (".csv", ".parquet") and not bad:
                try:
                    n = len(pd.read_parquet(p, columns=[a.required[0]]) if a.kind == ".parquet" and a.required
                            else pd.read_csv(p, usecols=[0]) if a.kind == ".csv" and p.stat().st_size > 1 else [])
                except Exception:
                    n = None
            rows.append({"artifact": a.rel, "producer_stage": a.producer, "requirement": a.requirement,
                         "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0, "rows": n,
                         "valid": bool(ok) and not bad, "problems": "; ".join(bad),
                         "typed_empty": n == 0, "consumers": ";".join(a.consumers)})
        C = pd.DataFrame(rows)
        figs = E.ws.load_table("figures/FIGURE_CATALOG.csv")
        for r in (figs.to_dict("records") if len(figs) else []):
            miss = [k for k in ("png", "svg", "data", "interpretation") if isinstance(r.get(k), str)
                    and not E.ws.path(r[k]).exists()]
            if miss:
                C.loc[len(C)] = {"artifact": f"figure {r['figure_id']}", "producer_stage": r.get("stage"),
                                 "requirement": "figure", "exists": False, "size_bytes": 0, "rows": None,
                                 "valid": False, "problems": f"catalogued files missing {miss}", "typed_empty": False,
                                 "consumers": "e15_synthesis"}
        return C

    def expected_fp(self, stage: str):
        E = self.E
        if stage in FORCED or stage in ("01_attach", "03_scenarios"):
            return None
        if stage == "e13_dem" and not E.c["dem"]["enabled"]:
            return E.fp(stage, *E.stage_parts(stage, ["disabled"]))
        if stage == "e11_forensics":                     # keyed on the frozen-engine state (03_scenarios)
            eng = E.fps.get("03_scenarios") or E.manifest("03_scenarios").get("fingerprint")
            return E.fp(stage, *E.stage_parts(stage, [{"engine": eng}]))
        return E.fp(stage, *E.stage_parts(stage, []))

    # =========================================================================== D. geometry precision
    def geometry(self) -> pd.DataFrame:
        from gems_eda.morphology import GEOMETRY_REV

        E = self.E
        p = E.ws.path("tables/fault_geometry.csv")
        if not p.exists():
            return pd.DataFrame([{"check": "fault_geometry.csv", "value": "absent", "ok": True}])
        T = pd.read_csv(p, usecols=lambda c: c in ("seg_id", "strike_deg", "linearity"))
        prov = E.ws.read_json("tables/fault_geometry.provenance.json") or {}
        s, lin = T.get("strike_deg", pd.Series(dtype=float)), T.get("linearity", pd.Series(dtype=float))
        rows = [{"check": "rows", "value": len(T), "ok": True},
                {"check": "strike_eq_180", "value": int((s == 180).sum()), "ok": int((s == 180).sum()) == 0},
                {"check": "strike_outside_0_180", "value": int(((s < 0) | (s > 180)).sum()),
                 "ok": int(((s < 0) | (s > 180)).sum()) == 0},
                {"check": "linearity_gt_1", "value": int((lin > 1).sum()), "ok": int((lin > 1).sum()) == 0},
                {"check": "linearity_gt_1_by_more_than_1e-9", "value": int((lin > 1 + 1e-9).sum()),
                 "ok": int((lin > 1 + 1e-9).sum()) == 0},
                {"check": "linearity_lt_0", "value": int((lin < 0).sum()), "ok": int((lin < 0).sum()) == 0},
                {"check": "min_linearity", "value": float(lin.min()) if len(lin) else None, "ok": True},
                {"check": "geometry_rev_of_cached_table", "value": prov.get("geometry_rev", "absent (pre-geom.2)"),
                 "ok": prov.get("geometry_rev") == GEOMETRY_REV,
                 "note": f"current producer {GEOMETRY_REV}; a different revision means e03 recomputes"}]
        return pd.DataFrame(rows)

    # =========================================================================== E. frozen models / DEM
    def models(self) -> pd.DataFrame:
        E = self.E
        rows = [{"item": "factory (frozen engine)", "status": "CONFIGURED" if E.F is not None else "ABSENT",
                 "note": "B0 / B1 / W2 provenance, zone ids, grid and held-out status are verified by the factory "
                         "at 01_attach (selftest, preflight, attach) - not inferred from directory existence here"}]
        for rel, what in (("validation/scenarios.csv", "03_scenarios validation units"),
                          ("atlas/cache/honest_maps.npz", "zone-stitched held-out maps (e11 cache)")):
            v = AR.validate(E.ws, rel, E.GRID_SHAPE) if E.ws.path(rel).exists() else ["absent"]
            rows.append({"item": what, "status": "VALID_CACHE" if not v else "ABSENT_OR_INVALID",
                         "note": "; ".join(v) or rel})
        for rel in ("outputs/leaderboard_focus/submission_factory", "outputs/m7_discovery_first/dem/features/"
                    "m7_dem_features_100m.tif", "data/raw/dem_1m/inventory", "data/raw/dem_1m/tnm_urls.json"):
            rows.append({"item": rel, "status": "PRESENT_UNVERIFIED" if (self.root / rel).exists() else "ABSENT",
                         "note": "existence only (content verified by its consumer)"})
        rows.append({"item": "dem.enabled", "status": str(bool(E.c["dem"]["enabled"])),
                     "note": "optional; missing DEM coverage is never evidence of absence; no download in preflight"})
        return pd.DataFrame(rows)

    # =========================================================================== matrix
    def stages(self, inputs_ok: bool, env_ok: bool, C: pd.DataFrame) -> pd.DataFrame:
        from gems_bench.geomoe.decide import PRODUCED, SKIPPED_OK

        E = self.E
        by_stage = C.groupby("producer_stage")
        pred, rows = {}, []
        for s in E.STAGES:
            m = E.manifest(s)
            probs = "; ".join(p for p in (by_stage.get_group(s).problems if s in by_stage.groups else []) if p)
            exp = self.expected_fp(s)
            ups = E.DEPENDS.get(s, ())
            up_bad = [u for u in ups if pred.get(u) not in RUNNABLE]
            up_opt = [u for u in up_bad if pred.get(u) == OPT_UNAVAILABLE]
            sw = _switch(E, s)
            if s == "e13_dem" and not E.c["dem"]["enabled"] or sw is False:
                st, why = OPT_UNAVAILABLE, "disabled in CONFIG"
            elif s in ("01_attach", "03_scenarios") and E.F is None:
                st, why = OPT_UNAVAILABLE, "no factory: frozen engine not configured"
            elif not env_ok:
                st, why = INVALID, "environment check failed (see audit/preflight_inputs.csv / env)"
            elif not inputs_ok and s != "00_tests":
                st, why = INVALID if s == "e01_inventory" else BLOCKED, "raw inputs invalid"
            elif up_bad and len(up_opt) == len(up_bad):
                st, why = OPT_UNAVAILABLE, f"needs optional / conditional {up_opt}"
            elif up_bad:
                st, why = BLOCKED, f"upstream cannot become valid: {up_bad}"
            elif s in FORCED:
                st, why = READY, "always runs"
            elif (m.get("status") in (*PRODUCED, SKIPPED_OK) and (exp is None or m.get("fingerprint") == exp)
                  and not probs):
                st, why = FROM_CACHE, f"fingerprint {str(m.get('fingerprint'))[:10]} matches, outputs valid"
            else:
                reason = ("never ran" if not m else f"manifest {m.get('status')}" if m.get("status") not in PRODUCED
                          else "fingerprint changed (code / contract / config)" if exp and m.get("fingerprint") != exp
                          else "cached outputs invalid")
                st, why = RECOMPUTE, reason
            pred[s] = st
            rows.append({"stage": s, "description": E.LABELS.get(s, s), "readiness": st,
                         "can_run_now": st in RUNNABLE and not [u for u in ups if pred.get(u) == RECOMPUTE],
                         "manifest_status": m.get("status", "NEVER_RUN"), "manifest_written": m.get("written"),
                         "fingerprint_cached": m.get("fingerprint"), "fingerprint_expected": exp,
                         "missing_or_invalid": probs, "upstream": ";".join(ups), "reason": why,
                         "resource": RESOURCE.get(s, ""), "recovery": _recovery(st, s, why)})
        return pd.DataFrame(rows)

    # =========================================================================== run
    def check(self) -> PreflightReport:
        t0 = time.time()
        E = self.E
        env = self.env()
        I = self.inputs()
        C = self.cache()
        G = self.geometry()
        M = self.models()
        env_ok = bool(env.ok.astype(bool).all())
        inputs_ok = bool(I[I.required.astype(bool)].ok.astype(bool).all())
        S = self.stages(inputs_ok, env_ok, C)
        problems = [f"environment: {r.check} = {r.value}" for r in env.itertuples() if not r.ok]
        problems += [f"input {r.input}: {r.note}" for r in I.itertuples() if r.required and not r.ok]
        core = getattr(E, "CORE_STAGES", ())
        problems += [f"core stage {r.stage} is {r.readiness}: {r.reason}" for r in S.itertuples()
                     if r.stage in core and r.readiness in (BLOCKED, INVALID)]
        R = PreflightReport(env, I, C, S, G, M, problems)
        E.ws.save_table(pd.concat([env.assign(section="environment"), I.assign(section="raw_inputs")],
                                  ignore_index=True), "audit/preflight_inputs.csv")
        E.ws.save_table(S, "audit/preflight_stage_readiness.csv")
        E.ws.save_table(AR.contract_table(type(E)), "audit/preflight_artifact_contract.csv")
        E.ws.save_table(C, "audit/preflight_cache_validation.csv")
        E.ws.save_table(G, "audit/preflight_geometry_precision.csv")
        E.ws.save_table(M, "audit/preflight_models.csv")
        E.ws.save_json({"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
                        "eda_code_rev": EDA_CODE_REV, "commit": E.head, "mode": E.c["mode"], "ready": R.ready,
                        "problems": problems, "readiness": S.set_index("stage").readiness.to_dict(),
                        "undeclared_consumption": AR.undeclared_consumption(type(E))},
                       "audit/preflight_report.json")
        E.ws.save_text(markdown(R, E), "reports/PREFLIGHT_READINESS.md")
        E.ws.push()
        print(f"[preflight] {'READY' if R.ready else 'NOT READY'} in {time.time() - t0:.0f}s | "
              f"{S.readiness.value_counts().to_dict()}")
        return R


# =============================================================================== helpers
def _is_sha(ref) -> bool:
    return isinstance(ref, str) and 7 <= len(ref) <= 40 and all(ch in "0123456789abcdef" for ch in ref.lower())


def _meminfo_gb() -> dict:
    try:
        out = {}
        for ln in Path("/proc/meminfo").read_text().splitlines():
            k, v = ln.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                out[k] = round(int(v.split()[0]) / 2 ** 20, 1)
        return out
    except OSError:
        return {}


def _switch(E, stage: str):
    from .run import STAGE_SWITCH

    k = STAGE_SWITCH.get(stage)
    return None if k is None else bool(E.c["stages"].get(k, True))


def _recovery(st: str, stage: str, why: str) -> str:
    return {READY: "", FROM_CACHE: "", RECOMPUTE: f"runs in E.run() ({why})",
            OPT_UNAVAILABLE: f"optional: {why}", BLOCKED: f"fix upstream first: {why}",
            INVALID: f"fix inputs: {why}"}.get(st, "")


def markdown(R: PreflightReport, E) -> str:
    def tbl(df):
        try:
            return df.to_markdown(index=False)
        except ImportError:
            return "```\n" + df.to_string(index=False) + "\n```"

    S = R.summary()
    return "\n".join([
        "# Preflight readiness", "", f"Code rev {EDA_CODE_REV}; commit {E.head}; mode {E.c['mode']}; generated "
        f"{time.strftime('%Y-%m-%d %H:%M')}. Static checks only: a READY stage can still fail at run time.", "",
        f"**{'READY' if R.ready else 'NOT READY'}**", "", *(f"* {p}" for p in R.problems), "",
        "## Stage readiness", "", tbl(S), "", "## Raw inputs", "",
        tbl(R.inputs[[c for c in ("input", "exists", "ok", "height", "width", "epsg", "count", "dtype", "note")
                      if c in R.inputs]]), "", "## Geometry precision (cached fault_geometry.csv)", "",
        tbl(R.geometry), "", "## Frozen engine / DEM", "", tbl(R.models), "",
        "Files: audit/preflight_inputs.csv, preflight_stage_readiness.csv, preflight_artifact_contract.csv, "
        "preflight_cache_validation.csv, preflight_geometry_precision.csv, preflight_models.csv, "
        "preflight_report.json", ""])


def load_last(E) -> dict:
    """Reload the last saved preflight (later session)."""
    return {"report": E.ws.read_json("audit/preflight_report.json"),
            "stages": E.ws.load_table("audit/preflight_stage_readiness.csv")}

