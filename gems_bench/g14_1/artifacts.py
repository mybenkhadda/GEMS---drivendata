"""Artifact discovery, provenance, Drive restoration and stage-level resumability (Phase B).

Search roots follow the conventions the repository already writes to (read from the code, see
docs/g14_1/ARTIFACT_REUSE.md):

* G14 experiment: ``<local_root>/experiments/<exp>/`` and ``<drive>/outputs/G14_GEOFAULTFORMER/experiments/<exp>/``
  (``checkpoints/<unit>_best.pt``, ``manifests/<unit>_train.json``, ``maps/<unit>.npy``, ``maps/full.npy``,
  ``evaluation/*__scorecard.json``, ``submissions/*.tif``); shared: ``.../shared/`` (stack manifest, folds).
* G13: ``/content/gems_g13_work`` and ``<drive>/outputs/G13_GEODISCOVERY`` (``splits/params.json``,
  ``predictions/<model>/{oof,disc<d>,full}.npy``, ``regimes/regimes.npy``, ``tables/scoreboard.csv``,
  ``features/registry.csv``, ``submissions/*.tif``).
* P6 W2: ``<drive>/outputs/p6_fast_track/stack/maps/{task}_{zone}.npy`` (unit crops, Phase-5 units),
  ``stack/maps/full_score.tif``; P6 boost submissions ``<drive>/outputs/p6_boost/submissions/*.tif``.
* P7: ``<drive>/outputs/p7_geofaultformer/**/{oof,disc,full_region}_*.tif``.
* G07: ``<drive>/outputs/g07_multiscale_fault_discovery/**`` (maps named oof / disc / full).

Rules: a persistent (Drive) copy is used in place when the local copy is absent; nothing is regenerated because it is
missing locally. Predictions are tagged with their protocol (backend and unit family) and grid; ``compatible`` refuses
to combine maps from different grids or protocols.

UNTESTED: written without execution.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================== registry
@dataclass
class ArtifactSpec:
    name: str
    kind: str                          # map | checkpoint | manifest | table | raster | submission | stack | splits
    patterns: tuple                    # glob patterns relative to each root of ``roots``
    roots: tuple                       # root keys (see ``Roots``)
    protocol: str = ""                 # g13 | phase5 | full | "" (protocol of a prediction)
    needed_by: tuple = ()              # stage ids that need it
    many: bool = True


@dataclass
class Roots:
    """Resolved search roots (local first, then the persistent Drive mirror)."""
    local_root: Path
    drive_root: Path
    g14_exp: str
    g14_drive_out: Path
    g13_local: Path = Path("/content/gems_g13_work")
    extra: dict = field(default_factory=dict)

    def get(self, key: str) -> list[Path]:
        d, L = self.drive_root / "outputs", self.local_root
        m = {
            "g14_exp": [L / "experiments" / self.g14_exp, self.g14_drive_out / "experiments" / self.g14_exp],
            "g14_shared": [L / "shared", self.g14_drive_out / "shared"],
            "g14_all": [L / "experiments", self.g14_drive_out / "experiments"],
            "g13": [self.g13_local, d / "G13_GEODISCOVERY"],
            "p6": [d / "p6_fast_track"],
            "p6_boost": [d / "p6_boost"],
            "p7": [d / "p7_geofaultformer"],
            "g07": [d / "g07_multiscale_fault_discovery"],
            "drive_outputs": [d],
            "drive": [self.drive_root],
        }
        return m.get(key, []) + [Path(p) for p in self.extra.get(key, [])]


REGISTRY: list[ArtifactSpec] = [
    ArtifactSpec("g14_full_map", "map", ("maps/full.npy",), ("g14_exp",), "full", ("R02", "R03", "R05", "R17"), False),
    ArtifactSpec("g14_full_mc_std", "map", ("maps/full_mc_std.npy",), ("g14_exp",), "full", ("R02",), False),
    ArtifactSpec("g14_unit_maps", "map", ("maps/geo_*.npy", "maps/disc_official_*.npy", "maps/regime_*.npy"),
                 ("g14_exp",), "g13", ("R02", "R03", "R04", "R05")),
    ArtifactSpec("g14_checkpoints", "checkpoint", ("checkpoints/*_best.pt",), ("g14_exp",), "", ("R06",)),
    ArtifactSpec("g14_train_manifests", "manifest", ("manifests/*_train.json",), ("g14_exp",), "", ("R00", "R06")),
    ArtifactSpec("g14_scorecards", "table", ("evaluation/*__scorecard.json",), ("g14_exp",), "", ("R03", "R17")),
    ArtifactSpec("g14_choice_tables", "table", ("evaluation/*__geo_choice.csv", "evaluation/*__disc_official_choice.csv",
                                                "evaluation/*__operating.csv"), ("g14_exp",), "", ("R03",)),
    ArtifactSpec("g14_submissions", "submission", ("submissions/*.tif",), ("g14_exp",), "full", ("R01",)),
    ArtifactSpec("g14_stack_manifest", "stack", ("stack/manifest.json",), ("g14_shared",), "", ("R06", "R08"), False),
    ArtifactSpec("g14_stack_levels", "stack", ("stack/stack_l*.npy",), ("g14_shared",), "", ("R06", "R08")),
    ArtifactSpec("g14_fold_manifests", "splits", ("folds/*/manifest.json", "folds/*/fold_audit.csv"),
                 ("g14_shared",), "g13", ("R07",)),
    ArtifactSpec("g13_splits", "splits", ("splits/params.json",), ("g13",), "g13", ("R03", "R07"), False),
    ArtifactSpec("g13_regimes", "raster", ("regimes/regimes.npy",), ("g13",), "", ("R04", "R05", "R13"), False),
    ArtifactSpec("g13_scoreboard", "table", ("tables/scoreboard.csv",), ("g13",), "", ("R05",), False),
    ArtifactSpec("g13_feature_registry", "table", ("features/registry.csv",), ("g13",), "", ("R08",), False),
    ArtifactSpec("g13_oof_maps", "map", ("predictions/*/oof.npy",), ("g13",), "g13", ("R05",)),
    ArtifactSpec("g13_disc_maps", "map", ("predictions/*/disc*.npy",), ("g13",), "g13", ("R03", "R05")),
    ArtifactSpec("g13_full_maps", "map", ("predictions/*/full.npy",), ("g13",), "full", ("R05",)),
    ArtifactSpec("g13_submissions", "submission", ("submissions/*.tif",), ("g13",), "full", ("R01", "R05")),
    ArtifactSpec("p6_w2_unit_maps", "map", ("stack/maps/geo_*.npy", "stack/maps/disc_official_*.npy"), ("p6",),
                 "phase5", ("R05",)),
    ArtifactSpec("p6_w2_full", "raster", ("stack/maps/full_score.tif",), ("p6",), "full", ("R05",), False),
    ArtifactSpec("p6_submissions", "submission", ("submissions/*.tif", "**/submissions/*.tif"), ("p6", "p6_boost"),
                 "full", ("R01", "R05")),
    ArtifactSpec("p7_maps", "raster", ("**/oof_*.tif", "**/disc_*.tif", "**/full_region_*.tif"), ("p7",), "phase5",
                 ("R05",)),
    ArtifactSpec("g07_maps", "map", ("**/*oof*.npy", "**/*disc*.npy", "**/*full*.npy", "**/*full*.tif"), ("g07",),
                 "", ("R05",)),
]
SPEC = {s.name: s for s in REGISTRY}


# =============================================================================== discovery
def _sha256(path: Path, limit_mb: float = 64.0) -> str:
    """sha256 of the file (or of its first ``limit_mb`` MB plus its size for very large files: tagged '~')."""
    h = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size <= limit_mb * 2**20:
            for chunk in iter(lambda: f.read(2**20), b""):
                h.update(chunk)
            return h.hexdigest()[:16]
        h.update(f.read(int(limit_mb * 2**20)))
    h.update(str(size).encode())
    return "~" + h.hexdigest()[:15]


def array_header(path: Path) -> dict:
    """Shape / dtype without reading the payload (npy memmap header, GeoTIFF profile)."""
    try:
        if path.suffix == ".npy":
            a = np.load(path, mmap_mode="r")
            return {"shape": tuple(a.shape), "dtype": str(a.dtype)}
        if path.suffix in (".tif", ".tiff"):
            import rasterio

            with rasterio.open(path) as s:
                return {"shape": (s.count, s.height, s.width) if s.count > 1 else (s.height, s.width),
                        "dtype": s.dtypes[0], "crs": str(s.crs), "transform": tuple(s.transform)[:6],
                        "res": tuple(s.res), "nodata": s.nodata}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    return {}


def discover(roots: Roots, grid_shape: tuple | None = None, specs: list[ArtifactSpec] | None = None,
             hash_files: bool = False, max_per_pattern: int = 200) -> pd.DataFrame:
    """Every artifact found under the roots (local first). One row per file; ``location`` = local | drive."""
    rows = []
    for s in specs or REGISTRY:
        seen = set()
        for rk in s.roots:
            for root in roots.get(rk):
                if not root.exists():
                    continue
                loc = "local" if str(root).startswith(str(roots.local_root)) or str(root).startswith(
                    str(roots.g13_local)) else "drive"
                for pat in s.patterns:
                    for p in sorted(root.glob(pat))[:max_per_pattern]:
                        if not p.is_file() or ".partial" in p.name:
                            continue
                        rel = str(p.relative_to(root))
                        st = p.stat()
                        hdr = array_header(p) if s.kind in ("map", "raster", "submission", "stack") else {}
                        shp = hdr.get("shape")
                        grid_ok = None
                        if grid_shape is not None and shp is not None:
                            grid_ok = tuple(shp[-2:]) == tuple(grid_shape)
                        rows.append({"artifact": s.name, "kind": s.kind, "protocol": s.protocol, "location": loc,
                                     "root": str(root), "rel": rel, "path": str(p), "bytes": st.st_size,
                                     "mtime": time.strftime("%Y-%m-%d %H:%M", time.gmtime(st.st_mtime)),
                                     "shape": shp, "dtype": hdr.get("dtype"), "grid_ok": grid_ok,
                                     "header_error": hdr.get("error"),
                                     "sha256": _sha256(p) if hash_files else "",
                                     "duplicate_of_local": rel in seen, "needed_by": ",".join(s.needed_by)})
                        seen.add(rel)
    cols = ["artifact", "kind", "protocol", "location", "root", "rel", "path", "bytes", "mtime", "shape", "dtype",
            "grid_ok", "header_error", "sha256", "duplicate_of_local", "needed_by"]
    return pd.DataFrame(rows, columns=cols)


def best_copy(inv: pd.DataFrame, artifact: str, rel: str | None = None) -> Path | None:
    """Local copy if present and not smaller than the Drive copy, else the Drive copy (used in place)."""
    t = inv[inv.artifact == artifact]
    if rel is not None:
        t = t[t.rel == rel]
    if t.empty:
        return None
    loc = t[t.location == "local"]
    drv = t[t.location == "drive"]
    if len(loc) and (drv.empty or int(loc.bytes.iloc[0]) >= int(drv.bytes.iloc[0])):
        return Path(loc.path.iloc[0])
    return Path(drv.path.iloc[0]) if len(drv) else Path(loc.path.iloc[0])


def restore(src: Path, dst: Path) -> Path:
    """Copy a persistent artifact to local storage only when the local copy is missing or differs in size."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return dst


def unit_maps(inv: pd.DataFrame, artifact: str = "g14_unit_maps") -> dict[str, Path]:
    """{'geo_1': path, 'disc_official_0': path, ...} for one map artifact (best copy of each)."""
    out = {}
    for rel in sorted(set(inv[inv.artifact == artifact].rel)):
        out[Path(rel).stem] = best_copy(inv, artifact, rel)
    return out


def g13_models(inv: pd.DataFrame) -> dict[str, dict[str, Path]]:
    """G13 predictions grouped by model: {model: {'oof': p, 'disc0': p, ..., 'full': p}}."""
    out: dict[str, dict] = {}
    for art in ("g13_oof_maps", "g13_disc_maps", "g13_full_maps"):
        for rel in sorted(set(inv[inv.artifact == art].rel)):
            parts = Path(rel).parts                     # predictions/<model>/<file>.npy
            if len(parts) >= 3:
                out.setdefault(parts[-2], {})[Path(rel).stem] = best_copy(inv, art, rel)
    return out


# =============================================================================== provenance / hashes
def experiment_hash_check(exp_name: str, manifests: list[Path]) -> pd.DataFrame:
    """The experiment directory name ends with the semantic config hash; every training manifest must agree."""
    want = exp_name.rsplit("__", 1)[-1] if "__" in exp_name else None
    rows = []
    for m in manifests:
        try:
            j = json.loads(Path(m).read_text())
        except Exception as e:  # noqa: BLE001
            rows.append({"manifest": str(m), "status": f"UNREADABLE: {e}"})
            continue
        got = j.get("cfg_hash")
        rows.append({"manifest": str(m), "unit": j.get("unit"), "cfg_hash": got, "expected": want,
                     "schema_hash": j.get("schema_hash"), "n_channels": len(j.get("channels", [])),
                     "best_epoch": j.get("best_epoch"), "best_inner_score": j.get("best_inner_score"),
                     "selection_metric": j.get("selection_metric"),
                     "status": "OK" if want is None or got == want else "HASH_MISMATCH"})
    return pd.DataFrame(rows)


def feature_provenance(manifests: list[Path]) -> pd.DataFrame:
    """Channel table per checkpoint (Phase K): which inputs each trained model actually consumed."""
    rows = []
    for m in manifests:
        j = json.loads(Path(m).read_text())
        mods = j.get("modalities", [])
        for i, ch in enumerate(j.get("channels", [])):
            src = "known" if ch.startswith("known:") else ("raw" if ch.startswith("raw:") else
                                                           ("g14" if ch.startswith("g14:") else "g13"))
            rows.append({"unit": j.get("unit"), "index": i, "channel": ch, "source": src,
                         "target_derived": src == "known", "modalities_in_model": ",".join(mods)})
    return pd.DataFrame(rows)


def code_revision(repo_dir: str | Path = ".") -> str:
    try:
        return subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def compatible(meta_a: dict, meta_b: dict) -> tuple[bool, str]:
    """Two prediction artifacts may be combined only on the same grid and the same protocol family."""
    if tuple(meta_a.get("shape", ())[-2:]) != tuple(meta_b.get("shape", ())[-2:]):
        return False, f"grid {meta_a.get('shape')} vs {meta_b.get('shape')}"
    pa, pb = meta_a.get("protocol", ""), meta_b.get("protocol", "")
    if pa and pb and pa != pb and "full" not in (pa, pb):
        return False, f"protocol {pa} vs {pb} (different validation units)"
    if (pa == "full") != (pb == "full"):
        return False, "a full-region map cannot be compared with held-out unit maps"
    return True, "ok"


# =============================================================================== stage ledger
class StageLedger:
    """``manifests/stages.json``: stage -> {status, cfg_hash, inputs digest, outputs, timings, code revision}.

    ``fresh(stage, digest)`` is True when a completed record with the same digest exists and all its outputs are still
    on disk: the stage is skipped (resume) unless ``force``."""

    def __init__(self, path: Path, cfg_hash: str, code_rev: str = "unknown"):
        self.path = Path(path)
        self.cfg_hash, self.code_rev = cfg_hash, code_rev
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def digest(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def fresh(self, stage: str, digest: str) -> bool:
        r = self.data.get(stage)
        return bool(r and r.get("status") == "done" and r.get("digest") == digest
                    and all(Path(o).exists() for o in r.get("outputs", [])))

    def start(self, stage: str, digest: str) -> float:
        self.data[stage] = {"status": "running", "digest": digest, "cfg_hash": self.cfg_hash,
                            "code_rev": self.code_rev, "started_unix": time.time()}
        self._save()
        return time.perf_counter()

    def finish(self, stage: str, t0: float, outputs: list, notes: str = "") -> None:
        r = self.data.setdefault(stage, {})
        r.update(status="done", outputs=[str(o) for o in outputs], seconds=round(time.perf_counter() - t0, 2),
                 finished_unix=time.time(), notes=notes)
        self._save()

    def fail(self, stage: str, t0: float, error: str) -> None:
        r = self.data.setdefault(stage, {})
        r.update(status="failed", error=error[:2000], seconds=round(time.perf_counter() - t0, 2))
        self._save()

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{"stage": k, **{kk: v for kk, v in r.items() if kk != "outputs"},
                              "n_outputs": len(r.get("outputs", []))} for k, r in self.data.items()])

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".partial")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        os.replace(tmp, self.path)


# =============================================================================== preflight
STAGE_NEEDS = {
    "R01": [("g14_full_map", True), ("g14_submissions", False)],
    "R02": [("g14_full_map", False), ("g14_unit_maps", False)],
    "R03": [("g14_unit_maps", True), ("g13_splits", False)],
    "R04": [("g14_unit_maps", True)],
    "R05": [("g14_unit_maps", False), ("g13_oof_maps", False), ("p6_w2_unit_maps", False), ("p6_w2_full", False)],
    "R06": [("g14_checkpoints", True), ("g14_train_manifests", True), ("g14_stack_manifest", False)],
    "R07": [("g13_splits", False)],
    "R17": [("g14_full_map", False), ("g14_scorecards", False)],
}


def preflight(inv: pd.DataFrame, official_inputs: dict | None, grid_shape: tuple | None) -> pd.DataFrame:
    """Readiness of every artifact-dependent stage: READY / PARTIAL / BLOCKED with the missing items."""
    have = set(inv.artifact) if len(inv) else set()
    bad_grid = set(inv[(inv.grid_ok == False)].artifact) if len(inv) else set()   # noqa: E712
    rows = [{"stage": "inputs", "status": "READY" if official_inputs else "BLOCKED",
             "missing_required": "" if official_inputs else "official rasters (raw, known faults, template)",
             "missing_optional": "", "grid_mismatch": ""}]
    for st, needs in STAGE_NEEDS.items():
        req = [a for a, r in needs if r and a not in have]
        opt = [a for a, r in needs if not r and a not in have]
        status = "BLOCKED" if req else ("PARTIAL" if opt else "READY")
        if st in ("R02", "R05") and all(a not in have for a, _ in needs):
            status = "BLOCKED"
        rows.append({"stage": st, "status": status, "missing_required": ", ".join(req),
                     "missing_optional": ", ".join(opt),
                     "grid_mismatch": ", ".join(sorted(bad_grid & {a for a, _ in needs}))})
    rows.append({"stage": "grid", "status": "READY" if grid_shape else "UNKNOWN", "missing_required": "",
                 "missing_optional": "", "grid_mismatch": ", ".join(sorted(bad_grid))})
    return pd.DataFrame(rows)
