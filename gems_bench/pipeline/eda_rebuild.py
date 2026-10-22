"""Minimal Phase-1 rebuild: only the EDA artifacts later phases read, regenerated from the raw rasters.

Downstream consumers (inspected in the code, not assumed):

* ``segments.load_segment_table`` / ``segment_id_raster``  -> ``datasets/known_fault_segments.{parquet,gpkg}``,
  ``datasets/segment_archetypes.parquet``  (segment attributes, strata, archetypes)
* ``features.FeatureStore._eda`` (``eda:*`` channels, G4 group, B0_lineament floor) -> ``features/lineament_maps.tif``,
  ``features/crossmodal_orientation.tif``
* ``p3.context.load_context`` -> ``outputs/benchmark/datasets/{native_segment_ids.tif, segment_attributes.parquet}``

The rebuild calls the *same* ``gems_eda`` section functions as notebook 01 (same ``EDAConfig`` defaults:
2048 px working grid, seed 7, profile scale max(2 km, 3 px)) on a private output folder
``outputs/eda_rebuild/<version>/``, then *installs* each artifact group into ``outputs/eda/`` only if that
group is entirely absent there. Groups are installed whole because their members share segment ids. A
historical artifact is never overwritten; a partially present historical group is reported, not mixed.
Figures and the remaining EDA tables are not regenerated (they are not consumed downstream).

Resumption:

* a finished rebuild folder carries ``REBUILD_COMPLETE.json``; a later session whose install was interrupted
  (or whose Drive push failed) installs from it instead of recomputing (same ``max_dim`` / ``seed`` only);
* ``outputs/eda/INSTALL_STATE.json`` is written ``in_progress`` before the first copy and ``complete`` after the
  manifest. A group left partial by *our own* interrupted install is completed from the recorded rebuild
  folder; a partial group without that record is historical and stays blocked.

Every installed file is listed in ``outputs/eda/manifest.json`` (entry key ``origin: rebuilt``) and in
``outputs/eda/REBUILD_PROVENANCE.json``. A rebuild is a new artifact version, not a claim that it is
byte-identical to the historical Phase-1 run (library versions can change skeletonisation / sampling).
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..artifacts.atomic import atomic_write_json, sha256_file
from ..artifacts.validation import EDA_GROUPS, EDA_MANIFEST_NAMES, check_eda_file, read_json, validate_eda_group
from .context import PipelineStop

NEEDS = {"segments": ("s2", "s3", "s6"), "lineaments": ("s4",), "crossmodal": ("s3", "s5")}
REBUILD_DIR = "outputs/eda_rebuild"
COMPLETE = "REBUILD_COMPLETE.json"
INSTALL_STATE = "INSTALL_STATE.json"
SEG_PROVENANCE = "native_segments_provenance.json"


def group_states(eda_dir: Path) -> dict[str, str]:
    """ok | missing | partial | invalid per group of ``outputs/eda``."""
    return {g: validate_eda_group(eda_dir, g).status for g in EDA_GROUPS}


def _rebuild_paths(root: Path, version: str):
    """``EDAPaths`` whose output folder is ``outputs/eda_rebuild/<version>`` (raw inputs resolve as usual).

    The version is a dataclass *field* (``rebuild_version``): a class-body default ``version: str = version``
    cannot see the enclosing function's argument and raised ``NameError`` on Colab."""
    from gems_eda.config import EDAPaths

    if not version or not re.fullmatch(r"[A-Za-z0-9_.-]+", version):
        raise ValueError(f"invalid EDA rebuild version {version!r} (letters, digits, '_', '.', '-')")

    @dataclass(frozen=True)
    class RebuildPaths(EDAPaths):
        rebuild_version: str = ""

        @property
        def out(self) -> Path:
            return self.root / "outputs" / "eda_rebuild" / self.rebuild_version

    return RebuildPaths(root=Path(root), rebuild_version=version).ensure()


def _preflight() -> None:
    """Fail in seconds, not after 30 minutes: ``gems_eda.io.write_lines`` silently falls back to a WKT Parquet
    (overwriting ``known_fault_segments.parquet``) when geopandas is missing, and the GeoPackage is required."""
    missing = []
    for mod in ("geopandas", "pyogrio", "shapely", "rasterio", "skimage"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise PipelineStop(f"the EDA rebuild needs {missing}, which cannot be imported in this runtime",
                           "install the Colab extra (pip install -e '/content/GEMS[colab]', done by the notebook's "
                           "installation cell), restart the session, and re-run")


def _load_context(root: Path, paths, max_dim: int | None, seed: int):
    """gems_eda.context.load_context with a redirected output folder (same steps, same defaults)."""
    from rasterio.enums import Resampling

    from gems_eda.config import EDAConfig, Manifest
    from gems_eda.context import EDAContext
    from gems_eda.enhance import Enhancer, standardize_signal
    from gems_eda.io import load_feature_stack, load_on_grid
    from gems_eda.morphology import build_fault_network

    cfg = EDAConfig(max_dim=max_dim, seed=seed)
    manifest = Manifest(paths)
    for key in ("features", "faults", "submission"):
        if not paths.raw(key).exists():
            raise PipelineStop(f"raw file '{key}' not found under {paths.root} (looked in {paths.raw(key)})",
                               "run the raw_data stage first (pipe.step('raw_data'))")
    stack = load_feature_stack(paths.raw("features"), cfg.max_dim)
    grid = stack.grid
    faults, finfo = load_on_grid(paths.raw("faults"), grid, Resampling.max)
    sub, sinfo = load_on_grid(paths.raw("submission"), grid, Resampling.nearest)
    mask = np.isfinite(faults) & (faults > 0)
    if not mask.any():
        raise PipelineStop("existing_faults.tif has no positive pixel on the EDA grid",
                           "re-acquire the labels file (raw_data stage)")
    finfo["positive_frac"] = float(mask.mean())
    cfg.profile_half_km = max(cfg.profile_half_km, 20 * grid.pixel_km)
    cfg.station_spacing_km = max(cfg.station_spacing_km, 2 * grid.pixel_km)
    cfg.background_min_km = max(cfg.background_min_km, cfg.profile_half_km)
    net = build_fault_network(mask, grid, cfg.region_grid)
    ctx = EDAContext(cfg, paths, manifest, stack, faults, finfo, sub, sinfo, net, Enhancer(grid),
                     np.random.default_rng(cfg.seed))
    for name in stack.names:
        ctx.signals[name], ctx.signal_notes[name] = standardize_signal(stack[name])
    return ctx


# ---------------------------------------------------------------------------
# finished rebuild folders / interrupted installs
# ---------------------------------------------------------------------------
def _params(max_dim, seed) -> dict:
    return {"max_dim": None if max_dim is None else int(max_dim), "seed": int(seed)}


def completed_rebuilds(root: Path, max_dim=2048, seed: int = 7) -> list[tuple[str, list[str]]]:
    """(version, groups) of finished rebuild folders under ``root`` made with these parameters, newest first."""
    base = Path(root) / REBUILD_DIR
    out = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir(), reverse=True):
        rec = read_json(d / COMPLETE, None) if d.is_dir() else None
        if rec and rec.get("params") == _params(max_dim, seed):
            out.append((d.name, list(rec.get("groups") or [])))
    return out


def install_state(root: Path) -> dict:
    return read_json(Path(root) / "outputs" / "eda" / INSTALL_STATE, {}) or {}


def rebuild_sources(root: Path, max_dim=2048, seed: int = 7) -> list[str]:
    """Rebuild folders worth staging from Drive before rebuilding: the source of an interrupted install and
    the newest finished rebuild with matching parameters (relative paths)."""
    rels = []
    st = install_state(root)
    if st.get("status") == "in_progress" and st.get("version"):
        rels.append(f"{REBUILD_DIR}/{st['version']}")
    done = completed_rebuilds(root, max_dim, seed)
    if done:
        rels.append(f"{REBUILD_DIR}/{done[0][0]}")
    return list(dict.fromkeys(rels))


def _source_valid(src_out: Path, groups: list[str]) -> bool:
    return all(check_eda_file(src_out / rel, spec).ok for g in groups for rel, spec in EDA_GROUPS[g].items())


def rebuild_minimal(root: Path, *, max_dim: int | None = 2048, seed: int = 7, groups: list[str] | None = None,
                    version: str | None = None) -> dict:
    """Regenerate the missing EDA groups into outputs/eda_rebuild/<version>/ and install them; returns a report."""
    root = Path(root)
    eda = root / "outputs" / "eda"
    states = group_states(eda)
    wanted = list(groups or EDA_GROUPS)
    todo = [g for g in wanted if states[g] == "missing"]
    blocked = {g: s for g, s in states.items() if g in wanted and s in ("partial", "invalid")}
    report = {"states_before": states, "rebuilt": [], "installed": [], "blocked": blocked, "version": None,
              "reused_rebuild": None, "resumed_install": []}

    # 1. our own interrupted install: complete it from the recorded source folder
    st = install_state(root)
    if blocked and st.get("status") == "in_progress":
        src = root / REBUILD_DIR / str(st.get("version"))
        resumable = [g for g in blocked if g in (st.get("groups") or [])]
        if resumable and _source_valid(src, resumable):
            report["installed"] += install(root, src, resumable, st["version"], resume=resumable)
            report["resumed_install"] = resumable
            for g in resumable:
                blocked.pop(g)
        elif resumable:
            report["blocked_detail"] = (f"interrupted install of {resumable} from {REBUILD_DIR}/{st.get('version')}, "
                                        "but that rebuild folder is missing or invalid")
    if not todo:
        return report

    # 2. a finished rebuild with the same parameters (e.g. the install or the Drive push was interrupted)
    for v, gs in completed_rebuilds(root, max_dim, seed):
        src = root / REBUILD_DIR / v
        if set(todo) <= set(gs) and _source_valid(src, todo):
            print(f"[eda] reusing finished rebuild {REBUILD_DIR}/{v} for {todo} (no recomputation)")
            report.update(version=v, reused_rebuild=v)
            report["installed"] += install(root, src, todo, v)
            return report

    # 3. recompute
    _preflight()
    import matplotlib

    matplotlib.use("Agg")                      # no figure windows: the rebuild writes data, not figures
    import matplotlib.pyplot as plt

    from gems_eda import sections as S
    from gems_eda.context import add_derived_signals

    version = version or time.strftime("v%Y%m%d_%H%M%S")
    paths = _rebuild_paths(root, version)
    t0 = time.perf_counter()
    print(f"[eda] rebuilding groups {todo} -> {paths.out}")
    ctx = _load_context(root, paths, max_dim, seed)
    need = {s for g in todo for s in NEEDS[g]}
    prof, derived = None, []
    # same call order as notebooks/01_geological_eda.ipynb (s2 -> s3 -> s4 -> s5 -> s6); s2 uses no RNG, so the
    # profile stations of s3 (the only RNG consumer before s6) are sampled as in the historical run
    if need & {"s3", "s5", "s6"}:
        derived = add_derived_signals(ctx, max(2.0, 3 * ctx.grid.pixel_km))
    if "s2" in need:
        S.s2_morphology(ctx)
    if need & {"s3", "s5", "s6"}:
        prof = S.s3_profiles(ctx, derived)
    if "s4" in need:
        S.s4_fused_lineaments(ctx)
    if "s5" in need:
        S.s5_crossmodal(ctx, prof)
    if "s6" in need:
        S.s6_archetypes(ctx, prof)
    plt.close("all")
    bad = {g: [rel for rel, spec in EDA_GROUPS[g].items() if not check_eda_file(paths.out / rel, spec).ok]
           for g in todo}
    bad = {g: v for g, v in bad.items() if v}
    if bad:
        raise RuntimeError(f"EDA rebuild {version} finished but produced invalid files: {bad} "
                           f"(inspect {paths.out}; nothing was installed)")
    from .._version import provenance

    atomic_write_json(paths.out / COMPLETE, {"version": version, "groups": todo, "params": _params(max_dim, seed),
                                             "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                             "minutes": round((time.perf_counter() - t0) / 60, 1), **provenance()})
    report["rebuilt"] = todo
    report["version"] = version
    report["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    report["installed"] += install(root, paths.out, todo, version)
    return report


def install(root: Path, src_out: Path, groups: list[str], version: str, resume: list[str] = ()) -> list[str]:
    """Copy whole groups from a rebuild folder into outputs/eda (only groups absent there, or ``resume`` groups
    left partial by an interrupted install of the same version), update the manifest."""
    eda = Path(root) / "outputs" / "eda"
    installed = []
    man_p = eda / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    todo = []
    for g in groups:
        if validate_eda_group(eda, g).status != "missing" and g not in resume:
            continue                                       # never mix with / overwrite historical files
        bad = [rel for rel, spec in EDA_GROUPS[g].items() if not check_eda_file(Path(src_out) / rel, spec).ok]
        if bad:
            raise RuntimeError(f"rebuilt group {g} failed validation: {bad}")
        todo.append(g)
    if not todo:
        return installed
    state_p = eda / INSTALL_STATE
    files = [rel for g in todo for rel in EDA_GROUPS[g]]
    atomic_write_json(state_p, {"status": "in_progress", "version": version, "groups": todo, "files": files,
                                "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    for g in todo:
        for rel in EDA_GROUPS[g]:
            dst = eda / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".partial")
            shutil.copyfile(Path(src_out) / rel, tmp)
            tmp.replace(dst)
            man[EDA_MANIFEST_NAMES[rel]] = {"path": f"outputs/eda/{rel}", "kind": _kind(rel),
                                            "description": f"rebuilt by the master pipeline ({version})",
                                            "origin": "rebuilt", "rebuild_version": version}
            installed.append(f"outputs/eda/{rel}")
    atomic_write_json(man_p, man)
    prov_p = eda / "REBUILD_PROVENANCE.json"
    prov = json.loads(prov_p.read_text(encoding="utf-8")) if prov_p.exists() else {"rebuilds": []}
    from .._version import provenance

    prov["rebuilds"].append({"version": version, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "groups": todo,
                             "files": installed, "source": f"{REBUILD_DIR}/{version}",
                             "historical_equivalence": False, **provenance()})
    atomic_write_json(prov_p, prov)
    atomic_write_json(state_p, {"status": "complete", "version": version, "groups": todo, "files": files,
                                "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return installed


def adopt_manifest(root: Path) -> list[str]:
    """Add manifest entries for valid historical files that were uploaded without ``manifest.json`` (adopted,
    not rebuilt). Existing entries are kept unchanged."""
    eda = Path(root) / "outputs" / "eda"
    man_p = eda / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    added = []
    for g, files in EDA_GROUPS.items():
        if validate_eda_group(eda, g).status != "ok":
            continue
        for rel in files:
            name = EDA_MANIFEST_NAMES[rel]
            if name not in man:
                man[name] = {"path": f"outputs/eda/{rel}", "kind": _kind(rel), "description": "historical Phase-1 artifact",
                             "origin": "adopted_existing"}
                added.append(name)
    if added:
        atomic_write_json(man_p, man)
    return added


def _kind(rel: str) -> str:
    return "vector" if rel.endswith(".gpkg") else "raster" if rel.endswith(".tif") else "dataset"


# ---------------------------------------------------------------------------
# native-grid segment rasters (Phase-2 section 6 logic)
# ---------------------------------------------------------------------------
def eda_segments_fingerprint(eda_dir: Path) -> dict:
    """sha256 of the Phase-1 segment files the native rasters are derived from (seg_id identity)."""
    out = {}
    for rel in ("datasets/known_fault_segments.parquet", "datasets/known_fault_segments.gpkg",
                "datasets/segment_archetypes.parquet"):
        p = Path(eda_dir) / rel
        out[rel] = sha256_file(p) if p.exists() else None
    return out


def build_native_segments(data, root: Path) -> list[str]:
    """native_segment_ids.tif + segment_attributes.parquet in outputs/benchmark/datasets (only missing ones).

    Both files are derived from one Phase-1 segment set; ``native_segments_provenance.json`` records its sha256
    so a later EDA rebuild cannot silently pair new segment ids with an old raster."""
    from .. import segments
    from ..artifacts.validation import check_geotiff, check_parquet

    root = Path(root)
    eda = root / "outputs" / "eda"
    rep = validate_eda_group(eda, "segments")
    if not rep.ok:
        raise PipelineStop(f"Phase-1 segment artifacts are {rep.status}: {rep.summary()}",
                           "run the geological_eda stage first (pipe.step('geological_eda')), or restore "
                           "outputs/eda/datasets from the local project (docs/ARTIFACT_RECOVERY.md)")
    d = root / "outputs" / "benchmark" / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    seg_p, tab_p, prov_p = d / "native_segment_ids.tif", d / "segment_attributes.parquet", d / SEG_PROVENANCE
    src_fp = eda_segments_fingerprint(eda)
    prov = read_json(prov_p, None)
    seg_ok = check_geotiff(seg_p, count=1, bands=("seg_id",)).ok
    tab_ok = check_parquet(tab_p, columns=("seg_id", "length_km", "primary_archetype")).ok
    if prov and prov.get("eda_segments") != src_fp and (seg_ok or tab_ok):
        raise PipelineStop("native segment rasters were derived from another Phase-1 segment set "
                           "(sha256 differs from outputs/eda/datasets)",
                           "existing files are preserved: move outputs/benchmark/datasets/native_segment_ids.tif, "
                           "segment_attributes.parquet and native_segments_provenance.json aside on Drive, then re-run")
    if seg_ok and tab_ok:
        return []
    if (seg_ok or tab_ok) and not prov:
        raise PipelineStop(f"only one of the native segment files is present ({'raster' if seg_ok else 'table'}) and "
                           "its source segment set is not recorded, so a regenerated partner could disagree on seg_id",
                           "restore the missing file from the same source, or move the present one aside on Drive")
    tab = segments.load_segment_table(eda)
    seg = segments.segment_id_raster(data)
    ids = set(np.unique(seg[seg >= 0]).tolist())
    unknown = ids - set(tab.seg_id.astype(int).tolist())
    if unknown:
        raise RuntimeError(f"{len(unknown)} seg_id(s) in known_fault_segments.gpkg are absent from "
                           "known_fault_segments.parquet: the Phase-1 segment group is internally inconsistent")
    n_lab = int(data.labels.sum())
    frac = float((seg >= 0).sum() / max(n_lab, 1))
    if frac == 0.0:
        raise RuntimeError("no native label pixel lies within 300 m of a Phase-1 segment: check the GeoPackage CRS "
                           "against the native grid")
    made = []
    if not seg_ok:
        tmp = seg_p.with_name("native_segment_ids.tif.partial")
        segments.save_segment_raster(data, seg, tmp)
        tmp.replace(seg_p)
        made.append("outputs/benchmark/datasets/native_segment_ids.tif")
    if not tab_ok:
        tmp = tab_p.with_name("segment_attributes.parquet.partial")
        tab.to_parquet(tmp, index=False)
        tmp.replace(tab_p)
        made.append("outputs/benchmark/datasets/segment_attributes.parquet")
    from .._version import provenance

    atomic_write_json(prov_p, {"eda_segments": src_fp, "n_segments": int(len(tab)), "label_px": n_lab,
                               "label_px_assigned_frac": round(frac, 4), "max_dist_m": 300.0,
                               "crs": str(data.grid.crs), "shape": [int(data.shape[0]), int(data.shape[1])],
                               "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **provenance()})
    print(f"[segments] label px assigned to a Phase-1 segment: {frac:.3f}")
    return made
