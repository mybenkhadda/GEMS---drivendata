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

Every installed file is listed in ``outputs/eda/manifest.json`` (entry key ``origin: rebuilt``) and in
``outputs/eda/REBUILD_PROVENANCE.json``. A rebuild is a new artifact version, not a claim that it is
byte-identical to the historical Phase-1 run (library versions can change skeletonisation / sampling).
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..artifacts.atomic import atomic_write_json
from ..artifacts.validation import EDA_GROUPS, EDA_MANIFEST_NAMES, check_eda_file, validate_eda_group

NEEDS = {"segments": ("s2", "s3", "s6"), "lineaments": ("s4",), "crossmodal": ("s3", "s5")}


def group_states(eda_dir: Path) -> dict[str, str]:
    """ok | missing | partial | invalid per group of ``outputs/eda``."""
    return {g: validate_eda_group(eda_dir, g).status for g in EDA_GROUPS}


def _rebuild_paths(root: Path, version: str):
    from gems_eda.config import EDAPaths

    @dataclass(frozen=True)
    class RebuildPaths(EDAPaths):
        version: str = version

        @property
        def out(self) -> Path:
            return self.root / "outputs" / "eda_rebuild" / self.version

    return RebuildPaths(Path(root)).ensure()


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
    stack = load_feature_stack(paths.raw("features"), cfg.max_dim)
    grid = stack.grid
    faults, finfo = load_on_grid(paths.raw("faults"), grid, Resampling.max)
    sub, sinfo = load_on_grid(paths.raw("submission"), grid, Resampling.nearest)
    mask = np.isfinite(faults) & (faults > 0)
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


def rebuild_minimal(root: Path, *, max_dim: int | None = 2048, seed: int = 7, groups: list[str] | None = None,
                    version: str | None = None) -> dict:
    """Regenerate the missing EDA groups into outputs/eda_rebuild/<version>/ and install them; returns a report."""
    import matplotlib

    matplotlib.use("Agg")                      # no figure windows: the rebuild writes data, not figures
    import matplotlib.pyplot as plt

    from gems_eda import sections as S
    from gems_eda.context import add_derived_signals

    root = Path(root)
    eda = root / "outputs" / "eda"
    states = group_states(eda)
    todo = [g for g in (groups or EDA_GROUPS) if states[g] == "missing"]
    blocked = {g: s for g, s in states.items() if s in ("partial", "invalid")}
    report = {"states_before": states, "rebuilt": [], "installed": [], "blocked": blocked, "version": None}
    if not todo:
        return report
    version = version or time.strftime("v%Y%m%d_%H%M%S")
    paths = _rebuild_paths(root, version)
    t0 = time.perf_counter()
    print(f"[eda] rebuilding groups {todo} -> {paths.out}")
    ctx = _load_context(root, paths, max_dim, seed)
    need = {s for g in todo for s in NEEDS[g]}
    prof, derived = None, []
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
    report["rebuilt"] = todo
    report["version"] = version
    report["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    report["installed"] = install(root, paths.out, todo, version)
    return report


def install(root: Path, src_out: Path, groups: list[str], version: str) -> list[str]:
    """Copy whole groups from a rebuild folder into outputs/eda (only groups absent there), update the manifest."""
    eda = Path(root) / "outputs" / "eda"
    installed = []
    man_p = eda / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    for g in groups:
        if validate_eda_group(eda, g).status != "missing":
            continue                                       # never mix with / overwrite historical files
        files = EDA_GROUPS[g]
        bad = [rel for rel, spec in files.items() if not check_eda_file(src_out / rel, spec).ok]
        if bad:
            raise RuntimeError(f"rebuilt group {g} failed validation: {bad}")
        for rel in files:
            dst = eda / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".partial")
            shutil.copyfile(src_out / rel, tmp)
            tmp.replace(dst)
            man[EDA_MANIFEST_NAMES[rel]] = {"path": f"outputs/eda/{rel}", "kind": _kind(rel),
                                            "description": f"rebuilt by the master pipeline ({version})",
                                            "origin": "rebuilt", "rebuild_version": version}
            installed.append(f"outputs/eda/{rel}")
    if installed:
        atomic_write_json(man_p, man)
        prov_p = eda / "REBUILD_PROVENANCE.json"
        prov = json.loads(prov_p.read_text(encoding="utf-8")) if prov_p.exists() else {"rebuilds": []}
        from .._version import provenance

        prov["rebuilds"].append({"version": version, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "groups": groups,
                                 "files": installed, "source": f"outputs/eda_rebuild/{version}",
                                 "historical_equivalence": False, **provenance()})
        atomic_write_json(prov_p, prov)
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
def build_native_segments(data, root: Path) -> list[str]:
    """native_segment_ids.tif + segment_attributes.parquet in outputs/benchmark/datasets (only missing ones)."""
    from .. import segments
    from ..artifacts.validation import check_geotiff, check_parquet

    d = Path(root) / "outputs" / "benchmark" / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    made = []
    seg_p, tab_p = d / "native_segment_ids.tif", d / "segment_attributes.parquet"
    if not check_geotiff(seg_p).ok:
        seg = segments.segment_id_raster(data)
        tmp = seg_p.with_name("native_segment_ids.partial.tif")
        segments.save_segment_raster(data, seg, tmp)
        tmp.replace(seg_p)
        made.append("outputs/benchmark/datasets/native_segment_ids.tif")
        print(f"[segments] label px assigned to a Phase-1 segment: {float((seg >= 0).sum() / max(data.labels.sum(), 1)):.3f}")
    if not check_parquet(tab_p).ok:
        tab = segments.load_segment_table(data.paths.eda)
        tmp = tab_p.with_name("segment_attributes.partial.parquet")
        tab.to_parquet(tmp, index=False)
        tmp.replace(tab_p)
        made.append("outputs/benchmark/datasets/segment_attributes.parquet")
    return made
