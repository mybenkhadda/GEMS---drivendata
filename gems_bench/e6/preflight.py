"""Read-only E6 Colab preflight: every input the E6 pipeline needs, checked on the REAL mounted files.

Nothing is written, staged, downloaded or computed: files are only listed and their headers opened. Each item gets
one outcome

    AVAILABLE_VALID      present and readable (where checked: right shape / schema)
    DERIVABLE_MISSING    absent, but the pipeline builds it itself (cost noted)
    BLOCKED_MISSING      absent or invalid, and nothing in the pipeline can produce it -> do not start training

and ``run_preflight`` prints every blocker in one report. A file present only locally or only on Drive counts as
present (the pipeline stages Drive files on first use). Runtime-only facts remain runtime facts: a header that opens
does not prove the pixels are right, so the stage-level checks of the pipeline still apply.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

AVAILABLE, DERIVABLE, BLOCKED = "AVAILABLE_VALID", "DERIVABLE_MISSING", "BLOCKED_MISSING"


def _first(roots, rel):
    for r in roots:
        p = Path(r) / rel
        if p.exists():
            return p
    return None


def _row(rows, artifact, status, path="", detail="", blocks=""):
    rows.append({"artifact": artifact, "status": status, "path": str(path or ""), "detail": str(detail)[:400],
                 "blocks": blocks})


def _raster_header(p: Path) -> dict:
    import rasterio

    with rasterio.open(p) as ds:
        return {"shape": (ds.height, ds.width), "count": ds.count, "crs": str(ds.crs), "res": tuple(ds.res),
                "transform": tuple(ds.transform)[:6], "descriptions": list(ds.descriptions)}


def _base_groups(code: Path, base: str) -> dict:
    p = Path(code) / "configs" / "baselines" / "legacy_specs" / f"{base}.json"
    return (json.loads(p.read_text(encoding="utf-8")).get("groups") or {}) if p.exists() else {}


def run_preflight(config_path, local_root="/content/gems_work", drive_root="/content/drive/MyDrive/GEMS_2",
                  code_root="/content/GEMS_E6", check_gpu: bool = True, verbose: bool = True) -> pd.DataFrame:
    """One row per required E6 input; prints the blockers. Returns the table (``.attrs['blocked']`` = list)."""
    from gems_eda.config import RAW_FILES

    from ..config import BenchConfig
    from ..experiment.config import ExperimentConfig
    from ..features import _fname

    rows: list[dict] = []
    local, drive, code = Path(local_root), Path(drive_root), Path(code_root)
    roots = [local, drive]
    if not drive.is_dir():
        _row(rows, "google_drive_mount", BLOCKED, drive, "Drive not mounted / GEMS_2 missing", "everything")
        return _finish(rows, verbose)
    _row(rows, "google_drive_mount", AVAILABLE, drive)
    cfg = ExperimentConfig.from_yaml(config_path)
    exp_id = cfg.experiment_id()
    _row(rows, "e6_config", AVAILABLE, config_path, f"experiment id {exp_id}; kind {cfg.get('kind')}")

    # -- original competition data -----------------------------------------------------------
    template = None
    for key, subs in (("features", ("", "data/raw/geodawn")), ("faults", ("", "data/raw/faults", "data/raw/geodawn")),
                      ("submission", ("", "data/raw/geodawn", "data/raw"))):
        p = next((q for s in subs for q in [_first(roots, f"{s}/{RAW_FILES[key]}".lstrip("/"))] if q), None)
        if p is None:
            _row(rows, f"geodawn_{key}", BLOCKED, "", f"{RAW_FILES[key]} not found under data/raw", "prepare")
            continue
        try:
            h = _raster_header(p)
            if key == "submission":
                template = h
            _row(rows, f"geodawn_{key}", AVAILABLE, p, f"{h['shape']} {h['crs']} res {h['res']} bands {h['count']}")
        except Exception as e:
            _row(rows, f"geodawn_{key}", BLOCKED, p, f"unreadable: {type(e).__name__}: {e}", "prepare")
    if template is not None and abs(template["res"][0] - 100) > 1e-6:
        _row(rows, "template_grid", BLOCKED, "", f"template resolution {template['res']} != 100 m", "export")

    # -- validation design ---------------------------------------------------------------------
    bc = BenchConfig()
    fold_rel = f"outputs/benchmark/folds/geo_k{bc.n_folds}_b{bc.block_km:g}km_buf{bc.buffer_km:g}km_s{bc.seed}"
    disc_rel = f"outputs/benchmark/folds/discovery_d{bc.n_discovery_folds}_link{bc.system_link_km:g}km_s{bc.seed}"
    for name, rel, f in (("geographic_folds", fold_rel, "fold_map.tif"), ("discovery_splits", disc_rel,
                                                                           "discovery_map.tif")):
        p = _first(roots, f"{rel}/{f}")
        _row(rows, name, AVAILABLE if p else BLOCKED, p or rel, "" if p else
             "folds / splits are the evaluation design; regenerating them would change every result", "prepare")
    seg = _first(roots, "outputs/benchmark/datasets/segment_attributes.parquet") or \
        _first(roots, "outputs/eda/datasets/known_fault_segments.parquet")
    _row(rows, "fault_segments", AVAILABLE if seg else BLOCKED, seg or "outputs/benchmark/datasets", "" if seg else
         "load_context5(with_segments=True) needs segment attributes", "prepare")

    # -- base channels (terrain + potential field) ---------------------------------------------
    model = cfg.get("model") or {}
    groups = _base_groups(code, model.get("base", ""))
    if not groups:
        _row(rows, "base_channel_schema", BLOCKED, model.get("base"), "legacy spec snapshot missing", "prepare")
    base_names = [c for g in groups.values() for c in g]
    miss = []
    for n in base_names:
        sub = "outputs/phase4/cache/features" if n.startswith("rs:") else "outputs/benchmark/cache/features"
        if _first(roots, f"{sub}/{_fname(n)}") is None:
            miss.append(n)
    _row(rows, "base_feature_caches", AVAILABLE if not miss else DERIVABLE, "outputs/{benchmark,phase4}/cache/features",
         f"{len(base_names) - len(miss)}/{len(base_names)} cached" + (f"; built from the raw raster on first use "
                                                                       f"(CPU, can take long): {miss[:6]}" if miss else ""))

    # -- DEM: G07 cached 10 m parts -> E6 100 m bank ------------------------------------------------
    from .features import PARTS_GLOB, _is_temporary

    parts, temp = {}, []
    for r in roots:
        for p in sorted(Path(r).glob(PARTS_GLOB)):
            if _is_temporary(p):
                temp.append(p.name)
                continue
            parts.setdefault(p.stem.rsplit("_", 1)[0], p)
    bad = []
    for tile, p in list(parts.items()):
        try:
            h = _raster_header(p)
            if h["count"] != 1 or not 5 <= abs(h["res"][0]) <= 25:
                bad.append(f"{p.name}: {h['count']} bands, res {h['res']}")
        except Exception as e:
            bad.append(f"{p.name}: {type(e).__name__}")
    n_ok = len(parts) - len(bad)
    _row(rows, "dem_cached_10m_parts", AVAILABLE if n_ok else BLOCKED, PARTS_GLOB,
         f"{n_ok} readable parts" + (f"; unreadable {bad[:3]}" if bad else "") +
         (f"; temporary files ignored {temp[:3]}" if temp else "") +
         ("" if n_ok else "; E6 never starts native 1 m processing: run the G07 coarse DEM build first"),
         "" if n_ok else "prepare")
    if bad:
        _row(rows, "dem_parts_integrity", BLOCKED, "", f"{len(bad)} unreadable part(s) would fail the DEM stage: {bad[:3]}",
             "prepare")
    max_tiles = int(cfg.get("e6.max_cached_dem_tiles", 32))
    if len(parts) > max_tiles:
        _row(rows, "dem_parts_limit", AVAILABLE, "", f"{len(parts)} parts, only {max_tiles} used (e6.max_cached_dem_tiles)")
    cache = [p for r in roots for p in Path(r).glob("outputs/e6_pipeline/cache/*/dem_features_100m.tif")]
    _row(rows, "dem_features_100m", AVAILABLE if cache else DERIVABLE, cache[0] if cache else "outputs/e6_pipeline/cache",
         "an existing bank is used only if its fingerprint (parts + extractor code) matches" if cache else
         "built from the cached parts at prepare (minutes per part)")

    # -- E1 warm-start checkpoints ------------------------------------------------------------------
    warm = Path(str((model.get("overrides") or {}).get("e6_warm_dir", "")))
    folds = list(range(bc.n_folds))
    if not str(warm) or not warm.is_dir():
        _row(rows, "e1_checkpoints", DERIVABLE, warm, "warm-start folder absent: every unit trains its baseline from "
                                                      "scratch (no E1 parity)")
    else:
        for unit in [f"cv_fold{k}" for k in folds] + [f"disc{d}" for d in range(bc.n_discovery_folds)]:
            p = warm / f"{unit}_model.pt"
            if not p.is_file():
                _row(rows, f"e1_{unit}", DERIVABLE, p, "absent: this unit trains its baseline from scratch")
                continue
            try:
                import torch

                ck = torch.load(p, map_location="cpu", weights_only=False)
                sp = ck.get("spec") or {}
                g = [list(v) for k, v in (sp.get("gsn_groups") or sp.get("groups") or {}).items() if k != "context"]
                want = [list(groups.get("terrain", [])), list(groups.get("subsurface", groups.get("potential", [])))]
                ok = "state_dict" in ck and g == want
                _row(rows, f"e1_{unit}", AVAILABLE if ok else BLOCKED, p,
                     "schema matches E6 terrain / potential groups" if ok else
                     f"E1 groups {[len(x) for x in g]} != E6 {[len(x) for x in want]} (order or channels): warm start "
                     "would be refused", "" if ok else "train")
            except Exception as e:
                _row(rows, f"e1_{unit}", BLOCKED, p, f"unreadable: {type(e).__name__}: {e}", "train")

    # -- E6 experiment state (resume) ---------------------------------------------------------------
    exp_dir = _first(roots, f"experiments/{exp_id}")
    if exp_dir is None:
        _row(rows, "e6_experiment", DERIVABLE, f"experiments/{exp_id}", "new experiment (nothing to resume)")
    else:
        ck = exp_dir / "checkpoints"
        final = sorted(p.name for p in ck.glob("*_model.pt"))
        resume = sorted(p.name for p in ck.glob("*_latest.pt")) + sorted(p.name for p in ck.glob("*_ep*.pt"))
        temp_ck = [p.name for p in ck.glob("*") if _is_temporary(p)]
        sub = exp_dir / "submissions"
        subs = sorted(p.name for p in sub.glob("submission_e6_*.tif")) if sub.is_dir() else []
        _row(rows, "e6_experiment", AVAILABLE, exp_dir, f"final models {final}; resume files {resume[:6]}; "
                                                       f"incomplete writes ignored {temp_ck[:3]}; submissions {subs}")

    # -- runtime ------------------------------------------------------------------------------------
    if check_gpu:
        try:
            import torch

            cuda = torch.cuda.is_available()
            _row(rows, "cuda_gpu", AVAILABLE if cuda else BLOCKED, "",
                 f"{torch.cuda.get_device_name(0)} torch {torch.__version__}" if cuda else
                 "no CUDA device: Runtime -> Change runtime type -> GPU", "" if cuda else "train")
        except ImportError:
            _row(rows, "cuda_gpu", BLOCKED, "", "torch not importable", "train")
    for mod in ("rasterio", "scipy", "yaml", "pyarrow"):
        try:
            __import__(mod)
            _row(rows, f"package_{mod}", AVAILABLE)
        except ImportError:
            _row(rows, f"package_{mod}", BLOCKED, "", "pip install -e .[colab] failed?", "prepare")
    free = shutil.disk_usage(local if local.exists() else Path("/")).free / 2**30
    _row(rows, "local_disk", AVAILABLE if free > 20 else BLOCKED, local, f"{free:.0f} GiB free",
         "" if free > 20 else "prepare")
    return _finish(rows, verbose)


def _finish(rows, verbose):
    T = pd.DataFrame(rows, columns=["artifact", "status", "path", "detail", "blocks"])
    blocked = T[T.status == BLOCKED]
    T.attrs["blocked"] = list(blocked.artifact)
    if verbose:
        print(f"[E6 PREFLIGHT] {int((T.status == AVAILABLE).sum())} available, {int((T.status == DERIVABLE).sum())} "
              f"derivable, {len(blocked)} BLOCKED")
        for r in blocked.itertuples():
            print(f"  BLOCKED {r.artifact}: {r.detail} [{r.path}] -> stops {r.blocks or 'the run'}")
        if not len(blocked):
            print("  no blocker found in the files: READY FOR RUNTIME PREFLIGHT (stage checks still apply)")
    return T
