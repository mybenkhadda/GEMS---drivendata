"""Colab session helpers: Drive mount, environment report, what to stage from Drive to /content.

Typical session (notebooks/colab/00_setup_and_drive.ipynb)::

    from gems_bench.runtime import colab
    colab.mount_drive()
    env = colab.configure()            # GEMS_ROOT=/content/gems_work, GEMS_DRIVE_ROOT=.../GEMS_2, GEMS_CODE_ROOT=/content/GEMS
    colab.stage_core()                 # raw data, folds, splits, segments (skip unchanged)
    colab.environment_report()
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from ..artifacts.sync import DriveSync, env_paths_for_colab

LOCAL_ROOT = "/content/gems_work"
DRIVE_ROOT = "/content/drive/MyDrive/GEMS_2"
CODE_ROOT = "/content/GEMS"

# Always needed by load_context5 (small: ~0.5 GB with the raw GeoTIFFs)
CORE = (
    "data/raw/geodawn", "data/raw/faults",
    "outputs/benchmark/folds", "outputs/benchmark/datasets",
    "outputs/eda/manifest.json", "outputs/eda/datasets",
    "experiments",          # Phase-5 manifests (maps are staged per experiment)
    "model_registry",
)


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")


def mount_drive(path: str = "/content/drive") -> None:
    from google.colab import drive  # type: ignore

    drive.mount(path)


def configure(local_root: str = LOCAL_ROOT, drive_root: str = DRIVE_ROOT, code_root: str = CODE_ROOT) -> dict:
    Path(local_root).mkdir(parents=True, exist_ok=True)
    return env_paths_for_colab(local_root, drive_root, code_root)


def sync(local_root: str = LOCAL_ROOT, drive_root: str = DRIVE_ROOT) -> DriveSync:
    return DriveSync(Path(local_root), Path(drive_root))


def stage_core(extra: tuple[str, ...] = (), local_root: str = LOCAL_ROOT, drive_root: str = DRIVE_ROOT) -> pd.DataFrame:
    s = sync(local_root, drive_root)
    rels = [r for r in CORE + tuple(extra) if (Path(drive_root) / r).exists()]
    missing = [r for r in CORE if not (Path(drive_root) / r).exists()]
    if missing:
        print(f"[stage] not on Drive: {missing} - the master pipeline regenerates the derived ones "
              "(notebooks/colab/GEMS_MASTER_PIPELINE.ipynb, docs/ARTIFACT_RECOVERY.md)")
    return s.stage_in(rels)


def feature_cache_rels(ctx_or_names, phase2_cache="outputs/benchmark/cache/features",
                       phase4_cache="outputs/phase4/cache/features", phase5_cache="data/feature_store/phase5") -> list[str]:
    """Relative paths of the cached .npy files for a list of channel names (stage only what a run needs)."""
    from ..features import _fname

    names = ctx_or_names
    out = []
    for n in names:
        f = _fname(n)
        base = phase5_cache if n.startswith(("p5:", "dem:")) else phase4_cache if n.startswith("rs:") else phase2_cache
        out.append(f"{base}/{f}")
    return out


def stage_features(names, local_root: str = LOCAL_ROOT, drive_root: str = DRIVE_ROOT) -> pd.DataFrame:
    """Stage only the feature caches a run needs (unchanged files skipped by the ledger). The master pipeline
    does this automatically with ``features.stage: lazy`` (copy on first read)."""
    rels = [r for r in feature_cache_rels(names) if (Path(drive_root) / r).exists()]
    missing = sorted(set(feature_cache_rels(names)) - set(rels))
    if missing:
        print(f"[stage] {len(missing)} caches not on Drive (built on demand): {missing[:5]}")
    return sync(local_root, drive_root).stage_in(rels)


def model_map_rels(drive_root: str, exp_dirs: list[str]) -> list[str]:
    """oof.tif / pred.tif / config.json / run.json of legacy experiment folders (models/ excluded)."""
    rels = []
    for d in exp_dirs:
        for f in ("oof.tif", "pred.tif", "config.json", "run.json", "area_crossfit.csv", "metrics_fold.csv"):
            if (Path(drive_root) / d / f).exists():
                rels.append(f"{d}/{f}")
    return rels


def stage_legacy(drive_root: str = DRIVE_ROOT, local_root: str = LOCAL_ROOT, features: bool = True,
                 checkpoints: bool = False) -> pd.DataFrame:
    """Stage what Phase-5 code reads from Phases 2-4: feature caches (~7 GB) and every legacy experiment's
    maps / configs (``models/`` only with ``checkpoints=True``, needed for full-raster inference of a zoo CNN)."""
    rels = []
    if features:
        rels += [r for r in ("outputs/benchmark/cache/features", "outputs/phase4/cache/features", "data/feature_store")
                 if (Path(drive_root) / r).exists()]
    for phase in ("benchmark", "phase3", "phase4"):
        base = Path(drive_root) / "outputs" / phase / "experiments"
        if base.exists():
            dirs = [f"outputs/{phase}/experiments/{p.name}" for p in sorted(base.iterdir()) if p.is_dir()]
            rels += model_map_rels(drive_root, dirs)
            if checkpoints:
                rels += [f"{d}/models" for d in dirs if (Path(drive_root) / d / "models").exists()]
    for extra in ("outputs/benchmark/ensembles", "outputs/phase4/experiments.yaml"):
        if (Path(drive_root) / extra).exists():
            rels.append(extra)
    return sync(local_root, drive_root).stage_in(rels)


def environment_report() -> pd.DataFrame:
    from .._version import provenance
    from ..training.gpu import detect

    g = detect()
    rows = {**provenance(), "python": sys.version.split()[0], "cuda_available": g.get("cuda"), "gpu": g.get("name"),
            "vram_gb": g.get("vram_gb"), "capability": g.get("capability")}
    try:
        import torch

        rows["torch"] = torch.__version__
        rows["cuda"] = torch.version.cuda
    except Exception:
        rows["torch"] = None
    for p in ("/content", DRIVE_ROOT):
        if os.path.exists(p):
            rows[f"free_gb:{p}"] = round(shutil.disk_usage(p).free / 2**30, 1)
    for k in ("GEMS_ROOT", "GEMS_DRIVE_ROOT", "GEMS_CODE_ROOT"):
        rows[k] = os.environ.get(k)
    return pd.DataFrame({"key": list(rows), "value": [str(v) for v in rows.values()]})
