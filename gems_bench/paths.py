"""Project locations for GEMS 2.0: code checkout, data / workspace root, and the Google Drive mirror.

Three roots, deliberately separate
----------------------------------
``code``    the Git checkout (configs/, notebooks/, the package). ``$GEMS_CODE_ROOT`` or the directory above
            this package when it contains ``configs/`` (editable install / clone).
``root``    the workspace that holds data and artifacts. Locally this is the original project folder
            (raw GeoTIFFs at its top, legacy ``outputs/phase*``); on Colab it is a fast local copy on
            ``/content`` (``$GEMS_ROOT``). Raw files are accepted flat or under ``data/raw/<source>/``
            (``gems_eda.config.resolve_raw``).
``drive``   the persistent mirror on Google Drive (``$GEMS_DRIVE_ROOT``, default ``MyDrive/GEMS_2``).
            Same relative layout as ``root`` so that synchronisation is a subtree mirror
            (``gems_bench.artifacts.sync``).

Workspace layout (identical under ``root`` and ``drive``)::

    data/raw/{geodawn,faults,dem_1m}/  data/{external,intermediate,processed,feature_store}/
    outputs/{eda,benchmark,phase3,phase4}/     legacy Phase 1-4 artifacts (read-only from Phase 5 on)
    experiments/<EXPERIMENT_ID>/               Phase 5+ experiments (config.yaml, manifest.json, ...)
    submissions/  model_registry/  cached_artifacts/  environment/  reports/
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from gems_eda.config import find_project_root

ENV_ROOT = "GEMS_ROOT"
ENV_CODE = "GEMS_CODE_ROOT"
ENV_DRIVE = "GEMS_DRIVE_ROOT"
DEFAULT_DRIVE = "/content/drive/MyDrive/GEMS_2"

WORKSPACE_DIRS = (
    "data/raw/geodawn", "data/raw/faults", "data/raw/dem_1m", "data/external", "data/intermediate",
    "data/processed", "data/feature_store", "experiments", "submissions", "model_registry",
    "cached_artifacts", "environment", "reports",
)


def code_root() -> Path:
    """Git checkout holding ``configs/`` (env override, then the package's parent, then cwd ancestors)."""
    env = os.environ.get(ENV_CODE)
    if env and (Path(env) / "configs").is_dir():
        return Path(env).resolve()
    here = Path(__file__).resolve().parents[1]
    if (here / "configs").is_dir():
        return here
    cwd = Path.cwd().resolve()
    for c in (cwd, *cwd.parents):
        pp = c / "pyproject.toml"
        if pp.exists() and "gems-fault-discovery" in pp.read_text(encoding="utf-8", errors="ignore"):
            return c
    raise FileNotFoundError(f"code checkout with configs/ not found: set ${ENV_CODE} to the cloned repository")


def config_path(*parts: str) -> Path:
    return code_root().joinpath("configs", *parts)


def drive_root(default: str = DEFAULT_DRIVE) -> Path | None:
    p = Path(os.environ.get(ENV_DRIVE, default))
    return p if p.exists() else None


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    code: Path
    drive: Path | None = None

    @classmethod
    def discover(cls, root: Path | str | None = None, code: Path | str | None = None,
                 drive: Path | str | None = None) -> "ProjectPaths":
        r = find_project_root(Path(root) if root else None)
        c = Path(code).resolve() if code else code_root()
        d = Path(drive) if drive else drive_root()
        return cls(r, c, d)

    # -- workspace ------------------------------------------------------
    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def submissions(self) -> Path:
        return self.root / "submissions"

    @property
    def registry(self) -> Path:
        return self.root / "model_registry"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def feature_store(self) -> Path:
        return self.root / "data" / "feature_store"

    @property
    def dem_raw(self) -> Path:
        return self.root / "data" / "raw" / "dem_1m"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def legacy(self) -> Path:
        return self.root / "outputs"

    def ensure(self) -> "ProjectPaths":
        for d in WORKSPACE_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        return self

    def on_drive(self, local: Path) -> Path | None:
        """Drive counterpart of a path under ``root`` (None without a Drive mirror)."""
        if self.drive is None:
            return None
        return self.drive / Path(local).resolve().relative_to(self.root.resolve())

    def rel(self, p: Path) -> str:
        return Path(p).resolve().relative_to(self.root.resolve()).as_posix()

    def as_dict(self) -> dict:
        return {"root": str(self.root), "code": str(self.code), "drive": str(self.drive) if self.drive else None}
