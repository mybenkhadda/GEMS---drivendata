"""Google Drive <-> Colab local-disk synchronisation (subtree mirror, verified, resumable).

Model: ``local_root`` (fast, ephemeral: ``/content/gems_work``) and ``drive_root`` (persistent:
``/content/drive/MyDrive/GEMS_2``) share one relative layout (``gems_bench.paths``).

* ``stage_in(rels)``  Drive -> local: raw data, feature caches, legacy model maps. Unchanged files are skipped.
* ``push(rels)``      local -> Drive: experiment folders, registry, submissions. Each file goes through
                      ``atomic_copy`` (``.partial`` + verification + rename), so an interrupted push never
                      corrupts the previous Drive copy; re-running a push resumes where it stopped.
* ``safe_delete_local(rels)``  removes local files only when a verified Drive copy exists.

Unchanged-file detection uses a ledger (``environment/sync_ledger.json`` under the local root, mirrored to
Drive) keyed by relative path: (size, mtime_ns) of the source at the last verified copy. This avoids
re-reading Drive files (slow on the FUSE mount) and repeated Drive API operations on thousands of files.

Nothing is ever deleted on Drive by this module.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .atomic import atomic_copy, atomic_write_json, is_temporary, sha256_file

LEDGER = "environment/sync_ledger.json"
SHA_SUFFIXES = (".pt", ".ckpt", ".tif", ".parquet")     # verified by checksum when verify="auto"


@dataclass
class DriveSync:
    local_root: Path
    drive_root: Path
    verify: str = "auto"            # "size" | "sha256" | "auto" (sha256 for checkpoints / rasters / parquet)
    exclude: tuple[str, ...] = ("__pycache__", ".ipynb_checkpoints")
    _ledger: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.local_root, self.drive_root = Path(self.local_root), Path(self.drive_root)
        p = self.local_root / LEDGER
        if p.exists():
            self._ledger = json.loads(p.read_text(encoding="utf-8"))
        else:
            d = self.drive_root / LEDGER
            self._ledger = json.loads(d.read_text(encoding="utf-8")) if d.exists() else {"in": {}, "out": {}}

    # -- public -----------------------------------------------------------
    def stage_in(self, rels: Iterable[str], verify: str | None = None) -> pd.DataFrame:
        """Copy Drive files / directories to local disk (skip unchanged); returns one row per file."""
        return self._mirror(rels, self.drive_root, self.local_root, "in", verify)

    def push(self, rels: Iterable[str], verify: str | None = None) -> pd.DataFrame:
        """Copy local files / directories to Drive (skip unchanged, atomic, verified)."""
        return self._mirror(rels, self.local_root, self.drive_root, "out", verify)

    def is_synced(self, local_path: Path) -> bool:
        """True if ``local_path`` has a Drive copy recorded in the ledger with the current size / mtime."""
        p = Path(local_path)
        rel = p.resolve().relative_to(self.local_root.resolve()).as_posix()
        e = self._ledger.get("out", {}).get(rel)
        dst = self.drive_root / rel
        if not e or not p.exists() or not dst.exists():
            return False
        st = p.stat()
        return e["size"] == st.st_size and e["mtime_ns"] == st.st_mtime_ns and dst.stat().st_size == st.st_size

    def safe_delete_local(self, rels: Iterable[str]) -> list[str]:
        """Delete local files whose verified Drive copy exists (sha256 compared); returns deleted paths."""
        gone = []
        for f in self._files(self.local_root, rels):
            rel = f.relative_to(self.local_root).as_posix()
            dst = self.drive_root / rel
            if dst.exists() and dst.stat().st_size == f.stat().st_size and sha256_file(dst) == sha256_file(f):
                f.unlink()
                gone.append(rel)
        return gone

    def plan(self, rels: Iterable[str], direction: str = "out") -> pd.DataFrame:
        """Dry run: which files would be copied and how many bytes."""
        src_root = self.local_root if direction == "out" else self.drive_root
        rows = []
        for f in self._files(src_root, rels):
            rel = f.relative_to(src_root).as_posix()
            rows.append({"rel": rel, "bytes": f.stat().st_size, "changed": self._changed(direction, rel, f)})
        return pd.DataFrame(rows)

    # -- internals ----------------------------------------------------------
    def _files(self, root: Path, rels: Iterable[str]):
        for rel in rels:
            p = root / rel
            if p.is_file():
                if not is_temporary(p):
                    yield p
            elif p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and not is_temporary(f) and not any(x in f.parts for x in self.exclude):
                        yield f

    def _changed(self, direction: str, rel: str, f: Path) -> bool:
        e = self._ledger.setdefault(direction, {}).get(rel)
        st = f.stat()
        dst = (self.drive_root if direction == "out" else self.local_root) / rel
        if not e or not dst.exists():
            return True
        return not (e["size"] == st.st_size and e["mtime_ns"] == st.st_mtime_ns and dst.stat().st_size == st.st_size)

    def _mode(self, f: Path, verify: str | None) -> str:
        v = verify or self.verify
        if v == "auto":
            return "sha256" if f.suffix in SHA_SUFFIXES else "size"
        return v

    def _mirror(self, rels, src_root: Path, dst_root: Path, direction: str, verify: str | None) -> pd.DataFrame:
        rows, t0 = [], time.perf_counter()
        rels = list(rels)
        try:
            for f in self._files(src_root, rels):
                rel = f.relative_to(src_root).as_posix()
                if not self._changed(direction, rel, f):
                    rows.append({"rel": rel, "action": "skip", "bytes": f.stat().st_size})
                    continue
                info = atomic_copy(f, dst_root / rel, self._mode(f, verify))
                st = f.stat()
                self._ledger.setdefault(direction, {})[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                                               "sha256": info.get("sha256"),
                                                               "synced": time.strftime("%Y-%m-%dT%H:%M:%S")}
                rows.append({"rel": rel, "action": "copy", "bytes": st.st_size, "verify": info["verify"]})
        finally:
            self._save_ledger()
        df = pd.DataFrame(rows, columns=["rel", "action", "bytes", "verify"])
        n = int((df.action == "copy").sum()) if len(df) else 0
        mb = df.loc[df.action == "copy", "bytes"].sum() / 2**20 if len(df) else 0.0
        print(f"[sync {direction}] {n} copied ({mb:.1f} MiB), {len(df) - n} unchanged, "
              f"{time.perf_counter() - t0:.0f}s: {rels}")
        return df

    def _save_ledger(self) -> None:
        atomic_write_json(self.local_root / LEDGER, self._ledger)
        if self.drive_root.exists():
            try:
                atomic_write_json(self.drive_root / LEDGER, self._ledger)
            except OSError as e:                         # Drive hiccup: local ledger is still authoritative
                print(f"[sync] could not mirror the ledger to Drive: {e}")


def local_free_gb(path: Path = Path("/content")) -> float:
    import shutil

    p = path if Path(path).exists() else Path.cwd()
    return shutil.disk_usage(p).free / 2**30


def tree_size(root: Path, rels: Iterable[str]) -> pd.DataFrame:
    """Bytes per top-level entry (for planning uploads / staging)."""
    rows = []
    for rel in rels:
        p = Path(root) / rel
        n = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0
        rows.append({"rel": rel, "exists": p.exists(), "gib": n / 2**30})
    return pd.DataFrame(rows)


def env_paths_for_colab(local_root: str = "/content/gems_work", drive_root: str = "/content/drive/MyDrive/GEMS_2",
                        code_root: str = "/content/GEMS") -> dict:
    """Set the three GEMS_* environment variables for this process (and child processes)."""
    os.environ["GEMS_ROOT"], os.environ["GEMS_DRIVE_ROOT"], os.environ["GEMS_CODE_ROOT"] = local_root, drive_root, code_root
    return {"GEMS_ROOT": local_root, "GEMS_DRIVE_ROOT": drive_root, "GEMS_CODE_ROOT": code_root}
