"""G07 artifacts, manifests and caches - the G06 machinery (``g06.persistence``) under the G07 names.

A cached stage is reused only when its manifest is COMPLETE, the fingerprint (config section + upstream
fingerprints + source identities + module source hash) matches and every required output is readable with its
columns; a successful output replaced by another fingerprint is moved to ``history/`` first. DEM tile products are
cached per tile and per DEM-processing fingerprint so a Colab disconnect never recomputes a finished tile.
"""
from __future__ import annotations

from pathlib import Path

from ..g06 import persistence as _P

# G06 machinery re-exported under the G07 names
BLOCKED_MISSING, BLOCKED_UPSTREAM, COMPLETE, FAILED = _P.BLOCKED_MISSING, _P.BLOCKED_UPSTREAM, _P.COMPLETE, _P.FAILED
NOT_REQUESTED, NOT_RUN, REUSED, SKIPPED, VALID = _P.NOT_REQUESTED, _P.NOT_RUN, _P.REUSED, _P.SKIPPED, _P.VALID
Workspace, archive_outputs, artifact_fingerprint, check_outputs = (_P.Workspace, _P.archive_outputs,
                                                                   _P.artifact_fingerprint, _P.check_outputs)
code_identity, load_cached_stage, read_json, read_table = (_P.code_identity, _P.load_cached_stage, _P.read_json,
                                                           _P.read_table)
sha256_file, stable_hash, write_artifact_inventory = _P.sha256_file, _P.stable_hash, _P.write_artifact_inventory
write_json, write_table = _P.write_json, _P.write_table

BLOCKED_INPUT = "BLOCKED_INPUT"
NO_GO = "COMPLETE_NO_GO"          # a decision stage that ran correctly and decided against continuing


def atomic_save(path, writer) -> Path:
    return _P.atomic_write(path, writer)


def save_manifest(root, stage: str, status: str, fingerprint: str | None, outputs, info: dict | None = None,
                  **extra) -> dict:
    return _P.save_stage_manifest(root, stage, status, fingerprint, outputs, info, **extra)


def validate_cache(root, stage: str, fingerprint: str, outputs, schema_hash: str | None = None,
                   grid: dict | None = None) -> tuple[bool, list[str]]:
    return _P.validate_cached_stage(root, stage, fingerprint, outputs, schema_hash, grid)


def sync_artifacts(ws: Workspace, rels=None) -> None:
    """Push to Drive (never fatal: local outputs stay intact if Drive is unavailable)."""
    try:
        ws.push(rels)
    except Exception as e:
        print(f"[g07] Drive push failed ({type(e).__name__}: {e}); local outputs are intact")


def source_fingerprint(path, cache_file) -> dict:
    """sha256 of a source file, cached by (path, size, mtime) so large rasters are hashed once."""
    p = Path(path)
    st = p.stat()
    c = read_json(cache_file) or {}
    k = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    if k not in c:
        c[k] = sha256_file(p)
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        write_json(cache_file, c)
    return {"bytes": int(st.st_size), "sha256": c[k], "path": str(p)}
