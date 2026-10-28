"""Artifact inventory, selective staging and fingerprints (notebook 00; needs no GEMS pipeline object).

``inventory(drive_root, cfg)`` inspects every configured artifact on Drive. Statuses:
    OK             the expected number of files exists (sha256 up to ``hash_max_mb``, else size + mtime)
    MISSING        nothing matches the pattern (or its alternates)
    COUNT_MISMATCH a different number of files than ``expect_count``
    AMBIGUOUS      a single-object pattern (an experiment directory) matches several objects
    ID_CHANGED     the matched experiment id differs from the id seen in the last recorded run (``expected_id``);
                   blocking under ``strict_provenance`` (a different model would silently replace B0 / B1)
A required artifact with any status other than OK blocks Milestone 1 (``blocking``). The inventory fingerprint
(id + hashes) keys every downstream stage: a changed input invalidates the cached stages instead of mixing them.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import BLOCKED, COMPLETE, artifact_path

OK, MISSING, COUNT, AMBIG, IDCH = "OK", "MISSING", "COUNT_MISMATCH", "AMBIGUOUS", "ID_CHANGED"


def file_digest(p: Path, max_mb: float) -> str:
    """sha256 when the file is at most ``max_mb``, otherwise ``size:<bytes>:mtime:<int>`` (labelled, never mixed)."""
    p = Path(p)
    st = p.stat()
    if st.st_size > max_mb * 2**20:
        return f"size:{st.st_size}:mtime:{int(st.st_mtime)}"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def _matches(root: Path, pattern: str) -> list[Path]:
    return sorted(Path(x) for x in glob.glob(str(Path(root) / pattern)))


def experiment_ids(paths: list[Path], root: Path) -> list[str]:
    """``experiments/<id>/...`` -> sorted distinct ids."""
    ids = set()
    for p in paths:
        parts = Path(p).relative_to(root).parts
        if len(parts) > 1 and parts[0] == "experiments":
            ids.add(parts[1])
    return sorted(ids)


def classify(entry: dict, files: list[Path], ids: list[str], strict: bool) -> tuple[str, str]:
    """(status, note) of one artifact (pure)."""
    if not files:
        return MISSING, "no file matches"
    if entry.get("expected_id") is not None or (ids and entry.get("expect_count") == 1):
        if len(ids) > 1:
            return AMBIG, f"several experiments match: {ids}"
    exp = entry.get("expected_id")
    if exp and ids and ids != [exp]:
        return (IDCH if strict else OK), f"experiment {ids} differs from the recorded {exp}"
    n = entry.get("expect_count")
    if n is not None and len(files) != int(n):
        return COUNT, f"{len(files)} files, expected {n}"
    return OK, ""


def inventory(drive_root: Path, cfg: dict, strict: bool = True) -> pd.DataFrame:
    root = Path(drive_root)
    rows = []
    for a in cfg["artifacts"]:
        pat = artifact_path(a, cfg)
        files, used = _matches(root, pat), pat
        for alt in a.get("alternates") or []:
            if files:
                break
            files, used = _matches(root, alt), alt
        files = [f for f in files if f.is_file()] if not Path(root / used).is_dir() else files
        ids = experiment_ids(files, root)
        st, note = classify(a, files, ids, strict)
        hashes = {}
        for f in files:
            if f.is_file():
                hashes[f.relative_to(root).as_posix()] = file_digest(f, float(cfg["hash_max_mb"]))
            elif f.is_dir():
                hashes[f.relative_to(root).as_posix()] = "dir:" + ",".join(sorted(x.name for x in f.iterdir()))
        rows.append({"id": a["id"], "pattern": used, "required": bool(a.get("required")), "staged": bool(a.get("staged")),
                     "owner": a.get("owner", ""), "status": st, "note": note, "n_files": len(files),
                     "experiment_ids": ";".join(ids), "bytes": int(sum(f.stat().st_size for f in files if f.is_file())),
                     "files": json.dumps(sorted(hashes)), "hashes": json.dumps(hashes, sort_keys=True)})
    return pd.DataFrame(rows)


def blocking(inv: pd.DataFrame) -> list[str]:
    bad = inv[inv.required.astype(bool) & (inv.status != OK)] if len(inv) else inv
    return [f"{r.id}: {r.status} ({r.note or r.pattern})" for r in bad.itertuples()]


def summary(inv: pd.DataFrame) -> dict:
    b = blocking(inv)
    return {"status": BLOCKED if b else COMPLETE, "blocking": b,
            "counts": inv.status.value_counts().to_dict() if len(inv) else {},
            "optional_missing": list(inv[~inv.required.astype(bool) & (inv.status != OK)].id) if len(inv) else [],
            "fingerprint": fingerprint(inv)}


def fingerprint(inv: pd.DataFrame) -> str:
    """Hash of (artifact id, file hashes): changes whenever any inspected input changes."""
    items = sorted((r.id, r.hashes) for r in inv.itertuples()) if len(inv) else []
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]


def staging_plan(inv: pd.DataFrame) -> list[str]:
    """Drive-relative files to copy to local disk: staged artifacts only (never the checkpoint directories)."""
    out = []
    for r in inv[inv.staged.astype(bool) & (inv.status == OK)].itertuples() if len(inv) else []:
        out += json.loads(r.files)
    return sorted(set(out))


def environment_record(head: str, requested_ref: str) -> dict:
    import platform
    import time

    from ..p6.gpu_runtime import dependency_record, memory_report

    return {"requested_ref": requested_ref, "resolved_commit": head, "python": platform.python_version(),
            "dependencies": dependency_record(), "memory": memory_report(),
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}


def run_environment(CONFIG: dict, head: str, m1_config: dict | None = None) -> dict:
    """Notebook 00: environment record, Drive inventory, selective staging (+ verification), Drive push.
    Needs no GEMS pipeline: only the mounted Drive and the cloned repository."""
    from ..artifacts.atomic import atomic_save, atomic_write_json
    from ..artifacts.sync import DriveSync
    from .config import load_m1_config

    cfg = load_m1_config(m1_config)
    drive, local = Path(CONFIG["drive_root"]), Path(CONFIG["local_root"])
    rel_root = cfg["output_dir"]
    root = local / rel_root
    env = environment_record(head, CONFIG.get("ref", ""))
    atomic_write_json(root / "environment" / "environment.json", env)
    inv = inventory(drive, cfg, bool(CONFIG.get("strict_provenance", True)))
    summ = summary(inv)
    atomic_save(root / "artifact_inventory.csv", lambda t: inv.to_csv(t, index=False))
    atomic_write_json(root / "artifact_inventory.json", {"summary": summ, "artifacts": inv.to_dict("records"),
                                                         "drive_root": str(drive)})
    sync = DriveSync(local, drive)
    plan = staging_plan(inv)
    prev = {}
    pm = root / "staging_manifest.json"
    if pm.exists():
        prev = json.loads(pm.read_text(encoding="utf-8"))
    if CONFIG.get("reuse_cached", True) and prev.get("inventory_fp") == summ["fingerprint"]:
        staged = pd.DataFrame(prev.get("verification") or [])
        note = "inventory fingerprint unchanged: staging reused"
    else:
        sync.stage_in(plan)
        staged = verify_staged(local, plan, inv, float(cfg["hash_max_mb"]))
        note = "staged"
    bad = list(staged[~staged.match.astype(bool)].file) if len(staged) else []
    man = {"inventory_fp": summ["fingerprint"], "files": plan, "n_files": len(plan), "verification":
           staged.to_dict("records"), "mismatch": bad, "note": note,
           "not_staged": "CNN checkpoint directories and prediction rasters stay on Drive (the pipeline stages "
                         "the maps it needs; checkpoints are never needed)"}
    atomic_write_json(pm, man)
    if bad:
        summ = {**summ, "status": "BLOCKED", "blocking": summ["blocking"] + [f"staged copy differs: {bad}"]}
        atomic_write_json(root / "artifact_inventory.json", {"summary": summ, "artifacts": inv.to_dict("records"),
                                                             "drive_root": str(drive)})
    from .._version import provenance

    atomic_write_json(root / "manifests" / "artifacts.json",
                      {"stage": "artifacts", "status": summ["status"], "reasons": summ["blocking"],
                       "fingerprint": summ["fingerprint"], "code": provenance(), "environment": env})
    sync.push([rel_root])
    return {"summary": summ, "inventory": inv, "staging": man, "environment": env}


def verify_staged(local_root: Path, rels: list[str], inv: pd.DataFrame, max_mb: float) -> pd.DataFrame:
    """Local copy vs the inventory hash, file by file."""
    want = {}
    for r in inv.itertuples():
        want.update(json.loads(r.hashes))
    rows = []
    for rel in rels:
        p = Path(local_root) / rel
        got = file_digest(p, max_mb) if p.exists() else None
        rows.append({"file": rel, "local_exists": p.exists(), "expected": want.get(rel), "local": got,
                     "match": got is not None and got == want.get(rel)})
    return pd.DataFrame(rows)
