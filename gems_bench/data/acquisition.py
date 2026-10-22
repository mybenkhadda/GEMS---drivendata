"""One-time acquisition of the competition files and the USGS 1 m DEM URL list (resumable, verified, atomic).

Per file the order is fixed and never re-downloads a valid cached copy:

1. look on Drive (canonical path, then accepted alternates)  -> valid: stage to local, done
2. look on the local disk (a previous session / an archive)  -> valid: push to Drive, done
3. reuse a local ``.partial`` (HTTP Range resume)            -> download with retries
4. verify length (Content-Length / known release size) and structure (GeoTIFF opens / JSON parses)
5. atomic copy to Drive (``.partial`` + sha256 + rename), record in ``data/raw/acquisition_manifest.json``

URLs
----
DrivenData links are personal and expire. They are *never* written to disk or Git by this module: the
manifest records only the host and a sha256 of the URL. URLs are resolved per file from (first hit wins)
an explicit ``urls`` mapping passed at runtime, environment variables, then Colab Secrets
(``GEMS_URL_FEATURES``, ``GEMS_URL_FAULTS``, ``GEMS_URL_SUBMISSION``, ``GEMS_URL_DEM_JSON``).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from ..artifacts.atomic import atomic_copy, atomic_write_json
from ..artifacts.validation import Check, check_geotiff, check_json

MANIFEST = "data/raw/acquisition_manifest.json"


@dataclass(frozen=True)
class RawFile:
    key: str                              # features | faults | submission | dem_json
    rel: str                              # canonical workspace path
    kind: str                             # tif | json
    required: bool = True
    alternates: tuple[str, ...] = ()      # other accepted locations (flat layout, earlier uploads)
    expected_bytes: int | None = None     # known release size (a mismatch is reported, not fatal)
    count: int | None = None              # expected band count (GeoTIFF)
    secret: str = ""
    note: str = ""


CORE_FILES = (
    RawFile("features", "data/raw/geodawn/gems-geodawn-numerical-features.tif", "tif",
            alternates=("gems-geodawn-numerical-features.tif",), expected_bytes=418_912_844, count=19,
            secret="GEMS_URL_FEATURES", note="19-band GeoDAWN stack, 100 m, EPSG:32611"),
    RawFile("faults", "data/raw/faults/existing_faults.tif", "tif",
            alternates=("existing_faults.tif", "data/raw/geodawn/existing_faults.tif"), expected_bytes=425_830,
            secret="GEMS_URL_FAULTS", note="mapped faults, value >= 1"),
    RawFile("submission", "data/raw/geodawn/example_submission.tif", "tif",
            alternates=("data/raw/example_submission.tif", "example_submission.tif"), expected_bytes=1_599_597,
            secret="GEMS_URL_SUBMISSION", note="submission template (grid / dtype reference)"),
    RawFile("dem_json", "data/raw/dem_1m/tnm_urls.json", "json", required=False,
            alternates=("data/raw/tnm_items.json", "data/raw/dem_1m/tnm_items.json", "tnm_items.json"),
            secret="GEMS_URL_DEM_JSON", note="USGS TNM 1 m DEM GeoTIFF URL list (metadata only)"),
)
BY_KEY = {f.key: f for f in CORE_FILES}


# ---------------------------------------------------------------------------
# URL resolution (never persisted)
# ---------------------------------------------------------------------------
def _colab_secret(name: str) -> str:
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name) or ""
    except Exception:                     # not on Colab, secret missing, or notebook access not granted
        return ""


def resolve_url(f: RawFile, urls: dict | None = None, use_secrets: bool = True) -> tuple[str, str]:
    """(url, source) for one file; ("", "") when none is configured. ``source`` never contains the URL."""
    u = (urls or {}).get(f.key, "") or ""
    if u:
        return u.strip(), "runtime"
    if f.secret and os.environ.get(f.secret):
        return os.environ[f.secret].strip(), f"env:{f.secret}"
    if use_secrets and f.secret:
        s = _colab_secret(f.secret)
        if s:
            return s.strip(), f"colab_secret:{f.secret}"
    return "", ""


def url_record(url: str) -> dict:
    """What may be stored about a URL: host and sha256 (never the query string / signature)."""
    if not url:
        return {}
    return {"host": urlparse(url).netloc, "url_sha256": hashlib.sha256(url.encode()).hexdigest()[:16]}


def mask(url: str) -> str:
    if not url:
        return ""
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/.../{Path(p.path).name}"


# ---------------------------------------------------------------------------
# validation / download
# ---------------------------------------------------------------------------
def validate_file(f: RawFile, path: Path) -> Check:
    if f.kind == "tif":
        return check_geotiff(path, count=f.count, expected_bytes=f.expected_bytes, min_bytes=1024)
    c = check_json(path)
    if c.ok:
        try:
            from ..dem.inventory import _records

            n = len(_records(json.loads(Path(path).read_text(encoding="utf-8"))))
            c.info["records"] = n
            if n == 0:
                c.status, c.reason = "invalid", "no URL records"
        except Exception as e:
            c.status, c.reason = "invalid", f"unsupported DEM list structure: {e}"
    return c


def remote_size(url: str, timeout: int = 60) -> int | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def download(url: str, dst: Path, *, expected_bytes: int | None = None, retries: int = 5, chunk: int = 8 << 20,
             timeout: int = 120, progress: bool = True) -> dict:
    """Resumable HTTP download into ``dst`` through ``dst.partial`` (Range requests, retries with backoff).

    Returns {"status": "downloaded" | "failed", "bytes", "attempts", "error"}. ``dst`` is only created
    after the size check; content validation is the caller's job (``validate_file``)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".partial")
    total = remote_size(url)              # the known release size is only reported (a new release may differ)
    if total is None and expected_bytes:
        print(f"  [{dst.name}] no Content-Length; known release size {expected_bytes} bytes is checked after download")
    err = ""
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if total and have == total:
            break
        if total and have > total:                         # corrupt partial (different release): restart
            part.unlink()
            have = 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "ab" if have else "wb") as fh:
                if have and getattr(r, "status", 200) != 206:   # server ignored Range: restart from zero
                    fh.seek(0)
                    fh.truncate()
                    have = 0
                t0, last = time.time(), 0.0
                while True:
                    b = r.read(chunk)
                    if not b:
                        break
                    fh.write(b)
                    have += len(b)
                    if progress and time.time() - last > 15:
                        last = time.time()
                        pct = f"{100 * have / total:.0f}%" if total else f"{have / 2**20:.0f} MiB"
                        print(f"  [{dst.name}] {pct} ({have / 2**20 / max(time.time() - t0, 1e-6):.1f} MiB/s)")
            if total and part.stat().st_size != total:
                raise IOError(f"incomplete: {part.stat().st_size} / {total} bytes")
            break
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  [{dst.name}] attempt {attempt}/{retries} failed: {err}")
            time.sleep(min(60, 5 * attempt))
    else:
        return {"status": "failed", "bytes": part.stat().st_size if part.exists() else 0, "attempts": retries,
                "error": err}
    os.replace(part, dst)
    return {"status": "downloaded", "bytes": dst.stat().st_size, "attempts": attempt, "error": ""}


# ---------------------------------------------------------------------------
# acquisition
# ---------------------------------------------------------------------------
@dataclass
class Acquirer:
    local_root: Path
    drive_root: Path | None
    urls: dict = field(default_factory=dict)
    download_missing: bool = True
    use_secrets: bool = True
    verify: str = "sha256"

    def __post_init__(self) -> None:
        self.local_root = Path(self.local_root)
        self.drive_root = Path(self.drive_root) if self.drive_root else None

    # -- locating -----------------------------------------------------------
    def candidates(self, f: RawFile) -> list[str]:
        return [f.rel, *f.alternates]

    def find(self, f: RawFile, root: Path | None) -> tuple[str, Check] | None:
        """First *valid* copy of ``f`` under ``root`` (canonical path first)."""
        if root is None:
            return None
        bad = None
        for rel in self.candidates(f):
            p = root / rel
            if p.exists():
                c = validate_file(f, p)
                if c.ok:
                    return rel, c
                bad = bad or (rel, c)
        return bad

    # -- manifest -----------------------------------------------------------
    def _manifest(self) -> dict:
        for root in (self.local_root, self.drive_root):
            if root and (root / MANIFEST).exists():
                try:
                    return json.loads((root / MANIFEST).read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {"files": {}}

    def _save_manifest(self, man: dict) -> None:
        atomic_write_json(self.local_root / MANIFEST, man)
        if self.drive_root and self.drive_root.exists():
            try:
                atomic_write_json(self.drive_root / MANIFEST, man)
            except OSError as e:
                print(f"[acquire] manifest not mirrored to Drive: {e}")

    # -- one file -----------------------------------------------------------
    def acquire(self, f: RawFile) -> dict:
        row = {"key": f.key, "required": f.required, "canonical": f.rel, "action": "", "location": "",
               "status": "missing", "detail": "", "bytes": None}
        drive_hit = self.find(f, self.drive_root)
        if drive_hit and drive_hit[1].ok:                                  # 1. valid on Drive
            rel, c = drive_hit
            local = self.local_root / rel
            if not (local.exists() and local.stat().st_size == (self.drive_root / rel).stat().st_size):
                atomic_copy(self.drive_root / rel, local, "size")
                row["action"] = "staged_from_drive"
            else:
                row["action"] = "cached"
            return {**row, "location": rel, "status": "ok", "bytes": c.info.get("bytes"),
                    "detail": c.info.get("size_note", "")}
        local_hit = self.find(f, self.local_root)
        if local_hit and local_hit[1].ok:                                  # 2. valid locally -> persist
            rel, c = local_hit
            self._push(rel)
            return {**row, "location": rel, "status": "ok", "action": "pushed_to_drive", "bytes": c.info.get("bytes")}
        invalid = [h for h in (drive_hit, local_hit) if h and not h[1].ok]
        url, source = resolve_url(f, self.urls, self.use_secrets)
        if not url or not self.download_missing:                           # nothing more we can do
            why = "download disabled" if url else f"no URL configured (Colab secret {f.secret} or urls['{f.key}'])"
            if invalid:
                why = f"existing copy invalid ({invalid[0][1].reason}); {why}"
            return {**row, "status": "missing", "action": "none", "detail": why}
        dst = self.local_root / f.rel                                      # 3-4. download + verify
        print(f"[acquire] {f.key}: downloading {mask(url)} ({source})")
        res = download(url, dst, expected_bytes=f.expected_bytes)
        if res["status"] != "downloaded":
            return {**row, "status": "failed", "action": "download", "detail": res["error"]}
        c = validate_file(f, dst)
        if not c.ok:
            bad = dst.with_name(dst.name + f".invalid-{int(time.time())}")
            os.replace(dst, bad)                                           # keep for inspection, never reuse
            return {**row, "status": "failed", "action": "download", "detail": f"downloaded file invalid: {c.reason}"}
        self._push(f.rel)                                                  # 5. atomic, verified copy to Drive
        man = self._manifest()
        man.setdefault("files", {})[f.key] = {"rel": f.rel, "bytes": c.info.get("bytes"), "source": source,
                                              "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S"), **url_record(url)}
        self._save_manifest(man)
        return {**row, "status": "ok", "action": "downloaded", "location": f.rel, "bytes": c.info.get("bytes"),
                "detail": c.info.get("size_note", "")}

    def _push(self, rel: str) -> None:
        if self.drive_root is None or not self.drive_root.exists():
            return
        dst = self.drive_root / rel
        src = self.local_root / rel
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            return
        atomic_copy(src, dst, self.verify)

    def acquire_all(self, keys: tuple[str, ...] | None = None) -> pd.DataFrame:
        rows = []
        for f in CORE_FILES:
            if keys and f.key not in keys:
                continue
            try:
                rows.append(self.acquire(f))
            except Exception as e:                                         # report and continue with the next file
                rows.append({"key": f.key, "required": f.required, "canonical": f.rel, "status": "failed",
                             "action": "error", "detail": f"{type(e).__name__}: {str(e)[:200]}"})
        df = pd.DataFrame(rows)
        man = self._manifest()
        man["last_check"] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "status": {r["key"]: r["status"] for r in rows},
                             "location": {r["key"]: r.get("location", "") for r in rows}}
        self._save_manifest(man)
        return df


def inventory(roots: dict[str, Path | None]) -> pd.DataFrame:
    """Read-only view: where each core file is (canonical / alternate) and whether it validates, per root."""
    rows = []
    for f in CORE_FILES:
        for label, root in roots.items():
            if root is None:
                continue
            for rel in (f.rel, *f.alternates):
                p = Path(root) / rel
                if p.exists():
                    c = validate_file(f, p)
                    rows.append({"key": f.key, "root": label, "rel": rel, "canonical": rel == f.rel, "status": c.status,
                                 "reason": c.reason, "bytes": p.stat().st_size, "required": f.required})
    df = pd.DataFrame(rows, columns=["key", "root", "rel", "canonical", "status", "reason", "bytes", "required"])
    found = set(df.key) if len(df) else set()
    miss = [{"key": f.key, "root": "-", "rel": f.rel, "canonical": True, "status": "missing", "reason": "not found",
             "bytes": None, "required": f.required} for f in CORE_FILES if f.key not in found]
    return pd.concat([df, pd.DataFrame(miss)], ignore_index=True) if miss else df


def canonicalize_dem_json(local_root: Path, drive_root: Path | None) -> Path | None:
    """Copy an alternate DEM URL list (e.g. ``data/raw/tnm_items.json``) to the canonical path read by the
    DEM pilot (``data/raw/dem_1m/tnm_urls.json``). Never overwrites; returns the canonical local path."""
    f = BY_KEY["dem_json"]
    local = Path(local_root) / f.rel
    if local.exists():
        return local
    for root in (Path(local_root), Path(drive_root) if drive_root else None):
        if root is None:
            continue
        for rel in f.alternates:
            p = root / rel
            if p.exists() and validate_file(f, p).ok:
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, local)
                if drive_root and Path(drive_root).exists() and not (Path(drive_root) / f.rel).exists():
                    atomic_copy(local, Path(drive_root) / f.rel, "sha256")
                return local
    return None
