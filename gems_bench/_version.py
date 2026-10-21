"""Package version and source revision (recorded in every experiment manifest)."""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

__version__ = "0.5.0"


@lru_cache(maxsize=1)
def git_commit() -> str:
    """Commit of the installed source: pip VCS metadata (``pip install git+...@sha``), else ``git`` in the
    checkout (suffix ``+dirty`` with uncommitted changes), else ``"unknown"``."""
    try:
        from importlib.metadata import distribution

        txt = distribution("gems-fault-discovery").read_text("direct_url.json")
        if txt:
            vcs = json.loads(txt).get("vcs_info", {})
            if vcs.get("commit_id"):
                return vcs["commit_id"]
    except Exception:
        pass
    here = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def provenance() -> dict:
    return {"package": "gems-fault-discovery", "version": __version__, "git_commit": git_commit()}
