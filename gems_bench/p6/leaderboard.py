"""P6 leaderboard bookkeeping: manual upload record + local vs public comparison. Nothing is ever uploaded, and the
public score never feeds a selection (it is a small spatial subset; the held-out zones remain the validation).

    lb.record(fg, "submission_p6_boost.tif", public_score=0.1234, date="2026-11-02", notes="first P6 boost file")
    lb.compare(fg)            # leaderboard/leaderboard_comparison.csv + leaderboard_report.md

``leaderboard/leaderboard_history.csv`` is append-only: a row is refused when the file on disk does not hash to its
manifest, when the same file already has a recorded score, or when the score is outside [0, 1].
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["submission_name", "file", "sha256", "candidate", "role", "model", "code_commit", "recipe_fp",
           "local_disc_official_dti", "local_disc_worst", "local_geo_dti", "objective", "public_lb_score", "date_utc",
           "recorded_utc", "postprocessing", "area_frac_eligible", "validation_status", "was_ready", "notes"]


# =============================================================================== pure
def make_row(man: dict, public_score: float, date: str, notes: str = "", name: str | None = None) -> dict:
    """One history row from a submission manifest (``submissions/<stem>_manifest.json``) and the typed score."""
    s = float(public_score)
    if not np.isfinite(s) or not 0.0 <= s <= 1.0:
        raise ValueError(f"public score {public_score!r} is not a DTI in [0, 1]")
    time.strptime(str(date), "%Y-%m-%d")                     # raises on a malformed date
    pol = (man.get("recipe") or {}).get("policy") or {}
    off = man.get("offline") or {}
    zones = man.get("zones") or []
    el = sum(int(z.get("eligible_px", 0)) for z in zones)
    pos = sum(int(z.get("positive_px", 0)) for z in zones)
    return {"submission_name": name or Path(man["submission"]).stem, "file": man["submission"],
            "sha256": man.get("sha256"), "candidate": man.get("candidate"), "role": man.get("role"),
            "model": (man.get("recipe") or {}).get("kind"), "code_commit": (man.get("code") or {}).get("git_commit"),
            "recipe_fp": man.get("recipe_fp"), "local_disc_official_dti": off.get("disc_dti"),
            "local_disc_worst": off.get("disc_worst"), "local_geo_dti": off.get("geo_dti"),
            "objective": off.get("objective"), "public_lb_score": s, "date_utc": str(date),
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "postprocessing": json.dumps({k: pol.get(k) for k in ("rep", "frac", "blend", "grid_candidate")} |
                                         {"construction": "zone_stitched", "ring_px": 0}),
            "area_frac_eligible": pos / el if el else None, "validation_status": man.get("validation_status"),
            "was_ready": (man.get("readiness") or {}).get("status") == "READY_FOR_UPLOAD", "notes": notes}


def append_row(H: pd.DataFrame, row: dict) -> pd.DataFrame:
    """History with ``row`` appended; one public score per file (sha256)."""
    H = H if H is not None and len(H.columns) else pd.DataFrame(columns=COLUMNS)
    if row.get("sha256") and len(H) and (H.sha256.astype(str) == str(row["sha256"])).any():
        prev = H[H.sha256.astype(str) == str(row["sha256"])].iloc[0]
        raise ValueError(f"{row['file']} (sha256 {str(row['sha256'])[:12]}) already recorded on {prev.date_utc} "
                         f"with public score {prev.public_lb_score}: one file, one score")
    return pd.concat([H, pd.DataFrame([row]).reindex(columns=COLUMNS)], ignore_index=True)


def uploads_in_window(H: pd.DataFrame, date: str, days: int = 7) -> int:
    """Recorded uploads in the rolling window ending on ``date`` (organizer: 3 scored per 7 days)."""
    if not len(H):
        return 0
    d = pd.to_datetime(H.date_utc, errors="coerce")
    end = pd.Timestamp(date)
    return int(((d > end - pd.Timedelta(days=days)) & (d <= end)).sum())


def comparison(H: pd.DataFrame, tie: float) -> tuple[pd.DataFrame, dict]:
    """Per submission: local (nested disc_official) vs public; pairwise order agreement; rank correlation when >= 3."""
    if not len(H):
        return pd.DataFrame(), {"n": 0}
    T = H.copy()
    T["local"] = pd.to_numeric(T.local_disc_official_dti, errors="coerce")
    T["public"] = pd.to_numeric(T.public_lb_score, errors="coerce")
    T["gap_public_minus_local"] = T.public - T.local
    pairs = []
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            a, b = T.iloc[i], T.iloc[j]
            dl, dp = a.local - b.local, a.public - b.public
            if not (np.isfinite(dl) and np.isfinite(dp)):
                continue
            pairs.append({"a": a.submission_name, "b": b.submission_name, "local_delta": dl, "public_delta": dp,
                          "verdict": "inconclusive" if abs(dp) < tie or abs(dl) < tie else
                          ("agree" if np.sign(dl) == np.sign(dp) else "disagree")})
    P = pd.DataFrame(pairs)
    s = {"n": int(len(T)), "pairs": int(len(P)),
         "agree": int((P.verdict == "agree").sum()) if len(P) else 0,
         "disagree": int((P.verdict == "disagree").sum()) if len(P) else 0,
         "inconclusive": int((P.verdict == "inconclusive").sum()) if len(P) else 0,
         "mean_gap": float(T.gap_public_minus_local.mean()) if T.gap_public_minus_local.notna().any() else None}
    ok = T[["local", "public"]].dropna()
    if len(ok) >= 3:
        from scipy.stats import spearmanr

        s["spearman_local_public"] = float(spearmanr(ok.local, ok.public)[0])
    return T, {**s, "pair_table": P.to_dict("records") if len(P) else []}


def gap_causes(row: dict, tie: float) -> list[str]:
    """Candidate explanations of a local / public gap (hypotheses to test offline, never fixes to tune on the LB)."""
    g = row.get("gap_public_minus_local")
    out = []
    if g is None or not np.isfinite(g):
        return ["no local estimate recorded for this file"]
    if abs(g) < tie:
        return [f"|gap| {abs(g):.4f} < {tie:g}: consistent with the local estimate within noise"]
    if g < 0:
        out += ["pseudo-discovery truth (withheld mapped systems) under-represents the hidden new faults: different "
                "fault types / data sources (organizer t/11527 does not disclose them)",
                "new truth next to known traces (corrections) is scored but rare in the proxy: the known-fault mask "
                "and any suppression near known faults cost TP there",
                "area budget: the pooled public truth density is unknown; the frac was chosen on the proxy",
                "public subset = an undisclosed spatial chunk: three validation zones cannot bound its variance"]
    else:
        out += ["the proxy is pessimistic for this policy (hidden truth denser or closer to the predicted structures)",
                "public subset favourable by chance: do not infer a private-leaderboard gain"]
    out.append("the public score is one subset: it is not a held-out validation and never enters selection")
    return out


# =============================================================================== Colab side
def history_path(fg) -> Path:
    return fg.root / fg.cfg["leaderboard"]["history"]


def load_history(fg) -> pd.DataFrame:
    p = history_path(fg)
    return pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame(columns=COLUMNS)


def record(fg, submission: str, public_score: float, date: str | None = None, notes: str = "") -> dict:
    """Append one manually obtained public score. The file must exist and hash to its manifest."""
    from ..artifacts.atomic import atomic_save
    from .boost_audit import file_hash

    sub = fg.root / "submissions"
    path = sub / Path(submission).name
    man_p = sub / f"{path.stem}_manifest.json"
    if not path.exists() or not man_p.exists():
        raise FileNotFoundError(f"{path} or its manifest {man_p.name} is missing (stage them from Drive first)")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    sha = file_hash(path, 1e9)
    if sha != man.get("sha256"):
        raise ValueError(f"{path.name}: sha256 on disk {sha} != manifest {man.get('sha256')} (the file changed)")
    date = date or time.strftime("%Y-%m-%d", time.gmtime())
    row = make_row(man, public_score, date, notes)
    H = append_row(load_history(fg), row)
    n7 = uploads_in_window(H, date)
    p = history_path(fg)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(p, lambda t: H.to_csv(t, index=False))
    fg.push()
    lim = int(fg.cfg["leaderboard"]["max_scored_per_week"])
    if n7 > lim:
        print(f"[leaderboard] WARNING {n7} uploads recorded in the 7 days to {date} (organizer limit {lim})")
    if not row["was_ready"]:
        print(f"[leaderboard] WARNING {path.name} was not READY_FOR_UPLOAD when generated")
    return {**row, "uploads_last_7_days": n7}


def compare(fg) -> dict:
    """leaderboard_comparison.csv + leaderboard_report.md from the history (no claim about the private board)."""
    from ..artifacts.atomic import atomic_write_text

    tie = float(fg.cfg["leaderboard"]["tie_delta"])
    T, s = comparison(load_history(fg), tie)
    if len(T):
        T["possible_causes"] = [" | ".join(gap_causes(r, tie)) for r in T.to_dict("records")]
    d = history_path(fg).parent
    d.mkdir(parents=True, exist_ok=True)
    fg.save_table(T, "leaderboard_comparison", "leaderboard")
    lines = ["# P6 boost: local validation vs public leaderboard", "",
             "Local = nested held-out disc_official DTI (benchmark); public = the score typed in after a manual "
             "upload. The public board is a spatial subset: agreement is evidence, not validation. Nothing here "
             "claims a private-leaderboard result.", "",
             f"Submissions: {s['n']}; pairs {s.get('pairs', 0)}: agree {s.get('agree', 0)}, disagree "
             f"{s.get('disagree', 0)}, inconclusive (|delta| < {tie:g}) {s.get('inconclusive', 0)}; mean gap "
             f"(public - local) {s.get('mean_gap')}; Spearman {s.get('spearman_local_public', 'n < 3')}", ""]
    if len(T):
        lines += ["| submission | candidate | local disc | public | gap | possible causes |", "|---|---|---:|---:|---:|---|"]
        for r in T.to_dict("records"):
            lines.append(f"| {r['submission_name']} | {r['candidate']} | {r['local']:.4f} | {r['public']:.4f} | "
                         f"{r['gap_public_minus_local']:+.4f} | {r['possible_causes'][:300]} |")
    atomic_write_text(d / "leaderboard_report.md", "\n".join(lines) + "\n")
    fg.push()
    return {"summary": s, "table": T}
