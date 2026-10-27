"""G01 review: what a persisted G01 run actually established (READ-ONLY: never writes into, nor creates, a G01 run).

    from gems_bench.m9.research.g01_review import review_g01, g01_run_dirs
    rev = review_g01(run_dir)              # run_dir = .../research/G01_wavelet_geology/g01_<fp>  (or None)
    rev["questions"], rev["decision_table"], rev["markdown"]

Every answer is derived from the run's tables (``hypothesis_contrasts``, ``bootstrap_vs_references``, ``ablations``,
``nested_choice``, ``summary_per_zone``) and, for the gradient-redundancy question, from the per-unit rasters. When a
table or a run is absent the answer says ABSENT and the component is UNVERIFIED: nothing is inferred from code or from
expectations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COMPONENTS = [  # (component, candidate, reference, interpretation if supported, interpretation if refuted)
    ("magnetic wavelet energy", "G01-A", "edge",
     "band-limited multiscale magnetic contrast localises faults better than a raw gradient",
     "wavelet energy adds cost, not information, over a plain magnetic gradient"),
    ("gravity wavelet energy", "G01-B", "edge",
     "density boundaries alone carry fault information beyond the magnetic gradient",
     "gravity alone is too smooth / deep to localise mapped faults at these distances"),
    ("tolerant magnetic-gravity co-location", "G01-C", "G01-A",
     "faults that juxtapose BOTH susceptibility and density contrasts are favoured; gravity adds to magnetics",
     "gravity co-location dilutes the magnetic signal (different sources or offset beyond tolerance)"),
    ("axial strike compatibility", "G01-D", "G01-C",
     "co-located but discordant energy (e.g. a dyke across a basin edge) is correctly demoted",
     "orientation agreement is uninformative or the gate removes real oblique intersections"),
    ("multiscale persistence", "G01-E", "G01-D",
     "deep-rooted structures persistent across scales are more often mapped faults",
     "real faults are visible at one scale only; persistence penalises them"),
    ("G01 fusion vs M9 engineered cross-modal", "G01-E", "G01-F",
     "physically constrained fusion beats M9's engineered cross-modal layers at equal area",
     "M9's engineered layers already capture (or exceed) the physical fusion"),
    ("G01 simple joint vs M9 engineered cross-modal", "G01-C", "G01-F",
     "the simple joint map is competitive with M9's engineered layers",
     "M9's engineered layers are stronger than the simple joint map"),
]


def g01_run_dirs(parent: Path) -> list[Path]:
    return sorted(Path(parent).glob("g01_*")) if Path(parent).exists() else []


def _table(root: Path, name: str, sub: str = "tables") -> pd.DataFrame:
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = Path(root) / sub / f"{name}{ext}"
        if p.exists():
            try:
                return reader(p)
            except Exception:
                continue
    return pd.DataFrame()


def _json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def load_g01_run(root: Path | None) -> dict:
    """Every G01 table that exists (empty DataFrames otherwise) + done markers + summary."""
    names = ("units", "summary_pooled", "summary_per_zone", "bootstrap_vs_references", "hypothesis_contrasts",
             "ablations", "nested_choice", "official_style_pooled", "channel_audit")
    if root is None or not Path(root).exists():
        return {"root": None, "steps_done": [], **{n: pd.DataFrame() for n in names}, "summary": None}
    root = Path(root)
    out = {"root": root, "steps_done": sorted(p.stem.replace("done_", "") for p in (root / "processed").glob("done_*.json"))}
    out.update({n: _table(root, n) for n in names})
    out["summary"] = _json(root / "reports" / "summary.json")
    out["inputs"] = _json(root / "config" / "inputs.json")
    return out


def _contrast(H: pd.DataFrame, cand: str, ref: str) -> pd.DataFrame:
    if not len(H) or not {"candidate", "reference"} <= set(H.columns):
        return pd.DataFrame()
    return H[(H.candidate == cand) & (H.reference == ref)]


def _boot(Bt: pd.DataFrame, cand: str, ref: str, metric: str) -> pd.DataFrame:
    if not len(Bt):
        return pd.DataFrame()
    return Bt[(Bt.candidate == cand) & (Bt.reference == ref) & (Bt.metric == metric)]


def _fmt(df: pd.DataFrame) -> str:
    if not len(df):
        return "ABSENT"
    parts = []
    for r in df.itertuples():
        v = getattr(r, "verdict", "")
        d = getattr(r, "delta", np.nan)
        p = getattr(r, "p_delta_positive", np.nan)
        w = getattr(r, "zone_win_share", np.nan)
        parts.append(f"{r.task}: delta={d:+.4f}, P(>0)={p:.2f}, zones won={w:.2f} {v}".strip())
    return "; ".join(parts)


def _verdict_to_action(df: pd.DataFrame) -> str:
    if not len(df) or "verdict" not in df:
        return "UNVERIFIED"
    v = set(df.verdict)
    if v == {"SUPPORTED"}:
        return "Keep"
    if "REFUTED" in v and "SUPPORTED" not in v:
        return "Reject"
    if v <= {"UNAVAILABLE"}:
        return "UNVERIFIED"
    return "Investigate"


def edge_redundancy(root: Path | None, n: int = 50000, seed: int = 2026) -> pd.DataFrame:
    """Spearman correlation of G01-E (and G01-C) with the simple edge on each unit's domain (unit rasters)."""
    rows = []
    if root is None:
        return pd.DataFrame(rows)
    for p in sorted((Path(root) / "rasters" / "units").glob("*.npz")):
        try:
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z["__meta__"]))
                inv = {v: k for k, v in (meta.get("keys") or {}).items()}
                get = lambda k: np.asarray(z[inv.get(k, k)], np.float32) if inv.get(k, k) in z.files else None  # noqa: E731
                dom, edge = get("domain"), get("edge")
                if dom is None or edge is None:
                    continue
                idx = np.flatnonzero(dom.ravel() > 0.5)
                if idx.size > n:
                    idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
                re = pd.Series(edge.ravel()[idx]).rank()
                for c in ("G01-A", "G01-C", "G01-E", "G01-F"):
                    a = get(c)
                    if a is None:
                        continue
                    rc = pd.Series(a.ravel()[idx]).rank()
                    ok = re.notna() & rc.notna()
                    rows.append({"unit": p.stem, "candidate": c, "spearman_vs_edge": float(np.corrcoef(re[ok], rc[ok])[0, 1])
                                 if ok.sum() > 10 else np.nan, "n": int(ok.sum())})
        except Exception as e:                                       # unreadable raster: reported, never fatal
            rows.append({"unit": p.stem, "candidate": "error", "spearman_vs_edge": np.nan, "n": 0, "error": str(e)})
    return pd.DataFrame(rows)


def review_g01(root: Path | None, metric: str = "f1") -> dict:
    R = load_g01_run(root)
    H, Bt, A, N, Z = (R["hypothesis_contrasts"], R["bootstrap_vs_references"], R["ablations"], R["nested_choice"],
                      R["summary_per_zone"])
    q = []

    def add(i, question, answer, source):
        q.append({"#": i, "question": question, "answer": answer, "source": source})

    have = R["root"] is not None
    add(1, "Which G01 experiments were actually completed?",
        (f"steps done: {', '.join(R['steps_done']) or 'none'}; units evaluated: {len(R['units'])}; candidates with "
         f"pooled results: {sorted(set(R['summary_pooled'].candidate)) if len(R['summary_pooled']) else 'none'}")
        if have else "ABSENT: no G01 run folder (G01 has never been executed)", "processed/done_*.json, tables/units")
    if len(A) and {"candidate", "f1"} <= set(A.columns):
        a = A[A.candidate.str.contains("@", regex=False)].copy()
        a["base"], a["lam"] = a.candidate.str.split("@").str[0], a.candidate.str.split("@").str[1]
        tops = []
        for (task, base), g in a.groupby(["task", "base"]):
            g = g.sort_values(metric, ascending=False)
            tops.append(f"{task}/{base}: " + ", ".join(f"{r.lam}={getattr(r, metric):.3f}" for r in g.itertuples()))
        ans = " | ".join(tops) if tops else "ABSENT"
        if len(N) and "chosen" in N:
            ans += f" || nested choices: {N.groupby(['family', 'chosen']).size().to_dict()}"
        add(2, "Which wavelengths contributed most to fault localisation?", ans, "tables/ablations, nested_choice")
    else:
        add(2, "Which wavelengths contributed most to fault localisation?", "ABSENT", "tables/ablations")
    add(3, "Did gravity improve magnetic-only evidence?", _fmt(_contrast(H, "G01-C", "G01-A")), "G01-C vs G01-A")
    add(4, "Did strike agreement help?", _fmt(_contrast(H, "G01-D", "G01-C")), "G01-D vs G01-C")
    add(5, "Did multiscale persistence help?", _fmt(_contrast(H, "G01-E", "G01-D")), "G01-E vs G01-D")
    if len(A) and A.candidate.str.contains("|tol=", regex=False).any():
        t = A[A.candidate.str.contains("|tol=", regex=False)].sort_values(["task", metric], ascending=[True, False])
        ans = " | ".join(f"{r.task}:{r.candidate.split('=')[1]}={getattr(r, metric):.3f}" for r in t.itertuples())
        add(6, "Did spatial tolerance improve alignment?", ans, "tables/ablations (G01-C|tol=...)")
    else:
        add(6, "Did spatial tolerance improve alignment?", "ABSENT (tolerance ablation not run)", "tables/ablations")
    e = pd.concat([_contrast(H, "G01-A", "edge"), _boot(Bt, "G01-E", "edge", metric)], ignore_index=True)
    add(7, "Did G01 outperform simple magnetic-edge detection?", _fmt(e), "G01-A vs edge, G01-E vs edge (bootstrap)")
    add(8, "Did G01 outperform the M9 cross-modal features?",
        _fmt(pd.concat([_contrast(H, "G01-E", "G01-F"), _contrast(H, "G01-C", "G01-F")], ignore_index=True)),
        "G01-E / G01-C vs G01-F")
    if len(H) and "zone_win_share" in H:
        cons = H[["hypothesis", "task", "candidate", "reference", "zone_win_share", "n_zones"]].dropna()
        ans = "; ".join(f"{r.candidate}>{r.reference} {r.task}: {r.zone_win_share:.2f} of {int(r.n_zones)}"
                        for r in cons.itertuples()) or "ABSENT"
    else:
        ans = "ABSENT"
    add(9, "Were improvements consistent across geographic folds?", ans, "hypothesis_contrasts.zone_win_share")
    red = edge_redundancy(R["root"])
    if len(red) and "spearman_vs_edge" in red:
        s = red.groupby("candidate").spearman_vs_edge.median()
        ans = (", ".join(f"{k}: median rho vs edge = {v:.2f}" for k, v in s.items())
               + ". rho > 0.9 = the map is essentially a gradient; geological plausibility still needs figures 9 / 12")
    else:
        ans = "ABSENT (no unit rasters); plausibility cannot be judged without figures 9 / 12"
    add(10, "Do candidate maps show plausible geology or merely strong gradients?", ans,
        "rasters/units/*.npz rank correlation with edge")
    rows = []
    for comp, c, r, yes, no in COMPONENTS:
        h = _contrast(H, c, r)
        act = _verdict_to_action(h)
        interp = yes if act == "Keep" else no if act == "Reject" else \
            "not measured" if act == "UNVERIFIED" else f"mixed: if confirmed, {yes}"
        inc = ", ".join(f"{x.task} {x.delta:+.4f}" for x in h.itertuples() if np.isfinite(getattr(x, "delta", np.nan))) \
            if len(h) and "delta" in h else "ABSENT"
        rows.append({"Component": comp, "Evidence": f"{c} vs {r}: {_fmt(h)}", "Incremental value": inc,
                     "Geological interpretation": interp, "Keep / Reject / Investigate": act})
    if len(N) and "held_out_value" in N:
        for fam, g in N.groupby("family"):
            ch = g.chosen.value_counts()
            stable = ch.iloc[0] / ch.sum() if len(ch) else np.nan
            rows.append({"Component": f"nested choice {fam}", "Evidence": f"chosen {ch.to_dict()}",
                         "Incremental value": f"held-out {metric} mean {g.held_out_value.mean():.4f}",
                         "Geological interpretation": "a stable choice across zones indicates a characteristic scale / "
                                                      "tolerance; an unstable one indicates none",
                         "Keep / Reject / Investigate": "Investigate" if stable < 0.67 else "Keep"})
    D = pd.DataFrame(rows)
    Q = pd.DataFrame(q)
    md = ["# G01 review (generated from persisted G01 results)", "",
          f"Run: `{R['root']}`" if have else "**No G01 run found: every G01 result is ABSENT; every component is "
                                              "UNVERIFIED.**", "",
          "## Ten review questions", "", _md(Q), "", "## Decision table", "", _md(D), ""]
    return {"questions": Q, "decision_table": D, "edge_redundancy": red, "available": have, "markdown": "\n".join(md),
            "run": R}


def _md(df: pd.DataFrame) -> str:
    if not len(df):
        return "_(empty)_"
    cols = list(df.columns)
    esc = lambda v: str(v).replace("|", "\\|").replace("\n", " ")   # noqa: E731
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(esc(v) for v in r) + " |" for r in df.itertuples(index=False, name=None)])
