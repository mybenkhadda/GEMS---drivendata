"""Notebook-native monitoring: stage table, runtime / storage summary, model and experiment status, figures.

Pandas tables and Matplotlib only. Everything shown is also saved under ``reports/pipeline/`` (and pushed
to Drive by the ``reports`` stage): ``stage_status.csv``, ``dashboard_summary.csv``, ``figures/*.png|svg``
and ``PIPELINE_REPORT.md``. Scores shown here are read from saved scoreboards (local protocol: geographic
CV + pseudo-discovery); public-leaderboard numbers only ever come from ``reports/leaderboard_log.csv``.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd

COLORS = {"done": "#1baf7a", "cached": "#2a78d6", "pending": "#bdbdbd", "failed": "#e34948", "blocked": "#eda100",
          "disabled": "#8a8985", "interrupted": "#e87ba4", "incompatible": "#6250d6"}


def stage_table(pc, reg, executor=None) -> pd.DataFrame:
    """One row per stage: state (from this session's results, else the persisted record), last run, outputs."""
    rows = []
    res = executor.results if executor is not None else {}
    for s in reg:
        rec = pc.manifests.load(s.name)
        r = res.get(s.name)
        if r is not None:
            state = {"SKIP_CACHED": "cached", "DONE": "done", "FAILED": "failed", "BLOCKED": "blocked",
                     "DISABLED": "disabled", "INTERRUPTED": "interrupted", "INCOMPATIBLE": "incompatible",
                     "MISSING_PREREQ": "blocked"}.get(r.outcome, "pending")
            reason, recovery = r.reason, r.recovery
        else:
            state = {"done": "cached", "failed": "failed", "interrupted": "interrupted",
                     "running": "interrupted"}.get(rec.get("status", ""), "pending")
            reason, recovery = rec.get("error", "") or rec.get("validation", ""), rec.get("recovery", "")
        rows.append({"stage": s.name, "state": state, "title": s.title, "resource": s.resource,
                     "origin": rec.get("origin", ""), "last_finished": rec.get("finished", ""),
                     "attempts": rec.get("attempts", 0), "reason": str(reason)[:200], "recovery": str(recovery)[:200],
                     "outputs": ", ".join(s.outputs(pc)[:2]) if _safe(lambda: s.outputs(pc)) else ""})
    return pd.DataFrame(rows)


def _safe(f):
    try:
        return f()
    except Exception:
        return None


def storage(pc) -> pd.DataFrame:
    rows = []
    for label, root in (("local", pc.local), ("drive", pc.drive)):
        if root is None or not Path(root).exists():
            continue
        u = shutil.disk_usage(root)
        rows.append({"root": label, "path": str(root), "free_gb": round(u.free / 2**30, 1),
                     "total_gb": round(u.total / 2**30, 1)})
    return pd.DataFrame(rows)


def summary(pc, reg, executor=None) -> pd.DataFrame:
    """Headline numbers of the dashboard (one key / value table)."""
    st = stage_table(pc, reg, executor)
    g = pc.gpu()
    rows = {"time": time.strftime("%Y-%m-%d %H:%M"), "mode": pc.cfg.mode,
            "device": g.get("name") or "CPU only", "vram_gb": g.get("vram_gb"),
            "drive": str(pc.drive) if pc.drive else "NOT MOUNTED"}
    for k in ("done", "cached", "pending", "failed", "blocked", "disabled", "interrupted", "incompatible"):
        rows[f"stages_{k}"] = int((st.state == k).sum())
    for r in storage(pc).itertuples():
        rows[f"free_gb_{r.root}"] = r.free_gb
    fr = pc.local / "data/feature_store/pipeline_feature_registry.csv"
    if fr.exists():
        f = pd.read_csv(fr)
        rows["features_required"] = len(f)
        rows["features_cached"] = int(f.cached.sum())
    bs = pc.local / "reports/pipeline/baseline_status.csv"
    if bs.exists():
        b = pd.read_csv(bs)
        rows["baselines_available"] = ", ".join(b.model[b.status == "AVAILABLE"]) or "none"
        rows["baselines_cv_only"] = ", ".join(b.model[b.status == "CV_ONLY"]) or "none"
        rows["baselines_not_available"] = ", ".join(b.model[b.status == "NOT_AVAILABLE"]) or "none"
    es = pc.local / "reports/pipeline/experiment_status.csv"
    if es.exists():
        e = pd.read_csv(es)
        rows["experiments_completed"] = int((e.get("export", pd.Series(dtype=str)) == "done").sum())
    sb = pc.local / "reports/pipeline/scoreboard_latest.csv"
    if sb.exists():
        b = pd.read_csv(sb)
        for col, lab in (("geo_dti_area", "best_geo_dti"), ("disc_dti_excl", "best_disc_dti_excl")):
            if col in b and b[col].notna().any():
                i = b[col].idxmax()
                rows[f"{lab} (local protocol)"] = f"{b.loc[i, 'model']}: {b.loc[i, col]:.4f}"
    figs = list((pc.local / "reports").rglob("*.png"))
    rows["figures_saved"] = len(figs)
    from ..artifacts.registry import ModelRegistry

    reg_dir = pc.local / "model_registry"
    ch = ModelRegistry(reg_dir).champion() if reg_dir.exists() else None
    rows["promoted_model"] = ch["name"] if ch else "none (promotion is manual)"
    subs = list((pc.local / "submissions").glob("*.tif")) if (pc.local / "submissions").exists() else []
    rows["submission_files"] = len(subs)
    rows["submission_ready"] = "yes" if subs else ("model promoted, export pending" if ch else "no promoted model")
    return pd.DataFrame({"key": list(rows), "value": [str(v) for v in rows.values()]})


def dashboard_figure(pc, reg, executor=None):
    import matplotlib.pyplot as plt

    st = stage_table(pc, reg, executor)
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(st) + 1.2))
    y = range(len(st))[::-1]
    ax.barh(list(y), [1] * len(st), color=[COLORS.get(s, "#bdbdbd") for s in st.state], edgecolor="white")
    for yi, r in zip(y, st.itertuples()):
        ax.text(0.02, yi, f"{r.stage}", va="center", ha="left", fontsize=8, color="white", fontweight="bold")
        ax.text(0.98, yi, r.state, va="center", ha="right", fontsize=8, color="white")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"GEMS master pipeline - {time.strftime('%Y-%m-%d %H:%M')} ({pc.cfg.mode})", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    ax.legend(handles, list(COLORS), ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False)
    return fig


def fold_figure(pc):
    """Geographic fold map + discovery withheld systems (from the saved rasters)."""
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    fdir, ddir = pc.local / pc.fold_rel, pc.local / pc.discovery_rel
    if not (fdir / "fold_map.tif").exists() or not (ddir / "discovery_map.tif").exists():
        return None
    with rasterio.open(fdir / "fold_map.tif") as src:
        fm = src.read(1, out_shape=(src.height // 4, src.width // 4))
    with rasterio.open(ddir / "discovery_map.tif") as src:
        n = src.count - 1
        dm = [src.read(i + 2, out_shape=(src.height // 4, src.width // 4)) for i in range(n)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(fm, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    ax[0].set_title(f"geographic folds ({pc.fold_rel.split('/')[-1]})", fontsize=9)
    wh = np.zeros_like(dm[0], dtype=float) * np.nan
    for d, a in enumerate(dm):
        wh[a >= 3] = d
    ax[1].imshow(np.isfinite(fm), cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
    ax[1].imshow(wh, cmap="Set1", vmin=0, vmax=8, interpolation="nearest")
    ax[1].set_title("withheld fault systems per discovery split", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    return fig


def save_fig(pc, fig, name: str) -> list[str]:
    import matplotlib.pyplot as plt

    if fig is None:
        return []
    out = []
    if pc.cfg.get("reports.save_figures", True):
        d = pc.reports / "figures"
        for ext in ("png", "svg"):
            fig.savefig(d / f"{name}.{ext}", dpi=160, bbox_inches="tight")
            out.append(f"reports/pipeline/figures/{name}.{ext}")
    plt.close(fig)
    return out


def write_report(pc, reg=None, executor=None) -> dict:
    """Tables + figures + PIPELINE_REPORT.md under reports/pipeline/ (pushed by the reports stage)."""
    import matplotlib

    matplotlib.use("Agg")
    from .stages import build_registry

    reg = reg or pc._cache.get("registry") or build_registry()
    executor = executor or pc._cache.get("executor")          # this session's results when run through Pipeline
    st = stage_table(pc, reg, executor)
    st.to_csv(pc.reports / "stage_status.csv", index=False)
    sm = summary(pc, reg, executor)
    sm.to_csv(pc.reports / "dashboard_summary.csv", index=False)
    figs = save_fig(pc, dashboard_figure(pc, reg, executor), "pipeline_dashboard")
    figs += save_fig(pc, _safe(lambda: fold_figure(pc)), "validation_design")
    lines = ["# GEMS master pipeline report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} - mode {pc.cfg.mode}.",
             "", "Scores are local-protocol numbers (geographic CV + pseudo-discovery), never public-leaderboard "
             "scores. Models marked NOT_AVAILABLE were neither found nor retrained; no number is reported for them.",
             "", "## Summary", "", _md(sm), "", "## Stages", "",
             _md(st[["stage", "state", "origin", "last_finished", "reason", "recovery"]]), ""]
    for name, title in (("baseline_status", "Baselines"), ("experiment_status", "Research experiments"),
                        ("scoreboard_latest", "Latest scoreboard (local protocol)")):
        p = pc.reports / f"{name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            keep = [c for c in ("model", "status", "origin", "discovery", "checkpoints", "detail", "experiment",
                                "experiment_id", "prepare", "train", "evaluate", "export", "error", "geo_dti_area",
                                "disc_dti_excl", "disc_dti_retained_fp", "public_lb") if c in df.columns]
            lines += [f"## {title}", "", _md(df[keep] if keep else df), ""]
    lines += ["## Figures", ""] + [f"* `{f}`" for f in figs if f.endswith(".png")]
    (pc.reports / "PIPELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"figures": figs, "report": "reports/pipeline/PIPELINE_REPORT.md"}


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:                                  # tabulate missing: plain CSV block
        return "```\n" + df.to_csv(index=False) + "```"
