"""Figure index, findings tables and the Markdown / HTML research report (sections 38-39).

Findings are only populated from measured tables: every ``Finding`` carries the measurement it rests on and a
confidence label computed from the evidence (``supported`` needs a positive effect whose uncertainty interval
excludes zero and holds in every held-out zone; ``tentative`` otherwise; ``not supported`` when the effect is <= 0).
"""

from __future__ import annotations

import html
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import atomic_save_table, atomic_write_text


# =============================================================================== figures
class FigureIndex:
    def __init__(self, ws):
        self.ws = ws
        self.path = ws.path("figures", "figure_index.csv")
        self.rows = pd.read_csv(self.path).to_dict("records") if self.path.exists() else []

    def save(self, fig, name: str, section: str, takeaway: str, dpi: int = 140) -> Path:
        import matplotlib.pyplot as plt

        p = self.ws.path("figures", f"{name}.png")
        tmp = p.with_name(p.stem + ".partial.png")
        fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        tmp.replace(p)
        self.rows = [r for r in self.rows if r["name"] != name]
        self.rows.append(
            {
                "name": name,
                "section": section,
                "file": str(p.relative_to(self.ws.root)),
                "takeaway": takeaway,
                "created_unix": time.time(),
            }
        )
        atomic_save_table(self.path, pd.DataFrame(self.rows))
        return p


def extent_km(data) -> tuple:
    t = data.transform
    h, w = data.shape
    return (t.c / 1000, (t.c + w * t.a) / 1000, (t.f + h * t.e) / 1000, t.f / 1000)


def map_axes(ax, data, arr, title, cmap="viridis", vmin=None, vmax=None, step=None):
    """Plot a full-grid raster in UTM km coordinates with a decimation step chosen for display only."""
    step = step or max(1, max(data.shape) // 1200)
    im = ax.imshow(arr[::step, ::step], extent=extent_km(data), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("UTM 11N easting (km)", fontsize=7)
    ax.set_ylabel("northing (km)", fontsize=7)
    ax.tick_params(labelsize=6)
    return im


# =============================================================================== findings
@dataclass
class Finding:
    section: str
    finding: str
    measurement: str
    effect: float
    ci_lo: float
    ci_hi: float
    zones_positive: str
    confidence: str
    follow_up: str


def judge(effect: float, ci_lo: float, ci_hi: float, zones_pos: int, zones: int) -> str:
    if not np.isfinite(effect):
        return "not evaluated"
    if effect <= 0:
        return "not supported"
    if np.isfinite(ci_lo) and ci_lo > 0 and zones_pos == zones and zones > 0:
        return "supported"
    return "tentative"


def finding(section, text, measurement, effect, ci_lo=np.nan, ci_hi=np.nan, zones_pos=0, zones=0, follow_up=""):
    return Finding(
        section,
        text,
        measurement,
        float(effect),
        float(ci_lo),
        float(ci_hi),
        f"{zones_pos}/{zones}",
        judge(effect, ci_lo, ci_hi, zones_pos, zones),
        follow_up,
    )


def findings_table(items: list[Finding]) -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in items])


# =============================================================================== report
SECTIONS = [
    "Executive summary",
    "Dataset characterization",
    "Geological hypotheses",
    "Feature engineering",
    "Key exploratory results",
    "Best-performing physical indicators",
    "Geological regimes",
    "Fault discovery candidates",
    "Model comparisons",
    "Validation methodology",
    "DTI analysis",
    "Failure categories",
    "Final submission selection",
    "Uncertainty and limitations",
    "Recommended next experiments",
]


def _md_table(df: pd.DataFrame | None, max_rows: int = 25) -> str:
    if df is None or len(df) == 0:
        return "_not available (stage not executed)_\n"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    more = f"\n_{len(df) - max_rows} more rows in the CSV._\n" if len(df) > max_rows else "\n"
    return "\n".join(lines) + more


def write_report(ws, content: dict[str, list], title: str = "GEMS G13 GeoDiscovery report") -> tuple[Path, Path]:
    """``content``: section -> list of str (markdown) or DataFrame. Writes report.md and report.html."""
    md = [
        f"# {title}",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. Offline DTI values are "
        "proxies against withheld known faults; no number here is an official leaderboard score unless it comes "
        "from `submission_history.csv`._",
        "",
    ]
    for i, sec in enumerate(SECTIONS, 1):
        md.append(f"## {i}. {sec}\n")
        for item in content.get(sec, ["_not executed in this run._"]):
            md.append(_md_table(item) if isinstance(item, pd.DataFrame) else str(item) + "\n")
    fig_idx = ws.root / "figures" / "figure_index.csv"
    if fig_idx.exists():
        md.append("## Figure index\n")
        md.append(_md_table(pd.read_csv(fig_idx)[["name", "section", "takeaway"]], 200))
    text = "\n".join(md)
    p_md = atomic_write_text(ws.path("reports", "report.md"), text)
    p_html = atomic_write_text(ws.path("reports", "report.html"), _to_html(text, title))
    return p_md, p_html


def _to_html(md: str, title: str) -> str:
    """Minimal Markdown -> HTML (headings, tables, paragraphs, code); no external dependency."""
    out, in_table = [], False
    for line in md.splitlines():
        if line.startswith("|") and line.endswith("|"):
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-"):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("#"):
            n = len(line) - len(line.lstrip("#"))
            out.append(f"<h{n}>{html.escape(line[n:].strip())}</h{n}>")
        elif line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        out.append("</table>")
    style = (
        "body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5}"
        "table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px}"
        "th{background:#f3f3f3}"
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{style}</style>"
        f"</head><body>{''.join(out)}</body></html>"
    )


def runtime_table(ws) -> pd.DataFrame:
    rows = []
    for p in sorted((ws.root / "manifests").glob("*.json")):
        m = json.loads(p.read_text())
        rows.append(
            {
                "stage": m["stage"],
                "status": m["status"],
                "seconds": m.get("seconds"),
                "outputs": len(m.get("outputs", {})),
                "output_mb": sum(v["size"] for v in m.get("outputs", {}).values()) / 2**20,
            }
        )
    return pd.DataFrame(rows).sort_values("seconds", ascending=False, na_position="last") if rows else pd.DataFrame()
