"""Self-contained HTML / Markdown EDA report assembled from computed tables and saved figures.

Text is generated from the numbers; sections whose inputs are missing say so explicitly.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd


def _img(path: Path) -> str:
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="max-width:100%">'


def html_report(title: str, sections: list[dict], out_path) -> Path:
    """``sections``: [{'title', 'text', 'tables': {name: df}, 'figures': [png paths]}]."""
    from ..registry import atomic_write_text

    parts = [f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>"
             "body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;color:#1d1d1d}"
             "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:3px 6px}"
             "@media(prefers-color-scheme:dark){body{background:#151515;color:#e6e6e6}td,th{border-color:#444}}"
             f"</style></head><body><h1>{html.escape(title)}</h1>"]
    for s in sections:
        parts.append(f"<h2>{html.escape(s['title'])}</h2><p>{html.escape(s.get('text', ''))}</p>")
        for name, df in (s.get("tables") or {}).items():
            if isinstance(df, pd.DataFrame) and len(df):
                parts.append(f"<h4>{html.escape(name)}</h4>" + df.head(60).to_html(index=False, float_format=lambda x: f"{x:.4g}"))
        for f in s.get("figures") or []:
            if Path(f).is_file():
                parts.append(_img(f))
    parts.append("</body></html>")
    return atomic_write_text(out_path, "\n".join(parts))


def recommendation_table(contrast: pd.DataFrame, util: pd.DataFrame, native: pd.DataFrame, compl: pd.DataFrame,
                         coverage: dict) -> pd.DataFrame:
    """Evidence-driven modelling recommendations (each row cites the numbers it is based on)."""
    import numpy as np

    rows = []
    fam_of = {}
    if len(util):
        fam_of = util.drop_duplicates("feature").set_index("feature").family.to_dict()
    sig = contrast[(contrast.ci_low > 0) | (contrast.ci_high < 0)] if len(contrast) else contrast
    allu = pd.concat([util[util.region == "all"] if len(util) else util, native if len(native) else native],
                     ignore_index=True)
    for fam, g in allu.groupby("family") if len(allu) else []:
        best = g.sort_values("auc", ascending=False).head(1)
        n_sig = int(sig.feature.map(fam_of).eq(fam).sum()) if len(sig) else 0
        auc = float(best.auc.iloc[0]) if len(best) and np.isfinite(best.auc.iloc[0]) else np.nan
        rec = ("use in the pixel / patch models" if (auc >= 0.6 and (n_sig or fam == "dem_native"))
               else "low priority (weak separation)")
        rows.append({"area": f"feature family {fam}", "recommendation": rec,
                     "evidence": f"best AUC {auc:.3f} ({best.feature.iloc[0] if len(best) else '-'}, scale "
                                 f"{best.scale_m.iloc[0] if len(best) else '-'}), {n_sig} feature(s) with contrast CI "
                                 "excluding 0"})
    fc = coverage.get("fault_fraction_with_usable_dem")
    if fc is not None:
        rows.append({"area": "native DEM (E4 gated fusion, segment DEM features)",
                     "recommendation": "train E4 / use cross-fault features" if fc >= 0.2 else
                     "expect limited benefit: most faults lack usable DEM",
                     "evidence": f"{100 * fc:.1f}% of mapped-fault pixels have usable (verified-unit) DEM; "
                                 f"{100 * coverage.get('fault_coverage_lost_to_unit_exclusion', 0):.1f}% lost to unit exclusions"})
    if len(native) and len(util):
        nb = native.auc.max()
        db = util[(util.family == "dem_100m") & (util.region == "all")].auc.max() if (util.family == "dem_100m").any() else np.nan
        rows.append({"area": "resolution", "recommendation": "keep native-resolution segment features" if nb > (db if np.isfinite(db) else 0)
                     else "100 m DEM summaries capture most of the terrain signal",
                     "evidence": f"best native AUC {nb:.3f} vs best 100 m DEM AUC {db:.3f}"})
    if len(compl) and "mapped_recall_top1pct" in compl:
        single = compl[compl.model_b.fillna("") == ""]
        if len(single):
            b = single.sort_values("mapped_recall_top1pct", ascending=False).iloc[0]
            rows.append({"area": "existing predictions", "recommendation": "reference for EDA / disagreement only "
                         "(original splits are not the G11 split)",
                         "evidence": f"best mapped recall@top-1% {b.mapped_recall_top1pct:.3f} ({b.model_a}, {b.provenance})"})
    return pd.DataFrame(rows, columns=["area", "recommendation", "evidence"])


def decision_report(contrast: pd.DataFrame, util: pd.DataFrame, redund: pd.DataFrame, failures: pd.DataFrame | None
                    ) -> str:
    """Markdown decision summary: strongest features (CI excludes 0), useful scales, candidate / hard-negative types."""
    L = ["# G11 EDA decision report", ""]
    if len(contrast):
        c = contrast[(contrast.ci_low > 0) | (contrast.ci_high < 0)].copy()
        c["abs"] = c.near_vs_local_control.abs()
        geo = c[~c.feature.str.startswith(("dem", "terrain"))].nlargest(10, "abs")
        dem = c[c.feature.str.startswith(("dem", "terrain"))].nlargest(10, "abs")
        L += ["## Most promising geophysical features (block-bootstrap CI excludes 0)", ""]
        L += [f"- {r.feature}: {r.near_vs_local_control:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" for r in geo.itertuples()] or ["- none"]
        L += ["", "## Most promising DEM features", ""]
        L += [f"- {r.feature}: {r.near_vs_local_control:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" for r in dem.itertuples()] or ["- none (or no DEM coverage)"]
    else:
        L += ["No contrast table (stage not run)."]
    if len(util):
        from .multiscale import best_scales

        b = best_scales(util).head(15)
        L += ["", "## Useful spatial scales (best AUC per feature base)", ""]
        L += [f"- {r.base}: {r.scale_m:g} m (AUC {r.auc:.3f})" for r in b.itertuples()]
    if len(redund):
        L += ["", f"## Redundancy: {redund.cluster.nunique()} clusters among {len(redund)} features "
              f"(|rho| >= threshold grouped)"]
    if failures is not None and len(failures):
        vc = failures.dominant_mode.value_counts(normalize=True)
        L += ["", "## Candidate / hard-negative types (dominant descriptive mode)", ""]
        L += [f"- {k}: {v:.1%}" for k, v in vc.items()]
    L += ["", "## Recommended modelling experiments", "",
          "Derived from the tables above: retain features whose contrast CI excludes 0 and are cluster representatives;"
          " run E2/E4 only if DEM features appear above; use the dominant hard-negative modes for hard-negative "
          "sampling. These are hypotheses for the experiment matrix, not results."]
    return "\n".join(L) + "\n"
