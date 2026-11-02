"""Stage B: failure atlas of the G05 investigation systems + matched recovered controls.

For every target system (config ``target_systems``: discovery zone -> system ids, from G05's larger union pool) and
its matched controls (systems of the same V1 unit RECOVERED by the frozen W2, nearest in length), on the V1 unit:
  * metrics: length, scored pixels, W2 pixel rank on the system (median / max), inventory coverage, frozen-policy
    TP support and hit share, selected pixels within the window, nearest selected pixel, candidate segments near the
    system and their proposers;
  * five figures (post-hoc: truth is drawn ONLY as an outline on a diagnostic figure, after every score was fixed):
    1 learned evidence (W2 pixel rank + frozen selection), 2 CNN B0 / B1, 3 potential fields (magnetic / gravity
    horizontal gradients), 4 structural / multiscale (G02 continuity, G01 persistence, PF persistence), 5 candidate
    inventory by proposer family + permitted known context;
  * an observation (measured numbers) and a HYPOTHESIS (rule-based reading, explicitly labelled, never a finding).
Truth here is evaluation-side only: nothing in this module feeds a candidate, a score or a model.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

CATEGORIES = ("NOT_FOUND_IN_UNIT", "NOT_IN_INVENTORY", "IN_INVENTORY_NOT_SELECTED", "PARTIAL_SELECTION",
              "RECOVERED")
PANELS = ("1_learned", "2_cnn", "3_potential", "4_structural", "5_candidates")


# =============================================================================== pure
def matched_controls(A: pd.DataFrame, recovered: pd.DataFrame, unit: str, sid: int, n: int = 2) -> list[int]:
    """System ids of ``unit`` recovered by the frozen W2, nearest in length to ``sid`` (deterministic tie-break by
    id). ``A``: unit, system_id, length_km; ``recovered``: unit, system_id, recovered."""
    a = A[A.unit == unit]
    if not len(a) or sid not in set(a.system_id.astype(int)):
        return []
    L = float(a[a.system_id.astype(int) == sid].length_km.iloc[0])
    rec = recovered[(recovered.unit == unit) & recovered.recovered.astype(bool)]
    c = a[a.system_id.astype(int).isin(set(rec.system_id.astype(int))) & (a.system_id.astype(int) != sid)].copy()
    if not len(c):
        return []
    c["d"] = (c.length_km - L).abs()
    return [int(x) for x in c.sort_values(["d", "system_id"]).system_id.head(n)]


def category(found: bool, covered: bool, hit_share: float, recovered: bool) -> str:
    if not found:
        return "NOT_FOUND_IN_UNIT"
    if recovered:
        return "RECOVERED"
    if not covered:
        return "NOT_IN_INVENTORY"
    return "PARTIAL_SELECTION" if (hit_share or 0) > 0 else "IN_INVENTORY_NOT_SELECTED"


def observation(m: dict) -> str:
    if m["category"] == "NOT_FOUND_IN_UNIT":
        return "System id not present among the scored truth pixels of this V1 unit."
    return (f"{m['length_km']:.1f} km, {m['n_px']} scored px; W2 pixel rank on the system median "
            f"{m['w2_rank_median']:.2f} (max {m['w2_rank_max']:.2f}); inventory near-share {m['inventory_near_share']:.2f}; "
            f"frozen-policy support {m['support_mean']:.2f}, hit share {m['hit_share']:.2f}; "
            f"{m['n_segments_near']} candidate segment(s) within 300 m (proposers: {m['proposers'] or 'none'}); nearest "
            f"selected pixel {m['nearest_selected_km']:.2f} km.")


def hypothesis(m: dict) -> str:
    """Rule-based reading of the measured numbers. Always prefixed HYPOTHESIS; never used as a result."""
    c = m["category"]
    if c == "NOT_FOUND_IN_UNIT":
        return "HYPOTHESIS: none (system not in the unit; check the zone mapping of the G05 id)."
    if c == "NOT_IN_INVENTORY":
        return ("HYPOTHESIS: generation-limited - no candidate traces it at the pool budget; a ranker cannot recover "
                "it, only a new candidate source could.")
    if c == "IN_INVENTORY_NOT_SELECTED":
        lo = m["w2_rank_median"] < 0.9
        return ("HYPOTHESIS: ranking-limited - candidates exist but W2 ranks them " +
                ("low; a residual ranker using non-W2 evidence could lift them, if that evidence is present on the "
                 "system (see panels 2-4)." if lo else "high yet below the budget cut; budget / representation, not "
                 "evidence, is the limiting factor."))
    if c == "PARTIAL_SELECTION":
        return ("HYPOTHESIS: localisation / continuity-limited - part of the system is selected; continuity (graph) or "
                "partial-segment selection could complete it.")
    return "HYPOTHESIS: none (recovered: control system)."


# =============================================================================== Colab side
def window(su, sid: int, pad_px: int) -> tuple | None:
    m = np.asarray(su.sysmap) == int(sid)
    if not m.any():
        return None
    rr, cc = np.nonzero(m)
    H, W = m.shape
    return (slice(max(rr.min() - pad_px, 0), min(rr.max() + pad_px + 1, H)),
            slice(max(cc.min() - pad_px, 0), min(cc.max() + pad_px + 1, W)))


def system_metrics(G, su, sid: int, L: dict, lab: np.ndarray, T: pd.DataFrame, q: np.ndarray, cov: pd.DataFrame,
                   hs: pd.DataFrame, tol_px: float) -> dict:
    from scipy import ndimage

    m = np.asarray(su.sysmap) == int(sid)
    base = {"unit": su.key, "zone": su.zone, "system_id": int(sid)}
    if not m.any():
        return {**base, "category": "NOT_FOUND_IN_UNIT"}
    w2 = L.get("w2")
    v = w2[m] if w2 is not None else np.array([np.nan])
    sel = np.asarray(q) > 0
    d_sel = ndimage.distance_transform_edt(~sel)[m].min() * 0.1 if sel.any() else np.nan
    near = ndimage.binary_dilation(m, iterations=int(tol_px))
    labs = sorted(set(np.unique(lab[near])) - {0})
    prop = T[T.label.isin(labs)].proposed_by.astype(str) if "proposed_by" in T else pd.Series(dtype=str)
    fams = sorted({p for s in prop for p in s.split(";") if p})
    h = hs[hs.system_id == int(sid)]
    c = cov[(cov.unit == su.key) & (cov.system_id.astype(int) == int(sid)) & (cov.pool == "inventory")]
    out = {**base, "n_px": int(m.sum()), "length_km": float(m.sum() * 0.112),
           "w2_rank_median": float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan,
           "w2_rank_max": float(np.nanmax(v)) if np.isfinite(v).any() else np.nan,
           "inventory_near_share": float(c.near_share.iloc[0]) if len(c) else np.nan,
           "covered": bool(c.covered.iloc[0]) if len(c) else False,
           "support_mean": float(h.support_mean.iloc[0]) if len(h) else 0.0,
           "hit_share": float(h.hit_share.iloc[0]) if len(h) else 0.0,
           "recovered": bool(h.recovered.iloc[0]) if len(h) else False,
           "selected_px_in_window": int(sel[window(su, sid, 30)].sum()) if window(su, sid, 30) else 0,
           "nearest_selected_km": float(d_sel), "n_segments_near": len(labs), "proposers": ";".join(fams)}
    for k in ("cnn_b0", "cnn_b1", "mag_hgrad", "grav_hgrad", "g02c", "g01_persistence", "pf_persistence", "network"):
        a = L.get(k)
        out[f"{k}_median"] = float(np.nanmedian(a[m])) if a is not None and np.isfinite(a[m]).any() else np.nan
    out["category"] = category(True, out["covered"], out["hit_share"], out["recovered"])
    return out


def write_panels(G, su, sid: int, L: dict, lab: np.ndarray, T: pd.DataFrame, q: np.ndarray, role: str, pad_px: int,
                 rel_dir: str) -> list[str]:
    """Five PNGs for one system (returns relative paths). Truth outline drawn last, labelled post-hoc."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = window(su, sid, pad_px)
    if w is None:
        return []
    truth = np.asarray(su.sysmap)[w] == int(sid)
    other = (np.asarray(su.sysmap)[w] >= 0) & ~truth
    known = np.asarray(su.known)[w]
    out = []

    def show(ax, a, title, cmap="viridis"):
        ax.imshow(a, cmap=cmap, interpolation="nearest")
        ax.contour(truth, levels=[0.5], colors="red", linewidths=0.8)
        if other.any():
            ax.contour(other, levels=[0.5], colors="orange", linewidths=0.5)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    def get(k):
        a = L.get(k)
        return None if a is None else np.asarray(a)[w]

    specs = {"1_learned": [("w2", "W2 pixel rank"), ("_q", "frozen W2 selection")],
             "2_cnn": [("cnn_b0", "CNN B0 rank"), ("cnn_b1", "CNN B1 rank")],
             "3_potential": [("mag_hgrad", "magnetic |grad| rank"), ("grav_hgrad", "gravity |grad| rank")],
             "4_structural": [("g02c", "G02 continuity"), ("g01_persistence", "G01 persistence"),
                              ("pf_persistence", "PF persistence")],
             "5_candidates": [("_lab", "candidate segments"), ("_known", "permitted known faults")]}
    for name, items in specs.items():
        fig, axs = plt.subplots(1, len(items), figsize=(3.2 * len(items), 3.4))
        axs = np.atleast_1d(axs)
        for ax, (k, t) in zip(axs, items):
            if k == "_q":
                show(ax, (np.asarray(q)[w] > 0).astype(float), t, "Greys")
            elif k == "_lab":
                show(ax, np.where(lab[w] > 0, (lab[w] % 17) + 1, np.nan), t, "tab20")
            elif k == "_known":
                show(ax, known.astype(float), t, "Greys")
            else:
                a = get(k)
                if a is None or not np.isfinite(a).any():
                    ax.text(0.5, 0.5, f"{k}: UNAVAILABLE", ha="center", va="center", fontsize=8)
                    ax.set_axis_off()
                else:
                    show(ax, a, t)
        fig.suptitle(f"{su.key} system {sid} ({role}) - red: this system, orange: other systems "
                     "(post-hoc truth overlay)", fontsize=8)
        fig.tight_layout()
        rel = f"{rel_dir}/{su.key.replace(':', '_')}_sys{sid}_{name}.png"
        p = G.ws.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=110)
        plt.close(fig)
        out.append(rel)
    return out


def browser_html(M: pd.DataFrame, panels: dict) -> str:
    """Self-contained case browser (relative PNG links)."""
    rows = []
    for r in M.itertuples():
        imgs = "".join(f'<a href="../{p}"><img src="../{p}" height="120"></a>'
                       for p in panels.get((r.unit, int(r.system_id)), []))
        rows.append(f"<tr><td>{html.escape(str(r.unit))}</td><td>{int(r.system_id)}</td><td>{html.escape(r.role)}</td>"
                    f"<td>{html.escape(r.category)}</td><td>{html.escape(str(getattr(r, 'observation', '')))}</td>"
                    f"<td><i>{html.escape(str(getattr(r, 'hypothesis', '')))}</i></td><td>{imgs}</td></tr>")
    return ("<!doctype html><meta charset='utf-8'><title>G05.1 failure atlas</title><style>body{font:13px sans-serif;"
            "margin:16px}td{vertical-align:top;border-top:1px solid #ccc;padding:4px}</style><h1>G05.1 failure atlas"
            "</h1><p>Observations are measured numbers; HYPOTHESIS lines are rule-based readings, not findings. Truth "
            "outlines are post-hoc.</p><table><tr><th>unit</th><th>system</th><th>role</th><th>category</th>"
            "<th>observation</th><th>hypothesis</th><th>panels</th></tr>" + "".join(rows) + "</table>")
