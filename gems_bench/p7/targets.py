"""P7 training targets: the M9 target set (built from VISIBLE labels only) + centerline and distance targets.

``m9.dataset.build_targets`` (reused, unchanged) gives seg / kern / (ocos, osin, ovalid) / ep / jn / topo_valid /
skel / weight / sysmap from the unit's visible labels (labels inside the permitted geography minus ignored systems).
P7 adds
* ``center``: skeleton of the visible labels dilated by ``dilate_px`` (100 m registration tolerance), uint8;
* ``dist``:   distance to visible geometry clipped at ``clip_px`` and divided by it (0 on a trace, 1 far away).
Both derive from the same visible raster, so the held-out fold / withheld systems never enter a target.
``audit_targets`` checks this containment and is written to ``folds/target_audit.csv``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def centerline_target(visible: np.ndarray, dilate_px: int = 1) -> np.ndarray:
    from ..geology.skeleton import skeletonize

    vis = np.asarray(visible, bool)
    if not vis.any():
        return np.zeros(vis.shape, np.uint8)
    sk = skeletonize(vis)
    if int(dilate_px) > 0:
        sk = ndimage.binary_dilation(sk, structure=np.ones((3, 3), bool), iterations=int(dilate_px))
    return sk.astype(np.uint8)


def distance_target(visible: np.ndarray, clip_px: float = 10.0) -> np.ndarray:
    vis = np.asarray(visible, bool)
    if not vis.any():
        return np.ones(vis.shape, np.float16)
    d = ndimage.distance_transform_edt(~vis)
    return (np.clip(d, 0.0, float(clip_px)) / float(clip_px)).astype(np.float16)


def build_p7_targets(m9, tu, ecfg: dict) -> dict:
    from ..m9.dataset import build_targets

    t = build_targets(m9, tu)
    dc = ecfg["data"]
    t["center"] = centerline_target(tu.visible, int(dc["centerline"]["dilate_px"]))
    t["dist"] = distance_target(tu.visible, float(dc["distance"]["clip_px"]))
    return t


def audit_targets(tu, t: dict, withheld: np.ndarray | None = None) -> dict:
    """Containment checks (pure): every positive target lies on (or within the centerline dilation of) visible
    labels; no positive or loss weight touches withheld systems; cv units carry no weight outside ``allowed``."""
    vis = np.asarray(tu.visible, bool)
    seg = np.asarray(t["seg"]) > 0
    cen = np.asarray(t.get("center", np.zeros_like(seg))) > 0
    w = np.asarray(t["weight"], np.float32) > 0
    near = ndimage.binary_dilation(vis, structure=np.ones((3, 3), bool), iterations=3) if vis.any() else vis
    row = {"unit": tu.name, "kind": tu.kind, "seg_outside_visible_px": int((seg & ~vis).sum()),
           "center_outside_visible_px": int((cen & ~near).sum()), "weight_px": int(w.sum()),
           "positives_px": int(seg.sum())}
    if tu.kind == "cv":
        row["weight_outside_allowed_px"] = int((w & ~np.asarray(tu.allowed, bool)).sum())
        if tu.inner is not None:
            row["inner_overlap_allowed_px"] = int((np.asarray(tu.inner, bool) & np.asarray(tu.allowed, bool)).sum())
        if tu.predict is not None:
            row["heldout_overlap_allowed_px"] = int((np.asarray(tu.predict, bool) & np.asarray(tu.allowed, bool)).sum())
    if withheld is not None:
        G = np.asarray(withheld, bool)
        row["withheld_visible_px"] = int((G & vis).sum())
        row["withheld_weighted_px"] = int((G & w).sum())
    bad = [k for k, v in row.items() if k.endswith("_px") and k not in ("weight_px", "positives_px") and v > 0]
    row["ok"] = not bad
    row["violations"] = ";".join(bad)
    return row
