"""Experiment A: three discovery validation scenarios on the existing units (no new split framework).

* V1 ``disc_official`` units: complete withheld fault systems (``ctx.splits``); CNN channels = the split's discovery
  map (models trained with the systems hidden), W2 = its cross-fitted held-out map; known context = retained faults.
* V2 network completion inside each geographic fold k: real mapped sub-sections of fold-k faults are withheld
  (``withhold_network``: tips, junction branches, short splays, interior sub-sections). Truth = the withheld pixels;
  the retained fold-k faults are the known context AND the pixel-exact exclusion (official convention, as
  ``disc_official``); the evaluation zone is the whole fold (never a ring around the truth). Every source map is the
  fold-k honest map (B0 / B1 out-of-fold, W2 held-out): no model saw any fold-k label, so the withheld pixels reach no
  input. Nothing is invented: no lateral shift, no synthetic trace.
* V3 ``geo`` units: the existing geographic folds (truth = every mapped fault of fold k; known context = faults outside
  fold k only).

``SUnit`` keeps apart the unit that is SCORED (``unit``) and the unit whose honest maps feed the sources (``base``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

S8 = np.ones((3, 3), bool)
SCENARIOS = ("V1", "V2", "V3")
V2_CLASSES = ("tip_extension", "branch", "short_splay", "subsection")
V2_TASK = "v2_net"
PIXEL_KM = 0.1
STEP_KM = 0.112                       # mean digital step (geology.skeleton.component_length_km)


@dataclass
class SUnit:
    scenario: str
    unit: object                      # gems3.official.Unit that is scored
    base: object                      # unit whose honest maps (CNN / W2 / G02) are the sources
    known: np.ndarray                 # cropped permitted known faults (network context; never truth)
    sysmap: np.ndarray                # cropped int32 system / event id on truth pixels, -1 elsewhere
    events: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def zone(self) -> int:
        return int(self.unit.zone)

    @property
    def key(self) -> str:
        return f"{self.scenario}:{self.zone}"


# =============================================================================== pure: V2 withholding
def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _disk_window(shape, r: int, c: int, rad: int) -> tuple[tuple, np.ndarray]:
    """(window slices, disk mask of the window) of radius ``rad`` px around (r, c), clipped at the edges."""
    h, w = shape
    r0, r1, c0, c1 = max(r - rad, 0), min(r + rad + 1, h), max(c - rad, 0), min(c + rad + 1, w)
    return (slice(r0, r1), slice(c0, c1)), _disk(rad)[rad - (r - r0):rad + (r1 - r), rad - (c - c0):rad + (c1 - c)]


def _pad(sl: tuple, shape, p: int = 1) -> tuple:
    return tuple(slice(max(s.start - p, 0), min(s.stop + p, n)) for s, n in zip(sl, shape))


def withhold_network(labels: np.ndarray, zone: np.ndarray, rng, cc: dict) -> tuple[np.ndarray, np.ndarray,
                                                                                     np.ndarray, pd.DataFrame]:
    """(retained, withheld, event id map, event table) of the mapped faults inside ``zone``.

    Only real mapped pixels of ``labels & zone`` are moved to ``withheld``; faults outside the zone are untouched and
    never returned. Classes (``cc['classes']``, ``sample_frac`` of the eligible features each, at most
    ``max_events_per_class``): ``short_splay`` whole components shorter than ``short_km``; ``branch`` skeleton
    branches touching a junction and shorter than ``branch_max_km`` (junction pixels stay); ``tip_extension`` a disk
    of radius ``tip_km`` at a trace tip; ``subsection`` a disk of radius ``subsection_km`` on the interior of a trace
    at least ``min_trace_km`` long. Every event withholds at least one pixel and keeps its own id."""
    from ..geology.skeleton import branch_points, endpoints, skeletonize
    from ..p3.lineaments import split_branches

    lab = np.asarray(labels, bool) & np.asarray(zone, bool)
    keep = lab.copy()
    out = np.zeros(lab.shape, bool)
    eid_map = np.full(lab.shape, -1, np.int32)
    rows, nxt = [], 0
    classes = list(cc.get("classes") or V2_CLASSES)
    frac, cap = float(cc["sample_frac"]), int(cc["max_events_per_class"])
    if not lab.any():
        return keep, out, eid_map, pd.DataFrame(columns=["event_id", "class", "px", "length_km"])
    sk = skeletonize(lab)
    comp, n = ndimage.label(lab, structure=S8)
    sk_len = np.bincount(comp[sk], minlength=n + 1).astype(float) * STEP_KM

    def pick(ids):
        ids = list(ids)
        k = min(int(round(frac * len(ids))), cap)
        return [ids[i] for i in rng.choice(len(ids), k, replace=False)] if k else []

    def add(cls, sl, m, extra=None):
        """Withhold the window mask ``m`` (only still-retained mapped pixels move)."""
        nonlocal nxt
        m = m & keep[sl]
        if not m.any():
            return
        out[sl] |= m
        eid_map[sl][m] = nxt
        keep[sl] &= ~m
        rows.append({"event_id": nxt, "class": cls, "px": int(m.sum()), "length_km": float(m.sum()) * STEP_KM,
                     **(extra or {})})
        nxt += 1

    objs = ndimage.find_objects(comp)
    if "short_splay" in classes:
        ids = [i for i in range(1, n + 1) if 0 < sk_len[i] < float(cc["short_km"])]
        for i in pick(ids):
            sl = objs[i - 1]
            add("short_splay", sl, comp[sl] == i)
    if "branch" in classes:
        bl, _, junction = split_branches((sk & keep).astype(np.uint8), 2)
        jz = ndimage.binary_dilation(junction, structure=S8, iterations=2)
        size = np.bincount(bl.ravel(), minlength=int(bl.max()) + 1)
        touch = set(np.unique(bl[ndimage.binary_dilation(jz, structure=S8) & (bl > 0)]).tolist())
        cand = [i for i in sorted(touch) if size[i] * STEP_KM <= float(cc["branch_max_km"])]
        bobj = ndimage.find_objects(bl)
        for i in pick(cand):
            sl = _pad(bobj[i - 1], lab.shape)
            add("branch", sl, ndimage.binary_dilation(bl[sl] == i, structure=S8) & lab[sl] & ~jz[sl])
    long_ = (sk_len[comp] >= float(cc["min_trace_km"])) & sk
    if "tip_extension" in classes:
        ep = np.argwhere(endpoints(sk) & long_ & keep)
        for j in pick(range(len(ep))):
            r, c = ep[j]
            rad = int(round(float(rng.choice(cc["tip_km"])) / PIXEL_KM))
            sl, d = _disk_window(lab.shape, int(r), int(c), rad)
            add("tip_extension", sl, d, {"radius_km": rad * PIXEL_KM})
    if "subsection" in classes:
        away = ~ndimage.binary_dilation(endpoints(sk) | branch_points(sk), structure=S8, iterations=10)
        pts = np.argwhere(long_ & away & keep)
        n_sub = min(int(round(frac * len(pts) / 50.0)), cap)
        for j in (rng.choice(len(pts), n_sub, replace=False) if n_sub else []):
            r, c = pts[j]
            rad = int(round(float(rng.choice(cc["subsection_km"])) / PIXEL_KM))
            sl, d = _disk_window(lab.shape, int(r), int(c), rad)
            add("subsection", sl, d, {"radius_km": rad * PIXEL_KM})
    E = pd.DataFrame(rows, columns=["event_id", "class", "px", "length_km", "radius_km"])
    return keep, out, eid_map, E


def leakage_checks(su_key: str, truth: np.ndarray, known: np.ndarray, exclude: np.ndarray | None,
                   labels: np.ndarray | None = None) -> list[dict]:
    """Pure checks on one scenario unit: the known context never touches the scored truth; the exclusion never
    removes truth; V2 truth is made of real mapped pixels only."""
    t, k = np.asarray(truth, bool), np.asarray(known, bool)
    out = [{"unit": su_key, "check": "known context shares no pixel with the truth", "ok": not bool((t & k).any()),
            "detail": f"{int((t & k).sum())} px"}]
    if exclude is not None:
        e = np.asarray(exclude, bool)
        out.append({"unit": su_key, "check": "exclusion shares no pixel with the truth", "ok": not bool((t & e).any()),
                    "detail": f"{int((t & e).sum())} px"})
    if labels is not None:
        out.append({"unit": su_key, "check": "every truth pixel is a real mapped pixel",
                    "ok": bool((t <= np.asarray(labels, bool)).all()), "detail": f"{int((t & ~labels).sum())} px"})
    return out


# =============================================================================== pure: distance bands
def band_of(dist_km: np.ndarray, edges) -> np.ndarray:
    """0 = near (<= e0), 1 = mid (<= e1), 2 = far."""
    d = np.asarray(dist_km, float)
    return np.digitize(d, [float(e) + 1e-9 for e in edges]).astype(np.int8)


BANDS = ("near", "mid", "far")


def banded_components(u, q: np.ndarray, dist_km: np.ndarray, edges) -> pd.DataFrame:
    """Official components split by distance to the permitted known faults: TP / FN from the truth pixels of the band,
    FP from the predicted pixels of the band (sums over bands == the unit's components)."""
    from ..metric import clean_prediction

    s = u.scorer
    p = clean_prediction(np.where(u.ev, np.nan_to_num(np.asarray(q, np.float64), nan=0.0), 0.0))
    p[s.zero] = 0.0
    band = band_of(dist_km, edges)
    fp = p * s.fp_w
    rows = []
    sup = s.support(p, cleaned=True) if s.n else np.zeros(0)
    gb = band[s.g_rc] if s.n else np.zeros(0, np.int8)
    for b, name in enumerate(BANDS):
        rows.append({"band": name, "tp_w": float(sup[gb == b].sum()), "fn_w": float((1.0 - sup[gb == b]).sum()),
                     "fp_w": float(fp[band == b].sum()), "n_truth": int((gb == b).sum())})
    return pd.DataFrame(rows)


# =============================================================================== pure: system recovery
def system_recovery(u, q: np.ndarray, sysmap: np.ndarray, support_min: float, frac_min: float) -> pd.DataFrame:
    """Per truth system / event: scored pixels, mean TP support, hit share, recovered (``m7.protocol.system_hits``
    with an explicit id map, so V2 events are scored like systems)."""
    s = u.scorer
    cols = ["system_id", "n_px", "support_mean", "hit_share", "recovered"]
    if not s.n:
        return pd.DataFrame(columns=cols)
    sup = s.support(np.asarray(q, np.float64))
    ids = np.asarray(sysmap)[s.g_rc]
    df = pd.DataFrame({"system_id": ids, "sup": sup, "hit": sup >= float(support_min)})
    df = df[df.system_id >= 0]
    g = df.groupby("system_id").agg(n_px=("sup", "size"), support_mean=("sup", "mean"), hit_share=("hit", "mean"))
    g["recovered"] = g.hit_share >= float(frac_min)
    return g.reset_index()[cols]


# =============================================================================== Colab side
def build(G) -> list[SUnit]:
    """Scenario units of the attached engine (``G``: run.DiscoveryGeoMoE)."""
    from ..gems3.official import Unit
    from ..metric import OFFICIAL_EXCLUDE_SUPPORTS_TP
    from ..p6.stack import permitted_known

    bst, ctx = G.bst, G.ctx
    lab = np.asarray(ctx.data.labels, bool)
    sysm = np.asarray(ctx.splits.system_map)
    out = []
    for u in bst.units():
        if u.task == "disc_official":
            k = permitted_known(ctx, u)[u.sl]
            out.append(SUnit("V1", u, u, k, np.where(u.eval_truth(), sysm[u.sl], -1).astype(np.int32)))
    geo = [u for u in bst.units() if u.task == "geo"]
    for u in geo:
        k = permitted_known(ctx, u)[u.sl]
        out.append(SUnit("V3", u, u, k, np.where(u.eval_truth(), sysm[u.sl], -1).astype(np.int32)))
    cc = G.cfg["v2"]
    for u in geo:
        rng = np.random.default_rng(int(cc["seed"]) + 101 * int(u.zone))
        keep, wh, eid, E = withhold_network(lab[u.sl], u.ev, rng, cc)
        v = Unit(V2_TASK, int(u.zone), u.sl, u.ev.copy(), wh, keep & u.ev, bool(OFFICIAL_EXCLUDE_SUPPORTS_TP), keep)
        ctx_known = permitted_known(ctx, u)[u.sl] | keep          # faults outside fold k + retained fold-k faults
        out.append(SUnit("V2", v, u, ctx_known, eid, E.assign(zone=int(u.zone))))
    return out


def v2_full_known(G, su: SUnit) -> np.ndarray:
    """Full-grid permitted known map of a V2 unit (outside fold k + retained fold-k faults)."""
    from ..p6.stack import permitted_known

    full = permitted_known(G.ctx, su.base).copy()
    full[su.unit.sl] |= su.known
    return full


def scenario_table(units: list[SUnit], cfg: dict) -> pd.DataFrame:
    rows = []
    for su in units:
        u = su.unit
        t = u.eval_truth()
        rows.append({"scenario": su.scenario, "zone": su.zone, "unit": su.key, "task": u.task,
                     "label": cfg["scenarios"][su.scenario]["label"], "base_unit": su.base.key,
                     "eval_px": int(u.ev.sum()), "scored_truth_px": int(t.sum()),
                     "excluded_px": int(u.exclude.sum()) if u.exclude is not None else 0,
                     "known_context_px": int(su.known.sum()),
                     "n_systems": int(len(np.unique(su.sysmap[su.sysmap >= 0]))),
                     "sources": "split discovery CNN + W2 held-out" if su.scenario == "V1" else
                     "fold-k OOF CNN + W2 held-out"})
    return pd.DataFrame(rows)
