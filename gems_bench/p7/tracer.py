"""T4 (H4): endpoint-conditioned recurrent fault tracer (``FaultTraceGRU``) vs geometric endpoint linking.

The tracer never reads a flattened raster: at each step it samples a small evidence window ROTATED to the current
heading around the current endpoint position and updates a GRU state.

Evidence channels (label-free, per evaluation-unit crop): P7 centerline probability (fault probability when no
centre head), P7 fault probability, magnetic and gravity horizontal-gradient ranks, the existing candidate map
(connectivity) and a dispersion / uncertainty layer (MC-dropout std, else 1 - orientation confidence).

Step outputs: turn (cos, sin of the heading change), step length in [min_step_px, max_step_px], continue / stop
probability, branch probability (recorded, not followed: branching is out of the predeclared scope).

Bounded rollout (``rollout``): stop at max steps, max continuation length, after more than ``max_gap_px`` of
consecutive unsupported length (evidence < ``min_evidence``), on a self-intersection (loop radius), outside the
evaluation mask (boundary), on a mapped fault (``known: stop``; the scorer masks known faults, so tracing through
them adds nothing scoreable), or when the model's stop probability exceeds 0.5.

Training (``fit``): trajectories are resampled skeleton paths of permitted geometry only. For held-out unit u they
come from the OTHER units of the same task (cross-fitting, as W2), never within ``exclude_buffer_km`` of u's
evaluation zone and never from systems withheld in u; the evidence of those units is their own honest held-out P7
map. Teacher forcing decays linearly from ``teacher_forcing[0]`` to ``[1]`` (scheduled sampling, evaluated as a
separate rollout); the evaluation always uses free rollout.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
STOP_REASONS = ("max_steps", "max_length", "gap", "loop", "boundary", "known", "model_stop")


# =============================================================================== pure geometry
def skeleton_paths(skel: np.ndarray) -> list[np.ndarray]:
    """Ordered pixel paths of a skeleton, split at junctions (>= 3 neighbours); each path starts at an endpoint or a
    junction neighbour. Closed loops without endpoints are ignored."""
    sk = np.asarray(skel, bool)
    H, W = sk.shape
    nb = ndimage.convolve(sk.astype(np.uint8), np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8),
                          mode="constant") * sk
    visited = np.zeros_like(sk)
    starts = [tuple(p) for p in np.argwhere(sk & (nb == 1))] + [tuple(p) for p in np.argwhere(sk & (nb >= 3))]
    paths = []
    for s in starts:
        if nb[s] == 1 and visited[s]:                     # already walked from its other end
            continue
        for dr, dc in ([(0, 0)] if nb[s] == 1 else _N8):
            r0, c0 = s[0] + dr, s[1] + dc
            if not (0 <= r0 < H and 0 <= c0 < W) or not sk[r0, c0] or (visited[r0, c0] and (dr, dc) != (0, 0)):
                continue
            path = [s] if (dr, dc) != (0, 0) else []
            cur = (r0, c0)
            while True:
                path.append(cur)
                visited[cur] = True
                if nb[cur] >= 3 and len(path) > 1:
                    break
                nxt = [(cur[0] + a, cur[1] + b) for a, b in _N8 if 0 <= cur[0] + a < H and 0 <= cur[1] + b < W
                       and sk[cur[0] + a, cur[1] + b] and not visited[cur[0] + a, cur[1] + b]]
                if not nxt:
                    break
                cur = nxt[0]
            if len(path) >= 2:
                paths.append(np.asarray(path, np.float32))
    return paths


def resample(path: np.ndarray, step: float) -> np.ndarray:
    """Points every ``step`` px along a polyline (linear interpolation of arc length)."""
    p = np.asarray(path, np.float64)
    if len(p) < 2:
        return p.astype(np.float32)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.arange(0.0, s[-1] + 1e-9, float(step))
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1).astype(np.float32)


def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-9 else np.array([0.0, 1.0], np.float32)


def endpoint_seeds(binary: np.ndarray, back_px: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """(position, outward heading) of every skeleton endpoint of a candidate map. Heading = endpoint minus the mean of
    the skeleton pixels of its component within ``back_px`` (label-free)."""
    from ..geology.skeleton import endpoints, skeletonize

    sk = skeletonize(np.asarray(binary, bool))
    ep = np.argwhere(endpoints(sk))
    lab, _ = ndimage.label(sk, structure=np.ones((3, 3), bool))
    out = []
    for r, c in ep:
        r0, r1, c0, c1 = max(r - back_px, 0), r + back_px + 1, max(c - back_px, 0), c + back_px + 1
        w = np.argwhere(lab[r0:r1, c0:c1] == lab[r, c]) + [r0, c0]
        if len(w) < 2:
            continue
        out.append((np.array([r, c], np.float32), unit(np.array([r, c]) - w.mean(0))))
    return out


def rasterize(path: np.ndarray, shape) -> np.ndarray:
    from skimage.draw import line

    m = np.zeros(shape, bool)
    p = np.rint(np.asarray(path)).astype(int)
    for a, b in zip(p[:-1], p[1:]):
        rr, cc = line(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        m[rr[ok], cc[ok]] = True
    return m


def rollout(step_fn, start: np.ndarray, heading: np.ndarray, evidence: np.ndarray, inside: np.ndarray,
            known: np.ndarray | None, tc: dict) -> tuple[np.ndarray, str, float]:
    """Bounded tracing from one endpoint. ``step_fn(pos, heading, t) -> (new heading, step px, p_stop)``;
    ``evidence``: (H, W) support in [0, 1]. Returns (path incl. start, stop reason, unsupported length px)."""
    H, W = inside.shape
    pos, h = np.asarray(start, np.float32), unit(heading)
    path, length, gap, unsupported = [pos.copy()], 0.0, 0.0, 0.0
    reason = "max_steps"
    for t in range(int(tc["max_steps"])):
        h_new, step, p_stop = step_fn(pos, h, t)
        if p_stop > 0.5:
            reason = "model_stop"
            break
        step = float(np.clip(step, float(tc["min_step_px"]), float(tc["max_step_px"])))
        h = unit(h_new)
        nxt = pos + step * h
        r, c = int(round(float(nxt[0]))), int(round(float(nxt[1])))
        if not (0 <= r < H and 0 <= c < W) or not inside[r, c]:
            reason = "boundary"
            break
        if known is not None and tc.get("known", "stop") == "stop" and known[r, c]:
            reason = "known"
            break
        if len(path) > 3:
            prev = np.asarray(path[:-3])
            if (np.hypot(prev[:, 0] - nxt[0], prev[:, 1] - nxt[1]) < float(tc["loop_radius_px"])).any():
                reason = "loop"
                break
        ok = float(evidence[r, c]) >= float(tc["min_evidence"])
        gap = 0.0 if ok else gap + step
        if not ok:
            unsupported += step
        if gap > float(tc["max_gap_px"]):
            reason = "gap"
            break
        length += step
        path.append(nxt.astype(np.float32))
        pos = nxt
        if length >= float(tc["max_length_px"]):
            reason = "max_length"
            break
    p = np.asarray(path, np.float32)
    if reason == "gap":                                   # drop the trailing unsupported stretch
        while len(p) > 1:
            r, c = int(round(float(p[-1, 0]))), int(round(float(p[-1, 1])))
            if float(evidence[r, c]) >= float(tc["min_evidence"]):
                break
            p = p[:-1]
    return p, reason, unsupported


def geometric_link(binary: np.ndarray, inside: np.ndarray, max_gap_px: float, max_angle_deg: float) -> np.ndarray:
    """Baseline: join facing endpoint pairs (distance <= max_gap, both headings within max_angle of the joining
    line, different components); returns the ADDED pixels."""
    seeds = endpoint_seeds(binary)
    lab, _ = ndimage.label(np.asarray(binary, bool), structure=np.ones((3, 3), bool))
    add = np.zeros(binary.shape, bool)
    used = set()
    cosmax = np.cos(np.radians(max_angle_deg))
    for i, (pa, ha) in enumerate(seeds):
        best, bj = None, None
        for j, (pb, hb) in enumerate(seeds):
            if j == i or j in used or i in used:
                continue
            if lab[int(pa[0]), int(pa[1])] == lab[int(pb[0]), int(pb[1])]:
                continue
            d = pb - pa
            dist = float(np.hypot(*d))
            if dist > max_gap_px or dist < 1:
                continue
            u = unit(d)
            if float(ha @ u) >= cosmax and float(hb @ -u) >= cosmax and (best is None or dist < best):
                best, bj = dist, j
        if bj is not None:
            used |= {i, bj}
            add |= rasterize(np.stack([pa, seeds[bj][0]]), binary.shape) & inside
    return add & ~np.asarray(binary, bool)


def trajectory_set(skel_paths: list[np.ndarray], step: float, min_px: int, drop_mask: np.ndarray | None) -> list:
    """Resampled training trajectories; points inside ``drop_mask`` split / remove a trajectory."""
    out = []
    for p in skel_paths:
        q = resample(p, step)
        if drop_mask is not None and len(q):
            keep = ~drop_mask[np.clip(np.rint(q[:, 0]).astype(int), 0, drop_mask.shape[0] - 1),
                              np.clip(np.rint(q[:, 1]).astype(int), 0, drop_mask.shape[1] - 1)]
            runs = np.split(q, np.flatnonzero(np.diff(keep.astype(int)) != 0) + 1)
            ks = np.split(keep, np.flatnonzero(np.diff(keep.astype(int)) != 0) + 1)
            q = [r for r, k in zip(runs, ks) if k.all()]
        else:
            q = [q]
        out += [r for r in q if len(r) * step >= min_px and len(r) >= 3]
    return out


# =============================================================================== torch module
def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def make_tracer(n_channels: int, embed_px: int, hidden: int):
    torch, nn, F = _torch()

    class FaultTraceGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = int(embed_px)
            self.embed = nn.Sequential(nn.Flatten(), nn.Linear(n_channels * self.k * self.k, hidden), nn.GELU())
            self.cell = nn.GRUCell(hidden + 2, hidden)
            self.turn = nn.Linear(hidden, 2)
            self.step = nn.Linear(hidden, 1)
            self.stop = nn.Linear(hidden, 1)
            self.branch = nn.Linear(hidden, 1)
            nn.init.zeros_(self.turn.weight)
            with torch.no_grad():
                self.turn.bias.copy_(torch.tensor([1.0, 0.0]))        # start straight

        def window(self, ev, pos, head):
            """ev (C, H, W); pos, head (B, 2) in (row, col) -> rotated (B, C, k, k) evidence windows."""
            k = self.k
            o = torch.arange(k, device=ev.device, dtype=torch.float32) - (k - 1) / 2
            along, across = torch.meshgrid(o, o, indexing="ij")
            n = torch.stack([-head[:, 1], head[:, 0]], 1)
            rr = pos[:, 0, None, None] + along * head[:, 0, None, None] + across * n[:, 0, None, None]
            cc = pos[:, 1, None, None] + along * head[:, 1, None, None] + across * n[:, 1, None, None]
            H, W = ev.shape[-2:]
            g = torch.stack([2 * cc / max(W - 1, 1) - 1, 2 * rr / max(H - 1, 1) - 1], -1)
            return F.grid_sample(ev[None].expand(pos.shape[0], -1, -1, -1), g, align_corners=True)

        def forward(self, ev, pos, head, h):
            e = self.embed(self.window(ev, pos, head))
            h = self.cell(torch.cat([e, head], 1), h)
            t = self.turn(h)
            t = t / (t.norm(dim=1, keepdim=True) + 1e-6)
            new_head = torch.stack([t[:, 0] * head[:, 0] - t[:, 1] * head[:, 1],
                                    t[:, 1] * head[:, 0] + t[:, 0] * head[:, 1]], 1)
            return new_head, self.step(h)[:, 0], self.stop(h)[:, 0], self.branch(h)[:, 0], h

    return FaultTraceGRU()


def fit(model, data: list[tuple], tc: dict, device, log=print) -> list[dict]:
    """``data`` = [(evidence (C, H, W) float32 tensor on device, [trajectories])]. Returns the per-epoch history."""
    torch, _, F = _torch()
    tr = tc["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]))
    E = int(tr["epochs"])
    tf0, tf1 = (float(x) for x in tr["teacher_forcing"])
    hidden = model.cell.hidden_size
    smin, smax = float(tc["min_step_px"]), float(tc["max_step_px"])
    hist = []
    rng = np.random.default_rng(0)
    for ep in range(E):
        tf = tf0 + (tf1 - tf0) * ep / max(E - 1, 1)
        tot, n = 0.0, 0
        for ev, trajs in data:
            order = rng.permutation(len(trajs))
            for i in range(0, len(order), int(tr["batch"])):
                batch = [trajs[j] for j in order[i:i + int(tr["batch"])]]
                L = max(len(t) for t in batch)
                P = np.zeros((len(batch), L, 2), np.float32)
                M = np.zeros((len(batch), L), np.float32)
                for b, t in enumerate(batch):
                    P[b, :len(t)], M[b, :len(t)] = t, 1
                P, M = torch.from_numpy(P).to(device), torch.from_numpy(M).to(device)
                h = torch.zeros(len(batch), hidden, device=device)
                pos = P[:, 1]
                head = F.normalize(P[:, 1] - P[:, 0], dim=1)
                loss = 0.0
                for t in range(1, L - 1):
                    m = M[:, t + 1]
                    tgt = P[:, t + 1] - P[:, t]
                    nh, st, sp, _, h = model(ev, pos, head, h)
                    tdir = F.normalize(tgt, dim=1)
                    l_dir = ((1 - (nh * tdir).sum(1)) * m).sum() / m.sum().clamp_min(1)
                    step = smin + (smax - smin) * torch.sigmoid(st)
                    l_len = (F.smooth_l1_loss(step, tgt.norm(dim=1), reduction="none") * m).sum() / m.sum().clamp_min(1)
                    stop_t = (M[:, t] > 0) & (m == 0)
                    l_stop = F.binary_cross_entropy_with_logits(sp, stop_t.float(), reduction="none")
                    l_stop = (l_stop * M[:, t]).sum() / M[:, t].sum().clamp_min(1)
                    loss = loss + l_dir + 0.1 * l_len + 0.5 * l_stop
                    use_true = torch.rand(len(batch), device=device) < tf
                    pred = pos + step[:, None] * nh
                    pos = torch.where(use_true[:, None], P[:, t + 1], pred.detach())
                    head = torch.where(use_true[:, None], tdir, nh.detach())
                opt.zero_grad(set_to_none=True)
                if torch.is_tensor(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    tot, n = tot + float(loss.detach()), n + 1
        hist.append({"epoch": ep + 1, "teacher_forcing": tf, "loss": tot / max(n, 1), "batches": n})
        log(f"[p7 tracer] epoch {ep + 1}/{E} loss {tot / max(n, 1):.4f} (teacher forcing {tf:.2f})")
    return hist


def model_step_fn(model, ev, device, tc: dict):
    """Adapter for ``rollout``: one GRU step per call (state kept in the closure)."""
    torch, _, _ = _torch()
    state = {"h": torch.zeros(1, model.cell.hidden_size, device=device)}
    smin, smax = float(tc["min_step_px"]), float(tc["max_step_px"])

    def step(pos, head, t):
        with torch.no_grad():
            p = torch.as_tensor(np.asarray(pos, np.float32)[None], device=device)
            hd = torch.as_tensor(np.asarray(head, np.float32)[None], device=device)
            nh, st, sp, _, state["h"] = model(ev, p, hd, state["h"])
        return nh[0].cpu().numpy(), float(smin + (smax - smin) * torch.sigmoid(st)[0]), float(torch.sigmoid(sp)[0])

    return step
