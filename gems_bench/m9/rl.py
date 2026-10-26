"""Section 13 (P2, optional): reinforcement-learning tracing as a CHALLENGER to beam search, never a dependency.

Environment (``TraceCore``; ``make_env`` wraps it as a Gymnasium env when gymnasium is installed)
    state        position, heading, gap counter, step count, branch count
    observation  compact, rotation-relative features: dense evidence sampled 1..6 px ahead along the straight / left
                 / right headings, the predicted-orientation alignment of each heading, the best evidence reachable
                 by a 2-4 px jump inside a +-30 deg cone, gap counter, step fraction, branch count
    actions      0 advance, 1 turn left, 2 turn right, 3 connect (jump across a gap to the best pixel in the cone),
                 4 branch (record a branch hypothesis here, keep going), 5 stop
    reward       (training episodes only) +on_target for each new pixel within 1 px of the target trace, off_target
                 otherwise, a step cost, +connect for a jump that lands on the target, +stop_near_end for stopping
                 within 3 px of the target's end.
Leakage: episodes are built from OTHER zones' held-out dense predictions and THEIR truth (a 2 km buffer around the
evaluated unit is removed); the evaluated unit's truth never enters a reward.

Learners: behavioural cloning (LightGBM multiclass on expert actions = greedy moves along the target trace),
PPO / Double-DQN via stable-baselines3 when installed (``rl.algorithm``), and a learned edge cost (logistic regression
of "pixel on a fault" from label-free dense heads, trained on the other zones) used as the evidence of an ordinary
beam search. Every challenger is scored exactly like beam search (same seeds, deduplication, oracle); Gate 5 promotes
none unless it adds held-out systems / DTI over beam search.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

ACTIONS = ("advance", "turn_left", "turn_right", "connect", "branch", "stop")
DISTS = (1, 2, 3, 4, 5, 6)
N_OBS = 3 * (len(DISTS) + 1) + 4                                         # obs() length


class TraceCore:
    def __init__(self, E: np.ndarray, theta: np.ndarray, valid: np.ndarray, rc: dict, target: np.ndarray | None = None):
        self.E, self.theta, self.valid, self.rc = E, theta, valid, rc
        self.turn = np.radians(float(rc["turn_deg"]))
        self.T = int(rc["max_steps"])
        self.target = target
        if target is not None and target.any():
            self.tdist = ndimage.distance_transform_edt(~target)
            from ..geology.skeleton import endpoints, skeletonize

            self.tend = ndimage.binary_dilation(endpoints(skeletonize(target)), iterations=3)
        else:
            self.tdist, self.tend = None, None

    def reset(self, r: float, c: float, phi: float):
        self.r, self.c, self.phi, self.gaps, self.t, self.nb = float(r), float(c), float(phi), 0, 0, 0
        self.path, self.branches, self.visited = [(self.r, self.c)], [], set()
        return self.obs()

    def _e(self, r, c):
        H, W = self.E.shape
        if not (0 <= r <= H - 1 and 0 <= c <= W - 1):
            return 0.0
        return float(ndimage.map_coordinates(self.E, [[r], [c]], order=1)[0])

    def _ahead(self, phi, d):
        return self.r - np.cos(phi) * d, self.c + np.sin(phi) * d

    def _align(self, phi, r, c):
        H, W = self.E.shape
        ri, ci = int(np.clip(round(r), 0, H - 1)), int(np.clip(round(c), 0, W - 1))
        return abs(np.cos(phi - self.theta[ri, ci]))

    def _cone_best(self):
        best = (0.0, None)
        for d in (2, 3, 4):
            for a in np.radians([-30, -15, 0, 15, 30]):
                r, c = self._ahead(self.phi + a, d)
                e = self._e(r, c)
                if e > best[0]:
                    best = (e, (r, c, self.phi + a))
        return best

    def obs(self) -> np.ndarray:
        f = []
        for dphi in (0.0, self.turn, -self.turn):
            ph = self.phi + dphi
            for d in DISTS:
                f.append(self._e(*self._ahead(ph, d)))
            f.append(self._align(ph, *self._ahead(ph, 1)))
        f.append(self._cone_best()[0])
        f += [self.gaps / max(int(self.rc.get("max_gap_px", 6)), 1), self.t / self.T, float(self.nb)]
        return np.asarray(f, np.float32)

    def step(self, a: int):
        rw = self.rc["reward"]
        done, reward = False, -abs(float(rw["step"]))                     # step cost
        if ACTIONS[a] == "stop":
            done = True
            if self.tend is not None:
                H, W = self.E.shape
                ri, ci = int(np.clip(round(self.r), 0, H - 1)), int(np.clip(round(self.c), 0, W - 1))
                reward += float(rw["stop_near_end"]) if self.tend[ri, ci] else 0.0
            return self.obs(), reward, done
        if ACTIONS[a] == "branch":
            self.branches.append((self.r, self.c, self.phi + self.turn * 2))
            self.nb += 1
        if ACTIONS[a] == "connect":
            e, best = self._cone_best()
            if best is not None:
                self.r, self.c, self.phi = best
        else:
            if ACTIONS[a] == "turn_left":
                self.phi += self.turn
            elif ACTIONS[a] == "turn_right":
                self.phi -= self.turn
            self.r, self.c = self._ahead(self.phi, 1.0)
        self.t += 1
        H, W = self.E.shape
        ri, ci = int(round(self.r)), int(round(self.c))
        if not (0 <= ri < H and 0 <= ci < W) or not self.valid[ri, ci] or self.t >= self.T:
            return self.obs(), reward, True
        self.path.append((self.r, self.c))
        self.gaps = self.gaps + 1 if self._e(self.r, self.c) < float(self.rc.get("gap_thr", 0.0)) else 0
        if self.tdist is not None:
            on = self.tdist[ri, ci] <= 1.0 and (ri, ci) not in self.visited
            reward += float(rw["on_target"]) if on else float(rw["off_target"])
            if on and ACTIONS[a] == "connect":
                reward += float(rw["connect"])
            self.visited.add((ri, ci))
        if self.gaps > int(self.rc.get("max_gap_px", 6)):
            done = True
        return self.obs(), reward, done


def make_env(core_factory):
    """Gymnasium wrapper (import only when gymnasium is installed)."""
    import gymnasium as gym

    class TraceEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.core, self.start = core_factory()
            n = len(self.core.reset(*self.start))
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
            self.action_space = gym.spaces.Discrete(len(ACTIONS))

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.core, self.start = core_factory()
            return self.core.reset(*self.start), {}

        def step(self, a):
            o, r, d = self.core.step(int(a))
            return o, r, d, False, {}

    return TraceEnv


def expert_action(core: TraceCore) -> int:
    """Greedy move that keeps closest to the unvisited target trace; stop when nothing on-target is reachable."""
    best_a, best_d = 5, np.inf
    H, W = core.E.shape
    for a in (0, 1, 2):
        ph = core.phi + (core.turn if a == 1 else -core.turn if a == 2 else 0.0)
        r, c = core._ahead(ph, 1.0)
        ri, ci = int(round(r)), int(round(c))
        if not (0 <= ri < H and 0 <= ci < W):
            continue
        d = core.tdist[ri, ci] + (0.5 if (ri, ci) in core.visited else 0.0)
        if d < best_d:
            best_a, best_d = a, d
    if best_d > 1.5:
        e, b = core._cone_best()
        if b is not None:
            ri, ci = int(round(b[0])), int(round(b[1]))
            if 0 <= ri < H and 0 <= ci < W and core.tdist[ri, ci] <= 1.0 and (ri, ci) not in core.visited:
                return 3
        return 5
    return best_a


# =============================================================================== episodes from other zones
def episode_sources(m9, u_eval, preds: dict) -> list[dict]:
    """(E, theta, valid, target) crops of every OTHER unit, minus a 2 km buffer around ``u_eval``."""
    from .tracing import fields

    H, W = m9.ctx.data.shape
    ez = np.zeros((H, W), bool)
    ez[u_eval.sl] = u_eval.ev
    buf = ndimage.distance_transform_edt(~ez, sampling=(0.1, 0.1)) <= 2.0
    out = []
    for key, pred in preds.items():
        u = m9.unit(key)
        if u is None or u.key == u_eval.key:
            continue
        ok = u.scored_mask() & ~buf[u.sl]
        tgt = u.eval_truth() & ok
        if tgt.sum() < 50:
            continue
        E, theta, _ = fields(pred, ok, float(m9.cfg["tracing"]["evidence_sigma_px"]))
        out.append({"unit": u.key, "E": E, "theta": theta, "valid": ok, "target": tgt})
    return out


def collect_bc(m9, sources: list[dict], n_episodes: int, rng) -> tuple[np.ndarray, np.ndarray]:
    from ..geology.skeleton import endpoints, skeletonize

    rc = {**m9.cfg["rl"], "max_gap_px": m9.cfg["tracing"]["max_gap_px"]}
    X, y = [], []
    per = max(n_episodes // max(len(sources), 1), 1)
    for s in sources:
        sk = skeletonize(s["target"])
        eps = np.argwhere(endpoints(sk))
        if not len(eps):
            continue
        core = TraceCore(s["E"], s["theta"], s["valid"], rc, s["target"])
        for i in rng.choice(len(eps), min(per, len(eps)), replace=False):
            r, c = eps[i]
            nb = np.argwhere(sk[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4]) + [max(r - 3, 0), max(c - 3, 0)]
            if len(nb) < 2:
                continue
            v = nb.mean(0) - (r, c)
            phi = float(np.arctan2(v[1], -v[0]))
            o = core.reset(r, c, phi)
            for _ in range(core.T):
                a = expert_action(core)
                X.append(o)
                y.append(a)
                o, _, d = core.step(a)
                if d:
                    break
    return (np.asarray(X, np.float32), np.asarray(y, np.int16)) if X else (np.zeros((0, N_OBS), np.float32),
                                                                           np.zeros(0, np.int16))


def train_bc(X: np.ndarray, y: np.ndarray, seed: int):
    import lightgbm as lgb

    if len(np.unique(y)) < 2:
        return None
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1).fit(X, y)


def train_sb3(m9, sources: list[dict], algo: str, steps: int, seed: int):
    """PPO / DQN (stable-baselines3) on random target-endpoint starts of the training sources; None if unavailable."""
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError:
        return None
    rc = {**m9.cfg["rl"], "max_gap_px": m9.cfg["tracing"]["max_gap_px"]}
    rng = np.random.default_rng(seed)
    from ..geology.skeleton import endpoints, skeletonize

    starts = []
    for s in sources:
        e = np.argwhere(endpoints(skeletonize(s["target"])))
        starts += [(s, tuple(p)) for p in e]
    if not starts:
        return None

    def factory():
        s, (r, c) = starts[int(rng.integers(len(starts)))]
        return TraceCore(s["E"], s["theta"], s["valid"], rc, s["target"]), (r, c, float(s["theta"][r, c]))
    Env = make_env(factory)
    model = (PPO if algo == "ppo" else DQN)("MlpPolicy", Env(), seed=seed, verbose=0)
    model.learn(total_timesteps=int(steps))
    return model


def rollout(policy, E, theta, valid, seeds_rc: np.ndarray, rc: dict) -> list[np.ndarray]:
    """Both-direction rollouts of a policy (``predict(obs)`` -> action) from the same seeds as beam search."""
    from .tracing import to_pixels

    core = TraceCore(E, theta, valid, rc)
    out = []
    for r, c in seeds_rc:
        parts = []
        for flip in (0.0, np.pi):
            o = core.reset(r, c, float(theta[r, c]) + flip)
            for _ in range(core.T):
                a = int(np.asarray(policy(o)).ravel()[0])
                o, _, d = core.step(a)
                if d:
                    break
            parts.append(np.asarray(core.path, np.float32))
        full = np.r_[parts[1][::-1], parts[0][1:]]
        out.append(to_pixels(full, E.shape))
    return out


def learned_edge_cost(sources: list[dict], preds_eval: dict, seed: int):
    """Logistic regression P(pixel within 1 px of a fault | label-free heads) fitted on the other zones."""
    from sklearn.linear_model import LogisticRegression

    X, y = [], []
    rng = np.random.default_rng(seed)
    for s in sources:
        m = s["valid"]
        tgt = ndimage.binary_dilation(s["target"], iterations=1)
        idx = np.flatnonzero(m.ravel())
        idx = rng.choice(idx, min(idx.size, 60_000), replace=False)
        X.append(_edge_features(s["E"])[idx])
        y.append(tgt.ravel()[idx])
    if not X:
        return None
    X, y = np.concatenate(X), np.concatenate(y)
    if y.min() == y.max():
        return None
    return LogisticRegression(max_iter=500, class_weight="balanced").fit(X, y)


def _edge_features(E: np.ndarray) -> np.ndarray:
    """(n_pixels, 3): evidence, its square, 5 x 5 local mean (label-free)."""
    return np.stack([E.ravel(), E.ravel() ** 2, ndimage.uniform_filter(E, 5).ravel()], 1)


def apply_edge_cost(model, E: np.ndarray) -> np.ndarray:
    return model.predict_proba(_edge_features(E))[:, 1].reshape(E.shape).astype(np.float32)


def gate5_rl(m9, comp: pd.DataFrame) -> dict:
    g = m9.cfg["gates"]["g5_extensions"]
    base = {"gate": "G5_extensions", "component": "rl", "task": m9.prim, "profile": m9.profile["name"],
            "provisional": bool(m9.profile["provisional"])}
    if not len(comp):
        return {**base, "decision": "NOT_EVALUATED", "reason": "no RL comparison"}
    dens = float(m9.cfg["tracing"]["oracle"]["gate_density"])
    c = comp[(comp.task == m9.prim) & (comp.density_cap == dens)]
    b = c[c.tracer == "beam"]
    best = c[c.tracer != "beam"].sort_values("system_recall", ascending=False).head(1)
    if not len(b) or not len(best):
        return {**base, "decision": "NOT_EVALUATED", "reason": "missing tracer rows"}
    gain_sys = int(best.systems_recovered.iloc[0] - b.systems_recovered.iloc[0])
    ok = gain_sys >= int(g["min_unique_systems"])
    return {**base, "decision": "PROMOTE_CHALLENGER" if ok else "RETAIN_BEAM", "challenger": best.tracer.iloc[0],
            "systems_gain_vs_beam": gain_sys,
            "rule": f"challenger recovers >= {g['min_unique_systems']} more held-out systems than beam at {dens:g} km/100 km^2"}
