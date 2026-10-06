"""Trunk-reservation admission with configurable routing (WP1 factorial cells).

Classical loss-network control (Key 1990; Gibbens, Kelly & Key) protects
future admissibility with a single analytic rule: accept a request on a path
only if, after reserving it, every link on that path keeps at least a fraction
`theta` of its capacity free. That is the hand-designed analogue of the
"admissibility preservation" the learned agents discover, so it is the
baseline a referee from the loss-network community will ask for: if a one-
parameter reservation rule matches the learned admission policy, learning
admission is not what pays.

Routing modes
-------------
  shortest : always path k=0 (theta=0 reproduces "greedy admission + shortest")
  widest   : path with the largest bottleneck (theta=0 reproduces Greedy)
  dar      : dynamic alternative routing -- the direct (shortest) path is used
             whenever it is feasible, with NO reservation; only overflow onto
             an alternate path is subject to the reservation threshold. This
             is the classic trunk-reservation scheme for alternate routing.

`theta` is calibrated per substrate on the TRAINING seed (calibrate_theta) and
then frozen for held-out evaluation, mirroring how the learned agents are
trained on one seed and evaluated on another.

The policy reads the environment directly (like RevenueHeuristic), because the
reservation test needs per-link capacities, which the observation does not
expose. That is a privilege the learned agents do not have; it can only make
this baseline stronger.
"""
from __future__ import annotations

import numpy as np

ROUTING_MODES = ("shortest", "widest", "dar")


class TrunkReservation:
    def __init__(self, env, theta: float = 0.0, routing: str = "widest",
                 mode: str = "unified") -> None:
        if routing not in ROUTING_MODES:
            raise ValueError(f"routing must be one of {ROUTING_MODES}, got {routing!r}")
        if mode not in ("unified", "separated"):
            raise ValueError(f"mode must be 'unified' or 'separated', got {mode!r}")
        self.env = env
        self.theta = float(theta)
        self.routing = routing
        self.mode = mode
        self.eps = 0.0  # interface parity with learned agents

    # ------------------------------------------------------------------

    def _connections(self):
        M = self.env.current_request["Mt"]
        V = self.env.V
        return [(i, j) for i in range(V) for j in range(V) if M[i, j] == 1]

    def _path_edges(self, i: int, j: int, k: int):
        topo = self.env.topo
        pl = topo.paths.get((topo.nodes[i], topo.nodes[j]), [])
        return pl[k] if k < len(pl) else None

    def _min_headroom(self, k: int) -> float:
        """Smallest post-admission free fraction over all links of path k,
        across every required connection. -inf if path k cannot carry it."""
        topo = self.env.topo
        bw = float(self.env.current_request["bandwidth"])
        worst = np.inf
        conns = self._connections()
        if not conns:
            return -np.inf
        for (i, j) in conns:
            path = self._path_edges(i, j, k)
            if path is None:
                return -np.inf
            for e in path:
                free_after = topo.avail.get(e, 0.0) - bw
                if free_after < 0:
                    return -np.inf
                worst = min(worst, free_after / topo._cap[e])
        return worst

    def _act(self, admit: bool, k: int):
        if self.mode == "unified":
            return (k + 1) if admit else 0
        return (1 if admit else 0, k)

    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray):
        K = self.env.K
        if self.routing == "shortest":
            h = self._min_headroom(0)
            return self._act(h >= self.theta, 0)
        if self.routing == "widest":
            heads = [self._min_headroom(k) for k in range(K)]
            k = int(np.argmax(heads))
            return self._act(heads[k] >= self.theta, k)
        # dar: direct route without reservation, alternates with reservation
        if self._min_headroom(0) >= 0.0:
            return self._act(True, 0)
        alt = [self._min_headroom(k) for k in range(1, K)]
        if not alt:
            return self._act(False, 0)
        k = 1 + int(np.argmax(alt))
        return self._act(alt[k - 1] >= self.theta, k)

    # duck-typing for the train/eval loops
    def store(self, *a, **kw) -> None:
        pass

    def learn(self):
        return None

    def update_target(self) -> None:
        pass


THETA_GRID = (0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30)


def calibrate_theta(make_env, routing: str, eval_fn, episodes: int = 40,
                    grid=THETA_GRID) -> tuple[float, dict]:
    """Pick the theta maximising acceptance on the TRAINING seed.

    make_env() must return a fresh env on the training seed; eval_fn(agent,
    env, episodes) must return a summary dict with 'acceptance_ratio'.
    Returns (best_theta, {theta: acceptance}).
    """
    scores = {}
    for th in grid:
        env = make_env()
        agent = TrunkReservation(env, theta=th, routing=routing)
        scores[th] = eval_fn(agent, env, episodes)["acceptance_ratio"]
    best = max(scores, key=scores.get)
    return best, scores
