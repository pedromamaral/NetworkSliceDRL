"""Exponential-pricing admission and routing (WP4.1 comparator).

The online-competitive line of work (Awerbuch, Azar & Plotkin, FOCS 1993, "AAP";
recent slicing variants such as OPA) controls admission AND routing with link prices
that grow exponentially in utilisation:

    price_e = (b / C_e) * (mu ** u_e - 1),     u_e = reserved_e / C_e   (before admission)

A request is routed on the candidate path with the lowest total price and admitted only
if that price does not exceed its benefit. Here the benefit of every slice is the same
(+1, count reward), so the admission rule is  min_k cost(k) <= sigma , with sigma a
threshold in price units. In this simulator one path index k serves every logical
connection of a slice, so cost(k) sums the prices of all connections on path k.

Routing variants (for the routing-gain test, as F for the learned controller):
  cheapest : min-price path among the K candidates (AAP)
  shortest : path k = 0 only (the same pricing rule, no path choice)

(mu, sigma) are calibrated per substrate on the TRAINING seed by grid search and frozen
for held-out evaluation, as theta is for trunk reservation. sigma = inf reduces to
"admit whenever feasible, route on the min-price path".

Like TrunkReservation, this reads per-link capacities from the environment, which the
learned agents do not see; that can only make the comparator stronger.
"""
from __future__ import annotations

import math

import numpy as np

ROUTINGS = ("cheapest", "shortest")
MU_GRID = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
SIGMA_GRID = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, math.inf)


class ExpPricing:
    def __init__(self, env, mu: float = 16.0, sigma: float = math.inf,
                 routing: str = "cheapest", mode: str = "unified") -> None:
        if routing not in ROUTINGS:
            raise ValueError(f"routing must be one of {ROUTINGS}, got {routing!r}")
        if mode != "unified":
            raise ValueError("ExpPricing is a unified-action controller")
        self.env, self.mu, self.sigma, self.routing = env, float(mu), float(sigma), routing
        self.eps = 0.0

    def _cost(self, k: int) -> float:
        """Total price of serving every connection of the request on path k;
        inf if path k is missing or lacks headroom on any link."""
        topo, req = self.env.topo, self.env.current_request
        bw = float(req["bandwidth"])
        M, V = req["Mt"], self.env.V
        conns = [(i, j) for i in range(V) for j in range(V) if M[i, j] == 1]
        if not conns:
            return math.inf
        total = 0.0
        for (i, j) in conns:
            pl = topo.paths.get((topo.nodes[i], topo.nodes[j]), [])
            if k >= len(pl):
                return math.inf
            for e in pl[k]:
                cap, free = topo._cap[e], topo.avail.get(e, 0.0)
                if free < bw:
                    return math.inf
                u = 1.0 - free / cap
                total += (bw / cap) * (self.mu ** u - 1.0)
        return total

    def select_action(self, state: np.ndarray):
        ks = range(self.env.K) if self.routing == "cheapest" else (0,)
        costs = [(self._cost(k), k) for k in ks]
        c, k = min(costs)
        if math.isinf(c) or c > self.sigma:
            return 0
        return k + 1

    def store(self, *a, **kw) -> None:
        pass

    def learn(self):
        return None

    def update_target(self) -> None:
        pass


def calibrate(make_env, routing: str, eval_fn, episodes: int = 40,
              mus=MU_GRID, sigmas=SIGMA_GRID):
    """Grid-search (mu, sigma) maximising acceptance on the TRAINING seed."""
    scores = {}
    for mu in mus:
        for sg in sigmas:
            env = make_env()
            scores[(mu, sg)] = eval_fn(ExpPricing(env, mu, sg, routing), env,
                                       episodes)["acceptance_ratio"]
    best = max(scores, key=scores.get)
    return best, scores
