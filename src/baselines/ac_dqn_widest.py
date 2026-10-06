"""Learned admission + widest-path routing (WP1 factorial cell).

The TNSM work plan decomposes the joint agent's advantage into two parts:
foresight in the ADMISSION decision, and flexibility in the ROUTING decision.
The existing agents leave one cell of that grid empty:

                      shortest path     widest path        learned routing
    greedy admission  greedy-shortest   Greedy             --
    learned admission AC-only DQN       THIS CLASS         Joint / Factored

This agent learns admit/reject exactly like AdmissionOnlyDQN (same network,
loss, schedule) but, when it admits, routes over the candidate path with the
largest feasibility signal phi -- the same rule Greedy uses. It therefore has
the admission foresight of the learned agents AND the path flexibility of
greedy, but cannot make a routing error when a feasible path exists.

Pre-registered prediction (see paper/preregistration_wp1.md): on the
controlled rho family this agent matches or beats the joint agent at high rho,
because the joint agent's advantage over greedy decays to ~0 there -- its
routing errors cancel its admission gain exactly where routing matters.
"""
from __future__ import annotations

import numpy as np

from src.baselines.admission_only_dqn import AdmissionOnlyDQN


class ACDQNWidest(AdmissionOnlyDQN):
    """AdmissionOnlyDQN whose admitted slices are routed widest-path.

    The Q-network still has two outputs (reject/admit); routing is a fixed
    heuristic and is therefore part of the environment from the learner's
    point of view.
    """

    def __init__(self, state_dim: int, cfg: dict, mode: str = "unified",
                 n_paths: int | None = None) -> None:
        super().__init__(state_dim, cfg, mode=mode)
        self.K: int = int(n_paths if n_paths is not None else cfg["k_shortest_paths"])

    def _widest(self, state: np.ndarray) -> int:
        """Index (0-based) of the path with the largest feasibility signal."""
        feas = np.asarray(state[-self.K:], dtype=np.float32)
        return int(np.argmax(feas))

    def select_action(self, state: np.ndarray):
        admit = self._admit_int(state)
        k = self._widest(state)
        if self.mode == "unified":
            return (k + 1) if admit == 1 else 0
        return (admit, k)

    def store(self, s, a, r: float, s_next, done: bool) -> None:
        # The env receives a path index (1..K) in unified mode; the Q-network
        # only knows admit(1)/reject(0), so collapse before storing.
        if isinstance(a, tuple):
            a = a[0]
        self.buf.push(s, 1 if int(a) > 0 else 0, r, s_next, done)
