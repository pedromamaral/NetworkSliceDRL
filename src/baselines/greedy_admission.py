"""Greedy admission baseline.

Always attempts to admit every arriving slice request via the shortest
pre-computed path (k=0).  The environment enforces feasibility; if
insufficient capacity exists the environment will not admit and the
episode reward is zero for that step.

Interface mirrors the DRL agents: ``select_action(state) -> action``.
The ``state`` argument is accepted but ignored — this is a stateless
heuristic.
"""
from __future__ import annotations

import numpy as np


class GreedyAdmission:
    """Accept all requests; always route via the shortest path (k=0).

    Args:
        mode: ``"unified"`` or ``"separated"``.
    """

    def __init__(self, mode: str = "unified") -> None:
        if mode not in ("unified", "separated"):
            raise ValueError(f"mode must be 'unified' or 'separated', got {mode!r}")
        self.mode = mode

    # ------------------------------------------------------------------
    # Public API — compatible with DRL agent interface
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int | tuple[int, int]:
        """Return an action that always tries to admit via path 0.

        Unified:   action = 1  (admit, path index 0 ≡ action 1)
        Separated: action = (1, 0)  (admit=1, path_k=0)
        """
        if self.mode == "unified":
            return 1
        return (1, 0)

    # Dummy store/learn to satisfy train-loop duck-typing if needed.
    def store(self, *args, **kwargs) -> None:  # noqa: D401
        pass

    def learn(self) -> None:
        return None

    def update_target(self) -> None:
        pass
