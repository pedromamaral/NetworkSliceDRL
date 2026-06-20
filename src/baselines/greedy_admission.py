"""Greedy admission baseline.

Admits every arriving slice request that is feasible on at least one of the
K pre-computed paths.  Among all feasible paths, chooses the one with the
highest bottleneck capacity (max–min available bandwidth across connections).

The current network state is read directly from the MDP observation vector,
so no special environment access is needed.

Interface mirrors the DRL agents: ``select_action(state) -> action``.
"""
from __future__ import annotations

import numpy as np


class GreedyAdmission:
    """Admit all feasible requests; route via the path with maximum bottleneck.

    Args:
        mode: ``"unified"`` or ``"separated"``.
        V:    Number of endpoint nodes (must match env.V).
        K:    Number of pre-computed shortest paths (must match env.K).
    """

    def __init__(self, mode: str = "unified", V: int = 21, K: int = 3) -> None:
        if mode not in ("unified", "separated"):
            raise ValueError(f"mode must be 'unified' or 'separated', got {mode!r}")
        self.mode = mode
        self.V = V
        self.K = K
        # Offsets into the flat observation vector (matches NetworkEnv._get_obs)
        self._bw_idx = 2                          # bandwidth is state[2]
        self._mt_start = 4                        # Mt_flat starts at index 4
        self._mt_end = 4 + V * V                  # Mt_flat ends here
        self._b_start = 4 + V * V + 2             # B_flat starts after active_counts
        self._b_end = 4 + V * V + 2 + V * V * K  # B_flat ends here

    # ------------------------------------------------------------------
    # Public API — compatible with DRL agent interface
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int | tuple[int, int]:
        """Return the action that admits via the best feasible path.

        Decodes bandwidth demand, M_t, and B from the observation vector,
        then finds the path index k that maximises the bottleneck capacity
        across all logical connections required by the slice.

        Returns a reject action only when no path can satisfy all connections.
        """
        bw = float(state[self._bw_idx])
        Mt = state[self._mt_start:self._mt_end].reshape(self.V, self.V)
        B = state[self._b_start:self._b_end].reshape(self.V, self.V, self.K)

        connections = [
            (i, j)
            for i in range(self.V)
            for j in range(self.V)
            if Mt[i, j] > 0.5
        ]

        best_k: int | None = None
        best_bottleneck: float = -1.0

        for k in range(self.K):
            if not connections:
                # Degenerate slice with no connections: admit on k=0.
                best_k = 0
                break
            bottleneck = float(min(B[i, j, k] for i, j in connections))
            if bottleneck >= bw and bottleneck > best_bottleneck:
                best_bottleneck = bottleneck
                best_k = k

        if best_k is None:
            return 0 if self.mode == "unified" else (0, 0)

        return best_k + 1 if self.mode == "unified" else (1, best_k)

    # Dummy store/learn to satisfy train-loop duck-typing if needed.
    def store(self, *args, **kwargs) -> None:
        pass

    def learn(self) -> None:
        return None

    def update_target(self) -> None:
        pass
