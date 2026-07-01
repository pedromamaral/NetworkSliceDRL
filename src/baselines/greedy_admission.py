"""Greedy admission baseline.

Admits every arriving slice request that is feasible on at least one of the
K pre-computed paths.  Among all feasible paths, chooses the one with the
highest bottleneck capacity (max–min available bandwidth across connections).

Uses the per-path feasibility signal that the environment appends as the last
K entries of the observation vector.  The env computes those from *raw* B and
bandwidth, so this baseline is immune to any normalisation applied to the rest
of the observation (unlike the old decode of B_flat / bandwidth, which broke
when the two were scaled by different constants).

Interface mirrors the DRL agents: ``select_action(state) -> action``.
"""
from __future__ import annotations

import numpy as np

# feasibility[k] = clip(bottleneck_k / bw, 0, 3) / 3, so a path is feasible
# (bottleneck >= bw) iff feasibility[k] >= 1/3.
_FEASIBLE_THRESH = 1.0 / 3.0 - 1e-6


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

    # ------------------------------------------------------------------
    # Public API — compatible with DRL agent interface
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int | tuple[int, int]:
        """Admit via the path with the highest feasibility, or reject if none.

        Reads the K feasibility values (last K entries of the observation),
        picks the path with maximum feasibility, and admits iff that path is
        actually feasible.  Normalisation-proof.
        """
        feasibility = np.asarray(state[-self.K:], dtype=np.float32)
        best_k = int(np.argmax(feasibility))

        if feasibility[best_k] < _FEASIBLE_THRESH:
            return 0 if self.mode == "unified" else (0, 0)

        return best_k + 1 if self.mode == "unified" else (1, best_k)

    # Dummy store/learn to satisfy train-loop duck-typing if needed.
    def store(self, *args, **kwargs) -> None:
        pass

    def learn(self) -> None:
        return None

    def update_target(self) -> None:
        pass
