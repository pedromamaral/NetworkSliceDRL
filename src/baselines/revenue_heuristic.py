"""Revenue-threshold admission heuristic.

Admits a slice request only when the *expected revenue* of the slice
(``duration × price`` in raw units) exceeds a configurable threshold;
otherwise rejects it.  When admitting, always routes via the shortest
path (k=0).

The raw request is read from the environment (``env.current_request``)
rather than the observation vector, because the observation normalises
``duration`` and ``price`` by different constants — decoding revenue from
it would be scale-dependent.  Holding an env reference keeps the threshold
in interpretable raw revenue units.
"""
from __future__ import annotations

import numpy as np


class RevenueHeuristic:
    """Admit only if ``duration × price ≥ threshold`` (raw units).

    Args:
        threshold: Minimum expected revenue (raw ``d × p``) to admit.
        mode:      ``"unified"`` or ``"separated"``.
        env:       NetworkEnv whose ``current_request`` supplies raw features.
    """

    def __init__(
        self, threshold: float = 500.0, mode: str = "unified", env=None
    ) -> None:
        if mode not in ("unified", "separated"):
            raise ValueError(f"mode must be 'unified' or 'separated', got {mode!r}")
        self.threshold = threshold
        self.mode = mode
        self.env = env

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int | tuple[int, int]:
        """Admit (via path 0) if raw ``duration × price ≥ threshold``."""
        req = self.env.current_request
        expected_revenue = float(req["duration"]) * float(req["price"])

        if expected_revenue >= self.threshold:
            return 1 if self.mode == "unified" else (1, 0)
        return 0 if self.mode == "unified" else (0, 0)

    # Dummy methods for train-loop duck-typing compatibility.
    def store(self, *args, **kwargs) -> None:
        pass

    def learn(self) -> None:
        return None

    def update_target(self) -> None:
        pass
