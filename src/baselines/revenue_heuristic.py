"""Revenue-threshold admission heuristic.

Admits a slice request only when the *expected revenue* of the slice
exceeds a configurable threshold; otherwise rejects it.  When admitting,
always routes via the shortest path (k=0).

Expected revenue is computed directly from the request features embedded
in the observation vector:

    expected_revenue = duration * price

Observation layout (matches ``NetworkEnv._get_obs``):
    index 0 – slice_type
    index 1 – duration
    index 2 – bandwidth
    index 3 – price
    index 4 … – Mt_flat, active_counts, B_flat  (not used here)
"""
from __future__ import annotations

import numpy as np


class RevenueHeuristic:
    """Admit only if ``duration × price ≥ threshold``.

    Args:
        threshold: Minimum expected revenue to trigger admission.
                   Default ``50.0`` corresponds roughly to a mid-range
                   slice (duration≈10, price≈5) in the default config.
        mode:      ``"unified"`` or ``"separated"``.
    """

    # Indices inside the flat observation vector
    _IDX_DURATION = 1
    _IDX_PRICE = 3

    def __init__(
        self, threshold: float = 50.0, mode: str = "unified"
    ) -> None:
        if mode not in ("unified", "separated"):
            raise ValueError(f"mode must be 'unified' or 'separated', got {mode!r}")
        self.threshold = threshold
        self.mode = mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int | tuple[int, int]:
        """Return admission decision based on expected revenue.

        Admit (route via path 0) if ``duration × price ≥ threshold``;
        reject otherwise.

        Unified:   1 (admit, path 0) or 0 (reject)
        Separated: (1, 0) (admit, path 0) or (0, 0) (reject)
        """
        duration = float(state[self._IDX_DURATION])
        price = float(state[self._IDX_PRICE])
        expected_revenue = duration * price

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
