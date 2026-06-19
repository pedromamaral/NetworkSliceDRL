"""Lightweight metrics accumulator for a training window.

Usage::

    tracker = MetricsTracker()
    for step in episode:
        tracker.update(reward, info)
    summary = tracker.summarise()   # dict of KPIs
    tracker.reset()                 # start next window
"""
from __future__ import annotations


class MetricsTracker:
    """Accumulates per-step statistics and computes window KPIs."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._total: int = 0
        self._admitted: int = 0
        self._fulfilled: int = 0
        self._revenue: float = 0.0
        self._sla_violations: int = 0
        self._ep_rewards: list[float] = []
        self._cur_ep_reward: float = 0.0
        self._utilization_samples: list[float] = []

    # ------------------------------------------------------------------

    def update(self, reward: float, info: dict) -> None:
        """Call once per environment step."""
        self._total += 1
        self._cur_ep_reward += reward
        if info.get("admitted"):
            self._admitted += 1
            self._revenue += reward
            if not info.get("sla_ok", True):
                self._sla_violations += 1
            else:
                self._fulfilled += 1
        util = info.get("avg_path_utilization")
        if util is not None:
            self._utilization_samples.append(util)

    def end_episode(self) -> None:
        """Signal end of episode so per-episode reward is recorded."""
        self._ep_rewards.append(self._cur_ep_reward)
        self._cur_ep_reward = 0.0

    # ------------------------------------------------------------------

    def summarise(self) -> dict:
        """Return a dict of KPIs for the current accumulation window."""
        n_eps = len(self._ep_rewards) or 1
        n_util = len(self._utilization_samples) or 1
        return {
            "acceptance_ratio": self._admitted / max(self._total, 1),
            "fulfillment_ratio": self._fulfilled / max(self._admitted, 1),
            "revenue": self._revenue,
            "sla_violation_rate": self._sla_violations / max(self._admitted, 1),
            "avg_path_utilization": sum(self._utilization_samples) / n_util,
            "mean_ep_reward": sum(self._ep_rewards) / n_eps,
        }
