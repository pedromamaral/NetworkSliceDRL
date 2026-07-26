"""Unit tests for src/baselines/ — Phase 3.

Tests cover all three baselines:
  - GreedyAdmission
  - RevenueHeuristic
  - AdmissionOnlyDQN

All tests run on CPU with tiny state dimensions.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.baselines.greedy_admission import GreedyAdmission
from src.baselines.revenue_heuristic import RevenueHeuristic
from src.baselines.admission_only_dqn import AdmissionOnlyDQN


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

STATE_DIM = 10
BATCH = 4


def _make_state(duration: float = 10.0, price: float = 8.0) -> np.ndarray:
    """Build a minimal state vector with known duration/price at indices 1/3."""
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[1] = duration  # index 1 = duration
    state[3] = price     # index 3 = price
    return state


def _greedy_state(feasibility, K: int = 3, dim: int = STATE_DIM) -> np.ndarray:
    """State whose last K entries hold the per-path feasibility signal.

    GreedyAdmission reads state[-K:] (env appends feasibility[k] = clip(
    bottleneck_k/bw, 0, 3)/3, so path k is feasible iff feasibility[k] >= 1/3).
    """
    state = np.zeros(dim, dtype=np.float32)
    state[-K:] = np.asarray(feasibility, dtype=np.float32)
    return state


class _StubEnv:
    """Minimal env exposing current_request, for RevenueHeuristic tests.

    RevenueHeuristic reads raw duration/price from env.current_request (not the
    normalised observation), so the tests supply a stub rather than a state."""

    def __init__(self, duration: float, price: float) -> None:
        self.current_request = {"duration": duration, "price": price}


def _baseline_cfg() -> dict:
    return {
        "hidden_size": 32,
        "lr": 1e-3,
        "replay_capacity": 32,
        "gamma": 0.99,
        "batch_size": BATCH,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 1000,
    }


# ===========================================================================
# GreedyAdmission
# ===========================================================================


class TestGreedyAdmission:
    def test_unified_admits_max_feasibility_path(self):
        # feasibility [0.2, 0.9, 0.4] → best path index 1 → action = index+1 = 2
        agent = GreedyAdmission(mode="unified", K=3)
        assert agent.select_action(_greedy_state([0.2, 0.9, 0.4])) == 2

    def test_unified_rejects_when_no_path_feasible(self):
        # all feasibility < 1/3 → reject
        agent = GreedyAdmission(mode="unified", K=3)
        assert agent.select_action(_greedy_state([0.1, 0.2, 0.05])) == 0

    def test_separated_admits_max_feasibility_path(self):
        # feasibility [0.2, 0.3, 0.9] → best path index 2 → (1, 2)
        agent = GreedyAdmission(mode="separated", K=3)
        assert agent.select_action(_greedy_state([0.2, 0.3, 0.9])) == (1, 2)

    def test_separated_rejects_when_no_path_feasible(self):
        agent = GreedyAdmission(mode="separated", K=3)
        assert agent.select_action(_greedy_state([0.1, 0.2, 0.05])) == (0, 0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            GreedyAdmission(mode="invalid")

    def test_store_learn_noop(self):
        agent = GreedyAdmission()
        s = np.zeros(STATE_DIM, dtype=np.float32)
        agent.store(s, 1, 1.0, s, False)
        assert agent.learn() is None
        agent.update_target()  # must not raise


# ===========================================================================
# RevenueHeuristic
# ===========================================================================


class TestRevenueHeuristic:
    def test_admits_when_revenue_above_threshold(self):
        # duration=10, price=10 → revenue=100 > 50 → admit
        agent = RevenueHeuristic(threshold=50.0, mode="unified",
                                 env=_StubEnv(duration=10.0, price=10.0))
        assert agent.select_action(_make_state()) == 1

    def test_rejects_when_revenue_below_threshold(self):
        # duration=2, price=5 → revenue=10 < 50 → reject
        agent = RevenueHeuristic(threshold=50.0, mode="unified",
                                 env=_StubEnv(duration=2.0, price=5.0))
        assert agent.select_action(_make_state()) == 0

    def test_admits_at_exact_threshold(self):
        # revenue = 5 * 10 = 50 == threshold → admit (>=)
        agent = RevenueHeuristic(threshold=50.0, mode="unified",
                                 env=_StubEnv(duration=5.0, price=10.0))
        assert agent.select_action(_make_state()) == 1

    def test_separated_admit_action_is_tuple(self):
        agent = RevenueHeuristic(threshold=0.0, mode="separated",
                                 env=_StubEnv(duration=10.0, price=5.0))
        assert agent.select_action(_make_state()) == (1, 0)

    def test_separated_reject_action_is_tuple(self):
        agent = RevenueHeuristic(threshold=1e9, mode="separated",
                                 env=_StubEnv(duration=1.0, price=1.0))
        assert agent.select_action(_make_state()) == (0, 0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            RevenueHeuristic(mode="bad")

    def test_store_learn_noop(self):
        agent = RevenueHeuristic()
        s = _make_state()
        agent.store(s, 1, 1.0, s, False)
        assert agent.learn() is None
        agent.update_target()


# ===========================================================================
# AdmissionOnlyDQN
# ===========================================================================


class TestAdmissionOnlyDQN:
    def test_instantiation_unified(self):
        agent = AdmissionOnlyDQN(STATE_DIM, _baseline_cfg(), mode="unified")
        # Q-network has 2 outputs (reject/admit)
        dummy = torch.zeros(1, STATE_DIM)
        out = agent.q(dummy)
        assert out.shape == (1, 2)

    def test_instantiation_separated(self):
        agent = AdmissionOnlyDQN(STATE_DIM, _baseline_cfg(), mode="separated")
        assert agent.mode == "separated"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            AdmissionOnlyDQN(STATE_DIM, _baseline_cfg(), mode="bad")

    def test_select_action_unified_is_int(self):
        agent = AdmissionOnlyDQN(STATE_DIM, _baseline_cfg(), mode="unified")
        a = agent.select_action(_make_state())
        assert isinstance(a, int)
        assert a in (0, 1)

    def test_select_action_separated_is_tuple(self):
        agent = AdmissionOnlyDQN(STATE_DIM, _baseline_cfg(), mode="separated")
        a = agent.select_action(_make_state())
        assert isinstance(a, tuple) and len(a) == 2
        assert a[0] in (0, 1)
        assert a[1] == 0  # path always fixed to 0

    def test_separated_path_always_zero(self):
        """Regardless of admission decision, path_k must always be 0."""
        cfg = {**_baseline_cfg(), "epsilon_start": 0.0}  # force greedy
        agent = AdmissionOnlyDQN(STATE_DIM, cfg, mode="separated")
        for _ in range(20):
            a = agent.select_action(_make_state())
            assert a[1] == 0

    def test_learn_returns_none_when_buffer_empty(self):
        agent = AdmissionOnlyDQN(STATE_DIM, _baseline_cfg())
        assert agent.learn() is None

    def test_learn_returns_float_after_filling_buffer(self):
        agent = AdmissionOnlyDQN(STATE_DIM, _baseline_cfg(), mode="unified")
        for _ in range(BATCH):
            s = _make_state()
            agent.store(s, 1, 1.0, s, False)
        loss = agent.learn()
        assert loss is not None and isinstance(loss, float) and loss >= 0.0

    def test_store_accepts_int_action(self):
        agent = AdmissionOnlyDQN(STATE_DIM, _baseline_cfg())
        s = _make_state()
        agent.store(s, 1, 2.0, s, False)  # int action
        assert len(agent.buf) == 1

    def test_store_accepts_tuple_action(self):
        agent = AdmissionOnlyDQN(STATE_DIM, _baseline_cfg(), mode="separated")
        s = _make_state()
        agent.store(s, (1, 0), 2.0, s, False)  # tuple action
        assert len(agent.buf) == 1

    def test_steps_increment(self):
        agent = AdmissionOnlyDQN(STATE_DIM, _baseline_cfg())
        for _ in range(BATCH):
            agent.store(_make_state(), 0, 0.0, _make_state(), False)
        agent.learn()
        assert agent.steps == 1

    def test_update_target_syncs_weights(self):
        agent = AdmissionOnlyDQN(STATE_DIM, _baseline_cfg())
        with torch.no_grad():
            for p in agent.q.parameters():
                p.add_(1.0)
        agent.update_target()
        for p_q, p_t in zip(agent.q.parameters(), agent.q_target.parameters()):
            assert torch.allclose(p_q, p_t)

    def test_epsilon_decays_over_calls(self):
        agent = AdmissionOnlyDQN(STATE_DIM, _baseline_cfg())
        eps_init = agent.eps
        for _ in range(50):
            agent.select_action(_make_state())
        assert agent.eps < eps_init

    def test_epsilon_floor(self):
        cfg = {**_baseline_cfg(), "epsilon_decay_steps": 1}
        agent = AdmissionOnlyDQN(STATE_DIM, cfg)
        for _ in range(5000):
            agent.select_action(_make_state())
        assert agent.eps >= agent.eps_end
