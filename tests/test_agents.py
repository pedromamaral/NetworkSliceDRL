"""Unit tests for src/agents/ — Phase 2.

All tests use tiny dimensions (state_dim=10, small buffer, batch=4) so they
run fast on CPU with no GPU required.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.replay_buffer import ReplayBuffer
from src.agents.dqn_unified import DQNUnified, MLP
from src.agents.dqn_separated import DQNSeparated
from src.agents.ddqn_unified import DDQNUnified, DuellingMLP
from src.agents.ddqn_separated import DDQNSeparated


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

STATE_DIM = 10
ACTION_DIM = 4        # K+1 for unified (K=3)
ACTION_DIMS = (2, 3)  # (n_admit=2, n_path=K=3) for separated
BATCH = 4


@pytest.fixture
def agent_cfg() -> dict:
    return {
        "hidden_size": 32,
        "lr": 1e-3,
        "replay_capacity": 64,
        "gamma": 0.99,
        "batch_size": BATCH,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 1000,
    }


def _rand_state() -> np.ndarray:
    return np.random.rand(STATE_DIM).astype(np.float32)


def _fill_buffer_unified(agent: DQNUnified, n: int = BATCH) -> None:
    for _ in range(n):
        s, s2 = _rand_state(), _rand_state()
        a = int(np.random.randint(ACTION_DIM))
        agent.store(s, a, float(np.random.rand()), s2, False)


def _fill_buffer_separated(agent: DQNSeparated, n: int = BATCH) -> None:
    # admit=1 so BOTH the admission buffer and the routing D_rt buffer fill
    # (routing only receives admitted transitions).
    for _ in range(n):
        s, s2 = _rand_state(), _rand_state()
        a = (1, int(np.random.randint(3)))
        agent.store(s, a, float(np.random.rand()), s2, False)


# ===========================================================================
# ReplayBuffer
# ===========================================================================


class TestReplayBuffer:
    def test_len_after_push(self):
        buf = ReplayBuffer(capacity=10)
        assert len(buf) == 0
        buf.push(_rand_state(), 0, 1.0, _rand_state(), False)
        assert len(buf) == 1

    def test_maxlen_overflow(self):
        buf = ReplayBuffer(capacity=3)
        for _ in range(5):
            buf.push(_rand_state(), 0, 0.0, _rand_state(), False)
        assert len(buf) == 3  # deque discards oldest

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=16)
        for _ in range(8):
            buf.push(_rand_state(), int(np.random.randint(4)), 0.5, _rand_state(), False)
        states, actions, rewards, next_states, dones = buf.sample(4)
        assert states.shape == (4, STATE_DIM)
        assert next_states.shape == (4, STATE_DIM)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(actions) == 4

    def test_sample_tuple_actions(self):
        buf = ReplayBuffer(capacity=16)
        for _ in range(8):
            a = (np.random.randint(2), np.random.randint(3))
            buf.push(_rand_state(), a, 0.0, _rand_state(), False)
        _, actions, _, _, _ = buf.sample(4)
        assert all(isinstance(a, tuple) and len(a) == 2 for a in actions)


# ===========================================================================
# DQNUnified
# ===========================================================================


class TestDQNUnified:
    def test_instantiation(self, agent_cfg):
        agent = DQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        assert isinstance(agent.q, MLP)
        assert isinstance(agent.q_target, MLP)

    def test_select_action_range(self, agent_cfg):
        agent = DQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        for _ in range(20):
            a = agent.select_action(_rand_state())
            assert 0 <= a < ACTION_DIM

    def test_learn_returns_none_when_buffer_empty(self, agent_cfg):
        agent = DQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        assert agent.learn() is None

    def test_learn_returns_float_when_full(self, agent_cfg):
        agent = DQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        _fill_buffer_unified(agent, n=BATCH)
        loss = agent.learn()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_steps_increments(self, agent_cfg):
        agent = DQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        _fill_buffer_unified(agent, n=BATCH)
        agent.learn()
        assert agent.steps == 1

    def test_update_target_copies_weights(self, agent_cfg):
        agent = DQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        # Modify online weights, then check target diverges, then re-sync.
        with torch.no_grad():
            for p in agent.q.parameters():
                p.add_(1.0)
        agent.update_target()
        for p_q, p_t in zip(agent.q.parameters(), agent.q_target.parameters()):
            assert torch.allclose(p_q, p_t)

    def test_epsilon_decays(self, agent_cfg):
        agent = DQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        eps_before = agent.eps
        for _ in range(10):
            agent.select_action(_rand_state())
        assert agent.eps < eps_before

    def test_epsilon_floor(self, agent_cfg):
        cfg = {**agent_cfg, "epsilon_decay_steps": 1}
        agent = DQNUnified(STATE_DIM, ACTION_DIM, cfg)
        for _ in range(5000):
            agent.select_action(_rand_state())
        assert agent.eps >= agent.eps_end


# ===========================================================================
# DQNSeparated
# ===========================================================================


class TestDQNSeparated:
    def test_instantiation(self, agent_cfg):
        agent = DQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        # Two INDEPENDENT networks: admission (2 outputs) + routing (K outputs).
        assert isinstance(agent.q, MLP)
        assert isinstance(agent.q_route, MLP)
        assert agent.n_admit == ACTION_DIMS[0]
        assert agent.n_path == ACTION_DIMS[1]
        # Separate D_rt buffer for routing (spec §4.2).
        assert agent.buf is not agent.buf_rt

    def test_select_action_is_valid_tuple(self, agent_cfg):
        agent = DQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        for _ in range(20):
            a = agent.select_action(_rand_state())
            assert isinstance(a, tuple) and len(a) == 2
            assert 0 <= a[0] < ACTION_DIMS[0]
            assert 0 <= a[1] < ACTION_DIMS[1]

    def test_reject_forces_path_zero(self, agent_cfg):
        """Routing net is only queried on accept; reject → path index 0."""
        agent = DQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        agent.eps = 0.0  # greedy; force deterministic admit head
        # Drive the admission head to always reject by zeroing accept logits is
        # hard; instead just assert the contract: whenever admit==0, path==0.
        for _ in range(50):
            a = agent.select_action(_rand_state())
            if a[0] == 0:
                assert a[1] == 0

    def test_routing_buffer_only_gets_admitted(self, agent_cfg):
        """D_rt must receive ONLY admitted transitions (spec §4.2, §11.7)."""
        agent = DQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        s = _rand_state()
        agent.store(s, (0, 2), 1.0, s, False)  # reject → routing NOT stored
        agent.store(s, (1, 1), 1.0, s, False)  # admit  → routing stored
        agent.store(s, (0, 0), 0.0, s, False)  # reject → routing NOT stored
        assert len(agent.buf) == 3       # admission sees every transition
        assert len(agent.buf_rt) == 1    # routing sees only the admitted one

    def test_both_heads_learn(self, agent_cfg):
        """learn() updates admission and routing nets independently."""
        agent = DQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        _fill_buffer_separated(agent, n=BATCH)  # admit=1 → both buffers full
        # Snapshot routing weights; they must change after learn().
        before = [p.detach().clone() for p in agent.q_route.parameters()]
        agent.learn()
        after = list(agent.q_route.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    def test_learn_returns_none_when_buffer_empty(self, agent_cfg):
        agent = DQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        assert agent.learn() is None

    def test_learn_returns_float_when_full(self, agent_cfg):
        agent = DQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        _fill_buffer_separated(agent, n=BATCH)
        loss = agent.learn()
        assert loss is not None and isinstance(loss, float) and loss >= 0.0

    def test_steps_increments(self, agent_cfg):
        agent = DQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        _fill_buffer_separated(agent, n=BATCH)
        agent.learn()
        assert agent.steps == 1

    def test_update_target_copies_weights(self, agent_cfg):
        agent = DQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        with torch.no_grad():
            for p in agent.q.parameters():
                p.add_(1.0)
            for p in agent.q_route.parameters():
                p.add_(1.0)
        agent.update_target()
        for p_q, p_t in zip(agent.q.parameters(), agent.q_target.parameters()):
            assert torch.allclose(p_q, p_t)
        for p_q, p_t in zip(agent.q_route.parameters(), agent.q_route_target.parameters()):
            assert torch.allclose(p_q, p_t)


# ===========================================================================
# DDQNUnified
# ===========================================================================


class TestDDQNUnified:
    def test_uses_duelling_mlp(self, agent_cfg):
        agent = DDQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        assert isinstance(agent.q, DuellingMLP)
        assert isinstance(agent.q_target, DuellingMLP)

    def test_duelling_mlp_has_v_and_a_heads(self, agent_cfg):
        agent = DDQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        assert hasattr(agent.q, "v_head")
        assert hasattr(agent.q, "a_head")

    def test_select_action_range(self, agent_cfg):
        agent = DDQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        for _ in range(20):
            a = agent.select_action(_rand_state())
            assert 0 <= a < ACTION_DIM

    def test_learn_returns_none_when_buffer_empty(self, agent_cfg):
        assert DDQNUnified(STATE_DIM, ACTION_DIM, agent_cfg).learn() is None

    def test_learn_returns_float_when_full(self, agent_cfg):
        agent = DDQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        _fill_buffer_unified(agent, n=BATCH)
        loss = agent.learn()
        assert loss is not None and isinstance(loss, float) and loss >= 0.0

    def test_update_target(self, agent_cfg):
        agent = DDQNUnified(STATE_DIM, ACTION_DIM, agent_cfg)
        with torch.no_grad():
            for p in agent.q.parameters():
                p.add_(1.0)
        agent.update_target()
        for p_q, p_t in zip(agent.q.parameters(), agent.q_target.parameters()):
            assert torch.allclose(p_q, p_t)


# ===========================================================================
# DDQNSeparated
# ===========================================================================


class TestDDQNSeparated:
    def test_uses_duelling_mlp_both_nets(self, agent_cfg):
        agent = DDQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        assert isinstance(agent.q, DuellingMLP)          # admission net
        assert isinstance(agent.q_route, DuellingMLP)    # routing net

    def test_duelling_nets_have_v_and_a_heads(self, agent_cfg):
        agent = DDQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        for net in (agent.q, agent.q_route):
            assert hasattr(net, "v_head")
            assert hasattr(net, "a_head")

    def test_select_action_is_valid_tuple(self, agent_cfg):
        agent = DDQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        for _ in range(20):
            a = agent.select_action(_rand_state())
            assert isinstance(a, tuple) and len(a) == 2
            assert 0 <= a[0] < ACTION_DIMS[0]
            assert 0 <= a[1] < ACTION_DIMS[1]

    def test_learn_returns_none_when_buffer_empty(self, agent_cfg):
        assert DDQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg).learn() is None

    def test_learn_returns_float_when_full(self, agent_cfg):
        agent = DDQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        _fill_buffer_separated(agent, n=BATCH)
        loss = agent.learn()
        assert loss is not None and isinstance(loss, float) and loss >= 0.0

    def test_steps_increments(self, agent_cfg):
        agent = DDQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        _fill_buffer_separated(agent, n=BATCH)
        agent.learn()
        assert agent.steps == 1

    def test_update_target(self, agent_cfg):
        agent = DDQNSeparated(STATE_DIM, ACTION_DIMS, agent_cfg)
        with torch.no_grad():
            for p in agent.q.parameters():
                p.add_(1.0)
        agent.update_target()
        for p_q, p_t in zip(agent.q.parameters(), agent.q_target.parameters()):
            assert torch.allclose(p_q, p_t)
