"""Unit tests for NetworkEnv (src/env/network_env.py)."""
import numpy as np
import pytest

from src.env.network_env import NetworkEnv


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------


class TestObservationSpace:
    def test_reset_obs_shape(self, env_unified):
        obs, info = env_unified.reset()
        assert obs.shape == (env_unified.state_dim,)

    def test_reset_obs_dtype(self, env_unified):
        obs, _ = env_unified.reset()
        assert obs.dtype == np.float32

    def test_reset_obs_in_space(self, env_unified):
        obs, _ = env_unified.reset()
        # Box space with infinite bounds — just check shape / dtype
        assert env_unified.observation_space.shape == obs.shape

    def test_step_obs_shape_unified(self, env_unified):
        env_unified.reset()
        obs, *_ = env_unified.step(0)
        assert obs.shape == (env_unified.state_dim,)

    def test_step_obs_shape_separated(self, env_separated):
        env_separated.reset()
        obs, *_ = env_separated.step((0, 0))
        assert obs.shape == (env_separated.state_dim,)

    def test_state_dim_formula(self, base_cfg):
        """state_dim must equal 4 + V² + 2 + V²·K + K (K = per-path feasibility)."""
        env = NetworkEnv(base_cfg, mode="unified")
        V, K = env.V, env.K
        expected = 4 + V ** 2 + 2 + V ** 2 * K + K
        assert env.state_dim == expected


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------


class TestActionSpace:
    def test_unified_action_space_size(self, env_unified):
        assert env_unified.action_space.n == env_unified.K + 1

    def test_unified_action_dims(self, env_unified):
        assert env_unified.action_dims == env_unified.K + 1

    def test_separated_action_dims(self, env_separated):
        assert env_separated.action_dims == (2, env_separated.K)


# ---------------------------------------------------------------------------
# Reject action (action = 0)
# ---------------------------------------------------------------------------


class TestReject:
    def test_reject_reward_is_zero(self, env_unified):
        env_unified.reset()
        _, reward, *_ = env_unified.step(0)
        assert reward == 0.0

    def test_reject_not_admitted(self, env_unified):
        env_unified.reset()
        _, _, _, _, info = env_unified.step(0)
        assert not info["admitted"]

    def test_reject_separated(self, env_separated):
        env_separated.reset()
        _, reward, _, _, info = env_separated.step((0, 0))
        assert reward == 0.0
        assert not info["admitted"]


# ---------------------------------------------------------------------------
# Admit action
# ---------------------------------------------------------------------------


class TestAdmit:
    def _find_admitted(self, env, max_steps=50):
        """Run until at least one slice is admitted or we give up."""
        for _ in range(max_steps):
            env.reset()
            for path_idx in range(1, env.K + 1):
                _, reward, _, _, info = env.step(path_idx)
                if info["admitted"]:
                    return reward, info
        return None, None

    def test_admission_gives_positive_reward(self, env_unified):
        reward, info = self._find_admitted(env_unified)
        assert info is not None, "No slice was admitted in test window"
        assert reward > 0.0

    def test_admitted_flag_set(self, env_unified):
        _, info = self._find_admitted(env_unified)
        assert info is not None
        assert info["admitted"] is True

    def test_admission_reward_is_one_under_count_mode(self, env_unified):
        """Default reward_mode='count': admission reward is exactly +1,
        regardless of duration/price (paper's 'accept more slices' objective)."""
        for _ in range(50):
            env_unified.reset()
            for path_idx in range(1, env_unified.K + 1):
                _, reward, _, _, info = env_unified.step(path_idx)
                if info["admitted"]:
                    assert reward == pytest.approx(1.0)
                    return
        pytest.skip("could not trigger admission in test window")

    def test_admission_reward_matches_duration_times_price_in_revenue_mode(
        self, base_cfg
    ):
        """reward_mode='revenue' (soft-capacity ablation): reward == d * p."""
        cfg = {**base_cfg, "admission_mode": "soft", "reward_mode": "revenue"}
        env = NetworkEnv(cfg, mode="unified")
        for _ in range(50):
            env.reset()
            req = env.current_request
            expected = req["duration"] * req["price"]
            for path_idx in range(1, env.K + 1):
                _, reward, _, _, info = env.step(path_idx)
                if info["admitted"]:
                    assert reward == pytest.approx(expected)
                    return
        pytest.skip("could not trigger admission in test window")


# ---------------------------------------------------------------------------
# Capacity model
# ---------------------------------------------------------------------------


class TestCapacity:
    def test_hard_capacity_rejects_oversized_request(self, env_unified):
        """Default admission_mode='hard': a huge-bw slice cannot be admitted —
        the attempt fails outright rather than oversubscribing a link."""
        env_unified.reset()
        env_unified.current_request["bandwidth"] = 1e9
        _, reward, _, _, info = env_unified.step(1)
        assert info["admitted"] is False
        assert reward == 0.0
        assert info["new_violations"] == 0

    def test_hard_capacity_never_oversubscribed(self, env_unified):
        """Capacity constraint (§2.6): avail must never go negative on any
        link after any sequence of steps under the default hard admission."""
        env_unified.reset()
        for _ in range(200):
            action = env_unified.action_space.sample()
            env_unified.step(action)
            assert all(v >= -1e-9 for v in env_unified.topo.avail.values())

    def test_soft_capacity_over_admits_and_violates(self, base_cfg):
        """Explicit admission_mode='soft' + reward_mode='revenue': a huge-bw
        slice IS over-admitted and immediately fails its SLA."""
        cfg = {**base_cfg, "admission_mode": "soft", "reward_mode": "revenue"}
        env = NetworkEnv(cfg, mode="unified")
        env.reset()
        env.current_request["bandwidth"] = 1e9
        _, reward, _, _, info = env.step(1)
        assert info["admitted"] is True                    # over-admission allowed
        assert info["over_admitted"] is True                # chosen path was infeasible
        assert info["new_violations"] >= 1                  # SLA violated this step
        assert reward < info["admit_revenue"]               # a penalty was applied

    def test_capacity_restored_after_expiry(self, env_unified):
        """After a slice expires, capacity must be fully restored."""
        env_unified.reset()
        # Force a 1-step slice with a modest, feasible bandwidth.  The tick
        # counter uses duration_steps, so set that (not duration).
        env_unified.current_request["duration_steps"] = 1
        env_unified.current_request["bandwidth"] = 50.0
        # Record available capacity before admission
        cap_before = dict(env_unified.topo.avail)

        # Admit (soft capacity ⟹ structurally valid path always admits)
        _, _, _, _, info = env_unified.step(1)
        if not info["admitted"]:
            pytest.skip("admission failed; cannot test expiry")

        # After one more step the slice should have expired
        env_unified.step(0)
        # All capacities should be back to pre-admission values
        assert env_unified.topo.avail == cap_before


# ---------------------------------------------------------------------------
# Multi-episode reproducibility
# ---------------------------------------------------------------------------


class TestEpisodes:
    def test_multiple_resets_work(self, env_unified):
        for _ in range(5):
            obs, info = env_unified.reset()
            assert obs.shape == (env_unified.state_dim,)
            assert info == {}

    def test_episode_does_not_crash(self, env_unified):
        env_unified.reset()
        for _ in range(100):
            action = env_unified.action_space.sample()
            obs, reward, terminated, truncated, info = env_unified.step(action)
            assert obs.shape == (env_unified.state_dim,)
            assert isinstance(reward, float)
            assert not terminated
            assert not truncated

    def test_separated_episode_does_not_crash(self, env_separated):
        env_separated.reset()
        for _ in range(100):
            action = env_separated.action_space.sample()
            obs, reward, _, _, info = env_separated.step(action)
            assert obs.shape == (env_separated.state_dim,)

    def test_deterministic_with_same_seed(self, base_cfg):
        """Two envs with the same seed must produce identical trajectories."""
        env1 = NetworkEnv(base_cfg, mode="unified")
        env2 = NetworkEnv(base_cfg, mode="unified")
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        np.testing.assert_array_equal(obs1, obs2)
        for _ in range(20):
            a = 0
            o1, r1, *_ = env1.step(a)
            o2, r2, *_ = env2.step(a)
            np.testing.assert_array_equal(o1, o2)
            assert r1 == r2
