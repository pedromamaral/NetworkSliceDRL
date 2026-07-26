"""Evaluate all rule-based and lightweight-DRL baselines.

Baselines covered:
  1. GreedyAdmission   – always admit via path-0; no training
  2. RevenueHeuristic  – threshold-based; no training
  3. AdmissionOnlyDQN  – admit/reject Q-network, path always 0;
                         trained here for ``train_episodes`` steps then evaluated

Usage::

    # evaluate on default base config, seed 42
    python experiments/eval_baselines.py

    # override seed (for multi-seed sweep)
    python experiments/eval_baselines.py --seed 43

    # custom config / episode counts
    python experiments/eval_baselines.py \\
        --config configs/base.yaml \\
        --seed 44 \\
        --train_episodes 500 \\
        --eval_episodes 200
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.admission_only_dqn import AdmissionOnlyDQN
from src.baselines.greedy_admission import GreedyAdmission
from src.baselines.revenue_heuristic import RevenueHeuristic
from src.env.network_env import NetworkEnv
from src.utils.metrics import MetricsTracker


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_path: str | None = cfg.pop("_base_", None)
    if base_path is not None:
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f) or {}
        base_cfg.update(cfg)
        cfg = base_cfg
    return cfg


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _eval_agent(agent, env: NetworkEnv, n_episodes: int) -> dict:
    """Run *n_episodes* evaluation episodes and return summarised metrics."""
    tracker = MetricsTracker()
    for _ in range(n_episodes):
        state, _ = env.reset()
        for _ in range(env.cfg.get("max_steps_per_episode", 500)):
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            tracker.update(reward, info)
            if terminated or truncated:
                break
        tracker.end_episode()
    return tracker.summarise()


def _train_dqn_baseline(agent: AdmissionOnlyDQN, env: NetworkEnv, n_episodes: int) -> None:
    """Training loop for AdmissionOnlyDQN, matching run_experiment machinery.

    Rewards are divided by ``reward_scale`` before entering the replay buffer
    (same as the main agents) and the target net is hard-copied every
    ``target_update_freq`` STEPS (not episodes)."""
    max_steps: int = env.cfg.get("max_steps_per_episode", 500)
    target_freq: int = env.cfg.get("target_update_freq", 100)
    reward_scale: float = float(env.cfg.get("reward_scale", 1.0))
    total_steps = 0
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.store(state, action, reward / reward_scale, next_state,
                        terminated or truncated)
            agent.learn()
            state = next_state
            total_steps += 1
            if total_steps % target_freq == 0:
                agent.update_target()
            if terminated or truncated:
                break


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _save_results(rows: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.csv")
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval_baselines] Results saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg_path: str, seed: int, train_episodes: int, eval_episodes: int) -> None:
    cfg = _load_config(cfg_path)
    _set_seeds(seed)
    # Mirror run_experiment.evaluate(): train on `seed`, evaluate every baseline
    # on the held-out `seed + 100` traffic so results are directly comparable to
    # the DRL agents' held-out eval.
    eval_seed = seed + 100
    modes = ("unified",)  # unified is the comparison against the joint DRL agent

    results_dir: str = cfg.get("results_dir", "results")
    run_tag = f"baselines_s{seed}"
    out_dir = os.path.join(results_dir, run_tag)

    rows: list[dict] = []

    def _make_env(s: int, mode: str) -> NetworkEnv:
        return NetworkEnv({**cfg, "seed": s}, mode=mode)

    def _record(name: str, mode: str, summary: dict) -> None:
        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "baseline": name,
            "mode": mode,
            "seed": seed,
            "eval_seed": eval_seed,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in summary.items()},
        }
        rows.append(row)
        parts = [f"{name}/{mode}"]
        for k, v in summary.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        print("  ".join(parts), flush=True)

    # --- GreedyAdmission (no training; eval on held-out) ---
    for mode in modes:
        env = _make_env(eval_seed, mode)
        agent = GreedyAdmission(mode=mode, V=env.V, K=env.K)
        summary = _eval_agent(agent, env, eval_episodes)
        _record("greedy_admission", mode, summary)

    # --- RevenueHeuristic (no training; eval on held-out) ---
    threshold: float = cfg.get("revenue_threshold", 500.0)
    for mode in modes:
        env = _make_env(eval_seed, mode)
        agent = RevenueHeuristic(threshold=threshold, mode=mode, env=env)
        summary = _eval_agent(agent, env, eval_episodes)
        _record("revenue_heuristic", mode, summary)

    # --- AdmissionOnlyDQN (train on `seed`, eval on held-out `seed+100`) ---
    for mode in modes:
        train_env = _make_env(seed, mode)
        agent = AdmissionOnlyDQN(train_env.state_dim, cfg, mode=mode)
        print(
            f"[eval_baselines] Training AdmissionOnlyDQN/{mode} "
            f"for {train_episodes} episodes …",
            flush=True,
        )
        _train_dqn_baseline(agent, train_env, train_episodes)
        # Freeze epsilon and evaluate on held-out traffic.
        agent.eps = 0.0
        eval_env = _make_env(eval_seed, mode)
        summary = _eval_agent(agent, eval_env, eval_episodes)
        _record("admission_only_dqn", mode, summary)

    _save_results(rows, out_dir)
    print("[eval_baselines] Done.", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate heuristic and lightweight-DRL baselines.")
    p.add_argument("--config", default="configs/base.yaml", help="Path to config YAML.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--train_episodes", type=int, default=500,
                   help="Training episodes for AdmissionOnlyDQN.")
    p.add_argument("--eval_episodes", type=int, default=200,
                   help="Evaluation episodes for all baselines.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.config, args.seed, args.train_episodes, args.eval_episodes)
