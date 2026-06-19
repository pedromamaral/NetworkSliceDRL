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
    """Quick training loop for AdmissionOnlyDQN."""
    max_steps: int = env.cfg.get("max_steps_per_episode", 500)
    target_freq: int = env.cfg.get("target_update_freq", 100)
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.store(state, action, reward, next_state, terminated or truncated)
            agent.learn()
            state = next_state
            if terminated or truncated:
                break
        if ep % target_freq == 0:
            agent.update_target()


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
    cfg["seed"] = seed
    _set_seeds(seed)

    results_dir: str = cfg.get("results_dir", "results")
    run_tag = f"baselines_s{seed}"
    out_dir = os.path.join(results_dir, run_tag)

    rows: list[dict] = []

    def _record(name: str, mode: str, summary: dict) -> None:
        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "baseline": name,
            "mode": mode,
            "seed": seed,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in summary.items()},
        }
        rows.append(row)
        parts = [f"{name}/{mode}"]
        for k, v in summary.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        print("  ".join(parts), flush=True)

    # --- GreedyAdmission ---
    for mode in ("unified", "separated"):
        env = NetworkEnv(cfg, mode=mode)
        agent = GreedyAdmission(mode=mode)
        summary = _eval_agent(agent, env, eval_episodes)
        _record("greedy_admission", mode, summary)

    # --- RevenueHeuristic ---
    threshold: float = cfg.get("revenue_threshold", 50.0)
    for mode in ("unified", "separated"):
        env = NetworkEnv(cfg, mode=mode)
        agent = RevenueHeuristic(threshold=threshold, mode=mode)
        summary = _eval_agent(agent, env, eval_episodes)
        _record("revenue_heuristic", mode, summary)

    # --- AdmissionOnlyDQN (train first, then eval) ---
    for mode in ("unified", "separated"):
        env = NetworkEnv(cfg, mode=mode)
        agent = AdmissionOnlyDQN(env.state_dim, cfg, mode=mode)
        print(
            f"[eval_baselines] Training AdmissionOnlyDQN/{mode} "
            f"for {train_episodes} episodes …",
            flush=True,
        )
        _train_dqn_baseline(agent, env, train_episodes)
        # Freeze epsilon for eval
        saved_eps = agent.eps
        agent.eps = 0.0
        summary = _eval_agent(agent, env, eval_episodes)
        agent.eps = saved_eps
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
