"""Experiment entry-point.

Usage::

    python experiments/run_experiment.py --config configs/dqn_unified.yaml
    python experiments/run_experiment.py --config configs/ddqn_separated.yaml --seed 43

Config loading
--------------
Each per-experiment YAML may contain a ``_base_`` key pointing to a base config
file.  The runner loads the base first, then overlays the experiment values, so
only differing keys need to appear in the per-experiment file.

The optional ``--seed`` flag overrides ``cfg["seed"]`` and appends ``_s<seed>``
to the run name, enabling multi-seed sweeps without editing config files.

Agent dispatch
--------------
Unified agents (DQNUnified, DDQNUnified) are instantiated as:
    AgentClass(state_dim, action_dim:int, cfg)

Separated agents (DQNSeparated, DDQNSeparated) are instantiated as:
    AgentClass(state_dim, action_dims:tuple[int,int], cfg)

Both variants are handled transparently via ``env.action_dims``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

# Make sure src/ is importable when running from the repo root inside Docker.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ddqn_separated import DDQNSeparated
from src.agents.ddqn_unified import DDQNUnified
from src.agents.dqn_separated import DQNSeparated
from src.agents.dqn_unified import DQNUnified
from src.env.network_env import NetworkEnv
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import Logger
from src.utils.metrics import MetricsTracker

AGENT_MAP = {
    "dqn_unified": DQNUnified,
    "dqn_separated": DQNSeparated,
    "ddqn_unified": DDQNUnified,
    "ddqn_separated": DDQNSeparated,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    """Load a (possibly base-inheriting) YAML config and return merged dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base_path: str | None = cfg.pop("_base_", None)
    if base_path is not None:
        # base_path is relative to repo root (where the script is launched from)
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f) or {}
        base_cfg.update(cfg)   # experiment keys take priority
        cfg = base_cfg

    return cfg


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _build_agent(env: NetworkEnv, cfg: dict):
    agent_name: str = cfg["agent"]
    if agent_name not in AGENT_MAP:
        raise ValueError(
            f"Unknown agent '{agent_name}'. "
            f"Valid choices: {list(AGENT_MAP.keys())}"
        )
    AgentClass = AGENT_MAP[agent_name]
    return AgentClass(env.state_dim, env.action_dims, cfg)


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------


def evaluate(agent, cfg: dict, run_name: str) -> dict:
    """Evaluate a trained agent on fresh traffic with epsilon frozen at 0.

    Uses eval_seed = training_seed + 100 so the traffic sequence is different
    from every training seed (42-46) but still fully deterministic.
    Results are written to results/<run_name>/eval_final.csv.
    """
    eval_seed = cfg["seed"] + 100
    eval_cfg = {**cfg, "seed": eval_seed}

    mode = cfg.get("mode", "unified")
    env = NetworkEnv(eval_cfg, mode=mode)

    # Freeze exploration: eps=0.0 is below eps_end so _decay_eps() won't touch it.
    saved_eps = agent.eps
    agent.eps = 0.0

    metrics = MetricsTracker()
    n_episodes: int = cfg.get("eval_episodes", 200)
    max_steps: int = cfg.get("max_steps_per_episode", 500)

    for _ in range(n_episodes):
        state, _ = env.reset()
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            metrics.update(reward, info)
            state = next_state
            if terminated or truncated:
                break
        metrics.end_episode()

    agent.eps = saved_eps  # restore in case caller wants to resume training

    summary = metrics.summarise()

    # Persist to CSV
    results_dir = cfg.get("results_dir", "results")
    out_path = os.path.join(results_dir, run_name, "eval_final.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, v in summary.items():
            writer.writerow({"metric": k, "value": v})

    print(f"[evaluate] eval_seed={eval_seed}  episodes={n_episodes}")
    for k, v in summary.items():
        print(f"  {k:30s}: {v:.4f}")

    return summary


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _checkpoint_path(cfg: dict, episode: int) -> str:
    run_name: str = cfg.get("run_name", cfg["agent"])
    ckpt_dir: str = cfg.get("checkpoint_dir", "results/checkpoints")
    return os.path.join(ckpt_dir, run_name, f"ep{episode:05d}.pt")


def train(cfg: dict) -> None:
    _set_seeds(cfg["seed"])

    mode: str = cfg.get("mode", "unified")
    env = NetworkEnv(cfg, mode=mode)
    agent = _build_agent(env, cfg)

    metrics = MetricsTracker()
    logger = Logger(cfg)

    num_episodes: int = cfg["num_episodes"]
    max_steps: int = cfg["max_steps_per_episode"]
    target_update_freq: int = cfg["target_update_freq"]
    eval_interval: int = cfg["eval_interval"]
    log_interval: int = cfg.get("log_interval", 10)

    last_loss: float | None = None
    # Rewards (d_t × p_t) can be ~600 per step → Q-values in the hundreds of
    # thousands → MSE loss diverges.  Normalise before storing in the replay
    # buffer so Q-values stay in a tractable range.  Metrics still use the
    # original (un-scaled) rewards for interpretability.
    reward_scale: float = cfg.get("reward_scale", 1000.0)
    total_steps: int = 0  # global step counter for target-network sync

    print(
        f"[run_experiment] agent={cfg['agent']}  mode={mode}  "
        f"seed={cfg['seed']}  episodes={num_episodes}  "
        f"device={'cuda' if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        ep_reward: float = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.store(state, action, reward / reward_scale, next_state, terminated or truncated)
            loss = agent.learn()
            if loss is not None:
                last_loss = loss

            metrics.update(reward, info)   # original scale for metrics
            ep_reward += reward
            total_steps += 1
            state = next_state

            # Target-network hard copy every target_update_freq STEPS
            if total_steps % target_update_freq == 0:
                agent.update_target()

            if terminated or truncated:
                break

        metrics.end_episode()

        # Evaluation log + checkpoint
        if ep % eval_interval == 0:
            summary = metrics.summarise()
            logger.log(episode=ep, metrics=summary, loss=last_loss)
            metrics.reset()

            ckpt = _checkpoint_path(cfg, ep)
            save_checkpoint(agent, ep, ckpt)

        elif ep % log_interval == 0:
            # Lightweight stdout-only progress ping
            print(
                f"  ep={ep:5d}  reward={ep_reward:.1f}  "
                f"eps={agent.eps:.3f}",
                flush=True,
            )

    logger.finish()
    print("[run_experiment] Training complete.", flush=True)

    run_name: str = cfg.get("run_name", cfg["agent"])
    print(f"\n[run_experiment] Running held-out evaluation ...", flush=True)
    evaluate(agent, cfg, run_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a DRL agent for network slice admission control.")
    p.add_argument("--config", required=True, help="Path to per-experiment YAML config.")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg['seed'] and append _s<seed> to the run name.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

    if args.seed is not None:
        cfg["seed"] = args.seed
        # Disambiguate run name for multi-seed sweeps
        base_name: str = cfg.get("run_name", cfg.get("agent", "run"))
        # Strip any existing seed suffix before appending
        if "_s" in base_name and base_name.rsplit("_s", 1)[-1].isdigit():
            base_name = base_name.rsplit("_s", 1)[0]
        cfg["run_name"] = f"{base_name}_s{args.seed}"

    train(cfg)


if __name__ == "__main__":
    main()
