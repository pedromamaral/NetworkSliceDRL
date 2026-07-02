"""Diagnostic: why is the DRL agent stuck below greedy?

Runs a trained agent (loaded from its latest checkpoint) and the greedy
baseline on the SAME held-out traffic, and breaks down every arrival into:

  admitted            – slice admitted, capacity reserved
  routing_error       – agent tried to admit but chose an infeasible path
                        while another path WAS feasible  → routing bug
  missed_admission    – agent chose reject while a feasible path existed
                        → admission conservatism
  true_reject         – agent rejected and no path was feasible anyway
                        → correct reject

If routing_error dominates, the fix is in path selection.
If missed_admission dominates, the agent is being needlessly conservative.
The greedy column is the reference (it never routing-errors, never misses).

Usage::
    python experiments/diagnose.py --config configs/dqn_unified.yaml --seed 42
    python experiments/diagnose.py --config configs/dqn_unified.yaml \
        --checkpoint results/checkpoints/dqn_unified_s42/ep02000.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ddqn_separated import DDQNSeparated
from src.agents.ddqn_unified import DDQNUnified
from src.agents.dqn_separated import DQNSeparated
from src.agents.dqn_unified import DQNUnified
from src.baselines.greedy_admission import GreedyAdmission
from src.env.network_env import NetworkEnv
from src.utils.checkpoint import load_checkpoint

AGENT_MAP = {
    "dqn_unified": DQNUnified,
    "dqn_separated": DQNSeparated,
    "ddqn_unified": DDQNUnified,
    "ddqn_separated": DDQNSeparated,
}


def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_path = cfg.pop("_base_", None)
    if base_path is not None:
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f) or {}
        base_cfg.update(cfg)
        cfg = base_cfg
    return cfg


def _latest_checkpoint(run_name: str, cfg: dict) -> str | None:
    ckpt_dir = cfg.get("checkpoint_dir", "results/checkpoints")
    pattern = os.path.join(ckpt_dir, run_name, "ep*.pt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _tally(agent, env, n_episodes: int, max_steps: int) -> dict:
    counts = {
        "arrivals": 0,
        "admitted": 0,
        "routing_error": 0,
        "missed_admission": 0,
        "true_reject": 0,
        "any_feasible": 0,
    }
    total_reward = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        for _ in range(max_steps):
            action = agent.select_action(state)
            state, reward, term, trunc, info = env.step(action)
            total_reward += reward
            counts["arrivals"] += 1
            counts["any_feasible"] += int(info["any_path_feasible"])
            if info["admitted"]:
                counts["admitted"] += 1
            elif info["routing_error"]:
                counts["routing_error"] += 1
            elif info["missed_admission"]:
                counts["missed_admission"] += 1
            else:
                counts["true_reject"] += 1
            if term or trunc:
                break
    counts["mean_ep_reward"] = total_reward / max(n_episodes, 1)
    return counts


def _report(name: str, c: dict) -> None:
    n = max(c["arrivals"], 1)
    print(f"\n=== {name} ===")
    print(f"  arrivals          : {c['arrivals']}")
    print(f"  any_path_feasible : {c['any_feasible']/n:6.2%}  (share of arrivals with a feasible path)")
    print(f"  admitted          : {c['admitted']/n:6.2%}")
    print(f"  routing_error     : {c['routing_error']/n:6.2%}  (tried admit, picked infeasible path, feasible existed)")
    print(f"  missed_admission  : {c['missed_admission']/n:6.2%}  (chose reject, feasible path existed)")
    print(f"  true_reject       : {c['true_reject']/n:6.2%}  (correct: no feasible path)")
    print(f"  mean_ep_reward    : {c.get('mean_ep_reward', 0.0):,.0f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--mask", action="store_true",
                   help="Enable feasibility action masking on the DRL agent.")
    args = p.parse_args()

    cfg = _load_config(args.config)
    cfg["seed"] = args.seed + 100  # same held-out traffic as evaluate()
    if args.mask:
        cfg["action_mask"] = True
    mode = cfg.get("mode", "unified")
    run_name = f"{cfg.get('agent')}_s{args.seed}"
    max_steps = cfg.get("max_steps_per_episode", 500)

    # --- DRL agent ---
    env = NetworkEnv(cfg, mode=mode)
    AgentClass = AGENT_MAP[cfg["agent"]]
    agent = AgentClass(env.state_dim, env.action_dims, cfg)
    ckpt = args.checkpoint or _latest_checkpoint(run_name, cfg)
    if ckpt is None:
        print(f"[diagnose] No checkpoint found for {run_name}; using untrained agent.")
    else:
        ep = load_checkpoint(agent, ckpt)
        print(f"[diagnose] Loaded {ckpt} (episode {ep})")
    agent.eps = 0.0
    drl_counts = _tally(agent, env, args.episodes, max_steps)

    # --- Greedy on identical traffic ---
    env_g = NetworkEnv(cfg, mode=mode)
    greedy = GreedyAdmission(mode=mode, V=env_g.V, K=env_g.K)
    greedy_counts = _tally(greedy, env_g, args.episodes, max_steps)

    _report(f"DRL ({cfg['agent']}_s{args.seed})", drl_counts)
    _report("Greedy", greedy_counts)

    # --- Verdict ---
    n = max(drl_counts["arrivals"], 1)
    re = drl_counts["routing_error"] / n
    ma = drl_counts["missed_admission"] / n
    print("\n=== VERDICT ===")
    if re > ma and re > 0.05:
        print(f"  ROUTING-DOMINATED: {re:.1%} routing errors vs {ma:.1%} missed admissions.")
        print("  → Agent identifies admittable slices but picks the wrong path.")
        print("  → Fix path selection (separated routing head, or better feasibility encoding).")
    elif ma > 0.05:
        print(f"  CONSERVATISM-DOMINATED: {ma:.1%} missed admissions vs {re:.1%} routing errors.")
        print("  → Agent declines slices it could profitably admit.")
        print("  → Points to reward/credit-assignment or under-exploration, not routing.")
    else:
        print(f"  Agent is near-optimal on feasibility ({re:.1%} routing, {ma:.1%} missed).")
        print("  → Remaining gap to greedy is the regime (low congestion), not a bug.")


if __name__ == "__main__":
    main()
