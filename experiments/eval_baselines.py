"""Evaluate all rule-based and lightweight-DRL baselines.

Baselines covered:
  1. GreedyAdmission   – always admit via path-0; no training
  2. RevenueHeuristic  – threshold-based; no training
  3. AdmissionOnlyDQN  – admit/reject Q-network, path always 0;
  4. AggregateStateDQN – admit/reject over an AGGREGATE resource state, no
                         path-level routing; stand-in for the 5G-core
                         literature (SARA/DSARA, FSAC, online AC+RA);
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
import random
import sys
from datetime import datetime

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.admission_only_dqn import AdmissionOnlyDQN
from src.baselines.aggregate_state_dqn import AggregateStateDQN
from src.baselines.ac_dqn_widest import ACDQNWidest
from src.baselines.trunk_reservation import TrunkReservation, calibrate_theta
from src.baselines.greedy_admission import GreedyAdmission
from src.baselines.revenue_heuristic import RevenueHeuristic
from src.env.network_env import NetworkEnv
from src.utils.metrics import MetricsTracker


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(path: str, _seen: tuple = ()) -> dict:
    """Load a YAML config, resolving `_base_` inheritance RECURSIVELY.

    The previous loader resolved only one level, so a config inheriting from a
    config that itself inherits base.yaml silently lost every base key (e.g.
    topology_file). Child keys take priority over their base at every level.
    """
    if path in _seen:
        raise ValueError(f"cyclic _base_ chain: {' -> '.join(_seen + (path,))}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_path = cfg.pop("_base_", None)
    if base_path is None:
        return cfg
    merged = _load_config(base_path, _seen + (path,))
    merged.update(cfg)
    return merged


def _set_seeds(seed: int) -> None:
    # stdlib random drives replay-buffer minibatch sampling (unseeded before
    # 2026-10-06). Called again before every trained baseline so its result
    # does not depend on which other baselines ran earlier in the process.
    random.seed(seed)
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
        # union of keys, first-seen order: some baselines add columns (theta)
        fields: list[str] = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval_baselines] Results saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg_path: str, seed: int, train_episodes: int, eval_episodes: int,
         baselines: str = "greedy,revenue,aconly,aggregate") -> None:
    cfg = _load_config(cfg_path)
    _set_seeds(seed)
    # Mirror run_experiment.evaluate(): train on `seed`, evaluate every baseline
    # on the held-out `seed + 100` traffic so results are directly comparable to
    # the DRL agents' held-out eval.
    eval_seed = seed + 100
    modes = ("unified",)  # unified is the comparison against the joint DRL agent

    results_dir: str = cfg.get("results_dir", "results")
    # Per-config output dir: the old regime-agnostic "baselines_s<seed>" was
    # silently overwritten whenever two sweeps reused a seed.
    cfg_stem = os.path.splitext(os.path.basename(cfg_path))[0]
    want = {b.strip() for b in baselines.split(',') if b.strip()}
    # Baselines that are run in separate processes for the same config/seed
    # (WP1 runs one trained baseline per container) must not overwrite each
    # other's metrics.csv: the subset becomes part of the directory name.
    run_tag = f"baselines_{cfg_stem}_s{seed}__{'-'.join(sorted(want))}"
    out_dir = os.path.join(results_dir, run_tag)

    print(f"[eval_baselines] running: {sorted(want)}", flush=True)

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
    for mode in (modes if 'greedy' in want else ()):
        env = _make_env(eval_seed, mode)
        agent = GreedyAdmission(mode=mode, V=env.V, K=env.K)
        summary = _eval_agent(agent, env, eval_episodes)
        _record("greedy_admission", mode, summary)

    # --- RevenueHeuristic (no training; eval on held-out) ---
    threshold: float = cfg.get("revenue_threshold", 500.0)
    for mode in (modes if 'revenue' in want else ()):
        env = _make_env(eval_seed, mode)
        agent = RevenueHeuristic(threshold=threshold, mode=mode, env=env)
        summary = _eval_agent(agent, env, eval_episodes)
        _record("revenue_heuristic", mode, summary)

    # --- AdmissionOnlyDQN (train on `seed`, eval on held-out `seed+100`) ---
    for mode in (modes if 'aconly' in want else ()):
        train_env = _make_env(seed, mode)
        _set_seeds(seed)
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

    # --- AggregateStateDQN: stand-in for the 5G-core literature ------------
    # Aggregate resource state, admission-only action space, but the SAME
    # count objective and training machinery as our agents, so the comparison
    # isolates the representation rather than the objective.
    for mode in (modes if 'aggregate' in want else ()):
        train_env = _make_env(seed, mode)
        agg_cfg = {**cfg, "num_nodes_eff": train_env.V,
                   "k_shortest_paths": train_env.K}
        _set_seeds(seed)
        agent = AggregateStateDQN(train_env.state_dim, agg_cfg, mode=mode)
        print(
            f"[eval_baselines] Training AggregateStateDQN/{mode} "
            f"for {train_episodes} episodes …",
            flush=True,
        )
        _train_dqn_baseline(agent, train_env, train_episodes)
        agent.eps = 0.0
        eval_env = _make_env(eval_seed, mode)
        summary = _eval_agent(agent, eval_env, eval_episodes)
        _record("aggregate_state_dqn", mode, summary)

    # --- WP1 factorial cells -------------------------------------------
    # Learned admission + widest-path routing.
    for mode in (modes if 'acwidest' in want else ()):
        train_env = _make_env(seed, mode)
        _set_seeds(seed)
        agent = ACDQNWidest(train_env.state_dim, cfg, mode=mode, n_paths=train_env.K)
        print(f"[eval_baselines] Training ACDQNWidest/{mode} "
              f"for {train_episodes} episodes …", flush=True)
        _train_dqn_baseline(agent, train_env, train_episodes)
        agent.eps = 0.0
        summary = _eval_agent(agent, _make_env(eval_seed, mode), eval_episodes)
        _record("ac_dqn_widest", mode, summary)

    # Greedy admission + shortest path (= trunk reservation, shortest, theta=0).
    for mode in (modes if 'greedyshortest' in want else ()):
        env = _make_env(eval_seed, mode)
        agent = TrunkReservation(env, theta=0.0, routing="shortest", mode=mode)
        _record("greedy_shortest", mode, _eval_agent(agent, env, eval_episodes))

    # Trunk reservation: theta calibrated on the TRAINING seed, then frozen.
    for routing in ("shortest", "widest", "dar"):
        if f"tr_{routing}" not in want:
            continue
        for mode in modes:
            theta, scores = calibrate_theta(
                lambda: _make_env(seed, mode), routing,
                lambda a, e, n: _eval_agent(a, e, n),
                episodes=cfg.get("tr_calibration_episodes", 40))
            print(f"[eval_baselines] TR/{routing} calibration "
                  + " ".join(f"{t:.2f}:{v:.4f}" for t, v in scores.items())
                  + f" -> theta={theta:.2f}", flush=True)
            env = _make_env(eval_seed, mode)
            agent = TrunkReservation(env, theta=theta, routing=routing, mode=mode)
            summary = _eval_agent(agent, env, eval_episodes)
            summary["theta"] = theta
            _record(f"tr_{routing}", mode, summary)

    _save_results(rows, out_dir)
    print("[eval_baselines] Done.", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate heuristic and lightweight-DRL baselines.")
    p.add_argument("--config", default="configs/base.yaml", help="Path to config YAML.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--baselines", default="greedy,revenue,aconly,aggregate",
                   help="comma-separated subset to run: greedy, revenue, "
                        "aconly, aggregate, acwidest, greedyshortest, "
                        "tr_shortest, tr_widest, tr_dar. Trained baselines (aconly, "
                        "aggregate) dominate runtime, so sweeps that only need "
                        "a subset should say so.")
    p.add_argument("--train_episodes", type=int, default=500,
                   help="Training episodes for AdmissionOnlyDQN.")
    p.add_argument("--eval_episodes", type=int, default=200,
                   help="Evaluation episodes for all baselines.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.config, args.seed, args.train_episodes, args.eval_episodes,
         args.baselines)
