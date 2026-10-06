"""Greedy acceptance as a function of capacity scale (the load-sensitivity
envelope in Fig. "load").

Greedy needs no training, so this runs on a laptop in a few minutes. Same
held-out traffic as every agent: eval seeds 142-146 (training seeds 42-46
+100), 200 episodes each.

Usage:  python3 scripts/greedy_load_envelope.py
Output: results/greedy_load_envelope.csv  (one row per scale x seed)
"""
from __future__ import annotations

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from experiments.run_experiment import _load_config  # noqa: E402
from src.baselines.greedy_admission import GreedyAdmission  # noqa: E402
from src.env.network_env import NetworkEnv  # noqa: E402
from src.utils.metrics import MetricsTracker  # noqa: E402

SCALES = (0.35, 0.5, 0.7, 1.0, 1.5)
SEEDS = (42, 43, 44, 45, 46)
EPISODES = 200
OUT = os.path.join(ROOT, "results", "greedy_load_envelope.csv")


def run(cfg: dict, scale: float, seed: int) -> float:
    env = NetworkEnv({**cfg, "capacity_scale": scale, "seed": seed + 100}, mode="unified")
    agent = GreedyAdmission(mode="unified", V=env.V, K=env.K)
    m = MetricsTracker()
    for _ in range(EPISODES):
        s, _ = env.reset()
        for _ in range(cfg["max_steps_per_episode"]):
            s, r, term, trunc, info = env.step(agent.select_action(s))
            m.update(r, info)
            if term or trunc:
                break
        m.end_episode()
    return m.summarise()["acceptance_ratio"]


def main() -> None:
    cfg = _load_config("configs/ddqn_unified_count.yaml")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capacity_scale", "seed", "eval_seed", "acceptance_ratio"])
        for scale in SCALES:
            for seed in SEEDS:
                acc = run(cfg, scale, seed)
                w.writerow([scale, seed, seed + 100, f"{acc:.6f}"])
                f.flush()
                print(f"scale={scale:.2f} seed={seed} acc={acc:.4f}", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
