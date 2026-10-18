"""WP4.1: evaluate the exponential-pricing comparator (paper/preregistration_wp4.md).

Per substrate and routing variant (cheapest / shortest): calibrate (mu, sigma) on the
TRAINING traffic of seed 42 (40 episodes, full grid), freeze, then evaluate on the
held-out traffic of every registered seed (seed + 100, 200 episodes). Deterministic, no
training; one process per substrate.

Usage:  python3 scripts/run_exp_pricing.py [--procs 4]
Output: results/exp_pricing/<substrate>.csv  (one row per routing x seed, plus the
        calibration grid in results/exp_pricing/<substrate>_calibration.csv)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from experiments.run_experiment import _load_config  # noqa: E402
from src.baselines.exp_pricing import ROUTINGS, ExpPricing, calibrate  # noqa: E402
from src.env.network_env import NetworkEnv  # noqa: E402
from src.utils.metrics import MetricsTracker  # noqa: E402

OUT = os.path.join("results", "exp_pricing")
SUBSTRATES = {f"rho_r{l}": f"configs/ddqn_unified_rho_r{l}.yaml"
              for l in ("0p4", "0p8", "1p0", "1p3", "1p6", "2p0", "2p5", "3p0")}
SUBSTRATES["operator"] = "configs/ddqn_unified_count.yaml"
SUBSTRATES["waxman"] = "configs/ddqn_unified_count_topo2.yaml"
# E023 (A036): SNDlib + germany50 family, seeds matched to the existing ac_dqn_widest rows.
for _n in ("abilene", "atlanta", "geant", "janos_us", "newyork", "nobel_eu", "nobel_germany",
           "nobel_us", "polska"):
    SUBSTRATES[f"sndlib_{_n}"] = f"configs/wp23_{_n}.yaml"
for _l in ("0p4", "0p8", "1p0", "1p3", "1p6", "2p0", "2p5", "3p0"):
    SUBSTRATES[f"g50_r{_l}"] = f"configs/g50_r{_l}.yaml"
SEEDS = {s: (list(range(42, 52)) if s.startswith("rho_") else list(range(42, 47)))
         for s in SUBSTRATES}
CAL_SEED, CAL_EPISODES, EVAL_EPISODES = 42, 40, 200


def evaluate(agent, env, episodes):
    m = MetricsTracker()
    steps = env.cfg.get("max_steps_per_episode", 500) if hasattr(env, "cfg") else 500
    for _ in range(episodes):
        s, _ = env.reset()
        for _ in range(steps):
            s, r, te, tr, info = env.step(agent.select_action(s))
            m.update(r, info)
            if te or tr:
                break
        m.end_episode()
    return m.summarise()


def one(sub: str) -> str:
    cfg = _load_config(SUBSTRATES[sub])
    make = lambda seed: NetworkEnv({**cfg, "seed": seed}, mode="unified")
    rows, cal_rows = [], []
    for routing in ROUTINGS:
        (mu, sg), scores = calibrate(lambda: make(CAL_SEED), routing, evaluate,
                                     episodes=CAL_EPISODES)
        cal_rows += [{"routing": routing, "mu": a, "sigma": b, "acceptance_ratio": v}
                     for (a, b), v in scores.items()]
        for seed in SEEDS[sub]:
            env = make(seed + 100)
            summ = evaluate(ExpPricing(env, mu, sg, routing), env, EVAL_EPISODES)
            rows.append({"substrate": sub, "routing": routing, "seed": seed,
                         "eval_seed": seed + 100, "mu": mu, "sigma": sg, **summ})
        print(f"{sub:10s} {routing:9s} mu={mu:g} sigma={sg:g} "
              f"mean acc={sum(r['acceptance_ratio'] for r in rows if r['routing']==routing)/len(SEEDS[sub]):.4f}",
              flush=True)
    os.makedirs(OUT, exist_ok=True)
    for path, data in ((f"{sub}.csv", rows), (f"{sub}_calibration.csv", cal_rows)):
        with open(os.path.join(OUT, path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    return sub


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    subs = a.only or list(SUBSTRATES)
    with Pool(a.procs) as pool:
        for s in pool.imap_unordered(one, subs):
            print(f"done {s}", flush=True)
    print("EXP_PRICING_DONE", flush=True)


if __name__ == "__main__":
    main()
