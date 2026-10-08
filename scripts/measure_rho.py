"""Measure routing headroom rho (and greedy acceptance) for every rho-family level
with ONE documented estimator, so all levels sit on the same x-axis.

The family was generated in --quick mode (rho from 20 episodes, eval seed 142).
WP2 uses this longer estimate instead: greedy policy, eval seeds 142-146
(training seeds 42-46 + 100), 100 episodes each = 250k arrivals per level.
The quick values stay in family.json as `rho`; this writes `rho_wp2`.

Usage:  python3 scripts/measure_rho.py            (parallel over levels)
Output: data/rho_family/rho_wp2.csv, and rho_wp2 / greedy_acceptance_wp2
        fields added to data/rho_family/family.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
from multiprocessing import Pool

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import generate_rho_family as g  # noqa: E402

SEEDS = (142, 143, 144, 145, 146)
EPISODES = 100
FAM = os.path.join("data", "rho_family", "family.json")


def one(level: dict) -> dict:
    base = yaml.safe_load(open("configs/base.yaml"))
    acc, rho = [], []
    for s in SEEDS:
        a, r = g.measure(level["topology"], level["capacity_scale"], base, EPISODES, seed=s)
        acc.append(a)
        rho.append(r)
        print(f"ratio={level['ratio']} seed={s} greedy={a:.4f} rho={r:.4f}", flush=True)
    return {"ratio": level["ratio"], "acc": acc, "rho": rho}


def main() -> None:
    fam = json.load(open(FAM))
    with Pool(len(fam)) as p:
        res = {r["ratio"]: r for r in p.map(one, fam)}
    with open(os.path.join("data", "rho_family", "rho_wp2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ratio", "eval_seed", "greedy_acceptance", "rho"])
        for lvl in fam:
            r = res[lvl["ratio"]]
            for s, a, x in zip(SEEDS, r["acc"], r["rho"]):
                w.writerow([lvl["ratio"], s, f"{a:.6f}", f"{x:.6f}"])
    for lvl in fam:
        r = res[lvl["ratio"]]
        lvl["rho_wp2"] = sum(r["rho"]) / len(r["rho"])
        lvl["greedy_acceptance_wp2"] = sum(r["acc"]) / len(r["acc"])
    json.dump(fam, open(FAM, "w"), indent=1)
    for lvl in fam:
        print(f"ratio={lvl['ratio']:4.1f}  rho(quick)={100*lvl['rho']:6.2f}%  "
              f"rho_wp2={100*lvl['rho_wp2']:6.2f}%  greedy_wp2={lvl['greedy_acceptance_wp2']:.4f}")


if __name__ == "__main__":
    main()
