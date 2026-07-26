"""Aggregate the multi-seed count/hard sweep into a paper-grade table.

Reads, for each seed:
  results/<joint_run>_s<seed>/eval_final.csv     – joint DDQN held-out eval
  results/baselines_s<seed>/metrics.csv          – greedy / revenue / AC-only

Reports mean +/- std across seeds, plus PAIRED per-seed deltas (joint - baseline).
Pairing matters: for a given seed every agent is evaluated on the *same* held-out
traffic (seed+100), so the per-seed difference removes traffic variance and is a
far more powerful test than comparing two independent means at n=5.

Usage::
    python experiments/aggregate_sweep.py
    python experiments/aggregate_sweep.py --metric acceptance_ratio --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Two-tailed t critical values at alpha=0.05, indexed by degrees of freedom.
_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

BASELINE_KEYS = ["greedy_admission", "admission_only_dqn", "revenue_heuristic"]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    """Sample standard deviation (ddof=1)."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _read_joint(results_dir: str, run_base: str, seed: int, metric: str) -> float | None:
    path = os.path.join(results_dir, f"{run_base}_s{seed}", "eval_final.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["metric"] == metric:
                return float(row["value"])
    return None


def _read_baselines(results_dir: str, seed: int, metric: str) -> dict[str, float]:
    path = os.path.join(results_dir, f"baselines_s{seed}", "metrics.csv")
    out: dict[str, float] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("mode") != "unified":
                continue
            if metric in row and row[metric] not in ("", None):
                out[row["baseline"]] = float(row[metric])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--run_base", default="ddqn_unified_count")
    p.add_argument("--metric", default="acceptance_ratio")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = p.parse_args()

    metric = args.metric
    per_agent: dict[str, dict[int, float]] = {"joint_ddqn": {}}
    for k in BASELINE_KEYS:
        per_agent[k] = {}

    for s in args.seeds:
        j = _read_joint(args.results_dir, args.run_base, s, metric)
        if j is not None:
            per_agent["joint_ddqn"][s] = j
        for k, v in _read_baselines(args.results_dir, s, metric).items():
            if k in per_agent:
                per_agent[k][s] = v

    print(f"\n=== metric: {metric} ===")
    print(f"{'agent':22s} {'n':>2s}  {'mean':>9s} {'std':>9s}   per-seed")
    for name in ["joint_ddqn"] + BASELINE_KEYS:
        vals_by_seed = per_agent[name]
        if not vals_by_seed:
            print(f"{name:22s}  0        --        --   (no data)")
            continue
        vals = [vals_by_seed[s] for s in sorted(vals_by_seed)]
        detail = " ".join(f"{s}:{vals_by_seed[s]:.4f}" for s in sorted(vals_by_seed))
        print(f"{name:22s} {len(vals):2d}  {_mean(vals):9.4f} {_std(vals):9.4f}   {detail}")

    # --- Paired comparisons (joint - baseline on identical traffic) ---
    print(f"\n=== PAIRED deltas: joint_ddqn - baseline (same held-out traffic per seed) ===")
    joint = per_agent["joint_ddqn"]
    for name in BASELINE_KEYS:
        other = per_agent[name]
        common = sorted(set(joint) & set(other))
        if len(common) < 2:
            print(f"  vs {name:22s} : insufficient paired seeds ({len(common)})")
            continue
        d = [joint[s] - other[s] for s in common]
        m, sd, n = _mean(d), _std(d), len(d)
        se = sd / (n ** 0.5) if sd > 0 else 0.0
        rel = 100.0 * m / _mean([other[s] for s in common])
        line = (f"  vs {name:22s} : mean Δ={m:+.4f} ({rel:+.1f}%)  std={sd:.4f}  n={n}")
        if se > 0:
            t = m / se
            tc = _T_CRIT.get(n - 1, 2.0)
            verdict = "SIGNIFICANT" if abs(t) > tc else "not significant"
            line += f"  t={t:+.2f} (crit±{tc:.2f}, df={n-1}) → {verdict}"
        else:
            line += "  (zero variance)"
        print(line)
        print(f"       per-seed Δ: " + " ".join(f"{s}:{joint[s]-other[s]:+.4f}" for s in common))
        wins = sum(1 for x in d if x > 0)
        print(f"       joint wins {wins}/{n} seeds")
    print()


if __name__ == "__main__":
    main()
