"""Collect every experimental result into ONE tidy CSV (WP0).

Replaces hand-typed numbers. Every table and figure in the paper must be
derived from results/all_results.csv, which this script builds from the raw
artefacts:

  * results/<run_name>_s<seed>/eval_final.csv   -- learned joint/factored agents
  * baseline logs (lines "<name>/unified  acceptance_ratio=...")
  * results/baselines_<config>_s<seed>/metrics.csv (WP1 onwards)

Sources searched: results/remote/gpu14, results/remote/gpu15 (pulled from the
servers, checkpoints excluded) and results/gpu15_backup (the copy taken before
gpu15 was cleaned).

Mapping from artefact to (substrate, condition, agent) is an EXPLICIT registry
below. Anything not in the registry is reported as unmapped and ignored, so a
new run can never silently enter a table.

Usage:  python3 scripts/collect_results.py
Output: results/all_results.csv  +  a printed audit of what was found
"""
from __future__ import annotations

import csv
import glob
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SOURCES = ["remote/gpu14", "remote/gpu15", "gpu15_backup"]
OUT = os.path.join(ROOT, "all_results.csv")

# --------------------------------------------------------------------------
# Registry: learned agents (eval_final.csv), keyed by run_name.
# value = (substrate, condition, agent). Sources matter where a run_name was
# trained twice (see ddqn_unified_count).
# --------------------------------------------------------------------------
EVAL_RUNS = {
    ("gpu15_backup", "ddqn_unified_count"):      ("operator", "K3", "joint_unified"),
    ("remote/gpu14", "ddqn_unified_count"):      ("operator", "K3", "joint_unified_rerun"),
    ("gpu15_backup", "ddqn_separated_count"):    ("operator", "K3", "factored"),
    # WP1 batch (--tag wp1) of the two runs above, wherever they are pulled to
    ("wp1", "ddqn_unified_count"):               ("operator", "K3", "joint_unified"),
    ("wp1", "ddqn_separated_count"):             ("operator", "K3", "factored"),
    ("*", "ddqn_unified_count_k6"):              ("operator", "K6", "joint_unified"),
    ("*", "ddqn_separated_count_k6"):            ("operator", "K6", "factored"),
    ("*", "ddqn_unified_count_k8"):              ("operator", "K8", "joint_unified"),
    ("*", "ddqn_separated_count_k8"):            ("operator", "K8", "factored"),
    ("*", "ddqn_unified_count_load"):            ("operator", "load0.5", "joint_unified"),
    ("*", "ddqn_unified_count_mask"):            ("operator", "K3", "joint_unified_masked"),
    ("*", "ddqn_unified_count_topo2"):           ("waxman", "nominal", "joint_unified"),
    ("*", "ddqn_separated_count_topo2"):         ("waxman", "nominal", "factored"),
}
for lvl in ("r0p4", "r0p8", "r1p3", "r2p0", "r3p0"):
    EVAL_RUNS[("*", f"ddqn_unified_rho_{lvl}")] = (f"rho_{lvl}", "nominal", "joint_unified")

# run_names that exist but belong to the abandoned revenue/soft model
EVAL_EXCLUDED = {"ddqn_unified", "ddqn_separated", "dqn_unified", "dqn_separated",
                 "ddqn_unified_stress", "ddqn_unified_stress_pw25"}

# --------------------------------------------------------------------------
# Registry: baseline logs, keyed by log-name prefix (before _s<seed>.log).
# --------------------------------------------------------------------------
LOG_RUNS = {
    "sweep_count_baselines":   ("operator", "K3"),
    "aggregate":               ("operator", "K3"),
    "sweep_load_baselines":    ("operator", "load0.5"),
    "sweep_k8_baselines":      ("operator", "K8"),
    "sweep_topo2_baselines":   ("waxman", "nominal"),
    "sweep_topo2ext_baselines": ("waxman", "nominal"),
}
for lvl in ("r0p4", "r0p8", "r1p3", "r2p0", "r3p0"):
    LOG_RUNS[f"rho_{lvl}_baselines"] = (f"rho_{lvl}", "nominal")

BASELINE_NAMES = {
    "greedy_admission": "greedy",
    "revenue_heuristic": "revenue",
    "admission_only_dqn": "aconly",
    "aggregate_state_dqn": "aggregate",
    "ac_dqn_widest": "ac_dqn_widest",
    "greedy_shortest": "greedy_shortest",
    "tr_shortest": "tr_shortest",
    "tr_widest": "tr_widest",
    "tr_dar": "tr_dar",
}

# WP1+ baselines: results/baselines_<config-stem>_s<seed>__<subset>/metrics.csv.
# Keyed by config stem.  These rows carry batch="wp1" so that a WP1 re-run of
# an existing cell (e.g. aconly on the new image) is a separate row, never a
# silent replacement of the original.
CSV_RUNS = {
    "ddqn_unified_count":       ("operator", "K3"),
    "ddqn_unified_count_topo2": ("waxman", "nominal"),
    "ddqn_unified_count_k6":    ("operator", "K6"),
    "ddqn_unified_count_k8":    ("operator", "K8"),
    "ddqn_unified_count_load":  ("operator", "load0.5"),
}
for lvl in ("r0p4", "r0p8", "r1p3", "r2p0", "r3p0"):
    CSV_RUNS[f"ddqn_unified_rho_{lvl}"] = (f"rho_{lvl}", "nominal")

# The K=8 baseline run trained AC-only for only 50 episodes as a throwaway
# reference: its AC-only value is NOT comparable and is flagged invalid.
INVALID = {("sweep_k8_baselines", "aconly")}

LINE = re.compile(r"^(\w+)/unified\s+(.*)$")
KV = re.compile(r"(\w+)=([-0-9.eE]+)")


def _seed(name: str):
    m = re.search(r"_s(\d+)$", name)
    return int(m.group(1)) if m else None


def collect():
    rows, unmapped = [], Counter()

    # ---- learned agents ----------------------------------------------------
    for src in SOURCES:
        for path in glob.glob(os.path.join(ROOT, src, "*", "eval_final.csv")):
            d = os.path.basename(os.path.dirname(path))
            seed = _seed(d)
            run = re.sub(r"_s\d+$", "", d)
            batch = "orig"
            if run.endswith("_wp1"):          # run_experiment.py --tag wp1
                run, batch = run[:-len("_wp1")], "wp1"
            if run in EVAL_EXCLUDED:
                continue
            key = (EVAL_RUNS.get(("wp1", run)) if batch == "wp1" else None) \
                or EVAL_RUNS.get((src, run)) or EVAL_RUNS.get(("*", run))
            if key is None:
                unmapped[f"eval:{src}:{run}"] += 1
                continue
            metrics = {r["metric"]: float(r["value"]) for r in csv.DictReader(open(path))}
            rows.append(dict(substrate=key[0], condition=key[1], agent=key[2],
                             seed=seed, batch=batch, source=f"{src}/{d}", valid=1,
                             **metrics))

    # ---- baselines from logs ----------------------------------------------
    for src in SOURCES:
        for path in glob.glob(os.path.join(ROOT, src, "*.log")):
            base = os.path.basename(path)[:-4]
            seed = _seed(base)
            prefix = re.sub(r"_s\d+$", "", base)
            if prefix not in LOG_RUNS:
                continue
            sub, cond = LOG_RUNS[prefix]
            for line in open(path, errors="replace"):
                m = LINE.match(line.strip())
                if not m or m.group(1) not in BASELINE_NAMES:
                    continue
                agent = BASELINE_NAMES[m.group(1)]
                vals = {k: float(v) for k, v in KV.findall(m.group(2))}
                rows.append(dict(substrate=sub, condition=cond, agent=agent, seed=seed,
                                 batch="orig", source=f"{src}/{base}.log",
                                 valid=0 if (prefix, agent) in INVALID else 1, **vals))

    # ---- WP1+ baselines (metrics.csv, per-config dirs) ---------------------
    for src in SOURCES:
        for path in glob.glob(os.path.join(ROOT, src, "baselines_*_s*", "metrics.csv")):
            d = os.path.basename(os.path.dirname(path))
            m = re.match(r"^baselines_(.+)_s\d+(?:__.+)?$", d)
            stem = m.group(1) if m else d
            if stem not in CSV_RUNS:
                unmapped[f"csv:{src}:{d}"] += 1
                continue
            sub, cond = CSV_RUNS[stem]
            for r in csv.DictReader(open(path)):
                if r.get("mode", "unified") != "unified" or r["baseline"] not in BASELINE_NAMES:
                    continue
                vals = {k: float(v) for k, v in r.items()
                        if k not in ("timestamp", "baseline", "mode", "seed", "eval_seed")
                        and v not in ("", None)}
                rows.append(dict(substrate=sub, condition=cond,
                                 agent=BASELINE_NAMES[r["baseline"]], seed=int(r["seed"]),
                                 batch="wp1", source=f"{src}/{d}/metrics.csv", valid=1,
                                 **vals))

    return rows, unmapped


def dedupe(rows):
    """The same (substrate, condition, agent, seed) can appear in several
    sources (e.g. a log copied into the backup). Keep one, and report any
    that DISAGREE -- that would mean two different runs under one label."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["substrate"], r["condition"], r["agent"], r["seed"], r["batch"])].append(r)
    kept, conflicts = [], []
    for k, rs in groups.items():
        accs = {round(r.get("acceptance_ratio", float("nan")), 4) for r in rs}
        if len(accs) > 1:
            conflicts.append((k, [(r["source"], r.get("acceptance_ratio")) for r in rs]))
        kept.append(rs[0])
    return kept, conflicts


def main():
    rows, unmapped = collect()
    rows, conflicts = dedupe(rows)
    fields = ["substrate", "condition", "agent", "seed", "batch", "valid", "acceptance_ratio",
              "fulfillment_ratio", "sla_violation_rate", "avg_path_utilization",
              "mean_ep_reward", "revenue", "theta", "source"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["substrate"], r["condition"], r["agent"],
                                           r["batch"], r["seed"] or 0)):
            w.writerow(r)

    print(f"wrote {OUT}: {len(rows)} rows")
    cov = Counter((r["substrate"], r["condition"], r["agent"], r["batch"])
                  for r in rows if r["valid"])
    print("\ncoverage (valid seeds per cell):")
    for k in sorted(cov):
        print(f"  {k[0]:9s} {k[1]:8s} {k[2]:22s} {k[3]:5s} n={cov[k]}")
    if conflicts:
        print("\nCONFLICTS (same label, different values):")
        for k, srcs in conflicts:
            print(f"  {k}: {srcs}")
    if unmapped:
        print("\nunmapped (ignored):")
        for k, n in sorted(unmapped.items()):
            print(f"  {k} x{n}")


if __name__ == "__main__":
    main()
