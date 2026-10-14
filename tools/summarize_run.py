#!/usr/bin/env python3
"""Raw run outputs -> .mem/results/E###.json + a <=15-line table.

Handles the three result formats of this repo:
  * learned runs   results/.../<run_name>_s<seed>/metrics.csv (per eval_interval) [+ eval_final.csv]
  * baselines      results/.../baselines_<cfg>_s<seed>__<subset>/metrics.csv (one row per baseline)
  * exp-pricing    results/.../exp_pricing/<substrate>.csv (rows with an arm column)
Each directory/CSV matched by the globs is one (arm, seed) sample; arms are aggregated as
mean/sd/n over seeds. Metric names are "<group>/<arm>/<metric>", group = config/substrate.

Usage:
  python3 tools/summarize_run.py E014 'results/remote/gpu15/baselines_wp23_*' [more globs] \
      [--metrics acceptance_ratio,mean_ep_reward] [--host gpu15] [--status finished]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM_COLS = ("baseline", "arm", "agent", "policy")


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def samples(path, metrics):
    """Yield (group, arm, seed, {metric: (final, best, best_step)}) for one dir or CSV."""
    name = os.path.basename(path.rstrip("/"))
    m = re.match(r"(?:baselines_)?(.+?)_s(\d+)(?:__(.+))?$", name)
    group, seed = (m.group(1), int(m.group(2))) if m else (name, None)
    if os.path.isdir(path):
        csv_path = os.path.join(path, "metrics.csv")
        if not os.path.exists(csv_path):
            return
        rows = _read(csv_path)
        if rows and "episode" in rows[0]:            # learned run: curve over evals
            final_path = os.path.join(path, "eval_final.csv")
            final = {r["metric"]: _float(r["value"]) for r in _read(final_path)} if os.path.exists(final_path) else {}
            out = {}
            for k in metrics:
                curve = [(int(r["episode"]), _float(r.get(k))) for r in rows if _float(r.get(k)) is not None]
                if not curve:
                    continue
                bstep, best = max(curve, key=lambda t: t[1])
                out[k] = (final.get(k, curve[-1][1]), best, bstep)
            yield group, "learned", seed, out
            return
        for r in rows:                                # baselines: one row per arm
            arm = next((r[c] for c in ARM_COLS if c in r), "?")
            s = int(r["seed"]) if r.get("seed") else seed
            yield group, arm, s, {k: (_float(r.get(k)), None, None) for k in metrics if _float(r.get(k)) is not None}
    elif path.endswith(".csv"):                       # flat CSV (exp-pricing)
        for r in _read(path):
            arm = next((r[c] for c in ARM_COLS if c in r), "?")
            yield name[:-4], arm, _float(r.get("seed")), {
                k: (_float(r.get(k)), None, None) for k in metrics if _float(r.get(k)) is not None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eid")
    p.add_argument("globs", nargs="+")
    p.add_argument("--metrics", default="acceptance_ratio")
    p.add_argument("--host", default="")
    p.add_argument("--config", default="")
    p.add_argument("--status", default="finished")
    a = p.parse_args()
    metrics = a.metrics.split(",")

    paths = sorted({q for g in a.globs for q in glob.glob(os.path.join(ROOT, g) if not os.path.isabs(g) else g)})
    agg = defaultdict(lambda: {"final": [], "best": [], "best_step": [], "seeds": []})
    for path in paths:
        for group, arm, seed, vals in samples(path, metrics):
            for k, (fin, best, bstep) in vals.items():
                d = agg[f"{group}/{arm}/{k}"]
                d["final"].append(fin)
                d["seeds"].append(seed)
                if best is not None:
                    d["best"].append(best)
                    d["best_step"].append(bstep)

    out_metrics = {}
    for name, d in sorted(agg.items()):
        f = d["final"]
        out_metrics[name] = {
            "final": round(statistics.mean(f), 5),
            "sd": round(statistics.stdev(f), 5) if len(f) > 1 else 0.0,
            "n": len(f),
            "best": round(statistics.mean(d["best"]), 5) if d["best"] else None,
            "best_step": int(statistics.median(d["best_step"])) if d["best_step"] else None,
        }
    try:
        sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "?"
    rec = {"id": a.eid, "sha": sha, "config": a.config, "host": a.host, "started": None,
           "status": a.status, "metrics": out_metrics,
           "artifacts": [os.path.relpath(q, ROOT) for q in paths][:50] + (["..."] if len(paths) > 50 else [])}
    os.makedirs(os.path.join(ROOT, ".mem", "results"), exist_ok=True)
    dest = os.path.join(ROOT, ".mem", "results", f"{a.eid}.json")
    with open(dest, "w") as f:
        json.dump(rec, f, indent=1)

    lines = [f"{a.eid}: {len(paths)} inputs -> {os.path.relpath(dest, ROOT)}"]
    for name, v in out_metrics.items():
        lines.append(f"  {name:<52} {v['final']:.4f} ± {v['sd']:.4f}  n={v['n']}")
    if len(lines) > 15:
        lines = lines[:14] + [f"  ... {len(lines) - 14} more rows in json"]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
