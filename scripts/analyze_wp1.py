"""WP1 analysis, exactly as pre-registered (paper/preregistration_wp1.md + Amendment 1).

Reads results/all_results.csv (batch == "wp1" only) and prints:
  * cell means / sd per substrate
  * H1  ac_dqn_widest >= joint at rho = 12.91 %, 14.42 % (per level, df=1) and pooled (df=3)
  * H2  ac_dqn_widest >= joint - 0.5 pp everywhere (descriptive check, no correction)
  * H3  ac_dqn_widest > best of {tr_widest, tr_dar} on operator and Waxman
  * Holm-Bonferroni over {H1@12.91, H1@14.42, H3 operator, H3 Waxman}
  * H4  decomposition: foresight = acwidest - greedy, flexibility = acwidest - aconly
  * replication check: original vs WP1 batch (never pooled)

Tests: two-sided paired t, alpha 0.05, pairs = training seeds.
Usage:  python3 scripts/analyze_wp1.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA = 0.05

cells = defaultdict(dict)
with open(os.path.join(ROOT, "results", "all_results.csv")) as f:
    for r in csv.DictReader(f):
        if r["valid"] == "1" and r["condition"] in ("K3", "nominal"):
            cells[(r["substrate"], r["agent"], r["batch"])][int(r["seed"])] = \
                float(r["acceptance_ratio"])

fam = json.load(open(os.path.join(ROOT, "data", "rho_family", "family.json")))
SUBS = {"operator": 11.0, "waxman": 0.0}
for f_ in fam:
    sub = "rho_r" + str(f_["ratio"]).replace(".", "p")
    if (sub, "joint_unified", "wp1") in cells:   # WP1 levels only
        SUBS[sub] = 100 * f_["rho"]
ORDER = sorted(SUBS, key=lambda s: (s.startswith("rho"), SUBS[s]))


def get(sub, agent, batch="wp1"):
    return cells.get((sub, agent, batch), {})


def diffs(sub, a, b):
    da, db = get(sub, a), get(sub, b)
    seeds = sorted(set(da) & set(db))
    return np.array([da[s] - db[s] for s in seeds])


def ttest(d):
    """two-sided paired t on differences d; returns mean, CI95, t, df, p."""
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t = m / se if se > 0 else np.inf * np.sign(m)
    df = n - 1
    p = 2 * stats.t.sf(abs(t), df)
    h = stats.t.ppf(0.975, df) * se
    return m, (m - h, m + h), t, df, p


def fmt(m, ci, t, df, p):
    return (f"{100*m:+6.2f} pp  CI95 [{100*ci[0]:+6.2f}, {100*ci[1]:+6.2f}]  "
            f"t={t:+7.2f}  df={df}  p={p:.4f}")


AGENTS = ["joint_unified", "factored", "ac_dqn_widest", "aconly", "greedy",
          "greedy_shortest", "tr_shortest", "tr_widest", "tr_dar"]

print("=== Cell means (acceptance, WP1 batch) ===")
print(f"{'substrate':10s} {'rho%':>6s} " + " ".join(f"{a[:13]:>13s}" for a in AGENTS))
for s in ORDER:
    vals = []
    for a in AGENTS:
        v = list(get(s, a).values())
        vals.append(f"{np.mean(v):.4f}±{np.std(v, ddof=1):.4f}"[:13] if v else "-")
    print(f"{s:10s} {SUBS[s]:6.2f} " + " ".join(f"{v:>13s}" for v in vals))

print("\n=== theta chosen by calibration (training seed) ===")
th = defaultdict(list)
with open(os.path.join(ROOT, "results", "all_results.csv")) as f:
    for r in csv.DictReader(f):
        if r["batch"] == "wp1" and r["agent"].startswith("tr_") and r["theta"]:
            th[(r["substrate"], r["agent"])].append(float(r["theta"]))
for s in ORDER:
    print(f"  {s:10s} " + "  ".join(f"{a}={th[(s, a)]}" for a in ("tr_shortest", "tr_widest", "tr_dar")))

# ---------------------------------------------------------------- primary family
print("\n=== H1: ac_dqn_widest >= joint at the two highest rho levels ===")
primary = {}
high = sorted((s for s in SUBS if s.startswith("rho")), key=SUBS.get)[-2:]
for s in high:
    r = ttest(diffs(s, "ac_dqn_widest", "joint_unified"))
    primary[f"H1 rho={SUBS[s]:.2f}%"] = r
    print(f"  rho={SUBS[s]:5.2f}%  acwidest-joint  {fmt(*r)}")
pooled = np.concatenate([diffs(s, "ac_dqn_widest", "joint_unified") for s in high])
rp = ttest(pooled)
print(f"  pooled (df=3)    acwidest-joint  {fmt(*rp)}")

print("\n=== H3: ac_dqn_widest > best of {tr_widest, tr_dar} ===")
for s in ("operator", "waxman"):
    best = max(("tr_widest", "tr_dar"), key=lambda a: np.mean(list(get(s, a).values())))
    r = ttest(diffs(s, "ac_dqn_widest", best))
    primary[f"H3 {s} (vs {best})"] = r
    print(f"  {s:9s} acwidest-{best:9s} {fmt(*r)}")

print("\n=== Holm-Bonferroni over the primary family ===")
items = sorted(primary.items(), key=lambda kv: kv[1][4])
k = len(items)
stop = False
for i, (name, r) in enumerate(items):
    thr = ALPHA / (k - i)
    rej = (not stop) and r[4] <= thr
    stop = stop or not rej
    print(f"  {name:28s} p={r[4]:.4f}  threshold={thr:.4f}  "
          f"{'SIGNIFICANT' if rej else 'n.s.'}  (mean {100*r[0]:+.2f} pp)")

# ---------------------------------------------------------------- secondary
print("\n=== H2: ac_dqn_widest >= joint - 0.5 pp everywhere (no correction) ===")
for s in ORDER:
    d = diffs(s, "ac_dqn_widest", "joint_unified")
    r = ttest(d)
    ok = r[0] >= -0.005
    print(f"  {s:10s} rho={SUBS[s]:5.2f}%  {fmt(*r)}  "
          f"lowerCI>{-0.5}pp: {r[1][0] >= -0.005}  mean ok: {ok}")

print("\n=== H4: decomposition (descriptive) ===")
print(f"  {'substrate':10s} {'rho%':>6s}  foresight(acw-greedy)  flexibility(acw-aconly)"
      f"  learned-routing(joint-acw)  joint-greedy")
for s in ORDER:
    m = lambda a: np.mean(list(get(s, a).values()))
    print(f"  {s:10s} {SUBS[s]:6.2f}  {100*(m('ac_dqn_widest')-m('greedy')):+8.2f} pp"
          f"            {100*(m('ac_dqn_widest')-m('aconly')):+8.2f} pp"
          f"               {100*(m('joint_unified')-m('ac_dqn_widest')):+8.2f} pp"
          f"           {100*(m('joint_unified')-m('greedy')):+8.2f} pp")

# ---------------------------------------------------------------- replication
print("\n=== Replication: original batch vs WP1 batch (not pooled) ===")
for s in ORDER:
    for a in ("joint_unified", "factored", "aconly"):
        o, w = get(s, a, "orig"), get(s, a, "wp1")
        if o and w:
            print(f"  {s:10s} {a:14s} orig {np.mean(list(o.values())):.4f} (n={len(o)})"
                  f"  wp1 {np.mean(list(w.values())):.4f} (n={len(w)})"
                  f"  diff {100*(np.mean(list(w.values()))-np.mean(list(o.values()))):+.2f} pp")
