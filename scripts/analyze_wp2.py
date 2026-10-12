"""WP2.1 analysis, exactly as pre-registered (paper/preregistration_wp2.md, sections 1-4).

F(rho) = acc(ac_dqn_widest) - acc(aconly)   [pp]   value of path flexibility
G(rho) = acc(ac_dqn_widest) - acc(greedy)   [pp]   value of admission foresight

W1  slope of F on rho_wp2 > 0; 95 % cluster-bootstrap CI (seeds resampled within level,
    10 000 replicates) excludes 0
W2  F at ratio 0.4 equivalent to 0 within +-0.5 pp (TOST, paired, n = 10, alpha 0.05)
W3  new levels 1.0 / 1.6 / 2.5: MAE of level means vs registered predictions <= 1.0 pp
    AND every observed mean inside its registered 95 % prediction interval
W4  fresh data only (seeds 44-51 at WP1 levels, all seeds at new levels):
    slope of G on rho_wp2 < 0, one-sided cluster bootstrap, alpha 0.05
W5  rho* = -intercept/slope of the W1 fit, with bootstrap CI (descriptive)
Holm-Bonferroni over {W1, W2, W3}; W3 is a pass/fail criterion, entered as p = 0 (pass)
or p = 1 (fail).

Usage:  python3 scripts/analyze_wp2.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA, B, EQ = 0.05, 10_000, 0.5
SEEDS = list(range(42, 52))
WP1_LEVELS = {0.4, 0.8, 1.3, 2.0, 3.0}
# registered in preregistration_wp2.md (W3 table), copied verbatim
PRED = {1.0: (1.06, 0.34, 1.77), 1.6: (3.70, 3.04, 4.35), 2.5: (5.24, 4.42, 6.06)}

fam = json.load(open(os.path.join(ROOT, "data", "rho_family", "family.json")))
RHO = {f["ratio"]: 100 * f["rho_wp2"] for f in fam}
SUB = {f["ratio"]: "rho_r" + str(f["ratio"]).replace(".", "p") for f in fam}

acc = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(ROOT, "results", "all_results.csv"))):
    if r["batch"] == "wp1" and r["valid"] == "1" and r["condition"] == "nominal":
        acc[(r["substrate"], r["agent"])][int(r["seed"])] = float(r["acceptance_ratio"])


def diff(ratio, a, b, seeds=SEEDS):
    da, db = acc[(SUB[ratio], a)], acc[(SUB[ratio], b)]
    missing = [s for s in seeds if s not in da or s not in db]
    if missing:
        raise SystemExit(f"missing seeds {missing} at ratio {ratio} for {a}/{b}")
    return np.array([100 * (da[s] - db[s]) for s in seeds])


def ols(x, y):
    s, i = np.polyfit(x, y, 1)
    return s, i


def cluster_boot(groups, rng):
    """groups: list of (rho, values). Resample values within each group."""
    slopes, icpts = np.empty(B), np.empty(B)
    xs = np.concatenate([np.full(len(v), r) for r, v in groups])
    for b in range(B):
        ys = np.concatenate([rng.choice(v, len(v), replace=True) for _, v in groups])
        slopes[b], icpts[b] = ols(xs, ys)
    return slopes, icpts


rng = np.random.default_rng(20261012)
ratios = sorted(RHO)

print("=== Level means (WP2.1, n = 10 seeds each) ===")
print(f"{'ratio':>5s} {'rho_wp2%':>8s} {'acwidest':>9s} {'aconly':>8s} {'greedy':>8s} "
      f"{'F pp':>13s} {'G pp':>13s}")
Fg, Gall = [], []
for r in ratios:
    F, G = diff(r, "ac_dqn_widest", "aconly"), diff(r, "ac_dqn_widest", "greedy")
    Fg.append((RHO[r], F))
    m = lambda a: np.mean([acc[(SUB[r], a)][s] for s in SEEDS])
    print(f"{r:5.1f} {RHO[r]:8.2f} {m('ac_dqn_widest'):9.4f} {m('aconly'):8.4f} "
          f"{m('greedy'):8.4f} {F.mean():+6.2f}±{F.std(ddof=1):4.2f} {G.mean():+6.2f}±{G.std(ddof=1):4.2f}")

# W1
x = np.concatenate([np.full(len(v), r) for r, v in Fg])
y = np.concatenate([v for _, v in Fg])
slope, icpt = ols(x, y)
bs, bi = cluster_boot(Fg, rng)
lo, hi = np.percentile(bs, [2.5, 97.5])
p_w1 = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
w1 = lo > 0
print(f"\n=== W1: slope of F on rho ===\n  F = {slope:.3f}·rho {icpt:+.3f} pp   "
      f"slope CI95 [{lo:.3f}, {hi:.3f}]   bootstrap p = {p_w1:.4f}   -> {'SUPPORTED' if w1 else 'not supported'}")

# W2
F0 = diff(0.4, "ac_dqn_widest", "aconly")
n = len(F0); se = F0.std(ddof=1) / np.sqrt(n)
p_lo = stats.t.sf((F0.mean() + EQ) / se, n - 1)       # H0: mean <= -0.5
p_hi = stats.t.cdf((F0.mean() - EQ) / se, n - 1)      # H0: mean >= +0.5
p_w2 = max(p_lo, p_hi)
ci90 = F0.mean() + np.array([-1, 1]) * stats.t.ppf(0.95, n - 1) * se
print(f"\n=== W2: F at ratio 0.4 (rho = {RHO[0.4]:.2f}%) within ±{EQ} pp ===\n"
      f"  mean {F0.mean():+.3f} pp  90% CI [{ci90[0]:+.3f}, {ci90[1]:+.3f}]  "
      f"TOST p = {p_w2:.4f}   -> {'SUPPORTED' if p_w2 < ALPHA else 'not supported'}")

# W3
print("\n=== W3: registered out-of-level predictions ===")
errs, inside = [], []
for r in (1.0, 1.6, 2.5):
    obs = diff(r, "ac_dqn_widest", "aconly").mean()
    p, a, b = PRED[r]
    errs.append(abs(obs - p)); inside.append(a <= obs <= b)
    print(f"  ratio {r}: rho {RHO[r]:.2f}%  predicted {p:+.2f} [{a:+.2f}, {b:+.2f}]  "
          f"observed {obs:+.2f}  error {obs - p:+.2f}  {'inside' if inside[-1] else 'OUTSIDE'}")
mae = float(np.mean(errs))
w3 = mae <= 1.0 and all(inside)
print(f"  MAE {mae:.2f} pp (≤ 1.0 required), inside {sum(inside)}/3 (3 required)"
      f"   -> {'SUPPORTED' if w3 else 'not supported'}")

# Holm over W1, W2, W3
print("\n=== Holm-Bonferroni over {W1, W2, W3} ===")
fam_p = sorted([("W1", p_w1), ("W2", p_w2), ("W3 (pass/fail)", 0.0 if w3 else 1.0)],
               key=lambda kv: kv[1])
stop, holm = False, {}
for i, (k, p) in enumerate(fam_p):
    thr = ALPHA / (len(fam_p) - i)
    rej = (not stop) and p <= thr
    stop = stop or not rej
    holm[k.split()[0]] = rej
    print(f"  {k:15s} p = {p:.4f}  threshold {thr:.4f}  {'significant' if rej else 'n.s.'}")

# W4: fresh data only
fresh = []
for r in ratios:
    seeds = [s for s in SEEDS if s >= 44] if r in WP1_LEVELS else SEEDS
    fresh.append((RHO[r], diff(r, "ac_dqn_widest", "greedy", seeds)))
xg = np.concatenate([np.full(len(v), r) for r, v in fresh])
yg = np.concatenate([v for _, v in fresh])
gs, gi = ols(xg, yg)
gbs, _ = cluster_boot(fresh, rng)
p_w4 = (gbs >= 0).mean()
glo, ghi = np.percentile(gbs, [2.5, 97.5])
print(f"\n=== W4: foresight G on rho, fresh data only (n = {len(yg)}) ===\n"
      f"  G = {gs:.3f}·rho {gi:+.3f} pp   slope CI95 [{glo:.3f}, {ghi:.3f}]   "
      f"one-sided bootstrap p = {p_w4:.4f}   -> {'SUPPORTED' if p_w4 < ALPHA else 'not supported'}")

# W5
rs = -bi / bs
rstar = -icpt / slope
rlo, rhi = np.percentile(rs[np.isfinite(rs)], [2.5, 97.5])
print(f"\n=== W5: sign-change point rho* (descriptive) ===\n"
      f"  rho* = {rstar:.2f}%   bootstrap CI95 [{rlo:.2f}, {rhi:.2f}]%")

# Gate 2 (WP2.1 part)
print("\n=== Gate 2 (WP2.1 part) ===")
W1, W2, W3 = holm.get("W1", False), holm.get("W2", False), holm.get("W3", False)
if W1 and W2 and W3:
    verdict = "predictive criterion (fitted slope + rho*); 'law' only if WP2.3 also passes"
elif W1 and W2:
    verdict = "criterion: monotone and zero at rho = 0, not quantitatively predictive"
elif W2:
    verdict = "necessary condition only: rho ~ 0 => no value from path flexibility"
else:
    verdict = "W2 fails: structural claim wrong as stated; stop and re-examine"
print(f"  W1={W1} W2={W2} W3={W3} -> {verdict}")
