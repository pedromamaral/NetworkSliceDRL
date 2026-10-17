"""E019 (exploratory, retrospective) — SP-penalty predictor & F decomposition. Card: .mem/experiments/E019-*.md

F = acc(ac_dqn_widest) - acc(aconly);  Dsp = acc(greedy) - acc(greedy_shortest);
G = acc(ac_dqn_widest) - acc(greedy);  H = acc(greedy) - acc(aconly)   (F = G + H).  All in pp, per-substrate means over seeds.
H1 quad term on rho vs on Dsp (F regressed on each, n=8 levels/family) + LOO MAE of F~Dsp vs F~rho, per family
H2 fit F~Dsp on wp21, predict 9 SNDlib nets: MAE<1.74, Spearman(Dsp,F)>.08 and p<.05  (also reverse g50 fit, extra)
H3 quad coef of G(rho) and H(rho), per family
H4 F ~ rho + Dsp partial R2 per family (and pooled), n small -> flagged

Usage: python3 scripts/analyze_e019.py > paper/e019_analysis_output.txt
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = ("ac_dqn_widest", "aconly", "greedy", "greedy_shortest")
acc = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(ROOT, "results", "all_results.csv"))):
    if r["batch"] == "wp1" and r["valid"] == "1" and r["condition"] == "nominal" and r["agent"] in AGENTS:
        acc[(r["substrate"], r["agent"])][int(r["seed"])] = float(r["acceptance_ratio"])


def point(sub):
    """per-substrate means over seeds common to all four agents; returns dict of pp metrics"""
    seeds = set.intersection(*[set(acc[(sub, a)]) for a in AGENTS])
    m = {a: 100 * np.mean([acc[(sub, a)][s] for s in seeds]) for a in AGENTS}
    return dict(n=len(seeds), F=m["ac_dqn_widest"] - m["aconly"], G=m["ac_dqn_widest"] - m["greedy"],
                H=m["greedy"] - m["aconly"], D=m["greedy"] - m["greedy_shortest"])


def family(dirname, prefix):
    fam = json.load(open(os.path.join(ROOT, "data", dirname, "family.json")))
    rows = []
    for f in sorted(fam, key=lambda f: f["ratio"]):
        p = point(prefix + str(f["ratio"]).replace(".", "p"))
        p.update(rho=100 * f["rho_wp2"], ratio=f["ratio"])
        rows.append(p)
    return rows


def col(rows, k):
    return np.array([r[k] for r in rows])


def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    dof = len(y) - X.shape[1]
    s2 = res @ res / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    p = 2 * stats.t.sf(np.abs(b / se), dof)
    r2 = 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))
    return b, p, r2, res


def quad_p(x, y):
    b, p, r2, _ = ols([x, x ** 2], y)
    return b[2], p[2], r2


def loo_mae(x, y):
    err = []
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        b = np.polyfit(x[k], y[k], 1)
        err.append(abs(np.polyval(b, x[i]) - y[i]))
    return float(np.mean(err))


fams = {"wp21": family("rho_family", "rho_r"), "g50": family("rho_family_germany50", "g50_r")}
nets = []
for n in json.load(open(os.path.join(ROOT, "data", "sndlib_topo", "networks.json"))):
    if n.get("status") != "ok":
        continue
    p = point("sndlib_" + n["name"].replace("-", "_"))
    p.update(rho=100 * n["rho_wp2"], name=n["name"])
    nets.append(p)

for name, rows in fams.items():
    print(f"=== {name}: level means (pp), n_seeds={rows[0]['n']} ===")
    print(f"{'ratio':>5s} {'rho%':>6s} {'F':>7s} {'G':>7s} {'H':>7s} {'Dsp':>7s}")
    for r in rows:
        print(f"{r['ratio']:5.1f} {r['rho']:6.2f} {r['F']:7.2f} {r['G']:7.2f} {r['H']:7.2f} {r['D']:7.2f}")
print(f"\n=== SNDlib nets (n_seeds={nets[0]['n']}) ===")
print(f"{'net':14s} {'rho%':>6s} {'F':>7s} {'G':>7s} {'H':>7s} {'Dsp':>7s}")
for r in sorted(nets, key=lambda r: r["rho"]):
    print(f"{r['name']:14s} {r['rho']:6.2f} {r['F']:7.2f} {r['G']:7.2f} {r['H']:7.2f} {r['D']:7.2f}")

print("\n=== H1: quadratic terms and LOO MAE of F (per family, n=8 level means) ===")
h1 = {}
for name, rows in fams.items():
    rho, D, F = col(rows, "rho"), col(rows, "D"), col(rows, "F")
    _, pr, r2r = quad_p(rho, F)
    _, pd_, r2d = quad_p(D, F)
    _, pdd, _ = quad_p(rho, D)
    mr, md = loo_mae(rho, F), loo_mae(D, F)
    h1[name] = (pd_ > .05 and pr <= .05 and md < mr)
    print(f"  {name}: quad p  F~rho {pr:.3f} | F~Dsp {pd_:.3f} | Dsp~rho {pdd:.3f}   LOO-MAE F~rho {mr:.3f}  F~Dsp {md:.3f}"
          f"   Pearson(Dsp,F)={np.corrcoef(D, F)[0, 1]:+.3f}  => per-family pass={h1[name]}")
print(f"H1 holds (as stated, both families): {all(h1.values())}  [falsified if fails in both: {not any(h1.values())}]")

print("\n=== H2: transfer to SNDlib (F ~ Dsp linear fit, 9 nets) ===")
Dn, Fn, rn = col(nets, "D"), col(nets, "F"), col(nets, "rho")
res2 = {}
for name, rows in fams.items():
    b = np.polyfit(col(rows, "D"), col(rows, "F"), 1)
    pred = np.polyval(b, Dn)
    mae = float(np.mean(np.abs(pred - Fn)))
    bias = float(np.mean(pred - Fn))
    res2[name] = mae
    print(f"  fit on {name}: F = {b[0]:.3f}·Dsp {b[1]:+.3f}  MAE {mae:.3f} pp  bias {bias:+.3f}  (threshold 1.74)")
rs, ps = stats.spearmanr(Dn, Fn)
rr, pr_ = stats.spearmanr(rn, Fn)
print(f"  Spearman(Dsp,F) = {rs:+.3f} (p={ps:.3f});  reference Spearman(rho,F) = {rr:+.3f} (p={pr_:.3f})")
print(f"  Dsp range nets [{Dn.min():.2f},{Dn.max():.2f}] vs wp21 [{col(fams['wp21'],'D').min():.2f},{col(fams['wp21'],'D').max():.2f}]"
      f" g50 [{col(fams['g50'],'D').min():.2f},{col(fams['g50'],'D').max():.2f}]")
print(f"H2 holds (wp21 fit): {res2['wp21'] < 1.74 and rs > .08 and ps < .05}")

print("\n=== H3: curvature of G and H vs rho (quad coef, p) ===")
for name, rows in fams.items():
    rho = col(rows, "rho")
    out = []
    for k in "GHF":
        c, p, _ = quad_p(rho, col(rows, k))
        out.append(f"{k}: {c:+.4f} (p={p:.3f})")
    print(f"  {name}: " + " | ".join(out))

print("\n=== H4: F ~ rho + Dsp (n small, overfit flag) ===")
pools = {**fams, "pooled": fams["wp21"] + fams["g50"]}
for name, rows in pools.items():
    rho, D, F = col(rows, "rho"), col(rows, "D"), col(rows, "F")
    _, _, r2_full, _ = ols([rho, D], F)
    _, _, r2_r, _ = ols([rho], F)
    _, _, r2_d, _ = ols([D], F)
    pr_rho = (r2_full - r2_d) / (1 - r2_d)   # partial R2 of rho given Dsp
    pr_D = (r2_full - r2_r) / (1 - r2_r)     # partial R2 of Dsp given rho
    print(f"  {name} (n={len(F)}): R2 full {r2_full:.3f} | rho only {r2_r:.3f} | Dsp only {r2_d:.3f} | partial R2 rho|Dsp {pr_rho:.3f}, Dsp|rho {pr_D:.3f}")
