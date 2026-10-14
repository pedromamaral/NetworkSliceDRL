"""Confirmatory analysis of WP2.3 (SNDlib out-of-sample) and WP4.1 (exponential pricing),
exactly as registered in paper/preregistration_wp2.md (§6, amendments 1-2) and
paper/preregistration_wp4.md (§2). Reads results/all_results.csv only.

WP2.3  MAE <= 1.0 pp; >= 7/9 observed means inside 95 % PI; MAE < constant predictor MAE;
       Spearman(pred, obs) > 0, one-sided permutation test (10 000), alpha 0.05.
WP4.1  P1 slope of F_AP = acc(ap_cheapest) - acc(ap_shortest) on rho_wp2 > 0, cluster bootstrap
       P2 F_AP at ratio 0.4 within +-0.5 pp (TOST, n = 10)
       P3 acc(ac_dqn_widest) > acc(ap_cheapest), paired two-sided t, operator and Waxman (n = 5)
       Holm over {P1, P2, P3-op, P3-Wx}.
"""

import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA, B, EQ = 0.05, 10_000, 0.5
rng = np.random.default_rng(20261014)

acc = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(ROOT, "results", "all_results.csv"))):
    if r["valid"] != "1" or r["batch"] not in ("wp1", "wp4"):
        continue
    if r["substrate"] == "operator" and r["condition"] not in ("K3", "nominal"):
        continue
    acc[(r["substrate"], r["agent"])][int(r["seed"])] = float(r["acceptance_ratio"])


def diff(sub, a, b, seeds):
    da, db = acc[(sub, a)], acc[(sub, b)]
    missing = [s for s in seeds if s not in da or s not in db]
    if missing:
        raise SystemExit(f"missing seeds {missing} on {sub} for {a}/{b}")
    return np.array([100 * (da[s] - db[s]) for s in seeds])


def cluster_boot(groups):
    xs = np.concatenate([np.full(len(v), r) for r, v in groups])
    out = np.empty(B)
    for b in range(B):
        ys = np.concatenate([rng.choice(v, len(v), replace=True) for _, v in groups])
        out[b] = np.polyfit(xs, ys, 1)[0]
    return out


# ---------------- WP2.3 ----------------
pred = json.load(open(os.path.join(ROOT, "data", "sndlib_topo", "predictions.json")))
S5 = pred["seeds"]
print("=== WP2.3: SNDlib out-of-sample (F = acwidest - aconly, n = 5) ===")
print(f"{'network':14s} {'rho%':>5s} {'pred':>6s} {'PI':>15s} {'obs':>11s} {'err':>6s} in  "
      f"{'G(acw-gr)':>10s}")
P, O, ins, G = [], [], 0, []
for n in pred["networks"]:
    sub = "sndlib_" + n["name"].replace("-", "_")
    F = diff(sub, "ac_dqn_widest", "aconly", S5)
    g = diff(sub, "ac_dqn_widest", "greedy", S5)
    inside = n["pi_lo"] <= F.mean() <= n["pi_hi"]
    ins += inside
    P.append(n["pred_F"]); O.append(F.mean()); G.append(g.mean())
    print(f"{n['name']:14s} {n['rho_wp2']:5.2f} {n['pred_F']:+6.2f} "
          f"[{n['pi_lo']:+.2f},{n['pi_hi']:+.2f}] {F.mean():+5.2f}±{F.std(ddof=1):4.2f} "
          f"{F.mean() - n['pred_F']:+6.2f} {'Y' if inside else 'n'}   {g.mean():+5.2f}±{g.std(ddof=1):4.2f}")
P, O = np.array(P), np.array(O)
mae = np.abs(O - P).mean()
mae_c = np.abs(O - pred["constant_predictor"]).mean()
rs = stats.spearmanr(P, O).statistic
perm = np.array([stats.spearmanr(P, rng.permutation(O)).statistic for _ in range(B)])
p_rs = (perm >= rs).mean()
rho = np.array([n["rho_wp2"] for n in pred["networks"]])
sl, ic = np.polyfit(rho, O, 1)
print(f"\nMAE {mae:.2f} pp (<=1.0: {mae <= 1.0}) | inside PI {ins}/9 (>=7: {ins >= 7}) | "
      f"constant-predictor MAE {mae_c:.2f} (beaten: {mae < mae_c}) | "
      f"Spearman {rs:+.3f} perm p {p_rs:.4f} ({p_rs < ALPHA})")
print(f"bias (mean obs - pred) {np.mean(O - P):+.2f} pp | descriptive OLS obs F = {sl:.3f}·rho {ic:+.2f}")
ok = mae <= 1.0 and ins >= 7
disc = mae < mae_c and p_rs < ALPHA
print("WP2.3 verdict:", "PASS (discriminating)" if ok and disc else
      "consistent but not discriminating" if ok else "FAIL")

# ---------------- WP4.1 ----------------
fam = json.load(open(os.path.join(ROOT, "data", "rho_family", "family.json")))
RHO = {"rho_r" + str(f["ratio"]).replace(".", "p"): 100 * f["rho_wp2"] for f in fam}
S10 = list(range(42, 52))
print("\n=== WP4.1: exponential pricing ===")
print(f"{'substrate':10s} {'rho%':>6s} {'ap_ch':>7s} {'ap_sh':>7s} {'F_AP':>11s} {'F(DQN)':>7s} "
      f"{'acw':>7s} {'acw-ap_ch':>10s} {'greedy':>7s}")
groups = []
for sub in sorted(RHO, key=RHO.get):
    Fap = diff(sub, "ap_cheapest", "ap_shortest", S10)
    F = diff(sub, "ac_dqn_widest", "aconly", S10)
    d = diff(sub, "ac_dqn_widest", "ap_cheapest", S10)
    groups.append((RHO[sub], Fap))
    m = lambda a: np.mean([acc[(sub, a)][s] for s in S10])
    print(f"{sub:10s} {RHO[sub]:6.2f} {m('ap_cheapest'):7.4f} {m('ap_shortest'):7.4f} "
          f"{Fap.mean():+5.2f}±{Fap.std(ddof=1):4.2f} {F.mean():+7.2f} {m('ac_dqn_widest'):7.4f} "
          f"{d.mean():+10.2f} {m('greedy'):7.4f}")

x = np.concatenate([np.full(len(v), r) for r, v in groups])
y = np.concatenate([v for _, v in groups])
s1, i1 = np.polyfit(x, y, 1)
bs = cluster_boot(groups)
lo, hi = np.percentile(bs, [2.5, 97.5])
p1 = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
print(f"\nP1: F_AP = {s1:.3f}·rho {i1:+.2f} pp  slope CI95 [{lo:.3f}, {hi:.3f}]  p = {p1:.4f}  "
      f"(DQN F slope 0.371; ratio {s1 / 0.371:.2f})")

F0 = diff("rho_r0p4", "ap_cheapest", "ap_shortest", S10)
se = F0.std(ddof=1) / np.sqrt(len(F0))
p2 = max(stats.t.sf((F0.mean() + EQ) / se, len(F0) - 1), stats.t.cdf((F0.mean() - EQ) / se, len(F0) - 1))
print(f"P2: F_AP(ratio 0.4) = {F0.mean():+.3f} pp  TOST p = {p2:.4f}")

p3 = {}
for sub in ("operator", "waxman"):
    d = diff(sub, "ac_dqn_widest", "ap_cheapest", S5)
    t = stats.ttest_1samp(d, 0.0)
    p3[sub] = t.pvalue
    print(f"P3-{sub}: acwidest - ap_cheapest = {d.mean():+.2f}±{d.std(ddof=1):.2f} pp  "
          f"t = {t.statistic:+.2f}  p = {t.pvalue:.4f}  (direction {'as predicted' if d.mean() > 0 else 'REVERSED'})")
    for a in ("ap_shortest", "greedy"):
        dd = diff(sub, "ap_cheapest", a, S5)
        print(f"    ap_cheapest - {a}: {dd.mean():+.2f}±{dd.std(ddof=1):.2f} pp")

# Holm; P3 counts as supported only if significant AND in the predicted direction
names = ["P1", "P2", "P3-operator", "P3-waxman"]
pv = [p1, p2, p3["operator"], p3["waxman"]]
direction = [s1 > 0, True,
             diff("operator", "ac_dqn_widest", "ap_cheapest", S5).mean() > 0,
             diff("waxman", "ac_dqn_widest", "ap_cheapest", S5).mean() > 0]
order = np.argsort(pv)
rej, stop = {}, False
for k, j in enumerate(order):
    thr = ALPHA / (len(pv) - k)
    stop = stop or pv[j] > thr
    rej[names[j]] = (not stop, thr)
print("\nHolm:")
for nm, p, dr in zip(names, pv, direction):
    r, thr = rej[nm]
    v = "SUPPORTED" if r and dr else ("REVERSED (sig.)" if r else "not supported")
    print(f"  {nm:12s} p = {p:.4f}  thr {thr:.4f}  -> {v}")
