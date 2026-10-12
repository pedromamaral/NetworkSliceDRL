"""WP2.3 predictions (paper/preregistration_wp2.md section 6 + amendment 1).

Predict F on each SNDlib network from the WP2.1 W1 fit (OLS of F on rho_wp2 over all
80 (level, seed) points), with the 95 % OLS prediction interval for the MEAN of the
5 seeds (42-46) that WP2.3 trains. Also records the constant predictor required by
amendment 1 (mean F over the WP2.1 family, all levels and seeds).

Usage:  python3 scripts/wp23_predictions.py   -> data/sndlib_topo/predictions.json
"""
import csv, json, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_NEW = 5
fam = json.load(open(os.path.join(ROOT, "data", "rho_family", "family.json")))
acc = {}
for r in csv.DictReader(open(os.path.join(ROOT, "results", "all_results.csv"))):
    if r["batch"] == "wp1" and r["valid"] == "1" and r["condition"] == "nominal":
        acc[(r["substrate"], r["agent"], int(r["seed"]))] = float(r["acceptance_ratio"])
x, y = [], []
for f in fam:
    sub = "rho_r" + str(f["ratio"]).replace(".", "p")
    for s in range(42, 52):
        x.append(100 * f["rho_wp2"])
        y.append(100 * (acc[(sub, "ac_dqn_widest", s)] - acc[(sub, "aconly", s)]))
x, y = np.array(x), np.array(y)
n = len(x)
slope, icpt = np.polyfit(x, y, 1)
res = y - (slope * x + icpt)
s2 = res @ res / (n - 2)
sxx = ((x - x.mean()) ** 2).sum()
tq = stats.t.ppf(0.975, n - 2)
const = float(y.mean())
nets = json.load(open(os.path.join(ROOT, "data", "sndlib_topo", "networks.json")))
out = {"fit": {"slope": slope, "intercept": icpt, "n": n, "resid_sd": float(np.sqrt(s2))},
       "constant_predictor": const, "seeds": [42, 43, 44, 45, 46], "networks": []}
print(f"W1 fit: F = {slope:.4f} rho {icpt:+.4f} pp (n={n}), constant predictor {const:+.3f} pp")
for net in sorted(nets, key=lambda r: r["rho_wp2"]):
    x0 = 100 * net["rho_wp2"]
    p = slope * x0 + icpt
    h = tq * np.sqrt(s2 * (1 / N_NEW + 1 / n + (x0 - x.mean()) ** 2 / sxx))
    out["networks"].append({"name": net["name"], "rho_wp2": x0, "pred_F": p,
                            "pi_lo": p - h, "pi_hi": p + h})
    print(f"  {net['name']:14s} rho={x0:5.2f}%  F_pred={p:+.2f} pp  PI95=[{p-h:+.2f}, {p+h:+.2f}]")
json.dump(out, open(os.path.join(ROOT, "data", "sndlib_topo", "predictions.json"), "w"), indent=1)
