"""W3 predictions for paper/preregistration_wp2.md.

Fits F = acc(ac_dqn_widest) - acc(aconly) [pp] on rho_wp2 [%] using the WP1 batch only
(5 original family levels x seeds 42-43), then predicts F at the three new levels with
95 % OLS prediction intervals for a mean of n = 10 seeds.

Fills the RHO_* / PRED_* / PI_* placeholders in the pre-registration in place.
Usage:  python3 scripts/wp2_predictions.py
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(ROOT, "paper", "preregistration_wp2.md")
N_NEW = 10  # seeds per new level

fam = json.load(open(os.path.join(ROOT, "data", "rho_family", "family.json")))
rho = {f["ratio"]: 100 * f["rho_wp2"] for f in fam}

acc = {}
for r in csv.DictReader(open(os.path.join(ROOT, "results", "all_results.csv"))):
    if r["batch"] == "wp1" and r["substrate"].startswith("rho_") and r["valid"] == "1":
        acc[(r["substrate"], r["agent"], int(r["seed"]))] = float(r["acceptance_ratio"])

x, y = [], []
for f in fam:
    sub = "rho_r" + str(f["ratio"]).replace(".", "p")
    for s in (42, 43):
        a, b = acc.get((sub, "ac_dqn_widest", s)), acc.get((sub, "aconly", s))
        if a is not None and b is not None:
            x.append(rho[f["ratio"]])
            y.append(100 * (a - b))
x, y = np.array(x), np.array(y)
n = len(x)
slope, icpt = np.polyfit(x, y, 1)
res = y - (slope * x + icpt)
s2 = res @ res / (n - 2)
sxx = ((x - x.mean()) ** 2).sum()
tq = stats.t.ppf(0.975, n - 2)
print(f"fit on WP1 (n={n} points): F = {slope:.3f} * rho {icpt:+.3f}   resid sd {np.sqrt(s2):.3f} pp")

text = open(PREREG).read()
for ratio, r in rho.items():
    text = text.replace(f"RHO_{ratio}", f"{r:.2f}")
for ratio in (1.0, 1.6, 2.5):
    x0 = rho[ratio]
    pred = slope * x0 + icpt
    # interval for the MEAN of N_NEW new observations at x0
    half = tq * np.sqrt(s2 * (1 / N_NEW + 1 / n + (x0 - x.mean()) ** 2 / sxx))
    print(f"ratio {ratio}: rho_wp2={x0:.2f}%  predicted F={pred:+.2f} pp  "
          f"PI95=[{pred-half:+.2f}, {pred+half:+.2f}]")
    text = text.replace(f"PRED_{ratio}", f"{pred:+.2f}")
    text = text.replace(f"PI_{ratio}", f"[{pred-half:+.2f}, {pred+half:+.2f}]")
text = text.replace("Predictions below come from the OLS fit",
                    f"Predictions below come from the OLS fit "
                    f"(F = {slope:.3f}·ρ {icpt:+.3f} pp, n = {n})")
open(PREREG, "w").write(text)
print("filled", PREREG)
