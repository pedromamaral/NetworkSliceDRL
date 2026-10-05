"""Generate every figure for the paper as a standalone vector PDF.

Run:   python3 paper/make_figures.py
Output: paper/figures/*.pdf

All numbers are the measured experimental values; nothing here is illustrative.
Figures are sized for IEEEtran: 3.5in wide for single-column, 7.16in for
figure* (double-column). Fonts are serif at 8pt to match the body text.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_JOINT = "#1f77b4"
C_FACT  = "#ff7f0e"
C_GREED = "#2ca02c"
C_ACO   = "#9467bd"
C_AGG   = "#8c564b"
C_REV   = "#7f7f7f"

SINGLE = (3.5, 2.35)
DOUBLE = (7.16, 2.7)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {name}")


# =====================================================================
# 1. THE ROUTING-HEADROOM LAW  (centrepiece, double column)
# =====================================================================
def fig_rho():
    rho   = np.array([0.00, 0.70, 5.08, 12.91, 14.42])
    joint = np.array([0.4906, 0.4829, 0.4820, 0.4843, 0.4776])
    greedy= np.array([0.4708, 0.4740, 0.4788, 0.4849, 0.4774])
    acon  = np.array([0.4934, 0.4910, 0.4728, 0.4386, 0.4236])
    gap   = 100 * (joint - acon) / acon

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)

    # (a) acceptance of each policy as rho grows
    ax1.plot(rho, joint,  "o-", color=C_JOINT, lw=1.3, ms=4, label="Joint (AC+RA)")
    ax1.plot(rho, greedy, "^-", color=C_GREED, lw=1.3, ms=4, label="Greedy")
    ax1.plot(rho, acon,   "s-", color=C_ACO,   lw=1.3, ms=4, label="AC-only")
    ax1.set_xlabel(r"Routing headroom $\rho$ (\%)" if False else "Routing headroom ρ (%)")
    ax1.set_ylabel("Acceptance ratio")
    ax1.set_title("(a) policies as routing headroom grows", fontsize=8)
    ax1.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
    ax1.set_ylim(0.41, 0.51)

    # (b) the law: gap vs rho with least-squares fit
    s, b = np.polyfit(rho, gap, 1)
    r = np.corrcoef(rho, gap)[0, 1]
    xs = np.linspace(-0.6, 15.4, 50)
    ax2.plot(xs, s * xs + b, "-", color="0.45", lw=1.1, zorder=1,
             label=f"fit: {s:.2f}ρ − {abs(b):.2f}  (r²={r*r:.3f})")
    ax2.scatter(rho, gap, s=34, color=C_JOINT, zorder=3,
                label="controlled ρ family")
    # the two real substrates, as out-of-family validation
    ax2.scatter([0.0], [-2.30], s=46, marker="D", facecolor="none",
                edgecolor=C_FACT, lw=1.3, zorder=4, label="operator / Waxman substrates")
    ax2.scatter([11.0], [7.40], s=46, marker="D", facecolor="none",
                edgecolor=C_FACT, lw=1.3, zorder=4)
    ax2.axhline(0, color="0.3", lw=0.8, ls=":", zorder=2)
    ax2.set_xlabel("Routing headroom ρ (%)")
    ax2.set_ylabel("Joint − AC-only (%)")
    ax2.set_title("(b) the routing-headroom law", fontsize=8)
    ax2.legend(frameon=True, framealpha=0.9, edgecolor="0.8", loc="upper left")
    ax2.set_xlim(-1.2, 16)
    # keep this clear of the data cluster near the origin
    ax2.annotate("below 0: AC-only wins", xy=(8.4, -2.4),
                 fontsize=6.3, color="0.35", ha="left", va="center")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_rho.pdf")


# =====================================================================
# 2. Operator topology: main comparison
# =====================================================================
def fig_main():
    names = ["Joint\n(unified)", "Joint\n(factored)", "Greedy", "AC-only", "Aggregate\n-state", "Revenue\nthresh."]
    vals  = [0.4806, 0.4739, 0.4637, 0.4476, 0.4195, 0.1308]
    errs  = [0.0043, 0.0049, 0.0028, 0.0022, 0.0012, 0.0011]
    cols  = [C_JOINT, C_FACT, C_GREED, C_ACO, C_AGG, C_REV]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(names))
    ax.bar(x, vals, yerr=errs, capsize=2.5, color=cols, edgecolor="black", lw=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.016, f"{v:.3f}", ha="center", fontsize=6.3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.3)
    ax.set_ylabel("Acceptance ratio")
    ax.set_ylim(0, 0.56)
    save(fig, "fig_main.pdf")


# =====================================================================
# 3. Topology 2 (rho = 0): the architecture choice
# =====================================================================
def fig_topo2():
    names = ["AC-only", "Joint\n(factored)", "Joint\n(unified)", "Greedy"]
    vals  = [0.4970, 0.4930, 0.4858, 0.4683]
    errs  = [0.0044, 0.0048, 0.0028, 0.0026]
    cols  = [C_ACO, C_FACT, C_JOINT, C_GREED]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(len(names))
    ax.bar(x, vals, yerr=errs, capsize=2.5, color=cols, edgecolor="black", lw=0.5)
    # labels must clear the error-bar caps, not just the bar tops
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.0012, f"{v:.4f}", ha="center", fontsize=6.3)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("Acceptance ratio")
    ax.set_ylim(0.455, 0.518)
    # significance brackets, stacked above the labels
    ax.plot([0, 0, 1, 1], [0.5052, 0.5066, 0.5066, 0.5052], lw=0.7, color="0.25")
    ax.text(0.5, 0.5070, "n.s.", ha="center", fontsize=6)
    ax.plot([0, 0, 2, 2], [0.5105, 0.5119, 0.5119, 0.5105], lw=0.7, color="0.25")
    ax.text(1.0, 0.5123, "significant ($t$=5.97)", ha="center", fontsize=6)
    save(fig, "fig_topo2.pdf")


# =====================================================================
# 4. Mechanism: admissibility preservation
# =====================================================================
def fig_feasibility():
    labels = ["Operator", "Waxman"]
    drl    = [54.89, 53.27]
    greedy = [46.19, 46.68]
    x = np.arange(len(labels)); w = 0.33
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.bar(x - w/2, drl,    w, label="Learned policy", color=C_JOINT, edgecolor="black", lw=0.5)
    ax.bar(x + w/2, greedy, w, label="Greedy",         color=C_GREED, edgecolor="black", lw=0.5)
    for i,(a,b) in enumerate(zip(drl, greedy)):
        ax.text(i - w/2, a + 0.5, f"{a:.1f}", ha="center", fontsize=6.3)
        ax.text(i + w/2, b + 0.5, f"{b:.1f}", ha="center", fontsize=6.3)
        ax.annotate("", xy=(i, b), xytext=(i, a),
                    arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.3"))
        ax.text(i + 0.045, (a+b)/2, f"+{a-b:.1f} pp", fontsize=6.2, color="0.25")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Feasible arrivals (%)")
    ax.set_ylim(40, 60); ax.legend(frameon=True, edgecolor="0.8")
    save(fig, "fig_feasibility.pdf")


# =====================================================================
# 5. Scaling with K
# =====================================================================
def fig_scaling():
    K = np.array([3, 6, 8])
    uni = np.array([0.4806, 0.4802, 0.4679]); uni_e = np.array([0.0043, 0.0048, 0.0071])
    fac = np.array([0.4739, 0.4752, 0.4727]); fac_e = np.array([0.0049, 0.0062, 0.0040])
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.errorbar(K, uni, yerr=uni_e, fmt="o-", color=C_JOINT, lw=1.3, ms=4,
                capsize=2.5, label="Unified")
    ax.errorbar(K, fac, yerr=fac_e, fmt="s-", color=C_FACT, lw=1.3, ms=4,
                capsize=2.5, label="Factored")
    ax.set_xlabel("Number of candidate paths $K$")
    ax.set_ylabel("Acceptance ratio")
    ax.set_xticks(K); ax.set_ylim(0.455, 0.492)
    ax.legend(frameon=True, edgecolor="0.8", loc="lower left")
    ax.annotate("−2.6%\n(significant)", xy=(8, 0.4679), xytext=(6.5, 0.462),
                fontsize=6.3, color="0.3",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.45"))
    save(fig, "fig_scaling.pdf")


# =====================================================================
# 6. Sensitivity to offered load
# =====================================================================
def fig_load():
    cap = np.array([0.35, 0.5, 0.7, 1.0, 1.5])
    gre = np.array([0.2375, 0.3039, 0.3742, 0.4650, 0.5917])
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(cap, gre, "^-", color=C_GREED, lw=1.3, ms=4, label="Greedy (envelope)")
    ax.plot([0.5, 1.0], [0.3036, 0.4806], "o", color=C_JOINT, ms=5, label="Joint (unified)")
    ax.annotate("+1.0% (n.s.)", xy=(0.5, 0.3036), xytext=(0.52, 0.345),
                fontsize=6.3, color=C_JOINT,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=C_JOINT))
    ax.annotate("+3.6%", xy=(1.0, 0.4806), xytext=(0.98, 0.53),
                fontsize=6.3, color=C_JOINT,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=C_JOINT))
    ax.set_xlabel("Capacity scale (higher = lighter load)")
    ax.set_ylabel("Acceptance ratio")
    ax.set_xlim(0.28, 1.62); ax.set_ylim(0.19, 0.63)
    ax.legend(frameon=True, edgecolor="0.8", loc="upper left")
    save(fig, "fig_load.pdf")


# =====================================================================
# 7. Learning curves
# =====================================================================
def fig_learning():
    ep = np.arange(100, 2001, 100)
    uni = [0.3222,0.3556,0.3801,0.4010,0.4167,0.4310,0.4366,0.4492,0.4560,0.4579,
           0.4595,0.4621,0.4640,0.4600,0.4667,0.4659,0.4693,0.4691,0.4668,0.4677]
    fac = [0.2655,0.3060,0.3381,0.3631,0.3879,0.3967,0.4168,0.4301,0.4401,0.4462,
           0.4423,0.4536,0.4500,0.4554,0.4588,0.4599,0.4641,0.4655,0.4633,0.4634]
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(ep, uni, "-o", color=C_JOINT, lw=1.2, ms=2.6, label="Unified")
    ax.plot(ep, fac, "-s", color=C_FACT,  lw=1.2, ms=2.6, label="Factored")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Acceptance ratio")
    ax.set_ylim(0.24, 0.50); ax.legend(frameon=True, edgecolor="0.8", loc="lower right")
    save(fig, "fig_learning.pdf")


if __name__ == "__main__":
    print("generating figures ->", OUT)
    fig_rho(); fig_main(); fig_topo2(); fig_feasibility()
    fig_scaling(); fig_load(); fig_learning()
    print("done")
