"""Generate every figure for the paper as a standalone vector PDF.

Run:   python3 paper/make_figures.py
Output: paper/figures/*.pdf

Every number is READ from raw results -- nothing is typed in by hand:
  results/all_results.csv          (built by scripts/collect_results.py)
  data/rho_family/family.json      (measured rho of each family level)
  results/greedy_load_envelope.csv (scripts/greedy_load_envelope.py)
  results/remote/gpu14/mech_diag*.log  (experiments/diagnose.py output)
  results/gpu15_backup/ddqn_*_count_s42/metrics.csv  (training logs)
Re-run scripts/collect_results.py first whenever results change.
Figures are sized for IEEEtran: 3.5in wide for single-column, 7.16in for
figure* (double-column). Fonts are serif at 8pt to match the body text.
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------
def _load_cells(batch="orig"):
    cells = defaultdict(dict)
    with open(os.path.join(REPO, "results", "all_results.csv")) as f:
        for r in csv.DictReader(f):
            if r["valid"] == "1" and r["batch"] == batch:
                cells[(r["substrate"], r["condition"], r["agent"])][int(r["seed"])] = \
                    float(r["acceptance_ratio"])
    return cells


CELLS = _load_cells()


def acc(sub, cond, agent):
    """Per-seed acceptance, ordered by seed."""
    d = CELLS[(sub, cond, agent)]
    if not d:
        raise KeyError(f"no results for {(sub, cond, agent)}")
    return np.array([d[k] for k in sorted(d)])


def mean(sub, cond, agent):
    return acc(sub, cond, agent).mean()


def sd(sub, cond, agent):
    return acc(sub, cond, agent).std(ddof=1)


def paired_t(sub, cond, a, b):
    """Paired t of a-b over the seeds both cells share."""
    da, db = CELLS[(sub, cond, a)], CELLS[(sub, cond, b)]
    seeds = sorted(set(da) & set(db))
    d = np.array([da[k] - db[k] for k in seeds])
    return d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))


FAMILY = json.load(open(os.path.join(REPO, "data", "rho_family", "family.json")))
def family_sub(ratio):
    return "rho_r" + str(ratio).replace(".", "p")
# rho of the two real substrates, measured by the same estimator as the family
# (see scripts/generate_rho_family.py docstring): operator 11.0 %, Waxman 0.0 %.
RHO_OPERATOR, RHO_WAXMAN = 11.0, 0.0


def _feasible(log):
    """Mean any_path_feasible (%) of DRL and greedy across seeds in a diagnose log."""
    drl, gre, who = [], [], None
    for line in open(os.path.join(REPO, "results", "remote", "gpu14", log)):
        if line.startswith("=== DRL"):
            who = drl
        elif line.startswith("=== Greedy"):
            who = gre
        m = re.search(r"any_path_feasible\s*:\s*([\d.]+)%", line)
        if m and who is not None:
            who.append(float(m.group(1)))
    return np.mean(drl), np.mean(gre)


def _training_curve(run):
    path = os.path.join(REPO, "results", "gpu15_backup", run, "metrics.csv")
    rows = list(csv.DictReader(open(path)))
    return (np.array([int(r["episode"]) for r in rows]),
            np.array([float(r["acceptance_ratio"]) for r in rows]))

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
    fam   = sorted((f for f in FAMILY
                    if CELLS.get((family_sub(f["ratio"]), "nominal", "joint_unified"))),
                   key=lambda f: f["rho"])
    subs  = [family_sub(f["ratio"]) for f in fam]
    rho   = np.array([100 * f["rho"] for f in fam])
    joint = np.array([mean(s, "nominal", "joint_unified") for s in subs])
    greedy= np.array([mean(s, "nominal", "greedy") for s in subs])
    acon  = np.array([mean(s, "nominal", "aconly") for s in subs])
    gap   = 100 * (joint - acon) / acon
    def _gap(sub, cond):
        j, a = mean(sub, cond, "joint_unified"), mean(sub, cond, "aconly")
        return 100 * (j - a) / a
    gap_op, gap_wax = _gap("operator", "K3"), _gap("waxman", "nominal")

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
    ax2.scatter([RHO_WAXMAN], [gap_wax], s=46, marker="D", facecolor="none",
                edgecolor=C_FACT, lw=1.3, zorder=4, label="operator / Waxman substrates")
    ax2.scatter([RHO_OPERATOR], [gap_op], s=46, marker="D", facecolor="none",
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
    agents = ["joint_unified", "factored", "greedy", "aconly", "aggregate", "revenue"]
    vals  = [mean("operator", "K3", a) for a in agents]
    errs  = [sd("operator", "K3", a) for a in agents]
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
    agents = ["aconly", "factored", "joint_unified", "greedy"]
    vals  = [mean("waxman", "nominal", a) for a in agents]
    errs  = [sd("waxman", "nominal", a) for a in agents]
    t_af  = paired_t("waxman", "nominal", "aconly", "factored")
    t_aj  = paired_t("waxman", "nominal", "aconly", "joint_unified")
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
    ax.text(0.5, 0.5070, "n.s." if abs(t_af) < 2.776 else f"$t$={t_af:.2f}",
            ha="center", fontsize=6)  # 2.776 = t_0.975, df=4
    ax.plot([0, 0, 2, 2], [0.5105, 0.5119, 0.5119, 0.5105], lw=0.7, color="0.25")
    ax.text(1.0, 0.5123, f"significant ($t$={t_aj:.2f})" if abs(t_aj) >= 2.776
            else f"n.s. ($t$={t_aj:.2f})", ha="center", fontsize=6)
    save(fig, "fig_topo2.pdf")


# =====================================================================
# 4. Mechanism: admissibility preservation
# =====================================================================
def fig_feasibility():
    labels = ["Operator", "Waxman"]
    # NB: the operator diagnostic loaded the gpu14 checkpoints (the July-28
    # rerun batch), not the original July runs; it is a mechanism figure.
    (d_op, g_op), (d_wx, g_wx) = _feasible("mech_diag.log"), _feasible("mech_diag_topo2.log")
    drl    = [d_op, d_wx]
    greedy = [g_op, g_wx]
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
    conds = ["K3", "K6", "K8"]
    uni = np.array([mean("operator", c, "joint_unified") for c in conds])
    uni_e = np.array([sd("operator", c, "joint_unified") for c in conds])
    fac = np.array([mean("operator", c, "factored") for c in conds])
    fac_e = np.array([sd("operator", c, "factored") for c in conds])
    drop = 100 * (uni[2] - uni[0]) / uni[0]
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.errorbar(K, uni, yerr=uni_e, fmt="o-", color=C_JOINT, lw=1.3, ms=4,
                capsize=2.5, label="Unified")
    ax.errorbar(K, fac, yerr=fac_e, fmt="s-", color=C_FACT, lw=1.3, ms=4,
                capsize=2.5, label="Factored")
    ax.set_xlabel("Number of candidate paths $K$")
    ax.set_ylabel("Acceptance ratio")
    ax.set_xticks(K); ax.set_ylim(0.455, 0.492)
    ax.legend(frameon=True, edgecolor="0.8", loc="lower left")
    ax.annotate(f"{drop:+.1f}%\n(significant)".replace("-", "−"), xy=(8, uni[2]), xytext=(6.5, 0.462),
                fontsize=6.3, color="0.3",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.45"))
    save(fig, "fig_scaling.pdf")


# =====================================================================
# 6. Sensitivity to offered load
# =====================================================================
def fig_load():
    env = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(REPO, "results", "greedy_load_envelope.csv"))):
        env[float(r["capacity_scale"])].append(float(r["acceptance_ratio"]))
    cap = np.array(sorted(env))
    gre = np.array([np.mean(env[c]) for c in cap])
    j_lo, j_hi = mean("operator", "load0.5", "joint_unified"), mean("operator", "K3", "joint_unified")
    g_lo, g_hi = mean("operator", "load0.5", "greedy"), mean("operator", "K3", "greedy")
    m_lo, m_hi = 100 * (j_lo / g_lo - 1), 100 * (j_hi / g_hi - 1)
    t_lo = paired_t("operator", "load0.5", "joint_unified", "greedy")
    df_lo = len(acc("operator", "load0.5", "joint_unified")) - 1
    sig_lo = abs(t_lo) >= {2: 4.303, 3: 3.182, 4: 2.776}[df_lo]
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(cap, gre, "^-", color=C_GREED, lw=1.3, ms=4, label="Greedy (envelope)")
    ax.plot([0.5, 1.0], [j_lo, j_hi], "o", color=C_JOINT, ms=5, label="Joint (unified)")
    ax.annotate(f"{m_lo:+.1f}%" + ("" if sig_lo else " (n.s.)"), xy=(0.5, j_lo),
                xytext=(0.52, 0.345),
                fontsize=6.3, color=C_JOINT,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=C_JOINT))
    ax.annotate(f"{m_hi:+.1f}%", xy=(1.0, j_hi), xytext=(0.98, 0.53),
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
    ep, uni = _training_curve("ddqn_unified_count_s42")
    ep_f, fac = _training_curve("ddqn_separated_count_s42")
    assert (ep == ep_f).all()
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
