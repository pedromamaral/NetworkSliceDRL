# Pre-registration — WP4.1 exponential-pricing comparator

**Registered:** 2026-10-13, on the commit that adds this file, before any evaluation of the
comparator. Code: `src/baselines/exp_pricing.py`, `scripts/run_exp_pricing.py`.

**Why.** The online-competitive line (Awerbuch–Azar–Plotkin 1993; recent slicing variants
such as OPA) is the strongest non-learned approach to joint admission and routing. Two
questions: (i) does the ρ criterion predict the routing gain of a controller that has
nothing to do with our DQN — if yes, ρ is a property of the substrate, not of the learner;
(ii) does learned admission beat exponential pricing?

## 1. Design

**Controller.** Link price (b/C_e)(μ^{u_e} − 1), u_e = utilisation before admission; cost of
path k = sum over the slice's connections (one path index serves all connections, as in the
simulator). Admit on the min-cost path if cost ≤ σ. Two variants: `ap_cheapest` (min over
the K = 3 candidates) and `ap_shortest` (k = 0 only).

**Calibration.** (μ, σ) per substrate and variant by full grid (μ ∈ {2, 4, …, 128},
σ ∈ {0.02, …, 5, ∞}) on the **training** traffic of seed 42, 40 episodes; frozen. (A
per-seed calibration, as for trunk reservation, would cost ≈ 35 core-hours; the
single-seed calibration is the cheaper choice and can only weaken the comparator.)

**Evaluation.** Held-out traffic, seed + 100, 200 episodes; seeds 42–51 on the eight WP2.1
family levels, 42–46 on operator and Waxman. Learned arms are the existing WP1/WP2.1 runs
(same traffic per seed).

## 2. Hypotheses

- **P1** — ρ predicts pricing's routing gain. F_AP = acc(`ap_cheapest`) − acc(`ap_shortest`)
  has positive slope on ρ_wp2 across the family; 95 % cluster-bootstrap CI excludes 0.
- **P2** — no routing gain without headroom: F_AP at ratio 0.4 equivalent to 0 within
  ±0.5 pp (TOST, n = 10).
- **P3** — learned admission beats exponential pricing: acc(`ac_dqn_widest`) >
  acc(`ap_cheapest`) on operator and on Waxman (paired, two-sided t, n = 5 each).

Holm–Bonferroni over {P1, P2, P3-operator, P3-Waxman}. Descriptive: slope of F_AP vs slope
of F (WP2.1), and acceptance of both pricing variants on every substrate.

## 3. Stopping rule

Complete when every substrate has both variants at all registered seeds. A substrate that
fails to run is re-run once; if it fails again it is reported missing.
