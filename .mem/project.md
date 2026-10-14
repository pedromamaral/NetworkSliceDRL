# Project — netslice-drl (updated 2026-10-14)

## Research question
When does *learning* help joint slice admission control (AC) + routing (RA) on an IP/MPLS core,
and which part of the decision (admit vs. route) carries the value? Answer framed via the
routing-headroom statistic rho.

## Model (fixed; changes need Pedro — CLAUDE.md §14)
- count reward (+1/admit) + hard capacity (admission_mode hard). Revenue/soft = ablation only.
- State = [tau,d,b,p,M_flat,n_inel,n_el,B_flat] (OSS counters only). K=3 default.
- rho = P[some candidate path feasible AND shortest infeasible], measured under greedy
  (scripts/measure_rho.py, eval seeds 142-146 x 100 eps).

## Target claims (current; original "47.5% more slices / 18% more fulfilled" are SUPERSEDED)
- C1 Learned admission + widest-path routing (ac_dqn_widest) beats greedy, trunk reservation
  and learned joint routing on acceptance, on every substrate. (WP1, supported — K11)
- C2 rho predicts the value of path flexibility: F = acc(acwidest)-acc(aconly) ~ 0.371*rho + 0.13pp,
  F~0 at rho=0. "criterion" (not "law") unless WP2.3 passes. (WP2.1 supported; WP2.2/2.3 pending)
- C3 Foresight/flexibility trade-off: G = acc(acwidest)-acc(greedy) falls with rho (-0.114*rho+2.05).
- C4 Mechanism: learned admission keeps more future arrivals feasible (54.9% vs 46.2%).
- C5 (theory, WP3) lemma: single-homed + access-limited => rho=0; proposition on slope ~0.37. Not started.
- C6 Learned AC vs analytic online pricing (exp-pricing, WP4.1). Pending analysis; may fail (K16).

## Metrics
- Primary: acceptance_ratio on held-out eval seed = train_seed+100, 200 eps (fulfillment ≡1 under hard).
- Seeds: 42-46 (5) default; WP2.1 uses 42-51. Report mean ± sd; paired tests across seeds.
- "Good" = difference > seed noise (learned sd 0.002-0.006) and pre-registered criterion met.

## Success criteria
- Every confirmatory test is pre-registered (paper/preregistration_wp{1,2,4}.md) before runs.
- WP2.3: MAE<=1pp, >=7/9 in PI, beat constant predictor (+2.864pp), Spearman>0.
- WP2.2: W6-W9, slope invariance ±0.10 (data/rho_family_germany50/predictions.json).
