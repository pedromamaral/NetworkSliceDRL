# Project — netslice-drl (updated 2026-10-18)

## Research question
When does *learning* help joint slice admission control (AC) + routing (RA) on an IP/MPLS core,
and which part of the decision (admit vs. route) carries the value? Answer framed via the
routing-headroom statistic rho.

## Model (fixed; changes need Pedro — CLAUDE.md §14)
- count reward (+1/admit) + hard capacity (admission_mode hard). Revenue/soft = ablation only.
- State = [tau,d,b,p,M_flat,n_inel,n_el,B_flat] (OSS counters only). K=3 default.
- rho = P[some candidate path feasible AND shortest infeasible], measured under greedy
  (scripts/measure_rho.py, eval seeds 142-146 x 100 eps).

## Target claims (D19, 2026-10-18; old C1-C6 superseded — history in decisions.md D16/D19)
- C1' Value of learning is in admission: learn admission + route widest ≥ learned joint routing (K11, K07). Supported.
- C2' Classical online control is the bar: exp-pricing beats DRL admission everywhere (K16); TR → greedy (K12);
  BR-EFP to add (A031). Honest negative result.
- C3' Flexibility value F predictable training-free: in-family by rho (K13,K17,K19), out-of-family by Δ_SP
  (K22 exploratory → A035 pre-registered), explained by Erlang reduced-load model (A029, kill-gated).
- C4' Mechanism: learned admission keeps future arrivals feasible (K06); foresight/flexibility trade-off G(rho) (K14, K22 caveat).
- Dropped: "DRL beats all baselines" (K16); "rho law transfers across cores" (K18,K20).
- Venue: IEEE TNSM; fallback Computer Networks.

## Metrics
- Primary: acceptance_ratio on held-out eval seed = train_seed+100, 200 eps (fulfillment ≡1 under hard).
- Seeds: 42-46 (5) default; WP2.1 uses 42-51. Report mean ± sd; paired tests across seeds.
- "Good" = difference > seed noise (learned sd 0.002-0.006) and pre-registered criterion met.

## Success criteria
- Every confirmatory test is pre-registered (paper/preregistration_wp{1,2,4}.md) before runs.
- WP2.3: MAE<=1pp, >=7/9 in PI, beat constant predictor (+2.864pp), Spearman>0.
- WP2.2: W6-W9, slope invariance ±0.10 (data/rho_family_germany50/predictions.json).
