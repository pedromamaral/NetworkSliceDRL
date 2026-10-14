# LEDGER — one line per experiment: E | date | sha | change | key params | result | verdict | avenue
# Older experiments reconstructed from git + .mem/_import.md on 2026-10-14 (pre-E011 not reproducible, K03).
E001 | 2026-07-03 | 9f0e55c | revenue/soft-capacity campaign (incl. onceG99, per-step penalty) | gamma .99, pw | net reward 222,239 vs greedy 219,365, but fewer admits | superseded (K01) | A001,A002
E002 | 2026-07-22 | fa54ad6 | count/hard model, joint DDQN seed 42 | K3 | joint 0.4786 > greedy 0.4611 > aconly 0.4461 | positive (pre-fix) | A003
E003 | 2026-07-27 | 238d164 | separated (factored) DDQN, 5 seeds | K3 | 0.4739±0.0049 vs unified 0.4806±0.0042, n.s. | equal (K05) | A005
E004 | 2026-10-02 | 48acf4b | topology-2 (Waxman) comparison n=5 | nominal | aconly > joint on Waxman, significant | sign reversal (K10) | A007
E005 | 2026-07-30 | ae5729c | K scaling K6/K8 (orig batch) | K6,K8 | superseded by E012 | superseded | A006
E006 | 2026-10-01 | 718b439 | controlled rho-sweep family (orig batch) | 5 levels | motivated rho law (v3) | superseded by E011/E013 | A008
E007 | 2026-07-02 | 0862327 | feasibility action masking | mask on | removed selectivity edge | negative (K02) | A004
E008 | 2026-10-01 | 3f85d55 | aggregate-state DQN baseline | — | see E012 | reproduced | A009
E009 | 2026-10-02 | c4f68aa | factored arch at rho=0 penalty test | — | see E011 | superseded | A005
E011 | 2026-10-08 | 42061a6 | WP1 factorial, 90 jobs (joint/factored/aconly/acwidest/heur/TR) | 5 seeds, op+Wx+5 rho lvls | acwidest ≥ joint everywhere | Gate 1 (K11,K12) | A010,A011
E012 | 2026-10-08 | c59cb11 | loose ends re-run under wp1 seeding (K6/K8, load, mask, aggregate) | 5 seeds | K07,K08,K02,K09 | done | A006,A004,A009
E013 | 2026-10-12 | 369189c | WP2.1 rho family 8 lvls × 10 seeds, aconly vs acwidest + greedy | seeds 42-51 | F=0.371ρ+0.13pp; W1-W4 supported | positive (K13,K14) | A008
E014 | 2026-10-14 | af20d0d | WP2.3 SNDlib 9 real nets, 90 trainings + 45 greedy | load-prop caps (rule a) | COMPLETE on gpu15, not pulled/analysed | pending | A012
E015 | 2026-10-14 | 3c69310 | WP2.2 germany50 rho family, 80 trainings + 40 greedy | 8 lvls | RUNNING gpu15 | pending | A013
E016 | 2026-10-14 | 68b7df1 | WP4.1 exp-pricing comparator (ap_cheapest/ap_shortest) | calibrated on seed 42 | COMPLETE gpu14, raw ~0.50, not analysed | pending (K16) | A014
E017 | 2026-10-02 | — | feasibility-mechanism diagnosis (diagnose.py, orig ckpts) | — | 54.9% vs 46.2% feasible | positive (K06) | A003
