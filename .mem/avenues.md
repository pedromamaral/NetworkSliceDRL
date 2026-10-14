# AVENUE REGISTRY — every idea ever considered (anti-repeat). Grep before proposing.
# A### | status | idea | claim | experiments | conclusion | date
A001 | dropped | revenue reward + soft capacity (once-penalty, gamma .99) | C1 | E001 | beats greedy on revenue only by admitting fewer (K01) | 2026-07-07
A002 | dropped | per-step proportional SLA penalty | C1 | E001 | floods (184k vs greedy 219k) | 2026-07-04
A003 | tried | count reward + hard capacity (current model) | C1,C4 | E002,E017 | learned beats greedy via foresight | 2026-07-22
A004 | dropped | feasibility action masking for unified agents | C1 | E007,E012 | removes selectivity; masked joint 0.4381 | 2026-10-08
A005 | tried | factored/separated architecture (two nets + D_rt) | C1 | E003,E009,E011 | ≈ unified at K3, worse than acwidest | 2026-10-08
A006 | tried | K scaling (K6/K8) | C1 | E005,E012 | joint degrades with K, acwidest gains (K07) | 2026-10-08
A007 | tried | second topology (Waxman core) | C1,C2 | E004,E011 | sign reversal, consistent with rho | 2026-10-02
A008 | tried | controlled rho family (capacity-ratio levels) | C2 | E006,E013 | F linear in rho (K13) | 2026-10-12
A009 | tried | aggregate-state DQN baseline | C4 | E008,E012 | 0.4195, aconly +2.74pp | 2026-10-08
A010 | tried | ac_dqn_widest (learn AC, route widest) | C1 | E011 | best arm everywhere (K11) | 2026-10-08
A011 | tried | trunk reservation (shortest/widest/DAR), calibrated theta | C1 | E011 | theta~0 = greedy (K12) | 2026-10-08
A012 | active | real SNDlib cores, load-proportional capacity rule (a) | C2 | E014 | analyse vs predictions | 2026-10-14
A013 | active | germany50 rho family (slope invariance across cores) | C2 | E015 | running | 2026-10-14
A014 | active | exponential/AAP-style pricing comparator | C6 | E016 | raw: pricing ≥ acwidest | 2026-10-14
A015 | dropped | reuse pre-fix runs as WP1 comparison arms | — | — | batch confound; all learned arms re-run | 2026-10-06
A016 | dropped | per-seed (mu,sigma) calibration for pricing | C6 | — | ~35 core-h; calibrate once on seed 42 | 2026-10-13
A017 | dropped | Abilene with real capacities in WP2.3 | C2 | — | needs ad-hoc access-capacity rule | 2026-10-09
A018 | dropped | uniform capacities for real nets (rule b) | C2 | — | rho~0 everywhere (weak) | 2026-10-09
A019 | dropped | SNDlib demand-based capacities (rule c) | C2 | — | mixes traffic models | 2026-10-09
A020 | dropped | joint/factored as recommended architecture | C1 | E011 | withdrawn after Gate 1 | 2026-10-08
A021 | proposed | WP3 lemma (single-homed+access-limited ⇒ rho=0) + slope proposition | C5 | — | not started | 2026-10-14
A022 | proposed | WP4.2 masked PPO comparator (subset) | C1 | — | needs resumable training first | 2026-10-14
A023 | proposed | WP5 multi-homing | C2,C5 | — | pending | 2026-10-14
A024 | parked | WP2.2 sub-studies: greedy 0.35/0.60 operating points, K=6 | C2 | — | not calibrated/registered | 2026-10-13
A025 | proposed | mid-training checkpoints / resumable training (power-cut lesson) | infra | — | before PPO | 2026-10-13
A026 | tried | slice_duration / arrival tuning to calibrate greedy operating point | — | — | greedy ~0.47 everywhere | 2026-06-21
