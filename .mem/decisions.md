# Decisions — D# | decision | why | alternatives rejected | date
D01 | count reward + hard capacity as default model | paper claims are about slice count; revenue never pays to use routing headroom | revenue/soft (kept as ablation) | 2026-07-07
D02 | action_mask off by default | masking removed selectivity edge | mask on | 2026-07-02
D03 | seed stdlib random; re-run ALL learned arms under tag wp1 (prereg WP1 amend 1) | pre-fix runs non-reproducible | reuse old runs (batch confound) | 2026-10-06
D04 | pre-register every confirmatory WP before any run (paper/preregistration_wp{1,2,4}.md + amendments) | credibility for TNSM | post-hoc analysis | 2026-10-06
D05 | all paper numbers from results/all_results.csv via scripts/collect_results.py (explicit registry) | no hand-typed numbers | ad-hoc tables | 2026-10-06
D06 | recommend "learn admission, route widest" (acwidest) | Gate 1 (K11) | joint/factored | 2026-10-08
D07 | WP2.1 thresholds as drafted; WP2.3 capacity rule (a) load-proportional; WP2.2 core = germany50 | delegated by Pedro | rules b/c (A018,A019) | 2026-10-09
D08 | WP2.3 criterion tightened (amend 1): beat constant predictor (+2.864pp) and Spearman>0 | narrow rho range of real nets | MAE-only | 2026-10-09
D09 | term "criterion", not "law", unless WP2.3 passes | honesty of claim | "law" | 2026-10-12
D10 | WP2.2 sub-studies deferred; registered separately or dropped and reported | scope | include now | 2026-10-13
D11 | exp-pricing calibrated once on training seed 42 | cost | per-seed calibration | 2026-10-13
D12 | rho family regeneration only via --extend or build_topology_det (hashlib) | hash() jitter depends on PYTHONHASHSEED | build_topology | 2026-10-11
D13 | runs on gpu15 stay there; gpu14 only small non-GPU tasks; never touch others' jobs (MADDPG etc.); no long jobs on laptop | Pedro's rule | — | 2026-10
D14 | checkpoints final-only (checkpoint_every_eval false) | gpu15 disk ~11G free | per-eval ckpts | 2026-10
D15 | rho estimator for WP2 = scripts/measure_rho.py, 5 eval seeds 142-146 × 100 eps | precision | 20-ep --quick (v3/WP1; max diff 0.28pp) | 2026-10-08
D16 | (ACCEPTED by Pedro 2026-10-14) C1 and C6 cannot be claimed as stated (K16); C2 limited to in-family (K18,K19). Freeze new runs except E015; re-plan claim set in research-think strategy | pre-registered P3 reversed + WP2.3 failed | keep C1 with caveat (rejected: pricing wins on every substrate) | 2026-10-14
D17 | (Pedro 2026-10-17) exception to D16 run freeze: E018 rho-plateau diagnostic (exploratory, not confirmatory) before strategy re-plan | K20 convex F(rho) misfit; cheap | go straight to strategy | 2026-10-17
D18 | (Pedro 2026-10-17) ideate on SP-baseline loss: E019 = exploratory retrospective analysis (I1 Δ_SP predictor + I5 bivariate, I2 F=G+H decomposition) on all_results.csv, 0 GPU, compatible with D16 freeze; A029 Erlang model proposed; A030 dropped (K21). Confirmatory use of Δ_SP needs prereg on new substrates | modelling aconly loss directly is circular (acwidest flat); greedy_shortest is training-free | model F on rho alone; rho on SP-only states | 2026-10-17
D19 | (Pedro via research-strategist 2026-10-18, digested in strategy.md) PIVOT: claim set C1'-C4' replaces C1-C6 — measurement paper: learning value is in admission; classical online control (pricing, BR-EFP) is the bar; F predictable training-free (rho in-family, Δ_SP out-of-family, Erlang optional). Paper core = A031 + A035; A029 gated by kill test; A032/A033 parked. D16 run freeze lift for A031/A035 still PENDING Pedro | K16,K18,K19,K22 + lit-graph drl-slicing-path-flex | keep 'DRL beats all' (refuted K16); rho law (refuted K18/K20) | 2026-10-18
D20 | (Pedro 2026-10-18) LIFT D16 run freeze for A031 (BR-EFP) and A035 (Δ_SP confirmation, ~80 runs gpu15) only; other avenues stay frozen; A035 confirmatory runs only after prereg | D19 plan core needs runs | keep freeze | 2026-10-18
