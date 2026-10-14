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
