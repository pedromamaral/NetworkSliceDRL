# Import from Claude Code session 42f2c721 (2026-10-01 .. 2026-10-14). Terse; one fact per line.

## Model / setup (not obvious from code)
- Active model: count reward + hard capacity (admission_mode hard, reward_mode count); revenue/soft kept only as ablation.
- Target venue IEEE TNSM; plan in paper/plano_publicacao_TNSM.md (user's, untracked); assessment in paper/publishability_assessment_rho_law.md (untracked; don't add without asking).
- Paper draft: paper/NetworkSlicing_v3.tex (figures via paper/make_figures.py, all values read from results/all_results.csv). v3 story is now OUTDATED (see conclusions).
- rho (routing headroom) = P[some candidate path feasible AND shortest infeasible], measured with greedy; WP2 estimator = scripts/measure_rho.py (5 eval seeds 142-146 x 100 eps). v3/WP1 used 20-ep --quick values (max diff 0.28pp).

## Conclusions (all post-seeding-fix batch "wp1" unless noted)
- Seeding bug: replay buffer used unseeded stdlib random -> no learned run before 2026-10-06 reproducible; fixed f9bf817; same-seed runs now byte-identical (CPU, GPU, concurrent).
- Replication of original batch: every cell within 0.7pp (operator joint orig 0.4806 vs wp1 0.4826). July-28 gpu14 rerun (0.4718) was the outlier batch.
- WP1 Gate 1 = "learn admission, route widest": ac_dqn_widest >= joint on every substrate; joint-acwidest (learned routing) -0.1..-1.9pp everywhere. H1 Holm-sig at rho=14.42% (+1.07pp), pooled p=0.0096; H3: calibrated trunk reservation picks theta~0 (=greedy), acwidest beats it +2.0pp operator / +2.5pp Waxman. Output paper/wp1_analysis_output.txt.
- Operator K3 wp1 means: acwidest 0.4839, joint 0.4826, factored 0.4742, greedy 0.4637, aconly 0.4469, greedy_shortest 0.4193, aggregate 0.4195.
- Waxman wp1: aconly 0.4945, acwidest 0.4935, factored 0.4928, joint 0.4830, greedy 0.4683.
- WP2.1 (rho family, 8 levels x 10 seeds): F = acc(acwidest)-acc(aconly) = 0.371*rho + 0.13 pp, slope CI [0.355,0.386]; F at rho=0 = -0.04pp (TOST p=0.0005); new-level predictions MAE 0.14pp, 3/3 in PI; foresight G = acc(acwidest)-acc(greedy) = -0.114*rho + 2.05 (fresh seeds); rho* ~ 0. Gate 2 (WP2.1 part): "predictive criterion". Output paper/wp2_analysis_output.txt.
- Level F means (rho%: F pp): 0.00:-0.04, 0.74:+0.60, 1.82:+0.89, 5.36:+2.07, 9.55:+3.49, 13.02:+4.94, 14.07:+5.28, 14.41:+5.67.
- K scaling (operator, wp1): joint 0.4826/0.4770/0.4659 (K3/6/8; K8-K3 -1.67pp p=0.007); acwidest 0.4839/0.4893/0.4883 (+0.44pp p=0.04); greedy 0.4637/0.4756/0.4764; acwidest-joint at K8 +2.24pp.
- Heavy load (capacity 0.5): joint 0.3004 ~ greedy 0.3014 (n.s.); acwidest 0.3071 (+0.58pp vs greedy, p=0.001); aconly 0.2863.
- Masked joint: wp1 0.4381 vs orig 0.4602 (orig sd 0.021, lucky batch) -> v3 masking text must be revised.
- Aggregate-state baseline reproduces exactly (operator 0.4195, Waxman 0.4705); aconly beats it +2.74pp.
- Feasibility mechanism (orig ckpts, mech_diag logs): learned keeps 54.9% arrivals feasible vs greedy 46.2% (operator); 53.3 vs 46.7 (Waxman).
- Real networks (SNDlib, load-proportional capacities): rho only 6.2-10.3% (nobel-eu 6.20 ... newyork 10.29); real backbones sit mid-range.
- germany50 family: rho 0.01..14.49% over ratios 0.4..3.0; greedy 0.460-0.479.

## Ideas tried / rejected
- Per-step SLA penalty (revenue model): floods, rejected earlier (pre-session).
- Revenue objective: beats greedy only by admitting fewer; abandoned for count (pre-session).
- Joint/factored as recommended architecture: withdrawn after Gate 1.
- Reusing pre-fix runs as WP1 comparison arms: rejected (batch confound) -> prereg WP1 amendment 1, all learned arms re-run.
- Per-seed (mu,sigma) calibration for pricing: rejected (~35 core-h); calibrate once on training seed 42.
- Abilene with real capacities in WP2.3: dropped (would need ad-hoc access-capacity rule).
- Capacity rules for real nets: (b) uniform -> rho~0 everywhere (weak); (c) SNDlib demand -> mixes traffic models; chose (a) load-proportional.
- Regenerating rho family via build_topology: unsafe (hash() jitter is PYTHONHASHSEED-dependent) -> use --extend (reuses saved core) or build_topology_det (hashlib).

## Decisions
- User delegated choices ("choose what you think is best"): WP2.1 thresholds as drafted; WP2.3 capacity rule (a); WP2.2 core germany50.
- WP2.3 criterion tightened (amendment 1) because narrow rho range: must beat constant predictor (+2.864pp) and Spearman>0.
- WP2.2 sub-studies (greedy 0.35/0.60 operating points, K=6) deferred; registered separately or dropped and reported.
- Term "criterion" (not "law") unless WP2.3 also passes.
- User: runs on gpu15 stay there; gpu14 only for small non-GPU tasks; leave user's MADDPG/others' jobs alone; laptop must not run long jobs.
- Pre-registrations: paper/preregistration_wp1.md (+amend 1), paper/preregistration_wp2.md (+amend 1,2,3), paper/preregistration_wp4.md.

## Running / queued (verified 2026-10-14 13:58)
- gpu15 (10.26.110.15): WP2.3 COMPLETE 2026-10-14 05:24 (135/135 jobs rc=0; incl. 7 re-run after 2026-10-13 power cut). Results on server: results/baselines_wp23_<net>_s<seed>__<arm>/metrics.csv; NOT yet pulled/analysed.
- gpu15: WP2.2 RUNNING since 2026-10-14 05:28: started by scripts/queue_wp22.sh (sid 7530) -> `JOBSET=jobs_wp22 bash scripts/run_wp1_next.sh` (P=7, commit 3c69310), log results/wp1_logs/driver_wp22.log; 21/80 trainings done (avg 2.25h), 7 containers (auto names); then 40 greedy jobs. ETA ~09:00 2026-10-15. Outputs results/baselines_g50_r<lvl>_s<seed>__<arm>/metrics.csv. Disk 11G free.
- gpu14 (10.26.110.14): WP4.1 COMPLETE (EXP_PRICING_DONE); outputs results/exp_pricing/<substrate>.csv + _calibration.csv, log run.log; NOT yet pulled/analysed. Raw log means (ap_cheapest/ap_shortest): operator 0.5006/0.4434, waxman 0.5021/0.5040, rho levels r0p4 0.5075/0.5077, r0p8 0.5113/0.5083, r1p0 0.5065/0.4969, r1p3 0.5143/0.4901, r1p6 0.5028/0.4589, r2p0 0.5070/0.4482, r2p5 0.4906/0.4234, r3p0 0.4974/0.4245. NB pricing acceptance (~0.50) is ABOVE ac_dqn_widest (operator 0.4839, waxman 0.4935) -> P3 likely fails; pricing routing gain grows with rho (P1 likely holds).
- Completion tracking: results/wp1_logs/done.txt ("<job> rc=N secs"); driver skips jobs with rc=0 (restart-safe).

## Baselines / noise / timings
- Greedy ~0.47 at calibrated operating point everywhere (family calibrated to 0.47).
- Seed-to-seed sd of acceptance ~0.002-0.006 (learned), ~0.001-0.003 (greedy); runs deterministic per seed post-fix.
- Run times at P=7 on gpu15: joint/aconly/acwidest ~2.3-3h, factored ~4.5h, K8 factored longer; WP2.3 2.1-3.1h. gpu15 = gpu14 = i7-9800X (8C/16T).
- Benchmark: 1 run 88s(30 eps), 4 concurrent 122s, 6 concurrent 130s; pinning irrelevant.

## Ops gotchas
- eval_baselines --train_episodes default 500; always pass 2000.
- run_experiment --tag wp1 for re-runs (else overwrites); baseline out dirs include __<subset>.
- Never overwrite a script bash is executing (use run_wp1_next.sh copy).
- Remote launch: `ssh -f -n ... setsid nohup ... &`; a "hung" launch may still have started -> check `ps -eo sid,cmd` before retrying (duplicate WP2 driver ran 44h, 2026-10-09..11).
- Killing docker run client doesn't stop container (python PID1 ignores SIGTERM): use docker stop.
- Container-written files are root-owned: delete via `docker run --rm -v ... alpine rm`.
- gpu15 disk ~10-12G free (99%); checkpoints final-only (checkpoint_every_eval false).
- Power cut 2026-10-13 ~11:05 (reboot 12:35): learned baselines can't resume; consider mid-training checkpoints before PPO.

## Open problems / next steps
- On WP2.3 completion: analysis vs registered predictions (data/sndlib_topo/predictions.json): MAE<=1pp, >=7/9 in PI, beat constant, Spearman>0.
- On WP2.2 completion: W6-W9 vs data/rho_family_germany50/predictions.json (slope invariance +-0.10).
- On WP4.1 completion: P1-P3 (pricing routing gain vs rho; acwidest vs ap_cheapest); register ap_* in collect_results.py.
- WP3: lemma (single-homed + access-limited => rho=0) and proposition (why slope ~0.37); not started.
- WP4.2 masked PPO (subset), WP5 multi-homing: pending; add resumable training first.
- Paper restructure around: learned admission + widest routing; rho predicts value of path flexibility; foresight/flexibility trade-off. Update v3 numbers from wp1 batch; switch make_figures.py to batch wp1; rewrite masking text.
- WP2.2 sub-studies (operating points, K=6) not yet calibrated/registered.
