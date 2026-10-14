# Codebase map (updated 2026-10-14). Full spec of env/agents: CLAUDE.md §2-4 (long — read by section).

## Env & agents (src/)
- src/env/network_env.py   NetworkEnv (gymnasium); reward/admission modes; state order fixed
- src/env/topology.py      NetworkTopology: K-shortest paths, avail, bottleneck tensor B
- src/env/slice_generator.py  Poisson arrivals
- src/agents/{dqn,ddqn}_{unified,separated}.py  joint (unified) / factored (separated, D_rt buffer)
- src/agents/replay_buffer.py  (stdlib random now seeded — f9bf817)
- src/baselines/: greedy_admission (widest/shortest), admission_only_dqn (aconly, k=0),
  ac_dqn_widest (acwidest), trunk_reservation (tr_shortest/widest/dar), aggregate_state_dqn,
  revenue_heuristic, exp_pricing (WP4.1 ap_cheapest/ap_shortest)
- src/utils/{metrics,logger,checkpoint}.py

## Entry points
- experiments/run_experiment.py --config C --seed S --tag wp1  (Docker ENTRYPOINT; learned joint/factored)
  → results/<run_name>[_<tag>]_s<S>/{metrics.csv (per eval_interval), eval_final.csv (metric,value)}
- experiments/eval_baselines.py --config C --seed S --baselines a,b --train_episodes 2000 (ALWAYS pass 2000)
  → results/baselines_<cfg>_s<S>__<subset>/metrics.csv (one row per baseline)
- experiments/diagnose.py (feasibility/mechanism), experiments/aggregate_sweep.py
- scripts/run_exp_pricing.py (WP4.1) → results/exp_pricing/<substrate>.csv

## Config system
- configs/*.yaml inherit via `_base_: configs/base.yaml`. Families: ddqn_*_count* (operator/Waxman/K/load/mask),
  ddqn_unified_rho_r<lvl> (rho family, data/rho_family/), g50_r<lvl> (germany50, data/rho_family_germany50/),
  wp23_<net> (SNDlib, data/sndlib_topo/). capacity_scale calibrated so greedy ≈ 0.47.
- Revenue/soft configs (ddqn_unified, *_stress*, revenue_soft) = abandoned model, excluded.

## Batch & analysis pipeline
- scripts/run_wp1.sh: job-set functions jobs/jobs_loose/jobs_wp2/jobs_wp23/jobs_wp22 (JOBSET=…); P parallel
  containers; skips jobs with rc=0 in results/wp1_logs/done.txt. On server run it as run_wp1_next.sh copy.
- scripts/queue_wp2*.sh: wait for previous driver, refuse duplicates, start next.
- scripts/collect_results.py → results/all_results.csv (explicit registry; register new arms here, e.g. ap_*).
- scripts/analyze_wp1.py, analyze_wp2.py → paper/wp{1,2}_analysis_output.txt (pre-registered tests).
- scripts/measure_rho.py (rho), wp2_predictions.py, wp23_predictions.py (predictions.json),
  generate_rho_family.py (--extend), build_sndlib.py, greedy_load_envelope.py.
- paper/make_figures.py reads results/all_results.csv → paper/figures/*.pdf (still on old batch → switch to wp1).
- tools/summarize_run.py, tools/runs_status.sh (memory tooling).

## How to add an experiment
1. config in configs/ (inherit base; calibrate capacity_scale) 2. pre-register in paper/preregistration_*.md
3. add job-set fn to scripts/run_wp1.sh 4. register arm in collect_results.py 5. LEDGER line + card.

## Tests
python -m pytest tests/ -v  (test_env, test_agents, test_baselines, test_topology)
