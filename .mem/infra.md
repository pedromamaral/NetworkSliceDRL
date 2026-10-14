# Infra (updated 2026-10-14). No secrets here — ssh keys in ~/.ssh (id_ed25519), WANDB_API_KEY in server env.

## Hosts
| name | ssh alias | IP | CPU/GPU | role |
| gpu15 | GPUserver1 | 10.26.110.15 (pedroamaral) | i7-9800X 8C/16T + GPU | ALL training batches (P=7) |
| gpu14 | GPUserver2 | 10.26.110.14 (pedroamaral) | i7-9800X 8C/16T + GPU | small non-GPU tasks only (e.g. WP4.1) |
- Repo on both: ~/netslice-drl (git pull; commit sha recorded in .wp1_commit). Image netslice-drl:latest.
- Others' jobs (MADDPG etc.) also run there — never touch them. Laptop: no long jobs.

## Launch (batch driver)
ssh -f -n GPUserver1 'cd ~/netslice-drl && setsid nohup env JOBSET=jobs_X P=7 bash scripts/run_wp1_next.sh > results/wp1_logs/driver_X.log 2>&1 &'
- BEFORE retrying a "hung" launch: `ps -eo sid,cmd | grep run_wp1` (duplicate driver ran 44h, 2026-10-09..11).
- Never overwrite a script bash is executing → copy run_wp1.sh to run_wp1_next.sh before launch.
- Build synchronously first: `docker build -q -t netslice-drl:latest . && echo BUILD_OK` (src/configs/data are mounted anyway).
- Single run: docker run --gpus all --rm -e OMP_NUM_THREADS=1 -v $(pwd)/{results,configs,data,src,experiments}:/workspace/... netslice-drl:latest --config C --seed S --tag wp1
- Baselines: same with `--entrypoint python ... experiments/eval_baselines.py --train_episodes 2000 --eval_episodes 200`.

## Monitor
- tools/runs_status.sh (one ssh per host). Manual: docker ps; wc -l results/wp1_logs/done.txt; tail -n 2 driver_*.log.

## Fetch
rsync -av --exclude='checkpoints/' GPUserver1:~/netslice-drl/results/ results/remote/gpu15/   (same for gpu14 → remote/gpu14)
then python3 scripts/collect_results.py → results/all_results.csv.

## Timings
- P=7 on gpu15: joint/aconly/acwidest 2.3-3h; factored ~4.5h; WP2.3 2.1-3.1h; WP2.2 ~2.25h avg.
- 1 run 88s/30 eps; 6 concurrent 130s; CPU pinning irrelevant.

## Gotchas
- Killing `docker run` client doesn't stop the container (python PID1 ignores SIGTERM) → `docker stop`.
- Container-written files root-owned → delete via `docker run --rm -v …:/x alpine rm …`.
- gpu15 disk ~11G free (99%): checkpoints final-only; `docker builder prune -f` after rebuilds.
- Checkpoints shared per run_name (results/checkpoints/<agent>_s<seed>/) — pass --checkpoint explicitly.
- --tag wp1 for re-runs (else overwrites). eval_baselines default --train_episodes 500 → always 2000.
- Learned baselines can't resume after crash (power cut 2026-10-13); driver re-runs rc≠0 jobs.
