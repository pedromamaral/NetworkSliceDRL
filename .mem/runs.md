# RUN REGISTRY — active on top. Format: id | host | container | cmd | started | ETA | output | status
# Batches run via a driver (scripts/run_wp1.sh as run_wp1_next.sh copy, JOBSET=<fn>, P=7, auto-named containers).
# Completion: <host>:~/netslice-drl/results/wp1_logs/done.txt ("<job> rc=N secs"). Check with tools/runs_status.sh.
E015 | gpu15 | auto ×7 (driver sid 7530, scripts/queue_wp22.sh) | JOBSET=jobs_wp22 run_wp1_next.sh @3c69310 | 10-14 05:28 | ~10-15 09:00 | gpu15:~/netslice-drl/results/baselines_g50_r<lvl>_s<seed>__<arm>/metrics.csv ; log results/wp1_logs/driver_wp22.log | running (80 train + 40 greedy)
E014 | gpu15 | — | JOBSET=jobs_wp23 | 10-12 | done 10-14 05:24 | gpu15:~/netslice-drl/results/baselines_wp23_<net>_s<seed>__<arm>/metrics.csv | finished, NOT pulled
E016 | gpu14 | — | scripts/run_exp_pricing.py | 10-13 | done (EXP_PRICING_DONE) | gpu14:~/netslice-drl/results/exp_pricing/<substrate>.csv + _calibration.csv, run.log | finished, NOT pulled
