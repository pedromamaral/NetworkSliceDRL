# E015 — WP2.2 slope invariance on a second core (germany50 rho family)
- Avenue A013 · claim C2 · prereg paper/preregistration_wp2.md amend 3 · sha 3c69310
- 8 levels r0p4..r3p0 (configs/g50_r<lvl>.yaml, data/rho_family_germany50/), rho 0.01-14.49%, greedy 0.460-0.479.
- Arms: aconly, acwidest (80 trainings) + greedy (40).
- Predictions W6-W9: data/rho_family_germany50/predictions.json; slope invariance ±0.10 vs 0.371.
- Status: RUNNING gpu15 since 2026-10-14 05:28 (scripts/queue_wp22.sh → JOBSET=jobs_wp22, P=7). ETA ~2026-10-15 09:00.
- Outputs: gpu15:~/netslice-drl/results/baselines_g50_r<lvl>_s<seed>__<arm>/metrics.csv; log results/wp1_logs/driver_wp22.log
- Sub-studies (operating points 0.35/0.60, K=6) deferred (A024, D10).
- Result: TODO
