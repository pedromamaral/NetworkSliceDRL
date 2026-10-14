# E014 — WP2.3 out-of-sample rho criterion on real SNDlib cores
- Avenue A012 · claim C2 · prereg paper/preregistration_wp2.md (+amend 1,2) · sha af20d0d
- Nets (9): abilene, atlanta, geant, janos_us, newyork, nobel_eu, nobel_germany, nobel_us, polska
  (configs/wp23_<net>.yaml, data/sndlib_topo/). Capacity rule (a) load-proportional (D07).
- Arms: aconly, acwidest (90 trainings) + greedy (45). Seeds 42-46.
- Measured rho 6.2-10.3% (K15). Predictions: data/sndlib_topo/predictions.json (scripts/wp23_predictions.py).
- Criterion (amend 1): MAE ≤ 1pp, ≥7/9 nets in PI, beat constant predictor (+2.864pp), Spearman > 0.
- Status: COMPLETE gpu15 2026-10-14 05:24 (135/135 rc=0; 7 re-run after power cut). NOT pulled/analysed.
- Outputs: gpu15:~/netslice-drl/results/baselines_wp23_<net>_s<seed>__<arm>/metrics.csv
- Next: rsync → register wp23 arms in collect_results.py → analysis script vs predictions.json.
- Result: TODO
