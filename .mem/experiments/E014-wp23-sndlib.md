# E014 — WP2.3 out-of-sample rho criterion on real SNDlib cores
- Avenue A012 · claim C2 · prereg paper/preregistration_wp2.md (+amend 1,2) · sha af20d0d
- Nets (9): abilene, atlanta, geant, janos_us, newyork, nobel_eu, nobel_germany, nobel_us, polska
  (configs/wp23_<net>.yaml, data/sndlib_topo/). Capacity rule (a) load-proportional (D07).
- Arms: aconly, acwidest (90 trainings) + greedy (45). Seeds 42-46.
- Measured rho 6.2-10.3% (K15). Predictions: data/sndlib_topo/predictions.json (scripts/wp23_predictions.py).
- Criterion (amend 1): MAE ≤ 1pp, ≥7/9 nets in PI, beat constant predictor (+2.864pp), Spearman > 0.
- Status: COMPLETE gpu15 2026-10-14 05:24 (135/135 rc=0; 7 re-run after power cut). Analysed 2026-10-14.
- Outputs: gpu15:~/netslice-drl/results/baselines_wp23_<net>_s<seed>__<arm>/metrics.csv
- Next: rsync → register wp23 arms in collect_results.py → analysis script vs predictions.json.
- Analysis: scripts/analyze_wp23_wp41.py → paper/wp23_wp41_analysis_output.txt (n=5 paired, all cells complete).
- Result: observed F (pp) nobel-eu +1.48, nobel-germany +1.63, abilene -0.16, janos-us +1.74, geant +1.75,
  atlanta +1.04, nobel-us +0.69, polska +0.46, newyork +3.50 (seed sd 0.1-0.7). Every net BELOW prediction.
  MAE 1.74 pp (>1.0) · 0/9 in PI · constant-predictor MAE 1.66 (not beaten) · Spearman +0.08, perm p 0.42.
  Mean bias -1.74 pp. G = acwidest-greedy ranges -0.00 (newyork) .. +2.64 (abilene), not monotone in rho.
  Exploratory (NOT registered): non-learned routing gain greedy-greedy_shortest also unranked by rho (Spearman +0.08).
- Interpretation: within the narrow real range (6.2-10.3%) rho_wp2 neither levels nor ranks F. Failure is not
  DQN-specific (greedy routing gain unranked too). Synthetic-family calibration (intercept/slope) does not
  transfer to real cores; between-network heterogeneity (~1 pp) >> rho-explained spread (1.5 pp predicted).
  Anomaly to flag: abilene aconly ≥ acwidest (F<0) while G=+2.6 — learned admission substitutes for routing.
- Verdict: WP2.3 FAILS all four registered conditions → C2 cannot be claimed out-of-sample; "criterion" stays
  in-family only (K13, K19). Report as pre-registered negative.
