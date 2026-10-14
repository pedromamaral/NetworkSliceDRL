# E016 — WP4.1 exponential-pricing comparator
- Avenue A014 · claim C6 · prereg paper/preregistration_wp4.md · sha 68b7df1
- Code: src/baselines/exp_pricing.py, scripts/run_exp_pricing.py (multiprocessing, --procs 4).
- Arms: ap_cheapest, ap_shortest; (mu,sigma) calibrated once on seed 42 (D11).
- Substrates: operator, waxman, rho levels r0p4..r3p0.
- Predictions: P1 pricing routing gain grows with rho; P3 acwidest ≥ ap_cheapest (see prereg).
- Status: COMPLETE gpu14 (EXP_PRICING_DONE). NOT pulled/analysed.
- Outputs: gpu14:~/netslice-drl/results/exp_pricing/<substrate>.csv + _calibration.csv; run.log
- Raw log means (ap_cheapest/ap_shortest): operator .5006/.4434, waxman .5021/.5040, r0p4 .5075/.5077,
  r0p8 .5113/.5083, r1p0 .5065/.4969, r1p3 .5143/.4901, r1p6 .5028/.4589, r2p0 .5070/.4482,
  r2p5 .4906/.4234, r3p0 .4974/.4245 → above acwidest (op .4839, wx .4935): P3 likely FAILS (K16).
- Next: pull, register ap_* in collect_results.py, run P1-P3 as registered. Then think: implication for C1/C6.
- Result: TODO
