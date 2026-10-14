# E016 — WP4.1 exponential-pricing comparator
- Avenue A014 · claim C6 · prereg paper/preregistration_wp4.md · sha 68b7df1
- Code: src/baselines/exp_pricing.py, scripts/run_exp_pricing.py (multiprocessing, --procs 4).
- Arms: ap_cheapest, ap_shortest; (mu,sigma) calibrated once on seed 42 (D11).
- Substrates: operator, waxman, rho levels r0p4..r3p0.
- Predictions: P1 pricing routing gain grows with rho; P3 acwidest ≥ ap_cheapest (see prereg).
- Status: COMPLETE gpu14 (EXP_PRICING_DONE). Analysed 2026-10-14.
- Outputs: gpu14:~/netslice-drl/results/exp_pricing/<substrate>.csv + _calibration.csv; run.log
- Raw log means (ap_cheapest/ap_shortest): operator .5006/.4434, waxman .5021/.5040, r0p4 .5075/.5077,
  r0p8 .5113/.5083, r1p0 .5065/.4969, r1p3 .5143/.4901, r1p6 .5028/.4589, r2p0 .5070/.4482,
  r2p5 .4906/.4234, r3p0 .4974/.4245 → above acwidest (op .4839, wx .4935): P3 likely FAILS (K16).
- Next: pull, register ap_* in collect_results.py, run P1-P3 as registered. Then think: implication for C1/C6.
- Analysis: scripts/analyze_wp23_wp41.py → paper/wp23_wp41_analysis_output.txt. Seeds matched (family n=10, op/Wx n=5).
- Result (Holm over 4):
  P1 SUPPORTED: F_AP = 0.481·rho - 0.05 pp, slope CI [0.476, 0.486] (DQN F slope 0.371; ×1.30).
  P2 SUPPORTED: F_AP(ratio 0.4) = -0.02 pp, TOST p<1e-4.
  P3-operator REVERSED (sig.): acwidest - ap_cheapest = -1.67±0.42 pp, p=0.0009.
  P3-Waxman  REVERSED (sig.): -0.85±0.40 pp, p=0.0087.
  Descriptive: ap_cheapest > acwidest at all 8 family levels (0.9-2.4 pp); ap_cheapest - greedy +3.7 op / +3.4 Wx.
- Artefact checks: fulfillment 1.0, sla 0 (hard capacity respected); held-out acc ≈ calibration acc (no overfit);
  sigma=inf (pure min-price routing) ≈ greedy 0.47 → the gain is from the admission threshold, not routing alone.
  Not decomposed: ap_cheapest routes on min-price, acwidest on widest — admission vs routing share of the 1.7 pp unknown.
- Verdict: rho predicts routing gain of a non-learned controller (in-family) → rho is a substrate property.
  Learned admission (DQN) is beaten by single-seed-calibrated exponential pricing on every substrate tested → C1, C6 refuted as stated.
