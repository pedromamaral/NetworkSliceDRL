# E015 — WP2.2 slope invariance on a second core (germany50 rho family)
- Avenue A013 · claim C2 · prereg paper/preregistration_wp2.md amend 3 · sha 3c69310
- 8 levels r0p4..r3p0 (configs/g50_r<lvl>.yaml, data/rho_family_germany50/), rho 0.01-14.49%, greedy 0.460-0.479.
- Arms: aconly, acwidest (80 trainings) + greedy (40).
- Predictions W6-W9: data/rho_family_germany50/predictions.json; slope invariance ±0.10 vs 0.371.
- Status: finished; fetched 2026-10-16 (160 registry rows). Analysis: scripts/analyze_wp22.py → paper/wp22_analysis_output.txt
- Outputs: gpu15:~/netslice-drl/results/baselines_g50_r<lvl>_s<seed>__<arm>/metrics.csv; log results/wp1_logs/driver_wp22.log
- Sub-studies (operating points 0.35/0.60, K=6) deferred (A024, D10).
## Results (F = acwidest − aconly, pp, n=5; seed sd of F 0.2-0.7pp)
| ratio | rho% | F | pred | in PI | G=acw−greedy |
|---|---|---|---|---|---|
| 0.4 | 0.01 | −0.13±0.73 | +0.14 | Y | +2.55 |
| 0.8 | 3.29 | +1.27 | +1.35 | Y | +1.33 |
| 1.0 | 6.40 | +2.67 | +2.50 | Y | +1.20 |
| 1.3 | 10.26 | +3.94 | +3.93 | Y | +0.94 |
| 1.6 | 12.78 | +5.41 | +4.87 | n | +1.01 |
| 2.0 | 13.83 | +6.55 | +5.26 | n | +0.96 |
| 2.5 | 14.45 | +7.26 | +5.49 | n | +1.11 |
| 3.0 | 14.49 | +7.29 | +5.50 | n | +1.26 |
- W6 slope 0.499 (CI95 .467-.530) → SUPPORTED (Holm p<.0125).
- W7 F(0.4) −0.13, CI90 [−0.83,+0.56], TOST p=.16 → INCONCLUSIVE (n=5, sd .73: underpowered; point est. ≈0).
- W8 slope diff +0.128, CI90 [+.099,+.157] vs ±0.10 → REFUTED (equivalence fails; CI also excludes 0 → germany50 steeper).
- W9 MAE 0.74 (≤1.0 ✓), beats constant 2.61 ✓, but 4/8 in PI (need 6) → FAIL. Exact for rho≤10.3 (4/4, err ≤0.17pp);
  under-predicts by 0.5-1.8pp on the 4 saturated levels (rho 12.8-14.5) where F keeps rising (+1.9pp) while rho moves 1.7pts.
## Interpretation
- rho orders F strongly on a second core (monotone, slope CI tight) but slope is core-specific (.37 vs .50) → no universal law.
- rho plateau ~14.5% checked 2026-10-17 (subagent + family.json): not a definitional ceiling (rho = P(any k feas) − P(SP feas),
  generate_rho_family.py:172, bound ≤ 0.47). ratio scales access links only; capacity_scale identical at 2.5/3.0 in both
  families → access slack, core-limited value under K=3. F also plateaus (7.26→7.29). Real misfit = convex F(rho) over
  ratio 1.3-2.5. Suspect: rho measured under greedy states, F under learned agents. Exploratory: slope on rho≤11 = 0.401.
- Cheap check (gpu14, CPU, not run): g.measure at ratio-3.0 topo with (a) ratio 100, (b) K=6, (c) acwidest-checkpoint states;
  log P(SP feas) and P(any feas). (a) flat → core-limited; (b) jump → K-set; (c) larger and growing with ratio → explains convexity.
- G (foresight) ≈ flat +1.0-1.3pp for rho≥6 (WP2.1 fit predicts 0.4-1.3) — admission gain over greedy persists on germany50.
## Verdict: mixed — W6 supported; W7 inconclusive; W8 refuted; W9 fail (Holm: only W6). C2 = in-family ordinal, not invariant (K20).
