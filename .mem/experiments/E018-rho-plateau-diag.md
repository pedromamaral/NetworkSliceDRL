# E018 — rho plateau / convexity diagnostic (exploratory)
- Avenue A027 · claim C2 · D17 · follows K20. NOT confirmatory; predictions below written before runs.
- Script: scripts/diag_rho_plateau.py. Estimator = D15 (eval seeds 142-146 × 100 eps), logs P(SP feas), P(any feas), rho, acc.
- Levels: ratio 1.3 / 2.0 / 3.0, both families (WP2.1 rho_family, germany50).
## Parts
- ab (gpu14 CPU): variants base (greedy, K3; must reproduce rho_wp2), acc100 (access caps ×100/ratio → 600·100, same scale), k6 (greedy, K=6).
- c (gpu15, 12 trainings, 2000 eps, seed 42 = E015/WP2.1 agents reproduced): rho measured on acwidest's and aconly's own state distributions.
## Predictions
- P-a: acc100 rho at ratio 3.0 within ±0.5pt of base → plateau is core-limited.
- P-b: k6 rho > base by ≥1pt → the 14.5% level is set by K=3 (else core structure).
- P-c: rho(acwidest states) − rho(greedy states) > 0 and grows from ratio 1.3 → 3.0 → explains convex F(rho).
  Falsified if the gap is ≤0.5pt or flat across ratios.
- Sanity: base reproduces rho_wp2 exactly (same code/seeds); part-c eval acceptance on seed 142 close to E015 metrics.
## Results (2026-10-17; compact numbers .mem/results/E018.json; rho in pt, mean of 5 eval seeds, eval sd 0.1-0.36)
Sanity ✓: base = rho_wp2 exactly (all 6 cells); part-c repro_acc = E015/WP2.1 seed-42 acceptance to 4 d.p. (12/12).
| fam r | base | acc100 | k6 | ρ(acwidest st) | ρ(aconly st) | acc acw / aconly | F s42 |
|---|---|---|---|---|---|---|---|
| wp21 1.3 | 5.36 | 17.07 | 5.45 | 5.38 (+.02) | 8.99 (+3.6) | .486/.465 | 2.06 |
| wp21 2.0 | 13.02 | 15.29 | 17.46 | 13.20 (+.18) | 17.66 (+4.6) | .487/.444 | 4.33 |
| wp21 3.0 | 14.41 | 14.47 | 19.35 | 14.93 (+.52) | 19.65 (+5.2) | .484/.426 | 5.77 |
| g50 1.3 | 10.26 | 15.47 | 13.45 | 10.64 (+.38) | 14.56 (+4.3) | .480/.438 | 4.25 |
| g50 2.0 | 13.83 | 14.24 | 18.55 | 14.30 (+.47) | 19.53 (+5.7) | .467/.401 | 6.54 |
| g50 3.0 | 14.49 | 14.50 | 19.55 | 14.67 (+.18) | 19.95 (+5.5) | .486/.410 | 7.53 |
## Interpretation
- P-a SUPPORTED: acc100 at r3.0 = base ±0.06pt both families → plateau is not access; at r1.3 access binds hard (wp21 5.4→17.1).
  So the ratio knob = access tightness; it saturates once access is slack (≈r2.0 g50, ≈r3.0 wp21).
- P-b SUPPORTED (r≥2.0): K=6 raises rho +3.2…+5.1pt (except wp21 r1.3, access-bound: +0.1) → plateau level = K=3 core path diversity.
  k6 still flattens r2.0→3.0 (+1.0/+1.9) → bound = core path diversity under K, shifted up by K.
- P-c REFUTED: rho(acwidest) − rho(greedy) ≤0.52pt everywhere, not growing (g50 +.38/+.47/+.18). Greedy-state rho is a
  faithful proxy for the flexible agent's own states → convexity is NOT a state-distribution artefact. Caveat: 1 train seed.
- New (unpredicted): acwidest acceptance is flat (.467-.487) across ratios; F's growth is entirely aconly's decline
  (.465→.426 wp21, .438→.410 g50). SP-only routing self-congests: aconly states carry +3.6…+5.7pt more rho.
  F/rho rises with ratio under either rho (greedy .38→.40 / .41→.52; aconly-state .23→.29 / .29→.38) → convexity unexplained,
  lives in the SP-only baseline's degradation, which rho (a one-step feasibility statistic) does not capture.
## Verdict: P-a supported · P-b supported · P-c refuted · sanity passed. Exploratory (D17), n=1 train seed for part c.
## Launch (done 2026-10-17; script already on both hosts, code hashes = laptop)
- gpu14 (CPU, 6 procs, ~10-15 min): docker run --rm --name e018-ab, entrypoint python, mounts results/configs/data/src/experiments/scripts,
  `scripts/diag_rho_plateau.py ab --procs 6` → results/diag_rho/ab.json, ab.log
- gpu15 (12 containers, ~2.3 h): results/diag_rho/run_c.sh (6 configs g50_r{1p3,2p0,3p0}, ddqn_unified_rho_r{...} × acwidest/aconly, seed 42)
  → results/diag_rho/c_*.json; completion marker c_done.txt. Sanity: repro_acc must equal E015/WP2.1 seed-42 acceptance.
- Smoke 2026-10-17 (2 eps, g50 r3.0 s142): base rho .154, acc100 .154 (identical), k6 .213.
