# E023 — exp-pricing on SNDlib + germany50 family (C2' scope)
- Avenue A036 · claims C2', C3' · D21 (freeze lift) · follows K16, K17, K22. CPU only (gpu14), training-free. No new code expected.
## Hypotheses (written before runs)
- H1 (C2' scope): ap_cheapest acc > ac_dqn_widest (acwidest) acc on ≥7/9 SNDlib nets and ≥6/8 g50 levels
  (paired over seeds 42-46, one-sided p<.05 each), as on op/Wx/wp21 (K16).
- H2 (exploratory, C3'): F_AP = acc(ap_cheapest) − acc(ap_shortest) is predicted by Δ_SP out-of-family:
  fit on wp21 (E016) → SNDlib MAE ≤ 1pp, Spearman(Δ_SP, F_AP) > 0 at p<.05. Not confirmatory (Δ_SP seen; A035 is the prereg test).
## Change / config
- Same protocol as E016: (mu,sigma) calibrated once on seed 42 per substrate (D11; A016: no per-seed calibration),
  eval = seed+100 for seeds 42-46 (matched to existing ac_dqn_widest rows), 200 eps, nominal traffic, hard capacity, count metric. Arms: ap_cheapest, ap_shortest.
- Substrates: 9 sndlib_*, 8 g50_r*. Rows into results/all_results.csv via collect_results.py (new batch tag e023).
## Success / falsification
- H1 falsified if pricing ≤ acwidest on ≥3/9 SNDlib (or ≥3/8 g50) → C2' restricted to "in-family + op/Wx" wording.
- Check: hard capacity never violated; calibrated parameter not at grid edge (else extend grid, record).
## Status 2026-10-18 — IMPLEMENTED, not deployed
- Code: 17 substrates added to SUBSTRATES in scripts/run_exp_pricing.py (sndlib_<net>→configs/wp23_<net>.yaml, g50_r<l>→configs/g50_r<l>.yaml; seeds 42-46).
  collect_results.py tags those rows batch=e023 (others stay wp4). Smoke (1 ep, abilene+g50_r1p0) OK; pytest 106 pass.
- Launch (gpu14): `python3 scripts/run_exp_pricing.py --procs 4 --only $(sndlib_* g50_r* names)` → results/exp_pricing/<sub>.csv. Then fetch, collect_results.py, analyse H1/H2.
