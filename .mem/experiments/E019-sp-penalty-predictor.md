# E019 — SP-penalty predictor & F decomposition (exploratory, retrospective)
- Avenue A028 · claims C2, C3 · D18 · follows K21, K18, K20. Data already seen → NOT confirmatory. 0 GPU.
- Data: results/all_results.csv, nominal, valid=1; agents ac_dqn_widest, aconly, greedy, greedy_shortest; substrates
  rho_r* (8), g50_r* (8), sndlib_* (9), waxman. rho from rho_wp2 registry (D15) / E014 per-net rho.
- Script (to implement): scripts/analyze_e019.py → paper/e019_analysis_output.txt. Per-substrate means over seeds.
## Hypotheses (written before analysis)
- H1 (I1): Δ_SP = acc(greedy) − acc(greedy_shortest) is linear in F within each family: quadratic term on Δ_SP not significant
  (p>.05) while quadratic on rho is; within-family LOO MAE(Δ_SP) < LOO MAE(rho).
- H2 (I1, transfer): fit on wp21 family, predict 9 SNDlib nets: MAE < 1.74pp (K18) and Spearman(Δ_SP, F) > .08 and > 0 at p<.05.
- H3 (I2): with G = acwidest−greedy and H = greedy−aconly, convexity of F(rho) sits in H: quadratic coef of H(rho) significant,
  G(rho) linear (K14 replicated on g50).
- H4 (I5, secondary): F ~ rho + Δ_SP; report partial R² of each; flag overfit (8 pts/family).
## Success criterion: H1 and H2 both hold → propose Δ_SP for a pre-registered confirmatory test on new substrates (A023/new nets).
## Falsified: H1 fails both families OR H2 MAE ≥ 1.74pp.
## Implementation
- scripts/analyze_e019.py → paper/e019_analysis_output.txt (0 GPU, runs <5 s). No config; reads all_results.csv (batch wp1) + family.json + sndlib networks.json.
- H1 coded literally: quad-term p on F~rho vs F~Dsp (level means, n=8), + LOO MAE of F~rho vs F~Dsp; pass = Dsp quad p>.05 AND rho quad p<=.05 AND MAE_Dsp<MAE_rho. Interpretation choice → confirm in analyze.
- H2 primary fit on wp21 (g50 fit reported as extra).
## Results: TODO (script output done; interpretation pending → research-think analyze E019)
