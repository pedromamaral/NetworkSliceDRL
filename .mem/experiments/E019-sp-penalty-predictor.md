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
## Results (paper/e019_analysis_output.txt, 2026-10-17)
| H | key numbers | verdict |
|---|---|---|
| H1 | quad p F~rho .41 (wp21) / .068 (g50) → premise "rho curvature significant" fails at n=8; quad p F~Dsp .41/.13; LOO-MAE F~Dsp .108 vs rho .166 (wp21), .234 vs .464 (g50); Pearson(Dsp,F) .999/.997 | REFUTED as stated (both fams) — but MAE sub-criterion holds in both |
| H2 | wp21 fit F=.844·Dsp+.06 → SNDlib MAE .37pp (bias +.30) vs rho 1.74; g50 fit F=.876·Dsp−.26 → MAE .23; Spearman(Dsp,F) +.97 p<.001 vs rho +.08 | SUPPORTED, strongly |
| H3 | quad coef G: +.013 p=.052 (wp21), +.016 p=.003 (g50); H: p=.23/.95 | REFUTED — curvature sits in G, not H; H ~linear in rho |
| H4 | pooled n=16: partial R² rho|Dsp .001, Dsp|rho .875; within-fam R² ≥.97 for either alone (n=8, overfit flag) | SUPPORTED: Dsp subsumes rho across families |
## Interpretation
- Failure of rho out-of-family (K18) is in the rho→routing-gain link, not routing-gain→F: Dsp ranks SNDlib nets (.97) where rho can't.
- F ≈ .84-.88·Dsp on 3 families; slopes differ by .032 (inside W8's ±.10 margin) where rho slopes differed by .128 → exploratory cross-family invariance.
- Caveat: semi-tautological (greedy's routing gain predicts learned-AC routing gain); Dsp needs simulation, not training. Nets' Dsp [0.4,3.7] is interpolation.
- H3 G-curvature is driven by high-ratio levels where rho saturates (K21) while G rebounds (.50→.64 wp21, .94→1.26 g50): partly a rho-plateau artefact, not clean convexity.
## Verdict: MIXED. Pre-written falsification rule triggered (H1 fails both) → no automatic promotion to confirmatory; H2/H4 strong. Next confirmatory use of Dsp needs Pedro's decision + prereg (D18).
