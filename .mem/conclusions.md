# Conclusions (batch "wp1" = post-seeding-fix unless noted). Source: .mem/_import.md + paper/wp{1,2}_analysis_output.txt

## Reference numbers (acceptance_ratio, mean over seeds)
REF | operator K3 acwidest | 0.4839 | E011
REF | operator K3 joint | 0.4826 | E011
REF | operator K3 factored | 0.4742 | E011
REF | operator K3 greedy | 0.4637 | E011
REF | operator K3 aconly | 0.4469 | E011
REF | operator K3 greedy_shortest | 0.4193 | E011
REF | operator K3 aggregate-state | 0.4195 | E012
REF | waxman aconly/acwidest/factored/joint/greedy | 0.4945/0.4935/0.4928/0.4830/0.4683 | E011
REF | greedy at calibrated operating point (all families) | ~0.47 | E013
REF | seed sd learned | 0.002-0.006 | E011
REF | seed sd greedy | 0.001-0.003 | E011
REF | real SNDlib rho (load-prop. capacities) | 6.2-10.3% | E014 prep
REF | germany50 family rho / greedy | 0.01-14.49% / 0.460-0.479 | E015 prep

## Conclusions
K01 | revenue/soft objective beats greedy only by admitting fewer; per-step SLA penalty floods | E001 | high | 2026-07-07
K02 | action masking removes agent's selectivity edge; masked joint wp1 0.4381 (orig 0.4602 was lucky) | E007,E012 | high | 2026-10-08
K03 | pre-2026-10-06 learned runs non-reproducible (unseeded stdlib random); now byte-identical per seed | E011 | high | 2026-10-06
K04 | wp1 replicates original batch within 0.7pp every cell; July-28 gpu14 rerun (0.4718) was outlier | E011 | high | 2026-10-08
K05 | separated(factored) ≈ unified at K3 (n.s.) | E003,E011 | med | 2026-07-27
K06 | learned AC keeps more arrivals feasible: 54.9% vs greedy 46.2% (op); 53.3 vs 46.7 (Waxman) | E017 | high | 2026-10-02
K07 | K scaling: joint degrades K3→K8 (0.4826/0.4770/0.4659, -1.67pp p=0.007); acwidest gains (0.4839/0.4893/0.4883); acwidest-joint +2.24pp at K8 | E012 | high | 2026-10-08
K08 | heavy load (cap 0.5): joint 0.3004 ≈ greedy 0.3014 n.s.; acwidest 0.3071 (+0.58pp p=0.001); aconly 0.2863 | E012 | high | 2026-10-08
K09 | aggregate-state baseline reproduces exactly (op 0.4195, Wx 0.4705); aconly beats it +2.74pp | E012 | high | 2026-10-08
K10 | Waxman: sign reversal vs operator (aconly ≥ joint), significant at n=5 | E004 | high | 2026-10-02
K11 | GATE 1: acwidest ≥ joint everywhere; learned routing costs 0.1-1.9pp; H1 Holm-sig at rho=14.42% (+1.07pp), pooled p=0.0096 | E011 | high | 2026-10-08
K12 | H3: calibrated trunk reservation picks theta~0 (=greedy); acwidest beats it +2.0pp op / +2.5pp Wx | E011 | high | 2026-10-08
K13 | WP2.1: F = 0.371*rho + 0.13pp, slope CI [0.355,0.386]; F(rho=0) = -0.04pp (TOST p=0.0005); new-level MAE 0.14pp, 3/3 in PI | E013 | high | 2026-10-12
K14 | WP2.1: G = acc(acwidest)-acc(greedy) = -0.114*rho + 2.05 (fresh seeds); rho*~0 → foresight/flexibility trade-off | E013 | high | 2026-10-12
K15 | real backbones sit mid-range of rho (6.2-10.3%) | E014 prep | high | 2026-10-09
K16 | (raw, unanalysed) exp-pricing acceptance ~0.50 > acwidest → WP4.1 P3 likely fails; pricing routing gain grows with rho (P1 likely holds) | E016 | low | 2026-10-14
