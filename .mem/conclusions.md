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
REF | exp-pricing ap_cheapest op/Wx/family | 0.5006/0.5021/0.491-0.514 | E016
REF | germany50 family rho / greedy | 0.01-14.49% / 0.460-0.479 | E015 prep
REF | germany50 F slope / seed sd of F | 0.499 (CI .467-.530) / 0.2-0.7pp | E015

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
K14 | (see K22: G curved at high ratio, rho-plateau) WP2.1: G = acc(acwidest)-acc(greedy) = -0.114*rho + 2.05 (fresh seeds); rho*~0 → foresight/flexibility trade-off | E013 | high | 2026-10-12
K15 | real backbones sit mid-range of rho (6.2-10.3%) | E014 prep | high | 2026-10-09
K16 | WP4.1 P3 REVERSED: exp-pricing ap_cheapest beats acwidest +1.67pp op (p=.0009), +0.85pp Wx (p=.0087), and at all 8 family levels (0.9-2.4pp); hard capacity respected, no calibration overfit; gain is from admission threshold (sigma=inf ≈ greedy) | E016 | high | 2026-10-14
K17 | WP4.1 P1/P2: rho predicts routing gain of non-learned pricing, F_AP = 0.481*rho - 0.05pp (CI .476-.486), 0 at rho=0 → rho is a substrate property in-family, not a learner artefact | E016 | high | 2026-10-14
K18 | WP2.3 FAIL: on 9 SNDlib nets F below prediction everywhere (bias -1.74pp), MAE 1.74, 0/9 in PI, constant predictor not beaten, Spearman +.08 (p=.42); greedy routing gain also unranked by rho (exploratory) → rho does not transfer out-of-family in the 6-10% range | E014 | high | 2026-10-14
K19 | C2 scope after WP2.3+WP2.2: rho→F is monotone and strongly linear within a controlled family on both cores (DQN slope .371 WP2.1, .499 germany50; pricing .481) but slope/intercept are family/core-specific (W8 refuted) and do not transfer out-of-family (K18) | E013,E014,E015,E016 | high | 2026-10-16
K20 | WP2.2 germany50: W6 ✓ slope .499; W7 inconcl. (F(0.4)=−0.13, n=5 underpowered); W8 ✗ Δslope +.128 CI90 [.099,.157]; W9 fail 4/8 in PI though MAE .74 and exact (≤.17pp) for rho≤10.3; rho plateau ~14.5% is NOT an estimator ceiling (bound is ≤ greedy acc 0.47): ratio scales access links only and calibrated capacity_scale is identical at ratio 2.5/3.0 → access slack, network = core under K=3; F plateaus too (7.26→7.29). Real misfit = convexity of F vs rho over ratio 1.3-2.5 (F 3.94→7.26, rho 10.3→14.5); state-distribution cause REFUTED by E018 (K21) | E015 | high | 2026-10-16
K21 | E018: rho plateau = K=3 core path-diversity bound (access×100 changes rho ≤.06pt at r3.0; K=6 lifts it +3-5pt); ratio knob = access tightness, saturating; rho on acwidest's own states = greedy-state rho ±.52pt (estimator valid for the flexible agent); convex F(rho) unexplained by rho — acwidest acc flat ~.48, F growth = aconly decline (SP-only self-congestion, aconly states +3.6-5.7pt rho). n=1 train seed (part c) | E018 | medium | 2026-10-17
K22 | E019 (exploratory, retrospective): F ≈ .84-.88·Dsp (Dsp=acc(greedy)−acc(greedy_shortest)) on wp21, g50 and 9 SNDlib nets — SNDlib MAE .37pp (vs rho 1.74), Spearman .97 (vs .08); cross-family slope Δ .032; pooled partial R² rho|Dsp .001. rho's out-of-family failure is rho→routing-gain, not routing-gain→F. Convexity of F(rho) not significant at n=8 levels; G(rho) curved (g50 p=.003), H linear — partly rho-plateau artefact (K21). Semi-tautological predictor; not pre-registered | E019 | medium | 2026-10-17
