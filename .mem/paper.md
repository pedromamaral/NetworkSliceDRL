# Paper (updated 2026-10-14). Venue IEEE TNSM.
- Draft: paper/NetworkSlicing_v3.tex (story OUTDATED: built on joint-DRL + "routing-headroom law", old batch).
- Figures: paper/make_figures.py ← results/all_results.csv → paper/figures/fig_*.pdf (switch to batch wp1).
- Plan: paper/plano_publicacao_TNSM.md. Prereg: paper/preregistration_wp{1,2,4}.md.

## New outline (to write): learned admission + widest routing; rho predicts value of flexibility; foresight/flexibility trade-off.

| claim | figure/table | experiments | status |
| C1 acwidest best; learned routing hurts | fig_main, table WP1 factorial, fig_scaling (K) | E011,E012 | data ready; text to rewrite |
| C1 heavy load | fig_load | E012 + greedy_load_envelope.csv | data ready |
| C2 F vs rho (synthetic) | fig_rho | E013 | data ready |
| C2 slope invariance (germany50) | fig_rho (2nd core) | E015 | running |
| C2 real nets out-of-sample | table/fig WP2.3 | E014 | to analyse |
| C3 foresight G vs rho | fig_rho / new | E013 | data ready |
| C4 feasibility mechanism | fig_feasibility | E017 (orig ckpts) | old batch, OK as mechanism |
| C5 lemma/proposition | §theory | WP3 | not started |
| C6 vs exp-pricing | table WP4.1 | E016 | to analyse |
| Waxman sign reversal | fig_topo2 | E004,E011 | data ready |
| masking | text | E012 (0.4381) | v3 masking text must be revised |
| learning curves | fig_learning | wp1 metrics.csv | to regenerate |
