# STATE — 2026-10-16 · main @5b4e022 (.mem edits + 2 new analysis scripts uncommitted)
Phase just done: analyze E018 (K21) + ideate (D18: E019 active A028, A029 proposed, A030 dropped).

## Changed this session
- NEW scripts/analyze_wp22.py → paper/wp22_analysis_output.txt (registered W6-W9, Holm); results/all_results.csv refreshed (160 g50 rows)
- Card E015 results+verdict; LEDGER; A013 → tried; K19 revised; K20 new; REF germany50 slope added

## Active runs
- none running (E018 ab+c both done and fetched 2026-10-17)

## Latest results
- E015: W6 ✓ slope .499; W7 inconclusive (n=5); W8 ✗ slope ≠ .371 (Δ+.128); W9 FAIL 4/8 in PI (MAE .74, exact for rho≤10.3).
- rho plateau ~14.5% explained (K20): access-link slack, not estimator ceiling; F plateaus too. Open: convex F(rho) over ratio 1.3-2.5.
- E014 WP2.3 FAIL (K18); E016 pricing beats acwidest (K16), pricing F slope .481 (K17).

## Open problem
C1/C6 refuted as stated; C2 = in-family ordinal predictor only (K19). Paper story must be re-planned (D16).

## EXACT next step
0. DONE: E019 implemented (scripts/analyze_e019.py, output in paper/e019_analysis_output.txt) → next: research-think analyze E019.
1. (done) research-do implement E019 (card experiments/E019-sp-penalty-predictor.md, D18): scripts/analyze_e019.py on all_results.csv
   (0 GPU, laptop OK) → paper/e019_analysis_output.txt; then research-think analyze E019.
2. Then research-think strategy: re-plan claim set using K16-K21 + E019 verdict (C2 scope: rho vs Δ_SP).
3. A029 (Erlang reduced-load model) only if E019 H1/H2 hold or strategy asks for C5 theory.

## Files next session must read
STATE.md, conclusions.md (K16-K20), decisions.md D16, project.md, paper/wp22_analysis_output.txt, paper/wp23_wp41_analysis_output.txt.
