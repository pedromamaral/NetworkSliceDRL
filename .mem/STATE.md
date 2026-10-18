# STATE — 2026-10-18 · main @d1abf51
Phase just done: ideate 2026-10-18 → D21 (order E020+E023 → E021 → E022; freeze lifted for A036). Cards E020-E023 written.

## Changed this session (ideate, committed d1abf51)
- avenues: A029 active (E020); A036 NEW active (E023); A035→E021, A031→E022 planned · decisions D21
- cards: E020 (full), E023 (full), E021/E022 skeletons
- previous session: E019/K22, D19 pivot, D20 (commit f37f0fa)

## Active runs
- none

## Latest results
- E019: F ≈ .84-.88·Δ_SP across wp21/g50/SNDlib; SNDlib MAE .37pp (rho 1.74), Spearman .97 (rho .08); exploratory (K22).
- E016: exp-pricing beats acwidest +1.67pp op (K16) → C2' honest negative.

## Open problem
C3' out-of-family predictor Δ_SP is exploratory and semi-tautological → needs pre-registered test on new
nets (A035) and ideally an analytic Erlang version (A029, kill-gated). BR-EFP baseline missing (A031).

## EXACT next step
1. research-do implement E023 (A036: reuse E016 pricing runner on 9 SNDlib + 8 g50, gpu14 CPU) and E020 (A029: new
   scripts/efp_predict.py, kill gate Spearman≥.7 & MAE≤1pp on 26 substrates) — in parallel (D21, freeze lifted for A036).
2. Then E021 (A035 pilot + preregistration_wp6.md), then E022 (A031 BR-EFP, timebox 2 wk). Cards exist as skeletons.

## Files next session must read
STATE.md, experiments/E020-efp-kill-test.md, E023-pricing-sndlib-g50.md, E016 card (runner), decisions D21.
## Housekeeping
Untracked from earlier sessions, not yet committed: scripts/analyze_wp22.py, diag_rho_plateau.py, run_e018.sh,
paper/wp22_analysis_output.txt, data/diag_rho/, lit-graph/ (repo copy — ask Pedro before adding).
