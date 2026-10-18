# STATE — 2026-10-18 · main @8dee9b9 (+ .mem commit below)
Phase just done: implement+analyze E019 (K22) → strategy re-plan (D19 PIVOT, D20 freeze lifted for A031/A035).

## Changed this session
- scripts/analyze_e019.py + paper/e019_analysis_output.txt (8dee9b9); card E019 results/verdict; LEDGER E019
- conclusions K22 new, K14 annotated · avenues A028 tried; A029/A031/A035 needed-for-paper; A032/A033 parked
- strategy.md = digest of lit-graph/drl-slicing-path-flex/strategy-plan.md (2026-10-18)
- project.md claims C1'-C4' (D19) · decisions D19 (pivot), D20 (freeze lift for A031/A035 only)

## Active runs
- none

## Latest results
- E019: F ≈ .84-.88·Δ_SP across wp21/g50/SNDlib; SNDlib MAE .37pp (rho 1.74), Spearman .97 (rho .08); exploratory (K22).
- E016: exp-pricing beats acwidest +1.67pp op (K16) → C2' honest negative.

## Open problem
C3' out-of-family predictor Δ_SP is exploratory and semi-tautological → needs pre-registered test on new
nets (A035) and ideally an analytic Erlang version (A029, kill-gated). BR-EFP baseline missing (A031).

## EXACT next step
1. research-think ideate (strongest model): cards E020 = A029 kill test (EFP-predicted vs measured Δ_SP,
   26 substrates, CPU; kill if Spearman<.7 or MAE>1pp), E021 = A035 pilot (2 new ITZ nets, greedy/
   greedy_shortest/pricing, no training; kill if Δ_SP range <1pp) + A035 prereg draft, E022 = A031 BR-EFP (timebox 2 wk).
2. Then research-do implement E020/E021.

## Files next session must read
STATE.md, strategy.md, project.md, conclusions.md (K16-K22), decisions.md D19-D20, avenues A029/A031/A035 lines.
## Housekeeping
Untracked from earlier sessions, not yet committed: scripts/analyze_wp22.py, diag_rho_plateau.py, run_e018.sh,
paper/wp22_analysis_output.txt, data/diag_rho/, lit-graph/ (repo copy — ask Pedro before adding).
