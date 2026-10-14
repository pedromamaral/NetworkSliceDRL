# STATE — 2026-10-14 · main @dce7bce (.mem edits + new analysis script uncommitted)
Phase just done: research-think analyze E014 + E016 vs pre-registrations.

## Changed this session
- NEW scripts/analyze_wp23_wp41.py → paper/wp23_wp41_analysis_output.txt (registered tests, Holm)
- Cards E014/E016 results+verdicts; LEDGER; A012/A014 → tried; K16 rewritten, K17-K19 new; D16 ACCEPTED; project.md claim flags

## Active runs (runs.md)
- E015 WP2.2 germany50 | gpu15 | running, ETA ~2026-10-15 09:00

## Latest results
- E014 WP2.3 FAIL: MAE 1.74pp, 0/9 in PI, constant not beaten, Spearman +.08; bias -1.74pp (K18).
- E016: P1/P2 supported (pricing F_AP slope .481, rho is substrate property, K17);
  P3 REVERSED: ap_cheapest beats acwidest +1.67 op / +0.85 Wx / all family levels (K16).

## Open problem
C1 and C6 refuted as stated; C2 holds only in-family (K19). Paper story must be re-planned (D16 accepted: no new runs except E015 until strategy re-plan).

## EXACT next step
1. E015: after ~10-15 09:00 `bash tools/runs_status.sh`, fetch, analyse vs W6-W9 (now decisive for C2 slope invariance).
2. research-think strategy (or ideate): re-plan claims (candidates: rho as substrate statistic across controllers; pricing-vs-DQN
   decomposition ap_widest; honest negatives) — no new runs before that.

## Files next session must read
STATE.md, conclusions.md (K16-K19), decisions.md D16, project.md, paper/wp23_wp41_analysis_output.txt.
