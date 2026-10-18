# E020 — Erlang reduced-load (EFP) predictor of Δ_SP: kill test
- Avenue A029 · claim C3' · D19, D21 · follows K22. CPU only (gpu14), ≤3 days. Kill gate from strategy-plan #1.
## Hypothesis
- H1: an Erlang fixed-point model of blocking under SP-only vs K=3 widest routing, fed only with topology, capacities and the
  traffic matrix (no simulation), gives Δ_SP_EFP that matches measured Δ_SP = acc(greedy) − acc(greedy_shortest).
## Change
- New scripts/efp_predict.py: per-link offered load → link blocking (Erlang B, multi-rate Kaufman-Roberts for slice classes
  if bandwidth classes differ) → fixed point → path/route acceptance under (a) shortest only, (b) first-feasible-of-K (greedy).
  Output Δ_SP_EFP per substrate. No changes to env/MDP.
## Config
- 26 substrates: rho_r* (8), g50_r* (8), sndlib_* (9), operator (+waxman if traffic spec same). Measured Δ_SP from all_results.csv.
## Success criterion (kill gate, fixed now)
- Pass: Spearman(Δ_SP_EFP, Δ_SP) ≥ .7 AND MAE ≤ 1pp over 26 substrates. Report also per-family and level bias.
- Kill: either fails → drop theory part of C3', keep Δ_SP empirical only (A029 → tried).
## Notes
- Exploratory (Δ_SP data already seen); model choices (routing approximation, multi-rate) frozen in the script header before
  the comparison is run; no tuning against measured Δ_SP.
