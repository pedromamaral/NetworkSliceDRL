# Strategy (updated 2026-10-18) — digest of strategy-plan.md
Source: ~/Library/CloudStorage/OneDrive-FCTNOVA/Docencia/Research/lit-graph/drl-slicing-path-flex/strategy-plan.md (2026-10-18, rev 0)
Avenue map: plan #1 = A029 Erlang · #2 = A031 BR-EFP · #3 = A032 LP bound · #4 = A033 implied costs · #6 = A035 Δ_SP confirm
Untracked Pedro docs: paper/plano_publicacao_TNSM.md, paper/publishability_assessment_rho_law.md (never git add).

- 2026-10-18 · topic drl-slicing-path-flex · status: new (revision 0) · this is the claim-set re-plan that D16 called for
- Chosen: AVENUE #2 (classical head-to-head incl. BR-EFP) + AVENUE #6 (pre-registered Δ_SP confirmation) = paper core; AVENUE #1 (Erlang reduced-load predictor) = strategic play, in the paper if its kill test passes by week 5
- RQ: When do learning and routing flexibility pay for online slice admission on IP/MPLS cores, and can that value be predicted without training?
- C1' Value of learning is in admission: learn admission, route widest ≥ learned joint routing (K11, K07). Supported.
- C2' Classical online control is the bar: exp-pricing beats DRL admission on every substrate (K16); TR collapses to greedy (K12); BR-EFP to be added (#2). Honest negative result.
- C3' Flexibility value F is predictable training-free: in-family by ρ (K13, K17, K19), out-of-family by Δ_SP (K22, confirmatory test #6), explained by an Erlang reduced-load model (#1, optional).
- C4' Mechanism: learned admission keeps future arrivals feasible (K06); foresight/flexibility trade-off G(ρ) (K14).
- Dropped claims: "DRL beats all baselines" (old C1/C6, K16); "ρ law transfers across cores" (old C2, K18/K20).
- Gaps addressed: G#1 classical baselines absent in DRL AC; G#2 no predictor of when flexibility pays; G#5 no structural explanation of DRL value/cost
- Novelty: first pre-registered DRL-vs-loss-network-control comparison for slice admission + routing, with a training-free predictor of the value of path flexibility that holds out of family
- Venue: IEEE TNSM (no APC). Fallback: Computer Networks.
- Baselines (literature): NDP CAC+routing +2.2% avg reward vs OSPF, 16-node · 10.1109/49.824797
- Baselines (literature): BR-EFP thresholds within 0.5% of exhaustive search, B4 eval · 10.1109/tnet.2023.3249480
- Baselines (literature): MA-PPO AC +29.96% revenue vs greedy, 9-node C-RAN · 10.1109/tnsm.2022.3222589
- Baselines (internal): exp-pricing beats acwidest by +1.67pp (op) and +0.85pp (Wx) acceptance (K16); acwidest beats calibrated TR by +2.0pp/+2.5pp (K12)
- Data: SNDlib (already used, 9 nets; cite Orlowski et al. 2010) https://sndlib.put.poznan.pl
- Data: Internet Topology Zoo, CC BY 4.0, >250 nets: https://doi.org/10.25909/30153949.v1 (also via pip `topohub`, licence not stated → use ITZ originals)
- Tool: HiGHS LP solver (MIT, pip highspy) https://github.com/ERGO-Code/HiGHS (only if AVENUE #3 is opened)
- KILL #1 (Erlang): EFP-predicted Δ_SP vs measured Δ_SP on the 26 existing substrates (CPU, ≤3 days). Spearman <0.7 or MAE >1pp → drop the theory claim, keep Δ_SP empirical only.
- KILL #2 (BR-EFP): implement and run on op/Wx, seed 42 (timebox 1 week). Not working by the end of week 2 → report BR-EFP as "not reproduced" and keep TR + pricing.
- KILL #6 (Δ_SP confirm): pilot on 2 new ITZ nets with greedy/greedy_shortest/pricing (no training). Δ_SP range <1pp across the pilot → choose more diverse nets before preregistering.
- Do not pursue: A001/A002 revenue or soft reward; A004 masking; A005/A020 factored as the recommended design; A015 reuse of pre-fix runs; A016 per-seed pricing calibration; A018/A019 capacity rules b/c; A030 ρ on SP-only states (K21)
- Do not pursue before submission: AVENUE #4 learned implied costs (changes the action space, needs Pedro, 4–6 weeks); AVENUE #3 LP bound (2–3 weeks, state-aggregation risk) → follow-up letter
- Constraints: Pedro alone and part-time; gpu15 runs batches at P=7 (≈2.5 h per learned run); gpu14 small CPU jobs; ~11 GB disk on gpu15; no paid compute; D16 freeze must be lifted explicitly for #2/#6 runs
- Budget: #6 ≈ 8 nets × 2 learned arms × 5 seeds = 80 runs ≈ 30 h on gpu15; #2 and #1 are CPU-only
