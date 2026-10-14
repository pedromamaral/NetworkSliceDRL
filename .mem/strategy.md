# Strategy (updated 2026-10-14)

- Target venue: IEEE TNSM. Plan: paper/plano_publicacao_TNSM.md (Pedro's, untracked).
  Assessment: paper/publishability_assessment_rho_law.md (untracked — don't git add without asking).
- Lit-graph topic folder: TODO(ask) — lit-graph has no topic folders yet
  (~/Library/CloudStorage/OneDrive-FCTNOVA/Docencia/Research/lit-graph).
- Work packages: WP0 tidy CSV · WP1 factorial (done) · WP2 rho criterion (2.1 done, 2.2 running,
  2.3 to analyse) · WP3 theory · WP4 literature comparators (4.1 exp-pricing, 4.2 masked PPO) · WP5 multi-homing.

## Novelty claim (post-Gate 1)
Not "DRL beats baselines" — rather: (i) the value of learning lies in admission (foresight), not
routing; widest routing suffices; (ii) a measurable topology/load statistic rho predicts how much
path flexibility is worth, with a pre-registered predictive test on synthetic + real (SNDlib) cores.

## Gaps
- G1 DRL slicing papers rarely isolate AC from RA contribution → factorial WP1.
- G2 No predictive criterion for when routing flexibility pays → rho (WP2).
- G3 Few compare to classical online admission (trunk reservation, exp/AAP pricing) → H3, WP4.1.
- G4 Reproducibility (seeding, pre-registration) weak in the area → prereg + byte-identical runs.

## Strongest baselines (numbers in conclusions.md)
- greedy widest (never rejects feasible) · trunk reservation (calibrated → theta~0) ·
  exp-pricing (WP4.1, raw ~0.50 acceptance, above acwidest) · aggregate-state DQN.
- Literature numbers: TODO(research-think with lit-graph).
