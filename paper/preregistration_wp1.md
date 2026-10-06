# Pre-registration — WP1 factorial decomposition

**Registered:** 2026-10-06, before any WP1 run. The commit that adds this file is the
registration timestamp; analysis code may change afterwards, the hypotheses, decision rules
and stopping rule below may not. Any deviation is reported in the paper as a deviation.

**Purpose.** Decompose the joint agent's advantage into (i) foresight in the *admission*
decision and (ii) flexibility in the *routing* decision, and decide which architecture the
paper recommends (Gate 1 of `plano_publicacao_TNSM.md`).

---

## 1. Design

Admission × routing grid. Cells marked *new* are run in WP1; the others exist.

| | shortest path | widest path | learned routing |
|---|---|---|---|
| greedy admission | `greedy_shortest` (new) | `greedy` (exists) | — |
| learned admission | `aconly` (exists) | **`ac_dqn_widest` (new)** | `joint` unified (exists), `factored` (exists) |
| trunk reservation | `tr_shortest` (new) | `tr_widest` (new) | — |
| trunk reservation, DAR | `tr_dar` (new): direct path unreserved, overflow to alternates subject to θ | | |

Substrates and seeds:

| Substrate | Seeds | Why |
|---|---|---|
| Operator (ρ = 11.0 %) | 42–46 | pairs with existing 5-seed joint / AC-only / greedy |
| Waxman (ρ = 0.0 %) | 42–46 | pairs with existing 5-seed joint / factored / AC-only |
| ρ family, all 5 levels | 42–43 | pairs with existing 2-seed sweep; extended to 10 seeds in WP2 **only if** Gate 1 selects the agent to sweep |

Everything else is frozen at `configs/base.yaml` as of this commit: 2000 training episodes,
evaluation on held-out seed+100 for 200 episodes with ε = 0, same network, loss, clipping and
schedules for every learned policy.

Trunk-reservation threshold θ is calibrated on the **training** seed by grid search over
θ ∈ {0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30} (40 episodes per value), then frozen for
held-out evaluation. θ = 0 reproduces greedy (verified: 0 action mismatches in 3,000 steps).

## 2. Hypotheses

**H1 (primary) — routing flexibility, not routing learning, is what pays.**
On the ρ family at the two highest levels (ρ = 12.91 %, 14.42 %),
`ac_dqn_widest` ≥ `joint` (unified).
*Rationale, recorded before seeing data:* across the family, joint − greedy falls
+4.2 → +1.9 → +0.7 → −0.1 → 0.0 % as ρ rises; the joint agent's routing errors (4.6 % of
arrivals on the operator substrate) appear to cancel its admission advantage exactly where
routing matters. `ac_dqn_widest` keeps the admission network and cannot make a routing error.

**H2 — non-inferiority everywhere.** On every substrate and level,
`ac_dqn_widest` ≥ `joint` − 0.5 percentage points (absolute acceptance).

**H3 — learned admission is not trunk reservation.** On the operator and Waxman substrates,
`ac_dqn_widest` > best of {`tr_widest`, `tr_dar`} with calibrated θ.
(A uniform one-parameter reservation rule does not reproduce the learned admission policy.)

**H4 — decomposition (descriptive, no test).** With routing fixed to widest,
admission foresight = `ac_dqn_widest` − `greedy`; with admission learned,
routing flexibility = `ac_dqn_widest` − `aconly`. Reported per substrate and per ρ level.
Expected: flexibility grows with ρ; foresight roughly constant.

## 3. Analysis

- Unit of analysis: training seed. All policies within a seed see identical held-out traffic,
  so comparisons are **paired** across seeds.
- Test: two-sided paired t-test, α = 0.05. Report mean difference, 95 % CI (t-based), t, df.
- Multiple comparisons: Holm–Bonferroni across the primary family {H1 at ρ=12.91 %,
  H1 at ρ=14.42 %, H3 operator, H3 Waxman}. H2 and H4 are reported without correction.
- With 2 seeds per ρ level (df = 1) the per-level tests have almost no power. H1 is therefore
  also evaluated pooled over the two highest levels (4 paired differences, df = 3). If both
  the per-level and pooled versions are inconclusive, H1 is reported as **inconclusive**,
  not as supported — and the WP2 10-seed extension is what resolves it.
- No result is excluded. Any crashed run is re-run with the same seed; if it crashes twice it
  is reported as missing.

## 4. Decision rule (Gate 1)

| Outcome | Paper's recommendation | WP2 sweeps |
|---|---|---|
| H1 supported, H2 holds | "Learn admission; route widest-path. ρ decides whether path flexibility is needed — a heuristic suffices." Factored-agent recommendation in v3 is withdrawn. | `ac_dqn_widest` |
| H1 rejected (joint > ac_dqn_widest at high ρ) | Learned routing has value of its own at high ρ; keep the factored agent. | `joint`/`factored` |
| Inconclusive | Report both; WP2 runs both arms at 10 seeds. | both |

H3 failing (trunk reservation matches learned admission) does **not** change the
architecture recommendation, but it changes the paper's claim C2 ("what is worth learning
is admission") to "what is worth having is reservation; learning reproduces it", and that
sentence will be written whichever way it falls.

## 5. Stopping rule

WP1 ends when every cell above has its registered seeds. No additional seeds are added to
WP1 in response to its results; extra seeds belong to WP2 and are registered there.

---

## Amendment 1 — 2026-10-06, before any WP1 run

**What was found.** The replay buffer samples minibatches with Python's stdlib `random`,
which no entry point seeded (`experiments/run_experiment.py`, `experiments/eval_baselines.py`
seeded only numpy and torch). Two trainings with the same seed on the same machine diverged
by episode 10, on CPU as well as GPU. Every learned-policy result produced before this date
(joint, factored, AC-only, aggregate-state) is therefore a valid random draw but cannot be
reproduced from its seed. This is the likely cause of the discrepancy between the original
operator joint runs (seeds 42–44: 0.4815 / 0.4743 / 0.4831) and their July-28 re-run
(0.4771 / 0.4705 / 0.4678). Code drift between the two batches cannot be excluded.

**Fix.** stdlib `random` is now seeded with the run seed, and `eval_baselines.py` re-seeds
before every trained baseline, so a baseline's result no longer depends on which other
baselines ran before it in the same process. Verified: two same-seed trainings on gpu15 (GPU,
run concurrently) produce byte-identical logs, and so do two on CPU under different
`PYTHONHASHSEED` values.

**Change to §1 (design).** "Exists" cells are no longer reused. The old runs and the new
cells come from different, non-reproducible batches, so pairing them would confound the
comparison with batch effects. Every learned arm of the grid is **re-run** with the fix, on
the same image as the new cells, under run tag `wp1`:

| Substrate | Learned arms re-run (new seeding) | Seeds |
|---|---|---|
| Operator | `joint` (unified), `factored`, `aconly`, + new `ac_dqn_widest` | 42–46 |
| Waxman | `joint` (unified), `factored`, `aconly`, + new `ac_dqn_widest` | 42–46 |
| ρ family, 5 levels | `joint` (unified), `aconly`, + new `ac_dqn_widest` | 42–43 |

The deterministic policies (`greedy`, `greedy_shortest`, `tr_*`) are evaluated in the same
batch for completeness; they do not depend on the seeding fix.

**Change to §3 (analysis).** All tests of H1–H4 use WP1-batch results only. Pre-amendment
results are reported separately as a replication check (original vs. re-run, per cell),
never pooled with the WP1 batch.

**Unchanged.** Hypotheses, rationale, thresholds, tests, multiplicity correction, decision
rule and stopping rule (§2, §4, §5) are unchanged.

**Consequence beyond WP1 (recorded now so it cannot be decided after seeing data).** The
headline numbers in `NetworkSlicing_v3.tex` that come from learned policies are re-estimated
from the WP1 batch, and the paper reports those, whichever direction they move. The original
batch appears only in the replication check.
