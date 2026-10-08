# Pre-registration — WP2 validation of the ρ criterion

**Status:** WP2.1 registered on the commit that adds this file, before any WP2.1 run.
WP2.2 and WP2.3 are outlined in §6 and get their own dated registration before their runs.
Analysis code may change afterwards; hypotheses, decision rules and stopping rule may not.

**Context.** Gate 1 (WP1, `wp1_analysis_output.txt`) selected row 1 of the decision rule:
learned admission + widest-path routing (`ac_dqn_widest`) matches or beats the joint agent on
every substrate, so WP2 sweeps `ac_dqn_widest`. The quantity ρ predicts is therefore the
value of **path flexibility** given learned admission:

  F(ρ) = acceptance(`ac_dqn_widest`) − acceptance(`aconly`)   [percentage points]

`aconly` is the same admission learner pinned to the shortest path, so F isolates what
access to the K candidate paths is worth, with the admission policy class held fixed.

---

## 1. Design (WP2.1 — extended controlled family)

Eight levels of the controlled family: the five WP1 levels plus three new ones added on
2026-10-08 with `generate_rho_family.py --extend 1.0 1.6 2.5` (same core graph, only
access capacity changes; verified to reproduce an existing level byte for byte).

| ratio | 0.4 | 0.8 | 1.0 | 1.3 | 1.6 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|---|
| ρ_wp2 (%) | 0.00 | 0.74 | 1.82 | 5.36 | 9.55 | 13.02 | 14.07 | 14.41 |
| status | WP1 | WP1 | **new** | WP1 | **new** | WP1 | **new** | WP1 |

**ρ estimator (fixed for all of WP2).** `scripts/measure_rho.py`: greedy policy, held-out
seeds 142–146, 100 episodes each (250k arrivals per level). The 20-episode values used in v3
and WP1 (`rho` in `family.json`) are reported alongside but not used in WP2 tests.

**Arms.** `ac_dqn_widest`, `aconly` (learned, 2000 training episodes, frozen `base.yaml`),
`greedy` and `greedy_shortest` (deterministic). Same image, seeding fix, held-out evaluation
(seed + 100, 200 episodes, ε = 0) as WP1. The joint agent is not swept (Gate 1).

**Seeds.** 42–51 at every level. At the five WP1 levels, seeds 42–43 are the WP1-batch runs
(same code, same image, same seeding); seeds 44–51 are new. Learned runs:
5 × 8 + 3 × 10 = 70 per arm, 140 in total.

## 2. Hypotheses

**W1 (primary) — F increases with ρ.** The OLS slope of F on ρ_wp2 over all 80 (level, seed)
points is positive; its 95 % cluster-bootstrap CI (resampling seeds within level, 10 000
replicates) excludes 0.

**W2 (primary) — no flexibility value without headroom.** At ratio 0.4 (ρ ≈ 0), F is
equivalent to 0 within ±0.5 pp: two one-sided paired t-tests (TOST), α = 0.05, n = 10.

**W3 (primary) — out-of-level prediction.** The three new levels were never used to fit
anything. Predictions below come from the OLS fit (F = 0.342·ρ +0.435 pp, n = 10) of F on ρ_wp2 using the WP1 data only
(5 levels × seeds 42–43), and are committed here before any run at those levels.

| ratio | ρ_wp2 (%) | predicted F (pp) | 95 % prediction interval |
|---|---|---|---|
| 1.0 | 1.82 | +1.06 | [+0.34, +1.77] |
| 1.6 | 9.55 | +3.70 | [+3.04, +4.35] |
| 2.5 | 14.07 | +5.24 | [+4.42, +6.06] |

Supported if the mean absolute error of the three level means (n = 10 each) is ≤ 1.0 pp
**and** every observed mean lies inside its prediction interval.

**W4 (secondary, confirmatory on fresh data) — foresight shrinks with ρ.** WP1 showed
admission foresight G(ρ) = acceptance(`ac_dqn_widest`) − acceptance(`greedy`) falling from
+2.7 pp at ρ = 0 to +0.3–0.8 pp at high ρ, against the WP1 expectation of a constant.
Because that pattern was seen before this registration, W4 is tested on **new data only**
(seeds 44–51 at WP1 levels and all seeds at new levels): slope of G on ρ_wp2 < 0, cluster
bootstrap CI as in W1, one-sided α = 0.05.

**W5 (descriptive) — sign-change point.** ρ\* = −intercept/slope of the W1 fit, with its
cluster-bootstrap CI. Reported, not tested.

## 3. Analysis

- Unit: training seed; all policies within a seed see identical held-out traffic.
- Multiplicity: Holm–Bonferroni over {W1, W2, W3}. W3's criterion is not a p-value; it
  enters Holm as pass/fail and is reported as such. W4 and W5 are reported uncorrected.
- No exclusions. A crashed run is re-run once with the same seed; if it crashes again it is
  reported missing and the level mean uses the remaining seeds.
- Everything is computed by a committed script from `results/all_results.csv`, batch `wp1`
  (= every run made after the seeding fix on the same image; WP2 runs share it).

## 4. Decision rule (Gate 2, WP2.1 part)

| Outcome | Paper |
|---|---|
| W1, W2, W3 supported | ρ presented as a **predictive criterion** with a fitted slope and ρ\*; "law" allowed only if WP2.3 (real topologies) also passes |
| W1, W2 supported, W3 fails | "criterion" — monotone and zero at ρ = 0, but not quantitatively predictive; slope reported as descriptive |
| W2 supported only | demote to a **necessary condition**: ρ ≈ 0 ⇒ no value from path flexibility |
| W2 fails | the structural claim is wrong as stated; stop and re-examine before WP2.2 |

## 5. Stopping rule

WP2.1 ends when every level has seeds 42–51 for both learned arms. No seeds are added in
response to results.

## 6. WP2.2 and WP2.3 — choices fixed now, predictions registered later

The choices below are fixed on this commit, **before any ρ is computed on these networks**,
so that neither the network set nor the capacity rule can be tuned to the result. Each part
gets a dated amendment with its run list (WP2.2) or its numerical predictions (WP2.3) before
its runs.

**WP2.2 — second controlled family.** Core graph: SNDlib `germany50` (50 nodes, 88 links,
average degree 3.52), a real regional backbone, twice the size of the current synthetic
24-node Waxman core. Construction identical to the current family: 15 endpoints attached by
degree-1 access links to the highest-degree core nodes; core capacity 600 Mbps with a
deterministic ±20 % per-edge jitter (hashlib-based, not Python `hash()`); access capacity
600 × ratio; ratios {0.4, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0}; capacity_scale calibrated to
greedy acceptance 0.47. Also, on a subset of levels: operating points greedy ≈ 0.35 and
≈ 0.60, and K = 6. Question: is the slope of F on ρ invariant across cores?

**WP2.3 — out-of-sample real topologies.**

*Network set (fixed):* the SNDlib networks with N ≤ 30 nodes, excluding `germany50` (used in
WP2.2), the near-full meshes with N ≤ 11 (`dfn-bwin`, `dfn-gwin`, `di-yuan`, `pdh`) and the
multi-layer/test instances (`brain`, `ta1`, `ta2`, `sun`, `zib54`): `abilene`, `polska`,
`nobel-us`, `atlanta`, `newyork`, `nobel-germany`, `geant`, `janos-us`, `nobel-eu`
(9 networks, N = 12–28, average degree 2.50–6.12). `france`, `norway` and `cost266` are
held in reserve, used only to replace a network that fails to build (disconnected after
collapsing parallel links); any replacement is reported.

*Endpoint and access rule:* every SNDlib node is a core router; parallel links are collapsed
into one undirected link with unit routing weight. Each core node gets exactly one slice
endpoint attached by its own degree-1 access link (V = N). Traffic is the simulator's own
model (uniform ordered endpoint pairs, 1–3 connections per slice), unchanged.

*Capacity rule — load-proportional provisioning:* for each link e, L_e = number of ordered
endpoint pairs (i, j) whose first candidate path (the k = 0 path the simulator computes)
traverses e in either direction. Capacity C_e = max(L_e, 0.1 · max_e L_e) in arbitrary
units, the same formula for access and core links. One global capacity_scale per network is
then calibrated by the existing binary search to greedy acceptance 0.47. No other per-network
parameter exists. Rationale: every link starts at the same utilisation under shortest-path
routing, so the rule does not choose where the bottleneck is; ρ is left to the network's
structure.

*Procedure:* compute ρ_wp2 with the WP2 estimator → predict F from the WP2.1 fit (the W1
regression, with its 95 % prediction interval) → commit predictions → train `ac_dqn_widest`
and `aconly`, seeds 42–46 → report mean absolute error and coverage of the intervals.
Pass criterion (fixed now): MAE ≤ 1.0 pp and at least 7 of 9 observed means inside their
intervals.

**WP2.4 — sensitivity and cost.** ρ estimated with greedy vs `aconly` as the reference
policy and at 20/100/500 episodes; wall-clock cost of estimating ρ vs training one agent.
