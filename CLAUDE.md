# CLAUDE.md — netslice-drl

DRL slice admission control + core routing (IP/MPLS), Pedro Amaral (FCT-UNL / IT). Target: IEEE TNSM.
Current thesis: learned *admission* + widest-path routing beats greedy/trunk reservation; the
routing-headroom statistic rho predicts the value of path flexibility (pre-registered WPs).

**Start every session by reading `.mem/STATE.md`, then `.mem/INDEX.md`; read only what it routes.**
Full original spec (MDP, state order, agent rules, metrics, infra): `docs/PROJECT_SPEC.md` — read by section
(`rg -n "^## " docs/PROJECT_SPEC.md`). Parts of it (claims "47.5%/18%", §10 status) are outdated; `.mem/` wins.

## Non-negotiable rules
- Model = count reward + hard capacity (admission_mode hard, reward_mode count). Revenue/soft = ablation only.
- Any change to MDP formulation, state vector, or reward → discuss with Pedro first.
- State order fixed: [τ, d, b, p, M_flat, n_inelastic, n_elastic, B_flat] (changing it breaks checkpoints).
- Gymnasium API (5-tuple step). Never reserve partial connections. Recompute B every step.
- Determinism: seed numpy, torch, AND stdlib random. Baselines on identical traffic seeds; eval seed = seed+100.
- Never change metric definitions to fit results; investigate bugs first.
- Confirmatory experiments are pre-registered (paper/preregistration_wp*.md) BEFORE runs; amendments, not edits.
- Paper numbers only from results/all_results.csv (scripts/collect_results.py registry) or .mem/results/*.json.
- Do not mock the topology. Do not git-add paper/plano_publicacao_TNSM.md or publishability_assessment_rho_law.md.
- GPU servers: batches on gpu15 only; gpu14 small non-GPU tasks; never touch others' jobs; no long jobs on laptop.

## Context rules (research-memory)
1. Start from STATE.md + INDEX.md; read code by line range (`rg -n` first).
2. Recall by grep: `rg -i "<kw>" .mem/avenues.md .mem/experiments/LEDGER.md .mem/conclusions.md`.
3. Raw outputs never enter context: `tools/summarize_run.py`, `tools/runs_status.sh`, `tail -n 30`, `rg -m 20`. Never cat logs.
4. Bulk reading → subagent returning ≤20 lines with file:line refs.
5. Replies ≤25 lines unless asked; tables over prose.
6. Write decisions/conclusions/avenues to `.mem/` immediately.
7. One phase per session; end with `/research-memory state`, then `/clear`.
8. Think phases on the strongest model; do phases (implement/deploy/status/fetch/figures) on a cheaper one.

## Quick commands
- Tests: `python -m pytest tests/ -v`
- Status: `bash tools/runs_status.sh` · Summarize: `python3 tools/summarize_run.py E### '<glob>' ...`
- Collect: `python3 scripts/collect_results.py` · Figures: `python3 paper/make_figures.py`
