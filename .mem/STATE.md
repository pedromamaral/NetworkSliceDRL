# STATE — 2026-10-14 · main @68b7df1 (+ memory-init commit)
Phase just done: research-memory init (.mem/ built from .mem/_import.md + git + repo scan).

## Active runs (runs.md)
- E015 WP2.2 germany50 | gpu15 | running, 7 containers, ETA ~2026-10-15 09:00 (checked 14:31: rc!=0 = 0)
- E014 WP2.3 SNDlib | gpu15 | finished 10-14 05:24, NOT pulled
- E016 WP4.1 exp-pricing | gpu14 | finished, NOT pulled

## Latest results
- E013 WP2.1: F = 0.371*rho + 0.13pp (K13); G falls with rho (K14). Gate 1: acwidest best everywhere (K11).
- E016 raw: pricing ~0.50 > acwidest 0.484 → P3 likely fails (K16, unverified).

## Open problem
Is C2 (rho criterion) robust out-of-sample (WP2.3) and across cores (WP2.2)? Does exp-pricing
beating acwidest (if confirmed) undermine C1 → need to reposition C1/C6 for TNSM.

## EXACT next step (research-do: fetch, then research-think: analyse)
1. `rsync -av --exclude='checkpoints/' GPUserver1:~/netslice-drl/results/ results/remote/gpu15/` and
   `rsync -av --exclude='checkpoints/' GPUserver2:~/netslice-drl/results/exp_pricing results/remote/gpu14/`
2. `python3 tools/summarize_run.py E014 'results/remote/gpu15/baselines_wp23_*'` ; same for E016 exp_pricing CSVs.
3. Register wp23 + ap_* arms in scripts/collect_results.py; run WP2.3 test vs data/sndlib_topo/predictions.json.
4. WP2.2 (E015): after ~10-15 09:00, `bash tools/runs_status.sh`, then same fetch/analyse vs W6-W9.

## Files next session must read
STATE.md, INDEX.md, runs.md, experiments/E014-wp23-sndlib.md, experiments/E016-wp41-exp-pricing.md, infra.md (Fetch).
