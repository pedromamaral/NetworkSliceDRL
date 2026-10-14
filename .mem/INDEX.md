# INDEX — task → files to read (and NOT to read). Always start with STATE.md.
| task | read | skip |
| status / monitor runs | runs.md, then `bash tools/runs_status.sh` | logs (never cat) |
| fetch + summarize results | infra.md (Fetch), runs.md, experiment card | raw CSVs → tools/summarize_run.py |
| analyse a WP vs pre-registration | experiments/E###.md, conclusions.md, paper/preregistration_wp*.md (by section) | _import.md |
| ideate next experiments | project.md, avenues.md (grep), conclusions.md, strategy.md | LEDGER-archive |
| implement / new config / job set | codebase.md, CLAUDE.md, docs/PROJECT_SPEC.md (by section) | results/ |
| deploy on GPU servers | infra.md, runs.md, decisions.md D13-D14 | — |
| paper writing / figures | paper.md, conclusions.md, project.md | old v2 files |
| any change to MDP/state/reward | docs/PROJECT_SPEC.md §2,§4, decisions.md D01 → ask Pedro first | — |
| recall "did we try X?" | `rg -i X .mem/avenues.md .mem/experiments/LEDGER.md .mem/conclusions.md` | — |
- Never read by default: .mem/_import.md (raw import, 2026-10-14), *-archive.md, results/**, checkpoints, wandb.
- Recall by grep, not by reading. Raw outputs → tools/summarize_run.py → .mem/results/E###.json.
