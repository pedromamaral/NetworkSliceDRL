"""Thin logging wrapper: prints to stdout, writes CSV, and optionally logs to WandB.

Usage::

    logger = Logger(cfg)
    logger.log(episode=100, metrics={"acceptance_ratio": 0.82, ...}, loss=0.034)
    # ... training ...
    logger.save_csv()
    logger.finish()
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from typing import Any


class Logger:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._rows: list[dict] = []
        self._use_wandb: bool = cfg.get("use_wandb", False)
        self._wandb_run = None

        # Build output path: results_dir/run_name/metrics.csv
        run_name: str = cfg.get("run_name", cfg.get("agent", "run"))
        results_dir: str = cfg.get("results_dir", "results")
        self._out_dir = os.path.join(results_dir, run_name)
        os.makedirs(self._out_dir, exist_ok=True)
        self._csv_path = os.path.join(self._out_dir, "metrics.csv")

        if self._use_wandb:
            self._init_wandb(cfg, run_name)

    # ------------------------------------------------------------------

    def _init_wandb(self, cfg: dict, run_name: str) -> None:
        try:
            import wandb  # type: ignore

            self._wandb_run = wandb.init(
                project=cfg.get("wandb_project", "netslice-drl"),
                name=run_name,
                config={k: v for k, v in cfg.items() if not k.startswith("_")},
                tags=cfg.get("wandb_tags", []),
                resume="allow",
            )
        except Exception as exc:  # wandb not installed or no API key
            print(f"[Logger] WandB disabled: {exc}", file=sys.stderr)
            self._use_wandb = False

    # ------------------------------------------------------------------

    def log(self, episode: int, metrics: dict[str, Any], loss: float | None) -> None:
        """Record one evaluation checkpoint."""
        row: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "episode": episode,
            "loss": round(loss, 6) if loss is not None else None,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        self._rows.append(row)

        # Stdout summary
        parts = [f"ep={episode:5d}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        for k, v in metrics.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        print("  ".join(parts), flush=True)

        # WandB
        if self._use_wandb and self._wandb_run is not None:
            try:
                import wandb  # type: ignore

                log_dict = {"episode": episode, **({"loss": loss} if loss is not None else {}), **metrics}
                wandb.log(log_dict, step=episode)
            except Exception:
                pass

    def save_csv(self) -> None:
        """Flush all accumulated rows to disk."""
        if not self._rows:
            return
        fieldnames = list(self._rows[0].keys())
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"[Logger] Metrics saved → {self._csv_path}")

    def finish(self) -> None:
        self.save_csv()
        if self._use_wandb and self._wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb.finish()
            except Exception:
                pass
