"""Save and restore agent checkpoints.

Only the minimal state needed to resume training or reproduce evaluation is
persisted.  All agents (DQNUnified, DQNSeparated, DDQNUnified, DDQNSeparated,
AdmissionOnlyDQN) expose the same attributes: ``q``, ``q_target``, ``opt``,
``eps``, ``steps``.

Usage::

    save_checkpoint(agent, episode=500, path="results/checkpoints/dqn_ep500.pt")
    load_checkpoint(agent, path="results/checkpoints/dqn_ep500.pt")
"""
from __future__ import annotations

import os

import torch


def save_checkpoint(agent, episode: int, path: str) -> None:
    """Persist agent weights + optimiser + training state to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "episode": episode,
        "q_state_dict": agent.q.state_dict(),
        "q_target_state_dict": agent.q_target.state_dict(),
        "optimizer_state_dict": agent.opt.state_dict(),
        "eps": agent.eps,
        "steps": agent.steps,
    }
    torch.save(payload, path)


def load_checkpoint(agent, path: str) -> int:
    """Restore weights into *agent* in-place.  Returns the saved episode number."""
    payload = torch.load(path, map_location=agent.q.net[0].weight.device
                         if hasattr(agent.q, "net") else "cpu")
    agent.q.load_state_dict(payload["q_state_dict"])
    agent.q_target.load_state_dict(payload["q_target_state_dict"])
    agent.opt.load_state_dict(payload["optimizer_state_dict"])
    agent.eps = payload["eps"]
    agent.steps = payload["steps"]
    return int(payload["episode"])
