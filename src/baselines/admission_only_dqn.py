"""Admission-only DQN baseline.

A standard DQN that learns only the binary *admit / reject* decision.
When the network decides to admit, it always routes via the shortest
pre-computed path (k=0).  Path selection is therefore fixed (no
learning), which makes this a useful ablation that isolates the value
of learned routing from learned admission.

Architecture: identical 3-layer MLP to ``DQNUnified`` but with
``action_dim = 2`` (reject=0, admit=1).

Unified mode output:
    Q-value for 0 (reject) and 1 (admit-via-path-0).
    ``select_action`` returns an ``int``:  0 or 1.

Separated mode output:
    Same binary Q-network; ``select_action`` returns a ``tuple[int, int]``:
    ``(admit_decision, 0)`` — path index is always 0.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.agents.replay_buffer import ReplayBuffer
from src.agents.dqn_unified import MLP


_ADMIT_DIM = 2  # reject=0, admit=1


class AdmissionOnlyDQN:
    """DQN that learns only admit/reject; always routes via path 0.

    Args:
        state_dim: Dimension of the flat observation vector.
        cfg:       Hyper-parameter dict (same keys as ``DQNUnified``).
        mode:      ``"unified"`` or ``"separated"``.
    """

    def __init__(self, state_dim: int, cfg: dict, mode: str = "unified") -> None:
        if mode not in ("unified", "separated"):
            raise ValueError(f"mode must be 'unified' or 'separated', got {mode!r}")
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden: int = cfg.get("hidden_size", 256)

        self.q = MLP(state_dim, hidden, _ADMIT_DIM).to(self.device)
        self.q_target = MLP(state_dim, hidden, _ADMIT_DIM).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.buf = ReplayBuffer(cfg["replay_capacity"])

        self.gamma: float = cfg["gamma"]
        self.batch: int = cfg["batch_size"]
        self.eps: float = cfg["epsilon_start"]
        self.eps_end: float = cfg["epsilon_end"]
        self.eps_decay: int = cfg["epsilon_decay_steps"]
        self.steps: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decay_eps(self) -> None:
        if self.eps > self.eps_end:
            self.eps = max(
                self.eps_end,
                self.eps - (self.eps - self.eps_end) / self.eps_decay,
            )

    def _admit_int(self, state: np.ndarray) -> int:
        """Return 0 (reject) or 1 (admit) using ε-greedy policy."""
        self._decay_eps()
        if np.random.random() < self.eps:
            return int(np.random.randint(_ADMIT_DIM))
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q(s).argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Public API (mirrors DRL agents)
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int | tuple[int, int]:
        """Return action with path always fixed to 0 when admitting.

        Unified:   int  — 0 (reject) or 1 (admit, path 0)
        Separated: tuple — (0, 0) (reject) or (1, 0) (admit, path 0)
        """
        admit = self._admit_int(state)
        if self.mode == "unified":
            return admit  # 0=reject, 1=admit-path-0 (matches env convention)
        return (admit, 0)

    def store(self, s, a, r: float, s_next, done: bool) -> None:
        """Store transition.  ``a`` must be the admit int (0 or 1)."""
        # Accept both int and tuple; extract just the admit decision.
        if isinstance(a, tuple):
            a = a[0]
        self.buf.push(s, int(a), r, s_next, done)

    def learn(self) -> float | None:
        """Standard DQN Bellman update over the admit/reject Q-network."""
        if len(self.buf) < self.batch:
            return None

        states, actions, rewards, next_states, dones = self.buf.sample(self.batch)
        S = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        A = torch.as_tensor(
            np.array(actions, dtype=np.int64), dtype=torch.long
        ).to(self.device)
        R = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        S2 = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
        D = torch.as_tensor(dones, dtype=torch.float32).to(self.device)

        q_val = self.q(S).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = R + self.gamma * (1.0 - D) * self.q_target(S2).max(dim=1).values

        loss = nn.functional.mse_loss(q_val, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        return float(loss.item())

    def update_target(self) -> None:
        self.q_target.load_state_dict(self.q.state_dict())
