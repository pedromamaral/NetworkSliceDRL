from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .replay_buffer import ReplayBuffer


class MLP(nn.Module):
    """Three-layer MLP Q-network."""

    def __init__(self, state_dim: int, hidden: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNUnified:
    """Standard DQN for unified ``Discrete(K+1)`` action space.

    Actions: 0 = reject, 1..K = admit via k-th shortest path.
    Update rule: standard Bellman target using target network max.
    Epsilon: linear decay from ``epsilon_start`` to ``epsilon_end``
             over ``epsilon_decay_steps`` calls to ``select_action``.
    """

    def __init__(self, state_dim: int, action_dim: int, cfg: dict) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden: int = cfg.get("hidden_size", 256)

        self.q = MLP(state_dim, hidden, action_dim).to(self.device)
        self.q_target = MLP(state_dim, hidden, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.buf = ReplayBuffer(cfg["replay_capacity"])

        self.gamma: float = cfg["gamma"]
        self.batch: int = cfg["batch_size"]
        self.eps: float = cfg["epsilon_start"]
        self.eps_end: float = cfg["epsilon_end"]
        self.eps_decay: int = cfg["epsilon_decay_steps"]
        self.action_dim: int = action_dim
        self.steps: int = 0

        # Optional action masking: never select "admit via path k" when the
        # observation's feasibility signal (last K entries) says path k cannot
        # satisfy the request.  Reject (action 0) is always permitted.  The
        # feasibility signal is part of the observation (not privileged info).
        self.use_mask: bool = bool(cfg.get("action_mask", False))
        self.n_paths: int = action_dim - 1                # K
        self._mask_thresh: float = 1.0 / 3.0 - 1e-6       # matches env feasibility encoding

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decay_eps(self) -> None:
        """One step of linear epsilon decay."""
        if self.eps > self.eps_end:
            self.eps = max(
                self.eps_end,
                self.eps - (self.eps - self.eps_end) / self.eps_decay,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        self._decay_eps()

        # Build feasibility mask over path actions 1..K (reject always allowed).
        feasible_path = None
        if self.use_mask:
            feas = np.asarray(state[-self.n_paths:], dtype=np.float32)
            feasible_path = feas >= self._mask_thresh  # bool over K paths

        if np.random.random() < self.eps:
            if feasible_path is not None:
                # Explore only among reject + feasible paths.
                choices = [0] + [k + 1 for k in range(self.n_paths) if feasible_path[k]]
                return int(np.random.choice(choices))
            return int(np.random.randint(self.action_dim))

        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q(s).squeeze(0).cpu().numpy()
        if feasible_path is not None:
            for k in range(self.n_paths):
                if not feasible_path[k]:
                    q[k + 1] = -np.inf  # forbid admitting via infeasible path k
        return int(np.argmax(q))

    def store(self, s, a: int, r: float, s_next, done: bool) -> None:
        self.buf.push(s, a, r, s_next, done)

    def learn(self) -> float | None:
        """Sample a mini-batch and apply one gradient update.

        Returns the MSE loss value, or ``None`` if the buffer is not yet
        large enough to form a full batch.
        """
        if len(self.buf) < self.batch:
            return None

        states, actions, rewards, next_states, dones = self.buf.sample(self.batch)
        S = torch.as_tensor(states).to(self.device)
        A = torch.as_tensor(
            np.array(actions, dtype=np.int64)
        ).to(self.device)
        R = torch.as_tensor(rewards).to(self.device)
        S2 = torch.as_tensor(next_states).to(self.device)
        D = torch.as_tensor(dones).to(self.device)

        q_val = self.q(S).gather(1, A.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target = R + self.gamma * (1.0 - D) * self.q_target(S2).max(dim=1).values

        loss = nn.functional.smooth_l1_loss(q_val, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.opt.step()
        self.steps += 1
        return float(loss.item())

    def update_target(self) -> None:
        """Hard copy online weights to target network."""
        self.q_target.load_state_dict(self.q.state_dict())
