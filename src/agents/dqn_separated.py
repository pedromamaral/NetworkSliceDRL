from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .replay_buffer import ReplayBuffer


class TwoHeadMLP(nn.Module):
    """Shared backbone with two independent Q-heads.

    Admit head : ``Discrete(2)``  — reject (0) or admit (1)
    Path head  : ``Discrete(K)``  — which of the K shortest paths to use
    """

    def __init__(
        self, state_dim: int, hidden: int, n_admit: int, n_path: int
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.admit_head = nn.Linear(hidden, n_admit)
        self.path_head = nn.Linear(hidden, n_path)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return self.admit_head(feat), self.path_head(feat)


class DQNSeparated:
    """DQN for separated ``Tuple(Discrete(2), Discrete(K))`` action space.

    Joint Q-value:
        Q(s, a_admit, a_path) = Q_admit(s)[a_admit] + Q_path(s)[a_path]

    Both heads share a backbone and are trained jointly via a single
    Bellman MSE loss.  The target uses greedy max on each head independently:
        target = r + γ(1−d) · (max Q_target_admit(s') + max Q_target_path(s'))
    """

    def __init__(
        self, state_dim: int, action_dims: tuple[int, int], cfg: dict
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden: int = cfg.get("hidden_size", 256)
        self.n_admit: int = action_dims[0]
        self.n_path: int = action_dims[1]

        self.q = TwoHeadMLP(state_dim, hidden, self.n_admit, self.n_path).to(
            self.device
        )
        self.q_target = TwoHeadMLP(
            state_dim, hidden, self.n_admit, self.n_path
        ).to(self.device)
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

    def _decay_eps(self) -> None:
        if self.eps > self.eps_end:
            self.eps = max(
                self.eps_end,
                self.eps - (self.eps - self.eps_end) / self.eps_decay,
            )

    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> tuple[int, int]:
        self._decay_eps()
        if np.random.random() < self.eps:
            return (
                int(np.random.randint(self.n_admit)),
                int(np.random.randint(self.n_path)),
            )
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_admit, q_path = self.q(s)
        return (
            int(q_admit.argmax(dim=1).item()),
            int(q_path.argmax(dim=1).item()),
        )

    def store(self, s, a: tuple[int, int], r: float, s_next, done: bool) -> None:
        self.buf.push(s, a, r, s_next, done)

    def learn(self) -> float | None:
        if len(self.buf) < self.batch:
            return None

        states, actions, rewards, next_states, dones = self.buf.sample(self.batch)
        S = torch.as_tensor(states).to(self.device)
        A_admit = torch.as_tensor(
            np.array([a[0] for a in actions], dtype=np.int64)
        ).to(self.device)
        A_path = torch.as_tensor(
            np.array([a[1] for a in actions], dtype=np.int64)
        ).to(self.device)
        R = torch.as_tensor(rewards).to(self.device)
        S2 = torch.as_tensor(next_states).to(self.device)
        D = torch.as_tensor(dones).to(self.device)

        qa, qp = self.q(S)
        q_val = (
            qa.gather(1, A_admit.unsqueeze(1)).squeeze(1)
            + qp.gather(1, A_path.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            qa2, qp2 = self.q_target(S2)
            target = R + self.gamma * (1.0 - D) * (
                qa2.max(dim=1).values + qp2.max(dim=1).values
            )

        loss = nn.functional.mse_loss(q_val, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        return float(loss.item())

    def update_target(self) -> None:
        self.q_target.load_state_dict(self.q.state_dict())
