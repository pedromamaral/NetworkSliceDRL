from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dqn_unified import DQNUnified


class DuellingMLP(nn.Module):
    """Duelling network architecture.

    Q(s,a) = V(s) + A(s,a) − mean_a'[A(s,a')]

    The mean-subtraction stabilises training by keeping advantage estimates
    centred around zero, decoupling value and advantage learning.
    """

    def __init__(self, state_dim: int, hidden: int, action_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.v_head = nn.Linear(hidden, 1)
        self.a_head = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        V = self.v_head(feat)
        A = self.a_head(feat)
        return V + A - A.mean(dim=1, keepdim=True)


class DDQNUnified(DQNUnified):
    """Duelling Double DQN for unified ``Discrete(K+1)`` action space.

    Architecture: ``DuellingMLP`` (V + A heads).
    Update rule  : Double DQN — the online network selects the greedy next
                   action; the target network evaluates its Q-value.
                   This decouples action selection from evaluation, reducing
                   overestimation bias compared to plain DQN.
    """

    def __init__(self, state_dim: int, action_dim: int, cfg: dict) -> None:
        # Initialise base class to set up device, hyperparams, and buffer.
        super().__init__(state_dim, action_dim, cfg)
        hidden: int = cfg.get("hidden_size", 256)

        # Replace standard MLP networks with duelling networks.
        self.q = DuellingMLP(state_dim, hidden, action_dim).to(self.device)
        self.q_target = DuellingMLP(state_dim, hidden, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])

    def learn(self) -> float | None:
        """Double DQN Bellman update.

        target = r + γ(1−d) · Q_target(s', argmax_a Q_online(s', a))
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
            # Online network selects the best next action.
            a_star = self.q(S2).argmax(dim=1, keepdim=True)
            # Target network evaluates that action (Double DQN).
            target = R + self.gamma * (1.0 - D) * (
                self.q_target(S2).gather(1, a_star).squeeze(1)
            )

        loss = nn.functional.mse_loss(q_val, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        return float(loss.item())
