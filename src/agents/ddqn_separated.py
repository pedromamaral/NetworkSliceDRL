from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dqn_separated import DQNSeparated


class DuellingTwoHeadMLP(nn.Module):
    """Shared backbone with two independent duelling streams.

    Each stream has its own V(s) and A(s,a) sub-heads:
        Q_admit(s, a) = V_admit(s) + A_admit(s,a) − mean[A_admit]
        Q_path(s, p)  = V_path(s)  + A_path(s,p)  − mean[A_path]
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
        # Admit duelling sub-heads
        self.v_admit = nn.Linear(hidden, 1)
        self.a_admit = nn.Linear(hidden, n_admit)
        # Path duelling sub-heads
        self.v_path = nn.Linear(hidden, 1)
        self.a_path = nn.Linear(hidden, n_path)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)

        V_ad = self.v_admit(feat)
        A_ad = self.a_admit(feat)
        q_admit = V_ad + A_ad - A_ad.mean(dim=1, keepdim=True)

        V_p = self.v_path(feat)
        A_p = self.a_path(feat)
        q_path = V_p + A_p - A_p.mean(dim=1, keepdim=True)

        return q_admit, q_path


class DDQNSeparated(DQNSeparated):
    """Duelling Double DQN for separated ``Tuple(Discrete(2), Discrete(K))`` action space.

    Architecture: ``DuellingTwoHeadMLP`` — one duelling stream per decision.
    Update rule  : Double DQN applied independently to each head.
                   The online network selects the best next action per head;
                   the target network evaluates those actions.

    Joint Q-value:
        Q(s, a_admit, a_path) = Q_admit(s)[a_admit] + Q_path(s)[a_path]

    Bellman target:
        a*_admit = argmax Q_online_admit(s')
        a*_path  = argmax Q_online_path(s')
        target   = r + γ(1−d) · (Q_target_admit(s')[a*_admit] + Q_target_path(s')[a*_path])
    """

    def __init__(
        self, state_dim: int, action_dims: tuple[int, int], cfg: dict
    ) -> None:
        # Initialise base class to set up device, hyperparams, and buffer.
        super().__init__(state_dim, action_dims, cfg)
        hidden: int = cfg.get("hidden_size", 256)

        # Replace TwoHeadMLP with duelling variant.
        self.q = DuellingTwoHeadMLP(
            state_dim, hidden, self.n_admit, self.n_path
        ).to(self.device)
        self.q_target = DuellingTwoHeadMLP(
            state_dim, hidden, self.n_admit, self.n_path
        ).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])

    def learn(self) -> float | None:
        """Double DQN Bellman update for both heads jointly."""
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
            # Online network selects best next actions per head.
            qa2_on, qp2_on = self.q(S2)
            a_admit_star = qa2_on.argmax(dim=1, keepdim=True)
            a_path_star = qp2_on.argmax(dim=1, keepdim=True)
            # Target network evaluates those actions.
            qa2_tgt, qp2_tgt = self.q_target(S2)
            target = R + self.gamma * (1.0 - D) * (
                qa2_tgt.gather(1, a_admit_star).squeeze(1)
                + qp2_tgt.gather(1, a_path_star).squeeze(1)
            )

        loss = nn.functional.mse_loss(q_val, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        return float(loss.item())
