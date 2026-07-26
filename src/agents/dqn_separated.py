from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .replay_buffer import ReplayBuffer
from .dqn_unified import MLP


class DQNSeparated:
    """Two INDEPENDENT DQNs for the separated ``Tuple(Discrete(2), Discrete(K))``
    action space (CLAUDE.md §2.4, §4.2).

    Decomposition:
      * ADMISSION network ``q``  : Q_a(s) over {reject=0, accept=1}. Trained on
        EVERY transition, from buffer ``buf``.
      * ROUTING   network ``q_route`` : Q_r(s) over the K shortest paths. Queried
        ONLY when the admission decision is "accept", and trained ONLY on
        admitted transitions, from a SEPARATE buffer ``buf_rt`` (the spec's
        D_rt).  Reject steps never reach the routing net.

    This is the honest AC/RA factorisation, and the reason it is expected to
    scale better with K than the unified ``Discrete(K+1)`` agent: the admission
    head stays size-2 no matter how large K grows — only the routing head does,
    and it learns from a cleaner (admitted-only) signal.

    The admission net is exposed as ``q``/``q_target``/``opt`` so the shared
    checkpoint helper persists it; the routing net lives in the parallel
    ``q_route``/``q_route_target``/``opt_route`` attributes.
    """

    def __init__(
        self, state_dim: int, action_dims: tuple[int, int], cfg: dict
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden: int = cfg.get("hidden_size", 256)
        self.n_admit: int = action_dims[0]   # 2
        self.n_path: int = action_dims[1]    # K

        # Admission network (all transitions).
        self.q = self._make_net(state_dim, hidden, self.n_admit).to(self.device)
        self.q_target = self._make_net(state_dim, hidden, self.n_admit).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])

        # Routing network (admitted transitions only).
        self.q_route = self._make_net(state_dim, hidden, self.n_path).to(self.device)
        self.q_route_target = self._make_net(state_dim, hidden, self.n_path).to(self.device)
        self.q_route_target.load_state_dict(self.q_route.state_dict())
        self.q_route_target.eval()
        self.opt_route = torch.optim.Adam(self.q_route.parameters(), lr=cfg["lr"])

        self.buf = ReplayBuffer(cfg["replay_capacity"])      # admission
        self.buf_rt = ReplayBuffer(cfg["replay_capacity"])   # routing (D_rt)

        self.gamma: float = cfg["gamma"]
        self.batch: int = cfg["batch_size"]
        self.eps: float = cfg["epsilon_start"]
        self.eps_end: float = cfg["epsilon_end"]
        self.eps_decay: int = cfg["epsilon_decay_steps"]
        self.steps: int = 0

    # ------------------------------------------------------------------
    # Hooks overridden by the duelling / Double-DQN subclass
    # ------------------------------------------------------------------

    def _make_net(self, state_dim: int, hidden: int, out_dim: int) -> nn.Module:
        return MLP(state_dim, hidden, out_dim)

    def _next_value(self, online: nn.Module, target: nn.Module,
                    S2: torch.Tensor) -> torch.Tensor:
        """Bootstrap value of s' (plain DQN: target-net max)."""
        return target(S2).max(dim=1).values

    # ------------------------------------------------------------------

    def _decay_eps(self) -> None:
        if self.eps > self.eps_end:
            self.eps = max(
                self.eps_end,
                self.eps - (self.eps - self.eps_end) / self.eps_decay,
            )

    def _greedy(self, net: nn.Module, state: np.ndarray) -> int:
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(net(s).argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Public API (mirrors the DRL agents)
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> tuple[int, int]:
        """Independent ε-greedy per head; routing queried ONLY on accept."""
        self._decay_eps()

        # Admission head.
        if np.random.random() < self.eps:
            admit = int(np.random.randint(self.n_admit))
        else:
            admit = self._greedy(self.q, state)

        if admit != 1:
            return (admit, 0)  # reject → routing net not queried

        # Routing head (only reached on accept).
        if np.random.random() < self.eps:
            path = int(np.random.randint(self.n_path))
        else:
            path = self._greedy(self.q_route, state)
        return (admit, path)

    def store(self, s, a: tuple[int, int], r: float, s_next, done: bool) -> None:
        admit = int(a[0])
        path = int(a[1])
        self.buf.push(s, admit, r, s_next, done)          # admission: all steps
        if admit == 1:
            self.buf_rt.push(s, path, r, s_next, done)    # routing: admitted only

    def _learn_head(self, online: nn.Module, target: nn.Module,
                    opt: torch.optim.Optimizer, buf: ReplayBuffer) -> float | None:
        if len(buf) < self.batch:
            return None
        states, actions, rewards, next_states, dones = buf.sample(self.batch)
        S = torch.as_tensor(states).to(self.device)
        A = torch.as_tensor(np.array(actions, dtype=np.int64)).to(self.device)
        R = torch.as_tensor(rewards).to(self.device)
        S2 = torch.as_tensor(next_states).to(self.device)
        D = torch.as_tensor(dones).to(self.device)

        q_val = online(S).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_val = R + self.gamma * (1.0 - D) * self._next_value(online, target, S2)

        loss = nn.functional.smooth_l1_loss(q_val, target_val)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
        opt.step()
        return float(loss.item())

    def learn(self) -> float | None:
        """Update both networks from their own buffers.  Returns the summed
        loss (or None if neither buffer has a full batch yet)."""
        loss_admit = self._learn_head(self.q, self.q_target, self.opt, self.buf)
        loss_route = self._learn_head(
            self.q_route, self.q_route_target, self.opt_route, self.buf_rt
        )
        if loss_admit is None and loss_route is None:
            return None
        self.steps += 1
        return (loss_admit or 0.0) + (loss_route or 0.0)

    def update_target(self) -> None:
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_route_target.load_state_dict(self.q_route.state_dict())
