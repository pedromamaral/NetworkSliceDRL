"""Aggregate-resource admission DQN -- a stand-in for the 5G-core literature.

Why this baseline exists
------------------------
Greedy, the revenue heuristic and the AC-only DQN are all our own
constructions. A referee will reasonably ask how the proposed agent compares
against a *published* method. The dominant family of DRL admission controllers
for the 5G core (e.g. SARA/DSARA [villota2022admission], FSAC [wang2024fsac],
and the online formulation of [sulaiman2025datadriven]) shares two structural
properties that distinguish it from this work:

  1. the substrate is represented as AGGREGATE resource pools -- totals and
     per-node-class capacities -- not as per-endpoint-pair, per-path state;
  2. path-level routing is not in the action space; placement is delegated to a
     heuristic or assumed.

Reproducing any one of those papers verbatim is not possible here: they target
different substrates and objectives, so a direct port would compare our problem
against their problem. Instead we reproduce the *methodological essence* -- the
aggregate state abstraction and the admission-only action space -- while
holding everything else identical to our own agents: the same count objective,
the same duelling-DQN machinery, the same loss, clipping, schedules and
training budget.

That makes the comparison informative rather than rhetorical. Against the
AC-only DQN (which sees the FULL state but also cannot route) it isolates the
value of the path-level state representation; against the joint agent it
isolates representation and routing together.

State abstraction
-----------------
The full observation is
    [ tau, d, b, p | Mt (V^2) | n_inel, n_el | B (V^2*K) | phi (K) ]
This agent consumes only an aggregate projection of it:
    [ tau, d, b, p, n_inel, n_el,
      mean(B), min(B), frac(B < b), std(B) ]
i.e. request features, active-slice counts, and four scalar summaries of
residual capacity. No per-pair or per-path resolution, and crucially no
feasibility signal phi. This is the information an aggregate resource model
would expose.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.agents.replay_buffer import ReplayBuffer
from src.agents.dqn_unified import MLP

_ADMIT_DIM = 2        # reject = 0, admit (via shortest path) = 1
_AGG_DIM = 10         # see module docstring


class AggregateStateDQN:
    """Admission-only DQN over an aggregate view of the substrate.

    Args:
        state_dim: dimension of the FULL observation (used to locate slices).
        cfg:       hyper-parameters; must carry ``num_nodes_eff`` (V) and
                   ``k_shortest_paths`` (K) so the observation can be decoded.
        mode:      ``"unified"`` or ``"separated"``.
    """

    def __init__(self, state_dim: int, cfg: dict, mode: str = "unified") -> None:
        if mode not in ("unified", "separated"):
            raise ValueError(f"mode must be 'unified' or 'separated', got {mode!r}")
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden: int = cfg.get("hidden_size", 256)

        self.V: int = int(cfg["num_nodes_eff"])
        self.K: int = int(cfg["k_shortest_paths"])
        self.state_dim = state_dim
        # observation layout offsets
        self._mt0 = 4
        self._cnt0 = 4 + self.V ** 2
        self._b0 = self._cnt0 + 2
        self._b1 = self._b0 + self.V ** 2 * self.K

        self.q = MLP(_AGG_DIM, hidden, _ADMIT_DIM).to(self.device)
        self.q_target = MLP(_AGG_DIM, hidden, _ADMIT_DIM).to(self.device)
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

    def project(self, state: np.ndarray) -> np.ndarray:
        """Compress the full observation to the aggregate view."""
        s = np.asarray(state, dtype=np.float32)
        req = s[0:4]                                   # tau, d, b, p
        cnt = s[self._cnt0:self._cnt0 + 2]             # active slice counts
        B = s[self._b0:self._b1]                       # residual capacities
        b_norm = s[2]                                  # normalised bandwidth demand
        agg = np.array([
            float(B.mean()),
            float(B.min()),
            float((B < b_norm).mean()),                # share of paths too small
            float(B.std()),
        ], dtype=np.float32)
        return np.concatenate([req, cnt, agg])

    def _decay_eps(self) -> None:
        if self.eps > self.eps_end:
            self.eps = max(self.eps_end,
                           self.eps - (self.eps - self.eps_end) / self.eps_decay)

    def _admit_int(self, state: np.ndarray) -> int:
        self._decay_eps()
        if np.random.random() < self.eps:
            return int(np.random.randint(_ADMIT_DIM))
        x = torch.as_tensor(self.project(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q(x).argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Public API (mirrors the other agents)
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray):
        """Admit via the shortest path (k=0) or reject; no routing choice."""
        admit = self._admit_int(state)
        if self.mode == "unified":
            return admit            # 0 = reject, 1 = admit via path 0
        return (admit, 0)

    def store(self, s, a, r: float, s_next, done: bool) -> None:
        if isinstance(a, tuple):
            a = a[0]
        # store the PROJECTED state so the buffer matches the network input
        self.buf.push(self.project(s), int(a), r, self.project(s_next), done)

    def learn(self) -> float | None:
        if len(self.buf) < self.batch:
            return None
        states, actions, rewards, next_states, dones = self.buf.sample(self.batch)
        S = torch.as_tensor(np.array(states, dtype=np.float32)).to(self.device)
        A = torch.as_tensor(np.array(actions, dtype=np.int64)).to(self.device)
        R = torch.as_tensor(np.array(rewards, dtype=np.float32)).to(self.device)
        S2 = torch.as_tensor(np.array(next_states, dtype=np.float32)).to(self.device)
        D = torch.as_tensor(np.array(dones, dtype=np.float32)).to(self.device)

        q_val = self.q(S).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = R + self.gamma * (1.0 - D) * self.q_target(S2).max(dim=1).values

        # identical machinery to the proposed agents, so the comparison
        # isolates representation rather than training stability
        loss = nn.functional.smooth_l1_loss(q_val, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.opt.step()
        self.steps += 1
        return float(loss.item())

    def update_target(self) -> None:
        self.q_target.load_state_dict(self.q.state_dict())
