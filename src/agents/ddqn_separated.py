from __future__ import annotations

import torch
import torch.nn as nn

from .dqn_separated import DQNSeparated
from .ddqn_unified import DuellingMLP


class DDQNSeparated(DQNSeparated):
    """Duelling Double DQN, separated ``Tuple(Discrete(2), Discrete(K))`` space.

    Same two-independent-network / separate-D_rt-buffer structure as
    :class:`DQNSeparated` (admission net trained on all transitions, routing net
    trained only on admitted transitions), but:

      * each network is a duelling ``DuellingMLP`` (``Q = V + A − mean A``), and
      * the Bellman target uses Double DQN — the online net picks the greedy
        next action, the target net evaluates it — applied to each network
        independently.

    Only the two hooks differ from the base class; ``__init__``, action
    selection, dual-buffer ``store`` and ``learn`` are inherited unchanged.
    """

    def _make_net(self, state_dim: int, hidden: int, out_dim: int) -> nn.Module:
        return DuellingMLP(state_dim, hidden, out_dim)

    def _next_value(self, online: nn.Module, target: nn.Module,
                    S2: torch.Tensor) -> torch.Tensor:
        """Double DQN bootstrap: online selects argmax, target evaluates it."""
        a_star = online(S2).argmax(dim=1, keepdim=True)
        return target(S2).gather(1, a_star).squeeze(1)
