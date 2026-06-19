from __future__ import annotations

import collections
import random

import numpy as np


class ReplayBuffer:
    """Circular experience replay buffer.

    Works with both unified (int) and separated (tuple[int, int]) actions —
    the ``actions`` element in each sample is a plain Python list whose
    elements are whatever was pushed, so callers must convert accordingly.
    """

    def __init__(self, capacity: int) -> None:
        self.buf: collections.deque = collections.deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Return a random mini-batch as numpy arrays (states/next_states/rewards/dones)
        and a plain Python list for actions (int or tuple depending on agent type)."""
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            list(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)
