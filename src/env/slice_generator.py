import numpy as np


class SliceGenerator:
    """Generates synthetic network slice requests.

    Each request contains:
        type      – 0 = inelastic, 1 = elastic (per §2.2; P(type=0) = inelastic_prob)
        duration  – holding time in time-steps (geometric distribution)
        bandwidth – required bandwidth in Mbps (uniform)
        price     – revenue per time-step (proportional to bandwidth)
        Mt        – (V × V) binary connectivity matrix with 1–3 logical links
    """

    def __init__(self, cfg: dict, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.V: int = cfg["num_nodes"]

    def sample(self) -> dict:
        bw = float(self.rng.uniform(*self.cfg["bandwidth_range"]))
        # duration_slots is in time-slot units (paper's d_t); used for state and reward.
        # duration_steps is the tick count: each MDP step is 1 arrival event, so there
        # are arrival_rate steps per time slot on average.
        arrival_rate = float(self.cfg.get("arrival_rate", 1.0))
        duration_slots = int(self.rng.geometric(1.0 / self.cfg["slice_duration_mean"]))
        duration_steps = max(1, int(round(duration_slots * arrival_rate)))
        price = bw * self.cfg["price_scale"] * (
            1.0 + 0.1 * float(self.rng.standard_normal())
        )
        slice_type = int(self.rng.random() > self.cfg["inelastic_prob"])

        n_conn = int(self.rng.integers(1, 4))  # 1, 2 or 3 logical connections
        Mt = np.zeros((self.V, self.V), dtype=np.int8)
        for _ in range(n_conn):
            i, j = self.rng.choice(self.V, 2, replace=False)
            Mt[int(i), int(j)] = 1

        return {
            "type": slice_type,
            "duration": duration_slots,        # paper's d_t (time slots) — state + reward
            "duration_steps": duration_steps,  # tick counter (MDP steps)
            "bandwidth": bw,
            "price": max(price, 0.1),
            "Mt": Mt,
        }
