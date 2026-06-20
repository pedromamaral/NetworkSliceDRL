from __future__ import annotations

import gymnasium as gym
import numpy as np

from .slice_generator import SliceGenerator
from .topology import NetworkTopology


class NetworkEnv(gym.Env):
    """Gymnasium environment for network slice admission control.

    Observation (float32, shape=(state_dim,)):
        [ request_features(4) | Mt_flat(V²) | active_counts(2) | B_flat(V²·K) ]
        where:
            request_features = [slice_type, duration, bandwidth, price]
            Mt_flat          = flattened logical connectivity matrix
            active_counts    = [#inelastic_active, #elastic_active]
            B_flat           = flattened bottleneck-capacity tensor B[i,j,p]

    Action – unified mode:
        Discrete(K+1):  0 = reject,  1..K = route via p-th shortest path

    Action – separated mode:
        Tuple(Discrete(2), Discrete(K)):  (admit∈{0,1}, path_k∈{0..K-1})

    Reward:
        0           – reject or infeasible attempt
        d · price   – admission; d = slice duration in time-steps
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, cfg: dict, mode: str = "unified") -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        self.topo = NetworkTopology(cfg["topology_file"], cfg["k_shortest_paths"])
        self.K: int = cfg["k_shortest_paths"]
        self.V: int = self.topo.V

        self.rng = np.random.default_rng(cfg.get("seed", 42))
        # Override num_nodes with the actual topology size so SliceGenerator's
        # Mt matrix always has the same V as the loaded graph.
        gen_cfg = {**cfg, "num_nodes": self.V}
        self.gen = SliceGenerator(gen_cfg, self.rng)
        self.penalty_weight: float = cfg.get("penalty_weight", 0.5)

        # state_dim = request(4) + Mt(V²) + active_counts(2) + B(V²·K)
        self.state_dim: int = 4 + self.V ** 2 + 2 + self.V ** 2 * self.K
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        if mode == "unified":
            self.action_space = gym.spaces.Discrete(self.K + 1)
            # action_dims is exposed so agents can inspect it without .n on Tuple
            self.action_dims: int | tuple[int, int] = self.K + 1
        else:
            self.action_space = gym.spaces.Tuple(
                (gym.spaces.Discrete(2), gym.spaces.Discrete(self.K))
            )
            self.action_dims = (2, self.K)

        # Runtime state – properly initialised in reset()
        self.active_slices: list = []
        self.current_request: dict = self.gen.sample()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.topo.reset_capacities()
        self.active_slices = []
        self.current_request = self.gen.sample()
        return self._get_obs(), {}

    def step(self, action):
        req = self.current_request

        # --- Decode action ---------------------------------------------------
        if self.mode == "unified":
            admit = int(action) > 0
            path_k = int(action) - 1 if admit else 0
        else:
            admit = bool(action[0])
            path_k = int(action[1])

        reward = 0.0
        info: dict = {"admitted": False, "sla_ok": True}

        # --- Admission decision ----------------------------------------------
        if admit:
            connections = [
                (i, j)
                for i in range(self.V)
                for j in range(self.V)
                if req["Mt"][i, j] == 1
            ]

            # Compute bottleneck once for this time-step
            B = self.topo.bottleneck_tensor()
            feasible = True
            chosen_paths: list = []

            for (i, j) in connections:
                path_list = self.topo.paths.get(
                    (self.topo.nodes[i], self.topo.nodes[j]), []
                )
                if not path_list or path_k >= len(path_list):
                    feasible = False
                    break
                path = path_list[path_k]
                if B[i, j, path_k] >= req["bandwidth"]:
                    chosen_paths.append((path, req["bandwidth"]))
                else:
                    feasible = False
                    break

            if feasible:
                for path, bw in chosen_paths:
                    self.topo.reserve(path, bw)
                # Tick counter uses duration_steps (MDP steps); reward uses
                # duration (time slots) per §2.5 of the paper formulation.
                self.active_slices.append(
                    (req, chosen_paths, req["duration_steps"], req["bandwidth"])
                )
                reward = float(req["duration"] * req["price"])
                info["admitted"] = True
                # With hard reservation SLA is guaranteed; P(s,a) = 0 always.
                info["sla_ok"] = True

        info["avg_path_utilization"] = self.topo.avg_link_utilization()

        # --- Tick active slices (age by one time-step) -----------------------
        new_active = []
        for (s_req, s_paths, s_rem, s_bw) in self.active_slices:
            s_rem -= 1
            if s_rem <= 0:
                for path, bw in s_paths:
                    self.topo.release(path, bw)
            else:
                new_active.append((s_req, s_paths, s_rem, s_bw))
        self.active_slices = new_active

        self.current_request = self.gen.sample()
        return self._get_obs(), reward, False, False, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        req = self.current_request
        rt_vec = np.array(
            [req["type"], req["duration"], req["bandwidth"], req["price"]],
            dtype=np.float32,
        )
        mt_flat = req["Mt"].flatten().astype(np.float32)
        n_inelastic = sum(1 for s in self.active_slices if s[0]["type"] == 0)
        n_elastic = sum(1 for s in self.active_slices if s[0]["type"] == 1)
        asl = np.array([n_inelastic, n_elastic], dtype=np.float32)
        B_flat = self.topo.bottleneck_tensor().flatten()
        return np.concatenate([rt_vec, mt_flat, asl, B_flat])
