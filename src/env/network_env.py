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

    Reward (controlled by cfg["reward_mode"], default "count"):
        "count"   – 0 reject/infeasible, +1 per admitted slice (paper's headline
                    "accept more slices" objective; used with admission_mode="hard").
        "revenue" – 0 reject/infeasible, d·price per admission, minus an SLA
                    penalty accrued while any of the slice's links are
                    oversubscribed (only reachable under admission_mode="soft").

    Admission (controlled by cfg["admission_mode"], default "hard"):
        "hard" – admission is gated on capacity: the chosen path must have
                 enough headroom for every logical connection, or the attempt
                 fails (no reservation, no reward).  Capacity can never go
                 negative; SLA violations cannot occur.
        "soft" – over-admission allowed; the chosen path is reserved even if
                 it drives a link negative (oversubscribed).  Only used with
                 reward_mode="revenue" to study the SLA-penalty regime.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, cfg: dict, mode: str = "unified") -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        self.topo = NetworkTopology(
            cfg["topology_file"],
            cfg["k_shortest_paths"],
            capacity_scale=cfg.get("capacity_scale", 1.0),
        )
        self.K: int = cfg["k_shortest_paths"]
        self.V: int = self.topo.V

        self.rng = np.random.default_rng(cfg.get("seed", 42))
        # Override num_nodes with the actual topology size so SliceGenerator's
        # Mt matrix always has the same V as the loaded graph.
        gen_cfg = {**cfg, "num_nodes": self.V}
        self.gen = SliceGenerator(gen_cfg, self.rng)
        self.penalty_weight: float = cfg.get("penalty_weight", 0.5)
        # SLA penalty model (only relevant when admission_mode="soft"):
        #   "once"     – full weight·frac·(d·p) charged the first step a slice
        #                violates (harsh on any violation → drives selectivity).
        #   "per_step" – weight·frac·(d·p)/duration_steps per oversubscribed step
        #                (lenient on partial starvation; the discount asymmetry
        #                vs the lump-sum admission reward makes over-admission
        #                look profitable → floody.  Kept for experimentation.)
        self.sla_penalty_mode: str = cfg.get("sla_penalty_mode", "once")
        # admission_mode: "hard" (gate on capacity, never oversubscribe — the
        # 2026-07-07 default, needed for the "accept more slices" objective)
        # or "soft" (over-admission allowed, SLA penalty applies; the
        # 2026-07-03/04 revenue-selectivity experiments used this).
        self.admission_mode: str = cfg.get("admission_mode", "hard")
        # reward_mode: "count" (+1 per admitted slice — paper's headline
        # metric) or "revenue" (d·price minus SLA penalty).
        self.reward_mode: str = cfg.get("reward_mode", "count")

        # Normalisation constants — keep all observation components in [0, 1].
        # Derived from config so they stay consistent across env instances.
        arrival_rate = float(cfg.get("arrival_rate", 1.0))
        dur_mean = float(cfg.get("slice_duration_mean", 10.0))
        self._norm_duration = max(1.0, 5.0 * dur_mean)           # 5× mean covers >99% of geometric
        self._norm_bw = float(cfg.get("bandwidth_range", [10, 100])[1])
        self._norm_price = self._norm_bw * float(cfg.get("price_scale", 1.0)) * 1.5
        # Steady-state active slices = 1 arrival/step × duration_steps_mean
        self._norm_active = max(1.0, 3.0 * arrival_rate * dur_mean)
        # Maximum link capacity from the loaded topology
        self._norm_cap = max(self.topo._cap.values()) if self.topo._cap else 1000.0

        # state_dim = request(4) + Mt(V²) + active_counts(2) + B(V²·K) + feasibility(K)
        self.state_dim: int = 4 + self.V ** 2 + 2 + self.V ** 2 * self.K + self.K
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
        info: dict = {"admitted": False, "admit_revenue": 0.0}

        connections = [
            (i, j)
            for i in range(self.V)
            for j in range(self.V)
            if req["Mt"][i, j] == 1
        ]
        B = self.topo.bottleneck_tensor()

        # Diagnostic: was ANY of the K paths feasible (no oversubscription)?
        any_path_feasible = False
        for k in range(self.K):
            ok = bool(connections)
            for (i, j) in connections:
                pl = self.topo.paths.get((self.topo.nodes[i], self.topo.nodes[j]), [])
                if not pl or k >= len(pl) or B[i, j, k] < req["bandwidth"]:
                    ok = False
                    break
            if ok:
                any_path_feasible = True
                break

        # --- Admission decision ------------------------------------------------
        over_admitted = False
        if admit:
            # Gather the chosen path for every logical connection.
            structurally_valid = True
            chosen_paths: list = []
            chosen_feasible = True
            for (i, j) in connections:
                path_list = self.topo.paths.get(
                    (self.topo.nodes[i], self.topo.nodes[j]), []
                )
                if not path_list or path_k >= len(path_list):
                    structurally_valid = False
                    break
                chosen_paths.append((path_list[path_k], req["bandwidth"]))
                if B[i, j, path_k] < req["bandwidth"]:
                    chosen_feasible = False

            # "hard": the chosen path must have headroom for every connection,
            # or the admission attempt fails outright (no reservation, reward
            # stays 0 — same as routing_error/missed_admission bookkeeping
            # below).  "soft": reserve regardless; avail may go negative.
            can_admit = structurally_valid and chosen_paths and (
                self.admission_mode == "soft" or chosen_feasible
            )
            if can_admit:
                for path, bw in chosen_paths:
                    self.topo.reserve(path, bw)
                # Slice entry is mutable so we can flag SLA violation later
                # (soft mode only — under hard mode this flag never flips).
                self.active_slices.append(
                    [req, chosen_paths, req["duration_steps"], req["bandwidth"], False]
                )
                rev = float(req["duration"] * req["price"])
                info["admitted"] = True
                info["admit_revenue"] = rev
                over_admitted = not chosen_feasible
                reward += 1.0 if self.reward_mode == "count" else rev

        # --- SLA violation cost (soft admission_mode only) ---------------------
        # A link is oversubscribed when its committed load exceeds capacity
        # (avail < 0) — structurally impossible under admission_mode="hard", so
        # this whole loop is skipped there.  Under "soft", any active slice
        # crossing such a link fails its SLA.  Charge is controlled by
        # self.sla_penalty_mode: "once" charges the full weight·frac·(d·p) the
        # first violating step (harsh → selective); "per_step" spreads it over
        # the lifetime (lenient → floody, per the 2026-07-04 experiment).  The
        # per-slice `violated` flag (s[4]) always tracks the first violation
        # for METRICS (fulfillment / sla rate count a slice once).
        new_violations = 0
        new_violations_inelastic = 0
        new_violations_elastic = 0
        if self.admission_mode == "soft":
            penalty = 0.0
            for s in self.active_slices:
                if self.sla_penalty_mode == "once" and s[4]:
                    continue
                oversub = any(
                    self.topo.avail.get(e, 0.0) < 0.0
                    for path, _ in s[1]
                    for e in path
                )
                if not oversub:
                    continue
                d_p = float(s[0]["duration"] * s[0]["price"])
                frac = 1.0 if s[0]["type"] == 0 else 0.5
                if self.sla_penalty_mode == "per_step":
                    steps = max(int(s[0]["duration_steps"]), 1)
                    penalty += self.penalty_weight * frac * d_p / steps
                else:
                    penalty += self.penalty_weight * frac * d_p
                if not s[4]:
                    s[4] = True
                    new_violations += 1
                    if s[0]["type"] == 0:
                        new_violations_inelastic += 1
                    else:
                        new_violations_elastic += 1
            if self.reward_mode == "revenue":
                reward -= penalty

        # --- Info / diagnostics ----------------------------------------------
        info["admit_attempt"] = admit
        info["any_path_feasible"] = any_path_feasible
        info["over_admitted"] = over_admitted
        info["new_violations"] = new_violations
        info["new_violations_inelastic"] = new_violations_inelastic
        info["new_violations_elastic"] = new_violations_elastic
        info["admit_type"] = int(req["type"]) if info["admitted"] else None
        info["missed_admission"] = bool(not admit and any_path_feasible)
        info["routing_error"] = bool(
            admit and not info["admitted"] and any_path_feasible
        )
        info["avg_path_utilization"] = self.topo.avg_link_utilization()

        # --- Tick active slices (age by one time-step) -----------------------
        new_active = []
        for s in self.active_slices:
            s[2] -= 1
            if s[2] <= 0:
                for path, bw in s[1]:
                    self.topo.release(path, bw)
            else:
                new_active.append(s)
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
            [
                req["type"],                               # {0, 1}
                req["duration"] / self._norm_duration,    # → [0, 1]
                req["bandwidth"] / self._norm_bw,         # → [0, 1]
                req["price"] / self._norm_price,          # → [0, 1]
            ],
            dtype=np.float32,
        )
        mt_flat = req["Mt"].flatten().astype(np.float32)  # binary, already [0, 1]
        n_inelastic = sum(1 for s in self.active_slices if s[0]["type"] == 0)
        n_elastic = sum(1 for s in self.active_slices if s[0]["type"] == 1)
        asl = np.array(
            [n_inelastic / self._norm_active, n_elastic / self._norm_active],
            dtype=np.float32,
        )
        B = self.topo.bottleneck_tensor()
        B_flat = (B / self._norm_cap).flatten()

        # Feasibility signal: for each of the K paths, the ratio of available
        # bottleneck capacity to the requested bandwidth across all connections
        # required by this slice.  A value ≥ 1 means path k is feasible.
        # Clamped to [0, 3] and normalised so the network gets a direct,
        # low-dimensional routing cue without having to decode 1323 B values.
        bw = req["bandwidth"]
        connections = [
            (i, j)
            for i in range(self.V)
            for j in range(self.V)
            if req["Mt"][i, j] == 1
        ]
        feasibility = np.zeros(self.K, dtype=np.float32)
        for k in range(self.K):
            if connections:
                bottleneck = float(min(B[i, j, k] for i, j in connections))
                feasibility[k] = np.clip(bottleneck / max(bw, 1e-6), 0.0, 3.0) / 3.0
            else:
                feasibility[k] = 1.0  # no connections required → always feasible

        return np.concatenate([rt_vec, mt_flat, asl, B_flat, feasibility])
