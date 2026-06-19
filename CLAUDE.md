# CLAUDE.md — netslice-drl Project Context

This file is the primary context document for Claude Code working on this repository.
Read it fully before making any changes. It describes the research goal, the mathematical
model, the codebase architecture, the infrastructure, and the exact rules that must be
respected during development.

---

## 1. What This Project Is

This is the **implementation and experimentation codebase** for an IEEE journal paper:

> *"Deep Reinforcement Learning based Admission Control and Network Resource Allocation
> for Core Network Slicing — feasibility and performance"*
> Pedro Amaral, FCT-UNL / Instituto de Telecomunicações

The paper proposes a DRL agent that **jointly** solves:
1. **Slice Admission Control (AC)** — accept or reject an incoming network slice request
2. **Resource Allocation (RA)** — select a routing path and reserve bandwidth on the
   IP/MPLS core network for admitted slices

The key novelty claims that experiments **must support**:
- DRL agents accept **47.5% more slices** than baseline methods
- Joint AC+RA achieves **18% more fulfilled slices** than AC-only approaches
- The state uses **only OSS/telemetry-available counters** (no deep packet inspection,
  no per-flow state unavailable in production)

---

## 2. Mathematical Model (Read Before Touching Any Code)

### 2.1 Network Model

The core network is a directed graph **G = (V, E)** where each link `e ∈ E` has
capacity `C_e` in Mbps.

### 2.2 Slice Request

A slice request arriving at time `t` is the tuple:

```
r_t = (τ_t, d_t, b_t, p_t, M_t)
```

| Symbol | Meaning |
|--------|---------|
| `τ_t` | Slice type: 0 = inelastic, 1 = elastic |
| `d_t` | Requested lifetime in time slots |
| `b_t` | Bandwidth demand per logical connection (Mbps) |
| `p_t` | Price offered by the tenant |
| `M_t` | V×V logical connectivity matrix (M_t[i,j]=1 means a logical link i→j is required) |

### 2.3 MDP State

```
s_t = (r_t, A_sl, B)
```

| Component | Description |
|-----------|-------------|
| `r_t` | Current slice request (flattened: τ, d, b, p, M_t flat → 4 + V² scalars) |
| `A_sl` | [n_inelastic, n_elastic] — count of active admitted slices by type |
| `B` | V×V×K bottleneck tensor: B[i,j,k] = min available capacity on k-th shortest path between nodes i and j |

**State vector size** = 4 + V² + 2 + V²×K (flattened to 1D float32)

### 2.4 Action Spaces

**DQN-1 / DDQN-1 (Unified action space):**
```
A_unified = {0, 1, 2, ..., K}
  0     → Reject slice
  1..K  → Accept slice and route via path k
```

**DQN-2 / DDQN-2 (Separated action space):**
```
A_admission = {0=reject, 1=accept}
A_routing   = {1, 2, ..., K}  (path index)
```
Two independent Q-networks; routing network is only queried if admission=1.

### 2.5 Reward

```
R(s, a) =
  0                      if slice rejected
  d_t × p_t + P(s,a)    if slice accepted
```

Performance penalty P(s,a) applied during slice lifetime:
```
P(s,a) =
  0          if bandwidth SLA is met throughout slice duration
  -p_t/2     if bandwidth NOT met AND slice is elastic (τ=1)
  -p_t       if bandwidth NOT met AND slice is inelastic (τ=0)
```

`λ` (penalty_weight in config) scales the penalty term.

### 2.6 Capacity Constraint

For every link `e ∈ E`, total reserved bandwidth must not exceed capacity:
```
Σ_{s ∈ S_active} Σ_{(i,j): M_s[i,j]=1} b_s × δ_{s,i,j,e} ≤ C_e
```

### 2.7 Duelling Architecture Decomposition

```
Q(s,a; θ,α,β) = V(s; θ,β) + A(s,a; θ,α) - (1/|A|) Σ_a' A(s,a'; θ,α)
```

Both unified and separated variants use this decomposition. The separated variant has
two independent duelling networks (one for admission, one for routing).

---

## 3. Codebase Architecture

```
netslice-drl/
├── CLAUDE.md                    ← THIS FILE
├── IMPLEMENTATION_PLAN.md       ← Phase-by-phase build plan
├── configs/
│   ├── base.yaml                ← Shared defaults (source of truth for hyperparams)
│   ├── dqn1_unified.yaml
│   ├── dqn2_separated.yaml
│   ├── ddqn1_unified.yaml
│   └── ddqn2_separated.yaml
├── src/
│   ├── env/
│   │   ├── network_env.py       ← Gymnasium env (core loop)
│   │   ├── topology.py          ← NetworkTopology class + bottleneck tensor
│   │   ├── slice_generator.py   ← Poisson/Pareto slice request generator
│   │   └── ovs_interface.py     ← Optional OVS CLI wrapper (not needed for training)
│   ├── agents/
│   │   ├── replay_buffer.py     ← Experience replay (capacity 50k default)
│   │   ├── dqn_unified.py       ← DQN-1: single MLP, unified action space
│   │   ├── dqn_separated.py     ← DQN-2: two MLPs, separated spaces
│   │   ├── ddqn_unified.py      ← DDQN-1: duelling MLP, unified
│   │   └── ddqn_separated.py    ← DDQN-2: two duelling MLPs, separated
│   ├── baselines/
│   │   ├── greedy_admission.py  ← Always admit if capacity exists, path k=0
│   │   ├── revenue_heuristic.py ← Admit if p_t > threshold, shortest path
│   │   └── admission_only_dqn.py← DQN that only learns admission (no path selection)
│   └── utils/
│       ├── metrics.py           ← MetricsTracker
│       ├── logger.py            ← WandB + CSV logger
│       └── checkpoint.py        ← Save/load agent checkpoints
├── experiments/
│   ├── run_experiment.py        ← Docker ENTRYPOINT; reads --config, runs training loop
│   └── eval_baselines.py        ← Evaluate all baselines on fixed traffic seeds
├── data/
│   └── operator_topology.json   ← Real operator topology in NetworkX node-link format
├── scripts/
│   ├── build_image.sh
│   ├── run_on_gpu14.sh          ← rsync + docker run on 10.26.110.14
│   ├── run_on_gpu15.sh          ← rsync + docker run on 10.26.110.15
│   └── sync_results.sh          ← Pull results/ from both servers
├── tests/
│   ├── test_env.py
│   ├── test_agents.py
│   └── test_topology.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 4. Key Implementation Rules

### 4.1 Environment Rules (DO NOT VIOLATE)

- `network_env.py` must implement the **Gymnasium** API exactly:
  `reset() → (obs, info)`, `step(action) → (obs, reward, terminated, truncated, info)`
- The state vector **must** be composed of exactly: `[τ, d, b, p, M_t_flat, n_inelastic, n_elastic, B_flat]`
  — in that order. Changing order invalidates all agent checkpoints.
- Slice duration is ticked **every step** (one step = one slice arrival event).
- When a slice expires, bandwidth is **released** back to `topology.avail`.
- If admission=1 but capacity is insufficient for the chosen path, the action is
  treated as a **forced reject** (reward=0, no capacity reserved). This is by design.
- The `mode` parameter ('unified' or 'separated') controls `action_space` type but
  not the state. State is identical for all four agent variants.

### 4.2 Agent Rules

- All four agents must use **identical hyperparameters** from `base.yaml` unless
  a per-config override is present.
- Target network update: **hard copy** every `target_update_freq` steps (not soft update).
- Replay buffer: sample only when `len(buffer) >= batch_size`.
- Epsilon decay: **linear** from `epsilon_start` to `epsilon_end` over `epsilon_decay_steps`.
- For DQN-2 / DDQN-2: routing network is only updated when `action_admission=1`.
  Routing transitions are stored in a **separate** replay buffer `D_rt`.

### 4.3 Metrics (These Are What the Paper Reports)

`MetricsTracker` must compute and expose:

| Metric | Definition |
|--------|-----------|
| `acceptance_ratio` | admitted / total_arrivals |
| `fulfillment_ratio` | fulfilled / admitted (fulfilled = admitted AND SLA met throughout lifetime) |
| `revenue` | Σ(d_t × p_t) for admitted slices |
| `sla_violation_rate` | sla_violations / admitted |
| `avg_path_utilization` | mean over all links: reserved / capacity |

The paper claims (higher % of accepted amd fulfilled vs baselines) must be
reproduced. If experiments show otherwise, **do not change metric definitions** —
investigate environment or agent bugs first.

### 4.4 Reproducibility Requirements

- Every run must log the seed used.
- Training and evaluation must be **deterministic given the same seed**:
  `np.random.default_rng(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed(seed)`
- Run each agent × 5 seeds (42, 43, 44, 45, 46) → 20 DRL training runs total.
- Report **mean ± std** across seeds for all metrics.

### 4.5 Baseline Comparison Rules

Baselines must be evaluated on the **exact same traffic sequences** as DRL agents.
Use `np.random.default_rng(seed)` with the same seeds. Do not generate fresh traffic
for baselines.

The three baselines:
1. **Greedy** — admit if any of the K paths has sufficient capacity; use the path
   with maximum bottleneck capacity; never reject a feasible request.
2. **Revenue heuristic** — admit only if `p_t > revenue_threshold` AND capacity
   is available; use shortest path.
3. **AC-only DQN** — DQN that learns admission (accept/reject) but always routes
   via path k=0. This isolates the contribution of joint RA.

---

## 5. Infrastructure

### 5.1 GPU Servers

| Server | IP | User | Assigned Experiments |
|--------|-----|------|---------------------|
| gpu14 | 10.26.110.14 | pedroamaral | DQN-1, DQN-2, greedy baseline |
| gpu15 | 10.26.110.15 | pedroamaral | DDQN-1, DDQN-2, revenue heuristic, AC-only DQN |

### 5.2 Docker Workflow

```bash
# Sync code to server
rsync -av --exclude='.git' --exclude='results/' ./ pedroamaral@10.26.110.15:~/netslice-drl/

# Build image on server
ssh pedroamaral@10.26.110.15 "cd ~/netslice-drl && docker build -t netslice-drl:latest ."

# Run experiment
ssh pedroamaral@10.26.110.15 \
  "cd ~/netslice-drl && docker run --gpus all --rm \
   -v \$(pwd)/results:/workspace/results \
   -v \$(pwd)/data:/workspace/data \
   -e WANDB_API_KEY=\$WANDB_API_KEY \
   netslice-drl:latest --config configs/ddqn1_unified.yaml --seed 42"

# Retrieve results
rsync -av pedroamaral@10.26.110.15:~/netslice-drl/results/ ./results/
```

---

## 6. Config System

All configs inherit from `base.yaml`. Per-experiment configs only override what differs.

### Critical base.yaml parameters

```yaml
num_nodes: 21               # Endpoint nodes: 7 BS + 7 CS + 7 MECS (core/dist S1-S65 are routing only)
k_shortest_paths: 3         # K — controls action space size and B tensor depth
arrival_rate: 5.0           # λ for Poisson arrivals (slices/time_slot)
slice_duration_mean: 10
bandwidth_range: [10, 100]  # Mbps
inelastic_prob: 0.5         # Fraction of inelastic (τ=0) slices

num_episodes: 2000
max_steps_per_episode: 500
gamma: 0.99
lr: 1.0e-4
hidden_size: 256
batch_size: 64
replay_capacity: 50000
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_steps: 10000
target_update_freq: 100

eval_episodes: 200
eval_interval: 100
seed: 42
```

**Do not change these defaults without discussing the implications for the paper claims.**

---

## 7. Topology File Format

`data/operator_topology.json` must be a **NetworkX node-link JSON**:

```json
{
  "directed": true,
  "multigraph": false,
  "graph": {},
  "nodes": [{"id": "R1"}, {"id": "R2"}, ...],
  "links": [
    {
      "source": "R1",
      "target": "R2",
      "capacity_mbps": 1000.0,
      "weight": 1.0
    }
  ]
}
```

**Do not mock or generate a fake topology** — use the real operator topology data.
If the file is missing, the environment will fail at startup with a clear error.

---

## 8. Testing Requirements

```bash
cd /workspace && python -m pytest tests/ -v
```

### test_env.py must verify:
- State vector has correct shape `(4 + V² + 2 + V²×K,)`
- `reset()` returns state within valid ranges
- `step()` with action=0 (reject) always returns reward=0
- Capacity constraint is never violated after `step()`
- Expired slices release bandwidth correctly
- Both modes ('unified', 'separated') produce valid action spaces

### test_agents.py must verify:
- `select_action()` on a random state returns valid action without error
- `learn()` returns a loss value after buffer has ≥ batch_size transitions
- Target network update does not change main network weights
- Checkpoint save/load round-trip preserves Q-values

### test_topology.py must verify:
- K-shortest paths are pre-computed for all (i,j) pairs
- Bottleneck tensor shape is `(V, V, K)`
- `reserve()` decreases `avail` correctly
- `release()` does not exceed original capacity

---

## 9. Results Structure

```
results/
├── {agent}_{seed}/
│   ├── training_log.csv        ← episode, reward, acceptance_ratio, fulfillment_ratio, loss
│   ├── eval_log.csv            ← eval metrics at each eval_interval
│   └── checkpoints/
│       └── ep_{N}.pt
└── baselines/
    ├── greedy_seed{N}.csv
    ├── revenue_heuristic_seed{N}.csv
    └── ac_only_dqn_seed{N}.csv
```

WandB project name: `netslice-drl`. Run name format: `{agent}_{seed}`. Group by agent.

---

## 10. Current Implementation Status

### Done (boilerplate from IMPLEMENTATION_PLAN.md):
- [x] Repository structure and directory scaffolding
- [x] `base.yaml` and per-experiment configs
- [x] `Dockerfile` and `docker-compose.yml`
- [x] Deployment scripts in `scripts/`
- [x] Initial skeleton for all `src/` modules

### Needs verification or completion:
- [ ] `topology.py` — verify K-shortest path pre-computation is complete
- [ ] `network_env.py` — verify capacity constraint enforcement and slice expiry
- [ ] `dqn_separated.py` — dual-buffer, dual-network logic
- [ ] `ddqn_separated.py` — same but with duelling heads
- [ ] `baselines/revenue_heuristic.py` — needs threshold calibration
- [ ] `baselines/admission_only_dqn.py` — AC-only DQN (routes via fixed k=0)
- [ ] `utils/metrics.py` — `avg_path_utilization` not yet implemented
- [ ] `experiments/eval_baselines.py` — baseline evaluation runner
- [ ] `data/operator_topology.json` — **must be provided by researcher**
- [ ] End-to-end smoke test (train DQN-1 for 10 episodes, confirm no crashes)
- [ ] WandB logger integration in `run_experiment.py`

---

## 11. Common Pitfalls

1. **Use `gymnasium` not `gym`** — the old API has a 4-tuple `step()` return;
   `gymnasium` returns 5-tuple `(obs, reward, terminated, truncated, info)`.

2. **Separated mode action space is a `Tuple`, not `Discrete`** — do not call
   `action_space.n` in separated mode. Always check `env.mode` first.

3. **Recompute `B` every step** — `B[i,j,k]` changes as slices are admitted and
   expire. Cache only the K-shortest path *edges*, not the capacity values.

4. **Check all logical connections before reserving** — verify `B[i,j,k] >= b_t`
   for ALL (i,j) pairs where `M_t[i,j]=1` before calling `topology.reserve()`.
   Never reserve partial connections.

5. **Reward is given at admission time** — the full `d_t × p_t` reward is given
   immediately. SLA penalties accumulate during slice lifetime and are subtracted.

6. **Topology edge keys are tuples** — `topology.avail` keys are `(source, target)`
   node ID tuples. If topology JSON uses string node IDs, path edges must also
   use string IDs.

7. **DQN-2 routing buffer only gets updates on admitted slices** — if admission=0,
   do not store the routing transition.

---

## 12. Paper Section → Code Mapping

| Paper Section | Code Location |
|---------------|--------------|
| §III Problem Formulation | `src/env/network_env.py`, `src/env/topology.py` |
| §III-B MDP State (s_t) | `NetworkEnv._get_state()` |
| §III-C Action Space | `NetworkEnv.__init__()`, action_space definitions |
| §III-D Reward Function | `NetworkEnv.step()` reward computation |
| §IV-A DQN Model | `src/agents/dqn_unified.py`, `dqn_separated.py` |
| §IV-B Duelling DQN | `src/agents/ddqn_unified.py`, `ddqn_separated.py` |
| §IV-C Loss Function | `agent.learn()` in all four agent files |
| §IV-D Training Algorithm | `experiments/run_experiment.py` main loop |
| §V Simulation Framework | `src/env/`, `Dockerfile`, `scripts/` |
| §V Baselines | `src/baselines/` |
| §VI Results | `notebooks/analysis.ipynb`, `results/` |

---

## 13. Quick Local Smoke Test (No Docker)

```bash
pip install -r requirements.txt
python experiments/run_experiment.py \
  --config configs/dqn1_unified.yaml \
  --seed 42 \
  --num_episodes 10 \
  --use_wandb false
```

This should complete in under 2 minutes on CPU. If it crashes, fix the bug before
running on the GPU servers.

---

## 14. Authority

- **Researcher**: Pedro Amaral (pfaf@fct.unl.pt)
- **Institution**: FCT-UNL / Instituto de Telecomunicações
- **Any change to the MDP formulation, state space, or reward function must be
  discussed with the researcher before implementation** — these directly affect
  the paper's claims.
