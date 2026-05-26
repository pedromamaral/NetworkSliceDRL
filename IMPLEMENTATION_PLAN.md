<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Ok so now I need to develop the experiment code. I am using VSCode (or other IDE that supports coding agents) I have remote GPU servers were I can run the experiements. Ideally the experiements shoud run in docker containers (that have access to GPU). I need you to prouduce an implementation plan in a format that can be used by me to feed an agentic software development workflow using Vscode and GitHubCopilot. The ideia is that the agent will code and run the experiments by remotely connecting to the GPU servers. I have two GPU servers that can be accessed via ssh with user name pedroamaral them are at IPs 10.26.110.15 and 10.26.110.14. Maybe beside the plan you can produce a boiler plate project also.

Here is a comprehensive implementation plan and overview of the boilerplate project. Since I hit the tool limit mid-generation, I'll deliver everything you need as structured content below — you can copy each file directly into your project.

***

# Network Slice DRL – Implementation \& Experiment Plan

> Feed each **Phase** to GitHub Copilot agent as an independent task. Complete phases sequentially before moving to the next.

***

## Repository Layout

```
netslice-drl/
├── .devcontainer/
│   └── devcontainer.json          ← VS Code Remote SSH + GPU config
├── configs/
│   ├── base.yaml                  ← shared defaults
│   ├── dqn1_unified.yaml
│   ├── dqn2_separated.yaml
│   ├── ddqn1_unified.yaml
│   └── ddqn2_separated.yaml
├── src/
│   ├── env/
│   │   ├── network_env.py         ← Gymnasium environment
│   │   ├── topology.py            ← graph, k-shortest paths, bottleneck tensor B
│   │   ├── slice_generator.py     ← Poisson/Pareto traffic generator
│   │   └── ovs_interface.py       ← optional OVS CLI wrapper
│   ├── agents/
│   │   ├── replay_buffer.py
│   │   ├── dqn_unified.py         ← DQN-1 (unified action space)
│   │   ├── dqn_separated.py       ← DQN-2 (two Q-networks)
│   │   ├── ddqn_unified.py        ← Duelling DQN-1
│   │   └── ddqn_separated.py      ← Duelling DQN-2
│   ├── baselines/
│   │   ├── greedy_admission.py
│   │   ├── revenue_heuristic.py
│   │   └── admission_only_dqn.py
│   └── utils/
│       ├── metrics.py
│       ├── logger.py              ← WandB/TensorBoard wrapper
│       └── checkpoint.py
├── experiments/
│   ├── run_experiment.py          ← Docker entry-point
│   └── eval_baselines.py
├── scripts/
│   ├── build_image.sh
│   ├── run_on_gpu14.sh            ← targets 10.26.110.14
│   ├── run_on_gpu15.sh            ← targets 10.26.110.15
│   └── sync_results.sh
├── notebooks/
│   └── analysis.ipynb
├── tests/
│   ├── test_env.py
│   ├── test_agents.py
│   └── test_topology.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```


***

## Phase 0 – Repository \& Remote Setup

**Copilot agent task:** initialise Git repo, create devcontainer config, verify GPU server connectivity.

### `.devcontainer/devcontainer.json`

```json
{
  "name": "netslice-drl",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "remoteUser": "pedroamaral",
  "features": {
    "ghcr.io/devcontainers/features/docker-outside-of-docker:1": {},
    "ghcr.io/devcontainers/features/nvidia-cuda:1": {}
  },
  "forwardPorts": [6006, 8888],
  "postCreateCommand": "pip install -r requirements.txt"
}
```


### `.gitignore`

```
__pycache__/
*.pt
*.pth
results/
wandb/
.env
*.egg-info/
dist/
```

**Acceptance criteria:**

- `ssh pedroamaral@10.26.110.14 nvidia-smi` returns GPU info
- `ssh pedroamaral@10.26.110.15 nvidia-smi` returns GPU info
- VS Code Remote SSH connects to both hosts successfully

***

## Phase 1 – Network Environment (`src/env/`)

**Copilot agent task:** implement the Gymnasium-compatible environment and all supporting modules.

### `src/env/topology.py`

```python
import networkx as nx
import numpy as np
import json
from itertools import islice

class NetworkTopology:
    def __init__(self, graph_file: str, k: int = 3):
        with open(graph_file) as f:
            data = json.load(f)
        self.G = nx.node_link_graph(data)
        self.k = k
        self.nodes = list(self.G.nodes)
        self.V = len(self.nodes)
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        # available capacity per edge (directed)
        self.avail = {e: self.G[e[^0]][e[^1]]['capacity_mbps']
                      for e in self.G.edges}
        self._precompute_paths()

    def _precompute_paths(self):
        self.paths = {}  # (i,j) -> list of k paths (each path = list of edges)
        for src in self.nodes:
            for dst in self.nodes:
                if src != dst:
                    gen = nx.shortest_simple_paths(self.G, src, dst, weight='weight')
                    node_paths = list(islice(gen, self.k))
                    self.paths[(src, dst)] = [
                        list(zip(p[:-1], p[1:])) for p in node_paths
                    ]

    def bottleneck_tensor(self) -> np.ndarray:
        """Returns B[i,j,k] = min available capacity on k-th path between i,j."""
        B = np.zeros((self.V, self.V, self.k))
        for (src, dst), path_list in self.paths.items():
            i, j = self.node_idx[src], self.node_idx[dst]
            for k_idx, path in enumerate(path_list):
                if path:
                    B[i, j, k_idx] = min(self.avail.get(e, 0) for e in path)
        return B

    def reserve(self, path_edges, bw: float):
        for e in path_edges:
            self.avail[e] -= bw

    def release(self, path_edges, bw: float):
        for e in path_edges:
            self.avail[e] = min(
                self.avail[e] + bw,
                self.G[e[^0]][e[^1]]['capacity_mbps']
            )
```


### `src/env/slice_generator.py`

```python
import numpy as np

class SliceGenerator:
    def __init__(self, cfg, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.V = cfg['num_nodes']

    def sample(self) -> dict:
        bw = self.rng.uniform(*self.cfg['bandwidth_range'])
        duration = int(self.rng.geometric(1.0 / self.cfg['slice_duration_mean']))
        price = bw * self.cfg['price_scale'] * (1 + 0.1 * self.rng.standard_normal())
        slice_type = int(self.rng.random() > self.cfg['inelastic_prob'])
        # sparse random connectivity matrix (1-3 logical connections)
        n_conn = self.rng.integers(1, 4)
        Mt = np.zeros((self.V, self.V), dtype=np.int8)
        for _ in range(n_conn):
            i, j = self.rng.choice(self.V, 2, replace=False)
            Mt[i, j] = 1
        return {
            'type': slice_type, 'duration': duration,
            'bandwidth': bw, 'price': max(price, 0.1), 'Mt': Mt
        }
```


### `src/env/network_env.py`

```python
import gymnasium as gym
import numpy as np
from .topology import NetworkTopology
from .slice_generator import SliceGenerator

class NetworkEnv(gym.Env):
    """
    State  = flatten(rt, Asl, B)
    Action = unified int {0..K}  OR  separated tuple (0|1, 1..K)
    Reward = 0 if reject; dt*pt - penalty if accept
    """
    def __init__(self, cfg: dict, mode: str = 'unified'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode  # 'unified' or 'separated'
        self.topo = NetworkTopology(cfg['topology_file'], cfg['k_shortest_paths'])
        self.K = cfg['k_shortest_paths']
        self.V = self.topo.V
        self.rng = np.random.default_rng(cfg.get('seed', 42))
        self.gen = SliceGenerator(cfg, self.rng)
        self.lam = cfg['arrival_rate']
        self.penalty_weight = cfg.get('penalty_weight', 0.5)

        # state dim: |rt| + 2 + V*V*K
        rt_dim = 2 + self.V * self.V  # type, duration, bw, price = 4; Mt flat = V*V
        self.state_dim = 4 + self.V * self.V + 2 + self.V * self.V * self.K
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        if mode == 'unified':
            self.action_space = gym.spaces.Discrete(self.K + 1)  # 0=reject
        else:
            self.action_space = gym.spaces.Tuple((
                gym.spaces.Discrete(2),
                gym.spaces.Discrete(self.K)
            ))

        self.active_slices = []  # list of (request, path_edges, remaining_steps, bw)
        self.current_request = None

    def _get_state(self):
        req = self.current_request
        rt_vec = np.array([req['type'], req['duration'],
                           req['bandwidth'], req['price']], dtype=np.float32)
        mt_flat = req['Mt'].flatten().astype(np.float32)
        asl = np.array([
            sum(1 for s in self.active_slices if s[^0]['type'] == 0),
            sum(1 for s in self.active_slices if s[^0]['type'] == 1)
        ], dtype=np.float32)
        B = self.topo.bottleneck_tensor().flatten().astype(np.float32)
        return np.concatenate([rt_vec, mt_flat, asl, B])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset topology capacities
        for e in self.topo.avail:
            self.topo.avail[e] = self.topo.G[e[^0]][e[^1]]['capacity_mbps']
        self.active_slices = []
        self.current_request = self.gen.sample()
        return self._get_state(), {}

    def step(self, action):
        req = self.current_request
        # decode action
        if self.mode == 'unified':
            admit = action > 0
            path_k = int(action) - 1 if admit else 0
        else:
            admit = bool(action[^0])
            path_k = int(action[^1])

        reward = 0.0
        info = {'admitted': False, 'sla_ok': True}

        if admit:
            # find a logical connection to route (simplified: first non-zero)
            connections = [(i, j) for i in range(self.V) for j in range(self.V)
                           if req['Mt'][i, j] == 1]
            feasible = True
            chosen_paths = []
            for (i, j) in connections:
                paths = self.topo.paths.get(
                    (self.topo.nodes[i], self.topo.nodes[j]), [])
                if not paths or path_k >= len(paths):
                    feasible = False; break
                path = paths[path_k]
                if self.topo.bottleneck_tensor()[i, j, path_k] >= req['bandwidth']:
                    chosen_paths.append((path, req['bandwidth']))
                else:
                    feasible = False; break

            if feasible:
                for path, bw in chosen_paths:
                    self.topo.reserve(path, bw)
                self.active_slices.append(
                    (req, chosen_paths, req['duration'], req['bandwidth']))
                reward = req['duration'] * req['price']
                info['admitted'] = True
            # else: cannot admit, treat as reject (reward=0)

        # tick active slices
        expired = []
        for idx, (s_req, s_paths, s_rem, s_bw) in enumerate(self.active_slices):
            new_rem = s_rem - 1
            if new_rem <= 0:
                for path, bw in s_paths:
                    self.topo.release(path, bw)
                expired.append(idx)
            else:
                self.active_slices[idx] = (s_req, s_paths, new_rem, s_bw)
        for idx in reversed(expired):
            self.active_slices.pop(idx)

        self.current_request = self.gen.sample()
        return self._get_state(), reward, False, False, info
```


***

## Phase 2 – DRL Agents (`src/agents/`)

### `src/agents/replay_buffer.py`

```python
import random, collections
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a),
                np.array(r, dtype=np.float32),
                np.array(s_next, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self): return len(self.buf)
```


### `src/agents/dqn_unified.py`

```python
import torch, torch.nn as nn, numpy as np
from .replay_buffer import ReplayBuffer

class MLP(nn.Module):
    def __init__(self, state_dim, hidden, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x): return self.net(x)

class DQNUnified:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q = MLP(state_dim, cfg.get('hidden_size', 256), action_dim).to(self.device)
        self.q_target = MLP(state_dim, cfg.get('hidden_size', 256), action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg['lr'])
        self.buf = ReplayBuffer(cfg['replay_capacity'])
        self.gamma = cfg['gamma']
        self.batch = cfg['batch_size']
        self.eps = cfg['epsilon_start']
        self.eps_end = cfg['epsilon_end']
        self.eps_decay = cfg['epsilon_decay_steps']
        self.action_dim = action_dim
        self.steps = 0

    def select_action(self, state):
        self.eps = max(self.eps_end,
                       self.eps - (self.eps - self.eps_end) / self.eps_decay)
        if np.random.random() < self.eps:
            return np.random.randint(self.action_dim)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q(s).argmax().item()

    def store(self, s, a, r, s_next, done):
        self.buf.push(s, a, r, s_next, done)

    def learn(self):
        if len(self.buf) < self.batch: return None
        s, a, r, s_next, done = [torch.FloatTensor(x).to(self.device)
                                  for x in self.buf.sample(self.batch)]
        a = a.long()
        q_val = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * (1 - done) * self.q_target(s_next).max(1)[^0]
        loss = nn.functional.mse_loss(q_val, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.steps += 1
        return loss.item()

    def update_target(self):
        self.q_target.load_state_dict(self.q.state_dict())
```


### `src/agents/ddqn_unified.py` (Duelling architecture)

```python
import torch, torch.nn as nn
from .dqn_unified import DQNUnified

class DuellingMLP(nn.Module):
    def __init__(self, state_dim, hidden, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU()
        )
        self.v_head = nn.Linear(hidden, 1)
        self.a_head = nn.Linear(hidden, action_dim)

    def forward(self, x):
        feat = self.backbone(x)
        V = self.v_head(feat)
        A = self.a_head(feat)
        return V + A - A.mean(dim=1, keepdim=True)

class DDQNUnified(DQNUnified):
    """Identical to DQNUnified but uses DuellingMLP."""
    def __init__(self, state_dim, action_dim, cfg):
        super().__init__(state_dim, action_dim, cfg)
        hidden = cfg.get('hidden_size', 256)
        self.q = DuellingMLP(state_dim, hidden, action_dim).to(self.device)
        self.q_target = DuellingMLP(state_dim, hidden, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg['lr'])
```


***

## Phase 3 – Baselines

### `src/baselines/greedy_admission.py`

```python
import numpy as np

class GreedyAdmission:
    """Accept all requests if capacity is available; route via shortest path (k=0)."""
    def select_action(self, state, mode='unified'):
        # Always try path 0 (shortest); env enforces feasibility
        if mode == 'unified':
            return 1  # accept, path index 1
        return (1, 0)
```


***

## Phase 4 – Docker \& Deployment

### `Dockerfile`

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    openvswitch-switch net-tools iproute2 rsync \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/workspace

ENTRYPOINT ["python", "experiments/run_experiment.py"]
```


### `docker-compose.yml`

```yaml
version: "3.9"

services:
  gpu14:
    build: .
    image: netslice-drl:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./results:/workspace/results
      - ./configs:/workspace/configs
      - ./data:/workspace/data
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
      - CUDA_VISIBLE_DEVICES=0
      - EXPERIMENT_CONFIG=${CONFIG:-configs/dqn1_unified.yaml}

  gpu15:
    extends:
      service: gpu14
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
      - CUDA_VISIBLE_DEVICES=0
      - EXPERIMENT_CONFIG=${CONFIG:-configs/ddqn1_unified.yaml}
```


### `scripts/run_on_gpu14.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
REMOTE="pedroamaral@10.26.110.14"
CONFIG=${1:-configs/dqn1_unified.yaml}

echo "[1/3] Syncing code to $REMOTE ..."
rsync -av --exclude='.git' --exclude='results/' --exclude='wandb/' \
      ./ $REMOTE:~/netslice-drl/

echo "[2/3] Building Docker image on $REMOTE ..."
ssh $REMOTE "cd ~/netslice-drl && docker build -t netslice-drl:latest . 2>&1"

echo "[3/3] Launching experiment: $CONFIG"
ssh $REMOTE "cd ~/netslice-drl && \
  docker run --gpus all --rm \
    -v \$(pwd)/results:/workspace/results \
    -v \$(pwd)/configs:/workspace/configs \
    -v \$(pwd)/data:/workspace/data \
    -e WANDB_API_KEY=\$WANDB_API_KEY \
    netslice-drl:latest --config $CONFIG"
echo "[✓] Done."
```


### `scripts/sync_results.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
for HOST in 10.26.110.14 10.26.110.15; do
  echo "Syncing from pedroamaral@$HOST ..."
  rsync -av pedroamaral@$HOST:~/netslice-drl/results/ ./results/
done
echo "[✓] All results synced."
```


***

## Phase 5 – Config Files

### `configs/base.yaml`

```yaml
topology_file: data/operator_topology.json
num_nodes: 12
k_shortest_paths: 3
arrival_rate: 5.0
slice_duration_mean: 10
bandwidth_range: [10, 100]
price_scale: 1.0
inelastic_prob: 0.5
num_episodes: 2000
max_steps_per_episode: 500
batch_size: 64
replay_capacity: 50000
gamma: 0.99
lr: 1.0e-4
hidden_size: 256
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_steps: 10000
target_update_freq: 100
penalty_weight: 0.5
eval_episodes: 200
eval_interval: 100
seed: 42
use_wandb: true
log_interval: 10
checkpoint_dir: results/checkpoints
results_dir: results
```

Each per-experiment config inherits from base and overrides only `agent:` and `seed:` — e.g.:

```yaml
# configs/ddqn2_separated.yaml
defaults:
  - base
agent: ddqn_separated
seed: 42
```


***

## Phase 6 – Experiment Runner

### `experiments/run_experiment.py`

```python
import argparse, yaml, os, numpy as np, torch
from src.env.network_env import NetworkEnv
from src.agents.dqn_unified import DQNUnified
from src.agents.dqn_separated import DQNSeparated
from src.agents.ddqn_unified import DDQNUnified
from src.agents.ddqn_separated import DDQNSeparated
from src.utils.metrics import MetricsTracker
from src.utils.logger import Logger

AGENT_MAP = {
    'dqn_unified': DQNUnified,
    'dqn_separated': DQNSeparated,
    'ddqn_unified': DDQNUnified,
    'ddqn_separated': DDQNSeparated,
}

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    mode = 'separated' if 'separated' in cfg['agent'] else 'unified'
    env = NetworkEnv(cfg, mode=mode)
    AgentClass = AGENT_MAP[cfg['agent']]
    agent = AgentClass(env.state_dim, env.action_space.n, cfg)
    metrics = MetricsTracker()
    logger = Logger(cfg)

    for ep in range(cfg['num_episodes']):
        state, _ = env.reset()
        ep_reward = 0
        for step in range(cfg['max_steps_per_episode']):
            action = agent.select_action(state)
            next_state, reward, done, trunc, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()
            metrics.update(reward, info)
            ep_reward += reward
            state = next_state
            if done or trunc:
                break

        if ep % cfg['target_update_freq'] == 0:
            agent.update_target()

        if ep % cfg['eval_interval'] == 0:
            summary = metrics.summarise()
            logger.log(ep, summary, loss)
            metrics.reset()
            print(f"Ep {ep:4d} | reward={ep_reward:.1f} | "
                  f"accept={summary['acceptance_ratio']:.3f} | "
                  f"fulfill={summary['fulfillment_ratio']:.3f}")

    logger.save_csv()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    main(p.parse_args().config)
```


***

## Phase 7 – Experiment Execution Plan

### Server assignment (run in parallel)

| Server | Experiments |
| :-- | :-- |
| `10.26.110.14` | DQN-1 unified, DQN-2 separated, greedy baseline |
| `10.26.110.15` | DDQN-1, DDQN-2, revenue heuristic, admission-only DQN |

### For statistical validity (required by reviewers)

- Run each agent with **5 seeds** (42–46) → 20 DRL runs total
- Report **mean ± std** for all KPIs
- Use **Wilcoxon signed-rank test** to confirm significance of the 47.5% / 18% gains
- Each seed is a separate `docker run` call; automate with a loop in the shell scripts

***

## Phase 8 – Metrics

### `src/utils/metrics.py`

```python
class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total, self.admitted, self.fulfilled = 0, 0, 0
        self.revenue, self.sla_violations = 0.0, 0

    def update(self, reward, info):
        self.total += 1
        if info.get('admitted'):
            self.admitted += 1
            self.revenue += reward
            if not info.get('sla_ok', True):
                self.sla_violations += 1
            else:
                self.fulfilled += 1

    def summarise(self):
        return {
            'acceptance_ratio':   self.admitted / max(self.total, 1),
            'fulfillment_ratio':  self.fulfilled / max(self.admitted, 1),
            'revenue':            self.revenue,
            'sla_violation_rate': self.sla_violations / max(self.admitted, 1),
        }
```


***

## Phase 9 – Requirements File

### `requirements.txt`

```
torch>=2.2.0
gymnasium>=0.29.0
networkx>=3.2
numpy>=1.26
pandas>=2.1
pyyaml>=6.0
wandb>=0.16
matplotlib>=3.8
seaborn>=0.13
scipy>=1.11
pytest>=7.4
tqdm>=4.66
```


***

## Copilot Agent Prompt Template

Use this exact prompt to feed each phase to GitHub Copilot:

```
Context: This is a DRL-based network slice admission control and resource
allocation project for an IP/MPLS core network (IEEE journal paper implementation).
Repository layout and specifications are in IMPLEMENTATION_PLAN.md.

Task: Implement Phase <N> – <Phase Name>.
Acceptance criteria are at the end of the phase section.

Constraints:
- PyTorch for all neural networks
- Gymnasium API for the environment
- NetworkX for graph operations
- PyYAML for config loading
- All training must run inside Docker containers with GPU access

After writing the code, run the acceptance criteria tests and fix any failures
before considering the phase complete.

File(s) to create: <filename(s)>
```


***

## Summary of What to Do First

1. **Create the repo** and paste this plan as `IMPLEMENTATION_PLAN.md`
2. **Open VS Code**, install the Remote SSH and Dev Containers extensions, connect to `10.26.110.15`
3. **Feed Phase 0** to Copilot agent to verify GPU server connectivity
4. **Feed Phase 1** to implement the environment — this is the foundation everything else depends on
5. **Provide your real operator topology** as `data/operator_topology.json` (NetworkX node-link format) before Phase 1 can complete
6. Run phases **in order** up to Phase 4, then trigger Phase 7 (experiments) across both GPU servers in parallel
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://github.com/orgs/community/discussions/52369

[^2]: https://github.com/microsoft/vscode-copilot-release/issues/11526

[^3]: https://github.com/microsoft/vscode/issues/259811

[^4]: https://code.visualstudio.com/blogs/2025/02/24/introducing-copilot-agent-mode

[^5]: https://github.com/microsoft/vscode/issues/256810

[^6]: https://pathofengineer.com/posts/containerization-of-cuda-environment-using-vscode/

[^7]: https://arxiv.org/html/2412.08293v1

[^8]: https://www.youtube.com/watch?v=uXOHOpqhkng

[^9]: https://docs.nvidia.com/infra-controller/documentation/development/visual-studio-code-remote-development

[^10]: https://github.com/rapariciopardo/PRISMA

[^11]: https://github.com/orgs/community/discussions/6942

[^12]: https://a-nau.github.io/blog/2024-03-17-VS_Code_Devcontainer/

[^13]: https://docs.ovn.org/en/stable/howto/docker.html

[^14]: https://github.blog/news-insights/product-news/github-copilot-agent-mode-activated/

[^15]: https://github.com/chunhokennethkong/tensorflow-gpu-container-demo

[^16]: NetworkSlicing.tex

