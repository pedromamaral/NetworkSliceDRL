<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>



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



## Phase 1 – Docker \& Deployment

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

