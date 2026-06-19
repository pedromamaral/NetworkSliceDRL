#!/usr/bin/env bash
# Run a 5-seed sweep for DQN variants on GPU server 10.26.110.14.
#
# Server assignment (Phase 7):
#   GPU14 → dqn_unified, dqn_separated
#   GPU15 → ddqn_unified, ddqn_separated  (see run_sweep_gpu15.sh)
#
# Usage:
#   bash scripts/run_sweep_gpu14.sh                  # seeds 42-46, both DQN configs
#   bash scripts/run_sweep_gpu14.sh 42 43             # custom seed list
#   bash scripts/run_sweep_gpu14.sh --eval-baselines  # only run baseline evaluation

set -euo pipefail

REMOTE="pedroamaral@10.26.110.14"
SEEDS=(42 43 44 45 46)
CONFIGS=(configs/dqn_unified.yaml configs/dqn_separated.yaml)
EVAL_BASELINES=false

# Parse optional flags / custom seed list
if [[ "${1:-}" == "--eval-baselines" ]]; then
    EVAL_BASELINES=true
    shift
fi
if [[ $# -gt 0 ]]; then
    SEEDS=("$@")
fi

# ---------------------------------------------------------------------------
echo "=== [1/3] Syncing workspace to $REMOTE:~/netslice-drl ==="
rsync -av --exclude='.git' --exclude='results/' --exclude='wandb/' \
      ./ "$REMOTE:~/netslice-drl/"

echo "=== [2/3] Building Docker image on $REMOTE ==="
ssh "$REMOTE" "cd ~/netslice-drl && docker build -t netslice-drl:latest . 2>&1"

# ---------------------------------------------------------------------------
echo "=== [3/3] Launching DQN sweep on $REMOTE (sequential — single GPU) ==="
for CFG in "${CONFIGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME=$(basename "$CFG" .yaml)_s${SEED}
        echo "  → $RUN_NAME"
        ssh "$REMOTE" "cd ~/netslice-drl && \
            docker run --gpus all --rm \
              -v \$(pwd)/results:/workspace/results \
              -v \$(pwd)/configs:/workspace/configs \
              -v \$(pwd)/data:/workspace/data \
              -e WANDB_API_KEY=\${WANDB_API_KEY:-} \
              netslice-drl:latest \
              --config $CFG --seed $SEED \
            2>&1 | tee -a ~/netslice-drl/results/${RUN_NAME}.log"
    done
done

if $EVAL_BASELINES; then
    echo "  → baselines sweep (sequential)"
    for SEED in "${SEEDS[@]}"; do
        ssh "$REMOTE" "cd ~/netslice-drl && \
            docker run --rm \
              -v \$(pwd)/results:/workspace/results \
              -v \$(pwd)/configs:/workspace/configs \
              -v \$(pwd)/data:/workspace/data \
              --entrypoint python \
              netslice-drl:latest \
              experiments/eval_baselines.py --seed $SEED \
            2>&1 | tee -a ~/netslice-drl/results/baselines_s${SEED}.log"
    done
fi

echo "[✓] All GPU14 jobs finished."
