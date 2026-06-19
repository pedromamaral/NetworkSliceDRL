#!/usr/bin/env bash
# Launch a single experiment on GPU server 10.26.110.15 (primary).
# Usage: bash scripts/run_on_gpu15.sh [config] [seed]
#   config  path to YAML relative to repo root  (default: configs/ddqn_unified.yaml)
#   seed    integer seed override                (default: use value in config)
#
# Examples:
#   bash scripts/run_on_gpu15.sh configs/ddqn_separated.yaml 44
#   bash scripts/run_on_gpu15.sh                                   # uses defaults
set -euo pipefail

REMOTE="pedroamaral@10.26.110.15"
REMOTE_DIR="~/netslice-drl"
CONFIG="${1:-configs/ddqn_unified.yaml}"
SEED="${2:-}"

echo "=== [1/4] Syncing workspace to $REMOTE:$REMOTE_DIR ==="
rsync -av --delete \
      --exclude='.git' \
      --exclude='results/' \
      --exclude='wandb/' \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      ./ "$REMOTE:$REMOTE_DIR/"

echo ""
echo "=== [2/4] Building Docker image on $REMOTE ==="
ssh "$REMOTE" "cd $REMOTE_DIR && docker build -t netslice-drl:latest . 2>&1"

echo ""
echo "=== [3/4] Launching experiment: $CONFIG (seed=${SEED:-from-config}) ==="

SEED_ARG=""
if [[ -n "$SEED" ]]; then
    SEED_ARG="--seed $SEED"
fi

ssh "$REMOTE" "cd $REMOTE_DIR && \
  docker run --gpus all --rm \
    -v \$(pwd)/results:/workspace/results \
    -v \$(pwd)/configs:/workspace/configs \
    -v \$(pwd)/data:/workspace/data \
    -e WANDB_API_KEY=\${WANDB_API_KEY:-} \
    netslice-drl:latest \
    --config $CONFIG $SEED_ARG 2>&1"

echo ""
echo "=== [4/4] Done — results available at results/ on $REMOTE ==="
