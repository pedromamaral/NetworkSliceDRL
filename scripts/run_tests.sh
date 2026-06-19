#!/usr/bin/env bash
# Run the full pytest suite inside a Docker container on the remote GPU server.
# Usage: bash scripts/run_tests.sh [host]
#   host defaults to pedroamaral@10.26.110.14
set -euo pipefail

REMOTE="${1:-pedroamaral@10.26.110.15}"
REMOTE_DIR="~/netslice-drl"

echo "=== [1/3] Syncing workspace to $REMOTE:$REMOTE_DIR ==="
rsync -av --delete \
      --exclude='.git' \
      --exclude='results/' \
      --exclude='wandb/' \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      ./ "$REMOTE:$REMOTE_DIR/"

echo ""
echo "=== [2/3] Building Docker image on $REMOTE ==="
ssh "$REMOTE" "cd $REMOTE_DIR && docker build -t netslice-drl:latest . 2>&1"

echo ""
echo "=== [3/3] Running pytest inside container ==="
ssh "$REMOTE" "cd $REMOTE_DIR && \
  docker run --rm \
    -v \$(pwd)/data:/workspace/data \
    netslice-drl:latest \
    python -m pytest tests/ -v --tb=short 2>&1"
