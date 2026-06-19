#!/usr/bin/env bash
# Run the full 25-job sweep (4 agents × 5 seeds + 5 baseline evals) on a single GPU server.
#
# Usage:
#   bash scripts/run_sweep_single_gpu.sh <user@host>
#   bash scripts/run_sweep_single_gpu.sh pedroamaral@10.26.110.15
#
# All jobs run sequentially — one GPU, one job at a time.
# Expected wall time: ~40–50 hrs for DRL runs + ~3 hrs for baselines.
# Run inside tmux/screen on the server so a dropped SSH session doesn't kill it.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <user@host>"
    exit 1
fi

REMOTE="$1"
REMOTE_DIR="~/netslice-drl"
SEEDS=(42 43 44 45 46)
DRL_CONFIGS=(
    configs/dqn_unified.yaml
    configs/dqn_separated.yaml
    configs/ddqn_unified.yaml
    configs/ddqn_separated.yaml
)

# ---------------------------------------------------------------------------
echo "=== [1/3] Pulling latest code on $REMOTE:$REMOTE_DIR ==="
ssh "$REMOTE" "cd $REMOTE_DIR && git pull"

echo ""
echo "=== [2/3] Building Docker image on $REMOTE ==="
ssh "$REMOTE" "cd $REMOTE_DIR && docker build -t netslice-drl:latest . 2>&1"

mkdir -p results   # ensure local results dir exists for later sync

# ---------------------------------------------------------------------------
echo ""
echo "=== [3/3] Running 25 jobs sequentially ==="
echo "    4 agents × 5 seeds = 20 DRL runs"
echo "    5 baseline evals"
echo ""

TOTAL=$((${#DRL_CONFIGS[@]} * ${#SEEDS[@]} + ${#SEEDS[@]}))
JOB=0

# --- DRL training runs ------------------------------------------------------
for CFG in "${DRL_CONFIGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB=$((JOB + 1))
        RUN_NAME=$(basename "$CFG" .yaml)_s${SEED}
        echo "--- Job $JOB/$TOTAL: $RUN_NAME ---"
        ssh "$REMOTE" "mkdir -p $REMOTE_DIR/results && \
            cd $REMOTE_DIR && \
            docker run --gpus all --rm \
              -v \$(pwd)/results:/workspace/results \
              -v \$(pwd)/configs:/workspace/configs \
              -v \$(pwd)/data:/workspace/data \
              netslice-drl:latest \
              --config $CFG --seed $SEED \
            2>&1 | tee -a $REMOTE_DIR/results/${RUN_NAME}.log"
        echo "    ✓ $RUN_NAME done"
    done
done

# --- Baseline evaluations (CPU-only, no --gpus flag needed) -----------------
for SEED in "${SEEDS[@]}"; do
    JOB=$((JOB + 1))
    RUN_NAME="baselines_s${SEED}"
    echo "--- Job $JOB/$TOTAL: $RUN_NAME ---"
    ssh "$REMOTE" "cd $REMOTE_DIR && \
        docker run --rm \
          -v \$(pwd)/results:/workspace/results \
          -v \$(pwd)/configs:/workspace/configs \
          -v \$(pwd)/data:/workspace/data \
          --entrypoint python \
          netslice-drl:latest \
          experiments/eval_baselines.py --seed $SEED \
        2>&1 | tee -a $REMOTE_DIR/results/${RUN_NAME}.log"
    echo "    ✓ $RUN_NAME done"
done

echo ""
echo "=== All $TOTAL jobs complete. Run sync_results.sh to pull results. ==="
