#!/bin/bash
# Phase-2 scaling probe: unified vs separated DDQN at K=8, seeds 42-43,
# count/hard. Sequential (shared box; concurrent runs starve sshd).
# Also evaluates greedy@K=8 once as the reference (no training).
set -u
cd ~/netslice-drl

MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"

for S in 42 43; do
  echo "=== [$(date +%H:%M:%S)] seed $S : unified K=8 ==="
  docker run --gpus all --rm $MOUNTS netslice-drl:latest \
    --config configs/ddqn_unified_count_k8.yaml --seed $S \
    > results/sweep_k8_unified_s${S}.log 2>&1
  echo "=== [$(date +%H:%M:%S)] seed $S : separated K=8 ==="
  docker run --gpus all --rm $MOUNTS netslice-drl:latest \
    --config configs/ddqn_separated_count_k8.yaml --seed $S \
    > results/sweep_k8_separated_s${S}.log 2>&1
done

# Greedy reference at K=8 (held-out 142), via eval_baselines (train_episodes=0
# skips AC-only training cost is not supported, so run greedy/revenue only by
# a short AC-only train — cheap enough at 50 eps just for the reference row).
echo "=== [$(date +%H:%M:%S)] baselines K=8 (greedy reference) ==="
docker run --gpus all --rm --entrypoint python $MOUNTS netslice-drl:latest \
  experiments/eval_baselines.py --config configs/ddqn_unified_count_k8.yaml \
  --seed 42 --train_episodes 50 --eval_episodes 200 \
  > results/sweep_k8_baselines_s42.log 2>&1

echo "=== [$(date +%H:%M:%S)] K8_SWEEP_COMPLETE ==="
