#!/bin/bash
# Action-masking test: unified DDQN with feasibility masking, count/hard, K=3,
# 5 seeds. Joint agent only — the K=3 baselines (greedy/AC-only/revenue) are
# unchanged by masking and already computed. Sequential.
set -u
cd ~/netslice-drl

CFG=configs/ddqn_unified_count_mask.yaml
MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"

for S in 42 43 44 45 46; do
  echo "=== [$(date +%H:%M:%S)] seed $S : unified+mask ==="
  docker run --gpus all --rm $MOUNTS netslice-drl:latest \
    --config $CFG --seed $S > results/sweep_mask_s${S}.log 2>&1
done

echo "=== [$(date +%H:%M:%S)] MASK_SWEEP_COMPLETE ==="
