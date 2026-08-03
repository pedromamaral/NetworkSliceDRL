#!/bin/bash
# Generalization test: joint unified DDQN + baselines on topology2
# (count/hard, cap_scale=0.85 ≈ operator regime), seeds 42-44. Sequential.
# Tests whether joint>greedy / joint>AC-only and the feasibility-preservation
# mechanism replicate on a structurally different graph.
set -u
cd ~/netslice-drl

CFG=configs/ddqn_unified_count_topo2.yaml
MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"

for S in 42 43 44; do
  echo "=== [$(date +%H:%M:%S)] seed $S : joint unified (topo2) ==="
  docker run --gpus all --rm $MOUNTS netslice-drl:latest \
    --config $CFG --seed $S > results/sweep_topo2_joint_s${S}.log 2>&1
  echo "=== [$(date +%H:%M:%S)] seed $S : baselines (topo2) ==="
  docker run --gpus all --rm --entrypoint python $MOUNTS netslice-drl:latest \
    experiments/eval_baselines.py --config $CFG --seed $S \
    --train_episodes 2000 --eval_episodes 200 \
    > results/sweep_topo2_baselines_s${S}.log 2>&1
done

echo "=== [$(date +%H:%M:%S)] TOPO2_SWEEP_COMPLETE ==="
