#!/bin/bash
# Heavier-load test: joint unified DDQN + baselines at capacity_scale=0.5,
# count/hard, K=3, seeds 42-44. Baselines MUST re-run here (greedy/AC-only
# numbers change with the load regime). Sequential.
set -u
cd ~/netslice-drl

CFG=configs/ddqn_unified_count_load.yaml
MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"

for S in 42 43 44; do
  echo "=== [$(date +%H:%M:%S)] seed $S : joint unified (load) ==="
  docker run --gpus all --rm $MOUNTS netslice-drl:latest \
    --config $CFG --seed $S > results/sweep_load_joint_s${S}.log 2>&1
  echo "=== [$(date +%H:%M:%S)] seed $S : baselines (load) ==="
  docker run --gpus all --rm --entrypoint python $MOUNTS netslice-drl:latest \
    experiments/eval_baselines.py --config $CFG --seed $S \
    --train_episodes 2000 --eval_episodes 200 \
    > results/sweep_load_baselines_s${S}.log 2>&1
done

echo "=== [$(date +%H:%M:%S)] LOAD_SWEEP_COMPLETE ==="
