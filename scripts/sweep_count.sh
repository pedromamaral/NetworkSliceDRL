#!/bin/bash
# 5-seed sweep under the count/hard model (paper objective).
# For each seed: train the joint DDQN, then evaluate all baselines
# (greedy, revenue_heuristic, AC-only-DQN trained at parity).
# Runs STRICTLY SEQUENTIALLY — concurrent runs previously starved sshd on
# 10.26.110.15 and made the box unreachable for a day.
set -u
cd ~/netslice-drl

CFG=configs/ddqn_unified_count.yaml
MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"

for S in 42 43 44 45 46; do
  echo "=== [$(date +%H:%M:%S)] seed $S : joint DDQN ==="
  docker run --gpus all --rm $MOUNTS netslice-drl:latest \
    --config $CFG --seed $S > results/sweep_count_joint_s${S}.log 2>&1
  echo "=== [$(date +%H:%M:%S)] seed $S : baselines ==="
  docker run --gpus all --rm --entrypoint python $MOUNTS netslice-drl:latest \
    experiments/eval_baselines.py --config $CFG --seed $S \
    --train_episodes 2000 --eval_episodes 200 \
    > results/sweep_count_baselines_s${S}.log 2>&1
done

echo "=== [$(date +%H:%M:%S)] SWEEP_COMPLETE ==="
