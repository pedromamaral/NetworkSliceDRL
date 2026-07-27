#!/bin/bash
# Phase 1 of the unified-vs-separated study: separated DDQN, 5 seeds, count/hard,
# K=3. Baselines and the unified arm already exist (sweep_count.sh), so this
# trains ONLY the separated joint agent. Sequential (concurrent runs starve
# sshd on 10.26.110.15).
set -u
cd ~/netslice-drl

CFG=configs/ddqn_separated_count.yaml
MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"

for S in 42 43 44 45 46; do
  echo "=== [$(date +%H:%M:%S)] seed $S : separated DDQN ==="
  docker run --gpus all --rm $MOUNTS netslice-drl:latest \
    --config $CFG --seed $S > results/sweep_sep_joint_s${S}.log 2>&1
done

echo "=== [$(date +%H:%M:%S)] SEPARATED_SWEEP_COMPLETE ==="
