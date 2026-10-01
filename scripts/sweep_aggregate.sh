#!/bin/bash
# Gap 3: aggregate-state DQN baseline on the MAIN (operator) topology.
#
# Trains the stand-in for the published 5G-core family -- aggregate resource
# state, admission-only action space -- on the same 5 seeds as the headline
# table, with the same count objective and training machinery as our agents.
# Fills the TBD row of the paper's main results table.
#
# Only the 'aggregate' baseline is requested: greedy/revenue/AC-only on this
# topology are already measured and would just be recomputed.
set -u
cd ~/netslice-drl

CFG=configs/ddqn_unified.yaml      # operator topology, K=3, count/hard
MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"

for S in 42 43 44 45 46; do
  echo "=== [$(date +%H:%M:%S)] seed $S : aggregate-state DQN ==="
  docker run --gpus all --rm --entrypoint python $MOUNTS netslice-drl:latest \
    experiments/eval_baselines.py --config $CFG --seed $S \
    --baselines aggregate \
    --train_episodes 2000 --eval_episodes 200 \
    > results/aggregate_s${S}.log 2>&1
done

echo "=== [$(date +%H:%M:%S)] AGGREGATE_SWEEP_COMPLETE ==="
