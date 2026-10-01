#!/bin/bash
# Plan B: extend the topology-2 generalization study from 3 to 5 seeds.
# Seeds 42-44 were run on gpu14; this adds 45-46 so the joint-vs-AC-only
# comparison on topo2 has n=5 (t_crit 2.776) instead of n=3 (t_crit 4.303).
# That comparison is the paper's second headline claim and is currently
# NOT significant at n=3 (t=-3.42), so this run decides whether it holds.
# Sequential: concurrent runs have starved sshd on this box before.
set -u
cd ~/netslice-drl

CFG=configs/ddqn_unified_count_topo2.yaml
MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"

for S in 45 46; do
  echo "=== [$(date +%H:%M:%S)] seed $S : joint unified (topo2) ==="
  docker run --gpus all --rm $MOUNTS netslice-drl:latest \
    --config $CFG --seed $S > results/sweep_topo2ext_joint_s${S}.log 2>&1
  echo "=== [$(date +%H:%M:%S)] seed $S : baselines (topo2) ==="
  docker run --gpus all --rm --entrypoint python $MOUNTS netslice-drl:latest \
    experiments/eval_baselines.py --config $CFG --seed $S \
    --train_episodes 2000 --eval_episodes 200 \
    > results/sweep_topo2ext_baselines_s${S}.log 2>&1
done

echo "=== [$(date +%H:%M:%S)] TOPO2EXT_COMPLETE ==="
