#!/bin/bash
# Plan A: controlled rho sweep.
#
# Tests the paper's central claim -- that routing headroom rho predicts whether
# joint AC+RA beats admission-only control -- on a family of topologies that
# vary ONLY in rho (same core graph, same endpoints, same traffic, and
# capacity_scale calibrated per level so greedy acceptance is ~0.47 everywhere).
# This removes the confound in the current two-topology evidence, where rho was
# entangled with size, core model and capacity layout.
#
# At each level we need BOTH arms of the comparison:
#   joint      -> run_experiment with the level's config
#   AC-only    -> eval_baselines (also gives greedy/revenue as references)
#
# Expected output: the (joint - AC-only) gap should rise with rho, and be <= 0
# at rho = 0.
#
# Sequential: concurrent runs have starved sshd on this box before.
set -u
cd ~/netslice-drl

SEEDS="${SEEDS:-42 43}"
LEVELS="r0p4 r0p8 r1p3 r2p0 r3p0"

MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"

for L in $LEVELS; do
  CFG=configs/ddqn_unified_rho_${L}.yaml
  for S in $SEEDS; do
    echo "=== [$(date +%H:%M:%S)] $L seed $S : joint ==="
    docker run --gpus all --rm $MOUNTS netslice-drl:latest \
      --config $CFG --seed $S > results/rho_${L}_joint_s${S}.log 2>&1
    echo "=== [$(date +%H:%M:%S)] $L seed $S : baselines (AC-only) ==="
    docker run --gpus all --rm --entrypoint python $MOUNTS netslice-drl:latest \
      experiments/eval_baselines.py --config $CFG --seed $S \
      --train_episodes 2000 --eval_episodes 200 \
      > results/rho_${L}_baselines_s${S}.log 2>&1
  done
done

echo "=== [$(date +%H:%M:%S)] RHO_SWEEP_COMPLETE ==="
