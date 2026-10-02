#!/bin/bash
# Does the FACTORED architecture avoid the rho=0 penalty?
#
# On topology 2 (rho=0) the unified joint agent loses to admission-only
# (0.4858 vs 0.4970, t=5.97). routing_error there is 0.00%, so the deficit is
# estimation dilution across K+1 actions, not bad routing. The factored agent
# keeps admission as a size-2 head trained on every transition -- the same
# learning problem admission-only solves -- so it should not pay that penalty.
#
# Reference values to beat on this substrate (5 seeds, held-out):
#   AC-only  0.4970 +- 0.0044   <- the bar
#   joint-U  0.4858 +- 0.0028
#   greedy   0.4683 +- 0.0026
#
# Outcome if factored >= AC-only: the paper recommends ONE architecture that
# dominates, rather than a rho-dependent choice between two methods.
# Outcome if factored also lags: the rho=0 penalty is intrinsic to learning
# routing at all, which is itself a cleaner and still-publishable statement.
#
# Baselines are NOT re-run -- they are already measured on this substrate.
set -u
cd ~/netslice-drl

CFG=configs/ddqn_separated_count_topo2.yaml
MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"

for S in 42 43 44 45 46; do
  echo "=== [$(date +%H:%M:%S)] seed $S : factored DDQN (topo2, rho=0) ==="
  docker run --gpus all --rm $MOUNTS netslice-drl:latest \
    --config $CFG --seed $S > results/factored_topo2_s${S}.log 2>&1
done

echo "=== [$(date +%H:%M:%S)] FACTORED_TOPO2_COMPLETE ==="
