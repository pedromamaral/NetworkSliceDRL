#!/bin/bash
# Consolidation batch (Phase A1 + Phase B), sequential on gpu14.
#  A1  count/hard K=3 cap=1.0 unified, seeds 42-44 -> mechanism-diagnostic
#      checkpoints (the originals were cleaned off gpu15).
#  B1  K=8 firm-up: unified+separated, seeds 44-46 (adds to existing 42-43 -> 5).
#  B2  K=6 midpoint: unified+separated, seeds 42-44 (K=3/6/8 trend line).
set -u
cd ~/netslice-drl

MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"

run() {  # $1=config  $2=seed  $3=logtag
  echo "=== [$(date +%H:%M:%S)] $3 seed $2 ==="
  docker run --gpus all --rm $MOUNTS netslice-drl:latest \
    --config "$1" --seed "$2" > "results/consolidate_$3_s$2.log" 2>&1
}

# --- A1: mechanism checkpoints (count/hard K=3 cap=1.0) ---
for S in 42 43 44; do run configs/ddqn_unified_count.yaml $S mech_unified; done

# --- B1: K=8 firm-up (seeds 44-46) ---
for S in 44 45 46; do
  run configs/ddqn_unified_count_k8.yaml   $S k8_unified
  run configs/ddqn_separated_count_k8.yaml $S k8_separated
done

# --- B2: K=6 midpoint (seeds 42-44) ---
for S in 42 43 44; do
  run configs/ddqn_unified_count_k6.yaml   $S k6_unified
  run configs/ddqn_separated_count_k6.yaml $S k6_separated
done

echo "=== [$(date +%H:%M:%S)] CONSOLIDATE_COMPLETE ==="
