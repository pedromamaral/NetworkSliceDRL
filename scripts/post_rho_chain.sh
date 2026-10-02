#!/bin/bash
# Runs after the rho sweep completes: the factored-at-rho=0 test first (it
# attacks the paper's main weakness), then the aggregate-state baseline.
set -u
cd ~/netslice-drl
while ! grep -q RHO_SWEEP_COMPLETE results/rho_MASTER.log 2>/dev/null; do sleep 180; done
echo "[$(date +%H:%M:%S)] rho sweep done -> factored topo2"
bash scripts/sweep_factored_topo2.sh
echo "[$(date +%H:%M:%S)] factored done -> aggregate baseline"
bash scripts/sweep_aggregate.sh
echo "[$(date +%H:%M:%S)] POST_RHO_COMPLETE"
