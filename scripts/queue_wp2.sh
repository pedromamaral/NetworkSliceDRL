#!/bin/bash
# Waits for the loose-end batch, then runs WP2.1 (paper/preregistration_wp2.md).
cd ~/netslice-drl
until grep -q WP1_COMPLETE results/wp1_logs/driver_loose.log; do sleep 300; done
echo b9a9489 > .wp1_commit
JOBSET=jobs_wp2 bash scripts/run_wp1_next.sh > results/wp1_logs/driver_wp2.log 2>&1
