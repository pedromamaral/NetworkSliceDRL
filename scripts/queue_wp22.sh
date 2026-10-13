#!/bin/bash
# Waits for WP2.3, then runs the WP2.2 main sweep (preregistration_wp2.md amendment 3).
# Refuses to start if another driver is already running (lesson of 2026-10-11).
cd ~/netslice-drl
until grep -q WP1_COMPLETE results/wp1_logs/driver_wp23.log; do sleep 300; done
if pgrep -f "JOBSET=jobs_wp22|run_wp1_next.sh" > /dev/null; then echo "driver already running"; exit 1; fi
echo 3c69310 > .wp1_commit
JOBSET=jobs_wp22 bash scripts/run_wp1_next.sh > results/wp1_logs/driver_wp22.log 2>&1
