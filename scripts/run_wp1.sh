#!/bin/bash
# WP1 factorial grid (paper/preregistration_wp1.md, incl. Amendment 1).
#
# Every learned arm is (re-)run with the seeded replay sampling, under tag wp1:
#   run_experiment  : joint (unified) + factored on operator/Waxman, joint on rho family
#   eval_baselines  : aconly, acwidest (2000 training episodes, same as the agents)
#   eval_baselines  : deterministic cells greedy, greedyshortest, tr_shortest/widest/dar
# One trained policy per container, P containers at once. Benchmark on gpu15
# (8 cores): 6 concurrent runs took 1.48x the single-run time, sshd stayed <1s.
#
# Usage (on gpu15):  nohup setsid bash scripts/run_wp1.sh > results/wp1_logs/driver.log 2>&1 &
set -u
cd ~/netslice-drl
P="${P:-7}"
LOG=results/wp1_logs
mkdir -p "$LOG"

MOUNTS="-v $(pwd)/results:/workspace/results -v $(pwd)/configs:/workspace/configs \
-v $(pwd)/data:/workspace/data -v $(pwd)/src:/workspace/src \
-v $(pwd)/experiments:/workspace/experiments"
RUN="docker run --gpus all --rm -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 $MOUNTS netslice-drl:latest"
BASE="docker run --gpus all --rm -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 --entrypoint python $MOUNTS netslice-drl:latest experiments/eval_baselines.py --train_episodes 2000 --eval_episodes 200"

OP=configs/ddqn_unified_count.yaml
WX=configs/ddqn_unified_count_topo2.yaml
SEEDS5="42 43 44 45 46"
LEVELS="r0p4 r0p8 r1p3 r2p0 r3p0"

jobs() {
  # long jobs first (learned policies), cheap deterministic cells last
  for S in $SEEDS5; do
    echo "op_joint_s$S|$RUN --config $OP --seed $S --tag wp1"
    echo "op_fact_s$S|$RUN --config configs/ddqn_separated_count.yaml --seed $S --tag wp1"
    echo "op_aconly_s$S|$BASE --config $OP --seed $S --baselines aconly"
    echo "op_acwidest_s$S|$BASE --config $OP --seed $S --baselines acwidest"
    echo "wx_joint_s$S|$RUN --config $WX --seed $S --tag wp1"
    echo "wx_fact_s$S|$RUN --config configs/ddqn_separated_count_topo2.yaml --seed $S --tag wp1"
    echo "wx_aconly_s$S|$BASE --config $WX --seed $S --baselines aconly"
    echo "wx_acwidest_s$S|$BASE --config $WX --seed $S --baselines acwidest"
  done
  for L in $LEVELS; do for S in 42 43; do
    C=configs/ddqn_unified_rho_$L.yaml
    echo "rho_${L}_joint_s$S|$RUN --config $C --seed $S --tag wp1"
    echo "rho_${L}_aconly_s$S|$BASE --config $C --seed $S --baselines aconly"
    echo "rho_${L}_acwidest_s$S|$BASE --config $C --seed $S --baselines acwidest"
  done; done
  H=greedy,greedyshortest,tr_shortest,tr_widest,tr_dar
  for S in $SEEDS5; do
    echo "op_heur_s$S|$BASE --config $OP --seed $S --baselines $H"
    echo "wx_heur_s$S|$BASE --config $WX --seed $S --baselines $H"
  done
  for L in $LEVELS; do for S in 42 43; do
    echo "rho_${L}_heur_s$S|$BASE --config configs/ddqn_unified_rho_$L.yaml --seed $S --baselines $H"
  done; done
}

one() {
  name="${1%%|*}"; cmd="${1#*|}"
  # skip jobs already completed (lets the driver be restarted safely)
  grep -q "^$name rc=0" "$LOG/done.txt" 2>/dev/null && return 0
  t0=$(date +%s)
  echo "$(date '+%F %T') START $name"
  $cmd > "$LOG/$name.log" 2>&1
  rc=$?
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s" >> "$LOG/done.txt"
  echo "$(date '+%F %T') END   $name rc=$rc"
}
export -f one
export LOG

echo "$(date '+%F %T') WP1 start, $(jobs | wc -l) jobs, P=$P, commit $(cat .wp1_commit 2>/dev/null)"
jobs | xargs -d '\n' -P "$P" -I{} bash -c 'one "$@"' _ {}
echo "$(date '+%F %T') WP1_COMPLETE"
