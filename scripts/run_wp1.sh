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

# Loose ends (2026-10-08, after Gate 1): old-batch results v3 still relies on,
# re-run with seeded replay sampling on the same image, same wp1 tag.
jobs_loose() {
  for S in $SEEDS5; do
    for K in k8 k6; do
      echo "${K}_fact_s$S|$RUN --config configs/ddqn_separated_count_$K.yaml --seed $S --tag wp1"
      echo "${K}_joint_s$S|$RUN --config configs/ddqn_unified_count_$K.yaml --seed $S --tag wp1"
      echo "${K}_acwidest_s$S|$BASE --config configs/ddqn_unified_count_$K.yaml --seed $S --baselines acwidest"
    done
    echo "op_aggregate_s$S|$BASE --config $OP --seed $S --baselines aggregate"
    echo "wx_aggregate_s$S|$BASE --config $WX --seed $S --baselines aggregate"
    echo "load_joint_s$S|$RUN --config configs/ddqn_unified_count_load.yaml --seed $S --tag wp1"
    echo "load_aconly_s$S|$BASE --config configs/ddqn_unified_count_load.yaml --seed $S --baselines aconly"
    echo "load_acwidest_s$S|$BASE --config configs/ddqn_unified_count_load.yaml --seed $S --baselines acwidest"
    echo "mask_joint_s$S|$RUN --config configs/ddqn_unified_count_mask.yaml --seed $S --tag wp1"
  done
  for S in $SEEDS5; do for C in k6 k8 load; do
    echo "${C}_heur_s$S|$BASE --config configs/ddqn_unified_count_$C.yaml --seed $S --baselines greedy,greedyshortest,revenue"
  done; done
}

# WP2.1 (paper/preregistration_wp2.md): extended rho family, ac_dqn_widest vs aconly,
# seeds 42-51; seeds 42-43 at the five WP1 levels already exist.
jobs_wp2() {
  for S in 44 45 46 47 48 49 50 51 42 43; do
    for L in r0p4 r0p8 r1p0 r1p3 r1p6 r2p0 r2p5 r3p0; do
      case "$S:$L" in 42:r0p4|42:r0p8|42:r1p3|42:r2p0|42:r3p0|43:r0p4|43:r0p8|43:r1p3|43:r2p0|43:r3p0) continue;; esac
      C=configs/ddqn_unified_rho_$L.yaml
      echo "rho_${L}_aconly_s$S|$BASE --config $C --seed $S --baselines aconly"
      echo "rho_${L}_acwidest_s$S|$BASE --config $C --seed $S --baselines acwidest"
    done
  done
  for S in 42 43 44 45 46 47 48 49 50 51; do for L in r0p4 r0p8 r1p0 r1p3 r1p6 r2p0 r2p5 r3p0; do
    case "$S:$L" in 42:r0p4|42:r0p8|42:r1p3|42:r2p0|42:r3p0|43:r0p4|43:r0p8|43:r1p3|43:r2p0|43:r3p0) continue;; esac
    echo "rho_${L}_greedy_s$S|$BASE --config configs/ddqn_unified_rho_$L.yaml --seed $S --baselines greedy,greedyshortest"
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

JOBSET="${JOBSET:-jobs}"   # jobs | jobs_loose | jobs_wp2
echo "$(date '+%F %T') WP1 start [$JOBSET], $($JOBSET | wc -l) jobs, P=$P, commit $(cat .wp1_commit 2>/dev/null)"
$JOBSET | xargs -d '\n' -P "$P" -I{} bash -c 'one "$@"' _ {}
echo "$(date '+%F %T') WP1_COMPLETE"
